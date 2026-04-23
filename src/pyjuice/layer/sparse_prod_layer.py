from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.nodes import InputNodes, ProdNodes
from pyjuice.nodes.distributions import SparseCategorical
from .prod_layer import ProdLayer
from .layer_group import LayerGroup
from .sparse_node_values import SparseNodeValues, LOG_EPS


class SparseProdLayer(ProdLayer):
    """
    Sparsity-propagating product layer. For each owned ``ProdNodes`` that has
    exactly one ``SparseCategorical`` input child (identity block-sparse edges)
    plus one or more dense (non-input) children, the forward pass produces a
    jagged :class:`SparseNodeValues` output — per-batch (row, value) pairs for
    rows active in the observed CSC column — instead of materialising a dense
    ``H``-vector. A scatter-to-dense step bridges the output to the downstream
    ``SumLayer`` / ``DenseSumLayer`` (which still expect dense ``element_mars``)
    by filling inactive rows with ``LOG_EPS``.

    Backward mirrors the sparsity: ``element_flows`` is gathered at the active
    row ids into a new :class:`SparseNodeValues`, and the result is handed to
    :meth:`SparseCategorical.custom_backward_sparse` for direct param-flow
    accumulation. The ``InputLayer`` for the sparse child is gated off via
    ``_skip_input_forward`` / ``_skip_input_backward`` on the ``InputNodes`` so
    no duplicate work happens.

    Constraints (checked at compile time; dispatching happens in
    :class:`TensorCircuit`):
      * exactly one ``InputNodes`` child with ``SparseCategorical`` dist;
      * the other children are non-input (sum nodes in practice);
      * block-sparse identity edges on the sparse slot;
      * ``ns.block_size == sparse_cs.block_size``,
        ``ns.num_node_blocks == sparse_cs.num_node_blocks``.

    Forward requires ``data`` (per-variable observed tokens) — pass it through
    as a ``data=...`` kwarg on ``layer_group(...)``.
    """

    def __init__(self, nodes: Sequence[ProdNodes],
                 global_nid_start: Optional[int] = None,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False,
                 input_layer_group: Optional[LayerGroup] = None,
                 **kwargs) -> None:

        super().__init__(
            nodes=nodes,
            global_nid_start=global_nid_start,
            layer_sparsity_tol=layer_sparsity_tol,
            max_num_partitions=max_num_partitions,
            disable_gpu_compilation=disable_gpu_compilation,
            force_gpu_compilation=force_gpu_compilation,
        )

        assert input_layer_group is not None, \
            "SparseProdLayer needs the compiled input_layer_group to resolve " \
            "the InputLayer that owns each SparseCategorical child."
        self._build_sparse_meta(input_layer_group)
        self._sparse_outputs: Dict[int, SparseNodeValues] = {}

        # Set to True by ``TensorCircuit._mark_sparse_prod_scatter_skip`` after
        # all layers compile when *every* consumer of this prod's outputs is a
        # :class:`SparseInputSumLayer` (which reads the sparse output directly).
        # In that case the forward pass produces only ``SparseNodeValues`` and
        # skips the O(H·B) ``scatter_to_dense``; on a B>1 fallback the
        # downstream ``SparseInputSumLayer`` materialises element_mars on
        # demand before calling ``super().forward/backward``.
        self._skip_scatter: bool = False

    def _build_sparse_meta(self, input_layer_group: LayerGroup) -> None:
        self._sparse_meta: List[dict] = []

        for ns_idx, ns in enumerate(self.nodes):
            sparse_ch_idxs = [
                i for i, cs in enumerate(ns.chs)
                if isinstance(cs, InputNodes)
                and isinstance(cs.dist, SparseCategorical)
            ]
            assert len(sparse_ch_idxs) == 1, (
                "SparseProdLayer requires exactly 1 SparseCategorical input "
                f"child per ProdNodes; got {len(sparse_ch_idxs)}."
            )
            sparse_ch_idx = sparse_ch_idxs[0]
            sparse_cs = ns.chs[sparse_ch_idx]

            dense_ch_idxs = [i for i in range(len(ns.chs)) if i != sparse_ch_idx]
            assert len(dense_ch_idxs) >= 1, \
                "SparseProdLayer requires at least one dense child."
            for i in dense_ch_idxs:
                assert not isinstance(ns.chs[i], InputNodes), (
                    f"SparseProdLayer: dense children must be non-input; "
                    f"ns.chs[{i}] is {type(ns.chs[i]).__name__}."
                )

            assert ns.is_block_sparse(), \
                "SparseProdLayer requires block-sparse edges."
            assert ns.block_size == sparse_cs.block_size, (
                f"SparseProdLayer: ns.block_size ({ns.block_size}) must match "
                f"sparse child's block_size ({sparse_cs.block_size})."
            )
            assert ns.num_node_blocks == sparse_cs.num_node_blocks, (
                f"SparseProdLayer: ns.num_node_blocks ({ns.num_node_blocks}) "
                f"must match sparse child's num_node_blocks "
                f"({sparse_cs.num_node_blocks})."
            )
            assert torch.equal(
                ns.edge_ids[:, sparse_ch_idx],
                torch.arange(ns.num_node_blocks, dtype=ns.edge_ids.dtype),
            ), "SparseProdLayer requires identity edges on the sparse slot."

            # Resolve owning InputLayer.
            sparse_input_layer = None
            for lyr in input_layer_group:
                if sparse_cs in lyr.nodes:
                    sparse_input_layer = lyr
                    break
            assert sparse_input_layer is not None, (
                "SparseProdLayer could not locate the InputLayer holding the "
                "SparseCategorical input child."
            )

            # Per-h lookup table for each dense child: global node_mars nid
            # for the h-th row of ns's output that corresponds to the dense
            # child's matching block.
            H = ns.num_nodes
            bs = ns.block_size
            h_range = torch.arange(H, dtype=torch.long)
            h_block = h_range // bs
            h_within = h_range % bs

            dense_lookups = []
            for ch_idx in dense_ch_idxs:
                cs = ns.chs[ch_idx]
                cs_base = cs._output_ind_range[0]
                eids = ns.edge_ids[:, ch_idx].to(torch.long)
                lookup = cs_base + eids[h_block] * bs + h_within
                dense_lookups.append(lookup)
            dense_lookup = torch.stack(dense_lookups, dim=0)  # [num_dense_chs, H]

            buf_name = f"_dense_ch_lookup_{ns_idx}"
            self.register_buffer(buf_name, dense_lookup)

            meta = {
                "ns_idx": ns_idx,
                "ns": ns,
                "sparse_input_ns": sparse_cs,
                "sparse_input_layer": sparse_input_layer,
                "var_id": sparse_cs.scope.to_list()[0],
                "num_rows": H,
                "block_size": bs,
                "num_dense_chs": len(dense_ch_idxs),
                "csc_values_base": sparse_cs._param_range[0],
                "csc_pflows_base": sparse_cs._param_flow_range[0],
                "max_nnz_per_col": sparse_cs.dist._max_nnz_per_col,
                "output_ind_base": ns._output_ind_range[0],
                "dense_ch_lookup_name": buf_name,
            }
            self._sparse_meta.append(meta)

            # Gate InputLayer from populating this ns's node_mars / node_flows.
            sparse_cs._skip_input_forward = True
            sparse_cs._skip_input_backward = True

    def __repr__(self) -> str:
        return (
            f"SparseProdLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sparse_ns={len(self._sparse_meta)})"
        )

    # ---------------- Forward ---------------- #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, data: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert data is not None, (
            "SparseProdLayer.forward requires `data` "
            "(per-var observed tokens, shape [num_vars, B]) as a kwarg."
        )
        assert not self.provided("fw_partition_local_ids"), \
            "SparseProdLayer does not support partial evaluation yet."

        batch_size = element_mars.size(1)

        for meta in self._sparse_meta:
            sv = self._compute_sparse_output(
                meta=meta,
                data=data,
                node_mars=node_mars,
                batch_size=batch_size,
            )
            self._sparse_outputs[meta["ns_idx"]] = sv
            if not self._skip_scatter:
                sv.scatter_to_dense(
                    element_mars, meta["output_ind_base"], fill_value=LOG_EPS,
                )

        return None

    def _compute_sparse_output(self, meta: dict, data: torch.Tensor,
                                node_mars: torch.Tensor,
                                batch_size: int) -> SparseNodeValues:
        """
        Build the sparse (row, value) output for one ProdNodes.

        Sparse-pattern tensors (``ptr``, ``indices``, ``csc_slots``,
        ``batch_ids``) are precomputed on the host via vectorised torch ops
        using the single ``.item()`` sync on ``total_nnz`` we already need for
        buffer allocation. The Triton kernel's only job is to fill
        ``values[j]`` — this gives it a 1D jagged grid
        ``(cdiv(total_nnz, BLOCK),)`` consistent with the backward /
        scatter / gather kernels (no 2D ``(B, max_K)`` padding, no masked
        wasted work).
        """
        device = node_mars.device
        var_id = meta["var_id"]
        csc_indptr = meta["sparse_input_ns"].dist._csc_indptr
        csc_indices = meta["sparse_input_ns"].dist._csc_indices
        sparse_input_layer = meta["sparse_input_layer"]

        # Normalise `data` to [num_vars * batch_size] contiguous layout so the
        # [var_id * B : (var_id+1) * B] slice is stride-1.
        data = data.reshape(-1).contiguous()

        # Per-batch column bounds.
        v = data[var_id * batch_size : (var_id + 1) * batch_size]
        col_start_per_batch = csc_indptr[v]                                   # [B]
        k_v = csc_indptr[v + 1] - col_start_per_batch                         # [B]

        # ptr = concat([0], cumsum(k_v))
        ptr = torch.empty(batch_size + 1, dtype=torch.long, device=device)
        ptr[0] = 0
        torch.cumsum(k_v, dim=0, out=ptr[1:])
        total_nnz = int(ptr[-1].item())

        if total_nnz == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_float = torch.empty(0, dtype=torch.float32, device=device)
            return SparseNodeValues(
                ptr=ptr, indices=empty_long, values=empty_float,
                csc_slots=empty_long, batch_ids=empty_long,
                num_rows=meta["num_rows"], batch_size=batch_size,
            )

        # Vectorised per-j structure:
        #   batch_ids[j] = b such that ptr[b] <= j < ptr[b+1]
        #   csc_slots[j] = col_start_per_batch[b] + (j - ptr[b])
        #   indices[j]   = csc_indices[csc_slots[j]]
        batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.long), k_v
        )
        col_start_expanded = torch.repeat_interleave(col_start_per_batch, k_v)
        ptr_expanded = torch.repeat_interleave(ptr[:-1], k_v)
        csc_slots = col_start_expanded + (
            torch.arange(total_nnz, device=device, dtype=torch.long) - ptr_expanded
        )
        indices = csc_indices[csc_slots]

        # Only `values` is kernel-computed.
        values = torch.empty(total_nnz, dtype=torch.float32, device=device)

        dense_ch_lookup = getattr(self, meta["dense_ch_lookup_name"])

        BLOCK = 256
        grid = (triton.cdiv(total_nnz, BLOCK),)
        _sparse_prod_forward_kernel[grid](
            params_ptr=sparse_input_layer.params,
            node_mars_ptr=node_mars,
            dense_ch_lookup_ptr=dense_ch_lookup,
            indices_ptr=indices,
            csc_slots_ptr=csc_slots,
            batch_ids_ptr=batch_ids,
            values_out_ptr=values,
            csc_values_base=meta["csc_values_base"],
            num_rows=meta["num_rows"],
            batch_size=batch_size,
            total_nnz=total_nnz,
            NUM_DENSE_CHS=meta["num_dense_chs"],
            BLOCK=BLOCK,
        )

        return SparseNodeValues(
            ptr=ptr, indices=indices, values=values,
            csc_slots=csc_slots, batch_ids=batch_ids,
            num_rows=meta["num_rows"], batch_size=batch_size,
        )

    # ---------------- Backward ---------------- #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False,
                 data: Optional[torch.Tensor] = None, **kwargs) -> None:
        # Dense-child node_flows: reuse the inherited plain-prod backward.
        # It will also scatter into the sparse input's node_flows slice, which
        # is harmless — InputLayer.backward skips that ns (set below via
        # `_skip_input_backward`), so no accidental second accumulation.
        super().backward(
            node_flows=node_flows, element_flows=element_flows,
            logspace_flows=logspace_flows, **kwargs,
        )

        # Sparse-input flows: gather element_flows → SparseNodeValues →
        # SparseCategorical.custom_backward_sparse.
        for meta in self._sparse_meta:
            sv = self._sparse_outputs.get(meta["ns_idx"])
            assert sv is not None, (
                "SparseProdLayer.backward called before forward "
                "(no cached sparse output)."
            )
            sv_flow = sv.gather_from_dense(element_flows, meta["output_ind_base"])

            meta["sparse_input_ns"].dist.custom_backward_sparse(
                input_layer=meta["sparse_input_layer"],
                sparse_flow=sv_flow,
                csc_pflows_base=meta["csc_pflows_base"],
                logspace_flows=logspace_flows,
            )

        return None


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit
def _sparse_prod_forward_kernel(
    params_ptr, node_mars_ptr, dense_ch_lookup_ptr,
    indices_ptr, csc_slots_ptr, batch_ids_ptr,
    values_out_ptr,
    csc_values_base,
    num_rows, batch_size,
    total_nnz,
    NUM_DENSE_CHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Per-entry 1D kernel. Grid = ``(cdiv(total_nnz, BLOCK),)``.

    For each active jagged slot ``j``:
      - ``row   = indices[j]``   (precomputed on host)
      - ``csc_slot = csc_slots[j]`` (precomputed)
      - ``b     = batch_ids[j]`` (precomputed)
      - ``log_emit = log(params[csc_values_base + csc_slot])``
      - ``dense_sum = Σ_ch node_mars[dense_ch_lookup[ch, row], b]``
      - ``values[j] = log_emit + dense_sum``
    """
    pid = tl.program_id(0)
    offs_j = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs_j < total_nnz

    row = tl.load(indices_ptr + offs_j, mask=mask, other=0)
    csc_slot = tl.load(csc_slots_ptr + offs_j, mask=mask, other=0)
    b = tl.load(batch_ids_ptr + offs_j, mask=mask, other=0)

    val = tl.load(
        params_ptr + csc_values_base + csc_slot, mask=mask, other=1.0
    )
    log_emit = tl.log(val)

    dense_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for ch in tl.static_range(NUM_DENSE_CHS):
        ch_nid = tl.load(
            dense_ch_lookup_ptr + ch * num_rows + row,
            mask=mask, other=0,
        )
        ch_addr = ch_nid * batch_size + b
        dense_sum += tl.load(node_mars_ptr + ch_addr, mask=mask, other=0.0)

    tl.store(values_out_ptr + offs_j, log_emit + dense_sum, mask=mask)
