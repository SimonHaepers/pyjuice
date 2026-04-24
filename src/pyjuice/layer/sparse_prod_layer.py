from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.nodes import InputNodes, ProdNodes, SparseProdNodes
from pyjuice.nodes.distributions import SparseCategorical
from .prod_layer import ProdLayer
from .input_layer import InputLayer
from .layer_group import LayerGroup
from .sparse_node_values import SparseNodeValues, LOG_EPS


class SparseProdLayer(ProdLayer):
    """
    Sparsity-propagating product layer. Each owned ``ns`` is a
    :class:`SparseProdNodes` (the DAG-stage subclass that validated the sparse
    pattern at build time: exactly one :class:`SparseCategorical` input child
    with identity block-sparse edges + one or more non-input children).

    Forward: produces a jagged :class:`SparseNodeValues` output — per-batch
    ``(row, value)`` pairs for rows active in the observed CSC column —
    instead of materialising a dense ``H``-vector. A scatter-to-dense step
    bridges the output to the downstream ``SumLayer``/``DenseSumLayer``
    (which still expect dense ``element_mars``) by filling inactive rows with
    ``LOG_EPS``. When *every* consumer is a :class:`SparseInputSumLayer` the
    scatter is skipped (``_skip_scatter=True``).

    Backward mirrors the sparsity: ``element_flows`` is gathered at the active
    row ids into a new :class:`SparseNodeValues`, then handed to
    :meth:`SparseCategorical.custom_backward_sparse` for direct param-flow
    accumulation. The ``InputLayer`` for the sparse child is gated off via
    ``_skip_input_forward`` / ``_skip_input_backward`` on the ``InputNodes`` so
    no duplicate work happens.

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
        for ns in self.nodes:
            assert isinstance(ns, SparseProdNodes), (
                f"SparseProdLayer expects SparseProdNodes; got {type(ns).__name__}. "
                "Build via `juice.multiply` (auto-detect) or `juice.sparse_multiply`."
            )

        # Parallel to `self.nodes`: per-ns resolved InputLayer + the name of
        # the dense-child-lookup buffer registered on this module.
        self._sparse_input_layers: List[InputLayer] = []
        self._dense_ch_lookups: List[str] = []
        for ns_idx, ns in enumerate(self.nodes):
            sparse_cs = ns.sparse_input_ns

            sparse_input_layer = None
            for lyr in input_layer_group:
                if sparse_cs in lyr.nodes:
                    sparse_input_layer = lyr
                    break
            assert sparse_input_layer is not None, (
                "SparseProdLayer could not locate the InputLayer holding the "
                "SparseCategorical input child."
            )
            self._sparse_input_layers.append(sparse_input_layer)

            # Per-row lookup for each dense child: the global ``node_mars`` nid
            # for the h-th row of ns's output. Registered as a buffer so the
            # Module's ``.to(device)`` moves it.
            H = ns.num_nodes
            bs = ns.block_size
            h_range = torch.arange(H, dtype=torch.long)
            h_block = h_range // bs
            h_within = h_range % bs
            dense_lookups = []
            for ch_idx in ns.dense_ch_idxs:
                cs = ns.chs[ch_idx]
                eids = ns.edge_ids[:, ch_idx].to(torch.long)
                dense_lookups.append(cs._output_ind_range[0] + eids[h_block] * bs + h_within)
            dense_lookup = torch.stack(dense_lookups, dim=0)  # [num_dense_chs, H]
            buf_name = f"_dense_ch_lookup_{ns_idx}"
            self.register_buffer(buf_name, dense_lookup)
            self._dense_ch_lookups.append(buf_name)

            # Gate InputLayer from populating this ns's node_mars / node_flows.
            sparse_cs._skip_input_forward = True
            sparse_cs._skip_input_backward = True

        # One slot per ns — forward populates, backward (sum + prod) reads.
        # Lists rather than dicts because ns_idx is contiguous over
        # ``range(len(self.nodes))`` and the buffers are always fully populated
        # after a forward call.
        self._sparse_outputs: List[Optional[SparseNodeValues]] = \
            [None] * len(self.nodes)
        # Backward companion: sv_flow containers written by the downstream
        # :class:`SparseInputSumLayer` fast path and consumed by this layer's
        # sparse backward. Only populated when ``_skip_scatter`` is True.
        self._sparse_flows: List[Optional[SparseNodeValues]] = \
            [None] * len(self.nodes)

        # Set to True by ``TensorCircuit._mark_sparse_prod_scatter_skip`` after
        # all layers compile when *every* consumer of this prod's outputs is a
        # :class:`SparseInputSumLayer` (which reads the sparse output directly).
        # In that case the forward pass produces only ``SparseNodeValues`` and
        # skips the O(H·B) ``scatter_to_dense``; on a B>1 fallback the
        # downstream ``SparseInputSumLayer`` materialises element_mars on
        # demand before calling ``super().forward/backward``.
        self._skip_scatter: bool = False

    def __repr__(self) -> str:
        return (
            f"SparseProdLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sparse_ns={len(self.nodes)})"
        )

    # ---------------- Forward ---------------- #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, data: Optional[torch.Tensor] = None,
                data_cpu: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert data is not None, (
            "SparseProdLayer.forward requires `data` "
            "(per-var observed tokens, shape [num_vars, 1]) as a kwarg."
        )
        assert not self.provided("fw_partition_local_ids"), \
            "SparseProdLayer does not support partial evaluation yet."

        batch_size = element_mars.size(1)
        assert batch_size == 1, (
            "SparseProdLayer is B=1 only (inference-only fast path matching "
            "SparseInputSumLayer). For B>1, build the DAG with plain "
            "`multiply` / `summate` (or `_force_plain=True`)."
        )

        # Prefer the CPU mirror cached by ``TensorCircuit.forward`` so the
        # per-ns ``build_sparse_pattern`` lookup is a host read, not a
        # device<->host sync. Falls back to ``data`` when called outside
        # TensorCircuit (e.g. unit tests that invoke the layer directly).
        data_for_pattern = data_cpu if data_cpu is not None else data

        torch.cuda.nvtx.range_push(
            f"SparseProdLayer.fwd(n_ns={len(self.nodes)},"
            f"skip_scatter={self._skip_scatter})"
        )
        for ns_idx, ns in enumerate(self.nodes):
            torch.cuda.nvtx.range_push(f"ns[{ns_idx}]")
            sv = self._compute_sparse_output(
                ns_idx=ns_idx, ns=ns,
                data=data_for_pattern, node_mars=node_mars,
            )
            self._sparse_outputs[ns_idx] = sv
            if not self._skip_scatter:
                torch.cuda.nvtx.range_push("scatter_to_dense")
                sv.scatter_to_dense(
                    element_mars, ns._output_ind_range[0], fill_value=LOG_EPS,
                )
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        return None

    def _compute_sparse_output(self, ns_idx: int, ns: SparseProdNodes,
                                data: torch.Tensor,
                                node_mars: torch.Tensor) -> SparseNodeValues:
        """
        Build the sparse ``(row, value)`` output for one
        :class:`SparseProdNodes` at B=1. Pattern construction
        (column-bounds lookup + ``indices`` view) lives on
        :meth:`SparseCategorical.build_sparse_pattern`; this method only
        launches the triton kernel that fills
        ``values[j] = log(params[csc_values_base + col_start + j]) +
                      Σ_ch node_mars[dense_ch_lookup[ch, row_j], 0]``.
        """
        device = node_mars.device
        sparse_cs = ns.sparse_input_ns
        sparse_input_layer = self._sparse_input_layers[ns_idx]

        torch.cuda.nvtx.range_push("build_sparse_pattern")
        sv = sparse_cs.dist.build_sparse_pattern(
            data=data, var_id=ns.var_id, num_rows=ns.num_nodes, device=device,
        )
        torch.cuda.nvtx.range_pop()

        if sv.total_nnz == 0:
            return sv

        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])

        torch.cuda.nvtx.range_push(f"_sparse_prod_fwd_kernel(nnz={sv.total_nnz})")
        BLOCK = 256
        grid = (triton.cdiv(sv.total_nnz, BLOCK),)
        _sparse_prod_forward_kernel[grid](
            params_ptr=sparse_input_layer.params,
            node_mars_ptr=node_mars,
            dense_ch_lookup_ptr=dense_ch_lookup,
            indices_ptr=sv.indices,
            values_out_ptr=sv.values,
            param_base=sparse_cs._param_range[0] + sv.col_start,
            num_rows=sv.num_rows,
            total_nnz=sv.total_nnz,
            NUM_DENSE_CHS=ns.num_dense_chs,
            BLOCK=BLOCK,
        )
        torch.cuda.nvtx.range_pop()

        return sv

    # ---------------- Backward ---------------- #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False,
                 data: Optional[torch.Tensor] = None, **kwargs) -> None:
        torch.cuda.nvtx.range_push(
            f"SparseProdLayer.bwd(n_ns={len(self.nodes)},"
            f"skip_scatter={self._skip_scatter})"
        )

        if self._skip_scatter:
            # Sparse fast path: downstream SparseInputSumLayer wrote sv_flow
            # straight into ``self._sparse_flows``. Route active-row flows to
            # each dense child's node_flows slice directly — no element_flows
            # round-trip, no super().backward() over the full H range.
            for ns_idx, ns in enumerate(self.nodes):
                sv_flow = self._sparse_flows[ns_idx]
                assert sv_flow is not None, (
                    "SparseProdLayer.backward (skip_scatter) expected a "
                    "sv_flow from the downstream SparseInputSumLayer; none "
                    "was stashed. Did the sum layer's backward run?"
                )
                # Consume-and-clear so stale state doesn't survive into the
                # next training step if a future bug changes eval order.
                self._sparse_flows[ns_idx] = None

                torch.cuda.nvtx.range_push(f"ns[{ns_idx}]/scatter_children")
                self._scatter_flow_to_children(ns_idx, ns, sv_flow, node_flows)
                torch.cuda.nvtx.range_pop()

                sparse_cs = ns.sparse_input_ns
                torch.cuda.nvtx.range_push(f"ns[{ns_idx}]/custom_backward_sparse")
                sparse_cs.dist.custom_backward_sparse(
                    input_layer=self._sparse_input_layers[ns_idx],
                    sparse_flow=sv_flow,
                    csc_pflows_base=sparse_cs._param_flow_range[0],
                    logspace_flows=logspace_flows,
                )
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()
            return None

        # Dense fallback: reuse the inherited plain-prod backward. It will
        # also scatter into the sparse input's node_flows slice, which is
        # harmless — InputLayer.backward skips that ns (gated via
        # ``_skip_input_backward``), so no accidental second accumulation.
        torch.cuda.nvtx.range_push("super.backward(dense_children)")
        super().backward(
            node_flows=node_flows, element_flows=element_flows,
            logspace_flows=logspace_flows, **kwargs,
        )
        torch.cuda.nvtx.range_pop()

        # Sparse-input flows: gather element_flows → SparseNodeValues →
        # SparseCategorical.custom_backward_sparse.
        for ns_idx, ns in enumerate(self.nodes):
            sv = self._sparse_outputs[ns_idx]
            assert sv is not None, (
                "SparseProdLayer.backward called before forward "
                "(no cached sparse output)."
            )
            sparse_cs = ns.sparse_input_ns
            torch.cuda.nvtx.range_push(f"ns[{ns_idx}]/gather_from_dense")
            sv_flow = sv.gather_from_dense(element_flows, ns._output_ind_range[0])
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"ns[{ns_idx}]/custom_backward_sparse")
            sparse_cs.dist.custom_backward_sparse(
                input_layer=self._sparse_input_layers[ns_idx],
                sparse_flow=sv_flow,
                csc_pflows_base=sparse_cs._param_flow_range[0],
                logspace_flows=logspace_flows,
            )
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()
        return None

    def _scatter_flow_to_children(self, ns_idx: int, ns: SparseProdNodes,
                                    sv_flow: SparseNodeValues,
                                    node_flows: torch.Tensor) -> None:
        """Write ``sv_flow.values`` into each child's ``node_flows``.

        Each active row ``h = sv_flow.indices[j]`` of the prod output has
        exactly one slot in each child (same rule as ``ProdLayer.backward``:
        ``node_flows[u_cid] = element_flows[parid]``):
          * Dense children: scatter via ``_dense_ch_lookup[ch, h]``.
          * Sparse input child: identity on the sparse input's row range.
            Even though :meth:`InputLayer.backward` skips this ns (gated via
            ``_skip_input_backward``), these posterior-marginal values at
            active states are a useful output exposed through
            ``pc.node_flows`` and are what the plain ProdLayer backward
            would have written.

        node_flows is zero-initialised at the top of ``TensorCircuit.backward``
        and no other layer writes to these rows on the sparse path, so
        inactive rows correctly stay at 0 (they contributed no LL path).
        """
        if sv_flow.total_nnz == 0:
            return
        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])
        # dense_ch_lookup: [num_dense_chs, H]. Per ch, fetch the active rows'
        # global nid indices and scatter values. B=1: only col 0 is used.
        for ch in range(ns.num_dense_chs):
            target_ids = dense_ch_lookup[ch].index_select(0, sv_flow.indices)
            node_flows[target_ids, 0] = sv_flow.values

        # Sparse input child: identity h → h within its _output_ind_range.
        sparse_input_base = ns.sparse_input_ns._output_ind_range[0]
        node_flows[sparse_input_base + sv_flow.indices, 0] = sv_flow.values


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit(
    do_not_specialize=["param_base", "num_rows", "total_nnz"],
    do_not_specialize_on_alignment=["indices_ptr", "values_out_ptr",
                                     "dense_ch_lookup_ptr"],
)
def _sparse_prod_forward_kernel(
    params_ptr, node_mars_ptr, dense_ch_lookup_ptr,
    indices_ptr,
    values_out_ptr,
    param_base,
    num_rows,
    total_nnz,
    NUM_DENSE_CHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Per-entry 1D kernel (B=1). Grid = ``(cdiv(total_nnz, BLOCK),)``.

    ``param_base`` is the caller-computed sum of the ns's CSC param base and
    the observed column's ``col_start`` — slots within the active column are
    contiguous, so slot ``j`` lives at ``params[param_base + j]``.

    For each active slot ``j``:
      - ``row      = indices[j]``
      - ``log_emit = log(params[param_base + j])``
      - ``dense_sum = Σ_ch node_mars[dense_ch_lookup[ch, row], 0]``
      - ``values[j] = log_emit + dense_sum``
    """
    pid = tl.program_id(0)
    offs_j = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs_j < total_nnz

    row = tl.load(indices_ptr + offs_j, mask=mask, other=0)

    val = tl.load(params_ptr + param_base + offs_j, mask=mask, other=1.0)
    log_emit = tl.log(val)

    dense_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for ch in tl.static_range(NUM_DENSE_CHS):
        ch_nid = tl.load(
            dense_ch_lookup_ptr + ch * num_rows + row,
            mask=mask, other=0,
        )
        # B=1: node_mars stride-1 along batch, so addr == ch_nid.
        dense_sum += tl.load(node_mars_ptr + ch_nid, mask=mask, other=0.0)

    tl.store(values_out_ptr + offs_j, log_emit + dense_sum, mask=mask)
