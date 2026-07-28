from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import ProdNodes, SparseProdNodes
from .sparse_prod_layer import SparseProdLayer
from .sparse_node_values import SparseNodeValues
from .layer_group import LayerGroup


_CO_BATCHED_BLOCK_K = 64
_CO_BATCHED_BLOCK_B = 8
"""[BLOCK_B, BLOCK_K] tile shape for the batched (B>1) variant of
:func:`_co_sparse_log_add_kernel`. Pure element-wise gathers — the shape
only trades program count against register pressure."""


@triton.jit(
    do_not_specialize=["param_offset", "n", "batch_size", "in_stride",
                       "out_stride"],
)
def _co_sparse_log_add_kernel(
    out_ptr, params_ptr, dense_values_ptr, max_out_ptr,
    col_starts_ptr, nnz_ptr,
    param_offset, n,
    batch_size, in_stride, out_stride,
    BLOCK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    IS_MISSING: tl.constexpr,
    IS_BATCHED: tl.constexpr,
):
    """Fused ``out[b, j] = log(params[param_offset + col_starts[b] + j]) +
    dense_values[b, j]`` and per-sample ``max_out[b] = max_j out[b, j]`` via
    per-tile atomic-max. Grid ``(cdiv(K, BLOCK), cdiv(B, BLOCK_B))``;
    degenerates to the 1-D ``(cdiv(n, BLOCK), 1)`` form at B=1
    (``IS_BATCHED == 0``, where ``param_offset`` already folds in the
    column's ``col_start`` and both value tensors are flat ``[n]``).

    Replaces the three-kernel ``torch.log(emit_params) + sv_dense.values``
    + ``sv.values.copy_(...)`` chain that otherwise launches one tiny
    element-wise kernel each (and allocates two intermediate temps) per
    timestep on the sparse HMM path. The atomic-max additionally elides the
    per-block ``sv.values.max()`` dispatch in the downstream sum layers —
    the values are already in registers here, and ``max_out`` is
    pre-initialized to ``-inf`` by the launcher so masked-out lanes
    contribute nothing.

    When ``IS_MISSING`` is set (B=1 only), this position is being
    marginalised: the emission contribution is ``log(Σ_v P(v|h)) = log(1) =
    0`` for every latent, so the kernel skips the ``params`` load and copies
    ``dense_values`` straight through.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    if IS_BATCHED:
        pid_b = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < batch_size
        cs_b = tl.load(col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_b = tl.load(nnz_ptr + offs_b, mask=mask_b, other=0)
        mask = mask_b[:, None] & (offs[None, :] < k_b[:, None])

        d = tl.load(
            dense_values_ptr + offs_b[:, None] * in_stride + offs[None, :],
            mask=mask, other=0.0,
        )
        p = tl.load(
            params_ptr + param_offset + cs_b[:, None] + offs[None, :],
            mask=mask, other=1.0,
        )
        out = tl.log(p) + d
        tl.store(
            out_ptr + offs_b[:, None] * out_stride + offs[None, :],
            out, mask=mask,
        )

        tile_val = tl.where(mask, out, -float("inf"))
        local_max = tl.max(tile_val, axis=1)
        tl.atomic_max(max_out_ptr + offs_b, local_max, mask=mask_b)
    else:
        mask = offs < n
        d = tl.load(dense_values_ptr + offs, mask=mask, other=0.0)
        if IS_MISSING:
            out = d
        else:
            p = tl.load(params_ptr + param_offset + offs, mask=mask, other=1.0)
            out = tl.log(p) + d
        tl.store(out_ptr + offs, out, mask=mask)

        tile_val = tl.where(mask, out, -float("inf"))
        local_max = tl.max(tile_val, axis=0)
        tl.atomic_max(max_out_ptr, local_max)


class CoSparseProdLayer(SparseProdLayer):
    """Co-sparse product layer. Both inputs are :class:`SparseNodeValues`
    with **identical** indices — the sparse ``SparseCategorical`` input's
    CSC column at ``var_id`` and the upstream :class:`SparseIOSumLayer`
    output for the same ``var_id``. The log-space product reduces to an
    element-wise add of the two packed ``.values`` tensors; ``indices`` is
    passed through untouched.

    Requires ``num_dense_chs == 1`` (the HMM chain case we support).
    Unconditionally ``_skip_scatter=True`` — never writes ``element_mars``.
    """

    def __init__(self, nodes: Sequence[ProdNodes],
                 global_nid_start: Optional[int] = None,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False,
                 input_layer_group: Optional[LayerGroup] = None,
                 inner_layer_groups: Optional[Sequence[LayerGroup]] = None,
                 **kwargs) -> None:
        super().__init__(
            nodes=nodes,
            global_nid_start=global_nid_start,
            layer_sparsity_tol=layer_sparsity_tol,
            max_num_partitions=max_num_partitions,
            disable_gpu_compilation=disable_gpu_compilation,
            force_gpu_compilation=force_gpu_compilation,
            input_layer_group=input_layer_group,
        )

        for ns in self.nodes:
            assert isinstance(ns, SparseProdNodes), (
                f"CoSparseProdLayer expects SparseProdNodes; got {type(ns).__name__}."
            )
            assert ns.num_dense_chs == 1, (
                f"CoSparseProdLayer requires exactly 1 dense (sum) child; "
                f"got {ns.num_dense_chs}."
            )

        assert inner_layer_groups is not None, (
            "CoSparseProdLayer needs inner_layer_groups to resolve the "
            "upstream SparseIOSumLayer that owns each prod's dense child."
        )
        self._dense_sum_refs: List[Tuple] = []
        # Imported lazily to avoid the circular dep
        # CoSparseProdLayer → SparseIOSumLayer → SparseInputSumLayer.
        # Also accept :class:`SparseIOBlockDiagonalSumLayer` (BD-pattern
        # variant of the sparse-IO sum) — both layer types expose the
        # same ``_sparse_outputs`` / ``_sparse_flows`` plumbing.
        from .sparse_io_sum_layer import SparseIOSumLayer
        from .sparse_io_block_diagonal_sum_layer import (
            SparseIOBlockDiagonalSumLayer,
        )
        from .sparse_output_block_diagonal_sum_layer import (
            SparseOutputBlockDiagonalSumLayer,
        )
        _sparse_io_layer_cls = (SparseIOSumLayer, SparseIOBlockDiagonalSumLayer,
                                SparseOutputBlockDiagonalSumLayer)

        for ns in self.nodes:
            dense_ch_ns = ns.chs[ns.dense_ch_idxs[0]]
            found = None
            for lg in inner_layer_groups:
                if lg.is_prod():
                    continue
                for layer in lg:
                    if not isinstance(layer, _sparse_io_layer_cls):
                        continue
                    for idx, sum_ns in enumerate(layer.nodes):
                        if sum_ns is dense_ch_ns:
                            found = (layer, idx)
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            assert found is not None, (
                "CoSparseProdLayer: dense child sum ns is not owned by any "
                "sparse-IO sum layer (SparseIOSumLayer or "
                "SparseIOBlockDiagonalSumLayer) in inner_layer_groups. The "
                "DAG pre-pass classified the sum as sparse_io but the "
                "layer dispatch did not compile it as a sparse-IO layer — "
                "check TensorCircuit compilation order."
            )
            self._dense_sum_refs.append(found)

        # Never scatter to element_mars — downstream is always another
        # sparse consumer (upstream sum or the final SparseInputSumLayer-via-
        # consumer-of-our-consumer). The chain invariant is guaranteed by the
        # eligibility check.
        self._skip_scatter = True

        # ``_fwd_values_workspaces`` / ``_fwd_max_workspace`` are created by
        # ``SparseProdLayer.__init__`` (shared per-ns values workspaces + the
        # ``[len(nodes), B]`` per-sample max buffer filled by the fused
        # log+add+max kernel).

    def __repr__(self) -> str:
        return (
            f"CoSparseProdLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sparse_ns={len(self.nodes)})"
        )

    # ---------------- Forward ---------------- #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, data: Optional[torch.Tensor] = None,
                data_cpu: Optional[torch.Tensor] = None,
                data_list: Optional[list] = None,
                pattern_cache: Optional[dict] = None,
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert data is not None, (
            "CoSparseProdLayer.forward requires `data` (per-var observed tokens)."
        )
        assert not self.provided("fw_partition_local_ids"), \
            "CoSparseProdLayer does not support partial evaluation."

        batch_size = element_mars.size(1)
        assert missing_mask is None or batch_size == 1, (
            "missing_mask on the sparse fast path is only supported at "
            "batch_size == 1 (conditional queries stay B=1 for now)."
        )

        data_for_pattern = data_cpu if data_cpu is not None else data

        # See ``SparseProdLayer.forward`` for the full explanation. At
        # missing positions the upstream :class:`SparseIOSumLayer` produced
        # an all-rows ``sv_dense`` (its ``output_sparsity_var_id`` matches
        # this prod's ``var_id``), and we replace this prod's input emission
        # contribution with ``log(1) = 0`` for every latent.
        missing_mask_cpu = None
        if missing_mask is not None:
            mm = missing_mask
            if mm.dim() == 2:
                if mm.size(0) == 1:
                    mm = mm[0]
                elif mm.size(1) == 1:
                    mm = mm[:, 0]
                else:
                    raise AssertionError(
                        "CoSparseProdLayer.forward got a 2D missing_mask with "
                        "neither dim == 1; missing_mask on the sparse path "
                        "is B=1 only."
                    )
            missing_mask_cpu = mm.cpu() if mm.device.type != "cpu" else mm

        if (self._fwd_max_workspace is None
                or self._fwd_max_workspace.device != node_mars.device
                or self._fwd_max_workspace.shape != (len(self.nodes), batch_size)):
            self._fwd_max_workspace = torch.empty(
                len(self.nodes), batch_size,
                dtype=torch.float32, device=node_mars.device,
            )
        # One memset per forward replaces N per-block ``sv.values.max()``
        # dispatches; empty-column (ns, sample) slots stay at -inf, which the
        # sum layer never reads (it short-circuits / masks on nnz == 0).
        self._fwd_max_workspace.fill_(float("-inf"))

        for ns_idx, ns in enumerate(self.nodes):
            sparse_cs = ns.sparse_input_ns
            sparse_input_layer = self._sparse_input_layers[ns_idx]

            is_missing = bool(missing_mask_cpu[ns.var_id].item()) if missing_mask_cpu is not None else False
            self._was_missing[ns_idx] = is_missing
            H = ns.num_nodes

            if is_missing:
                indices = self._missing_indices_cache.get(H)
                if indices is None or indices.device != node_mars.device:
                    indices = torch.arange(H, dtype=torch.long, device=node_mars.device).contiguous()
                    self._missing_indices_cache[H] = indices
                # Workspace must hold up to H entries at missing positions
                # (vs. ``_max_nnz_per_col`` entries at observed ones).
                ws = self._fwd_values_workspaces[ns_idx]
                if (ws is None or ws.device != node_mars.device
                        or ws.numel() < H):
                    ws = torch.empty(H, dtype=torch.float32, device=node_mars.device)
                    self._fwd_values_workspaces[ns_idx] = ws
                values = ws.narrow(0, 0, H)
                sv = SparseNodeValues(
                    col_start=0, total_nnz=H,
                    indices=indices, values=values, num_rows=H,
                )
                sv.max_val = self._fwd_max_workspace[ns_idx]
                self._sparse_outputs[ns_idx] = sv

                sum_layer, sum_ns_idx = self._dense_sum_refs[ns_idx]
                sv_dense = sum_layer._sparse_outputs[sum_ns_idx]
                assert sv_dense.total_nnz == H, (
                    "CoSparseProdLayer at a missing position expected the "
                    "upstream SparseIOSumLayer to have emitted an all-rows "
                    "sv_dense; got total_nnz=%d, num_rows=%d." % (
                        sv_dense.total_nnz, H,
                    )
                )

                BLOCK = 256
                grid = (triton.cdiv(H, BLOCK), 1)
                _co_sparse_log_add_kernel[grid](
                    out_ptr=sv.values,
                    params_ptr=sparse_input_layer.params,
                    dense_values_ptr=sv_dense.values,
                    max_out_ptr=sv.max_val,
                    col_starts_ptr=sv.values, nnz_ptr=sv.values,  # unused at B=1
                    param_offset=0,
                    n=H,
                    batch_size=1, in_stride=0, out_stride=0,
                    BLOCK=BLOCK,
                    BLOCK_B=1,
                    IS_MISSING=True,
                    IS_BATCHED=False,
                )
                continue

            ws = self._fwd_values_workspaces[ns_idx]
            needed = (batch_size * sparse_cs.dist._max_nnz_per_col
                      if batch_size > 1
                      else max(sparse_cs.dist._max_nnz_per_col, H))
            if ws is None or ws.device != node_mars.device or ws.numel() < needed:
                ws = torch.empty(
                    needed, dtype=torch.float32, device=node_mars.device,
                )
                self._fwd_values_workspaces[ns_idx] = ws

            sv = sparse_cs.dist.build_sparse_pattern(
                data=data_for_pattern, var_id=ns.var_id,
                num_rows=ns.num_nodes, device=node_mars.device,
                values_out=ws, data_list=data_list,
                pattern_cache=pattern_cache,
            )
            sv.max_val = self._fwd_max_workspace[ns_idx]
            self._sparse_outputs[ns_idx] = sv

            if sv.total_nnz == 0:
                continue

            sum_layer, sum_ns_idx = self._dense_sum_refs[ns_idx]
            sv_dense = sum_layer._sparse_outputs[sum_ns_idx]

            # Invariant: indices coincide (both built from
            # ``build_sparse_pattern(var_id=ns.var_id)``, i.e. the same
            # (dist, var) pattern). At B=1, total_nnz + col_start equality
            # certifies the views alias the same memory; at B>1 the shared
            # per-query pattern_cache yields identical col_starts tensors
            # (fall back to the host nnz_list comparison for direct-layer
            # callers that built the two patterns without a shared cache).
            if sv.is_batched:
                assert sv_dense.is_batched and (
                    sv_dense.col_starts is sv.col_starts
                    or sv_dense.nnz_list == sv.nnz_list
                ), (
                    "CoSparseProdLayer: batched dense sum output sparsity "
                    "does not match this prod's sparse input sparsity "
                    "(expected the same (dist, var) pattern, ideally via a "
                    "shared pattern_cache)."
                )
            else:
                assert sv_dense.col_start == sv.col_start \
                       and sv_dense.total_nnz == sv.total_nnz, (
                    "CoSparseProdLayer: dense sum output sparsity does not "
                    "match this prod's sparse input sparsity. Expected "
                    "identical views of dist._csc_indices (same var_id)."
                )

            # Emission params live flat-packed in sparse_input_layer.params,
            # indexed by ``_param_range[0] + col_start + j`` for slot j — same
            # addressing as ``_sparse_prod_forward_kernel``. At B>1 the
            # per-sample col_start moves into the kernel.
            if sv.is_batched:
                BLOCK = _CO_BATCHED_BLOCK_K
                BLOCK_B = _CO_BATCHED_BLOCK_B
                grid = (triton.cdiv(sv.total_nnz, BLOCK),
                        triton.cdiv(batch_size, BLOCK_B))
                _co_sparse_log_add_kernel[grid](
                    out_ptr=sv.values,
                    params_ptr=sparse_input_layer.params,
                    dense_values_ptr=sv_dense.values,
                    max_out_ptr=sv.max_val,
                    col_starts_ptr=sv.col_starts, nnz_ptr=sv.nnz,
                    param_offset=sparse_cs._param_range[0],
                    n=sv.total_nnz,
                    batch_size=batch_size,
                    in_stride=sv_dense.values.stride(0),
                    out_stride=sv.values.stride(0),
                    BLOCK=BLOCK,
                    BLOCK_B=BLOCK_B,
                    IS_MISSING=False,
                    IS_BATCHED=True,
                )
            else:
                param_base = sparse_cs._param_range[0] + sv.col_start
                BLOCK = 256
                grid = (triton.cdiv(sv.total_nnz, BLOCK), 1)
                _co_sparse_log_add_kernel[grid](
                    out_ptr=sv.values,
                    params_ptr=sparse_input_layer.params,
                    dense_values_ptr=sv_dense.values,
                    max_out_ptr=sv.max_val,
                    col_starts_ptr=sv.values, nnz_ptr=sv.values,  # unused at B=1
                    param_offset=param_base,
                    n=sv.total_nnz,
                    batch_size=1, in_stride=0, out_stride=0,
                    BLOCK=BLOCK,
                    BLOCK_B=1,
                    IS_MISSING=False,
                    IS_BATCHED=False,
                )
        return None

    # ---------------- Backward ---------------- #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False,
                 data: Optional[torch.Tensor] = None, **kwargs) -> None:
        # Log-space product forward: ``out = log_emit + sv_dense``, so
        # ∂out/∂log_emit = ∂out/∂sv_dense = 1. The incoming sv_flow becomes
        # the outgoing flow on both inputs unchanged.
        for ns_idx, ns in enumerate(self.nodes):
            sv_flow = self._sparse_flows[ns_idx]
            assert sv_flow is not None, (
                "CoSparseProdLayer.backward expected sv_flow from the "
                "downstream sum layer; none was stashed."
            )
            # NOTE: do NOT clear ``self._sparse_flows[ns_idx]`` — leaving it
            # populated post-backward is what lets the sparse conditional
            # query read the per-active-row flow without re-running backward
            # (matches the contract on plain ``SparseProdLayer``).

            sparse_cs = ns.sparse_input_ns

            # Hand the SAME container to the upstream SparseIOSumLayer —
            # its backward reads sv_flow.values at K_out positions.
            sum_layer, sum_ns_idx = self._dense_sum_refs[ns_idx]
            sum_layer._sparse_flows[sum_ns_idx] = sv_flow

            if self._was_missing[ns_idx]:
                # Marginalised position: emission contribution was forced to
                # ``log(1) = 0`` in the forward, so it has zero gradient.
                # Skip the param-flow accumulation that would otherwise
                # spuriously credit this ns's emission column.
                continue

            # Emission param flow accumulation (unchanged from SparseProdLayer).
            sparse_cs.dist.custom_backward_sparse(
                input_layer=self._sparse_input_layers[ns_idx],
                sparse_flow=sv_flow,
                csc_pflows_base=sparse_cs._param_flow_range[0],
                logspace_flows=logspace_flows,
            )

            # Note: we deliberately do NOT scatter into ``node_flows`` here.
            # The dense child is a :class:`SparseIOSumLayer` whose backward
            # reads ``self._sparse_flows[blk_idx]`` (set above), not
            # ``node_flows``; and the sparse-input flow is exposed via
            # :attr:`SparseProdLayer._sparse_flows` for downstream consumers
            # (e.g. :func:`pyjuice.queries.conditional`) — see ``_sparse_flow_owner``
            # for the input-ns → owning-layer back-reference. ``pc.node_flows``
            # at chain-interior and SparseCategorical-input rows stays zero on
            # the sparse path, mirroring how forward leaves ``node_mars``
            # untouched there.

        return None
