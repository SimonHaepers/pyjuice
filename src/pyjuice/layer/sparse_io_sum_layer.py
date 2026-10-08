from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .sparse_input_sum_layer import SparseInputSumLayer
from .sparse_node_values import SparseNodeValues
from .sparse_prod_layer import SparseProdLayer


_FWD_BLOCK_K = 64
"""Fixed K_in-axis tile size for :func:`_sparse_io_sum_forward_kernel`.
Matches ``sparse_input_sum_layer._FWD_BLOCK_K``; kept as a separate constant
so the two kernels can diverge in tuning if needed."""

_FWD_BATCHED_BLOCK_B = 1
"""Batch-axis tile size for the batched (B>1) forward. Per-sample columns
mean per-lane weight gathers — BLOCK_B > 1 widens the W tile to
``[TILE_M, BLOCK_B, BLOCK_K]``; raise only if profiling justifies."""

_BWD_BLOCK_P = 128
"""K_out-chunk tile size for :func:`_sparse_io_sum_backward_kernel`. Chunks
the serial reduction over K_out parents per program to bound register
pressure at large K_out (same reasoning as the dense-parent version in
``sparse_input_sum_layer``)."""

_BWD_BATCHED_BLOCK_B = 1
"""Batch-axis tile size for the batched (B>1) backward. Per-sample columns
mean per-lane child ids and weight gathers; BLOCK_B > 1 widens the K_out
chunks to ``[BLOCK_P, BLOCK_B]``."""

# --------------------------------------------------------------------------- #
# Weight-stationary (WS) batched path
# --------------------------------------------------------------------------- #
#
# The per-sample gather kernels above read ``W[h, c]`` once per (sample, active
# pair), each from its own 32B sector. At training batch sizes the union of the
# samples' supports covers essentially every tile of W at every step, so they
# move ~B * K_in * K_out sectors (~2 GB per step at H=16k, B=64) where one
# streaming pass over W is ~1 GB. The WS path instead loads each W tile into
# shared memory once per step and applies it to the whole batch tile:
# programs own a block of rows (forward: parents, backward: children), walk the
# reduction axis block by block, and multiply against the batch's activations
# densified for that block (zeros outside each sample's support) on the tensor
# cores. Cost is one pass over W per step, independent of the support size.
#
# Parameter flows of tied transitions (the homogeneous HMM chain) are deferred:
# each step stashes its densified factors and the pass ends with ONE GEMM,
# ``pflow += W * sum_t G_t^T X_t``, instead of B * K_in * K_out atomics per step.
#
# This is ON BY DEFAULT from ``_WS_MIN_BATCH`` samples up (EM training); B=1
# inference keeps the gather kernels, which are faster there. It costs one fp32
# copy of W plus the stash, and :meth:`_WSGroup.ensure_ready` falls back to the
# gather kernels rather than OOM when those do not fit. Results differ from the
# gather path only by tf32 rounding (~1e-3 nats per sequence on a T=128 chain);
# ``_WS_PRECISION="ieee"`` is bit-comparable but gives up most of the speedup.

_WS_MODE = os.environ.get("PYJUICE_SPARSE_IO_WS", "auto")
"""``"auto"`` (the default) — WS from ``_WS_MIN_BATCH`` samples up, as long as
the circuit is big enough for a W pass to pay off (``_WS_COST_RATIO``);
``"always"`` — WS whenever eligible (tests / A-B runs); ``"never"`` — gather
kernels only."""

_WS_MIN_BATCH = int(os.environ.get("PYJUICE_SPARSE_IO_WS_MIN_BATCH", "16"))
"""Smallest batch size that takes the WS kernels in ``"auto"`` mode. Below it
the gather kernels win: they touch only the active ``(sample, in, out)`` pairs,
while a W pass costs the same whatever the batch. That makes the choice a
near-pure function of B — measured EM step (fwd+bwd, trained H=16k / d=0.01
chain, T=128) WS vs gather: B=2 0.85x, B=4 0.87x, B=8 0.94x, B=16 1.06x,
B=32 1.32x, B=64 2.4x (3.8x on an uncontended H100). The crossover sits
between 8 and 16, hence 16. WS step time is nearly flat in B (477 ms at B=2,
501 ms at B=32), so raising the training batch size is close to free."""

_WS_PRECISION = os.environ.get("PYJUICE_SPARSE_IO_WS_PRECISION", "tf32")
"""``"tf32"`` — tensor-core tf32 on operands pre-rounded to nearest (a per-pass
rounded copy of W plus rounding at densify time), so the error is zero-mean
(~1e-4 relative) rather than the ~-6e-4 bias of tf32 truncation; ``"ieee"`` —
exact fp32 FMA, ~3x slower than tf32 but still memory-bound at B=64."""

_WS_COST_RATIO = 32.0
"""Measured (H100) cost of one gathered (sample, in, out) pair in the gather
kernels relative to one streamed W element in the WS kernels. Guards the
batch-size rule against circuits too small to amortise a W pass: a chain with
H=1024 / d=0.01 gathers ~1e4 pairs per step against 1e6 W elements, so it stays
on the gather kernels at any batch size. On the H=16k chains this is slack
(B=16 gathers ~5e8 weighted pairs against H^2 = 2.7e8), i.e. the effective rule
there is ``B >= _WS_MIN_BATCH``."""

_WS_FLUSH_PANEL_ELEMS = 1 << 26
"""Max elements of the dense ``[rows, H_in]`` panel the deferred param-flow
GEMM materialises at a time (256 MB fp32)."""


class SparseIOSumLayer(SparseInputSumLayer):
    """Sparse-in, sparse-out sum layer. Used on the interior of a sparse
    HMM chain where this sum's sole consumer is a :class:`CoSparseProdLayer`
    whose sparse input's CSC column defines *this* sum's output sparsity.

    Reduces the dense ``H``-row sum to a ``K_out × K_in`` tile: the parameter
    matrix is both **row-sliced** (by the consumer's emission-active rows)
    and **column-sliced** (by the child prod's emission-active rows).
    ``node_mars`` is never written — the consumer reads the packed
    :class:`SparseNodeValues` directly via ``self._sparse_outputs``.

    Any batch size (B>1 = batched per-sample columns; ``missing_mask`` is
    B=1-only), propagation_alg=='LL'. Backward writes the upstream prod's
    ``sv_flow_in`` from the sparse fast path and (optionally) scatters
    per-edge contributions into ``param_flows`` for EM training — same
    accumulation pattern as the BD sibling
    (:class:`SparseIOBlockDiagonalSumLayer`), atomic_add into the canonical
    ``[NB, NB_ch, cbs, bs]`` flat pflow buffer.
    """

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int,
                 global_pid_start: int, global_pfid_start: int,
                 node2tiednodes: dict,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_block: int = 8,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False,
                 inner_layer_groups: Optional[list] = None,
                 output_sparsity_var_ids: Optional[Sequence[int]] = None,
                 **kwargs) -> None:
        super().__init__(
            nodes=nodes, global_nid_start=global_nid_start,
            global_pid_start=global_pid_start, global_pfid_start=global_pfid_start,
            node2tiednodes=node2tiednodes,
            layer_sparsity_tol=layer_sparsity_tol,
            max_num_partitions=max_num_partitions,
            max_tied_ns_per_parflow_block=max_tied_ns_per_parflow_block,
            disable_gpu_compilation=disable_gpu_compilation,
            force_gpu_compilation=force_gpu_compilation,
            inner_layer_groups=inner_layer_groups,
        )

        assert output_sparsity_var_ids is not None \
               and len(output_sparsity_var_ids) == len(self.nodes), (
            "SparseIOSumLayer requires one `output_sparsity_var_id` per sum "
            "node — the variable whose CSC column defines this sum's output "
            "sparsity pattern (supplied by TensorCircuit from the consumer "
            "CoSparseProdNodes' sparse_input_ns.var_id)."
        )
        self._output_sparsity_var_ids: List[int] = list(output_sparsity_var_ids)

        # Resolve the downstream sparse input ns for each sum ns once, so
        # forward can call build_sparse_pattern without re-walking the DAG.
        # We identify the consumer by iterating sibling ProdNodes in
        # ``self.nodes[i].consumers`` — but we don't have a consumers link at
        # ns level. Instead, the caller passes var_ids + we look up the dist
        # on the input_layer_group via TensorCircuit's set-up.
        # For now, we let ``build_sparse_pattern`` be called on the parent
        # prod's sparse_input_ns.dist directly by threading the dist refs.
        # Since the consumer resolution needs the DAG graph, TensorCircuit
        # also passes ``output_sparsity_dists`` and ``output_sparsity_param_ranges``.
        output_sparsity_dists = kwargs.pop("output_sparsity_dists", None)
        output_sparsity_num_rows = kwargs.pop("output_sparsity_num_rows", None)
        assert output_sparsity_dists is not None \
               and len(output_sparsity_dists) == len(self.nodes), (
            "SparseIOSumLayer requires `output_sparsity_dists` (the "
            "SparseCategorical distribution of the consumer's sparse input) "
            "per sum node."
        )
        assert output_sparsity_num_rows is not None \
               and len(output_sparsity_num_rows) == len(self.nodes), (
            "SparseIOSumLayer requires `output_sparsity_num_rows` (H of the "
            "consumer prod output, == H of this sum) per sum node."
        )
        self._output_sparsity_dists = list(output_sparsity_dists)
        self._output_sparsity_num_rows: List[int] = list(output_sparsity_num_rows)

        # Forward-cached sv_out per ns, read by downstream CoSparseProdLayer.
        self._sparse_outputs: List[Optional[SparseNodeValues]] = [None] * len(self.nodes)
        # Backward flow container written by downstream CoSparseProdLayer.
        self._sparse_flows: List[Optional[SparseNodeValues]] = [None] * len(self.nodes)

        # Per-block GPU workspaces re-used every call. Sized to the relevant
        # ``_max_nnz_per_col`` so the per-step ``cudaMalloc`` in
        # ``build_sparse_pattern`` (forward) and ``torch.empty_like`` for
        # ``sv_flow_in.values`` (backward) become free slices. Allocated lazily
        # on first use (device unknown at __init__).
        self._fwd_values_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)
        self._bwd_flow_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)

        # Lazily-allocated arange tensors used as ``indices`` for the
        # all-rows ``sv_out`` produced when ``output_sparsity_var_id`` is
        # marginalised (see ``forward``). Keyed by ``out_num_rows``.
        self._missing_indices_cache: dict = {}

        # ``(group, slot)`` per block for the weight-stationary batched path,
        # assigned by :func:`build_ws_groups` (TensorCircuit compile time).
        # ``None`` keeps the block on the gather kernels.
        self._ws_slots: List[Optional[Tuple["_WSGroup", int]]] = [None] * len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"SparseIOSumLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sum_ns={len(self._sparse_input_refs)}, "
            f"out_vars={self._output_sparsity_var_ids})"
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, force_use_bf16: bool = False,
                force_use_fp32: bool = False, propagation_alg: str = "LL",
                data: Optional[torch.Tensor] = None,
                data_cpu: Optional[torch.Tensor] = None,
                data_list: Optional[list] = None,
                pattern_cache: Optional[dict] = None,
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert missing_mask is None or batch_size == 1, (
            "missing_mask on the sparse fast path is only supported at "
            "batch_size == 1 (conditional queries stay B=1 for now)."
        )
        assert propagation_alg == "LL", (
            "SparseIOSumLayer requires propagation_alg == 'LL'."
        )
        assert params.dim() == 1
        assert data is not None, (
            "SparseIOSumLayer.forward requires `data` (per-var observed "
            "tokens) so it can build the output-sparsity pattern for each ns."
        )

        data_for_pattern = data_cpu if data_cpu is not None else data

        # ``output_sparsity_var_id`` is the var of the *consumer*
        # CoSparseProdNodes. When that var is marginalised, the consumer's
        # input emission contribution becomes ``log(1) = 0`` for every row,
        # which means the consumer needs all-rows sv_dense from us — not
        # the column-keyed pattern derived from ``data[out_var_id]`` (which
        # would be junk at a missing position anyway).
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
                        "SparseIOSumLayer.forward got a 2D missing_mask with "
                        "neither dim == 1; missing_mask on the sparse path "
                        "is B=1 only."
                    )
            missing_mask_cpu = mm.cpu() if mm.device.type != "cpu" else mm

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._sparse_input_refs,
        )):
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, BS, CBS = block
            sv_in = sparse_prod._sparse_outputs[ns_idx]
            K_in = sv_in.total_nnz

            out_dist = self._output_sparsity_dists[blk_idx]
            out_var_id = self._output_sparsity_var_ids[blk_idx]
            out_num_rows = self._output_sparsity_num_rows[blk_idx]
            out_is_missing = bool(missing_mask_cpu[out_var_id].item()) if missing_mask_cpu is not None else False

            if out_is_missing:
                # Build an all-rows sv_out: indices = arange(out_num_rows),
                # values workspace must be at least ``out_num_rows`` long.
                indices = self._missing_indices_cache.get(out_num_rows)
                if indices is None or indices.device != node_mars.device:
                    indices = torch.arange(out_num_rows, dtype=torch.long, device=node_mars.device).contiguous()
                    self._missing_indices_cache[out_num_rows] = indices
                ws_out = self._fwd_values_workspaces[blk_idx]
                if (ws_out is None or ws_out.device != node_mars.device
                        or ws_out.numel() < out_num_rows):
                    ws_out = torch.empty(out_num_rows, dtype=torch.float32, device=node_mars.device)
                    self._fwd_values_workspaces[blk_idx] = ws_out
                values = ws_out.narrow(0, 0, out_num_rows)
                sv_out = SparseNodeValues(
                    col_start=0, total_nnz=out_num_rows,
                    indices=indices, values=values, num_rows=out_num_rows,
                )
            else:
                ws_out = self._fwd_values_workspaces[blk_idx]
                needed = (batch_size * out_dist._max_nnz_per_col
                          if batch_size > 1
                          else max(out_dist._max_nnz_per_col, out_num_rows))
                if (ws_out is None or ws_out.device != node_mars.device
                        or ws_out.numel() < needed):
                    ws_out = torch.empty(
                        needed, dtype=torch.float32, device=node_mars.device,
                    )
                    self._fwd_values_workspaces[blk_idx] = ws_out
                sv_out = out_dist.build_sparse_pattern(
                    data=data_for_pattern, var_id=out_var_id,
                    num_rows=out_num_rows, device=node_mars.device,
                    values_out=ws_out, data_list=data_list,
                    pattern_cache=pattern_cache,
                )

            self._sparse_outputs[blk_idx] = sv_out
            K_out = sv_out.total_nnz

            if K_out == 0:
                # No active output rows (for any sample at B>1): downstream
                # CoSparseProdLayer will emit a zero-length sv too.
                continue

            if K_in == 0:
                # No active input rows (for any sample at B>1): log(0) for
                # every K_out parent.
                sv_out.values.fill_(float("-inf"))
                continue

            # Prefer the fused (per-sample) max attached by the upstream prod
            # layer — mandatory at B>1 (per-sample maxes of a jagged [B, K]
            # region), and at B=1 it skips the per-block torch dispatch the
            # old code paid here.
            if sv_in.max_val is not None:
                max_val = sv_in.max_val
            else:
                assert not sv_in.is_batched, (
                    "batched SparseIOSumLayer.forward requires the fused "
                    "per-sample sv_in.max_val from the upstream prod layer."
                )
                max_val = sv_in.values.max()

            TILE_M = 32
            while TILE_M > 1 and TILE_M > K_out:
                TILE_M //= 2
            if TILE_M < 1:
                TILE_M = 1

            if sv_in.is_batched and self._ws_use(blk_idx, block, sv_in, sv_out, params):
                self._ws_forward_block(blk_idx, block, sv_in, sv_out, max_val, params)
            elif sv_in.is_batched:
                assert sv_out.is_batched
                BLOCK_B = _FWD_BATCHED_BLOCK_B
                grid = (triton.cdiv(K_out, TILE_M),
                        triton.cdiv(batch_size, BLOCK_B))
                _sparse_io_sum_forward_kernel[grid](
                    values_out_ptr=sv_out.values,
                    mparams_ptr=params,
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    max_val_ptr=max_val,
                    out_indices_ptr=sv_out.indices,
                    in_col_starts_ptr=sv_in.col_starts,
                    in_nnz_ptr=sv_in.nnz,
                    out_col_starts_ptr=sv_out.col_starts,
                    out_nnz_ptr=sv_out.nnz,
                    pid_start=pid_start,
                    K_in=K_in,
                    K_out=K_out,
                    batch_size=batch_size,
                    in_stride=sv_in.values.stride(0),
                    out_stride=sv_out.values.stride(0),
                    NB_ch=NB_ch,
                    BS=BS,
                    CBS=CBS,
                    TILE_M=TILE_M,
                    BLOCK_K=_FWD_BLOCK_K,
                    BLOCK_B=BLOCK_B,
                    IS_BATCHED=True,
                )
            else:
                grid = (triton.cdiv(K_out, TILE_M), 1)
                _sparse_io_sum_forward_kernel[grid](
                    values_out_ptr=sv_out.values,
                    mparams_ptr=params,
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    max_val_ptr=max_val,
                    out_indices_ptr=sv_out.indices,
                    in_col_starts_ptr=sv_in.values,   # unused at B=1
                    in_nnz_ptr=sv_in.values,
                    out_col_starts_ptr=sv_in.values,
                    out_nnz_ptr=sv_in.values,
                    pid_start=pid_start,
                    K_in=K_in,
                    K_out=K_out,
                    batch_size=1,
                    in_stride=0,
                    out_stride=0,
                    NB_ch=NB_ch,
                    BS=BS,
                    CBS=CBS,
                    TILE_M=TILE_M,
                    BLOCK_K=_FWD_BLOCK_K,
                    BLOCK_B=1,
                    IS_BATCHED=False,
                )
        return None

    # ------------------------------------------------------------------ #
    # Backward (element flows only)
    # ------------------------------------------------------------------ #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 allow_modify_flows: bool = False, propagation_alg: str = "LL",
                 logspace_flows: bool = False, negate_pflows: bool = False,
                 accumulate_ch_flows: bool = False, allow_neg_flows: bool = False,
                 force_use_fp32: bool = False, **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL" and not logspace_flows \
               and not allow_neg_flows, (
            "SparseIOSumLayer.backward requires propagation_alg='LL' + "
            "logspace_flows=False + allow_neg_flows=False."
        )
        # ``allow_modify_flows`` only governs whether the upstream SparseInputSumLayer /
        # SparseIOSumLayer pre-transforms ``node_flows`` dense cells before its
        # kernel consumes them. We don't read from ``node_flows`` at all —
        # our "nflow" lives in the packed ``sv_flow_out`` container produced
        # by the downstream CoSparseProdLayer, which is always raw. Safe to
        # ignore the flag.
        assert not accumulate_ch_flows, (
            "SparseIOSumLayer.backward writes sv_flow_in straight into the "
            "upstream prod layer; accumulate_ch_flows is not supported."
        )

        compute_pflows = param_flows is not None

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._sparse_input_refs,
        )):
            _nid_start, _cid_start, pid_start, pfid_start, _NB, NB_ch, BS, CBS = block

            sv_flow_out = self._sparse_flows[blk_idx]
            assert sv_flow_out is not None, (
                "SparseIOSumLayer.backward expected sv_flow_out from the "
                "downstream CoSparseProdLayer."
            )
            # Consume-and-clear to avoid stale state across passes.
            self._sparse_flows[blk_idx] = None

            sv_in = sparse_prod._sparse_outputs[ns_idx]
            sv_out = self._sparse_outputs[blk_idx]
            K_in = sv_in.total_nnz
            K_out = sv_out.total_nnz

            # Mirror sv_in pattern for the sparse flow handed to upstream prod.
            # ``K_in`` can exceed ``_max_nnz_per_col`` when the upstream
            # CoSparseProdLayer's ns is at a marginalised position (its sv
            # then carries all H rows, not just the active CSC column) —
            # B=1 only; at B>1 missing is disallowed so the static
            # ``B * _max_nnz_per_col`` bound suffices.
            ws_flow = self._bwd_flow_workspaces[blk_idx]
            in_dist = sparse_prod.nodes[ns_idx].sparse_input_ns.dist
            if sv_in.is_batched:
                needed = batch_size * in_dist._max_nnz_per_col
            else:
                in_max_nnz = max(in_dist._max_nnz_per_col, in_dist._num_nodes)
                needed = max(K_in, in_max_nnz)
            if (ws_flow is None or ws_flow.device != node_mars.device
                    or ws_flow.numel() < needed):
                ws_flow = torch.empty(
                    needed, dtype=torch.float32, device=node_mars.device,
                )
                self._bwd_flow_workspaces[blk_idx] = ws_flow
            if sv_in.is_batched:
                K_stride = in_dist._max_nnz_per_col
                flow_values = ws_flow.narrow(
                    0, 0, batch_size * K_stride).view(batch_size, K_stride)
            else:
                flow_values = ws_flow.narrow(0, 0, K_in)
            sv_flow_in = sv_in.like_pattern(flow_values)
            sparse_prod._sparse_flows[ns_idx] = sv_flow_in

            if K_in == 0:
                continue
            if K_out == 0:
                # No active output rows anywhere: zero flow reaches every
                # active input slot. The workspace is reused across calls,
                # so an explicit zero-fill is required before handing it to
                # the upstream prod backward.
                sv_flow_in.values.zero_()
                continue

            if sv_in.is_batched and self._ws_use(blk_idx, block, sv_in, sv_out,
                                                 params, need_stash=compute_pflows):
                self._ws_backward_block(
                    blk_idx, block, sv_in, sv_out, sv_flow_out, sv_flow_in,
                    params, param_flows if compute_pflows else None, negate_pflows,
                )
            elif sv_in.is_batched:
                BLOCK_B = _BWD_BATCHED_BLOCK_B
                grid = (K_in, triton.cdiv(batch_size, BLOCK_B))
                _sparse_io_sum_backward_kernel[grid](
                    flow_in_ptr=sv_flow_in.values,
                    sv_flow_out_ptr=sv_flow_out.values,
                    sv_out_values_ptr=sv_out.values,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows else sv_flow_in.values),
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    out_indices_ptr=sv_out.indices,
                    in_col_starts_ptr=sv_in.col_starts,
                    in_nnz_ptr=sv_in.nnz,
                    out_col_starts_ptr=sv_out.col_starts,
                    out_nnz_ptr=sv_out.nnz,
                    pid_start=pid_start,
                    pfid_start=pfid_start,
                    K_out=K_out,
                    batch_size=batch_size,
                    in_stride=sv_in.values.stride(0),
                    flow_out_stride=sv_flow_out.values.stride(0),
                    out_stride=sv_out.values.stride(0),
                    flow_in_stride=sv_flow_in.values.stride(0),
                    NB_ch=NB_ch,
                    BS=BS,
                    CBS=CBS,
                    BLOCK_P=_BWD_BLOCK_P,
                    BLOCK_B=BLOCK_B,
                    IS_BATCHED=True,
                    COMPUTE_PFLOWS=1 if compute_pflows else 0,
                    NEGATE_PFLOWS=1 if negate_pflows else 0,
                )
            else:
                grid = (K_in, 1)
                _sparse_io_sum_backward_kernel[grid](
                    flow_in_ptr=sv_flow_in.values,
                    sv_flow_out_ptr=sv_flow_out.values,
                    sv_out_values_ptr=sv_out.values,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows else sv_flow_in.values),
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    out_indices_ptr=sv_out.indices,
                    in_col_starts_ptr=sv_in.values,  # unused at B=1
                    in_nnz_ptr=sv_in.values,
                    out_col_starts_ptr=sv_in.values,
                    out_nnz_ptr=sv_in.values,
                    pid_start=pid_start,
                    pfid_start=pfid_start,
                    K_out=K_out,
                    batch_size=1,
                    in_stride=0,
                    flow_out_stride=0,
                    out_stride=0,
                    flow_in_stride=0,
                    NB_ch=NB_ch,
                    BS=BS,
                    CBS=CBS,
                    BLOCK_P=_BWD_BLOCK_P,
                    BLOCK_B=1,
                    IS_BATCHED=False,
                    COMPUTE_PFLOWS=1 if compute_pflows else 0,
                    NEGATE_PFLOWS=1 if negate_pflows else 0,
                )
        return None

    # ------------------------------------------------------------------ #
    # Weight-stationary batched path
    # ------------------------------------------------------------------ #

    def _ws_use(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                sv_out: SparseNodeValues, params: torch.Tensor,
                need_stash: bool = False) -> bool:
        """Take the WS kernels for this batched block? (see ``_WS_MODE``)

        ``"auto"`` asks two cheap questions: is the batch at least
        ``_WS_MIN_BATCH`` samples (below that the gather kernels win), and is
        the circuit big enough that one pass over W beats gathering the active
        pairs (``_WS_COST_RATIO``). Then :meth:`_WSGroup.ensure_ready` has the
        last word, so a group whose buffers do not fit stays on the gather
        kernels instead of failing the run.
        """
        ws = self._ws_slots[blk_idx]
        if ws is None or _WS_MODE == "never":
            return False
        group = ws[0]
        if not group.eligible:
            return False
        if _WS_MODE != "always":
            if sv_in.batch_size < _WS_MIN_BATCH:
                return False
            pairs = sum(a * b for a, b in zip(sv_in.nnz_list, sv_out.nnz_list))
            if pairs * _WS_COST_RATIO < group.H_out * group.H_in:
                return False
        return group.ensure_ready(params, sv_in.batch_size, need_stash)

    def _ws_forward_block(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                          sv_out: SparseNodeValues, max_val: torch.Tensor,
                          params: torch.Tensor) -> None:
        """``sv_out = log(W @ exp(sv_in - max)) + max`` for the whole batch:
        densify the inputs, stream W once through :func:`_ws_fwd_kernel`,
        gather the active parents."""
        group, _ = self._ws_slots[blk_idx]
        _, _, _, _, _NB, NB_ch, BS, CBS = block
        B = sv_in.batch_size
        device = params.device
        H_in, H_out = group.H_in, group.H_out
        rnd = _WS_PRECISION == "tf32"
        w, w_start = group.weights(params)

        xd = _ws_scratch("x", (B, H_in), device)
        xd.zero_()
        _ws_densify_x_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), B)](
            xd, sv_in.indices, sv_in.values, max_val, sv_in.col_starts, sv_in.nnz,
            sv_in.values.stride(0), H_in, 1, 0,
            BLOCK=_WS_VEC, ROUND=rnd,
        )

        yd = _ws_scratch("y", (B, H_out), device)
        TM, BB, warps, stages = _ws_tiles(H_out, B, BS, 128, device)
        TK = min(32, CBS)
        _ws_fwd_kernel[(triton.cdiv(B, BB), H_out // TM)](
            yd, w, xd, max_val, sv_in.nnz, w_start, B, H_in, H_out,
            NB_ch=NB_ch, BS=BS, CBS=CBS, TM=TM, TK=TK, BB=BB,
            PREC="tf32" if rnd else "ieee", num_warps=warps, num_stages=stages,
        )

        _ws_gather_kernel[(triton.cdiv(sv_out.total_nnz, _WS_VEC), B)](
            sv_out.values, yd, sv_out.indices, sv_out.col_starts, sv_out.nnz,
            sv_out.values.stride(0), H_out, BLOCK=_WS_VEC,
        )

    def _ws_backward_block(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                           sv_out: SparseNodeValues, sv_flow_out: SparseNodeValues,
                           sv_flow_in: SparseNodeValues, params: torch.Tensor,
                           param_flows: Optional[torch.Tensor],
                           negate_pflows: bool) -> None:
        """Element flows through :func:`_ws_bwd_flow_kernel`; the parameter
        flows are only *stashed* here (densified ``G`` / ``X`` factors in
        this block's slot) and accumulated by :meth:`_WSGroup.flush` once
        every tied step of the pass has run.

        With ``gs[b, h] = nflow[b, h] * exp(max_b - nmars[b, h])`` and
        ``x[b, c] = exp(in[b, c] - max_b)`` (zero off-support):
          flow_in[b, c] = x[b, c] * sum_h gs[b, h] W[h, c]
          pflow[h, c]  += W[h, c] * sum_b gs[b, h] x[b, c]
        """
        group, slot = self._ws_slots[blk_idx]
        _, _, _, _, _NB, NB_ch, BS, CBS = block
        B = sv_in.batch_size
        device = params.device
        H_in, H_out = group.H_in, group.H_out
        rnd = _WS_PRECISION == "tf32"
        max_val = sv_in.max_val
        assert max_val is not None, (
            "WS backward needs the fused per-sample sv_in.max_val from the "
            "upstream prod layer's re-forward."
        )
        w, w_start = group.weights(params)

        gd = _ws_scratch("g", (B, H_out), device)
        gd.zero_()
        if param_flows is not None:
            gt, xt, col0 = group.stash_slot(slot, B, device, negate_pflows)
        else:
            gt, xt, col0 = gd, None, 0
        _ws_densify_g_kernel[(triton.cdiv(sv_out.total_nnz, _WS_VEC), B)](
            gd, gt, sv_out.indices, sv_flow_out.values, sv_out.values, max_val,
            sv_in.nnz, sv_out.col_starts, sv_out.nnz,
            sv_flow_out.values.stride(0), sv_out.values.stride(0), H_out,
            gt.stride(0), col0,
            BLOCK=_WS_VEC, ROUND=rnd, WRITE_T=param_flows is not None,
        )
        if param_flows is not None:
            _ws_densify_x_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), B)](
                xt, sv_in.indices, sv_in.values, max_val, sv_in.col_starts, sv_in.nnz,
                sv_in.values.stride(0), 1, xt.stride(0), col0,
                BLOCK=_WS_VEC, ROUND=rnd,
            )

        fd = _ws_scratch("f", (B, H_in), device)
        TK, BB, warps, stages = _ws_tiles(H_in, B, CBS, 128, device)
        TM = min(32, BS)
        _ws_bwd_flow_kernel[(triton.cdiv(B, BB), H_in // TK)](
            fd, w, gd, w_start, B, H_in, H_out,
            NB_ch=NB_ch, BS=BS, CBS=CBS, TK=TK, TM=TM, BB=BB,
            PREC="tf32" if rnd else "ieee", num_warps=warps, num_stages=stages,
        )

        _ws_gather_flow_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), B)](
            sv_flow_in.values, fd, sv_in.indices, sv_in.values, max_val,
            sv_in.col_starts, sv_in.nnz, sv_in.values.stride(0),
            sv_flow_in.values.stride(0), H_in, BLOCK=_WS_VEC,
        )


# =====================================================================
# Weight-stationary path: shared state
# =====================================================================

_WS_VEC = 256
"""Element tile of the densify / gather kernels."""

_WS_SCRATCH: Dict[Tuple[str, torch.device], torch.Tensor] = {}
_WS_W_OWNER: Dict[torch.device, Tuple["_WSGroup", int]] = {}
_WS_NUM_SMS: Dict[torch.device, int] = {}


def _ws_scratch(name: str, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Process-wide grow-only fp32 scratch. Layers run one at a time on one
    stream, so every WS call can reuse the same buffers."""
    n = 1
    for s in shape:
        n *= s
    buf = _WS_SCRATCH.get((name, device))
    if buf is None or buf.numel() < n:
        _WS_SCRATCH.pop((name, device), None)
        buf = torch.empty(n, dtype=torch.float32, device=device)
        _WS_SCRATCH[(name, device)] = buf
    return buf[:n].view(shape)


def _ws_tiles(n_rows: int, B: int, blk: int, t_max: int,
              device: torch.device) -> Tuple[int, int, int, int]:
    """``(row tile, batch tile, num_warps, num_stages)``: largest tiles that
    still give one program per SM. Row tiles divide ``blk`` (a power of two
    >= 16); batch slices of the same row block are launched adjacently so
    they share W through L2. From B=128 on the in-tile MMA starts to bind
    (H=16k, B=256: 128-wide batch tiles 0.90 ms/step vs 1.35 with 64-wide)."""
    n_sm = _WS_NUM_SMS.get(device)
    if n_sm is None:
        n_sm = torch.cuda.get_device_properties(device).multi_processor_count
        _WS_NUM_SMS[device] = n_sm
    BB = 128 if B >= 128 else (64 if B >= 64 else max(16, triton.next_power_of_2(B)))
    T = min(t_max, blk)
    while T > 16 and (n_rows // T) * triton.cdiv(B, BB) < n_sm:
        T //= 2
    while BB > 16 and (n_rows // T) * triton.cdiv(B, BB) < n_sm:
        BB //= 2
    return (T, BB, 8, 3) if BB >= 128 else (T, BB, 4, 4)


class _WSGroup:
    """WS state shared by every :class:`SparseIOSumLayer` block that reads
    the same (tied) transition matrix, i.e. the same ``pid_start`` /
    ``pfid_start`` / shape — all 127 steps of a homogeneous HMM chain:

    * the per-pass tf32-rounded copy of W (``_WS_PRECISION == "tf32"``);
    * the stash of densified backward factors, one ``B``-column slot per
      member block: ``Gt[H_out, slots*B]`` and ``Xt[H_in, slots*B]``, whose
      product :meth:`flush` turns into the parameter flows at pass end.

    ``TensorCircuit`` calls :meth:`begin_pass` at the start of every
    forward / backward and :meth:`flush` after the inner backward.
    """

    def __init__(self, pid_start: int, pfid_start: int, NB: int, NB_ch: int,
                 BS: int, CBS: int):
        self.pid_start = pid_start
        self.pfid_start = pfid_start
        self.NB, self.NB_ch, self.BS, self.CBS = NB, NB_ch, BS, CBS
        self.H_out = NB * BS
        self.H_in = NB_ch * CBS
        self.num_slots = 0
        # tl.dot needs every tile dim >= 16; tiles must divide the PC blocks.
        self.eligible = BS % 16 == 0 and CBS % 16 == 0
        self._w_valid = False
        self._gt: Optional[torch.Tensor] = None
        self._xt: Optional[torch.Tensor] = None
        self._B = 0
        self._dirty: set = set()      # slots whose stash columns may be non-zero
        self._written: set = set()    # slots written in the current pass
        self._negate: Optional[bool] = None

    def begin_pass(self) -> None:
        """Params may have changed since the last pass (EM step): rebuild the
        rounded copy on next use. Slots written by a pass that never reached
        :meth:`flush` stay dirty and are zeroed by the next flush."""
        self._w_valid = False
        self._written.clear()
        self._negate = None

    def weights(self, params: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """``(tensor, offset of W in it)`` for the WS GEMMs. In tf32 mode
        this is a process-wide buffer holding W rounded to nearest tf32,
        rebuilt once per pass (or whenever another group used the buffer)."""
        if _WS_PRECISION != "tf32":
            return params, self.pid_start
        device = params.device
        n = self.H_out * self.H_in
        buf = _ws_scratch("w_rn", (n,), device)
        owner = _WS_W_OWNER.get(device)
        if not (self._w_valid and owner is not None and owner[0] is self
                and owner[1] == buf.data_ptr()):
            BLOCK = 4096
            _ws_round_copy_kernel[(triton.cdiv(n, BLOCK),)](
                buf, params, self.pid_start, n, BLOCK=BLOCK,
            )
            _WS_W_OWNER[device] = (self, buf.data_ptr())
            self._w_valid = True
        return buf, 0

    def ensure_ready(self, params: torch.Tensor, B: int, need_stash: bool) -> bool:
        """Allocate this group's WS buffers up front — the per-pass rounded W
        copy (``H^2`` floats) and, when ``need_stash``, the deferred param-flow
        stash (``(H_out + H_in) * num_slots * B`` floats). Returns ``False``
        after disabling the group if they do not fit, so that enabling WS by
        default can never turn a run that used to fit into an OOM crash; the
        blocks then stay on the gather kernels for the rest of the session.
        """
        try:
            if _WS_PRECISION == "tf32":
                _ws_scratch("w_rn", (self.H_out * self.H_in,), params.device)
            if need_stash:
                self._ensure_stash(B, params.device)
        except torch.cuda.OutOfMemoryError:
            self.eligible = False
            self._gt = self._xt = None
            _WS_SCRATCH.pop(("w_rn", params.device), None)
            extra = (self.H_out * self.H_in
                     + (self.H_out + self.H_in) * self.num_slots * B) * 4 / 2 ** 30
            warnings.warn(
                "Not enough GPU memory for the weight-stationary sparse-IO "
                f"path (needs up to ~{extra:.1f} GiB beyond the circuit: one "
                "fp32 copy of the transition matrix plus the deferred "
                "param-flow stash); falling back to the per-sample gather "
                "kernels, which are several times slower at this batch size. "
                "Lower the batch size, or set PYJUICE_SPARSE_IO_WS_PRECISION="
                "ieee to drop the W copy (exact, but its backward is slower "
                "than the gather path).",
                RuntimeWarning,
            )
            return False
        return True

    def _ensure_stash(self, B: int, device: torch.device) -> None:
        N = self.num_slots * B
        if self._gt is None or self._B != B or self._gt.device != device:
            self._gt = self._xt = None       # free the old pair before retrying
            self._gt = torch.zeros(self.H_out, N, dtype=torch.float32, device=device)
            self._xt = torch.zeros(self.H_in, N, dtype=torch.float32, device=device)
            self._B = B
            self._dirty.clear()

    def stash_slot(self, slot: int, B: int, device: torch.device,
                   negate: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """``(Gt, Xt, first column)`` of ``slot``, zeroed and marked written."""
        self._ensure_stash(B, device)
        assert self._negate is None or self._negate == negate, (
            "negate_pflows changed within one backward pass."
        )
        self._negate = negate
        col0 = slot * B
        if slot in self._dirty:
            self._gt[:, col0:col0 + B].zero_()
            self._xt[:, col0:col0 + B].zero_()
        self._dirty.add(slot)
        self._written.add(slot)
        return self._gt, self._xt, col0

    def flush(self, params: torch.Tensor, param_flows: torch.Tensor) -> None:
        """``pflow[h, c] += W[h, c] * sum_n Gt[h, n] Xt[c, n]`` over every
        slot of the pass: one cuBLAS GEMM (in row panels to bound the dense
        temporary) plus an epilogue that reads W and read-modify-writes the
        param flows once per pass."""
        if not self._written:
            return
        B = self._B
        for s in self._dirty - self._written:      # stale slots of an aborted pass
            self._gt[:, s * B:(s + 1) * B].zero_()
            self._xt[:, s * B:(s + 1) * B].zero_()
        self._dirty &= self._written

        device = params.device
        TMe = min(64, self.BS)
        TKe = min(64, self.CBS)
        rows = min(self.H_out, max(TMe, _WS_FLUSH_PANEL_ELEMS // self.H_in // TMe * TMe))
        panel = _ws_scratch("pflow_panel", (rows, self.H_in), device)
        prev = torch.backends.cuda.matmul.allow_tf32
        # Gt / Xt were rounded to nearest tf32 at densify time, so cuBLAS's
        # tf32 truncation is exact on them.
        torch.backends.cuda.matmul.allow_tf32 = _WS_PRECISION == "tf32"
        try:
            for r0 in range(0, self.H_out, rows):
                r = min(rows, self.H_out - r0)
                out = panel[:r]
                torch.mm(self._gt[r0:r0 + r], self._xt.t(), out=out)
                _ws_pflow_epilogue_kernel[(r // TMe, self.H_in // TKe)](
                    param_flows, params, out, r0, self.pid_start, self.pfid_start,
                    self.H_in, NB_ch=self.NB_ch, BS=self.BS, CBS=self.CBS,
                    TM=TMe, TK=TKe, NEGATE=bool(self._negate),
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        self._written.clear()
        self._negate = None


def build_ws_groups(inner_layer_groups) -> List[_WSGroup]:
    """Group the blocks of every :class:`SparseIOSumLayer` by the transition
    matrix they read and hand each block its ``(group, slot)``. Called once
    by ``TensorCircuit`` after compilation."""
    groups: Dict[tuple, _WSGroup] = {}
    for lg in inner_layer_groups:
        for layer in lg:
            if type(layer) is not SparseIOSumLayer:
                continue
            for blk_idx, block in enumerate(layer._dense_blocks):
                _nid, _cid, pid_start, pfid_start, NB, NB_ch, BS, CBS = block
                key = (pid_start, pfid_start, NB, NB_ch, BS, CBS)
                group = groups.get(key)
                if group is None:
                    group = groups[key] = _WSGroup(*key)
                layer._ws_slots[blk_idx] = (group, group.num_slots)
                group.num_slots += 1
    return list(groups.values())


# =====================================================================
# Triton kernels
# =====================================================================
#
# NOTE: These kernels mirror ``_sparse_input_sum_{forward,backward_sv}_kernel``
# in ``sparse_input_sum_layer.py`` but gather parent rows from ``out_indices``
# instead of iterating contiguously over ``NB * BS`` dense parents, and write
# to packed ``values_out`` / read from packed ``sv_flow_out`` / ``sv_out.values``
# instead of ``node_mars`` / ``node_flows`` / ``node_mars`` respectively.
# Keep the logsumexp math in sync with the sibling kernels.


@triton.jit(
    do_not_specialize=["pid_start", "K_in", "K_out", "batch_size",
                       "in_stride", "out_stride"],
    do_not_specialize_on_alignment=[
        "in_indices_ptr", "in_values_ptr", "max_val_ptr",
        "out_indices_ptr", "values_out_ptr",
        "in_col_starts_ptr", "in_nnz_ptr",
        "out_col_starts_ptr", "out_nnz_ptr",
    ],
)
def _sparse_io_sum_forward_kernel(
    values_out_ptr,
    mparams_ptr,
    in_indices_ptr, in_values_ptr, max_val_ptr,
    out_indices_ptr,
    in_col_starts_ptr, in_nnz_ptr,
    out_col_starts_ptr, out_nnz_ptr,
    pid_start,
    K_in,
    K_out,
    batch_size,
    in_stride, out_stride,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr, IS_BATCHED: tl.constexpr,
):
    """Forward for one SparseIOSumLayer block. Grid =
    ``(cdiv(K_out, TILE_M), cdiv(B, BLOCK_B))`` (axis 1 degenerate at B=1).

    Per-program (K_out-tile) math at ``IS_BATCHED == 0``:

      v_k = exp(log_in_values[k] - max_val)                    [per K_in chunk]
      h_m   = out_indices[m]                                   [TILE_M gather]
      pblock_m, within_m = h_m // BS, h_m % BS
      cblock_k, cslot_k  = in_indices[k] // CBS, in_indices[k] % CBS
      W[m, k] = mparams[pid_start + (pblock_m*NB_ch + cblock_k)*CBS*BS
                                   + cslot_k*BS + within_m]
      acc_sum[m] += Σ_{k ∈ chunk} W[m, k] · v_k                [TILE_M]
      values_out[m] = log(acc_sum[m] + 1e-24) + max_val

    ``IS_BATCHED == 1``: each batch lane reads its own in/out columns
    (``in_indices[in_col_starts[b] + k]`` masked ``k < in_nnz[b]``,
    ``out_indices[out_col_starts[b] + m]`` masked ``m < out_nnz[b]``),
    ``K_in`` / ``K_out`` become the per-batch max bounds, per-sample
    ``max_val[b]`` is clamped to ``-1e30``, and lanes with
    ``in_nnz[b] == 0`` store exact ``-inf``.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)              # [TILE_M]

    if IS_BATCHED:
        pid_b = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)        # [BLOCK_B]
        mask_b = offs_b < batch_size
        cs_in_b = tl.load(in_col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_in_b = tl.load(in_nnz_ptr + offs_b, mask=mask_b, other=0)
        cs_out_b = tl.load(out_col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_out_b = tl.load(out_nnz_ptr + offs_b, mask=mask_b, other=0)

        mask_mb = mask_b[None, :] & (offs_m[:, None] < k_out_b[None, :])
        h_m = tl.load(
            out_indices_ptr + cs_out_b[None, :] + offs_m[:, None],
            mask=mask_mb, other=0,
        )                                                       # [TILE_M, BLOCK_B]
        pblock_m = h_m // BS
        within_m = h_m % BS

        max_b = tl.load(max_val_ptr + offs_b, mask=mask_b, other=0.0)
        max_b = tl.maximum(max_b, -1e30)                        # empty col guard

        acc_sum = tl.zeros([TILE_M, BLOCK_B], dtype=tl.float32)
        for k0 in tl.range(0, K_in, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)                 # [BLOCK_K]
            mask_kb = mask_b[:, None] & (offs_k[None, :] < k_in_b[:, None])

            child_ids = tl.load(
                in_indices_ptr + cs_in_b[:, None] + offs_k[None, :],
                mask=mask_kb, other=0,
            )                                                   # [BLOCK_B, BLOCK_K]
            log_vals = tl.load(
                in_values_ptr + offs_b[:, None] * in_stride + offs_k[None, :],
                mask=mask_kb, other=-float("inf"),
            )

            v_k = tl.where(mask_kb, tl.exp(log_vals - max_b[:, None]), 0.0)

            cblock = child_ids // CBS
            cslot = child_ids % CBS

            W_ptr_off = (
                pid_start
                + (pblock_m[:, :, None] * NB_ch + cblock[None, :, :]) * CBS * BS
                + cslot[None, :, :] * BS
                + within_m[:, :, None]
            )                                                   # [TILE_M, BLOCK_B, BLOCK_K]
            combined_mask = mask_mb[:, :, None] & mask_kb[None, :, :]
            W = tl.load(mparams_ptr + W_ptr_off, mask=combined_mask,
                        other=0.0).to(tl.float32)

            acc_sum += tl.sum(W * v_k[None, :, :], axis=2)      # [TILE_M, BLOCK_B]

        result = tl.log(acc_sum + 1e-24) + max_b[None, :]
        result = tl.where(k_in_b[None, :] == 0, -float("inf"), result)

        tl.store(
            values_out_ptr + offs_b[None, :] * out_stride + offs_m[:, None],
            result, mask=mask_mb,
        )
    else:
        mask_m = offs_m < K_out

        h_m = tl.load(out_indices_ptr + offs_m, mask=mask_m, other=0)
        pblock_m = h_m // BS                                    # [TILE_M]
        within_m = h_m % BS                                     # [TILE_M]

        max_val = tl.load(max_val_ptr)                          # scalar

        acc_sum = tl.zeros([TILE_M], dtype=tl.float32)
        for k0 in tl.range(0, K_in, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)                 # [BLOCK_K]
            mask_k = offs_k < K_in

            child_ids = tl.load(in_indices_ptr + offs_k, mask=mask_k, other=0)
            log_vals = tl.load(in_values_ptr + offs_k, mask=mask_k,
                               other=-float("inf"))

            v_k = tl.where(mask_k, tl.exp(log_vals - max_val), 0.0)

            cblock = child_ids // CBS
            cslot = child_ids % CBS

            W_ptr_off = (
                pid_start
                + (pblock_m[:, None] * NB_ch + cblock[None, :]) * CBS * BS
                + cslot[None, :] * BS
                + within_m[:, None]
            )
            combined_mask = mask_m[:, None] & mask_k[None, :]
            W = tl.load(mparams_ptr + W_ptr_off, mask=combined_mask,
                        other=0.0).to(tl.float32)

            acc_sum += tl.sum(W * v_k[None, :], axis=1)

        result = tl.log(acc_sum + 1e-24) + max_val
        tl.store(values_out_ptr + offs_m, result, mask=mask_m)


@triton.jit(
    do_not_specialize=["pid_start", "pfid_start", "K_out", "batch_size",
                       "in_stride", "flow_out_stride", "out_stride",
                       "flow_in_stride"],
    do_not_specialize_on_alignment=[
        "flow_in_ptr", "sv_flow_out_ptr", "sv_out_values_ptr",
        "in_indices_ptr", "in_values_ptr", "out_indices_ptr",
        "pflows_ptr",
        "in_col_starts_ptr", "in_nnz_ptr",
        "out_col_starts_ptr", "out_nnz_ptr",
    ],
)
def _sparse_io_sum_backward_kernel(
    flow_in_ptr,
    sv_flow_out_ptr, sv_out_values_ptr,
    mparams_ptr,
    pflows_ptr,
    in_indices_ptr, in_values_ptr, out_indices_ptr,
    in_col_starts_ptr, in_nnz_ptr,
    out_col_starts_ptr, out_nnz_ptr,
    pid_start, pfid_start,
    K_out,
    batch_size,
    in_stride, flow_out_stride, out_stride, flow_in_stride,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_B: tl.constexpr, IS_BATCHED: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr, NEGATE_PFLOWS: tl.constexpr,
):
    """Element-flow + (optional) param-flow backward for one SparseIOSumLayer
    block. Grid = ``(K_in_max, cdiv(B, BLOCK_B))`` (axis 1 degenerate at B=1).

    Per-program (one active K_in slot) math at ``IS_BATCHED == 0``:

      c_k, log_val_k = in_indices[k], in_values[k]
      cblock, cslot  = c_k // CBS, c_k % CBS
      for j in chunks of K_out:
        h_j         = out_indices[j]
        pblock_j    = h_j // BS,  within_j = h_j % BS
        nflow_j     = sv_flow_out[j]           # packed K_out flow
        nmars_j     = sv_out_values[j]         # packed K_out forward result
        W_j         = mparams[pid_start + (pblock_j*NB_ch + cblock)*CBS*BS
                                        + cslot*BS + within_j]
        chunk       = nflow_j * W_j * exp(log_val_k - nmars_j)
        acc        += chunk
      flow_in[k] = Σ acc
      # When COMPUTE_PFLOWS == 1:
      param_flows[pfid_start + (pblock_j*NB_ch + cblock)*CBS*BS
                              + cslot*BS + within_j] += chunk

    ``IS_BATCHED == 1``: each batch lane reads its own in/out columns
    (``in_indices[in_col_starts[b] + k]`` masked ``k < in_nnz[b]``, packed
    values / flows via per-tensor row strides) and ``K_out`` is the
    per-batch max chunk bound; lanes with ``k >= in_nnz[b]`` are inert.

    Param-flow scatter uses ``atomic_add`` for accumulation across multiple
    backward calls (tied SumNodes across HMM time steps share the source's
    ``pfid_start``, so each launch atomically adds its contribution). At
    B=1, within a single launch no two programs collide on the same address
    (CSC indices are unique per column, and inside a program different
    ``j`` indices hit different ``(pblock_j, within_j)`` pairs). At B>1,
    lanes of *different samples* observing the same token DO collide on the
    same address — the atomic is exactly the cross-sample reduction EM
    needs (the pflow buffer is batch-less by design).
    """
    pid_k = tl.program_id(0)

    if IS_BATCHED:
        pid_b = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)        # [BLOCK_B]
        mask_b = offs_b < batch_size
        cs_in_b = tl.load(in_col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_in_b = tl.load(in_nnz_ptr + offs_b, mask=mask_b, other=0)
        cs_out_b = tl.load(out_col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_out_b = tl.load(out_nnz_ptr + offs_b, mask=mask_b, other=0)
        lane_mask = mask_b & (pid_k < k_in_b)                   # [BLOCK_B]

        n_active = tl.sum(lane_mask.to(tl.int32), axis=0)
        if n_active > 0:
            child_id = tl.load(in_indices_ptr + cs_in_b + pid_k,
                               mask=lane_mask, other=0)         # [BLOCK_B]
            log_val = tl.load(in_values_ptr + offs_b * in_stride + pid_k,
                              mask=lane_mask, other=-float("inf"))
            cblock = child_id // CBS
            cslot = child_id % CBS

            acc = tl.zeros([BLOCK_P, BLOCK_B], dtype=tl.float32)
            for j0 in tl.range(0, K_out, BLOCK_P):
                offs_j = j0 + tl.arange(0, BLOCK_P)             # [BLOCK_P]
                mask_jb = lane_mask[None, :] & (offs_j[:, None] < k_out_b[None, :])

                h_j = tl.load(
                    out_indices_ptr + cs_out_b[None, :] + offs_j[:, None],
                    mask=mask_jb, other=0,
                )                                               # [BLOCK_P, BLOCK_B]
                pblock_j = h_j // BS
                within_j = h_j % BS

                nflows = tl.load(
                    sv_flow_out_ptr + offs_b[None, :] * flow_out_stride + offs_j[:, None],
                    mask=mask_jb, other=0.0,
                )
                nmars = tl.load(
                    sv_out_values_ptr + offs_b[None, :] * out_stride + offs_j[:, None],
                    mask=mask_jb, other=0.0,
                )

                W_ptr = (
                    pid_start
                    + (pblock_j * NB_ch + cblock[None, :]) * CBS * BS
                    + cslot[None, :] * BS
                    + within_j
                )
                W = tl.load(mparams_ptr + W_ptr, mask=mask_jb,
                            other=0.0).to(tl.float32)

                chunk_full = nflows * W * tl.exp(log_val[None, :] - nmars)
                chunk = tl.where(mask_jb, chunk_full, 0.0)
                acc += chunk

                if COMPUTE_PFLOWS == 1:
                    pf_ptr = (
                        pfid_start
                        + (pblock_j * NB_ch + cblock[None, :]) * CBS * BS
                        + cslot[None, :] * BS
                        + within_j
                    )
                    pf_val = -chunk if NEGATE_PFLOWS == 1 else chunk
                    tl.atomic_add(pflows_ptr + pf_ptr, pf_val, mask=mask_jb)

            tl.store(flow_in_ptr + offs_b * flow_in_stride + pid_k,
                     tl.sum(acc, axis=0), mask=lane_mask)
    else:
        child_id = tl.load(in_indices_ptr + pid_k)
        log_val = tl.load(in_values_ptr + pid_k)
        cblock = child_id // CBS
        cslot = child_id % CBS

        acc = tl.zeros([BLOCK_P], dtype=tl.float32)

        for j0 in tl.range(0, K_out, BLOCK_P):
            offs_j = j0 + tl.arange(0, BLOCK_P)                 # [BLOCK_P]
            mask_j = offs_j < K_out

            h_j = tl.load(out_indices_ptr + offs_j, mask=mask_j, other=0)
            pblock_j = h_j // BS
            within_j = h_j % BS

            nflows = tl.load(sv_flow_out_ptr + offs_j, mask=mask_j, other=0.0)
            nmars = tl.load(sv_out_values_ptr + offs_j, mask=mask_j, other=0.0)

            W_ptr = (
                pid_start
                + (pblock_j * NB_ch + cblock) * CBS * BS
                + cslot * BS
                + within_j
            )
            W = tl.load(mparams_ptr + W_ptr, mask=mask_j, other=0.0).to(tl.float32)

            chunk_full = nflows * W * tl.exp(log_val - nmars)
            chunk = tl.where(mask_j, chunk_full, 0.0)
            acc += chunk

            if COMPUTE_PFLOWS == 1:
                # Address layout mirrors mparams exactly (same per-block
                # stride, same intra-block ``(cslot, within_j)`` order); the
                # BD layer uses an identical pattern, see
                # ``_bk_bd_sparse_io_kernel``.
                pf_ptr = (
                    pfid_start
                    + (pblock_j * NB_ch + cblock) * CBS * BS
                    + cslot * BS
                    + within_j
                )
                pf_val = -chunk if NEGATE_PFLOWS == 1 else chunk
                tl.atomic_add(pflows_ptr + pf_ptr, pf_val, mask=mask_j)

        tl.store(flow_in_ptr + pid_k, tl.sum(acc))


# =====================================================================
# Weight-stationary path: Triton kernels
# =====================================================================


@triton.jit
def _ws_round_tf32(x):
    """Round non-negative finite fp32 to the nearest tf32 value (10-bit
    mantissa). Tensor cores drop the 13 low mantissa bits of fp32 operands;
    on pre-rounded operands that truncation is exact, so the tf32 error
    becomes zero-mean instead of a systematic ~-3e-4 relative bias per
    operand (~-6e-4 per step in log space, ~-0.08 nats over a T=128 chain)."""
    xi = x.to(tl.int32, bitcast=True)
    xi = (xi + 0x1000) & -8192
    return xi.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["src_start", "n"])
def _ws_round_copy_kernel(dst_ptr, src_ptr, src_start, n, BLOCK: tl.constexpr):
    """``dst[i] = round_tf32(src[src_start + i])`` — the per-pass W copy."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src_ptr + src_start + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, _ws_round_tf32(x), mask=mask)


@triton.jit(do_not_specialize=["v_stride", "dst_stride_b", "dst_stride_r", "dst_off"])
def _ws_densify_x_kernel(dst_ptr, idx_ptr, val_ptr, max_ptr, cs_ptr, nnz_ptr,
                         v_stride, dst_stride_b, dst_stride_r, dst_off,
                         BLOCK: tl.constexpr, ROUND: tl.constexpr):
    """Scatter ``exp(values[b, k] - max_b)`` of a batched sparse column to
    ``dst[dst_off + b*dst_stride_b + row*dst_stride_r]`` (a pre-zeroed dense
    ``[B, H]`` (strides ``H, 1``) or a transposed stash ``[H, N]`` slot
    (strides ``1, N``)). Grid ``(cdiv(max nnz, BLOCK), B)``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    row = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    v = tl.load(val_ptr + b * v_stride + k, mask=mask, other=-float("inf"))
    mx = tl.maximum(tl.load(max_ptr + b), -1e30)
    x = tl.exp(v - mx)
    if ROUND:
        x = _ws_round_tf32(x)
    tl.store(dst_ptr + dst_off + b * dst_stride_b + row.to(tl.int64) * dst_stride_r, x,
             mask=mask)


@triton.jit(do_not_specialize=["out_stride"])
def _ws_gather_kernel(out_ptr, src_ptr, idx_ptr, cs_ptr, nnz_ptr, out_stride, H,
                      BLOCK: tl.constexpr):
    """``out[b, k] = src[b, indices[col_starts[b] + k]]`` for ``k < nnz[b]``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    row = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    v = tl.load(src_ptr + b * H + row, mask=mask, other=0.0)
    tl.store(out_ptr + b * out_stride + k, v, mask=mask)


@triton.jit(do_not_specialize=["w_start", "B"])
def _ws_fwd_kernel(yd_ptr, w_ptr, xd_ptr, max_ptr, kin_ptr, w_start, B, H_in, H_out,
                   NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                   TM: tl.constexpr, TK: tl.constexpr, BB: tl.constexpr,
                   PREC: tl.constexpr):
    """Weight-stationary forward of one SparseIOSumLayer block. Grid
    ``(cdiv(B, BB), H_out // TM)``: program = (batch slice, block of TM
    parents). It walks the children — the softmax / reduction axis — TK at a
    time; every W tile is loaded once and applied to all BB samples:

        acc[b, m]      = sum_c xd[b, c] * W[h0 + m, c]
        yd[b, h0 + m]  = log(acc[b, m] + 1e-24) + max_b   (-inf if nnz_in[b] == 0)

    W[h, c] sits at ``w_start + (pb*NB_ch + cb)*CBS*BS + cs*BS + w`` with
    ``h = pb*BS + w`` and ``c = cb*CBS + cs`` (parents contiguous), so the
    loaded ``[TK, TM]`` tile is the dot's ``[K, N]`` operand as is.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    h0 = pid_m * TM
    pb = h0 // BS
    w0 = h0 % BS
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, TK)
    offs_b = pid_b * BB + tl.arange(0, BB)
    mask_b = offs_b < B

    w_rows = w_ptr + w_start + pb.to(tl.int64) * (NB_ch * CBS * BS) + w0
    acc = tl.zeros([BB, TM], dtype=tl.float32)
    for k0 in range(0, H_in, TK):
        cb = k0 // CBS
        cs0 = k0 % CBS
        wt = tl.load(w_rows + cb * (CBS * BS) + (cs0 + offs_k)[:, None] * BS + offs_m[None, :])
        x = tl.load(xd_ptr + offs_b[:, None] * H_in + (k0 + offs_k)[None, :],
                    mask=mask_b[:, None], other=0.0)
        acc += tl.dot(x, wt, input_precision=PREC)

    mx = tl.maximum(tl.load(max_ptr + offs_b, mask=mask_b, other=0.0), -1e30)
    kin = tl.load(kin_ptr + offs_b, mask=mask_b, other=0)
    res = tl.log(acc + 1e-24) + mx[:, None]
    res = tl.where(kin[:, None] == 0, -float("inf"), res)
    tl.store(yd_ptr + offs_b[:, None] * H_out + (h0 + offs_m)[None, :], res,
             mask=mask_b[:, None])


@triton.jit(do_not_specialize=["flow_stride", "mars_stride", "gt_ld", "gt_off"])
def _ws_densify_g_kernel(gd_ptr, gt_ptr, idx_ptr, nflow_ptr, nmars_ptr, max_ptr, kin_ptr,
                         cs_ptr, nnz_ptr, flow_stride, mars_stride, H_out, gt_ld, gt_off,
                         BLOCK: tl.constexpr, ROUND: tl.constexpr, WRITE_T: tl.constexpr):
    """Scatter ``gs = nflow * exp(max_b - nmars)`` (0 for samples without
    active inputs) of the batched output column into the pre-zeroed dense
    ``gd[B, H_out]`` and, with ``WRITE_T``, into the transposed stash slot
    ``gt[h, gt_off + b]``. ``max_b`` is the input max the forward used, so
    ``gs * x = nflow * exp(in - nmars)`` exactly as the gather kernel's
    per-pair term."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    h = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    nf = tl.load(nflow_ptr + b * flow_stride + k, mask=mask, other=0.0)
    nm = tl.load(nmars_ptr + b * mars_stride + k, mask=mask, other=0.0)
    mx = tl.maximum(tl.load(max_ptr + b), -1e30)
    g = nf * tl.exp(mx - nm)
    dead = (tl.load(kin_ptr + b) == 0) | (nm == -float("inf"))
    g = tl.where(dead, 0.0, g)
    if ROUND:
        g = _ws_round_tf32(g)
    tl.store(gd_ptr + b * H_out + h, g, mask=mask)
    if WRITE_T:
        tl.store(gt_ptr + gt_off + h * gt_ld + b, g, mask=mask)


@triton.jit(do_not_specialize=["w_start", "B"])
def _ws_bwd_flow_kernel(fd_ptr, w_ptr, gd_ptr, w_start, B, H_in, H_out,
                        NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                        TK: tl.constexpr, TM: tl.constexpr, BB: tl.constexpr,
                        PREC: tl.constexpr):
    """Weight-stationary element-flow backward. Grid ``(cdiv(B, BB),
    H_in // TK)``: program = (batch slice, block of TK children); walks the
    parents TM at a time, ``fd[b, c] = sum_h gd[b, h] * W[h, c]``. The
    ``[TM, TK]`` W tile is parents-contiguous, i.e. K-major for the dot."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    c0 = pid_c * TK
    cb = c0 // CBS
    cs0 = c0 % CBS
    offs_k = tl.arange(0, TK)
    offs_m = tl.arange(0, TM)
    offs_b = pid_b * BB + tl.arange(0, BB)
    mask_b = offs_b < B

    w_cols = w_ptr + w_start + cb * (CBS * BS) + cs0 * BS
    acc = tl.zeros([BB, TK], dtype=tl.float32)
    for h0 in range(0, H_out, TM):
        pb = h0 // BS
        w0 = h0 % BS
        w = tl.load(w_cols + pb.to(tl.int64) * (NB_ch * CBS * BS)
                    + offs_k[None, :] * BS + (w0 + offs_m)[:, None])
        g = tl.load(gd_ptr + offs_b[:, None] * H_out + (h0 + offs_m)[None, :],
                    mask=mask_b[:, None], other=0.0)
        acc += tl.dot(g, w, input_precision=PREC)
    tl.store(fd_ptr + offs_b[:, None] * H_in + (c0 + offs_k)[None, :], acc,
             mask=mask_b[:, None])


@triton.jit(do_not_specialize=["v_stride", "out_stride"])
def _ws_gather_flow_kernel(out_ptr, fd_ptr, idx_ptr, val_ptr, max_ptr, cs_ptr, nnz_ptr,
                           v_stride, out_stride, H_in, BLOCK: tl.constexpr):
    """``flow_in[b, k] = exp(in[b, k] - max_b) * fd[b, indices[cs[b] + k]]``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    c = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    v = tl.load(val_ptr + b * v_stride + k, mask=mask, other=-float("inf"))
    mx = tl.maximum(tl.load(max_ptr + b), -1e30)
    f = tl.load(fd_ptr + b * H_in + c, mask=mask, other=0.0)
    tl.store(out_ptr + b * out_stride + k, tl.exp(v - mx) * f, mask=mask)


@triton.jit(do_not_specialize=["r0", "pid_start", "pfid_start"])
def _ws_pflow_epilogue_kernel(pf_ptr, w_ptr, s_ptr, r0, pid_start, pfid_start, H_in,
                              NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                              TM: tl.constexpr, TK: tl.constexpr, NEGATE: tl.constexpr):
    """``param_flows[(h, c)] += W[(h, c)] * S[h - r0, c]`` for one
    ``[rows, H_in]`` panel ``S = Gt[r0:r0+rows] @ Xt^T`` of the deferred
    param-flow GEMM (same ``(pb*NB_ch + cb)*CBS*BS + cs*BS + w`` layout for
    params and param flows). Grid ``(rows // TM, H_in // TK)``."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    h0 = r0 + pid_m * TM
    c0 = pid_c * TK
    pb = h0 // BS
    w0 = h0 % BS
    cb = c0 // CBS
    cs0 = c0 % CBS
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, TK)
    s = tl.load(s_ptr + (pid_m * TM + offs_m).to(tl.int64)[:, None] * H_in
                + (c0 + offs_k)[None, :])
    off = ((pb.to(tl.int64) * NB_ch + cb) * (CBS * BS)
           + (cs0 + offs_k)[None, :] * BS + (w0 + offs_m)[:, None])
    upd = tl.load(w_ptr + pid_start + off) * s
    if NEGATE:
        upd = -upd
    pf = tl.load(pf_ptr + pfid_start + off)
    tl.store(pf_ptr + pfid_start + off, pf + upd)
