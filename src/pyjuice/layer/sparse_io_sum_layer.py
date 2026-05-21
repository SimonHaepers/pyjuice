from __future__ import annotations

from typing import List, Optional, Sequence

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

_BWD_BLOCK_P = 128
"""K_out-chunk tile size for :func:`_sparse_io_sum_backward_kernel`. Chunks
the serial reduction over K_out parents per program to bound register
pressure at large K_out (same reasoning as the dense-parent version in
``sparse_input_sum_layer``)."""


class SparseIOSumLayer(SparseInputSumLayer):
    """Sparse-in, sparse-out sum layer. Used on the interior of a sparse
    HMM chain where this sum's sole consumer is a :class:`CoSparseProdLayer`
    whose sparse input's CSC column defines *this* sum's output sparsity.

    Reduces the dense ``H``-row sum to a ``K_out × K_in`` tile: the parameter
    matrix is both **row-sliced** (by the consumer's emission-active rows)
    and **column-sliced** (by the child prod's emission-active rows).
    ``node_mars`` is never written — the consumer reads the packed
    :class:`SparseNodeValues` directly via ``self._sparse_outputs``.

    B=1, propagation_alg=='LL'. Backward writes the upstream prod's
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
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert batch_size == 1, (
            "SparseIOSumLayer is B=1 only (matches SparseInputSumLayer / "
            "SparseProdLayer fast path)."
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
                        "neither dim == 1; sparse path is B=1 only."
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
                if (ws_out is None or ws_out.device != node_mars.device
                        or ws_out.numel() < max(out_dist._max_nnz_per_col, out_num_rows)):
                    ws_out = torch.empty(
                        max(out_dist._max_nnz_per_col, out_num_rows),
                        dtype=torch.float32, device=node_mars.device,
                    )
                    self._fwd_values_workspaces[blk_idx] = ws_out
                sv_out = out_dist.build_sparse_pattern(
                    data=data_for_pattern, var_id=out_var_id,
                    num_rows=out_num_rows, device=node_mars.device,
                    values_out=ws_out, data_list=data_list,
                )

            self._sparse_outputs[blk_idx] = sv_out
            K_out = sv_out.total_nnz

            if K_out == 0:
                # No active output rows: downstream CoSparseProdLayer will
                # emit a zero-length sv too.
                continue

            if K_in == 0:
                # No active input rows: log(0) for every K_out parent.
                sv_out.values.fill_(float("-inf"))
                continue

            max_val = sv_in.values.max()

            TILE_M = 32
            while TILE_M > 1 and TILE_M > K_out:
                TILE_M //= 2
            if TILE_M < 1:
                TILE_M = 1

            grid = (triton.cdiv(K_out, TILE_M),)
            _sparse_io_sum_forward_kernel[grid](
                values_out_ptr=sv_out.values,
                mparams_ptr=params,
                in_indices_ptr=sv_in.indices,
                in_values_ptr=sv_in.values,
                max_val_ptr=max_val,
                out_indices_ptr=sv_out.indices,
                pid_start=pid_start,
                K_in=K_in,
                K_out=K_out,
                NB_ch=NB_ch,
                BS=BS,
                CBS=CBS,
                TILE_M=TILE_M,
                BLOCK_K=_FWD_BLOCK_K,
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
        assert batch_size == 1, "SparseIOSumLayer is B=1 only."
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
            # then carries all H rows, not just the active CSC column).
            ws_flow = self._bwd_flow_workspaces[blk_idx]
            in_dist = sparse_prod.nodes[ns_idx].sparse_input_ns.dist
            in_max_nnz = max(in_dist._max_nnz_per_col, in_dist._num_nodes)
            if (ws_flow is None or ws_flow.device != node_mars.device
                    or ws_flow.numel() < max(K_in, in_max_nnz)):
                ws_flow = torch.empty(
                    max(K_in, in_max_nnz), dtype=torch.float32, device=node_mars.device,
                )
                self._bwd_flow_workspaces[blk_idx] = ws_flow
            sv_flow_in = SparseNodeValues(
                col_start=sv_in.col_start, total_nnz=K_in,
                indices=sv_in.indices,
                values=ws_flow.narrow(0, 0, K_in),
                num_rows=sv_in.num_rows,
            )
            sparse_prod._sparse_flows[ns_idx] = sv_flow_in

            if K_in == 0 or K_out == 0:
                continue

            grid = (K_in,)
            _sparse_io_sum_backward_kernel[grid](
                flow_in_ptr=sv_flow_in.values,
                sv_flow_out_ptr=sv_flow_out.values,
                sv_out_values_ptr=sv_out.values,
                mparams_ptr=params,
                pflows_ptr=(param_flows if compute_pflows else sv_flow_in.values),
                in_indices_ptr=sv_in.indices,
                in_values_ptr=sv_in.values,
                out_indices_ptr=sv_out.indices,
                pid_start=pid_start,
                pfid_start=pfid_start,
                K_out=K_out,
                NB_ch=NB_ch,
                BS=BS,
                CBS=CBS,
                BLOCK_P=_BWD_BLOCK_P,
                COMPUTE_PFLOWS=1 if compute_pflows else 0,
                NEGATE_PFLOWS=1 if negate_pflows else 0,
            )
        return None


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
    do_not_specialize=["pid_start", "K_in", "K_out"],
    do_not_specialize_on_alignment=[
        "in_indices_ptr", "in_values_ptr", "max_val_ptr",
        "out_indices_ptr", "values_out_ptr",
    ],
)
def _sparse_io_sum_forward_kernel(
    values_out_ptr,
    mparams_ptr,
    in_indices_ptr, in_values_ptr, max_val_ptr,
    out_indices_ptr,
    pid_start,
    K_in,
    K_out,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Forward for one SparseIOSumLayer block. Grid = ``(cdiv(K_out, TILE_M),)``.

    Per-program (K_out-tile) math:

      v_k = exp(log_in_values[k] - max_val)                    [per K_in chunk]
      h_m   = out_indices[m]                                   [TILE_M gather]
      pblock_m, within_m = h_m // BS, h_m % BS
      cblock_k, cslot_k  = in_indices[k] // CBS, in_indices[k] % CBS
      W[m, k] = mparams[pid_start + (pblock_m*NB_ch + cblock_k)*CBS*BS
                                   + cslot_k*BS + within_m]
      acc_sum[m] += Σ_{k ∈ chunk} W[m, k] · v_k                [TILE_M]
      values_out[m] = log(acc_sum[m] + 1e-24) + max_val
    """
    pid_m = tl.program_id(0)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)              # [TILE_M]
    mask_m = offs_m < K_out

    h_m = tl.load(out_indices_ptr + offs_m, mask=mask_m, other=0)
    pblock_m = h_m // BS                                        # [TILE_M]
    within_m = h_m % BS                                         # [TILE_M]

    max_val = tl.load(max_val_ptr)                              # scalar

    acc_sum = tl.zeros([TILE_M], dtype=tl.float32)
    for k0 in tl.range(0, K_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                     # [BLOCK_K]
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
    do_not_specialize=["pid_start", "pfid_start", "K_out"],
    do_not_specialize_on_alignment=[
        "flow_in_ptr", "sv_flow_out_ptr", "sv_out_values_ptr",
        "in_indices_ptr", "in_values_ptr", "out_indices_ptr",
        "pflows_ptr",
    ],
)
def _sparse_io_sum_backward_kernel(
    flow_in_ptr,
    sv_flow_out_ptr, sv_out_values_ptr,
    mparams_ptr,
    pflows_ptr,
    in_indices_ptr, in_values_ptr, out_indices_ptr,
    pid_start, pfid_start,
    K_out,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr, NEGATE_PFLOWS: tl.constexpr,
):
    """Element-flow + (optional) param-flow backward for one SparseIOSumLayer
    block. Grid = ``(K_in,)``.

    Per-program (one active K_in slot) math:

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

    Param-flow scatter uses ``atomic_add`` for accumulation across multiple
    backward calls (tied SumNodes across HMM time steps share the source's
    ``pfid_start``, so each launch atomically adds its contribution). Within
    a single launch no two programs collide on the same address: CSC indices
    are unique per column so different ``k`` programs map to different
    ``(cblock, cslot)`` pairs, and inside a program different ``j`` indices
    write to different ``(pblock_j, within_j)`` pairs.
    """
    pid_k = tl.program_id(0)

    child_id = tl.load(in_indices_ptr + pid_k)
    log_val = tl.load(in_values_ptr + pid_k)
    cblock = child_id // CBS
    cslot = child_id % CBS

    acc = tl.zeros([BLOCK_P], dtype=tl.float32)

    for j0 in tl.range(0, K_out, BLOCK_P):
        offs_j = j0 + tl.arange(0, BLOCK_P)                     # [BLOCK_P]
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
            # Address layout mirrors mparams exactly (same per-block stride,
            # same intra-block ``(cslot, within_j)`` order); the BD layer uses
            # an identical pattern, see ``_bk_bd_sparse_io_kernel``.
            pf_ptr = (
                pfid_start
                + (pblock_j * NB_ch + cblock) * CBS * BS
                + cslot * BS
                + within_j
            )
            pf_val = -chunk if NEGATE_PFLOWS == 1 else chunk
            tl.atomic_add(pflows_ptr + pf_ptr, pf_val, mask=mask_j)

    tl.store(flow_in_ptr + pid_k, tl.sum(acc))
