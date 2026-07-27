from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .dense_sum_layer import DenseSumLayer
from .sparse_node_values import SparseNodeValues
from .sparse_prod_layer import SparseProdLayer


_FWD_BLOCK_K = 64
"""Fixed K-axis tile size for :func:`_sparse_input_sum_forward_kernel`.

The forward kernel reduces over ``total_nnz`` active child slots. Keeping
``BLOCK_K`` a module-level constant means the kernel compiles **once** —
independent of per-call nnz — so we can ship AOT-compiled kernels for
deployment on platforms without a Triton JIT. Tuning knob: larger = fewer
inner-loop iterations but more register pressure per program.
"""

_FWD_BATCHED_BLOCK_B = 1
"""Batch-axis tile size for the batched (B>1) forward. Each batch lane
gathers its own weight entries (per-sample columns), so BLOCK_B > 1 widens
the W tile to ``[TILE_M, BLOCK_B, BLOCK_K]`` — raise only if profiling
shows the per-sample-program default leaves the GPU underutilised."""


_BWD_BATCHED_BLOCK_B = 1
"""Batch-axis tile size for the batched (B>1) sv backward. Per-sample
columns mean per-lane child ids and weight gathers; BLOCK_B > 1 widens the
serial parent tiles to ``[BLOCK_P, BLOCK_B]``."""

_BWD_SV_BLOCK_P = 128
"""Parent-chunk tile size for :func:`_sparse_input_sum_backward_sv_kernel`.

Each program handles one active child ``c_k`` and reduces serially over all
``NB * BS`` parents. At ``bs=1024``, a single-tile load of the full parent
block would spill registers; chunking by ``BLOCK_P`` keeps live tiles
bounded at ``BLOCK_P`` floats for ``nflows``, ``W[:, c_k]``, and (optionally)
``nmars``. Compiled once — runtime handles any ``BS`` that's a multiple.
"""


class SparseInputSumLayer(DenseSumLayer):
    """
    Block-dense sum layer whose single child is a :class:`SparseProdLayer`.
    Forward/backward read the upstream :class:`SparseNodeValues` directly
    (O(nnz·M)) instead of the full element_mars tile (O(H·M)).

    Supports any batch size on the pure sparse chain (every upstream
    ``SparseProdLayer`` with ``_skip_scatter=True``): B=1 runs the classic
    single-column kernels, B>1 the batched per-sample-column variants.
    Param-flow backward (EM training) is supported on the sparse fast path;
    the dense element_flows fallback (mixed-consumer topologies) and
    ``missing_mask`` handling remain B=1-only.
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
        )

        assert inner_layer_groups is not None, (
            "SparseInputSumLayer needs the already-compiled inner_layer_groups "
            "to resolve the upstream SparseProdLayer that owns each sum's child."
        )
        self._build_sparse_input_refs(inner_layer_groups)

    def _build_sparse_input_refs(self, inner_layer_groups: list) -> None:
        """For each :class:`SparseSumNodes` in ``self.nodes``, store
        ``(sparse_prod_layer, ns_idx_in_prod)`` so forward/backward can fetch
        the per-call :class:`SparseNodeValues`.
        """
        self._sparse_input_refs: List[Tuple[SparseProdLayer, int]] = []

        for ns in self.nodes:
            assert len(ns.chs) == 1, \
                "SparseInputSumLayer requires num_chs == 1 per SumNodes."
            cs = ns.chs[0]

            found = None
            for lg in inner_layer_groups:
                if not lg.is_prod():
                    continue
                for layer in lg:
                    if not isinstance(layer, SparseProdLayer):
                        continue
                    for idx, prod_ns in enumerate(layer.nodes):
                        if prod_ns is cs:
                            found = (layer, idx)
                            break
                    if found is not None:
                        break
                if found is not None:
                    break

            assert found is not None, (
                "SparseInputSumLayer: child ProdNodes is not owned by any "
                "SparseProdLayer in the already-compiled inner_layer_groups."
            )
            self._sparse_input_refs.append(found)

    def __repr__(self) -> str:
        return (
            f"SparseInputSumLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sum_ns={len(self._sparse_input_refs)})"
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, force_use_bf16: bool = False,
                force_use_fp32: bool = False, propagation_alg: str = "LL",
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL", (
            "SparseInputSumLayer requires propagation_alg == 'LL'."
        )

        assert params.dim() == 1

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._sparse_input_refs,
        )):
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, BS, CBS = block
            sv = sparse_prod._sparse_outputs[ns_idx]
            total_nnz = sv.total_nnz
            if total_nnz == 0:
                # Empty active column (every sample's column at B>1):
                # log(0) = -inf for all parents.
                node_mars[nid_start:nid_start + NB * BS, :].fill_(float("-inf"))
                continue

            # Per-call max of sparse values (for linear-sum numerical stability).
            # Prefer the upstream-fused per-sample max produced inline by the
            # prod layers' fused kernels — that leaves a ready-to-use
            # ``sv.max_val`` so we can skip the per-block ``sv.values.max()``
            # torch dispatch entirely. Fall back to the eager reduction for
            # B=1 producers that didn't fuse; at B>1 the fused max is
            # mandatory (per-sample maxes of a jagged [B, K] region).
            if sv.max_val is not None:
                max_val = sv.max_val
            else:
                assert not sv.is_batched, (
                    "batched SparseInputSumLayer.forward requires the fused "
                    "per-sample sv.max_val from the upstream prod layer."
                )
                max_val = sv.values.max()

            # Tile parents: TILE_M divides BS and is a power of 2.
            TILE_M = min(BS, 32)
            while BS % TILE_M != 0 and TILE_M > 1:
                TILE_M //= 2

            if sv.is_batched:
                BLOCK_B = _FWD_BATCHED_BLOCK_B
                grid = (NB * (BS // TILE_M),
                        triton.cdiv(batch_size, BLOCK_B))
                _sparse_input_sum_forward_kernel[grid](
                    node_mars_ptr=node_mars,
                    mparams_ptr=params,
                    indices_ptr=sv.indices,
                    values_ptr=sv.values,
                    max_val_ptr=max_val,
                    col_starts_ptr=sv.col_starts, nnz_ptr=sv.nnz,
                    nid_start=nid_start,
                    pid_start=pid_start,
                    batch_size=batch_size,
                    total_nnz=total_nnz,
                    v_stride=sv.values.stride(0),
                    NB_ch=NB_ch,
                    BS=BS,
                    CBS=CBS,
                    TILE_M=TILE_M,
                    BLOCK_K=_FWD_BLOCK_K,
                    BLOCK_B=BLOCK_B,
                    IS_BATCHED=True,
                )
            else:
                grid = (NB * (BS // TILE_M), 1)
                _sparse_input_sum_forward_kernel[grid](
                    node_mars_ptr=node_mars,
                    mparams_ptr=params,
                    indices_ptr=sv.indices,
                    values_ptr=sv.values,
                    max_val_ptr=max_val,
                    col_starts_ptr=sv.values, nnz_ptr=sv.values,  # unused at B=1
                    nid_start=nid_start,
                    pid_start=pid_start,
                    batch_size=batch_size,
                    total_nnz=total_nnz,
                    v_stride=0,
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
            "SparseInputSumLayer backward requires propagation_alg='LL' + "
            "logspace_flows=False + allow_neg_flows=False."
        )

        compute_pflows = param_flows is not None
        if compute_pflows:
            # Dense fallback path does not (yet) support param-flow scatter.
            # The fast path below covers every real HMM construction we have,
            # so gate the fallback rather than silently producing wrong pflows.
            all_skip = all(
                sp._skip_scatter for sp, _ in self._sparse_input_refs
            )
            assert all_skip, (
                "SparseInputSumLayer.backward with param_flows requires the "
                "sparse fast path (every upstream SparseProdLayer must have "
                "_skip_scatter=True)."
            )

        # Fast path: every upstream SparseProdLayer has _skip_scatter=True, so
        # it doesn't need element_flows and we can write sv_flow directly onto
        # the prod. Mixed-consumer topologies take the dense path below.
        all_skip = all(
            sp._skip_scatter for sp, _ in self._sparse_input_refs
        )
        if all_skip:
            if accumulate_ch_flows:
                raise NotImplementedError(
                    "SparseInputSumLayer (skip_scatter) backward does not "
                    "support accumulate_ch_flows=True; the sparse fast path "
                    "writes sv_flow straight into the upstream prod layer."
                )
            # When ``allow_modify_flows`` is True, the kernel below assumes
            # ``node_flows[parent_rows]`` has been pre-transformed to
            # ``log(flow) - node_mars``. DenseSumLayer.backward does this in
            # its own body (sparse fast path skips super().backward), so we
            # replicate the transform here. Without it, the root sum's
            # ``node_flows=1`` would enter the kernel unmodified and the
            # ``allow_modify_flows`` branch's ``exp(nflow + log_val)`` would
            # be off by ``exp(1) * exp(nmars[root])`` — silently breaking the
            # backward flow chain at the root and compounding through every
            # downstream consumer.
            if allow_modify_flows:
                propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
                propagation_alg_kwargs = self._get_propagation_alg_kwargs(
                    propagation_alg,
                )
                alpha = float(propagation_alg_kwargs.get("alpha", 0.0))
                for block in self._dense_blocks:
                    nid_start, _cid_start, _pid_start, _pfid_start, NB, _NB_ch, bs, _cbs = block
                    layer_n_nodes = NB * bs
                    BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
                    BLOCK_B = min(2048, BATCH_SIZE_NP2)
                    BLOCK_M = min(max(2048 // BLOCK_B, 1), bs)
                    if BLOCK_M < 1:
                        BLOCK_M = 1
                    grid = (triton.cdiv(batch_size, BLOCK_B),
                            triton.cdiv(layer_n_nodes, BLOCK_M))
                    self._bk_triton_dense_modify_flow_kernel[grid](
                        node_flows=node_flows,
                        node_mars=node_mars,
                        nid_start=nid_start,
                        batch_size=batch_size,
                        num_parents=layer_n_nodes,
                        BLOCK_B=BLOCK_B,
                        BLOCK_M=BLOCK_M,
                        propagation_alg_id=propagation_alg_id,
                        alpha=alpha,
                    )
            for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
                self._dense_blocks, self._sparse_input_refs,
            )):
                nid_start, _cid_start, pid_start, pfid_start, NB, NB_ch, BS, CBS = block
                sv = sparse_prod._sparse_outputs[ns_idx]
                total_nnz = sv.total_nnz

                # Mirror sv's pattern so SparseProdLayer.backward (sparse path)
                # can find the flow. The prod layer will pop this slot.
                sv_flow = sv.like_pattern(torch.empty_like(sv.values))
                sparse_prod._sparse_flows[ns_idx] = sv_flow

                if total_nnz == 0:
                    continue

                if sv.is_batched:
                    BLOCK_B = _BWD_BATCHED_BLOCK_B
                    grid = (total_nnz, triton.cdiv(batch_size, BLOCK_B))
                    _sparse_input_sum_backward_sv_kernel[grid](
                        node_flows_ptr=node_flows,
                        node_mars_ptr=node_mars,
                        mparams_ptr=params,
                        indices_ptr=sv.indices,
                        values_ptr=sv.values,
                        flow_out_ptr=sv_flow.values,
                        pflows_ptr=(param_flows if compute_pflows
                                    else sv_flow.values),
                        col_starts_ptr=sv.col_starts,
                        nnz_ptr=sv.nnz,
                        nid_start=nid_start,
                        pid_start=pid_start,
                        pfid_start=pfid_start,
                        batch_size=batch_size,
                        v_stride=sv.values.stride(0),
                        f_stride=sv_flow.values.stride(0),
                        NB=NB,
                        NB_ch=NB_ch,
                        BS=BS,
                        CBS=CBS,
                        BLOCK_P=_BWD_SV_BLOCK_P,
                        BLOCK_B=BLOCK_B,
                        IS_BATCHED=True,
                        allow_modify_flows=1 if allow_modify_flows else 0,
                        COMPUTE_PFLOWS=1 if compute_pflows else 0,
                        NEGATE_PFLOWS=1 if negate_pflows else 0,
                    )
                else:
                    grid = (total_nnz, 1)
                    _sparse_input_sum_backward_sv_kernel[grid](
                        node_flows_ptr=node_flows,
                        node_mars_ptr=node_mars,
                        mparams_ptr=params,
                        indices_ptr=sv.indices,
                        values_ptr=sv.values,
                        flow_out_ptr=sv_flow.values,
                        pflows_ptr=(param_flows if compute_pflows
                                    else sv_flow.values),
                        col_starts_ptr=sv.values,  # unused at B=1
                        nnz_ptr=sv.values,
                        nid_start=nid_start,
                        pid_start=pid_start,
                        pfid_start=pfid_start,
                        batch_size=batch_size,
                        v_stride=0,
                        f_stride=0,
                        NB=NB,
                        NB_ch=NB_ch,
                        BS=BS,
                        CBS=CBS,
                        BLOCK_P=_BWD_SV_BLOCK_P,
                        BLOCK_B=1,
                        IS_BATCHED=False,
                        allow_modify_flows=1 if allow_modify_flows else 0,
                        COMPUTE_PFLOWS=1 if compute_pflows else 0,
                        NEGATE_PFLOWS=1 if negate_pflows else 0,
                    )

            return None

        # Dense fallback: write to element_flows with the atomic-add kernel.
        # Zero the child range unless we're explicitly accumulating — matches
        # DenseSumLayer's accumulate_ch_flows contract (the kernel uses
        # atomic_add, so we must clear beforehand when NOT accumulating).
        assert batch_size == 1, (
            "SparseInputSumLayer dense-fallback backward (mixed-consumer "
            "topology) is B=1 only; the batched chain requires "
            "_skip_scatter=True on every upstream SparseProdLayer."
        )
        if not accumulate_ch_flows:
            for block, (sparse_prod, ns_idx) in zip(
                self._dense_blocks, self._sparse_input_refs,
            ):
                _nid_start, cid_start, _pid_start, _pfid_start, _NB, NB_ch, _BS, CBS = block
                element_flows[cid_start:cid_start + NB_ch * CBS, 0].zero_()

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._sparse_input_refs,
        )):
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, BS, CBS = block
            sv = sparse_prod._sparse_outputs[ns_idx]
            total_nnz = sv.total_nnz
            if total_nnz == 0:
                continue

            grid = (total_nnz, NB)
            _sparse_input_sum_backward_kernel[grid](
                node_flows_ptr=node_flows,
                node_mars_ptr=node_mars,
                mparams_ptr=params,
                indices_ptr=sv.indices,
                values_ptr=sv.values,
                element_flows_ptr=element_flows,
                nid_start=nid_start,
                cid_start=cid_start,
                pid_start=pid_start,
                batch_size=batch_size,
                NB_ch=NB_ch,
                BS=BS,
                CBS=CBS,
                allow_modify_flows=1 if allow_modify_flows else 0,
            )

        return None


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit(
    do_not_specialize=["nid_start", "pid_start", "total_nnz", "v_stride"],
    do_not_specialize_on_alignment=["indices_ptr", "values_ptr", "max_val_ptr",
                                     "col_starts_ptr", "nnz_ptr"],
)
def _sparse_input_sum_forward_kernel(
    node_mars_ptr, mparams_ptr,
    indices_ptr, values_ptr, max_val_ptr,
    col_starts_ptr, nnz_ptr,
    nid_start, pid_start,
    batch_size: tl.constexpr,
    total_nnz,
    v_stride,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr, IS_BATCHED: tl.constexpr,
):
    """
    Forward for one SparseInputSumLayer block. Grid
    ``(NB * (BS // TILE_M), cdiv(B, BLOCK_B))`` (axis 1 degenerate at B=1).

    Split-K reduction: the kernel compiles once for a fixed ``BLOCK_K`` and
    loops over ``ceil(total_nnz / BLOCK_K)`` chunks at runtime — the grid is
    purely parent-tile-shaped, so ``total_nnz`` never drives a recompile.
    ``nid_start`` / ``pid_start`` are runtime ints for the same reason
    (they change per layer block and we want one compiled kernel across all
    of them).

    Per-program (parent tile) math at ``IS_BATCHED == 0``:

      v_k = exp(log_values[k] - max_val)                       [per-chunk]
      W[m, k] = mparams[pid_start + (pblock * NB_ch + cblock_k) * CBS * BS
                                   + cslot_k * BS + parent_m]
      acc_sum[m] += Σ_{k ∈ chunk} W[m, k] · v_k                 [TILE_M]
      node_mars[nid_start + pblock * BS + parent_m, 0]
          = log(acc_sum[m] + 1e-24) + max_val

    where cblock_k = indices[k] // CBS and cslot_k = indices[k] % CBS.

    ``IS_BATCHED == 1``: each batch lane b reads its own column
    (``slot = col_starts[b] + k`` masked ``k < nnz[b]``, values from
    ``values[b * v_stride + k]``, per-sample ``max_val[b]`` clamped to
    ``-1e30`` so an empty column cannot poison the exp), the W tile widens
    to ``[TILE_M, BLOCK_B, BLOCK_K]``, ``total_nnz`` is the per-batch max
    (loop bound), stores land at ``node_mars[nid, b]``, and lanes with
    ``nnz[b] == 0`` store exact ``-inf`` (log of an empty sum).
    """
    pid_m = tl.program_id(0)
    pblock_id = pid_m // (BS // TILE_M)
    ntile_id = pid_m % (BS // TILE_M)

    offs_node = tl.arange(0, TILE_M) + ntile_id * TILE_M          # [TILE_M]
    off_nid = nid_start + pblock_id * BS + offs_node              # [TILE_M]

    if IS_BATCHED:
        pid_b = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)          # [BLOCK_B]
        mask_b = offs_b < batch_size
        cs_b = tl.load(col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_b = tl.load(nnz_ptr + offs_b, mask=mask_b, other=0)

        max_b = tl.load(max_val_ptr + offs_b, mask=mask_b, other=0.0)
        max_b = tl.maximum(max_b, -1e30)                          # empty col guard

        acc_sum = tl.zeros([TILE_M, BLOCK_B], dtype=tl.float32)
        for k0 in tl.range(0, total_nnz, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)                   # [BLOCK_K]
            mask_kb = mask_b[:, None] & (offs_k[None, :] < k_b[:, None])

            child_ids = tl.load(
                indices_ptr + cs_b[:, None] + offs_k[None, :],
                mask=mask_kb, other=0,
            )                                                     # [BLOCK_B, BLOCK_K]
            log_vals = tl.load(
                values_ptr + offs_b[:, None] * v_stride + offs_k[None, :],
                mask=mask_kb, other=-float("inf"),
            )

            v_k = tl.where(mask_kb, tl.exp(log_vals - max_b[:, None]), 0.0)

            cblock = child_ids // CBS
            cslot = child_ids % CBS

            W_ptr_off = (
                pid_start
                + (pblock_id * NB_ch + cblock[None, :, :]) * CBS * BS
                + cslot[None, :, :] * BS
                + offs_node[:, None, None]
            )                                                     # [TILE_M, BLOCK_B, BLOCK_K]
            W = tl.load(mparams_ptr + W_ptr_off,
                        mask=mask_kb[None, :, :], other=0.0).to(tl.float32)

            acc_sum += tl.sum(W * v_k[None, :, :], axis=2)        # [TILE_M, BLOCK_B]

        result = tl.log(acc_sum + 1e-24) + max_b[None, :]
        result = tl.where(k_b[None, :] == 0, -float("inf"), result)

        # int64 cast: num_nodes * B crosses 2^31 at (H=32k, B=1024)-scale
        # circuits — same overflow class as the DenseSumLayer parent-block
        # address fix.
        tl.store(
            node_mars_ptr
            + off_nid[:, None].to(tl.int64) * batch_size + offs_b[None, :],
            result, mask=mask_b[None, :],
        )
    else:
        max_val = tl.load(max_val_ptr)                            # scalar

        acc_sum = tl.zeros([TILE_M], dtype=tl.float32)
        for k0 in tl.range(0, total_nnz, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)                   # [BLOCK_K]
            mask_k = offs_k < total_nnz

            child_ids = tl.load(indices_ptr + offs_k, mask=mask_k, other=0)
            log_vals = tl.load(values_ptr + offs_k, mask=mask_k,
                               other=-float("inf"))

            v_k = tl.where(mask_k, tl.exp(log_vals - max_val), 0.0)  # [BLOCK_K]

            cblock = child_ids // CBS
            cslot = child_ids % CBS

            W_ptr_off = (
                pid_start
                + (pblock_id * NB_ch + cblock[None, :]) * CBS * BS
                + cslot[None, :] * BS
                + offs_node[:, None]
            )
            W = tl.load(mparams_ptr + W_ptr_off,
                        mask=mask_k[None, :], other=0.0).to(tl.float32)

            acc_sum += tl.sum(W * v_k[None, :], axis=1)           # [TILE_M]

        result = tl.log(acc_sum + 1e-24) + max_val

        # B=1: node_mars[off_nid, 0] = node_mars_ptr[off_nid * batch_size + 0]
        tl.store(node_mars_ptr + off_nid * batch_size, result)


@triton.jit(
    do_not_specialize=["nid_start", "cid_start", "pid_start"],
    do_not_specialize_on_alignment=["indices_ptr", "values_ptr"],
)
def _sparse_input_sum_backward_kernel(
    node_flows_ptr, node_mars_ptr, mparams_ptr,
    indices_ptr, values_ptr, element_flows_ptr,
    nid_start, cid_start, pid_start,
    batch_size: tl.constexpr,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    allow_modify_flows: tl.constexpr,
):
    """
    Backward for one SparseInputSumLayer block, gated to batch_size == 1 and
    propagation_alg == "LL". Grid = (total_nnz, NB).

    Per-program math:

      For a single active child c_k = indices[k] and one parent block pblock:
        contribution[p] = nflow[p] · W[p, c_k] · exp(log_val_k - nmars[p])
      partial_flow = Σ_{p in pblock} contribution[p]
      element_flows[cid_start + c_k, 0] += partial_flow   (atomic)

    When ``allow_modify_flows`` was set by TensorCircuit, ``nflow[p]`` has been
    pre-transformed to ``log(flow) - nmars``; the contribution is rewritten
    analytically to ``exp(nflow[p] + log_val_k) · W[p, c_k]``.
    """
    pid_k = tl.program_id(0)
    pid_pblock = tl.program_id(1)

    child_id = tl.load(indices_ptr + pid_k)
    log_val = tl.load(values_ptr + pid_k)
    cblock = child_id // CBS
    cslot = child_id % CBS

    p_range = tl.arange(0, BS)                                    # [BS]
    p_nid = nid_start + pid_pblock * BS + p_range
    p_addr = p_nid * batch_size                                   # B=1, offs_b=0

    nflows = tl.load(node_flows_ptr + p_addr)                     # [BS]
    nmars = tl.load(node_mars_ptr + p_addr)                       # [BS]

    W_ptr = pid_start + (pid_pblock * NB_ch + cblock) * CBS * BS + cslot * BS + p_range
    W = tl.load(mparams_ptr + W_ptr).to(tl.float32)               # [BS]

    if allow_modify_flows:
        contribution = tl.exp(nflows + log_val) * W
    else:
        contribution = nflows * W * tl.exp(log_val - nmars)

    partial_flow = tl.sum(contribution)

    tl.atomic_add(
        element_flows_ptr + (cid_start + child_id) * batch_size,
        partial_flow,
    )


@triton.jit(
    do_not_specialize=["nid_start", "pid_start", "pfid_start", "v_stride",
                       "f_stride"],
    do_not_specialize_on_alignment=[
        "indices_ptr", "values_ptr", "flow_out_ptr", "pflows_ptr",
        "col_starts_ptr", "nnz_ptr",
    ],
)
def _sparse_input_sum_backward_sv_kernel(
    node_flows_ptr, node_mars_ptr, mparams_ptr,
    indices_ptr, values_ptr, flow_out_ptr,
    pflows_ptr,
    col_starts_ptr, nnz_ptr,
    nid_start, pid_start, pfid_start,
    batch_size: tl.constexpr,
    v_stride, f_stride,
    NB: tl.constexpr, NB_ch: tl.constexpr,
    BS: tl.constexpr, CBS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_B: tl.constexpr, IS_BATCHED: tl.constexpr,
    allow_modify_flows: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr, NEGATE_PFLOWS: tl.constexpr,
):
    """
    Sparse-fast-path backward: writes ``flow_out[b, k] = Σ_p nflow[p, b] ·
    W[p, c_k] · exp(log_val[b, k] - nmars[p, b])`` for each active child
    ``c_k`` of each sample. Grid ``(K_max, cdiv(B, BLOCK_B))`` — no atomics
    on flow_out, each program owns its ``(k, batch-tile)`` slots. Parent
    reduction is serial over ``NB`` blocks × ``BS / BLOCK_P`` inner tiles so
    register pressure stays bounded at ``bs=1024``.

    ``IS_BATCHED == 1``: each batch lane reads its own column
    (``c_k = indices[col_starts[b] + k]``, masked ``k < nnz[b]``); a program
    whose ``k`` is beyond every lane's ``nnz[b]`` exits before the serial
    loop. At B=1 the classic single-column body runs unchanged.

    When ``allow_modify_flows`` is set, ``nflow`` is already the
    pre-transformed ``log(flow) - nmars``; contribution becomes
    ``exp(nflow + log_val) * W`` (matches the dense-atomic kernel above).

    When ``COMPUTE_PFLOWS == 1``, the contribution is also scattered into
    ``param_flows[pfid_start + (pblock*NB_ch + cblock) * CBS * BS + cslot *
    BS + p]`` via ``atomic_add`` — same address scheme as the params buffer.
    Programs with the same (child_id, parent) address collide atomically;
    at B>1 that includes lanes of *different samples* observing the same
    token — the atomic IS the batch reduction (EM pflows sum over samples).
    """
    pid_k = tl.program_id(0)

    if IS_BATCHED:
        pid_b = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)          # [BLOCK_B]
        mask_b = offs_b < batch_size
        cs_b = tl.load(col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_b = tl.load(nnz_ptr + offs_b, mask=mask_b, other=0)
        lane_mask = mask_b & (pid_k < k_b)                        # [BLOCK_B]

        n_active = tl.sum(lane_mask.to(tl.int32), axis=0)
        if n_active > 0:
            child_id = tl.load(indices_ptr + cs_b + pid_k,
                               mask=lane_mask, other=0)           # [BLOCK_B]
            log_val = tl.load(values_ptr + offs_b * v_stride + pid_k,
                              mask=lane_mask, other=-float("inf"))
            cblock = child_id // CBS
            cslot = child_id % CBS

            acc = tl.zeros([BLOCK_P, BLOCK_B], dtype=tl.float32)
            for pblock in tl.range(0, NB):
                edge_block_base = pid_start \
                    + (pblock * NB_ch + cblock) * CBS * BS + cslot * BS
                pf_block_base = pfid_start \
                    + (pblock * NB_ch + cblock) * CBS * BS + cslot * BS
                parent_base = nid_start + pblock * BS

                for p0 in tl.range(0, BS, BLOCK_P):
                    offs_p = p0 + tl.arange(0, BLOCK_P)           # [BLOCK_P]
                    mask_p = offs_p < BS
                    tile_mask = mask_p[:, None] & lane_mask[None, :]

                    # int64: parent nid * B crosses 2^31 at large H * B.
                    p_addr = (parent_base + offs_p[:, None]).to(tl.int64) \
                        * batch_size + offs_b[None, :]
                    nflows = tl.load(node_flows_ptr + p_addr,
                                     mask=tile_mask, other=0.0)
                    W = tl.load(
                        mparams_ptr + edge_block_base[None, :] + offs_p[:, None],
                        mask=tile_mask, other=0.0,
                    ).to(tl.float32)

                    if allow_modify_flows:
                        chunk = tl.exp(nflows + log_val[None, :]) * W
                    else:
                        nmars = tl.load(node_mars_ptr + p_addr,
                                        mask=tile_mask, other=0.0)
                        chunk = nflows * W * tl.exp(log_val[None, :] - nmars)

                    chunk = tl.where(tile_mask, chunk, 0.0)
                    acc += chunk

                    if COMPUTE_PFLOWS == 1:
                        pf_val = -chunk if NEGATE_PFLOWS == 1 else chunk
                        tl.atomic_add(
                            pflows_ptr + pf_block_base[None, :] + offs_p[:, None],
                            pf_val,
                            mask=tile_mask,
                        )

            tl.store(flow_out_ptr + offs_b * f_stride + pid_k,
                     tl.sum(acc, axis=0), mask=lane_mask)
    else:
        child_id = tl.load(indices_ptr + pid_k)
        log_val = tl.load(values_ptr + pid_k)
        cblock = child_id // CBS
        cslot = child_id % CBS

        # Accumulate contributions across (pblock, p0) tiles on BLOCK_P
        # parallel lanes; final reduction via tl.sum folds into one scalar
        # per program.
        acc = tl.zeros([BLOCK_P], dtype=tl.float32)

        for pblock in tl.range(0, NB):
            # Edge block base for this (pblock, cblock).
            edge_block_base = pid_start + (pblock * NB_ch + cblock) * CBS * BS + cslot * BS
            # Param-flow block base mirrors the param layout exactly.
            pf_block_base = pfid_start + (pblock * NB_ch + cblock) * CBS * BS + cslot * BS
            # Parent-node base for this pblock (B=1 → address == nid).
            parent_base = nid_start + pblock * BS

            for p0 in tl.range(0, BS, BLOCK_P):
                offs_p = p0 + tl.arange(0, BLOCK_P)               # [BLOCK_P]
                mask_p = offs_p < BS

                p_addr = (parent_base + offs_p) * batch_size      # [BLOCK_P]
                nflows = tl.load(node_flows_ptr + p_addr, mask=mask_p, other=0.0)
                W = tl.load(
                    mparams_ptr + edge_block_base + offs_p,
                    mask=mask_p, other=0.0,
                ).to(tl.float32)

                if allow_modify_flows:
                    chunk = tl.exp(nflows + log_val) * W
                else:
                    nmars = tl.load(
                        node_mars_ptr + p_addr,
                        mask=mask_p, other=0.0,
                    )
                    chunk = nflows * W * tl.exp(log_val - nmars)

                chunk = tl.where(mask_p, chunk, 0.0)
                acc += chunk

                if COMPUTE_PFLOWS == 1:
                    pf_val = -chunk if NEGATE_PFLOWS == 1 else chunk
                    tl.atomic_add(
                        pflows_ptr + pf_block_base + offs_p,
                        pf_val,
                        mask=mask_p,
                    )

        tl.store(flow_out_ptr + pid_k, tl.sum(acc))
