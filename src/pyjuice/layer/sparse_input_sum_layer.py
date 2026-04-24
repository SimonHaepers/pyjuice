from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .dense_sum_layer import DenseSumLayer
from .sparse_prod_layer import SparseProdLayer


_FWD_BLOCK_K = 64
"""Fixed K-axis tile size for :func:`_sparse_input_sum_forward_kernel`.

The forward kernel reduces over ``total_nnz`` active child slots. Keeping
``BLOCK_K`` a module-level constant means the kernel compiles **once** —
independent of per-call nnz — so we can ship AOT-compiled kernels for
deployment on platforms without a Triton JIT. Tuning knob: larger = fewer
inner-loop iterations but more register pressure per program.
"""


class SparseInputSumLayer(DenseSumLayer):
    """
    Block-dense sum layer whose single child is a :class:`SparseProdLayer`.
    Forward/backward read the upstream :class:`SparseNodeValues` directly
    (O(nnz·M)) instead of the full element_mars tile (O(H·M)).

    Inference-only + **batch_size == 1 only** (the underlying sparse fast path
    is B=1 by design, matching :class:`SparseProdLayer`). For B>1 circuits,
    build with plain `summate` / `multiply` (or ``_force_plain=True``).
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
        assert batch_size == 1, (
            "SparseInputSumLayer is B=1 only. Build with plain `summate` "
            "(or `_force_plain=True`) for B>1."
        )
        assert propagation_alg == "LL", (
            "SparseInputSumLayer requires propagation_alg == 'LL'."
        )

        assert params.dim() == 1

        torch.cuda.nvtx.range_push(
            f"SparseInputSumLayer.fwd(n_blocks={len(self._dense_blocks)})"
        )
        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._sparse_input_refs,
        )):
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, BS, CBS = block
            sv = sparse_prod._sparse_outputs[ns_idx]
            total_nnz = sv.total_nnz
            if total_nnz == 0:
                # Empty active column: log(0) = -inf for all parents.
                torch.cuda.nvtx.range_push(f"block[{blk_idx}]/empty_col_fill")
                node_mars[nid_start:nid_start + NB * BS, 0].fill_(float("-inf"))
                torch.cuda.nvtx.range_pop()
                continue

            torch.cuda.nvtx.range_push(f"block[{blk_idx}]")
            # Per-call max of sparse values (for linear-sum numerical stability).
            # `values` is already length `total_nnz` (no padding); scalar reduction
            # on device keeps the kernel pure.
            torch.cuda.nvtx.range_push("values.max()")
            max_val = sv.values.max()
            torch.cuda.nvtx.range_pop()

            # Tile parents: TILE_M divides BS and is a power of 2.
            TILE_M = min(BS, 32)
            while BS % TILE_M != 0 and TILE_M > 1:
                TILE_M //= 2

            torch.cuda.nvtx.range_push(f"_sparse_input_sum_fwd_kernel(nnz={total_nnz})")
            grid = (NB * (BS // TILE_M),)
            _sparse_input_sum_forward_kernel[grid](
                node_mars_ptr=node_mars,
                mparams_ptr=params,
                indices_ptr=sv.indices,
                values_ptr=sv.values,
                max_val_ptr=max_val,
                nid_start=nid_start,
                pid_start=pid_start,
                batch_size=batch_size,
                total_nnz=total_nnz,
                NB_ch=NB_ch,
                BS=BS,
                CBS=CBS,
                TILE_M=TILE_M,
                BLOCK_K=_FWD_BLOCK_K,
            )
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

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

        if param_flows is not None:
            raise NotImplementedError(
                "SparseInputSumLayer (like DenseSumLayer) is inference-only; "
                "parameter-flow accumulation is not supported."
            )

        batch_size = node_mars.size(1)
        assert batch_size == 1, (
            "SparseInputSumLayer is B=1 only. Build with plain `summate` "
            "(or `_force_plain=True`) for B>1."
        )
        assert propagation_alg == "LL" and not logspace_flows and not negate_pflows \
               and not allow_neg_flows, (
            "SparseInputSumLayer backward requires propagation_alg='LL' + "
            "logspace_flows=False + negate_pflows=False + allow_neg_flows=False."
        )

        torch.cuda.nvtx.range_push(
            f"SparseInputSumLayer.bwd(n_blocks={len(self._dense_blocks)})"
        )

        # Zero the child range unless we're explicitly accumulating — matches
        # DenseSumLayer's accumulate_ch_flows contract (the kernel uses
        # atomic_add, so we must clear beforehand when NOT accumulating).
        if not accumulate_ch_flows:
            torch.cuda.nvtx.range_push("zero_child_ranges(pytorch .zero_())")
            for block, (sparse_prod, ns_idx) in zip(
                self._dense_blocks, self._sparse_input_refs,
            ):
                _nid_start, cid_start, _pid_start, _pfid_start, _NB, NB_ch, _BS, CBS = block
                element_flows[cid_start:cid_start + NB_ch * CBS, 0].zero_()
            torch.cuda.nvtx.range_pop()

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._sparse_input_refs,
        )):
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, BS, CBS = block
            sv = sparse_prod._sparse_outputs[ns_idx]
            total_nnz = sv.total_nnz
            if total_nnz == 0:
                continue

            torch.cuda.nvtx.range_push(
                f"block[{blk_idx}]/_sparse_input_sum_bwd_kernel(nnz={total_nnz})"
            )
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
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()
        return None


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit(
    do_not_specialize=["nid_start", "pid_start", "total_nnz"],
    do_not_specialize_on_alignment=["indices_ptr", "values_ptr", "max_val_ptr"],
)
def _sparse_input_sum_forward_kernel(
    node_mars_ptr, mparams_ptr,
    indices_ptr, values_ptr, max_val_ptr,
    nid_start, pid_start,
    batch_size: tl.constexpr,
    total_nnz,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Forward for one SparseInputSumLayer block, gated to batch_size == 1.

    Split-K reduction: the kernel compiles once for a fixed ``BLOCK_K`` and
    loops over ``ceil(total_nnz / BLOCK_K)`` chunks at runtime — the grid is
    purely parent-tile-shaped, so ``total_nnz`` never drives a recompile.
    ``nid_start`` / ``pid_start`` are runtime ints for the same reason
    (they change per layer block and we want one compiled kernel across all
    of them).

    Per-program (parent tile) math:

      v_k = exp(log_values[k] - max_val)                       [per-chunk]
      W[m, k] = mparams[pid_start + (pblock * NB_ch + cblock_k) * CBS * BS
                                   + cslot_k * BS + parent_m]
      acc_sum[m] += Σ_{k ∈ chunk} W[m, k] · v_k                 [TILE_M]
      node_mars[nid_start + pblock * BS + parent_m, 0]
          = log(acc_sum[m] + 1e-24) + max_val

    where cblock_k = indices[k] // CBS and cslot_k = indices[k] % CBS.
    """
    pid_m = tl.program_id(0)
    pblock_id = pid_m // (BS // TILE_M)
    ntile_id = pid_m % (BS // TILE_M)

    offs_node = tl.arange(0, TILE_M) + ntile_id * TILE_M          # [TILE_M]
    off_nid = nid_start + pblock_id * BS + offs_node              # [TILE_M]

    max_val = tl.load(max_val_ptr)                                # scalar

    acc_sum = tl.zeros([TILE_M], dtype=tl.float32)
    for k0 in tl.range(0, total_nnz, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                       # [BLOCK_K]
        mask_k = offs_k < total_nnz

        child_ids = tl.load(indices_ptr + offs_k, mask=mask_k, other=0)
        log_vals = tl.load(values_ptr + offs_k, mask=mask_k,
                           other=-float("inf"))

        v_k = tl.where(mask_k, tl.exp(log_vals - max_val), 0.0)   # [BLOCK_K]

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

        acc_sum += tl.sum(W * v_k[None, :], axis=1)               # [TILE_M]

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
