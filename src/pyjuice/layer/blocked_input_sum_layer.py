from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .dense_sum_layer import DenseSumLayer
from .blocked_prod_layer import BlockedProdLayer


_FWD_BLOCK_K = 64
"""Child-axis (``j`` within the active block) chunk of the forward kernels."""

_BWD_BLOCK_P = 64
"""Parent-axis chunk of the backward kernels (bounds the serial reduction's
register footprint)."""

_BWD_TILE_J = 32
"""Child-axis tile of the backward kernels (one program per
``[TILE_J]`` slice of a sample's active block)."""


def _pow2_tile(n: int, cap: int) -> int:
    """Largest power of two ``<= min(n, cap)`` that divides ``n`` when ``n`` is
    itself a power of two; otherwise ``next_pow2(n)`` capped (masked tail)."""
    t = min(triton.next_power_of_2(n), cap)
    return max(t, 1)


class BlockedInputSumLayer(DenseSumLayer):
    """
    Blocked-in / **dense**-out sum layer: single child compiled as a
    :class:`BlockedProdLayer`. Forward reads the child's packed
    :class:`BlockedNodeValues` (``k`` active children per sample) and writes
    every parent of ``node_mars``::

        node_mars[h, b] = log Σ_{j<k} W[h, m_b·k + j] · exp(values[b, j])

    i.e. one ``[NB·BS, k]`` column stripe of the block-dense parameter matrix
    per sample instead of the full ``[NB·BS, NB_ch·CBS]`` GEMV — the blocked
    twin of :class:`SparseInputSumLayer`, used by the root sum of an HMM
    chain (and by any blocked sum whose consumers are not blocked-aware).

    Any batch size; ``propagation_alg == 'LL'``. Backward writes the upstream
    prod's flow container directly and (optionally) the parameter flows of the
    touched stripe. The dense-``element_flows`` fallback for mixed-consumer
    topologies delegates to :class:`DenseSumLayer`.
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
            "BlockedInputSumLayer needs the already-compiled inner_layer_groups "
            "to resolve the upstream BlockedProdLayer that owns each sum's child."
        )
        self._build_blocked_input_refs(inner_layer_groups)

        for block, (prod, ns_idx) in zip(self._dense_blocks, self._blocked_input_refs):
            _nid, _cid, _pid, _pfid, NB, NB_ch, BS, CBS = block
            k = prod.nodes[ns_idx].k
            assert k % CBS == 0, (
                f"BlockedInputSumLayer: k={k} must be a multiple of ch_block_size={CBS}."
            )
            assert NB_ch * CBS == prod.nodes[ns_idx].num_nodes

        # Backward flow workspaces (``[B, k]`` per block), lazily allocated.
        self._bwd_flow_workspaces: List[Optional[torch.Tensor]] = [None] * len(self._dense_blocks)

    def _build_blocked_input_refs(self, inner_layer_groups: list) -> None:
        self._blocked_input_refs: List[Tuple[BlockedProdLayer, int]] = []
        for ns in self.nodes:
            assert len(ns.chs) == 1, "BlockedInputSumLayer requires num_chs == 1 per SumNodes."
            cs = ns.chs[0]
            found = None
            for lg in inner_layer_groups:
                if not lg.is_prod():
                    continue
                for layer in lg:
                    if not isinstance(layer, BlockedProdLayer):
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
                "BlockedInputSumLayer: child ProdNodes is not owned by any "
                "BlockedProdLayer in the already-compiled inner_layer_groups."
            )
            self._blocked_input_refs.append(found)

    def __repr__(self) -> str:
        return (
            f"BlockedInputSumLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sum_ns={len(self._blocked_input_refs)})"
        )

    def _all_skip_scatter(self) -> bool:
        return all(p._skip_scatter for p, _ in self._blocked_input_refs)

    def _flow_workspace(self, blk_idx: int, batch_size: int, k: int,
                        device: torch.device) -> torch.Tensor:
        ws = self._bwd_flow_workspaces[blk_idx]
        if ws is None or ws.device != device or ws.shape != (batch_size, k):
            ws = torch.empty(batch_size, k, dtype=torch.float32, device=device)
            self._bwd_flow_workspaces[blk_idx] = ws
        return ws

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, force_use_bf16: bool = False,
                force_use_fp32: bool = False, propagation_alg: str = "LL",
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL", "BlockedInputSumLayer requires propagation_alg == 'LL'."
        assert params.dim() == 1

        for blk_idx, (block, (prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._blocked_input_refs,
        )):
            nid_start, _cid_start, pid_start, _pfid_start, NB, NB_ch, BS, CBS = block
            bv = prod._blocked_outputs[ns_idx]
            assert bv is not None, "upstream BlockedProdLayer has no cached output."
            k = bv.k
            assert bv.max_val is not None

            TILE_M = min(BS, 32)
            while BS % TILE_M != 0 and TILE_M > 1:
                TILE_M //= 2
            BLOCK_K = _pow2_tile(k, _FWD_BLOCK_K)
            grid = (NB * (BS // TILE_M), batch_size)
            _blocked_input_sum_forward_kernel[grid](
                node_mars_ptr=node_mars,
                mparams_ptr=params,
                values_ptr=bv.values,
                max_val_ptr=bv.max_val,
                data_ptr=bv.data,
                token_block_ptr=bv.token_block,
                var_id=bv.var_id,
                nid_start=nid_start,
                pid_start=pid_start,
                batch_size=batch_size,
                K=k,
                NB_ch=NB_ch, BS=BS, CBS=CBS,
                TILE_M=TILE_M, BLOCK_K=BLOCK_K,
            )
        return None

    # ------------------------------------------------------------------ #
    # Backward
    # ------------------------------------------------------------------ #

    def _modify_flows_prepass(self, node_flows, node_mars, propagation_alg, batch_size):
        """Replicates ``DenseSumLayer.backward``'s ``log(flow) - node_mars``
        pre-transform (the fast path skips ``super().backward``)."""
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg)
        alpha = float(propagation_alg_kwargs.get("alpha", 0.0))
        for block in self._dense_blocks:
            nid_start, _cid_start, _pid_start, _pfid_start, NB, _NB_ch, bs, _cbs = block
            layer_n_nodes = NB * bs
            BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
            BLOCK_B = min(2048, BATCH_SIZE_NP2)
            BLOCK_M = max(min(max(2048 // BLOCK_B, 1), bs), 1)
            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, BLOCK_M))
            self._bk_triton_dense_modify_flow_kernel[grid](
                node_flows=node_flows, node_mars=node_mars,
                nid_start=nid_start, batch_size=batch_size,
                num_parents=layer_n_nodes,
                BLOCK_B=BLOCK_B, BLOCK_M=BLOCK_M,
                propagation_alg_id=propagation_alg_id, alpha=alpha,
            )

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 allow_modify_flows: bool = False, propagation_alg: str = "LL",
                 logspace_flows: bool = False, negate_pflows: bool = False,
                 accumulate_ch_flows: bool = False, allow_neg_flows: bool = False,
                 force_use_fp32: bool = False, **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL" and not logspace_flows and not allow_neg_flows, (
            "BlockedInputSumLayer.backward requires propagation_alg='LL' + "
            "logspace_flows=False + allow_neg_flows=False."
        )
        compute_pflows = param_flows is not None

        if not self._all_skip_scatter():
            # Mixed-consumer topology: the upstream prod scattered its values
            # into element_mars, so the plain dense backward applies (the prod
            # layer gathers element_flows afterwards).
            super().backward(
                node_flows=node_flows, element_flows=element_flows,
                node_mars=node_mars, element_mars=element_mars, params=params,
                param_flows=param_flows, allow_modify_flows=allow_modify_flows,
                propagation_alg=propagation_alg, logspace_flows=logspace_flows,
                negate_pflows=negate_pflows, accumulate_ch_flows=accumulate_ch_flows,
                allow_neg_flows=allow_neg_flows, force_use_fp32=force_use_fp32,
                **kwargs,
            )
            return None

        if accumulate_ch_flows:
            raise NotImplementedError(
                "BlockedInputSumLayer (skip_scatter) backward writes the flow "
                "container straight into the upstream prod layer; "
                "accumulate_ch_flows=True is not supported."
            )
        if allow_modify_flows:
            self._modify_flows_prepass(node_flows, node_mars, propagation_alg, batch_size)

        for blk_idx, (block, (prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._blocked_input_refs,
        )):
            nid_start, _cid_start, pid_start, pfid_start, NB, NB_ch, BS, CBS = block
            bv = prod._blocked_outputs[ns_idx]
            k = bv.k
            flow_values = self._flow_workspace(blk_idx, batch_size, k, node_mars.device)
            bv_flow = bv.like_pattern(flow_values)
            prod._blocked_flows[ns_idx] = bv_flow

            TILE_J = _pow2_tile(k, _BWD_TILE_J)
            grid = (triton.cdiv(k, TILE_J), batch_size)
            _blocked_input_sum_backward_kernel[grid](
                node_flows_ptr=node_flows,
                node_mars_ptr=node_mars,
                mparams_ptr=params,
                values_ptr=bv.values,
                flow_out_ptr=bv_flow.values,
                pflows_ptr=(param_flows if compute_pflows else bv_flow.values),
                data_ptr=bv.data,
                token_block_ptr=bv.token_block,
                var_id=bv.var_id,
                nid_start=nid_start,
                pid_start=pid_start,
                pfid_start=pfid_start,
                batch_size=batch_size,
                K=k,
                num_parents=NB * BS,
                NB_ch=NB_ch, BS=BS, CBS=CBS,
                TILE_J=TILE_J, BLOCK_P=_BWD_BLOCK_P,
                allow_modify_flows=1 if allow_modify_flows else 0,
                COMPUTE_PFLOWS=1 if compute_pflows else 0,
                NEGATE_PFLOWS=1 if negate_pflows else 0,
                ATOMIC_PFLOWS=1 if batch_size > 1 else 0,
            )
        return None


# =====================================================================
# Triton kernels
# =====================================================================
#
# Parameter addressing (shared with DenseSumLayer's flat layout): for parent
# ``h`` and child ``c``,
#   W[h, c] = mparams[pid_start + (h // BS * NB_ch + c // CBS) * CBS * BS
#                              + (c % CBS) * BS + h % BS]
# With ``c = m_b * K + j`` and ``K % CBS == 0`` every K-chunk of ``j`` lies in
# one child block and is contiguous along ``h % BS`` — coalesced loads, no
# index arrays.


@triton.jit(
    do_not_specialize=["var_id", "nid_start", "pid_start", "batch_size", "K"],
)
def _blocked_input_sum_forward_kernel(
    node_mars_ptr, mparams_ptr,
    values_ptr, max_val_ptr,
    data_ptr, token_block_ptr,
    var_id, nid_start, pid_start, batch_size, K,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Grid ``(NB * BS / TILE_M, B)``: one program per (parent tile, sample).
    ``node_mars[nid_start + h, b] = log(Σ_j W[h, m_b·K + j] · exp(v[b, j] − max_b)) + max_b``."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1)

    offs_h = pid_m * TILE_M + tl.arange(0, TILE_M)                  # [TILE_M]
    pblock = offs_h // BS
    within = offs_h % BS

    v = tl.load(data_ptr + var_id * batch_size + b)
    m = tl.load(token_block_ptr + v)
    max_val = tl.load(max_val_ptr + b)
    max_val = tl.maximum(max_val, -1e30)

    acc = tl.zeros([TILE_M], dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_j = k0 + tl.arange(0, BLOCK_K)                         # [BLOCK_K]
        mask_j = offs_j < K
        c = m * K + offs_j
        cblock = c // CBS
        cslot = c % CBS
        log_vals = tl.load(values_ptr + b * K + offs_j, mask=mask_j, other=-float("inf"))
        vk = tl.where(mask_j, tl.exp(log_vals - max_val), 0.0)
        W_off = (
            pid_start
            + (pblock[:, None] * NB_ch + cblock[None, :]) * CBS * BS
            + cslot[None, :] * BS
            + within[:, None]
        )                                                           # [TILE_M, BLOCK_K]
        W = tl.load(mparams_ptr + W_off, mask=mask_j[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(W * vk[None, :], axis=1)

    result = tl.log(acc + 1e-24) + max_val
    tl.store(node_mars_ptr + (nid_start + offs_h) * batch_size + b, result)


@triton.jit(
    do_not_specialize=["var_id", "nid_start", "pid_start", "pfid_start",
                       "batch_size", "K", "num_parents"],
)
def _blocked_input_sum_backward_kernel(
    node_flows_ptr, node_mars_ptr, mparams_ptr,
    values_ptr, flow_out_ptr, pflows_ptr,
    data_ptr, token_block_ptr,
    var_id, nid_start, pid_start, pfid_start, batch_size, K, num_parents,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_J: tl.constexpr, BLOCK_P: tl.constexpr,
    allow_modify_flows: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr, NEGATE_PFLOWS: tl.constexpr,
    ATOMIC_PFLOWS: tl.constexpr,
):
    """Grid ``(cdiv(K, TILE_J), B)``: one program per (child tile, sample).

      chunk[h, j] = nflow[h, b] · W[h, m_b·K + j] · exp(v[b, j] − nmars[h, b])
      flow_out[b, j] = Σ_h chunk[h, j]
      pflows[W-address(h, m_b·K + j)] += chunk[h, j]      (if COMPUTE_PFLOWS)

    ``allow_modify_flows``: ``node_flows`` was pre-transformed to
    ``log(flow) − nmars`` so ``chunk = exp(nflow + v) · W``.

    Param flows: at B=1 every (h, j) address is touched by exactly one program
    of this launch and launches are stream-ordered, so a plain RMW is safe
    (``ATOMIC_PFLOWS=0``); at B>1 samples sharing a block collide and the
    atomic is the cross-sample reduction."""
    pid_j = tl.program_id(0)
    b = tl.program_id(1)

    offs_j = pid_j * TILE_J + tl.arange(0, TILE_J)                  # [TILE_J]
    mask_j = offs_j < K

    v = tl.load(data_ptr + var_id * batch_size + b)
    m = tl.load(token_block_ptr + v)
    c = m * K + offs_j
    cblock = c // CBS
    cslot = c % CBS
    log_val = tl.load(values_ptr + b * K + offs_j, mask=mask_j, other=-float("inf"))

    acc = tl.zeros([TILE_J], dtype=tl.float32)
    for h0 in tl.range(0, num_parents, BLOCK_P):
        offs_h = h0 + tl.arange(0, BLOCK_P)                         # [BLOCK_P]
        mask_h = offs_h < num_parents
        pblock = offs_h // BS
        within = offs_h % BS
        nflow = tl.load(node_flows_ptr + (nid_start + offs_h) * batch_size + b,
                        mask=mask_h, other=0.0)
        nmars = tl.load(node_mars_ptr + (nid_start + offs_h) * batch_size + b,
                        mask=mask_h, other=0.0)
        W_off = (
            pid_start
            + (pblock[:, None] * NB_ch + cblock[None, :]) * CBS * BS
            + cslot[None, :] * BS
            + within[:, None]
        )                                                           # [BLOCK_P, TILE_J]
        mask_hj = mask_h[:, None] & mask_j[None, :]
        W = tl.load(mparams_ptr + W_off, mask=mask_hj, other=0.0).to(tl.float32)
        if allow_modify_flows == 1:
            chunk = tl.exp(nflow[:, None] + log_val[None, :]) * W
        else:
            chunk = nflow[:, None] * W * tl.exp(log_val[None, :] - nmars[:, None])
        chunk = tl.where(mask_hj, chunk, 0.0)
        acc += tl.sum(chunk, axis=0)

        if COMPUTE_PFLOWS == 1:
            pf_off = (
                pfid_start
                + (pblock[:, None] * NB_ch + cblock[None, :]) * CBS * BS
                + cslot[None, :] * BS
                + within[:, None]
            )
            pf_val = -chunk if NEGATE_PFLOWS == 1 else chunk
            if ATOMIC_PFLOWS == 1:
                tl.atomic_add(pflows_ptr + pf_off, pf_val, mask=mask_hj)
            else:
                old = tl.load(pflows_ptr + pf_off, mask=mask_hj, other=0.0)
                tl.store(pflows_ptr + pf_off, old + pf_val, mask=mask_hj)

    tl.store(flow_out_ptr + b * K + offs_j, acc, mask=mask_j)
