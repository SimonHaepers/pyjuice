from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .blocked_input_sum_layer import BlockedInputSumLayer, _pow2_tile
from .blocked_node_values import BlockedNodeValues


_FWD_TILE_M = 32
_FWD_BLOCK_K = 64
_BWD_TILE_J = 32
_BWD_BLOCK_P = 64


class BlockedIOSumLayer(BlockedInputSumLayer):
    """Blocked-in / blocked-out sum layer — the interior transition of a
    blocked-emission HMM chain (the twin of :class:`SparseIOSumLayer`).

    The sole consumer is a :class:`CoBlockedProdLayer` whose
    :class:`BlockedCategorical` input at ``out_var`` selects the *output*
    block, while the child :class:`BlockedProdLayer`'s input at ``in_var``
    selects the *input* block. Per sample the sum therefore touches one
    ``k_out × k_in`` tile of the block-dense parameter matrix::

        out[b, m] = log Σ_{j<k_in} W[m_out·k_out + m, m_in·k_in + j] · exp(in[b, j])
        m_in  = token_block_in[data[in_var, b]],  m_out = token_block_out[data[out_var, b]]

    ``node_mars`` is never written; the consumer reads
    ``self._blocked_outputs[blk_idx]``. Backward reads the consumer's packed
    flow from ``self._blocked_flows[blk_idx]`` and writes the upstream prod's
    flow container plus (optionally) the tile's parameter flows.
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
                 output_var_ids: Optional[Sequence[int]] = None,
                 output_dists: Optional[Sequence] = None,
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
        assert output_var_ids is not None and len(output_var_ids) == len(self.nodes), (
            "BlockedIOSumLayer requires one `output_var_id` per sum node (the "
            "consumer CoBlockedProdNodes' var_id)."
        )
        assert output_dists is not None and len(output_dists) == len(self.nodes), (
            "BlockedIOSumLayer requires one `output_dist` (the consumer's "
            "BlockedCategorical) per sum node."
        )
        self._output_var_ids: List[int] = list(output_var_ids)
        self._output_dists = list(output_dists)

        for block, dist in zip(self._dense_blocks, self._output_dists):
            _nid, _cid, _pid, _pfid, NB, _NB_ch, BS, _CBS = block
            assert dist.k * dist.num_blocks == NB * BS, (
                f"BlockedIOSumLayer: consumer emission blocks ({dist.num_blocks} x "
                f"{dist.k}) do not tile this sum's {NB * BS} parents."
            )
            assert dist.k % BS == 0, (
                f"BlockedIOSumLayer: k_out={dist.k} must be a multiple of block_size={BS}."
            )

        self._blocked_outputs: List[Optional[BlockedNodeValues]] = [None] * len(self.nodes)
        self._blocked_flows: List[Optional[BlockedNodeValues]] = [None] * len(self.nodes)
        self._fwd_values_workspaces: List[Optional[torch.Tensor]] = [None] * len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"BlockedIOSumLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sum_ns={len(self.nodes)}, "
            f"out_vars={self._output_var_ids})"
        )

    def _out_workspace(self, blk_idx: int, batch_size: int, k: int,
                       device: torch.device) -> torch.Tensor:
        ws = self._fwd_values_workspaces[blk_idx]
        if ws is None or ws.device != device or ws.shape != (batch_size, k):
            ws = torch.empty(batch_size, k, dtype=torch.float32, device=device)
            self._fwd_values_workspaces[blk_idx] = ws
        return ws

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, force_use_bf16: bool = False,
                force_use_fp32: bool = False, propagation_alg: str = "LL",
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL", "BlockedIOSumLayer requires propagation_alg == 'LL'."
        assert params.dim() == 1

        for blk_idx, (block, (prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._blocked_input_refs,
        )):
            _nid_start, _cid_start, pid_start, _pfid_start, NB, NB_ch, BS, CBS = block
            bv_in = prod._blocked_outputs[ns_idx]
            assert bv_in is not None and bv_in.max_val is not None
            k_in = bv_in.k
            out_dist = self._output_dists[blk_idx]
            k_out = out_dist.k
            out_var = self._output_var_ids[blk_idx]

            values_out = self._out_workspace(blk_idx, batch_size, k_out, node_mars.device)
            bv_out = BlockedNodeValues(
                values=values_out, k=k_out, num_rows=NB * BS, data=bv_in.data,
                var_id=out_var, token_block=out_dist._token_block,
                batch_size=batch_size, max_val=None,
            )
            self._blocked_outputs[blk_idx] = bv_out

            TILE_M = _pow2_tile(k_out, _FWD_TILE_M)
            BLOCK_K = _pow2_tile(k_in, _FWD_BLOCK_K)
            grid = (triton.cdiv(k_out, TILE_M), batch_size)
            _blocked_io_sum_forward_kernel[grid](
                values_out_ptr=values_out,
                mparams_ptr=params,
                in_values_ptr=bv_in.values,
                max_val_ptr=bv_in.max_val,
                data_ptr=bv_in.data,
                in_token_block_ptr=bv_in.token_block,
                out_token_block_ptr=out_dist._token_block,
                in_var_id=bv_in.var_id,
                out_var_id=out_var,
                pid_start=pid_start,
                batch_size=batch_size,
                K_in=k_in, K_out=k_out,
                NB_ch=NB_ch, BS=BS, CBS=CBS,
                TILE_M=TILE_M, BLOCK_K=BLOCK_K,
            )
        return None

    # ------------------------------------------------------------------ #
    # Backward
    # ------------------------------------------------------------------ #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 allow_modify_flows: bool = False, propagation_alg: str = "LL",
                 logspace_flows: bool = False, negate_pflows: bool = False,
                 accumulate_ch_flows: bool = False, allow_neg_flows: bool = False,
                 force_use_fp32: bool = False, **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL" and not logspace_flows and not allow_neg_flows, (
            "BlockedIOSumLayer.backward requires propagation_alg='LL' + "
            "logspace_flows=False + allow_neg_flows=False."
        )
        # ``allow_modify_flows`` only concerns node_flows, which this layer never
        # reads: the parent flow arrives packed (raw) from the consumer.
        assert not accumulate_ch_flows, (
            "BlockedIOSumLayer.backward writes the upstream flow container "
            "directly; accumulate_ch_flows is not supported."
        )
        compute_pflows = param_flows is not None

        for blk_idx, (block, (prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._blocked_input_refs,
        )):
            _nid_start, _cid_start, pid_start, pfid_start, NB, NB_ch, BS, CBS = block
            bv_flow_out = self._blocked_flows[blk_idx]
            assert bv_flow_out is not None, (
                "BlockedIOSumLayer.backward expected the packed parent flow "
                "from the downstream CoBlockedProdLayer."
            )
            self._blocked_flows[blk_idx] = None

            bv_in = prod._blocked_outputs[ns_idx]
            bv_out = self._blocked_outputs[blk_idx]
            k_in, k_out = bv_in.k, bv_out.k

            flow_values = self._flow_workspace(blk_idx, batch_size, k_in, node_mars.device)
            bv_flow_in = bv_in.like_pattern(flow_values)
            prod._blocked_flows[ns_idx] = bv_flow_in

            TILE_J = _pow2_tile(k_in, _BWD_TILE_J)
            grid = (triton.cdiv(k_in, TILE_J), batch_size)
            _blocked_io_sum_backward_kernel[grid](
                flow_in_ptr=bv_flow_in.values,
                flow_out_ptr=bv_flow_out.values,
                out_values_ptr=bv_out.values,
                in_values_ptr=bv_in.values,
                mparams_ptr=params,
                pflows_ptr=(param_flows if compute_pflows else bv_flow_in.values),
                data_ptr=bv_in.data,
                in_token_block_ptr=bv_in.token_block,
                out_token_block_ptr=bv_out.token_block,
                in_var_id=bv_in.var_id,
                out_var_id=bv_out.var_id,
                pid_start=pid_start,
                pfid_start=pfid_start,
                batch_size=batch_size,
                K_in=k_in, K_out=k_out,
                NB_ch=NB_ch, BS=BS, CBS=CBS,
                TILE_J=TILE_J, BLOCK_P=_BWD_BLOCK_P,
                COMPUTE_PFLOWS=1 if compute_pflows else 0,
                NEGATE_PFLOWS=1 if negate_pflows else 0,
                ATOMIC_PFLOWS=1 if batch_size > 1 else 0,
            )
        return None


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit(
    do_not_specialize=["in_var_id", "out_var_id", "pid_start", "batch_size",
                       "K_in", "K_out"],
)
def _blocked_io_sum_forward_kernel(
    values_out_ptr, mparams_ptr,
    in_values_ptr, max_val_ptr,
    data_ptr, in_token_block_ptr, out_token_block_ptr,
    in_var_id, out_var_id, pid_start, batch_size, K_in, K_out,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Grid ``(cdiv(K_out, TILE_M), B)``: one program per (output tile, sample).
    ``out[b, m] = log(Σ_j W[m_out·K_out + m, m_in·K_in + j] · exp(in[b, j] − max_b)) + max_b``."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1)

    v_in = tl.load(data_ptr + in_var_id * batch_size + b)
    v_out = tl.load(data_ptr + out_var_id * batch_size + b)
    m_in = tl.load(in_token_block_ptr + v_in)
    m_out = tl.load(out_token_block_ptr + v_out)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)                  # [TILE_M]
    mask_m = offs_m < K_out
    h = m_out * K_out + offs_m
    pblock = h // BS
    within = h % BS

    max_val = tl.load(max_val_ptr + b)
    max_val = tl.maximum(max_val, -1e30)

    acc = tl.zeros([TILE_M], dtype=tl.float32)
    for k0 in tl.range(0, K_in, BLOCK_K):
        offs_j = k0 + tl.arange(0, BLOCK_K)
        mask_j = offs_j < K_in
        c = m_in * K_in + offs_j
        cblock = c // CBS
        cslot = c % CBS
        log_vals = tl.load(in_values_ptr + b * K_in + offs_j, mask=mask_j, other=-float("inf"))
        vk = tl.where(mask_j, tl.exp(log_vals - max_val), 0.0)
        W_off = (
            pid_start
            + (pblock[:, None] * NB_ch + cblock[None, :]) * CBS * BS
            + cslot[None, :] * BS
            + within[:, None]
        )                                                           # [TILE_M, BLOCK_K]
        W = tl.load(mparams_ptr + W_off, mask=mask_m[:, None] & mask_j[None, :],
                    other=0.0).to(tl.float32)
        acc += tl.sum(W * vk[None, :], axis=1)

    result = tl.log(acc + 1e-24) + max_val
    tl.store(values_out_ptr + b * K_out + offs_m, result, mask=mask_m)


@triton.jit(
    do_not_specialize=["in_var_id", "out_var_id", "pid_start", "pfid_start",
                       "batch_size", "K_in", "K_out"],
)
def _blocked_io_sum_backward_kernel(
    flow_in_ptr, flow_out_ptr, out_values_ptr, in_values_ptr,
    mparams_ptr, pflows_ptr,
    data_ptr, in_token_block_ptr, out_token_block_ptr,
    in_var_id, out_var_id, pid_start, pfid_start, batch_size, K_in, K_out,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_J: tl.constexpr, BLOCK_P: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr, NEGATE_PFLOWS: tl.constexpr,
    ATOMIC_PFLOWS: tl.constexpr,
):
    """Grid ``(cdiv(K_in, TILE_J), B)``: one program per (input tile, sample).

      chunk[m, j] = flow_out[b, m] · W[h_m, c_j] · exp(in[b, j] − out[b, m])
      flow_in[b, j] = Σ_m chunk[m, j];   pflows[addr(h_m, c_j)] += chunk[m, j]

    See :func:`_blocked_input_sum_backward_kernel` for the pflow atomicity
    argument (plain RMW at B=1, atomics at B>1)."""
    pid_j = tl.program_id(0)
    b = tl.program_id(1)

    v_in = tl.load(data_ptr + in_var_id * batch_size + b)
    v_out = tl.load(data_ptr + out_var_id * batch_size + b)
    m_in = tl.load(in_token_block_ptr + v_in)
    m_out = tl.load(out_token_block_ptr + v_out)

    offs_j = pid_j * TILE_J + tl.arange(0, TILE_J)
    mask_j = offs_j < K_in
    c = m_in * K_in + offs_j
    cblock = c // CBS
    cslot = c % CBS
    log_val = tl.load(in_values_ptr + b * K_in + offs_j, mask=mask_j, other=-float("inf"))

    acc = tl.zeros([TILE_J], dtype=tl.float32)
    for m0 in tl.range(0, K_out, BLOCK_P):
        offs_m = m0 + tl.arange(0, BLOCK_P)
        mask_m = offs_m < K_out
        h = m_out * K_out + offs_m
        pblock = h // BS
        within = h % BS
        nflow = tl.load(flow_out_ptr + b * K_out + offs_m, mask=mask_m, other=0.0)
        nmars = tl.load(out_values_ptr + b * K_out + offs_m, mask=mask_m, other=0.0)
        W_off = (
            pid_start
            + (pblock[:, None] * NB_ch + cblock[None, :]) * CBS * BS
            + cslot[None, :] * BS
            + within[:, None]
        )                                                           # [BLOCK_P, TILE_J]
        mask_mj = mask_m[:, None] & mask_j[None, :]
        W = tl.load(mparams_ptr + W_off, mask=mask_mj, other=0.0).to(tl.float32)
        chunk = nflow[:, None] * W * tl.exp(log_val[None, :] - nmars[:, None])
        chunk = tl.where(mask_mj, chunk, 0.0)
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
                tl.atomic_add(pflows_ptr + pf_off, pf_val, mask=mask_mj)
            else:
                old = tl.load(pflows_ptr + pf_off, mask=mask_mj, other=0.0)
                tl.store(pflows_ptr + pf_off, old + pf_val, mask=mask_mj)

    tl.store(flow_in_ptr + b * K_in + offs_j, acc, mask=mask_j)
