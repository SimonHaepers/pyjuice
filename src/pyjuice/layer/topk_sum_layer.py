from __future__ import annotations

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Sequence, List, Tuple, Optional

if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math

from pyjuice.nodes import SumNodes
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.parameter_list import FastParamList
from .layer import Layer
from .sum_layer import SumLayer


class TopKSumLayer(SumLayer):
    """
    Block-shared dynamic-top-K sum layer paired with :class:`TopKLayer`.

    Each annotated :class:`SumNodes` reads the K (index, log-value) pairs
    written by the upstream :class:`TopKLayer` into the circuit-level
    ``topk_indices`` / ``topk_values`` buffers (per-summate slot range
    stored on the SumNodes as ``_topk_slot_range``), gathers the
    corresponding K weight columns for every parent row in the block, and
    approximates the H-way logsumexp by a K-way one. K is fixed at compile
    time per summate, so the kernels carry no partition tables / variable-
    nnz book-keeping the general :class:`SumLayer` block-sparse path needs.

    Tied :class:`SumNodes` are supported and are critical for homogeneous
    HMM chains: the ``_param_range`` of a tied duplicate aliases the
    source's, so the flat params buffer holds one shared transition matrix
    instead of T-1 copies. Backward atomic-adds into ``param_flows`` go to
    the same slots from every timestep, accumulating EM stats correctly.

    Constraints (asserted at compile time):
      * Every ``SumNodes`` is block-dense and has a single child group.
      * ``logspace_flows=True`` is rejected at backward time (would need
        atomic-scatter into a ``-inf`` initialised buffer).
      * Param-flow accumulation only writes through atomic-add; the
        source's ``_param_flow_range`` is used (tied case) or a fresh
        range is allocated.
    """

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int,
                 global_pid_start: int, global_pfid_start: int,
                 node2tiednodes: dict,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_block: int = 8,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False) -> None:

        # Bypass ``SumLayer.__init__`` — we replace all of its block-sparse
        # compile-time bookkeeping with the direct-layout metadata below.
        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."
        assert len(nodes) == len(set(nodes)), "Input node list contains duplicates."

        for ns in nodes:
            assert getattr(ns, "_topk_k", None) is not None, (
                "TopKSumLayer requires every SumNodes to have ``_topk_k``."
            )
            # Dense-block layout is load-bearing: the forward kernel reads
            # ``params`` as a contiguous ``[NB, H, BS]`` row-major slice
            # (offset ``pid_start + n*H*BS + h*BS + r``). Without
            # block-dense, missing (parent_block, child_block) edges leave
            # gaps in that slice and the kernel silently reads garbage.
            assert ns.is_block_dense, (
                "TopKSumLayer requires every SumNodes to be block-dense."
            )
            assert len(ns.chs) == 1, (
                "TopKSumLayer requires a single child group per SumNodes."
            )
            assert ns.provided("_topk_slot_range"), (
                "TopKSumLayer: ``_topk_slot_range`` was not set on the "
                "SumNodes — make sure :class:`TopKLayer` is compiled "
                "first (TensorCircuit appends the TopK layer-group "
                "between prod and sum)."
            )
            if ns.is_tied():
                source_ns = ns.get_source_ns()
                assert source_ns.provided("_param_range"), (
                    "TopKSumLayer: tied sum node encountered before its "
                    "source was compiled. The source must appear in an "
                    "earlier depth — check the DAG topo order."
                )

        layer_nid_start = global_nid_start
        layer_pid_start = global_pid_start
        layer_pfid_start = global_pfid_start

        layer_num_nodes = 0
        layer_num_edges = 0

        # Per-SumNodes metadata: see ``_build_meta`` below for the field
        # ordering. One CUDA kernel launch per entry per direction.
        blocks: List[tuple] = []

        curr_nid = layer_nid_start
        curr_pid = layer_pid_start
        curr_pfid = layer_pfid_start

        for ns in nodes:
            cs = ns.chs[0]
            NB = ns.num_node_blocks
            NB_ch = ns.num_ch_node_blocks
            bs = ns.block_size
            cbs = ns.ch_block_size
            H = NB_ch * cbs        # total candidate children
            K = ns._topk_k
            slot_start, slot_end = ns._topk_slot_range
            assert slot_end - slot_start == K, (
                "TopKSumLayer: slot range width mismatch with ``_topk_k``."
            )

            ns._output_ind_range = (curr_nid, curr_nid + ns.num_nodes)

            # Param range — same alias-on-tied protocol as DenseSumLayer.
            if ns.is_tied():
                source_ns = ns.get_source_ns()
                block_pid_start, _ = source_ns._param_range
                block_pfid_start, _ = source_ns._param_flow_range
                ns._param_range = source_ns._param_range
                ns._param_flow_range = source_ns._param_flow_range
            else:
                pid_end = curr_pid + ns.num_edges
                pfid_end = curr_pfid + ns.num_edges
                ns._param_range = (curr_pid, pid_end)
                ns._param_flow_range = (curr_pfid, pfid_end)
                block_pid_start = curr_pid
                block_pfid_start = curr_pfid
                curr_pid = pid_end
                curr_pfid = pfid_end

            # Match DenseSumLayer's parameter ordering so
            # ``gather_parameters`` / ``_init_parameters`` lays out
            # ``_params[edge_block, bs, cbs]`` into the flat buffer in the
            # same row-major ``(parent_block, child_block, child_offs,
            # parent_offs)`` order our kernels assume.
            edge_ids = ns.edge_ids
            edge_lin_ids = edge_ids[0] * NB_ch + edge_ids[1]
            ns._param_ids = block_pid_start + edge_lin_ids * bs * cbs
            ns._inverse_param_ids = torch.argsort(edge_lin_ids)

            blocks.append((
                curr_nid,            # nid_start in node_mars
                block_pid_start,     # pid_start in flat params
                block_pfid_start,    # pfid_start in flat param_flows
                cs._output_ind_range[0],  # cid_start in element_mars
                NB, H, bs, cbs, K, slot_start,
            ))

            curr_nid += ns.num_nodes
            layer_num_nodes += ns.num_nodes
            if not ns.is_tied():
                layer_num_edges += ns.num_edges

        self.num_nodes = layer_num_nodes
        self.num_edges = layer_num_edges
        self._layer_nid_range = (layer_nid_start, layer_nid_start + layer_num_nodes)
        self._layer_pid_range = (layer_pid_start, curr_pid)
        self._layer_pfid_range = (layer_pfid_start, curr_pfid)

        self._topk_blocks = blocks

        # Bring along the block-sparse stubs the parent's accidental
        # fallbacks read so they no-op cleanly.
        self.num_fw_partitions = 0
        self.num_bk_partitions = 0
        self.partitioned_nids = FastParamList([])
        self.partitioned_cids = FastParamList([])
        self.partitioned_pids = FastParamList([])
        self.partitioned_pfids = FastParamList([])
        self.partitioned_chids = FastParamList([])
        self.partitioned_parids = FastParamList([])
        self.partitioned_parpids = FastParamList([])
        self.cs_block_sizes = []
        self._cached_fw_pcids = dict()
        self._cached_bk_parids = dict()

    def __repr__(self):
        return (
            f"TopKSumLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_blocks={len(self._topk_blocks)})"
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, *,
                topk_indices: Optional[torch.Tensor] = None,
                topk_values: Optional[torch.Tensor] = None,
                force_use_bf16: bool = False, force_use_fp32: bool = False,
                propagation_alg: str = "LL", **kwargs) -> None:

        assert topk_indices is not None and topk_values is not None, (
            "TopKSumLayer.forward requires `topk_indices` and `topk_values` "
            "buffers (allocated by TensorCircuit)."
        )
        assert params.dim() == 1, (
            "TopKSumLayer only supports flat 1-D params; got "
            f"params.dim()={params.dim()}."
        )
        assert propagation_alg == "LL", (
            "TopKSumLayer currently supports only LL propagation."
        )

        batch_size = node_mars.size(1)

        for block in self._topk_blocks:
            nid_start, pid_start, _pfid_start, _cid_start, NB, H, bs, cbs, K, slot_start = block

            # Output dim ``BS`` is tiled by ``BLOCK_S``: each program owns a
            # small ``[BLOCK_S, BLOCK_B]`` output tile and reduces over K.
            # This is the prune-pc ``_sparse_gms_kernel`` shape, ported.
            BLOCK_S = min(triton.next_power_of_2(bs), 64)
            BLOCK_B = min(triton.next_power_of_2(batch_size), 16)
            if BLOCK_B < 1:
                BLOCK_B = 1
            K_PADDED = triton.next_power_of_2(K)
            grid = (
                triton.cdiv(bs, BLOCK_S),
                NB,
                triton.cdiv(batch_size, BLOCK_B),
            )

            self._fw_topk_sum_kernel[grid](
                node_mars = node_mars,
                topk_indices = topk_indices,
                topk_values = topk_values,
                mparams = params,
                batch_size = batch_size,
                nid_start = nid_start,
                pid_start = pid_start,
                slot_start = slot_start,
                H = H,
                BS = bs,
                K = K,
                K_PADDED = K_PADDED,
                BLOCK_S = BLOCK_S,
                BLOCK_B = BLOCK_B,
            )

        return None

    # ------------------------------------------------------------------ #
    # Backward
    # ------------------------------------------------------------------ #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 *,
                 topk_indices: Optional[torch.Tensor] = None,
                 topk_values: Optional[torch.Tensor] = None,
                 allow_modify_flows: bool = False, propagation_alg: str = "LL",
                 logspace_flows: bool = False, negate_pflows: bool = False,
                 accumulate_ch_flows: bool = False, allow_neg_flows: bool = False,
                 force_use_fp32: bool = False, **kwargs) -> None:

        assert topk_indices is not None and topk_values is not None, (
            "TopKSumLayer.backward requires `topk_indices` and `topk_values`."
        )
        assert propagation_alg == "LL", (
            "TopKSumLayer backward currently supports only LL propagation."
        )
        assert not logspace_flows, (
            "TopKSumLayer backward does not support logspace_flows: the "
            "scatter into element_flows / param_flows uses atomic-add, "
            "which can't accumulate into a -inf-initialised buffer."
        )
        assert not allow_neg_flows, (
            "TopKSumLayer backward does not support allow_neg_flows."
        )

        batch_size = node_mars.size(1)

        for block in self._topk_blocks:
            nid_start, pid_start, pfid_start, _cid_start, NB, H, bs, cbs, K, slot_start = block

            # Output dim ``BS`` is tiled by ``BLOCK_S`` (mirrors forward).
            # element_flows atomic-adds across BLOCK_S tiles for the same
            # (parent_block, batch) accumulate the per-(k, b) child flow
            # correctly (atomic-add is associative). param_flows atomic-adds
            # across BLOCK_S tiles target disjoint ``offs_s`` ranges and
            # never collide.
            BLOCK_S = min(triton.next_power_of_2(bs), 64)
            BLOCK_B = min(triton.next_power_of_2(batch_size), 16)
            if BLOCK_B < 1:
                BLOCK_B = 1
            K_PADDED = triton.next_power_of_2(K)
            grid = (
                triton.cdiv(bs, BLOCK_S),
                NB,
                triton.cdiv(batch_size, BLOCK_B),
            )

            cid_start = _cid_start

            self._bk_topk_sum_kernel[grid](
                node_flows = node_flows,
                element_flows = element_flows,
                node_mars = node_mars,
                topk_indices = topk_indices,
                topk_values = topk_values,
                mparams = params,
                param_flows = (param_flows if param_flows is not None
                               else element_flows),  # dummy ptr; gated by COMPUTE_PFLOWS
                batch_size = batch_size,
                nid_start = nid_start,
                cid_start = cid_start,
                pid_start = pid_start,
                pfid_start = pfid_start,
                slot_start = slot_start,
                H = H,
                BS = bs,
                K = K,
                K_PADDED = K_PADDED,
                BLOCK_S = BLOCK_S,
                BLOCK_B = BLOCK_B,
                ALLOW_MODIFY_FLOWS = 1 if allow_modify_flows else 0,
                COMPUTE_PFLOWS = 1 if param_flows is not None else 0,
                NEGATE_PFLOWS = 1 if negate_pflows else 0,
            )

        return None

    # ------------------------------------------------------------------ #
    # Triton kernels
    # ------------------------------------------------------------------ #

    @staticmethod
    @triton_jit
    def _fw_topk_sum_kernel(node_mars, topk_indices, topk_values, mparams,
                            batch_size,
                            nid_start, pid_start, slot_start,
                            H, BS,
                            K: tl.constexpr,
                            K_PADDED: tl.constexpr,
                            BLOCK_S: tl.constexpr,
                            BLOCK_B: tl.constexpr):
        """
        Forward for one ``TopKSumLayer`` block, output-tiled.

        Per (parent_offs_tile, parent_block_id, batch_tile) program: load
        the K (index, value) pairs from the topk side buffers, gather a
        ``[BLOCK_S, K_PADDED, BLOCK_B]`` weight tile, reduce over K with
        ``log(sum exp(x - max)) + max``. The output ``BS`` dim is tiled by
        ``BLOCK_S`` so register pressure is bounded by ``BLOCK_S * K_PADDED
        * BLOCK_B`` regardless of how large ``BS`` is — earlier revisions
        had ``BS`` as ``tl.constexpr`` and ptxas blew up at ``BS=4096``.
        ``BS`` and ``H`` are runtime ints so a single binary serves every
        (tied or untied) timestep regardless of summate-specific shape.

        Padded K slots get neutral fill (-inf for values, 0 for indices,
        masked out of the reduction).
        """

        pid_s = tl.program_id(0)        # offs tile within a parent block
        pid_n = tl.program_id(1)        # parent_block_id
        pid_b = tl.program_id(2)        # batch tile

        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < BS

        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        offs_k = tl.arange(0, K_PADDED)
        mask_k = offs_k < K

        # Load topk indices/values for this batch tile.
        load_kb = mask_k[:, None] & mask_batch[None, :]
        idx_ptr = topk_indices + (slot_start + offs_k)[:, None] * batch_size + offs_batch[None, :]
        val_ptr = topk_values + (slot_start + offs_k)[:, None] * batch_size + offs_batch[None, :]
        topk_idx = tl.load(idx_ptr, mask = load_kb, other = 0)
        topk_val = tl.load(val_ptr, mask = load_kb, other = -float("inf"))

        # Per-(b) max over K for stabilisation; zero padded-k positions in
        # the linearised tile so they don't contribute.
        val_max = tl.max(topk_val, axis = 0)                                        # [BLOCK_B]
        val_max_safe = tl.where(val_max == -float("inf"), 0.0, val_max)
        val_lin = tl.exp(topk_val - val_max_safe[None, :])                          # [K_PADDED, BLOCK_B]
        val_lin = tl.where(mask_k[:, None], val_lin, 0.0)

        # Gather weights for the output tile: one address per
        # (parent_offs, k, batch). Dense ``[NB, H, BS]`` row-major layout —
        # block-dense is asserted at compile time.
        topk_idx64 = topk_idx.to(tl.int64)
        parent_block_base = pid_start + pid_n * H * BS
        weight_addr = (
            parent_block_base
            + topk_idx64[None, :, :] * BS
            + offs_s[:, None, None]
        )                                                                            # [BLOCK_S, K_PADDED, BLOCK_B]
        weight_mask = (
            mask_s[:, None, None]
            & mask_k[None, :, None]
            & mask_batch[None, None, :]
        )
        weight = tl.load(mparams + weight_addr, mask = weight_mask, other = 0.0)
        weight = weight.to(tl.float32)

        # Sum over K with linearised values.
        acc = tl.sum(weight * val_lin[None, :, :], axis = 1)                        # [BLOCK_S, BLOCK_B]

        nmars = tl.where(
            val_max[None, :] == -float("inf"),
            -float("inf"),
            tl.log(acc + 1e-32) + val_max[None, :],
        )

        out_ptr = (
            (nid_start + pid_n * BS + offs_s)[:, None] * batch_size
            + offs_batch[None, :]
        )
        tl.store(node_mars + out_ptr, nmars,
                 mask = mask_s[:, None] & mask_batch[None, :])

    @staticmethod
    @triton_jit
    def _bk_topk_sum_kernel(node_flows, element_flows, node_mars,
                            topk_indices, topk_values, mparams, param_flows,
                            batch_size,
                            nid_start, cid_start, pid_start, pfid_start,
                            slot_start,
                            H, BS,
                            K: tl.constexpr,
                            K_PADDED: tl.constexpr,
                            BLOCK_S: tl.constexpr,
                            BLOCK_B: tl.constexpr,
                            ALLOW_MODIFY_FLOWS: tl.constexpr,
                            COMPUTE_PFLOWS: tl.constexpr,
                            NEGATE_PFLOWS: tl.constexpr):
        """
        Combined backward for one ``TopKSumLayer`` block, output-tiled
        (mirrors the forward kernel's grid).

        Per (parent_offs_tile, parent_block_id, batch_tile) program:

          * computes ``log_n_fdm = log(parent_flow) - log_marg`` for this
            ``BLOCK_S`` tile of parent rows,
          * gathers a ``[BLOCK_S, K_PADDED, BLOCK_B]`` weight tile,
          * computes ``contrib[s, k, b] = weight[s, k, b]
            * exp(topk_val[k, b] + log_n_fdm[s, b])``,
          * atomic-adds the per-(k, b) partial sum (over this BLOCK_S
            range of parent rows) into
            ``element_flows[cid_start + h_kb, b]``,
          * optionally atomic-adds ``contrib`` into
            ``param_flows[pfid_start + parent_block * H * BS + h_kb * BS
            + offs_s]`` for this BLOCK_S range.

        Atomic-adds into element_flows from sibling BLOCK_S tiles
        accumulate the full sum-over-parents associatively. Atomic-adds
        into param_flows from sibling BLOCK_S tiles target disjoint
        ``offs_s`` ranges and never collide.

        ``BS`` and ``H`` are runtime ints; the constexpr cache key is just
        ``(K, K_PADDED, BLOCK_S, BLOCK_B, ALLOW_MODIFY_FLOWS, COMPUTE_PFLOWS,
        NEGATE_PFLOWS)`` so a single binary serves every (tied or untied)
        timestep.
        """

        pid_s = tl.program_id(0)        # offs tile within a parent block
        pid_n = tl.program_id(1)        # parent_block_id
        pid_b = tl.program_id(2)        # batch tile

        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < BS

        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        offs_k = tl.arange(0, K_PADDED)
        mask_k = offs_k < K

        # Load topk indices/values for this batch tile.
        load_kb = mask_k[:, None] & mask_batch[None, :]
        idx_ptr = topk_indices + (slot_start + offs_k)[:, None] * batch_size + offs_batch[None, :]
        val_ptr = topk_values + (slot_start + offs_k)[:, None] * batch_size + offs_batch[None, :]
        topk_idx = tl.load(idx_ptr, mask = load_kb, other = 0)
        topk_val = tl.load(val_ptr, mask = load_kb, other = -float("inf"))

        # Parent flows / log-marginals for this BLOCK_S tile of rows.
        nid_off = (nid_start + pid_n * BS + offs_s)[:, None] * batch_size + offs_batch[None, :]
        nf_mask = mask_s[:, None] & mask_batch[None, :]
        nflows = tl.load(node_flows + nid_off, mask = nf_mask, other = 0.0)
        nmars  = tl.load(node_mars  + nid_off, mask = nf_mask, other = -float("inf"))

        # ``log_n_fdm = log(parent_flow) - log_marg`` — same canonical form as
        # SumLayer's backward. ``ALLOW_MODIFY_FLOWS=1`` says nflows already
        # holds this value (TensorCircuit pre-pass converted it).
        if ALLOW_MODIFY_FLOWS == 1:
            log_n_fdm = nflows
        else:
            log_n_fdm = tl.where(
                nmars == -float("inf"), -float("inf"),
                tl.log(nflows + 1e-32) - nmars,
            )                                                                       # [BLOCK_S, BLOCK_B]

        # Gather weights for this output tile: [BLOCK_S, K_PADDED, BLOCK_B].
        topk_idx64 = topk_idx.to(tl.int64)
        parent_block_base = pid_start + pid_n * H * BS
        weight_addr = (
            parent_block_base
            + topk_idx64[None, :, :] * BS
            + offs_s[:, None, None]
        )
        weight_mask = (
            mask_s[:, None, None]
            & mask_k[None, :, None]
            & mask_batch[None, None, :]
        )
        weight = tl.load(mparams + weight_addr, mask = weight_mask, other = 0.0)
        weight = weight.to(tl.float32)

        # contrib[s, k, b] = weight[s, k, b] * exp(topk_val[k, b] + log_n_fdm[s, b])
        log_arg = topk_val[None, :, :] + log_n_fdm[:, None, :]                       # [BLOCK_S, K_PADDED, BLOCK_B]
        contrib = tl.where(
            log_arg == -float("inf"),
            0.0,
            weight * tl.exp(log_arg),
        )
        # Zero out padded slots so they don't reach the atomic-adds.
        contrib = tl.where(mask_s[:, None, None], contrib, 0.0)
        contrib = tl.where(mask_k[None, :, None], contrib, 0.0)

        # Per-(k, b) partial child-flow contribution = sum over THIS BLOCK_S
        # tile of parents. Sibling tiles atomic-add their partials into the
        # same element_flows slot.
        flow_kb = tl.sum(contrib, axis = 0)                                          # [K_PADDED, BLOCK_B]

        # Atomic-add into element_flows[cid_start + topk_idx[k, b], b].
        ef_ptr = (
            element_flows
            + (cid_start + topk_idx64) * batch_size
            + offs_batch[None, :]
        )                                                                            # [K_PADDED, BLOCK_B]
        tl.atomic_add(ef_ptr, flow_kb, mask = load_kb)

        if COMPUTE_PFLOWS == 1:
            # Per-(s, k, batch) param-flow into
            # param_flows[pfid_start + parent_block * H * BS + h_kb * BS + s].
            pf_addr = (
                pfid_start
                + pid_n * H * BS
                + topk_idx64[None, :, :] * BS
                + offs_s[:, None, None]
            )                                                                        # [BLOCK_S, K_PADDED, BLOCK_B]
            pf_val = -contrib if NEGATE_PFLOWS == 1 else contrib
            pf_mask = (
                mask_s[:, None, None]
                & mask_k[None, :, None]
                & mask_batch[None, None, :]
            )
            tl.atomic_add(
                param_flows + pf_addr,
                pf_val,
                mask = pf_mask,
            )
