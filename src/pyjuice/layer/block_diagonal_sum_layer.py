from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import List, Optional, Sequence

if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math

from pyjuice.nodes import SumNodes
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.parameter_list import FastParamList
from .layer import Layer
from .sum_layer import SumLayer
from .dense_sum_layer import DenseSumLayer, _select_bk_tiles, _select_par_tiles


class BlockDiagonalSumLayer(SumLayer):
    """
    Inference + element-flow backward fast path for sum layers whose
    transition matrix is block-diagonal at the block level: parent block
    ``i`` is connected to exactly child block ``i`` (and only that one).

    This is the structure produced by a Monarch factorisation
    (``BD₁ → permutation → BD₂``) when each BD is built with
    ``edge_ids = arange(NB)[None, :].repeat(2, 1)``. The natural data-model
    shape is ``NB`` independent ``[bs, cbs]`` weight matrices concatenated;
    a single Triton launch handles all ``NB`` blocks in parallel via the
    grid's parent-block axis.

    Asymptotically the kernel is ``NB``-fold cheaper than the general
    sum kernel: only the diagonal ``(pblock, pblock)`` weight blocks
    exist, and the per-parent-block child marginals live in a disjoint
    contiguous slice of ``element_mars``, so each program handles its
    block independently with no cross-block reduction.

    Constraints (asserted at compile time):
      * Every ``SumNodes`` has a single child group.
      * ``NB == NB_ch`` and ``bs == cbs`` (square blocks — the natural
        Monarch case; relax later if a non-square use case appears).
      * ``edge_ids`` is exactly ``arange(NB)[None, :].repeat(2, 1)``.

    Param flows are supported: :meth:`backward` accumulates them via
    :meth:`_bk_bd_par_kernel` — each program owns one ``[TILE_M, TILE_N]``
    tile of a single diagonal block and loops over the batch, so writes
    are deterministic load-add-stores (no atomics), same contract as
    :class:`DenseSumLayer`.

    Tied ``SumNodes`` are supported: ``_param_range`` / ``_param_flow_range``
    alias the source's, same protocol as :class:`DenseSumLayer`.
    """

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int,
                 global_pid_start: int, global_pfid_start: int,
                 node2tiednodes: dict,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_block: int = 8,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False) -> None:

        # Bypass ``SumLayer.__init__`` — we replace its block-sparse
        # bookkeeping with the direct-layout metadata below (same pattern
        # as ``DenseSumLayer``).
        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."
        assert len(nodes) == len(set(nodes)), "Input node list contains duplicates."

        for ns in nodes:
            assert len(ns.chs) == 1, (
                "BlockDiagonalSumLayer requires num_chs == 1 per SumNodes."
            )
            assert ns.num_node_blocks == ns.num_ch_node_blocks, (
                "BlockDiagonalSumLayer requires NB == NB_ch (square block "
                f"diagonal); got NB={ns.num_node_blocks}, "
                f"NB_ch={ns.num_ch_node_blocks}."
            )
            assert ns.block_size == ns.ch_block_size, (
                "BlockDiagonalSumLayer requires bs == cbs (square per-block "
                f"weight matrix); got bs={ns.block_size}, "
                f"cbs={ns.ch_block_size}."
            )
            assert ns.edge_ids.size(1) == ns.num_node_blocks, (
                "BlockDiagonalSumLayer: edge_ids has wrong number of "
                f"columns ({ns.edge_ids.size(1)} != NB={ns.num_node_blocks})."
            )
            if ns.is_tied():
                source_ns = ns.get_source_ns()
                assert source_ns.provided("_param_range"), (
                    "BlockDiagonalSumLayer: tied sum node encountered "
                    "before its source was compiled."
                )

        layer_nid_start = global_nid_start
        layer_pid_start = global_pid_start
        layer_pfid_start = global_pfid_start

        layer_num_nodes = 0
        layer_num_edges = 0

        # Per-SumNodes metadata: ``(nid_start, cid_start, pid_start,
        # pfid_start, NB, bs, cbs)`` — one CUDA kernel launch per entry per
        # direction.
        blocks: List[tuple] = []

        curr_nid = layer_nid_start
        curr_pid = layer_pid_start
        curr_pfid = layer_pfid_start

        for ns in nodes:
            cs = ns.chs[0]
            assert cs.provided("_output_ind_range"), (
                "BlockDiagonalSumLayer: child has no _output_ind_range; "
                "make sure the product layer is compiled before the sum."
            )

            NB = ns.num_node_blocks
            bs = ns.block_size
            cbs = ns.ch_block_size

            ns._output_ind_range = (curr_nid, curr_nid + ns.num_nodes)

            # Param range — alias-on-tied (same protocol as
            # DenseSumLayer). Block-diagonal needs exactly ``NB * bs * cbs``
            # scalars per ns (NB_ch-fold smaller than the dense case).
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

            # Validate the block-diagonal pattern AFTER computing num_edges
            # so a stray non-BD ns doesn't silently consume the wrong slice
            # of the param buffer. We need edge_ids[0] == edge_ids[1] ==
            # arange(NB); ``_init_layers`` already gates this layer behind
            # ``_block_diagonal_eligible``, but the assert here keeps the
            # invariant local to the layer for anyone constructing it
            # directly (e.g. tests).
            edge_ids = ns.edge_ids
            expected = torch.arange(NB, dtype = edge_ids.dtype,
                                    device = edge_ids.device)
            assert torch.equal(edge_ids[0], expected) and \
                   torch.equal(edge_ids[1], expected), (
                "BlockDiagonalSumLayer: edge_ids is not the block-diagonal "
                "pattern (expected arange(NB)[None,:].repeat(2,1))."
            )

            # ``_params`` is shape ``[NB, bs, cbs]`` ordered by edge_ids
            # columns. The BD edge ordering is already the natural
            # ``arange(NB)``, so ``_inverse_param_ids`` is identity and
            # ``gather_parameters`` lays out the flat buffer as
            # ``[NB, cbs, bs]`` row-major. The kernel addresses an entry
            # ``(pblock, c, s)`` at
            # ``pid_start + pblock*bs*cbs + c*bs + s`` — see the docstrings
            # on ``_fw_bd_kernel`` / ``_bk_bd_kernel`` for the full layout.
            edge_lin_ids = torch.arange(NB, dtype = torch.long)
            ns._param_ids = block_pid_start + edge_lin_ids * bs * cbs
            ns._inverse_param_ids = torch.argsort(edge_lin_ids)
            # ``_param_flow_ids`` mirrors ``_param_ids`` in the BD layout —
            # the param-flow buffer reuses the params' ``[NB, bs, cbs]`` flat
            # stride exactly (one ``bs*cbs`` block per edge, in arange order).
            # Needed by :meth:`SumNodes.update_param_flows` and
            # :meth:`TensorCircuit.get_node_param_flows` to pull this ns's
            # accumulated pflows back into canonical ``[NB, bs, cbs]`` shape.
            ns._param_flow_ids = block_pfid_start + edge_lin_ids * bs * cbs

            blocks.append((
                curr_nid,                      # nid_start in node_mars
                cs._output_ind_range[0],       # cid_start in element_mars
                block_pid_start,               # pid_start in flat params
                block_pfid_start,              # pfid_start in flat param_flows
                NB, bs, cbs,
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

        self._bd_blocks = blocks

        # Stub out the block-sparse bookkeeping that some parent
        # introspection paths read — mirrors the same stubs in
        # ``DenseSumLayer``.
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

    def __repr__(self) -> str:
        return (
            f"BlockDiagonalSumLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_blocks={len(self._bd_blocks)})"
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, force_use_bf16: bool = False,
                force_use_fp32: bool = False, propagation_alg: str = "LL",
                **kwargs) -> None:

        assert params.dim() == 1, (
            "BlockDiagonalSumLayer only supports flat 1-D params; got "
            f"params.dim()={params.dim()}."
        )
        assert propagation_alg == "LL", (
            "BlockDiagonalSumLayer currently supports only LL propagation."
        )

        batch_size = node_mars.size(1)

        for block in self._bd_blocks:
            nid_start, cid_start, pid_start, _pfid_start, NB, bs, cbs = block

            CBS_PADDED = triton.next_power_of_2(cbs)
            BS_PADDED = triton.next_power_of_2(bs)
            BLOCK_B = min(triton.next_power_of_2(batch_size), 64)
            if BLOCK_B < 1:
                BLOCK_B = 1

            # Grid: one program per (parent_block, batch_tile). The
            # forward kernel handles all ``bs`` parent rows in the block
            # within a single program because per-block compute is small
            # (BS_PADDED * CBS_PADDED weights) and reusing the linearised
            # ``emars_lin`` across all parents in the block is the main
            # arithmetic win over the general path.
            grid = (NB, triton.cdiv(batch_size, BLOCK_B))

            self._fw_bd_kernel[grid](
                node_mars = node_mars,
                element_mars = element_mars,
                mparams = params,
                batch_size = batch_size,
                nid_start = nid_start,
                cid_start = cid_start,
                pid_start = pid_start,
                BS = bs,
                CBS = cbs,
                BS_PADDED = BS_PADDED,
                CBS_PADDED = CBS_PADDED,
                BLOCK_B = BLOCK_B,
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

        if param_flows is not None:
            assert param_flows.dim() == 1, (
                "BlockDiagonalSumLayer only supports flat 1-D param_flows; "
                f"got param_flows.dim()={param_flows.dim()}."
            )
        assert propagation_alg == "LL", (
            "BlockDiagonalSumLayer.backward currently supports only LL."
        )
        assert not allow_neg_flows, (
            "BlockDiagonalSumLayer.backward does not support allow_neg_flows."
        )
        assert not (allow_modify_flows and logspace_flows), (
            "`allow_modify_flows` must be False when `logspace_flows=True` "
            "(same contract as SumLayer)."
        )

        batch_size = node_mars.size(1)

        # In-place ``log(flow) - log_marg`` transform on this layer's own
        # parent rows — the pre-pass the ``ALLOW_MODIFY_FLOWS`` kernel
        # branch reads. Same kernel + tiling as ``DenseSumLayer.backward``.
        if allow_modify_flows:
            for block in self._bd_blocks:
                nid_start, _cid_start, _pid_start, _pfid_start, NB, bs, _cbs = block
                layer_n_nodes = NB * bs
                BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
                MF_BLOCK_B = min(2048, BATCH_SIZE_NP2)
                MF_BLOCK_M = min(max(2048 // MF_BLOCK_B, 1), bs)
                if MF_BLOCK_M < 1:
                    MF_BLOCK_M = 1
                grid = (triton.cdiv(batch_size, MF_BLOCK_B),
                        triton.cdiv(layer_n_nodes, MF_BLOCK_M))
                DenseSumLayer._bk_triton_dense_modify_flow_kernel[grid](
                    node_flows = node_flows,
                    node_mars = node_mars,
                    nid_start = nid_start,
                    batch_size = batch_size,
                    num_parents = layer_n_nodes,
                    BLOCK_B = MF_BLOCK_B,
                    BLOCK_M = MF_BLOCK_M,
                    propagation_alg_id = 0,
                    alpha = 0.0,
                )

        for block in self._bd_blocks:
            nid_start, cid_start, pid_start, pfid_start, NB, bs, cbs = block

            # Same tile policy as the dense backward: 2-D [TILE_N, BLOCK_B]
            # output tiles with a serial TILE_K loop over the parent rows.
            # Passing ``NB_ch = NB`` makes the grid-saturation heuristic see
            # the true total child-slot count (NB blocks × cbs slots each).
            TILE_N, TILE_K, BLOCK_B, use_tl_dot = _select_bk_tiles(
                cbs = cbs, bs = bs, batch_size = batch_size,
                NB_ch = NB, force_use_fp32 = force_use_fp32,
            )
            K_NUM_TILES = bs // TILE_K

            grid = (triton.cdiv(batch_size, BLOCK_B), NB * (cbs // TILE_N))

            self._bk_bd_kernel[grid](
                node_flows = node_flows,
                element_flows = element_flows,
                node_mars = node_mars,
                element_mars = element_mars,
                mparams = params,
                batch_size = batch_size,
                nid_start = nid_start,
                cid_start = cid_start,
                pid_start = pid_start,
                BS = bs,
                CBS = cbs,
                BLOCK_B = BLOCK_B,
                TILE_N = TILE_N,
                TILE_K = TILE_K,
                K_NUM_TILES = K_NUM_TILES,
                ALLOW_MODIFY_FLOWS = 1 if allow_modify_flows else 0,
                ACCUMULATE_CH_FLOWS = 1 if accumulate_ch_flows else 0,
                LOGSPACE_FLOWS = 1 if logspace_flows else 0,
                USE_TL_DOT = use_tl_dot,
            )

            if param_flows is not None:
                # ``NB_ch = 1``: only the diagonal ``(pblock, pblock)`` edge
                # blocks exist, so the true launch grid is
                # ``NB * (cbs/TILE_N) * (bs/TILE_M)`` — with the child-slot
                # stand-in reduced to a single block's ``cbs``, the
                # grid-saturation estimate inside ``_select_par_tiles``
                # matches it exactly.
                P_TILE_M, P_TILE_N, P_TILE_B, use_tl_dot_p = _select_par_tiles(
                    bs = bs, cbs = cbs, batch_size = batch_size,
                    NB = NB, NB_ch = 1,
                )
                B_NUM_TILES = triton.cdiv(batch_size, P_TILE_B)
                par_grid = (NB * (cbs // P_TILE_N), bs // P_TILE_M)

                self._bk_bd_par_kernel[par_grid](
                    node_flows = node_flows,
                    node_mars = node_mars,
                    element_mars = element_mars,
                    mparams = params,
                    param_flows = param_flows,
                    batch_size = batch_size,
                    nid_start = nid_start,
                    cid_start = cid_start,
                    pid_start = pid_start,
                    pfid_start = pfid_start,
                    BS = bs,
                    CBS = cbs,
                    TILE_M = P_TILE_M,
                    TILE_N = P_TILE_N,
                    TILE_B = P_TILE_B,
                    B_NUM_TILES = B_NUM_TILES,
                    ALLOW_MODIFY_FLOWS = 1 if allow_modify_flows else 0,
                    LOGSPACE_FLOWS = 1 if logspace_flows else 0,
                    NEGATE_PFLOWS = 1 if negate_pflows else 0,
                    USE_TL_DOT = use_tl_dot_p,
                )

        return None

    # ------------------------------------------------------------------ #
    # Triton kernels
    # ------------------------------------------------------------------ #

    @staticmethod
    @triton_jit
    def _fw_bd_kernel(node_mars, element_mars, mparams,
                      batch_size,
                      nid_start, cid_start, pid_start,
                      BS, CBS,
                      BS_PADDED: tl.constexpr,
                      CBS_PADDED: tl.constexpr,
                      BLOCK_B: tl.constexpr):
        """
        Forward for one BlockDiagonalSumLayer block, one program per
        ``(parent_block, batch_tile)``.

        Param layout (matches ``gather_parameters``'s flat layout):
          ``mparams[pid_start + pblock * BS * CBS + c * BS + s]``
          = weight from child_offs ``c`` to parent_offs ``s`` within block
          ``pblock``. Outer-block ordering is dense — block ``pblock``
          occupies ``[pid_start + pblock*BS*CBS, pid_start + (pblock+1)*BS*CBS)``,
          there are no gaps (BD has one block per parent block, no missing
          edges).

        Math (LL propagation):
          1. Load all ``CBS`` child marginals from ``element_mars`` at
             ``cid_start + pblock * CBS + c``.
          2. Stabilise: ``max_b = max_c emars[c, b]`` then ``emars_lin =
             exp(emars - max_b)`` (zero for ``c >= CBS`` padding slots).
          3. Load the ``[BS_PADDED, CBS_PADDED]`` weight tile.
          4. ``acc[s, b] = sum_c weight[s, c] * emars_lin[c, b]``.
          5. ``nmars[s, b] = log(acc + 1e-24) + max_b`` (with -inf
             passthrough for fully-masked batches).

        ``BS`` and ``CBS`` are runtime ints so a single binary serves every
        (tied or untied) block regardless of summate-specific shape;
        register pressure is bounded by ``BS_PADDED * CBS_PADDED *
        BLOCK_B``, both padded to powers of two.
        """

        pid_nb = tl.program_id(0)       # parent block id
        pid_b = tl.program_id(1)        # batch tile

        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        offs_c = tl.arange(0, CBS_PADDED)
        mask_c = offs_c < CBS

        offs_s = tl.arange(0, BS_PADDED)
        mask_s = offs_s < BS

        # Load all CBS child marginals for this parent block. Children for
        # block ``pid_nb`` live at ``cid_start + pid_nb*CBS + c`` — disjoint
        # from every other parent block (no cross-block read).
        emars_ptr = (
            (cid_start + pid_nb * CBS + offs_c)[:, None] * batch_size
            + offs_batch[None, :]
        )
        emars = tl.load(
            element_mars + emars_ptr,
            mask = mask_c[:, None] & mask_batch[None, :],
            other = -float("inf"),
        )                                                                       # [CBS_PADDED, BLOCK_B]

        # Stabilise + linearise per-batch.
        emars_max = tl.max(emars, axis = 0)                                     # [BLOCK_B]
        emars_max_safe = tl.where(emars_max == -float("inf"), 0.0, emars_max)
        emars_lin = tl.where(
            mask_c[:, None],
            tl.exp(emars - emars_max_safe[None, :]),
            0.0,
        )                                                                       # [CBS_PADDED, BLOCK_B]

        # Load weights for this block: row-major ``[BS, CBS]`` with strides
        # ``(1, BS)`` — i.e., ``mparams[pid_start + pblock*BS*CBS + c*BS + s]``.
        # Cast pid_nb to int64 so worst-case range analysis on
        # ``pblock * BS * CBS`` doesn't trip Triton's int32 overflow check
        # for large NB.
        block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
        weight_addr = (
            block_base
            + offs_c[None, :] * BS
            + offs_s[:, None]
        )                                                                       # [BS_PADDED, CBS_PADDED]
        weight = tl.load(
            mparams + weight_addr,
            mask = mask_s[:, None] & mask_c[None, :],
            other = 0.0,
        )
        weight = weight.to(tl.float32)

        # acc[s, b] = sum_c weight[s, c] * emars_lin[c, b]
        acc = tl.sum(weight[:, :, None] * emars_lin[None, :, :], axis = 1)      # [BS_PADDED, BLOCK_B]

        # Reassemble log-space output. ``emars_max == -inf`` => every child
        # is -inf => output stays -inf (acc would be 0 anyway, but the
        # explicit branch avoids ``log(1e-24) + (-inf)`` arithmetic on the
        # padded path).
        nmars = tl.where(
            emars_max[None, :] == -float("inf"),
            -float("inf"),
            tl.log(acc + 1e-24) + emars_max[None, :],
        )

        out_ptr = (
            (nid_start + pid_nb * BS + offs_s)[:, None] * batch_size
            + offs_batch[None, :]
        )
        tl.store(
            node_mars + out_ptr,
            nmars,
            mask = mask_s[:, None] & mask_batch[None, :],
        )

    @staticmethod
    @triton_jit
    def _bk_bd_kernel(node_flows, element_flows, node_mars, element_mars,
                      mparams,
                      batch_size,
                      nid_start, cid_start, pid_start,
                      BS, CBS,
                      BLOCK_B: tl.constexpr,
                      TILE_N: tl.constexpr,
                      TILE_K: tl.constexpr,
                      K_NUM_TILES: tl.constexpr,
                      ALLOW_MODIFY_FLOWS: tl.constexpr,
                      ACCUMULATE_CH_FLOWS: tl.constexpr,
                      LOGSPACE_FLOWS: tl.constexpr,
                      USE_TL_DOT: tl.constexpr):
        """
        Element-flow backward, one program per ``(batch_tile,
        child_tile)`` where ``pid_n`` decomposes into ``(parent_block,
        TILE_N-slot tile)``. Same 2-D-tile + serial-K structure as
        ``DenseSumLayer._bk_triton_dense_ele_kernel`` — the block-diagonal
        restriction just shrinks the K loop to THIS block's ``BS`` parent
        rows (``K_NUM_TILES = BS // TILE_K``).

        Math (LL, no param_flows), per K tile ``kt``:

          1. ``log_n_fdm[s, b] = log(parent_flow[s, b]) - log_marg[s, b]``
             (read directly if ``ALLOW_MODIFY_FLOWS=1`` — the pre-pass in
             :meth:`backward` already wrote it in place).
          2. Per-tile stabilisation: ``m_b = max_s log_n_fdm[s, b]``,
             ``n_fdm_sub = exp(log_n_fdm - m_b)`` (0 if m_b is -inf).
          3. ``partial[n, b] = Σ_s W[s, n] · n_fdm_sub[s, b]`` — ``tl.dot``
             when every tile dim is ≥ 16, broadcast-sum otherwise.
          4. Linear: ``acc += partial · exp(emars + m_b)``; logspace:
             online logsumexp merge of ``log(partial) + emars + m_b``.

        The former single-program-per-block version materialised the full
        ``[BS, CBS, BLOCK_B]`` broadcast product in registers — at
        ``bs = 128, BLOCK_B = 64`` that's 1M floats/program, which spilled
        to local memory and made backward scale ~linearly with B (and OOM
        under memory pressure). The tiled form keeps live registers at
        ``TILE_N × (TILE_K + BLOCK_B)``-scale and engages tensor cores.

        Each child slot belongs to exactly one parent block ⇒ plain
        stores, no atomics. ``ACCUMULATE_CH_FLOWS=1`` adds onto existing
        ``element_flows`` (logaddexp under ``LOGSPACE_FLOWS=1``).

        ``TILE_N`` divides ``CBS`` and ``TILE_K`` divides ``BS`` (both are
        powers of two from ``_select_bk_tiles``), so the child / parent
        dims need no masks — only the batch dim is masked.
        """

        pid_b = tl.program_id(0)        # batch tile
        pid_n = tl.program_id(1)        # (parent block, child tile)

        pblock_id = pid_n // (CBS // TILE_N)
        tile_id_n = pid_n % (CBS // TILE_N)

        offs_child = tl.arange(0, TILE_N) + tile_id_n * TILE_N        # [TILE_N]
        offs_child = tl.max_contiguous(tl.multiple_of(offs_child, TILE_N), TILE_N)
        off_cid = cid_start + pblock_id * CBS + offs_child

        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        # Child marginals for this tile (log space). Masked batch lanes are
        # never stored, so no ``other`` needed.
        emars = tl.load(
            element_mars + off_cid[:, None] * batch_size + offs_batch[None, :],
            mask = mask_batch[None, :],
        )                                                              # [TILE_N, BLOCK_B]

        if LOGSPACE_FLOWS == 1:
            acc = tl.zeros([TILE_N, BLOCK_B], dtype = tl.float32) - float("inf")
        else:
            acc = tl.zeros([TILE_N, BLOCK_B], dtype = tl.float32)

        # int64 cast on pblock_id: same int32 worst-case-range fix as the
        # forward kernel.
        block_base = pid_start + pblock_id.to(tl.int64) * BS * CBS

        for kt in range(0, K_NUM_TILES):
            off_pwithin = tl.arange(0, TILE_K) + kt * TILE_K           # [TILE_K]
            off_pwithin = tl.max_contiguous(
                tl.multiple_of(off_pwithin, TILE_K), TILE_K
            )
            off_mid = nid_start + pblock_id * BS + off_pwithin

            # Weights [TILE_N, TILE_K]: mparams[block_base + c*BS + s]
            # (child-major rows, parent stride 1 — coalesced inner dim).
            epars = tl.load(
                mparams + block_base
                + offs_child[:, None] * BS + off_pwithin[None, :]
            ).to(tl.float32)                                           # [TILE_N, TILE_K]

            off_mb = off_mid[:, None] * batch_size + offs_batch[None, :]
            nflows = tl.load(node_flows + off_mb, mask = mask_batch[None, :])

            if ALLOW_MODIFY_FLOWS == 1:
                # The pre-pass in ``backward`` wrote ``log(flow) - nmars``
                # in place (-inf where nmars is -inf).
                log_n_fdm = nflows
            else:
                nmars = tl.load(node_mars + off_mb, mask = mask_batch[None, :])
                if LOGSPACE_FLOWS == 1:
                    # ``node_flows`` is already log-space — subtract
                    # log-marg directly.
                    log_n_fdm = tl.where(
                        nmars == -float("inf"),
                        -float("inf"),
                        nflows - nmars,
                    )
                else:
                    log_n_fdm = tl.where(
                        nmars == -float("inf"),
                        -float("inf"),
                        tl.log(nflows + 1e-32) - nmars,
                    )                                                  # [TILE_K, BLOCK_B]

            # Per-tile stabilisation (exact rescaling — merged back via
            # ``exp(emars + m_b)`` / the logspace merge below).
            log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None, :]       # [1, BLOCK_B]
            n_fdm_sub = tl.where(
                log_n_fdm_max != -float("inf"),
                tl.exp(log_n_fdm - log_n_fdm_max),
                0.0,
            )                                                          # [TILE_K, BLOCK_B]

            if USE_TL_DOT == 1:
                partial = tl.dot(epars, n_fdm_sub)                     # [TILE_N, BLOCK_B]
            else:
                partial = tl.sum(
                    epars[:, :, None] * n_fdm_sub[None, :, :], axis = 1,
                )

            if LOGSPACE_FLOWS == 1:
                # Online logsumexp merge:
                # acc <- logaddexp(acc, log(partial) + emars + m_b).
                partial_max = emars + log_n_fdm_max                    # [TILE_N, BLOCK_B]
                acc = tl.where(
                    partial_max == -float("inf"),
                    acc,
                    tl.where(
                        partial_max > acc,
                        tl.log(partial + tl.exp(acc - partial_max) + 1e-32) + partial_max,
                        tl.log(tl.exp(partial_max - acc) * partial + 1.0) + acc,
                    ),
                )
            else:
                factor = tl.where(
                    log_n_fdm_max == -float("inf"),
                    0.0,
                    tl.exp(emars + log_n_fdm_max),
                )
                acc = acc + partial * factor

        out_ptr = off_cid[:, None] * batch_size + offs_batch[None, :]
        if ACCUMULATE_CH_FLOWS == 1:
            if LOGSPACE_FLOWS == 1:
                ori = tl.load(
                    element_flows + out_ptr,
                    mask = mask_batch[None, :],
                    other = -float("inf"),
                )
                # logaddexp(acc, ori) with -inf passthrough.
                hi = tl.maximum(acc, ori)
                lo = tl.minimum(acc, ori)
                acc = tl.where(
                    hi == -float("inf"),
                    -float("inf"),
                    hi + tlmath.log1p(tl.exp(lo - hi)),
                )
            else:
                ori = tl.load(
                    element_flows + out_ptr,
                    mask = mask_batch[None, :],
                    other = 0.0,
                )
                acc = acc + ori
        tl.store(element_flows + out_ptr, acc, mask = mask_batch[None, :])

    @staticmethod
    @triton_jit
    def _bk_bd_par_kernel(node_flows, node_mars, element_mars, mparams,
                          param_flows,
                          batch_size,
                          nid_start, cid_start, pid_start, pfid_start,
                          BS, CBS,
                          TILE_M: tl.constexpr,
                          TILE_N: tl.constexpr,
                          TILE_B: tl.constexpr,
                          B_NUM_TILES: tl.constexpr,
                          ALLOW_MODIFY_FLOWS: tl.constexpr,
                          LOGSPACE_FLOWS: tl.constexpr,
                          NEGATE_PFLOWS: tl.constexpr,
                          USE_TL_DOT: tl.constexpr):
        """
        Param-flow kernel for one BlockDiagonalSumLayer block — the BD
        specialisation of ``DenseSumLayer._bk_triton_dense_par_kernel``:
        only the diagonal ``(pblock, pblock)`` edge blocks exist, so
        ``pid_n`` decomposes into ``(parent_block, TILE_N child tile)`` and
        the parent/child rows both come from block ``pblock_id``.

        Each program owns one ``[TILE_M, TILE_N]`` (parent, child) tile of
        a single diagonal block and loops over the full batch, so every
        flat pflow offset is written by exactly one program per launch —
        the final accumulate is a plain load-add-store (deterministic, no
        atomics). Aliased pflow regions (tied duplicates, e.g. an HMM chain
        sharing one ``pfid_start``) are safe because their launches are
        ordered on the CUDA stream.

        Math per batch column (LL only — the layer asserts LL):
        ``pflow[s, c] += w[s, c] * sum_b exp(emars[c, b] + log_n_fdm[s, b])``
        with ``log_n_fdm = log(parent_flow) - log_marg`` and its per-column
        max factored out for stability.

        Grid: ``(NB * CBS // TILE_N, BS // TILE_M)``.

        ``TILE_M`` divides ``BS`` and ``TILE_N`` divides ``CBS`` (both
        powers of two from ``_select_par_tiles``), so only the batch dim is
        masked.
        """

        pid_n = tl.program_id(0)        # (parent block, child tile)
        pid_m = tl.program_id(1)        # parent tile within the block

        pblock_id = pid_n // (CBS // TILE_N)
        tile_id_n = pid_n % (CBS // TILE_N)

        offs_par = tl.arange(0, TILE_M) + pid_m * TILE_M           # [TILE_M]
        offs_par = tl.max_contiguous(tl.multiple_of(offs_par, TILE_M), TILE_M)
        offs_ch = tl.arange(0, TILE_N) + tile_id_n * TILE_N        # [TILE_N]
        offs_ch = tl.max_contiguous(tl.multiple_of(offs_ch, TILE_N), TILE_N)

        off_nid = nid_start + pblock_id * BS + offs_par            # [TILE_M]
        off_cid = cid_start + pblock_id * CBS + offs_ch            # [TILE_N]

        # Flat (pid, pfid) offsets share the BD layout: diagonal block
        # ``pblock`` at ``pblock * BS * CBS``, entry (c, s) at ``c*BS + s``
        # within it. int64 cast: same int32 worst-case-range fix as the
        # other BD kernels.
        block_off = pblock_id.to(tl.int64) * BS * CBS
        intra_block_offs = offs_ch[None, :] * BS + offs_par[:, None]  # [TILE_M, TILE_N]

        offs_batch = tl.arange(0, TILE_B)

        acc = tl.zeros([TILE_M, TILE_N], dtype = tl.float32)

        for _b in range(0, B_NUM_TILES):
            mask_batch = offs_batch < batch_size

            # [TILE_M, TILE_B] node-side offsets; the TILE_B inner dim is
            # stride 1 (coalesced).
            off_node_b = off_nid[:, None] * batch_size + offs_batch[None, :]

            # ``other=0.0`` on emars is safe: masked batch columns have
            # ``log_n_fdm_max == -inf`` so their ``scaled_emars`` column is
            # exp(-inf) = 0.
            emars = tl.load(
                element_mars + off_cid[None, :] * batch_size +
                offs_batch[:, None],
                mask = mask_batch[:, None], other = 0.0,
            )  # [TILE_B, TILE_N]

            if ALLOW_MODIFY_FLOWS == 1:
                # node_flows pre-transformed to log(flow) - nmars by the
                # modify pre-pass in :meth:`backward`.
                log_n_fdm = tl.load(
                    node_flows + off_node_b, mask = mask_batch[None, :],
                    other = -float("inf"),
                )  # [TILE_M, TILE_B]
            else:
                nmars = tl.load(
                    node_mars + off_node_b, mask = mask_batch[None, :],
                    other = 0.0,
                )  # [TILE_M, TILE_B]
                if LOGSPACE_FLOWS == 1:
                    nflows_log = tl.load(
                        node_flows + off_node_b,
                        mask = mask_batch[None, :],
                        other = -float("inf"),
                    )
                    log_n_fdm = tl.where(
                        nmars == -float("inf"), -float("inf"),
                        nflows_log - nmars,
                    )
                else:
                    nflows = tl.load(
                        node_flows + off_node_b,
                        mask = mask_batch[None, :], other = 0.0,
                    )
                    log_n_fdm = tl.where(
                        nmars == -float("inf"), -float("inf"),
                        tl.log(nflows) - nmars,
                    )

            # Factor out the per-batch-column max across the parent tile.
            log_n_fdm_max = tl.max(log_n_fdm, axis = 0)            # [TILE_B]
            n_fdm_sub = tl.where(
                log_n_fdm_max[None, :] != -float("inf"),
                tl.exp(log_n_fdm - log_n_fdm_max[None, :]), 0.0,
            )  # [TILE_M, TILE_B]
            scaled_emars = tl.exp(
                emars + log_n_fdm_max[:, None],
            )  # [TILE_B, TILE_N]

            if USE_TL_DOT == 1:
                partial_flows = tl.dot(n_fdm_sub, scaled_emars)
            else:
                partial_flows = tl.sum(
                    n_fdm_sub[:, :, None] * scaled_emars[None, :, :],
                    axis = 1,
                )

            acc += partial_flows

            offs_batch += TILE_B

        # Upcast covers bf16 params transparently; pflows always accumulate
        # in fp32.
        epars = tl.load(mparams + pid_start + block_off + intra_block_offs)
        pflows = acc * epars.to(tl.float32)

        pf_offs = pfid_start + block_off + intra_block_offs
        ori = tl.load(param_flows + pf_offs)
        if NEGATE_PFLOWS == 1:
            tl.store(param_flows + pf_offs, ori - pflows)
        else:
            tl.store(param_flows + pf_offs, ori + pflows)
