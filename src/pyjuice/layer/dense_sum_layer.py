from __future__ import annotations

import functools
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Sequence, Optional, List

if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math

from pyjuice.nodes import SumNodes
from pyjuice.utils.kernel_launcher import triton_jit
from .layer import Layer
from .sum_layer import SumLayer


class DenseSumLayer(SumLayer):
    """
    Fast path for sum layers with fully-connected (dense) block topology.
    Skips the block-sparse partitioning in ``SumLayer.__init__`` and uses
    Triton kernels that address parameters by direct pointer arithmetic.

    Only supports:
      * ``SumNodes`` with ``is_block_dense == True``
      * ``num_chs == 1`` per ``SumNodes`` (children contiguous in element_mars)

    Tied ``SumNodes`` are fine — the layer reuses the source's
    ``_param_range`` so the flat params buffer stays at the source's size
    (critical for homogeneous HMMs: one shared H×H transition instead of
    ``T-1`` copies). The tied source must appear in an earlier depth in the
    same circuit so its ``_param_range`` is already set by the time the
    tied duplicates compile.

    Parameter flows (EM learning) are supported: :meth:`backward` with a
    ``param_flows`` buffer accumulates per-edge flows at the same flat
    offsets as the params (``_param_flow_ids`` mirrors ``_param_ids``).
    Tied duplicates alias the source's pflow region, so their flows
    accumulate directly into one shared block — no ``compute_cum_par_flows``
    fusing needed. The pflow kernel loops over the batch inside each
    program, so every flat offset is owned by exactly one program per
    launch and accumulation is a deterministic read-modify-write (kernel
    launches for aliased tied blocks are ordered by the CUDA stream).
    """

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int,
                 global_pid_start: int, global_pfid_start: int, node2tiednodes: dict,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_block: int = 8,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False) -> None:

        # Deliberately do NOT call ``SumLayer.__init__`` — we replace all of
        # its compile-time bookkeeping with direct-layout metadata below.
        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."
        assert len(nodes) == len(set(nodes)), "Input node list contains duplicates."

        for ns in nodes:
            assert ns.is_block_dense, (
                "DenseSumLayer requires every node to have fully-connected "
                "block topology; got a SumNodes with sparse edge_ids."
            )
            assert len(ns.chs) == 1, (
                "DenseSumLayer currently requires num_chs == 1 per SumNodes "
                "(so children are contiguous in element_mars)."
            )
            if ns.is_tied():
                source_ns = ns.get_source_ns()
                assert source_ns.provided("_param_range"), (
                    "DenseSumLayer: tied sum node encountered before its "
                    "source was compiled. The source must appear in an "
                    "earlier depth — check the DAG topo order."
                )

        layer_nid_start = global_nid_start
        layer_pid_start = global_pid_start
        layer_pfid_start = global_pfid_start

        layer_num_nodes = 0
        layer_num_edges = 0

        # Per-SumNode metadata: each tuple (nid_start, cid_start, pid_start,
        # pfid_start, NB, NB_ch, bs, cbs). One CUDA kernel launch per entry.
        blocks: List[tuple] = []

        curr_nid = layer_nid_start
        curr_pid = layer_pid_start
        curr_pfid = layer_pfid_start

        for ns in nodes:
            cs = ns.chs[0]
            assert cs.provided("_output_ind_range"), (
                "Child of DenseSumLayer has no _output_ind_range; make sure "
                "the product layer is compiled before the sum layer."
            )

            NB = ns.num_node_blocks
            NB_ch = ns.num_ch_node_blocks
            bs = ns.block_size
            cbs = ns.ch_block_size

            ns._output_ind_range = (curr_nid, curr_nid + ns.num_nodes)

            # Param range (linear-domain params, NB * NB_ch * cbs * bs scalars).
            # Tied nodes alias the source's range — do not advance curr_pid /
            # curr_pfid. The flat params tensor stays at the source's size.
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

            # Establish the inverse-permutation used by ``gather_parameters``
            # so ``TensorCircuit._init_parameters`` can copy user-provided
            # params into the flat tensor in the same order we read them back.
            #
            # SumNodes stores ``self._params`` of shape
            # ``[num_edge_blocks, bs, cbs]`` ordered by ``edge_ids`` columns.
            # ``gather_parameters`` writes
            #   params[psid:peid] = self._params[self._inverse_param_ids]
            #                           .permute(0, 2, 1).reshape(-1)
            # so the flat layout becomes ``[num_edge_blocks, cbs, bs]``
            # indexed by ``_inverse_param_ids``. We want that layout to match
            # the contiguous ``(pblock, cblock)`` row-major order the dense
            # kernel assumes (edge index = pblock * NB_ch + cblock).
            #
            # ``edge_ids`` for a dense node (default ``_construct_edges``) is
            # already in that order, so ``_inverse_param_ids == arange``.
            edge_ids = ns.edge_ids
            edge_lin_ids = edge_ids[0] * NB_ch + edge_ids[1]
            # Rebuild ``_param_ids`` / ``_inverse_param_ids`` the same way the
            # block-sparse path does during compilation.
            ns._param_ids = block_pid_start + edge_lin_ids * bs * cbs
            ns._inverse_param_ids = torch.argsort(edge_lin_ids)
            # ``_param_flow_ids`` mirrors ``_param_ids`` — the param-flow
            # buffer shares the same per-block ``bs * cbs`` stride and intra-
            # block ``(c, s)`` order as the params buffer. Needed by
            # :meth:`SumNodes.update_param_flows` for canonical extraction.
            ns._param_flow_ids = block_pfid_start + edge_lin_ids * bs * cbs

            blocks.append((
                curr_nid,                      # nid_start in node_mars
                cs._output_ind_range[0],       # cid_start in element_mars
                block_pid_start,               # pid_start in flat params
                block_pfid_start,              # pfid_start in flat param_flows
                NB, NB_ch, bs, cbs,
            ))

            curr_nid += ns.num_nodes
            layer_num_nodes += ns.num_nodes
            # Tied nodes contribute no new edges to the flat params — only the
            # source ns's num_edges counts (and that was counted by the layer
            # group that owns the source).
            if not ns.is_tied():
                layer_num_edges += ns.num_edges

        self.num_nodes = layer_num_nodes
        self.num_edges = layer_num_edges
        self._layer_nid_range = (layer_nid_start, layer_nid_start + layer_num_nodes)
        self._layer_pid_range = (layer_pid_start, curr_pid)
        self._layer_pfid_range = (layer_pfid_start, curr_pfid)

        # Host-side metadata driving kernel launches — never read on GPU,
        # so keep it as a plain Python list to avoid a D2H copy + implicit
        # stream sync on every forward/backward.
        self._dense_blocks = blocks

        # The forward path overwrites ``element_mars`` in place per block
        # (log -> exp(log_emars - per_tile_max)); overlapping child ranges
        # would re-exp already-linearised values. Reject at compile time so
        # the runtime can stay simple.
        child_ranges = sorted(
            (b[1], b[1] + b[5] * b[7]) for b in blocks
        )
        for (_, prev_end), (next_start, _) in zip(child_ranges, child_ranges[1:]):
            assert prev_end <= next_start, (
                "DenseSumLayer blocks have overlapping child ranges; the "
                "forward path requires disjoint cid ranges per block."
            )

        # Scratch buffer holding the per-k-tile max recorded by the forward
        # precompute kernel. Allocated lazily on the first forward call.
        self._element_mars_max = None

        # Scratch buffer holding the bf16 linear-domain emars written by the
        # precompute kernel on the LL/bf16 fast path (read back by the main
        # forward kernel without any dtype cast). Allocated lazily, reused
        # across dense blocks — each block's NB_ch*CBS rows fit within the
        # single buffer by using the max over all blocks as the leading dim.
        self._emars_linear_bf16 = None

        # Stubs that the parent's forward/backward loops look at; keep them
        # empty so any accidental fallback into the sparse path does nothing.
        self.num_fw_partitions = 0
        self.num_bk_partitions = 0
        self.partitioned_nids = nn.ParameterList()
        self.partitioned_cids = nn.ParameterList()
        self.partitioned_pids = nn.ParameterList()
        self.partitioned_pfids = nn.ParameterList()
        self.partitioned_chids = nn.ParameterList()
        self.partitioned_parids = nn.ParameterList()
        self.partitioned_parpids = nn.ParameterList()
        self.cs_block_sizes = []
        self._cached_fw_pcids = dict()
        self._cached_bk_parids = dict()

    def __repr__(self):
        return (
            f"DenseSumLayer(nid_range=({self._layer_nid_range[0]},"
            f" {self._layer_nid_range[1]}), num_nodes={self.num_nodes},"
            f" num_edges={self.num_edges})"
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, force_use_bf16: bool = False,
                force_use_fp32: bool = False, propagation_alg: str = "LL",
                **kwargs) -> None:

        assert params.dim() == 1, (
            "DenseSumLayer only supports flat 1-D params; got "
            f"params.dim()={params.dim()}."
        )

        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(
            propagation_alg, **kwargs
        )
        alpha = float(propagation_alg_kwargs.get("alpha", 0.0))

        batch_size = node_mars.size(1)

        # LL fast path: params already bf16 + plain LL + tl.dot-friendly tiles.
        # Pre-linearises emars into a dedicated bf16 scratch so the main kernel
        # loads both ``epars`` and ``emars`` as bf16 with zero in-kernel casts.
        bf16_fast = (
            propagation_alg_id == 0
            and not force_use_fp32
            and params.dtype == torch.bfloat16
        )
        if bf16_fast:
            max_k_rows = max(b[5] * b[7] for b in self._dense_blocks)  # NB_ch * cbs
            emars_linear_bf16 = self._get_or_alloc_emars_linear_bf16(
                max_k_rows, batch_size, element_mars.device,
            )
        else:
            emars_linear_bf16 = element_mars  # unused on non-bf16 path; must be a valid pointer

        for block in self._dense_blocks:
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, bs, cbs = block

            total_edges = NB_ch * cbs

            TILE_M, TILE_K, BLOCK_B, use_bf16, use_tl_dot = _select_fw_tiles(
                bs = bs, total_edges = total_edges, batch_size = batch_size,
                NB = NB,
                force_use_bf16 = force_use_bf16, force_use_fp32 = force_use_fp32,
            )

            # Enable the bf16 fast path only when tiles are tl.dot-eligible.
            bf16_path = 1 if (bf16_fast and use_bf16 == 1 and use_tl_dot == 1) else 0

            K_NUM_TILES = total_edges // TILE_K
            grid = (triton.cdiv(batch_size, BLOCK_B), NB * (bs // TILE_M))

            element_mars_max = self._get_or_alloc_emars_max(
                K_NUM_TILES, batch_size, element_mars.device,
            )

            # LL / GeneralLL precompute once per block: exp(log_emars -
            # per_tile_max) + the per-tile max. MPE needs raw log-space
            # emars, so skip precompute there and let the main kernel do
            # the max-based update in log space.
            if propagation_alg_id != 1:
                precompute_grid = (triton.cdiv(batch_size, BLOCK_B), K_NUM_TILES)
                self._fw_triton_dense_precompute_kernel[precompute_grid](
                    element_mars = element_mars,
                    element_mars_max = element_mars_max,
                    emars_bf16_out = emars_linear_bf16,
                    batch_size = batch_size,
                    cid_start = cid_start,
                    TILE_K = TILE_K,
                    K_NUM_TILES = K_NUM_TILES,
                    BLOCK_B = BLOCK_B,
                    propagation_alg_id = propagation_alg_id,
                    BF16_PATH = bf16_path,
                    alpha = alpha,
                )

            self._fw_triton_dense_kernel[grid](
                node_mars = node_mars,
                element_mars = element_mars,
                element_mars_max = element_mars_max,
                mparams = params,
                emars_bf16_in = emars_linear_bf16,
                batch_size = batch_size,
                nid_start = nid_start,
                cid_start = cid_start,
                pid_start = pid_start,
                NB_ch = NB_ch,
                BS = bs,
                CBS = cbs,
                BLOCK_B = BLOCK_B,
                TILE_M = TILE_M,
                TILE_K = TILE_K,
                K_NUM_TILES = K_NUM_TILES,
                propagation_alg_id = propagation_alg_id,
                use_bf16 = use_bf16,
                use_tl_dot = use_tl_dot,
                BF16_PATH = bf16_path,
                alpha = alpha,
            )

        return None

    def _get_or_alloc_emars_max(self, K_NUM_TILES: int, batch_size: int,
                                 device: torch.device) -> torch.Tensor:
        """
        Lazily allocate (or grow) the per-k-tile max scratch buffer. One
        buffer per DenseSumLayer, reused across dense blocks within a single
        forward pass.
        """
        buf = self._element_mars_max
        if (
            buf is None
            or buf.size(0) < K_NUM_TILES
            or buf.size(1) < batch_size
            or buf.device != device
        ):
            buf = torch.empty(
                K_NUM_TILES, batch_size,
                device = device, dtype = torch.float32,
            )
            self._element_mars_max = buf
        return buf

    def _get_or_alloc_emars_linear_bf16(self, total_rows: int, batch_size: int,
                                          device: torch.device) -> torch.Tensor:
        """
        Lazily allocate (or grow) the bf16 linear-domain emars scratch used by
        the LL fast path. Sized to the max ``NB_ch * cbs`` across this layer's
        dense blocks, so all blocks in one forward share it (sequentially —
        each block overwrites the leading ``NB_ch * cbs`` rows).
        """
        buf = self._emars_linear_bf16
        if (
            buf is None
            or buf.size(0) < total_rows
            or buf.size(1) < batch_size
            or buf.device != device
        ):
            buf = torch.empty(
                total_rows, batch_size,
                device = device, dtype = torch.bfloat16,
            )
            self._emars_linear_bf16 = buf
        return buf

    # ------------------------------------------------------------------ #
    # Backward (element flows + optional param flows)
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
                "DenseSumLayer only supports flat 1-D param_flows; got "
                f"param_flows.dim()={param_flows.dim()}."
            )

        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(
            propagation_alg, **kwargs
        )
        alpha = float(propagation_alg_kwargs.get("alpha", 0.0))

        batch_size = node_mars.size(1)

        # Optional preprocessing: replace ``node_flows`` with
        # ``log(node_flows) - node_mars`` on this layer's parent nodes.
        if allow_modify_flows:
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
                    node_flows = node_flows,
                    node_mars = node_mars,
                    nid_start = nid_start,
                    batch_size = batch_size,
                    num_parents = layer_n_nodes,
                    BLOCK_B = BLOCK_B,
                    BLOCK_M = BLOCK_M,
                    propagation_alg_id = propagation_alg_id,
                    alpha = alpha,
                )

        for block in self._dense_blocks:
            nid_start, cid_start, pid_start, pfid_start, NB, NB_ch, bs, cbs = block

            TILE_N, TILE_K, BLOCK_B, use_tl_dot = _select_bk_tiles(
                cbs = cbs, bs = bs, batch_size = batch_size,
                NB_ch = NB_ch,
                force_use_fp32 = force_use_fp32,
            )

            K_NUM_TILES = NB * (bs // TILE_K)
            grid = (triton.cdiv(batch_size, BLOCK_B), NB_ch * (cbs // TILE_N))

            self._bk_triton_dense_ele_kernel[grid](
                node_flows = node_flows,
                element_flows = element_flows,
                node_mars = node_mars,
                element_mars = element_mars,
                mparams = params,
                batch_size = batch_size,
                nid_start = nid_start,
                cid_start = cid_start,
                pid_start = pid_start,
                NB_ch = NB_ch,
                BS = bs,
                CBS = cbs,
                allow_modify_flows = 1 if allow_modify_flows else 0,
                BLOCK_B = BLOCK_B,
                TILE_N = TILE_N,
                TILE_K = TILE_K,
                K_NUM_TILES = K_NUM_TILES,
                propagation_alg_id = propagation_alg_id,
                accumulate_ch_flows = 1 if accumulate_ch_flows else 0,
                logspace_flows = 1 if logspace_flows else 0,
                allow_neg_flows = 1 if allow_neg_flows else 0,
                use_tl_dot = use_tl_dot,
                alpha = alpha,
            )

            if param_flows is not None:
                P_TILE_M, P_TILE_N, P_TILE_B, use_tl_dot_p = _select_par_tiles(
                    bs = bs, cbs = cbs, batch_size = batch_size,
                    NB = NB, NB_ch = NB_ch,
                    mpe = (propagation_alg_id == 1),
                )
                B_NUM_TILES = triton.cdiv(batch_size, P_TILE_B)
                par_grid = (NB_ch * (cbs // P_TILE_N), NB * (bs // P_TILE_M))

                self._bk_triton_dense_par_kernel[par_grid](
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
                    NB_ch = NB_ch,
                    BS = bs,
                    CBS = cbs,
                    allow_modify_flows = 1 if allow_modify_flows else 0,
                    logspace_flows = 1 if logspace_flows else 0,
                    allow_neg_flows = 1 if allow_neg_flows else 0,
                    negate_pflows = 1 if negate_pflows else 0,
                    TILE_M = P_TILE_M,
                    TILE_N = P_TILE_N,
                    TILE_B = P_TILE_B,
                    B_NUM_TILES = B_NUM_TILES,
                    propagation_alg_id = propagation_alg_id,
                    use_tl_dot = use_tl_dot_p,
                    alpha = alpha,
                )

        return None

    # ------------------------------------------------------------------ #
    # Triton kernels
    # ------------------------------------------------------------------ #

    @staticmethod
    @triton_jit
    def _fw_triton_dense_kernel(node_mars, element_mars, element_mars_max,
                                mparams, emars_bf16_in,
                                batch_size: tl.constexpr,
                                nid_start: tl.constexpr, cid_start: tl.constexpr,
                                pid_start: tl.constexpr, NB_ch: tl.constexpr,
                                BS: tl.constexpr, CBS: tl.constexpr,
                                BLOCK_B: tl.constexpr, TILE_M: tl.constexpr,
                                TILE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                propagation_alg_id: tl.constexpr,
                                use_bf16: tl.constexpr, use_tl_dot: tl.constexpr,
                                BF16_PATH: tl.constexpr,
                                alpha = 0.0):
        """
        Forward kernel for one DenseSumLayer block. Caller contract:

          * LL (0) / GeneralLL (2): the pre-computed linear values
            ``exp(log_emars - per_tile_max)`` live either in ``element_mars``
            at ``[cid_start, cid_start + NB_ch*CBS)`` (``BF16_PATH == 0``,
            fp32) or in ``emars_bf16_in`` at zero-based rows
            ``[0, NB_ch*CBS)`` (``BF16_PATH == 1``, bf16). Per-k-tile maxima
            live in ``element_mars_max[K_NUM_TILES, batch_size]``.
          * MPE (1): ``element_mars`` holds raw log-space values;
            ``element_mars_max`` is not read and ``BF16_PATH`` must be 0.

        On ``BF16_PATH == 1`` the K-loop loads ``epars`` (from bf16 ``mparams``)
        and ``emars`` (from the bf16 scratch) as bf16 directly — no in-kernel
        dtype casts before the ``tl.dot``. Running ``{linear sum, log max}``
        accumulator matches the legacy path.
        """

        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)

        nblock_id = pid_m // (BS // TILE_M)
        tile_id_m = pid_m % (BS // TILE_M)

        # Parent node offsets within block (BS dim) and global node ids.
        # tile_id_m comes from pid_m % (BS//TILE_M) so Triton can't infer
        # divisibility by TILE_M on its own — hint it explicitly.
        offs_node = tl.arange(0, TILE_M) + tile_id_m * TILE_M      # [TILE_M]
        offs_node = tl.max_contiguous(tl.multiple_of(offs_node, TILE_M), TILE_M)
        off_nid = nid_start + nblock_id * BS + offs_node           # [TILE_M]

        # Batch offsets: chunk of BLOCK_B starting at pid_b*BLOCK_B, so values
        # are multiple_of(BLOCK_B) and max_contiguous(BLOCK_B).
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        # Edge offsets; the k-th edge within a parent block corresponds to
        # global child id ``cid_start + k`` and edge-block base pid
        # ``pid_start + (nblock_id*NB_ch*CBS + k) * BS``.
        # offs_edge = arange(0,TILE_K) is already contig(TILE_K)/mult(TILE_K).
        # Cast ``nblock_id`` to int64 so the address arithmetic doesn't
        # tickle Triton's int32 bounds check: the constexpr ``NB_ch*CBS*BS``
        # can be ~2^24, and Triton's worst-case range analysis on the
        # int32 ``nblock_id`` then folds to ~2^31 even when the runtime
        # range is bounded by NB.
        par_block_base = pid_start + nblock_id.to(tl.int64) * NB_ch * CBS * BS
        offs_edge = tl.arange(0, TILE_K)

        # epars tile: [TILE_M, TILE_K]
        #   outer dim (TILE_M, rows): stride 1   — contiguous in memory
        #   inner dim (TILE_K, cols): stride BS  — strided (not fast axis)
        # Base aligned to TILE_M (par_block_base mult of BS, BS mult of TILE_M).
        epars_ptr = mparams + par_block_base + \
            offs_edge[None, :] * BS + offs_node[:, None]   # [TILE_M, TILE_K]
        # emars tile: [TILE_K, BLOCK_B]
        #   inner dim (BLOCK_B): stride 1         — fully coalesced
        #   outer dim (TILE_K):  stride batch_size — strided
        if BF16_PATH == 1:
            # bf16 scratch is zero-based per block: row = k_tile*TILE_K + offs_edge.
            emars_ptr = emars_bf16_in + \
                offs_edge[:, None] * batch_size + offs_batch[None, :]
        else:
            emars_ptr = element_mars + \
                (cid_start + offs_edge)[:, None] * batch_size + \
                offs_batch[None, :]                         # [TILE_K, BLOCK_B]
        # Pointer into the per-tile max: element_mars_max[k_tile, batch_tile].
        emax_ptr = element_mars_max + pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

        # Running {linear sum, log max}. Invariant for LL/GeneralLL:
        # true log-space result = log(acc_sum) + acc_max. For MPE only
        # acc_max is used (as a plain running max).
        acc_sum = tl.zeros([TILE_M, BLOCK_B], dtype = tl.float32)
        acc_max = tl.zeros([TILE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for _k in range(0, K_NUM_TILES):
            # epars: TILE_M rows each contiguous in memory (stride 1 along M).
            # emars: BLOCK_B inner dim contiguous (stride 1 along batch).
            # On BF16_PATH both loads yield bf16 tiles directly.
            epars = tl.load(epars_ptr)
            emars = tl.load(emars_ptr, mask = mask_batch[None, :])

            if propagation_alg_id == 1:
                # MPE: log-space emars, update running max.
                lpars = tl.log(epars)
                nmars = tl.max(lpars[:, :, None] + emars[None, :, :], axis = 1)
                acc_max = tl.maximum(acc_max, nmars)
            else:
                # LL / GeneralLL: emars is already linearised and the per-tile
                # max was recorded by the precompute kernel.
                if propagation_alg_id == 2:
                    epars = tl.exp(tl.log(epars) * alpha)

                emars_max = tl.load(emax_ptr, mask = mask_batch)[None, :]
                emax_ptr += batch_size

                if use_tl_dot == 1:
                    if BF16_PATH == 1:
                        # Already bf16 — go straight into tl.dot.
                        nmars = tl.dot(epars, emars).to(tl.float32)
                    elif use_bf16 == 1:
                        epars_b = epars.to(tl.bfloat16)
                        emars_b = emars.to(tl.bfloat16)
                        nmars = tl.dot(epars_b, emars_b).to(tl.float32)
                    else:
                        nmars = tl.dot(epars, emars)
                else:
                    nmars = tl.sum(
                        epars[:, :, None] * emars[None, :, :], axis = 1
                    )

                # 1 exp/tile, 0 log/tile.
                mask = emars_max > acc_max
                diff = tl.where(mask, acc_max - emars_max, emars_max - acc_max)
                # NaN guard for -inf - (-inf).
                diff = tl.where(diff != diff, 0.0, diff)
                scale = tl.exp(diff)
                acc_sum = tl.where(
                    mask, acc_sum * scale + nmars, acc_sum + nmars * scale,
                )
                acc_max = tl.maximum(acc_max, emars_max)

            # Advance along K: next TILE_K edges = TILE_K*BS scalars (params)
            # and TILE_K*batch_size scalars (emars rows). Both are multiples
            # of TILE_M / BLOCK_B respectively, so alignment is preserved.
            epars_ptr += TILE_K * BS
            emars_ptr += TILE_K * batch_size

        if propagation_alg_id == 1:
            acc = acc_max
        else:
            acc = tl.log(acc_sum + 1e-24) + acc_max

        if propagation_alg_id == 2:
            # Rescale back: node_mars = (log sum w^alpha * p^alpha) / alpha
            acc = acc * (1.0 / alpha)

        # Store [TILE_M, BLOCK_B]: inner dim BLOCK_B is stride 1 (contiguous,
        # coalesced), outer dim TILE_M is stride batch_size.
        off_out = off_nid[:, None] * batch_size + offs_batch[None, :]
        tl.store(node_mars + off_out, acc, mask = mask_batch[None, :])

    @staticmethod
    @triton_jit
    def _fw_triton_dense_precompute_kernel(element_mars, element_mars_max,
                                           emars_bf16_out,
                                           batch_size: tl.constexpr,
                                           cid_start: tl.constexpr,
                                           TILE_K: tl.constexpr,
                                           K_NUM_TILES: tl.constexpr,
                                           BLOCK_B: tl.constexpr,
                                           propagation_alg_id: tl.constexpr,
                                           BF16_PATH: tl.constexpr,
                                           alpha = 0.0):
        """
        Pre-compute per-k-tile max and ``exp(emars - tile_max)`` for a single
        DenseSumLayer block. Maxima land in
        ``element_mars_max[k_tile_id, batch_tile]``. Only called for LL /
        GeneralLL — MPE needs the raw log-space emars.

        ``BF16_PATH`` routes the linearised values:
          * 0 — overwrite ``element_mars`` in place (fp32), matching the
            historical behaviour.
          * 1 — write bf16 into ``emars_bf16_out`` at zero-based row
            ``k_tile_id*TILE_K + offs_edge``; leave ``element_mars``
            untouched so the log-space buffer stays valid for downstream
            consumers.

        Grid: ``(cdiv(batch_size, BLOCK_B), K_NUM_TILES)``.
        """

        pid_b = tl.program_id(0)
        k_tile_id = tl.program_id(1)

        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        offs_edge = tl.arange(0, TILE_K)
        offs_child = cid_start + k_tile_id * TILE_K + offs_edge

        # emars tile: [TILE_K, BLOCK_B], BLOCK_B inner dim stride 1.
        emars_ptr = element_mars + offs_child[:, None] * batch_size + \
            offs_batch[None, :]
        emars = tl.load(
            emars_ptr, mask = mask_batch[None, :], other = -float("inf"),
        )

        # Per-tile max across TILE_K edges.
        emars_max = tl.max(emars, axis = 0)   # [BLOCK_B]

        if propagation_alg_id == 0:
            emars_linear = tl.where(
                emars_max[None, :] != -float("inf"),
                tl.exp(emars - emars_max[None, :]), 0.0,
            )
        if propagation_alg_id == 2:
            emars_linear = tl.where(
                emars_max[None, :] != -float("inf"),
                tl.exp((emars - emars_max[None, :]) * alpha), 0.0,
            )
            emars_max = emars_max * alpha

        if BF16_PATH == 1:
            # Zero-based row within the dedicated bf16 scratch — independent
            # of ``cid_start`` so blocks can share one scratch buffer.
            offs_row_out = k_tile_id * TILE_K + offs_edge
            out_ptr = emars_bf16_out + offs_row_out[:, None] * batch_size + \
                offs_batch[None, :]
            tl.store(
                out_ptr, emars_linear.to(tl.bfloat16),
                mask = mask_batch[None, :],
            )
        else:
            # Legacy path: overwrite log-space emars with linear values.
            tl.store(emars_ptr, emars_linear, mask = mask_batch[None, :])

        # Store per-tile max: element_mars_max[k_tile_id, batch_tile].
        max_ptr = element_mars_max + k_tile_id * batch_size + offs_batch
        tl.store(max_ptr, emars_max, mask = mask_batch)

    @staticmethod
    @triton_jit
    def _bk_triton_dense_modify_flow_kernel(node_flows, node_mars,
                                            nid_start: tl.constexpr,
                                            batch_size: tl.constexpr,
                                            num_parents: tl.constexpr,
                                            BLOCK_B: tl.constexpr,
                                            BLOCK_M: tl.constexpr,
                                            propagation_alg_id: tl.constexpr,
                                            alpha = 0.0):

        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)

        # Parent-node offsets: chunk of BLOCK_M starting at a multiple of BLOCK_M.
        offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        mask_m = offs_m < num_parents

        # Batch offsets: chunk of BLOCK_B starting at a multiple of BLOCK_B.
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        # [BLOCK_M, BLOCK_B] tile:
        #   inner dim (BLOCK_B): stride 1         — contiguous / coalesced
        #   outer dim (BLOCK_M): stride batch_size
        off_nmfs = (nid_start + offs_m)[:, None] * batch_size + offs_batch[None, :]
        mask = mask_m[:, None] & mask_batch[None, :]

        # Both loads: BLOCK_B inner dim is stride 1 (contiguous).
        nmars = tl.load(node_mars + off_nmfs, mask = mask)
        nflows = tl.load(node_flows + off_nmfs, mask = mask)

        if propagation_alg_id == 0:
            uflows = tl.where(
                nmars != -float("inf"), tl.log(nflows) - nmars, -float("inf")
            )
        if propagation_alg_id == 1:
            uflows = nflows
        if propagation_alg_id == 2:
            uflows = tl.where(
                nmars != -float("inf"),
                tl.log(nflows) - nmars * alpha,
                -float("inf"),
            )

        # Store [BLOCK_M, BLOCK_B]: BLOCK_B inner dim is stride 1 (contiguous).
        tl.store(node_flows + off_nmfs, uflows, mask = mask)

    @staticmethod
    @triton_jit
    def _bk_triton_dense_ele_kernel(node_flows, element_flows, node_mars,
                                    element_mars, mparams,
                                    batch_size: tl.constexpr,
                                    nid_start: tl.constexpr,
                                    cid_start: tl.constexpr,
                                    pid_start: tl.constexpr,
                                    NB_ch: tl.constexpr,
                                    BS: tl.constexpr, CBS: tl.constexpr,
                                    allow_modify_flows: tl.constexpr,
                                    BLOCK_B: tl.constexpr, TILE_N: tl.constexpr,
                                    TILE_K: tl.constexpr,
                                    K_NUM_TILES: tl.constexpr,
                                    propagation_alg_id: tl.constexpr,
                                    accumulate_ch_flows: tl.constexpr,
                                    logspace_flows: tl.constexpr,
                                    allow_neg_flows: tl.constexpr,
                                    use_tl_dot: tl.constexpr, alpha = 0.0):

        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)

        cblock_id = pid_n // (CBS // TILE_N)
        tile_id_n = pid_n % (CBS // TILE_N)

        # Child node offsets. tile_id_n comes from a pid modulo so divisibility
        # by TILE_N is not inferred automatically — hint it.
        offs_child = tl.arange(0, TILE_N) + tile_id_n * TILE_N    # [TILE_N]
        offs_child = tl.max_contiguous(tl.multiple_of(offs_child, TILE_N), TILE_N)
        off_cid = cid_start + cblock_id * CBS + offs_child

        # Batch offsets: mult/contig of BLOCK_B.
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        # emars [TILE_N, BLOCK_B]:
        #   inner dim (BLOCK_B): stride 1         — contiguous / coalesced
        #   outer dim (TILE_N):  stride batch_size
        emars = tl.load(
            element_mars + off_cid[:, None] * batch_size + offs_batch[None, :],
            mask = mask_batch[None, :],
        )  # [TILE_N, BLOCK_B]

        if propagation_alg_id == 2:
            emars = emars * alpha

        if logspace_flows == 1:
            acc = tl.zeros([TILE_N, BLOCK_B], dtype = tl.float32) - float("inf")
        else:
            acc = tl.zeros([TILE_N, BLOCK_B], dtype = tl.float32)

        # Loop over parent tiles: kt -> (pblock_id, within_tile_id)
        # across all NB parent blocks, (BS // TILE_K) tiles each.
        for kt in range(0, K_NUM_TILES):
            pblock_id = kt // (BS // TILE_K)
            within_tile_id = kt % (BS // TILE_K)

            # Parent-within-block offsets: arange(TILE_K) + mult_of(TILE_K).
            off_pwithin = tl.arange(0, TILE_K) + within_tile_id * TILE_K   # [TILE_K]
            off_pwithin = tl.max_contiguous(
                tl.multiple_of(off_pwithin, TILE_K), TILE_K
            )
            off_mid = nid_start + pblock_id * BS + off_pwithin

            # Edge block base for (pblock_id, cblock_id). Scalar multiple of BS.
            # int64 cast: same int32-range fix as the forward kernel — the
            # constexpr ``CBS*BS`` can be ~2^24 and Triton's worst-case
            # range analysis on int32 ``pblock_id``/``cblock_id`` overflows.
            edge_block_base = pid_start + \
                (pblock_id.to(tl.int64) * NB_ch + cblock_id) * CBS * BS
            # epars [TILE_N, TILE_K] — note: the inner/outer layout is the
            # TRANSPOSE of the forward kernel's epars load.
            #   inner dim (TILE_K, cols): stride 1  — contiguous / coalesced
            #   outer dim (TILE_N, rows): stride BS — strided
            epars = tl.load(
                mparams + edge_block_base +
                offs_child[:, None] * BS + off_pwithin[None, :]
            )  # [TILE_N, TILE_K]
            # Upcast to fp32: downstream ops (tl.dot against fp32 flows, log,
            # elementwise multiplies with fp32 accumulators) assume fp32.
            # Works transparently whether ``mparams`` is fp32 or bf16.
            epars = epars.to(tl.float32)

            # [TILE_K, BLOCK_B] offset — BLOCK_B inner dim is stride 1.
            off_mb = off_mid[:, None] * batch_size + offs_batch[None, :]
            # nflows: contiguous along BLOCK_B inner dim.
            nflows = tl.load(node_flows + off_mb, mask = mask_batch[None, :])

            if propagation_alg_id == 1:
                # nmars: contiguous along BLOCK_B inner dim.
                nmars = tl.load(node_mars + off_mb, mask = mask_batch[None, :])
                elpars = tl.log(tl.trans(epars))                           # [TILE_K, TILE_N]
                eflows = tl.sum(
                    tl.where(
                        tl.abs(
                            elpars[:, :, None] + emars[None, :, :]
                            - nmars[:, None, :]
                        ) < 1e-6,
                        nflows[:, None, :], 0.0,
                    ),
                    axis = 0,
                )

                if logspace_flows == 1:
                    diff = acc - eflows
                    acc = tl.where(
                        diff == 0,
                        acc + 0.69314718055994530942,  # log(2)
                        tl.where(
                            diff > 0,
                            acc + tlmath.log1p(tl.exp(-diff)),
                            eflows + tlmath.log1p(tl.exp(diff)),
                        ),
                    )
                else:
                    acc = acc + eflows
            else:
                if propagation_alg_id == 2:
                    epars = tl.exp(tl.log(epars) * alpha)

                if allow_modify_flows == 1:
                    log_n_fdm = nflows
                else:
                    # nmars: contiguous along BLOCK_B inner dim.
                    nmars = tl.load(node_mars + off_mb, mask = mask_batch[None, :])
                    if logspace_flows == 1:
                        if propagation_alg_id == 0:
                            log_n_fdm = tl.where(
                                nmars == -float("inf"), -float("inf"),
                                nflows - nmars,
                            )
                        if propagation_alg_id == 2:
                            log_n_fdm = tl.where(
                                nmars == -float("inf"), -float("inf"),
                                nflows - nmars * alpha,
                            )
                    elif allow_neg_flows == 1:
                        if propagation_alg_id == 0:
                            log_n_fdm = tl.where(
                                nmars == -float("inf"), -float("inf"), -nmars,
                            )
                        if propagation_alg_id == 2:
                            log_n_fdm = tl.where(
                                nmars == -float("inf"), -float("inf"),
                                -nmars * alpha,
                            )
                    else:
                        if propagation_alg_id == 0:
                            log_n_fdm = tl.where(
                                nmars == -float("inf"), -float("inf"),
                                tl.log(nflows) - nmars,
                            )
                        if propagation_alg_id == 2:
                            log_n_fdm = tl.where(
                                nmars == -float("inf"), -float("inf"),
                                tl.log(nflows) - nmars * alpha,
                            )

                log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None, :]
                n_fdm_sub = tl.where(
                    log_n_fdm_max != -float("inf"),
                    tl.exp(log_n_fdm - log_n_fdm_max), 0.0,
                )

                if allow_neg_flows == 1:
                    if use_tl_dot == 1:
                        partial_flows = tl.dot(epars, n_fdm_sub * nflows)
                    else:
                        partial_flows = tl.sum(
                            epars[:, :, None] * n_fdm_sub[None, :, :]
                            * nflows[None, :, :],
                            axis = 1,
                        )
                else:
                    if use_tl_dot == 1:
                        partial_flows = tl.dot(epars, n_fdm_sub)
                    else:
                        partial_flows = tl.sum(
                            epars[:, :, None] * n_fdm_sub[None, :, :], axis = 1,
                        )

                if logspace_flows == 1:
                    partial_flows_max = emars + log_n_fdm_max
                    acc = tl.where(
                        partial_flows_max == -float("inf"), acc,
                        tl.where(
                            partial_flows_max > acc,
                            tl.log(partial_flows + tl.exp(acc - partial_flows_max) + 1e-32) + partial_flows_max,
                            tl.log(tl.exp(partial_flows_max - acc) * partial_flows + 1.0) + acc,
                        ),
                    )
                else:
                    acc = acc + partial_flows * tl.exp(emars + log_n_fdm_max)

        # [TILE_N, BLOCK_B] offset — BLOCK_B inner dim is stride 1 (contiguous).
        off_emfs = off_cid[:, None] * batch_size + offs_batch[None, :]
        if accumulate_ch_flows == 1:
            # Load existing element_flows: contiguous along BLOCK_B inner dim.
            ori = tl.load(
                element_flows + off_emfs, mask = mask_batch[None, :],
                other = 0.0,
            )
            acc = acc + ori
        # Store [TILE_N, BLOCK_B]: contiguous / coalesced along BLOCK_B.
        tl.store(element_flows + off_emfs, acc, mask = mask_batch[None, :])

    @staticmethod
    @triton_jit
    def _bk_triton_dense_par_kernel(node_flows, node_mars, element_mars,
                                    mparams, param_flows,
                                    batch_size: tl.constexpr,
                                    nid_start: tl.constexpr,
                                    cid_start: tl.constexpr,
                                    pid_start: tl.constexpr,
                                    pfid_start: tl.constexpr,
                                    NB_ch: tl.constexpr,
                                    BS: tl.constexpr, CBS: tl.constexpr,
                                    allow_modify_flows: tl.constexpr,
                                    logspace_flows: tl.constexpr,
                                    allow_neg_flows: tl.constexpr,
                                    negate_pflows: tl.constexpr,
                                    TILE_M: tl.constexpr, TILE_N: tl.constexpr,
                                    TILE_B: tl.constexpr,
                                    B_NUM_TILES: tl.constexpr,
                                    propagation_alg_id: tl.constexpr,
                                    use_tl_dot: tl.constexpr, alpha = 0.0):
        """
        Param-flow kernel for one DenseSumLayer block. Each program owns one
        ``[TILE_M, TILE_N]`` (parent, child) tile of a single ``(pblock,
        cblock)`` edge block and loops over the full batch, so every flat
        pflow offset is written by exactly one program per launch — the final
        accumulate is a plain load-add-store (deterministic, no atomics).
        Aliased pflow regions (tied duplicates) are safe because their
        launches are ordered on the CUDA stream.

        Math per batch column (LL, mirrors the plain block-sparse par
        kernel): ``pflow[m, n] += w[m, n] * sum_b exp(emars[n, b] +
        log_n_fdm[m, b])`` with the per-column max of ``log_n_fdm`` factored
        out for stability.

        Grid: ``(NB_ch * CBS // TILE_N, NB * BS // TILE_M)``.
        """

        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        cblock_id = pid_n // (CBS // TILE_N)
        tile_id_n = pid_n % (CBS // TILE_N)
        pblock_id = pid_m // (BS // TILE_M)
        tile_id_m = pid_m % (BS // TILE_M)

        # Parent (s) / child (c) offsets within their blocks. Both come from
        # pid modulos, so hint divisibility explicitly.
        offs_par = tl.arange(0, TILE_M) + tile_id_m * TILE_M       # [TILE_M]
        offs_par = tl.max_contiguous(tl.multiple_of(offs_par, TILE_M), TILE_M)
        offs_ch = tl.arange(0, TILE_N) + tile_id_n * TILE_N        # [TILE_N]
        offs_ch = tl.max_contiguous(tl.multiple_of(offs_ch, TILE_N), TILE_N)

        off_nid = nid_start + pblock_id * BS + offs_par            # [TILE_M]
        off_cid = cid_start + cblock_id * CBS + offs_ch            # [TILE_N]

        # Flat (pid, pfid) offsets share the dense layout: edge block
        # (pblock, cblock) at ``(pblock*NB_ch + cblock) * CBS * BS``, entry
        # (c, s) at ``c*BS + s`` within it. int64 cast: same int32-range fix
        # as the other dense kernels.
        edge_block_off = (pblock_id.to(tl.int64) * NB_ch + cblock_id) * CBS * BS
        intra_block_offs = offs_ch[None, :] * BS + offs_par[:, None]  # [TILE_M, TILE_N]
        epars_offs = pid_start + edge_block_off + intra_block_offs

        # MPE compares against log-params inside the batch loop.
        if propagation_alg_id == 1:
            epars = tl.load(mparams + epars_offs).to(tl.float32)
            elpars = tl.log(epars)

        offs_batch = tl.arange(0, TILE_B)

        acc = tl.zeros([TILE_M, TILE_N], dtype = tl.float32)

        for _b in range(0, B_NUM_TILES):
            mask_batch = offs_batch < batch_size

            # [TILE_M, TILE_B] node-side offsets; BLOCK/TILE_B inner dim is
            # stride 1 (coalesced).
            off_node_b = off_nid[:, None] * batch_size + offs_batch[None, :]

            if propagation_alg_id == 1:
                # MPE: count argmax matches. ``other=inf`` on emars keeps
                # masked batch columns from ever satisfying the match
                # predicate.
                emars = tl.load(
                    element_mars + off_cid[None, :] * batch_size +
                    offs_batch[:, None],
                    mask = mask_batch[:, None], other = float("inf"),
                )  # [TILE_B, TILE_N]
                nmars = tl.load(
                    node_mars + off_node_b, mask = mask_batch[None, :],
                    other = 0.0,
                )  # [TILE_M, TILE_B]
                if logspace_flows == 1:
                    nflows_log = tl.load(
                        node_flows + off_node_b, mask = mask_batch[None, :],
                        other = -float("inf"),
                    )
                    nflows = tl.exp(nflows_log)
                else:
                    nflows = tl.load(
                        node_flows + off_node_b, mask = mask_batch[None, :],
                        other = 0.0,
                    )

                cond = tl.abs(
                    elpars[:, None, :] + emars[None, :, :]
                    - nmars[:, :, None]
                ) < 1e-6
                acc += tl.sum(
                    tl.where(cond, nflows[:, :, None], 0.0), axis = 1,
                )
            else:
                # LL / GeneralLL. emars ``other=0.0`` is safe: masked batch
                # columns have ``log_n_fdm_max == -inf`` so their
                # ``scaled_emars`` column is exp(-inf) = 0.
                emars = tl.load(
                    element_mars + off_cid[None, :] * batch_size +
                    offs_batch[:, None],
                    mask = mask_batch[:, None], other = 0.0,
                )  # [TILE_B, TILE_N]

                if allow_modify_flows == 1:
                    # node_flows pre-transformed to log(flow) - nmars
                    # (times alpha for GeneralLL) by the modify pre-pass.
                    log_n_fdm = tl.load(
                        node_flows + off_node_b, mask = mask_batch[None, :],
                        other = -float("inf"),
                    )  # [TILE_M, TILE_B]
                    if propagation_alg_id == 2:
                        nmars = tl.load(
                            node_mars + off_node_b,
                            mask = mask_batch[None, :], other = 0.0,
                        )
                        log_n_fdm += (alpha - 1.0) * nmars
                else:
                    nmars = tl.load(
                        node_mars + off_node_b, mask = mask_batch[None, :],
                        other = 0.0,
                    )  # [TILE_M, TILE_B]
                    if logspace_flows == 1:
                        nflows_log = tl.load(
                            node_flows + off_node_b,
                            mask = mask_batch[None, :],
                            other = -float("inf"),
                        )
                        log_n_fdm = tl.where(
                            nmars == -float("inf"), -float("inf"),
                            nflows_log - nmars,
                        )
                    elif allow_neg_flows == 1:
                        log_n_fdm = tl.where(
                            nmars == -float("inf"), -float("inf"), -nmars,
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
                log_n_fdm_max = tl.max(log_n_fdm, axis = 0)        # [TILE_B]
                n_fdm_sub = tl.where(
                    log_n_fdm_max[None, :] != -float("inf"),
                    tl.exp(log_n_fdm - log_n_fdm_max[None, :]), 0.0,
                )  # [TILE_M, TILE_B]
                scaled_emars = tl.exp(
                    emars + log_n_fdm_max[:, None],
                )  # [TILE_B, TILE_N]

                if allow_neg_flows == 1:
                    nflows = tl.load(
                        node_flows + off_node_b, mask = mask_batch[None, :],
                        other = 0.0,
                    )
                    if use_tl_dot == 1:
                        partial_flows = tl.dot(n_fdm_sub * nflows, scaled_emars)
                    else:
                        partial_flows = tl.sum(
                            (n_fdm_sub * nflows)[:, :, None]
                            * scaled_emars[None, :, :], axis = 1,
                        )
                else:
                    if use_tl_dot == 1:
                        partial_flows = tl.dot(n_fdm_sub, scaled_emars)
                    else:
                        partial_flows = tl.sum(
                            n_fdm_sub[:, :, None] * scaled_emars[None, :, :],
                            axis = 1,
                        )

                acc += partial_flows

            offs_batch += TILE_B

        if propagation_alg_id != 1:
            # Upcast covers bf16 params transparently (mirrors the ele
            # kernel); pflows always accumulate in fp32.
            epars = tl.load(mparams + epars_offs).to(tl.float32)
            pflows = acc * epars
        else:
            pflows = acc

        pf_offs = pfid_start + edge_block_off + intra_block_offs
        ori = tl.load(param_flows + pf_offs)
        if negate_pflows == 1:
            tl.store(param_flows + pf_offs, ori - pflows)
        else:
            tl.store(param_flows + pf_offs, ori + pflows)


def _greatest_power_of_2_divisor(n: int, cap: int) -> int:
    # Largest power of 2 that divides n and is ≤ cap.
    p = 1
    while p * 2 <= cap and n % (p * 2) == 0:
        p *= 2
    return p


@functools.lru_cache(maxsize=16)
def _target_grid_size(device_index: int) -> int:
    # Target ~1 program per SM. Going higher (e.g. 4x) forces smaller tiles
    # that lose arithmetic intensity faster than they gain from occupancy at
    # HMM-shaped workloads. Falls back to a fixed value if CUDA properties
    # can't be queried.
    try:
        n_sm = torch.cuda.get_device_properties(device_index).multi_processor_count
        return n_sm
    except Exception:
        return 128


def _shrink_for_grid(tile_m: int, block_b: int, m_total: int, batch_size: int,
                     target_grid: int, dot_floor: int = 16):
    # Halve whichever of (tile_m, block_b) is currently larger until the grid
    # reaches target_grid or both tiles hit the tl.dot-friendly floor.
    def _grid() -> int:
        return triton.cdiv(batch_size, block_b) * (m_total // tile_m)

    while _grid() < target_grid:
        can_shrink_m = tile_m > dot_floor and (tile_m // 2) >= 1
        can_shrink_b = block_b > dot_floor
        if can_shrink_m and can_shrink_b:
            if tile_m >= block_b:
                tile_m //= 2
            else:
                block_b //= 2
        elif can_shrink_m:
            tile_m //= 2
        elif can_shrink_b:
            block_b //= 2
        else:
            break
    return tile_m, block_b


def _select_fw_tiles(bs: int, total_edges: int, batch_size: int, NB: int,
                     force_use_bf16: bool, force_use_fp32: bool):
    # Start from the largest tl.dot-friendly tiles (same cap as before), then
    # shrink TILE_M / BLOCK_B toward the 16x16 floor until the launch grid
    # (ceil(B/BLOCK_B) * NB * bs/TILE_M) saturates the SMs. TILE_K is left
    # large — it does not affect the grid, only the inner K-loop trip count.
    TILE_M = _greatest_power_of_2_divisor(bs, 64)
    TILE_K = _greatest_power_of_2_divisor(total_edges, 64)
    BLOCK_B = min(triton.next_power_of_2(batch_size), 64)
    if BLOCK_B < 1:
        BLOCK_B = 1
    # Triton (observed on 3.6 / H100) miscompiles a 3-D broadcast reduction
    # over the MIDDLE axis — ``tl.sum(a[:,:,None] * b[None,:,:], axis=1)`` —
    # when that axis is < 8 and BOTH outer dims are >= 16: the result comes
    # out scaled by 8/axis_size. The non-tl.dot path reduces over TILE_K, so
    # when TILE_K < 8 keep one outer dim (BLOCK_B) below 16 to stay on
    # verified-safe shapes.
    if TILE_K < 8:
        BLOCK_B = min(BLOCK_B, 8)

    target_grid = _target_grid_size(torch.cuda.current_device())
    TILE_M, BLOCK_B = _shrink_for_grid(
        tile_m = TILE_M, block_b = BLOCK_B,
        m_total = NB * bs, batch_size = batch_size,
        target_grid = target_grid,
    )

    use_tl_dot = 1 if (TILE_M >= 16 and TILE_K >= 16 and BLOCK_B >= 16) else 0
    # Match the sparse-path policy: bf16 inputs to tl.dot whenever tl.dot is
    # chosen. Keeps numerical behaviour aligned between the two kernels, which
    # is what tests (and users comparing outputs) expect.
    if force_use_fp32:
        use_bf16 = 0
    else:
        use_bf16 = 1 if use_tl_dot else 0

    return TILE_M, TILE_K, BLOCK_B, use_bf16, use_tl_dot


def _select_par_tiles(bs: int, cbs: int, batch_size: int, NB: int, NB_ch: int,
                      mpe: bool = False):
    # Param-flow grid is (NB_ch * cbs/TILE_N, NB * bs/TILE_M); the batch is
    # looped inside each program (deterministic accumulation, no atomics), so
    # only TILE_M / TILE_N can grow the grid — shrink them toward the
    # tl.dot-friendly 16 floor when the launch is too small. Reuse
    # ``_shrink_for_grid`` with the child-edge count standing in for the
    # batch axis.
    #
    # MPE materialises a [TILE_M, TILE_B, TILE_N] broadcast (no tl.dot), so
    # cap its tiles at 16 to keep register pressure sane.
    tile_cap = 16 if mpe else 64
    TILE_M = _greatest_power_of_2_divisor(bs, tile_cap)
    TILE_N = _greatest_power_of_2_divisor(cbs, tile_cap)
    # Floor TILE_B at 8: Triton (observed on 3.x / H100) miscompiles
    # ``tl.sum(a[:,:,None] * b[None,:,:], axis=1)`` when the reduction axis
    # is < 8, scaling the result by 8/axis_size. Masked tail lanes
    # contribute exact zeros, so an oversized batch tile is always safe.
    TILE_B = max(min(triton.next_power_of_2(batch_size), tile_cap), 8)

    target_grid = _target_grid_size(torch.cuda.current_device())
    TILE_M, TILE_N = _shrink_for_grid(
        tile_m = TILE_M, block_b = TILE_N,
        m_total = NB * bs, batch_size = NB_ch * cbs,
        target_grid = target_grid,
    )

    use_tl_dot = 1 if (TILE_M >= 16 and TILE_N >= 16 and TILE_B >= 16) else 0
    return TILE_M, TILE_N, TILE_B, use_tl_dot


def _select_bk_tiles(cbs: int, bs: int, batch_size: int, NB_ch: int,
                     force_use_fp32: bool):
    # Backward grid is (ceil(B/BLOCK_B), NB_ch * cbs/TILE_N). TILE_K loops
    # over the parent dimension inside the kernel, so only TILE_N and BLOCK_B
    # can grow the grid — shrink them toward 16 when the launch is too small.
    TILE_N = _greatest_power_of_2_divisor(cbs, 64)
    TILE_K = _greatest_power_of_2_divisor(bs, 64)
    BLOCK_B = min(triton.next_power_of_2(batch_size), 64)
    if BLOCK_B < 1:
        BLOCK_B = 1
    # Same Triton middle-axis-reduction guard as ``_select_fw_tiles``: the
    # non-tl.dot backward path reduces over TILE_K (the parent axis), which
    # is < 8 for bs < 8 layers (e.g. every block_size-1 root). Without the
    # cap, element flows at BLOCK_B >= 16 come out scaled by 8/TILE_K —
    # invisible to normalised queries but fatal for param-flow accumulation.
    if TILE_K < 8:
        BLOCK_B = min(BLOCK_B, 8)

    target_grid = _target_grid_size(torch.cuda.current_device())
    TILE_N, BLOCK_B = _shrink_for_grid(
        tile_m = TILE_N, block_b = BLOCK_B,
        m_total = NB_ch * cbs, batch_size = batch_size,
        target_grid = target_grid,
    )

    use_tl_dot = 1 if (TILE_N >= 16 and TILE_K >= 16 and BLOCK_B >= 16) else 0
    return TILE_N, TILE_K, BLOCK_B, use_tl_dot
