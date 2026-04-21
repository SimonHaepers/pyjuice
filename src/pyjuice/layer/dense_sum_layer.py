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
    Inference-only fast path for sum layers with fully-connected (dense) block
    topology. Skips the block-sparse partitioning in ``SumLayer.__init__`` and
    uses Triton kernels that address parameters by direct pointer arithmetic.

    Only supports:
      * ``SumNodes`` with ``is_block_dense == True``
      * ``num_chs == 1`` per ``SumNodes`` (children contiguous in element_mars)
      * ``not is_tied()``
    Parameter flow accumulation (learning) is not implemented; callers must
    pass ``param_flows=None`` to :meth:`backward`.
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
            assert not ns.is_tied(), (
                "DenseSumLayer does not support tied parameters yet."
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
            pid_end = curr_pid + ns.num_edges
            pfid_end = curr_pfid + ns.num_edges
            ns._param_range = (curr_pid, pid_end)
            ns._param_flow_range = (curr_pfid, pfid_end)

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
            ns._param_ids = curr_pid + edge_lin_ids * bs * cbs
            ns._inverse_param_ids = torch.argsort(edge_lin_ids)

            blocks.append((
                curr_nid,                      # nid_start in node_mars
                cs._output_ind_range[0],       # cid_start in element_mars
                curr_pid,                      # pid_start in flat params
                curr_pfid,                     # pfid_start (unused for inference)
                NB, NB_ch, bs, cbs,
            ))

            curr_nid += ns.num_nodes
            curr_pid = pid_end
            curr_pfid = pfid_end
            layer_num_nodes += ns.num_nodes
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

        for block in self._dense_blocks:
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, bs, cbs = block

            total_edges = NB_ch * cbs

            TILE_M, TILE_K, BLOCK_B, use_bf16, use_tl_dot = _select_fw_tiles(
                bs = bs, total_edges = total_edges, batch_size = batch_size,
                NB = NB,
                force_use_bf16 = force_use_bf16, force_use_fp32 = force_use_fp32,
            )

            K_NUM_TILES = total_edges // TILE_K
            grid = (triton.cdiv(batch_size, BLOCK_B), NB * (bs // TILE_M))

            self._fw_triton_dense_kernel[grid](
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
                BLOCK_B = BLOCK_B,
                TILE_M = TILE_M,
                TILE_K = TILE_K,
                K_NUM_TILES = K_NUM_TILES,
                propagation_alg_id = propagation_alg_id,
                use_bf16 = use_bf16,
                use_tl_dot = use_tl_dot,
                alpha = alpha,
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

        if param_flows is not None:
            raise NotImplementedError(
                "DenseSumLayer is inference-only; parameter-flow accumulation "
                "is not supported. Set use_dense_sum_layer=False on "
                "TensorCircuit for learning."
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
            nid_start, cid_start, pid_start, _pfid_start, NB, NB_ch, bs, cbs = block

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

        return None

    # ------------------------------------------------------------------ #
    # Triton kernels
    # ------------------------------------------------------------------ #

    @staticmethod
    @triton_jit
    def _fw_triton_dense_kernel(node_mars, element_mars, mparams,
                                batch_size: tl.constexpr,
                                nid_start: tl.constexpr, cid_start: tl.constexpr,
                                pid_start: tl.constexpr, NB_ch: tl.constexpr,
                                BS: tl.constexpr, CBS: tl.constexpr,
                                BLOCK_B: tl.constexpr, TILE_M: tl.constexpr,
                                TILE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                propagation_alg_id: tl.constexpr,
                                use_bf16: tl.constexpr, use_tl_dot: tl.constexpr,
                                alpha = 0.0):

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
        par_block_base = pid_start + nblock_id * NB_ch * CBS * BS
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
        emars_ptr = element_mars + \
            (cid_start + offs_edge)[:, None] * batch_size + \
            offs_batch[None, :]                            # [TILE_K, BLOCK_B]

        acc = tl.zeros([TILE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for _k in range(0, K_NUM_TILES):
            # epars: TILE_M rows each contiguous in memory (stride 1 along M).
            epars = tl.load(epars_ptr)
            # emars: BLOCK_B inner dim contiguous (stride 1 along batch).
            emars = tl.load(emars_ptr, mask = mask_batch[None, :])

            if propagation_alg_id == 1:
                # MPE propagation
                lpars = tl.log(epars)
                nmars = tl.max(lpars[:, :, None] + emars[None, :, :], axis = 1)
                acc = tl.maximum(acc, nmars)
            else:
                if propagation_alg_id == 0:
                    emars_max = tl.max(emars, axis = 0)[None, :]
                    emars_sub = tl.where(
                        emars_max != -float("inf"),
                        tl.exp(emars - emars_max), 0.0,
                    )

                if propagation_alg_id == 2:
                    emars_max = tl.max(emars, axis = 0)[None, :]
                    emars_sub = tl.where(
                        emars_max != -float("inf"),
                        tl.exp((emars - emars_max) * alpha), 0.0,
                    )
                    epars = tl.exp(tl.log(epars) * alpha)
                    emars_max *= alpha

                if use_tl_dot == 1:
                    if use_bf16 == 1:
                        epars_b = epars.to(tl.bfloat16)
                        emars_b = emars_sub.to(tl.bfloat16)
                        nmars = tl.dot(epars_b, emars_b).to(tl.float32)
                    else:
                        nmars = tl.dot(epars, emars_sub)
                else:
                    nmars = tl.sum(
                        epars[:, :, None] * emars_sub[None, :, :], axis = 1
                    )

                # Numerically stable running-max logsumexp — matches the
                # block-sparse kernel verbatim.
                acc = tl.where(
                    emars_max > acc,
                    tl.log(nmars + tl.exp(acc - emars_max) + 1e-24) + emars_max,
                    tl.where(
                        acc != -float("inf"),
                        tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc,
                        -float("inf"),
                    ),
                )

            # Advance along K: next TILE_K edges = TILE_K*BS scalars (params)
            # and TILE_K*batch_size scalars (emars rows). Both are multiples
            # of TILE_M / BLOCK_B respectively, so alignment is preserved.
            epars_ptr += TILE_K * BS
            emars_ptr += TILE_K * batch_size

        if propagation_alg_id == 2:
            # Rescale back: node_mars = (log sum w^alpha * p^alpha) / alpha
            acc = acc * (1.0 / alpha)

        # Store [TILE_M, BLOCK_B]: inner dim BLOCK_B is stride 1 (contiguous,
        # coalesced), outer dim TILE_M is stride batch_size.
        off_out = off_nid[:, None] * batch_size + offs_batch[None, :]
        tl.store(node_mars + off_out, acc, mask = mask_batch[None, :])

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
            edge_block_base = pid_start + \
                (pblock_id * NB_ch + cblock_id) * CBS * BS
            # epars [TILE_N, TILE_K] — note: the inner/outer layout is the
            # TRANSPOSE of the forward kernel's epars load.
            #   inner dim (TILE_K, cols): stride 1  — contiguous / coalesced
            #   outer dim (TILE_N, rows): stride BS — strided
            epars = tl.load(
                mparams + edge_block_base +
                offs_child[:, None] * BS + off_pwithin[None, :]
            )  # [TILE_N, TILE_K]

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

    target_grid = _target_grid_size(torch.cuda.current_device())
    TILE_N, BLOCK_B = _shrink_for_grid(
        tile_m = TILE_N, block_b = BLOCK_B,
        m_total = NB_ch * cbs, batch_size = batch_size,
        target_grid = target_grid,
    )

    use_tl_dot = 1 if (TILE_N >= 16 and TILE_K >= 16 and BLOCK_B >= 16) else 0
    return TILE_N, TILE_K, BLOCK_B, use_tl_dot
