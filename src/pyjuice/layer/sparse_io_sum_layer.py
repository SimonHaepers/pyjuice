from __future__ import annotations


import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from pyjuice.nodes.distributions import sparse_categorical as _scat
from .sparse_input_sum_layer import SparseInputSumLayer
from .sparse_node_values import SparseNodeValues
from . import sparse_io_cuda as _sio_cuda


_FWD_BLOCK_K = 64
"""Fixed K_in-axis tile size for :func:`_sparse_io_sum_forward_kernel` and
:func:`_sparse_io_sum_forward_single_kernel`. Matches
``sparse_input_sum_layer._FWD_BLOCK_K``; kept as a separate constant so the
kernels can diverge in tuning if needed."""

_FWD_SINGLE_PROGRAMS_PER_SM = 8
"""Programs per SM of the single-sample forward
(:func:`_sparse_io_sum_forward_single_kernel`). Every loop step of a program
gathers one ``[TILE_M, BLOCK_K]`` tile of scattered weights and then waits a
full memory latency for it, so the kernel only approaches HBM bandwidth with
many programs in flight per SM. One program per parent tile gives 15-94 of
them on the trained H=8k-16k chains (under one wave of 132 SMs, 0.5-1 TB/s
where cuSPARSE reads 2.8 TB/s), so the children are split as well (split-K:
one partial sum per (parent tile, child range), reduced by
:func:`_sparse_io_sum_forward_single_epilogue_kernel`) up to this many
programs per SM. The split is computed on the device from the actual column
lengths, so a captured B=1 graph adapts to the batch it replays on."""

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
# Batched sparse pass
# --------------------------------------------------------------------------- #
#
# The per-sample gather kernels above read ``W[h, c]`` once per (sample, active
# pair), each from its own 32-byte sector; at training batch sizes the samples'
# supports together cover nearly every row of W at every step, so they move
# many passes' worth of W (~2 GB per step at H=16k, B=64, where W is 1 GB).
# From B=2 on, every batched block whose PC blocks have at least 16 rows and
# whose columns ascend runs the sparse pass of :mod:`.sparse_io_cuda` instead:
# W is copied once per pass into slab order (tiles of ``_WS_SPARSE_SLAB_TM``
# parent rows, ranges of children whose ``[KS, TMS]`` slab fits
# ``_WS_SPARSE_SLAB_BYTES`` of shared memory, in bf16 with
# ``_WS_SPARSE_SLAB_W16``), a CTA keeps one slab and walks the packed input
# column of every sample with an active parent in its tile, forming only the
# products of the active pairs on the CUDA cores and writing the packed
# output; the element-flow backward is the same kernel with tiles of children
# and ranges of parents. B=1 keeps the gather kernels, which are faster there.
#
# Parameter flows of tied transitions (the homogeneous HMM chain) are deferred
# to the pass end (:class:`_WSGroup`): either every step's packed sparse
# inputs are stashed and accumulated every few steps on stationary tiles in
# shared memory (``_WS_SPARSE_PFLOW`` "tile", large W), or every step's
# densified factors are stashed and bounded chunks feed one cuBLAS GEMM,
# ``S += G_t^T X_t`` ("gemm", small W); ``pflow += W * S`` once per pass.
# :meth:`_WSGroup.ensure_ready` falls back to the gather kernels rather than
# OOM when the stash does not fit.
#
# Until 2026-11-02 the layer also carried tensor-core passes over dense tiles
# of W (a full pass, a pass over the batch's active children, a pass over the
# per-sample active parent tiles) and a cost model choosing among them per
# batch; on the density-0.01 chains the sparse pass wins from B=2 on, so they
# were removed (prune-pc REPORT_2026-10-26_tiling.md, section 13).

_WS_MODE = os.environ.get("PYJUICE_SPARSE_IO_WS", "auto")
"""``"auto"`` (the default) — batched sparse-IO blocks take the sparse pass
from B=2 on, the block-diagonal layers their dense batched path from
``_WS_MIN_BATCH`` samples up; ``"always"`` — the block-diagonal layers go
dense at any B > 1 (tests; the sparse-IO layer as in ``"auto"``);
``"never"`` — the per-sample gather kernels only."""

_WS_MIN_BATCH = int(os.environ.get("PYJUICE_SPARSE_IO_WS_MIN_BATCH", "16"))
"""Smallest batch the block-diagonal (Monarch) layers run their dense batched
path at in ``"auto"`` mode (``_bd_dense_batched`` in
``sparse_io_block_diagonal_sum_layer``); below it their per-sample kernels
win. The sparse-IO layer takes its sparse pass from B=2 on."""

_WS_PRECISION = os.environ.get("PYJUICE_SPARSE_IO_WS_PRECISION", "tf32")
"""Precision of the deferred param-flow GEMM (``_WS_SPARSE_PFLOW`` "gemm"):
``"tf32"`` — tensor-core tf32 on factors rounded to nearest tf32 at densify
time, so the error is zero-mean (~1e-4 relative) rather than the ~-6e-4 bias
of tf32 truncation; ``"ieee"`` — exact fp32. The sparse forward and backward
kernels do not depend on it."""

_WS_SPARSE_SLAB_TM = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_SLAB_TM", "4"))
"""Parent rows of a slab tile, 4 / 8 / 16 (the backward: child rows). 4 by
default: with the bf16 slabs the 4-row kernels ran faster than the 8-row
ones at every measured point (E35, 2026-11-02, H100, forward kernel ms per
batch of 127 steps: 21.8 vs 25.9 at 8k d0.01 B=1024, 9.7 vs 11.1 at 8k
B=256, 49.7 vs 50.8 at 16k B=512; in fp32 24.3 vs 32.3, 12.2 vs 12.8, 72.1
vs 85.4)."""
assert _WS_SPARSE_SLAB_TM in _sio_cuda.TILE_HEIGHTS, _WS_SPARSE_SLAB_TM
_WS_SPARSE_SLAB_BYTES = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_SLAB_KB", "64")) * 1024
"""Shared memory of one fp32 slab: the children are split into ranges of that
size (64 KB: three CTAs per SM; 2 ranges on an 8k chain with 4-row tiles; a
bf16 slab of the same ranges takes half). 128 KB (one range at 8k, 2 at 16k)
measured slower: 85 vs 50 ms per batch at 16k B=512 (E35)."""
_WS_SPARSE_SLAB_NWARP: Dict[int, int] = {4: 16, 8: 12}
"""Warps per CTA by tile height (16 for others); env
``PYJUICE_SPARSE_IO_WS_SPARSE_SLAB_NWARP`` overrides. Chosen 2026-10-24 on 8k /
16k d0.01 (B=64..2048): occupancy-bound, the most warps whose registers fit
three CTAs per SM without spills (40 / 56 registers)."""
if os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_SLAB_NWARP"):
    _WS_SPARSE_SLAB_NWARP = {t: int(os.environ["PYJUICE_SPARSE_IO_WS_SPARSE_SLAB_NWARP"])
                             for t in _sio_cuda.TILE_HEIGHTS}
_WS_SPARSE_SLAB_MINB = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_SLAB_MINB", "3"))
"""CTAs per SM the compiler budgets registers for (``__launch_bounds__``)."""
_WS_SPARSE_SLAB_W16 = os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_SLAB_W16", "1") == "1"
"""The slab-order copies of W in bf16 (half the bytes of every W read from shared memory and of
every slab load; the kernels unpack to fp32 before the multiply). The forward and the element-flow
backward then round W to 8 bits of mantissa (log outputs within about 6e-4 of fp32 at 8k d0.01).
Measured 2026-11-02 (E35, H100): forward + backward kernels -10 % at 8k B=1024, -22 % at 8k B=256,
-28 % at 16k B=512. fp16 is unusable: 99 % of the trained W's entries lie below its normal range."""
_WS_SPARSE_PFLOW = os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_PFLOW", "auto")
"""Parameter flows of a block on the sparse pass: ``"gemm"`` — the deferred dense GEMM of
:class:`_WSGroup` (every step's densified factors stashed, bounded chunks accumulated by cuBLAS);
``"tile"`` — each step's sparse inputs are packed into a stash and every ``_WS_SPARSE_PF_TILE_SLOTS``
steps accumulated on stationary accumulator tiles of 4 parent rows in shared memory, the samples with
the same child list (next token) summed in registers first, into the group's deferred accumulator,
multiplied by W at the pass end (:func:`.sparse_io_cuda.pflow_tile_flush`); ``"auto"`` — ``"tile"`` for a
W of at least ``_WS_SPARSE_PF_TILE_MIN_W`` elements from ``_WS_SPARSE_PF_TILE_MIN_BATCH`` samples on (a
larger W from proportionally fewer), else ``"gemm"`` (:func:`_pflow_mode`)."""
_WS_SPARSE_PF_TILE_MIN_W = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_PF_TILE_MIN_W", str(1 << 26)))
_WS_SPARSE_PF_TILE_MIN_BATCH = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_PF_TILE_MIN_BATCH", "256"))
"""``"auto"`` param flows take the tile kernel for a W of at least ``_WS_SPARSE_PF_TILE_MIN_W`` elements (8k x 8k)
from this many samples on, a larger W from proportionally fewer (``B W >= MIN_BATCH MIN_W``; the dense GEMM grows
with W, the tile kernel with the active pairs). EM steps on the trained d0.01 chains, H100 (2026-10-29, snapshot
2026-10-29_pf-tile3 / pf-tile4, tile vs dense GEMM, ms per EM step): 8k B=256 41.6 vs 49.6 (GPU kernels 37.8 vs 47.0),
B=384 47.6 vs 62.6; 16k B=64 104.3 vs 107.8, B=128 129.4 vs 145.5, B=256 166.2 vs 201.0; 2k B=1024 38.6 vs 37.7, B=2048
64.8 vs 60.4 (the dense GEMM of a small W is cheap)."""
_WS_SPARSE_PF_TILE_SLOTS = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_PF_TILE_SLOTS", "32"))
"""Steps stored per accumulation of ``_WS_SPARSE_PFLOW = "tile"`` (at most 32; fewer where the stash would exceed
``_WS_SPARSE_PF_TILE_STASH_BYTES``). Each holds the per-tile sample lists, ``[H_out / 4, B]`` entries, and the packed
children and parents of the batch. Every accumulation zeroes its tiles and adds them to the accumulator, so fewer,
longer ones are faster (pf_bench2 2026-10-29, pf_tile per EM step with 8 / 32 steps: 8k B=1024 25.6 / 22.7 ms, 16k
B=512 83.0 / 70.0 ms)."""
_WS_SPARSE_PF_TILE_STASH_BYTES = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_PF_TILE_STASH_MB", "1024")) << 20
"""Memory of the ``"tile"`` stash; it bounds the steps per accumulation (32 steps at 8k B=1024: about 0.7 GB)."""
_WS_SPARSE_PF_TILE_NWARP = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_PF_TILE_NWARP", "32"))
"""Warps sharing one accumulator tile of ``_WS_SPARSE_PFLOW = "tile"`` (a power of two up to 32; the tile's list
entries are sorted in chunks of ``32 NWARP``)."""
_WS_SPARSE_PF_TILE_BYTES = int(os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_PF_TILE_KB", "128")) * 1024
"""Shared memory of one accumulator tile (4 parent rows x as many children as fit, fp32): 128 KB is the whole
8k state range per tile, one tile per SM (2026-10-28: 4 x 8192 with 32 warps beat 4 x 4096 with 16 warps)."""


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

    output_storage = "packed"
    graph_host_state = ("_sparse_outputs", "_sparse_flows")

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

        # ``(group, slot)`` per block for the batched sparse pass and its
        # deferred parameter flows, assigned by :func:`build_ws_groups`
        # (TensorCircuit compile time).
        # ``None`` keeps the block on the gather kernels.
        self._ws_slots: List[Optional[Tuple["_WSGroup", int]]] = [None] * len(self.nodes)
        self._ws_sorted_memo: Dict[int, bool] = {}    # block -> both its columns ascend

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
                static_ctx=None,
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
        if static_ctx is not None:
            self._forward_static(params, static_ctx)
            return None
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

            if sv_in.is_batched and self._ws_use(blk_idx, batch_size, params):
                self._ws_forward_block(blk_idx, block, sv_in, sv_out, max_val, params)
            elif sv_in.is_batched:
                assert sv_out.is_batched
                self._launch_batched_forward(block, sv_in, sv_out, max_val, params,
                                             batch_size, TILE_M)
            else:
                self._launch_single_forward(blk_idx, block, sv_in, sv_out, max_val, params)
        return None

    def _launch_batched_forward(self, block: tuple, sv_in: SparseNodeValues,
                                sv_out: SparseNodeValues, max_val: torch.Tensor,
                                params: torch.Tensor, batch_size: int,
                                TILE_M: int) -> None:
        """Batched-layout launch of :func:`_sparse_io_sum_forward_kernel` (B>1,
        eager or static containers). Both ``total_nnz`` only bound the grid:
        the kernel bounds its loop by the per-sample sizes on device."""
        _, _, pid_start, _, _, NB_ch, BS, CBS = block
        assert batch_size > 1
        BLOCK_B = _FWD_BATCHED_BLOCK_B
        grid = (triton.cdiv(sv_out.total_nnz, TILE_M),
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
            K_in=sv_in.total_nnz,
            K_out=sv_out.total_nnz,
            batch_size=batch_size,
            in_stride=sv_in.values.stride(0),
            out_stride=sv_out.values.stride(0),
            NB_ch=NB_ch,
            BS=BS,
            CBS=CBS,
            TILE_M=TILE_M,
            BLOCK_K=_FWD_BLOCK_K,
            BLOCK_B=BLOCK_B,
        )

    def _launch_single_forward(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                               sv_out: SparseNodeValues, max_val: torch.Tensor,
                               params: torch.Tensor, static_ctx=None) -> None:
        """Single-sample launch (eager B=1, or the static B=1 containers of a
        CUDA-graph forward, which read their column offsets / sizes on the
        device). The grid is ``min(parent tiles * child tiles,
        _FWD_SINGLE_PROGRAMS_PER_SM * SMs)`` programs that walk the (parent
        tile, child range) work items — see
        :func:`_sparse_io_sum_forward_single_kernel`. An eager launch whose
        children fit one range per parent tile writes the result directly;
        otherwise the kernel writes one partial sum per work item and the
        epilogue reduces them."""
        _, _, pid_start, _, _, NB_ch, BS, CBS = block
        TILE_M, BLOCK_K = _scat.B1_TILE, _FWD_BLOCK_K
        static = sv_in.static
        assert static == sv_out.static and (not static or static_ctx is not None)
        # Stream / gather descriptors of the output column (see
        # ``SparseCategorical.b1_descriptors``): dense parent tiles are streamed
        # whole, the remaining active parents gathered. Eager: looked up by the
        # token; static: the per-step ``[4]`` device view the graph computed.
        desc = None
        if _scat.b1_hybrid_enabled() and (sv_out.b1_desc is not None or sv_out.col_id >= 0):
            desc = self._output_sparsity_dists[blk_idx].b1_descriptors(params.device)
        hybrid = desc is not None
        if hybrid and not static:
            v = sv_out.col_id
            d0, d1 = desc["dense_indptr_list"][v], desc["dense_indptr_list"][v + 1]
            s0, s1 = desc["sparse_indptr_list"][v], desc["sparse_indptr_list"][v + 1]
            n_dense, n_sparse = d1 - d0, s1 - s0
            k_out_eff = n_dense * TILE_M + n_sparse
        else:
            d0 = s0 = n_dense = n_sparse = 0
            # a bound: every dense tile adds at most TILE_M - 1 inactive lanes
            k_out_eff = sv_out.total_nnz + (sv_out.num_rows if hybrid else 0)
        n_m = triton.cdiv(k_out_eff, TILE_M)                # a bound for static containers
        k_tiles = triton.cdiv(sv_in.total_nnz, BLOCK_K)
        n_prog = max(1, min(n_m * k_tiles, _FWD_SINGLE_PROGRAMS_PER_SM * _num_sms(params.device)))
        direct = not static and not hybrid and _single_forward_splits(k_tiles, n_m, n_prog)[0] == 1
        if direct:
            partials = sv_out.values                       # unused
        else:
            n = max(n_prog, n_m) * TILE_M                  # >= splits * parent tiles * TILE_M
            if static:
                partials = static_ctx.buffer((id(self), "fwd_partials", blk_idx), (n,))
            else:
                partials = _ws_scratch("fwd_partials", (n,), params.device)
        dummy = sv_in.values
        _sparse_io_sum_forward_single_kernel[(n_prog,)](
            values_out_ptr=sv_out.values,
            partials_ptr=partials,
            mparams_ptr=params,
            in_indices_ptr=sv_in.indices,
            in_values_ptr=sv_in.values,
            max_val_ptr=max_val,
            out_indices_ptr=sv_out.indices,
            in_col_starts_ptr=sv_in.col_starts if static else dummy,  # unused unless static
            in_nnz_ptr=sv_in.nnz if static else dummy,
            out_col_starts_ptr=sv_out.col_starts if static else dummy,
            out_nnz_ptr=sv_out.nnz if static else dummy,
            dense_tiles_ptr=desc["dense_tiles"] if hybrid else dummy,
            sparse_rows_ptr=desc["sparse_rows"] if hybrid else dummy,
            desc_ptr=sv_out.b1_desc if (hybrid and static) else dummy,
            dense_start=d0, n_dense=n_dense, sparse_start=s0, n_sparse=n_sparse,
            num_rows=sv_out.num_rows,
            pid_start=pid_start,
            K_in=sv_in.total_nnz,
            K_out=sv_out.total_nnz,
            n_prog=n_prog,
            NB_ch=NB_ch,
            BS=BS,
            CBS=CBS,
            TILE_M=TILE_M,
            BLOCK_K=BLOCK_K,
            DEVICE_SIZES=static,
            DIRECT=direct,
            HYBRID=hybrid,
            ALIGNED=BS % TILE_M == 0,
            ROWS_MASK=sv_out.num_rows % TILE_M != 0,
        )
        if not direct:
            BLOCK = 256
            _sparse_io_sum_forward_single_epilogue_kernel[(triton.cdiv(sv_out.total_nnz, BLOCK),)](
                values_out_ptr=sv_out.values,
                partials_ptr=partials,
                max_val_ptr=max_val,
                in_nnz_ptr=sv_in.nnz if static else dummy,
                out_nnz_ptr=sv_out.nnz if static else dummy,
                out_col_starts_ptr=sv_out.col_starts if static else dummy,
                slot_ptr=desc["slot"] if hybrid else dummy,
                desc_ptr=sv_out.b1_desc if (hybrid and static) else dummy,
                col_start=sv_out.col_start,
                n_dense=n_dense, n_sparse=n_sparse,
                K_in=sv_in.total_nnz,
                K_out=sv_out.total_nnz,
                n_prog=n_prog,
                TILE_M=TILE_M,
                BLOCK_K=BLOCK_K,
                BLOCK=BLOCK,
                DEVICE_SIZES=static,
                HYBRID=hybrid,
            )

    def _forward_static(self, params: torch.Tensor, static_ctx) -> None:
        """CUDA-graph forward (see ``pyjuice.model.sparse_cudagraph``): output
        columns resolved on device, workspaces owned by ``static_ctx``. There
        are no empty-column shortcuts — the kernels already write ``-inf`` for
        a sample with an empty input column and nothing for one with an empty
        output column."""
        B = static_ctx.B
        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._dense_blocks, self._sparse_input_refs,
        )):
            sv_in = sparse_prod._sparse_outputs[ns_idx]
            assert sv_in.static and sv_in.max_val is not None, (
                "SparseIOSumLayer: the upstream prod output is not a static "
                "(CUDA-graph) container."
            )
            out_dist = self._output_sparsity_dists[blk_idx]
            out_var = self._output_sparsity_var_ids[blk_idx]
            ws = static_ctx.buffer((id(self), "fwd", blk_idx),
                                   (B * static_ctx.capacity(out_dist, out_var),))
            sv_out = static_ctx.pattern(
                out_dist, out_var, self._output_sparsity_num_rows[blk_idx], ws,
            )
            self._sparse_outputs[blk_idx] = sv_out

            if self._ws_use_static(blk_idx, params, static_ctx):
                self._ws_forward_block(blk_idx, block, sv_in, sv_out, sv_in.max_val, params)
            elif B == 1:
                self._launch_single_forward(blk_idx, block, sv_in, sv_out, sv_in.max_val,
                                            params, static_ctx)
            else:
                TILE_M = 32
                self._launch_batched_forward(block, sv_in, sv_out, sv_in.max_val,
                                             params, B, TILE_M)

    def _ws_use_static(self, blk_idx: int, params: torch.Tensor, static_ctx) -> bool:
        """:meth:`_ws_use` for the CUDA-graph forward, memoised in the container
        so that warmup and capture dispatch alike."""
        key = (id(self), blk_idx)
        use = static_ctx.ws_decisions.get(key)
        if use is None:
            use = static_ctx.ws_decisions[key] = self._ws_use(blk_idx, static_ctx.B, params)
        return use

    def _sparse_eligible(self, blk_idx: int) -> bool:
        """Whether the block can take the sparse pass: it has a weight group
        whose buffers fit (PC blocks of at least 16 rows,
        :func:`build_ws_groups`; :meth:`_WSGroup.ensure_ready`), and both its
        input and output columns ascend (the CSC rows ascend in every column,
        :meth:`SparseCategorical.columns_sorted`; a host-side scan of the index
        arrays, memoised per block)."""
        ws = self._ws_slots[blk_idx]
        if ws is None or not ws[0].eligible:
            return False
        hit = self._ws_sorted_memo.get(blk_idx)
        if hit is None:
            prod, ns_idx = self._sparse_input_refs[blk_idx]
            in_dist = prod.nodes[ns_idx].sparse_input_ns.dist
            hit = self._ws_sorted_memo[blk_idx] = bool(
                in_dist.columns_sorted() and self._output_sparsity_dists[blk_idx].columns_sorted())
        return hit

    def graph_columns(self):
        return list(zip(self._output_sparsity_dists, self._output_sparsity_var_ids))

    def graph_rebind_outputs(self, static_ctx, data_cpu: torch.Tensor,
                             data_list: Optional[list],
                             pattern_cache: Optional[dict]) -> None:
        """After a CUDA-graph replay, before an eager backward: rebuild
        ``self._sparse_outputs`` the way the eager forward would have (host
        ``total_nnz`` / ``nnz_list``, from which the backward sizes its grids
        and picks its kernels), with ``values`` aliasing the ``static_ctx``
        buffers the replay wrote — same memory layout at every B."""
        B = static_ctx.B
        for blk_idx in range(len(self._sparse_outputs)):
            out_dist = self._output_sparsity_dists[blk_idx]
            ws = static_ctx.buffer((id(self), "fwd", blk_idx),
                                   (B * out_dist._max_nnz_per_col,))
            self._sparse_outputs[blk_idx] = out_dist.build_sparse_pattern(
                data=data_cpu, var_id=self._output_sparsity_var_ids[blk_idx],
                num_rows=self._output_sparsity_num_rows[blk_idx], device=ws.device,
                values_out=ws, data_list=data_list, pattern_cache=pattern_cache,
            )

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
                needed = sv_in.values.numel()
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
                K_stride = sv_in.values.stride(0)
                flow_values = ws_flow.narrow(
                    0, 0, batch_size * K_stride).view(batch_size, K_stride)
            else:
                flow_values = ws_flow.narrow(0, 0, K_in)
            if kwargs.get("static_ctx") is not None:
                flow_values = kwargs["static_ctx"].buffer((id(self), "bwd", blk_idx), tuple(sv_in.values.shape))
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

            if sv_in.is_batched and self._ws_use(blk_idx, batch_size, params, need_stash=compute_pflows):
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
    # Batched sparse pass
    # ------------------------------------------------------------------ #

    def _ws_use(self, blk_idx: int, B: int, params: torch.Tensor, need_stash: bool = False) -> bool:
        """Whether this batched block runs the sparse pass: from B=2 on
        (``_WS_MODE``), when the block is eligible (:meth:`_sparse_eligible`)
        and, with ``need_stash`` (a backward that accumulates parameter flows
        on the deferred GEMM), its stash fits (:meth:`_WSGroup.ensure_ready`;
        a group whose buffers do not fit stays on the gather kernels instead
        of failing the run)."""
        if _WS_MODE == "never" or B < 2 or not self._sparse_eligible(blk_idx):
            return False
        group = self._ws_slots[blk_idx][0]
        return group.ensure_ready(params, B, need_stash and _pflow_mode(group.H_in, group.H_out, B) == "gemm")

    def _ws_forward_block(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                          sv_out: SparseNodeValues, max_val: torch.Tensor,
                          params: torch.Tensor) -> None:
        """``sv_out = log(W @ exp(sv_in - max)) + max`` for the whole batch on
        the sparse pass (:func:`.sparse_io_cuda.ws_slab_forward`): tiles of
        ``_WS_SPARSE_SLAB_TM`` parents from the slab-order copy of W
        (:meth:`_WSGroup.slab_copy`, rebuilt once per pass), only the
        products of the active pairs, the packed output written directly."""
        group, _ = self._ws_slots[blk_idx]
        _, _, _, _, _NB, _NB_ch, BS, _CBS = block
        B = sv_in.batch_size
        device = params.device
        H_in, H_out = group.H_in, group.H_out
        TMS = _WS_SPARSE_SLAB_TM
        n_m = -(-H_out // TMS)
        K = sv_in.total_nnz
        _KS, nsplit = _sio_cuda.slab_shape(H_in, TMS, _WS_SPARSE_SLAB_BYTES)
        ent = _ws_scratch("sp_ent", (n_m, 2 * B), device, torch.int32)
        cnt = _ws_scratch("sp_cnt", (n_m,), device, torch.int32)
        xp = _ws_scratch("sp_xpack", (B, 2 * K), device, torch.int32)
        cofs = _ws_scratch("sp_cofs", (B, nsplit + 1), device, torch.int32)
        wslab = group.slab_copy(params, TMS, _WS_SPARSE_SLAB_BYTES, BS, w16=_WS_SPARSE_SLAB_W16)
        _sio_cuda.ws_slab_forward(
            sv_out.values, wslab, sv_in, sv_out, max_val, ent, cnt, xp, cofs, B, H_in, H_out, BS,
            TMS=TMS, slab_bytes=_WS_SPARSE_SLAB_BYTES, NWARP=_WS_SPARSE_SLAB_NWARP.get(TMS, 16),
            min_blocks=_WS_SPARSE_SLAB_MINB, w16=_WS_SPARSE_SLAB_W16,
        )

    def _ws_backward_block(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                           sv_out: SparseNodeValues, sv_flow_out: SparseNodeValues,
                           sv_flow_in: SparseNodeValues, params: torch.Tensor,
                           param_flows: Optional[torch.Tensor], negate_pflows: bool) -> None:
        """Element flows on the sparse pass
        (:func:`.sparse_io_cuda.ws_slab_backward`: tiles of
        ``_WS_SPARSE_SLAB_TM`` children, ranges of parents, the packed flows
        written directly). The parameter flows are only *stashed* here — the
        step's packed inputs for the tile accumulation (``_WS_SPARSE_PFLOW``
        "tile"), or its densified ``G`` / ``X`` factors in this block's slot
        of the group's stash ("gemm") — and accumulated by
        :meth:`_WSGroup.flush`.

        With ``gs[b, h] = nflow[b, h] * exp(max_b - nmars[b, h])`` and
        ``x[b, c] = exp(in[b, c] - max_b)`` (zero off-support):
          flow_in[b, c] = x[b, c] * sum_h gs[b, h] W[h, c]
          pflow[h, c]  += W[h, c] * sum_b gs[b, h] x[b, c]
        """
        group, slot = self._ws_slots[blk_idx]
        _, _, _, _, _NB, _NB_ch, BS, _CBS = block
        B = sv_in.batch_size
        device = params.device
        H_in, H_out = group.H_in, group.H_out
        max_val = sv_in.max_val
        assert max_val is not None, (
            "the sparse backward needs the fused per-sample sv_in.max_val from "
            "the upstream prod layer's re-forward."
        )
        if param_flows is not None:
            if _pflow_mode(H_in, H_out, B) == "tile":
                group.tile_stash(params, param_flows, sv_in, sv_out, sv_flow_out, max_val, B, negate_pflows)
            else:
                rnd = _WS_PRECISION == "tf32"
                gt, xt, col0 = group.stash_slot(slot, B, device, negate_pflows, params, param_flows)
                _ws_densify_g_kernel[(triton.cdiv(sv_out.total_nnz, _WS_VEC), B)](
                    gt, sv_out.indices, sv_flow_out.values, sv_out.values, max_val,
                    sv_in.nnz, sv_out.col_starts, sv_out.nnz,
                    sv_flow_out.values.stride(0), sv_out.values.stride(0), gt.stride(0), col0,
                    BLOCK=_WS_VEC, ROUND=rnd,
                )
                _ws_densify_x_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), B)](
                    xt, sv_in.indices, sv_in.values, max_val, sv_in.col_starts, sv_in.nnz,
                    sv_in.values.stride(0), xt.stride(0), col0,
                    BLOCK=_WS_VEC, ROUND=rnd,
                )

        TMS = _WS_SPARSE_SLAB_TM
        n_t = -(-H_in // TMS)
        _KS, nsplit = _sio_cuda.slab_shape(H_out, TMS, _WS_SPARSE_SLAB_BYTES)
        ent = _ws_scratch("sp_ent", (n_t, 2 * B), device, torch.int32)
        cnt = _ws_scratch("sp_cnt", (n_t,), device, torch.int32)
        xp = _ws_scratch("sp_xpack", (B, 2 * sv_out.total_nnz), device, torch.int32)
        cofs = _ws_scratch("sp_cofs", (B, nsplit + 1), device, torch.int32)
        wslab = group.slab_copy(params, TMS, _WS_SPARSE_SLAB_BYTES, BS, bwd=True, w16=_WS_SPARSE_SLAB_W16)
        _sio_cuda.ws_slab_backward(
            sv_flow_in.values, wslab, sv_in, sv_out, sv_flow_out, max_val,
            ent, cnt, xp, cofs, B, H_in, H_out, BS, TMS=TMS, slab_bytes=_WS_SPARSE_SLAB_BYTES,
            NWARP=_WS_SPARSE_SLAB_NWARP.get(TMS, 16), min_blocks=_WS_SPARSE_SLAB_MINB,
            w16=_WS_SPARSE_SLAB_W16,
        )


# =====================================================================
# Batched sparse pass: shared state
# =====================================================================

_WS_STASH_BYTES = int(os.environ.get("PYJUICE_SPARSE_IO_WS_STASH_MIB", "256")) * 2**20
"""Factor-buffer budget per tied matrix of the deferred param-flow GEMM; at
least one batch slot is retained. Set to 0 to retain all slots (reference
behavior). Chunking changes only the order of parameter-flow accumulation,
not the batch or the EM update."""

_WS_VEC = 256
"""Element tile of the densify / gather kernels."""

_WS_SCRATCH: Dict[Tuple[str, torch.device], torch.Tensor] = {}
_WS_SLAB_OWNER: Dict[torch.device, tuple] = {}
_WS_NUM_SMS: Dict[torch.device, int] = {}


def _ws_scratch(name: str, shape: Tuple[int, ...], device: torch.device,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Process-wide grow-only scratch (one dtype per name). Layers run one at
    a time on one stream, so every call of the sparse pass can reuse the same
    buffers."""
    n = 1
    for s in shape:
        n *= s
    buf = _WS_SCRATCH.get((name, device))
    if buf is None or buf.numel() < n:
        _WS_SCRATCH.pop((name, device), None)
        buf = torch.empty(n, dtype=dtype, device=device)
        _WS_SCRATCH[(name, device)] = buf
    return buf[:n].view(shape)


def _num_sms(device: torch.device) -> int:
    n_sm = _WS_NUM_SMS.get(device)
    if n_sm is None:
        n_sm = torch.cuda.get_device_properties(device).multi_processor_count
        _WS_NUM_SMS[device] = n_sm
    return n_sm


def _single_forward_splits(k_tiles: int, n_m: int, n_prog: int) -> Tuple[int, int]:
    """Host mirror of the split rule in :func:`_sparse_io_sum_forward_single_kernel`:
    ``(splits, child tiles per split)`` for ``k_tiles`` child tiles, ``n_m``
    parent tiles and ``n_prog`` programs."""
    n_s = max(1, min(k_tiles, n_prog // max(n_m, 1)))
    ct = -(-k_tiles // n_s)
    return -(-k_tiles // ct) if ct > 0 else 1, ct


def _pflow_mode(H_in: int, H_out: int, B: int) -> str:
    """The param-flow path of a sparse-dispatch block (``_WS_SPARSE_PFLOW``, ``"auto"`` resolved)."""
    if _WS_SPARSE_PFLOW != "auto":
        return _WS_SPARSE_PFLOW
    W = H_in * H_out
    return "tile" if W >= _WS_SPARSE_PF_TILE_MIN_W and B * W >= _WS_SPARSE_PF_TILE_MIN_BATCH * _WS_SPARSE_PF_TILE_MIN_W \
        else "gemm"


class _WSGroup:
    """State shared by every :class:`SparseIOSumLayer` block that reads the
    same (tied) transition matrix, i.e. the same ``pid_start`` /
    ``pfid_start`` / shape — all 127 steps of a homogeneous HMM chain:

    * the per-pass copy of W in slab order (:meth:`slab_copy`);
    * the deferred parameter flows: the ``[H_out, H_in]`` accumulator ``S``
      (``pflow += W * S`` at the pass end) and its stash — the stored steps'
      packed inputs (``_WS_SPARSE_PFLOW`` "tile") or the densified factors
      ``Gt[H_out, slots*B]`` / ``Xt[H_in, slots*B]`` of up to the configured
      number of ``B``-column slots ("gemm"; full chunks flush during
      traversal, the remainder at the pass end).

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
        # blocks below 16 rows stay on the gather kernels (the sparse pass
        # and the param-flow epilogue tile the PC blocks in 16s and 64s)
        self.eligible = BS % 16 == 0 and CBS % 16 == 0
        self._slab_valid = False
        self._s: Optional[torch.Tensor] = None      # the deferred accumulator S
        self._s_dirty = False
        self._gt: Optional[torch.Tensor] = None
        self._xt: Optional[torch.Tensor] = None
        self._B = 0
        self._capacity = 0
        self._used = 0
        self._negate: Optional[bool] = None
        self._tbufs: Optional[dict] = None          # _WS_SPARSE_PFLOW "tile": stored steps
        self._tbytes = 0                            # the tile size for the device (pflow_tile_bytes)
        self._tused = 0
        self._tnegate: Optional[bool] = None

    def tile_stash(self, params: torch.Tensor, param_flows: torch.Tensor, sv_in, sv_out, sv_flow_out,
                   max_val: torch.Tensor, B: int, negate: bool) -> None:
        """Store one step's packed parameter-flow inputs (``_WS_SPARSE_PFLOW = "tile"``), accumulating
        the stored steps first when the stash is full or the batch / sign changed."""
        device = params.device
        if self._s is None:
            self._s = torch.zeros(self.H_out, self.H_in, dtype=torch.float32, device=device)
        Kx, Kg = sv_in.values.shape[-1], sv_out.values.shape[-1]
        b = self._tbufs
        if (b is None or b["B"] != B or b["xo"].shape[2] < Kx or b["gp"].shape[2] < Kg
                or (self._tused and self._tnegate != negate)):
            self.tile_flush()
            self._tbytes = _sio_cuda.pflow_tile_bytes(self.H_in, self.H_out, device, _WS_SPARSE_PF_TILE_BYTES,
                                                      _WS_SPARSE_PF_TILE_NWARP)
            _KS, nsplit, n_m = _sio_cuda.pflow_tile_shape(self.H_in, self.H_out, self._tbytes)
            per_step = n_m * (B * 8 + 4) + B * (Kx * 8 + (nsplit + 1) * 4 + 8 + Kg * 4)
            S = max(1, min(_WS_SPARSE_PF_TILE_SLOTS, _WS_SPARSE_PF_TILE_STASH_BYTES // per_step))
            i32 = dict(dtype=torch.int32, device=device)
            b = self._tbufs = {"B": B, "ent": torch.empty(S, n_m, B, 2, **i32), "cnt": torch.zeros(S, n_m, **i32),
                               "xo": torch.empty(S, B, Kx, **i32), "cofs": torch.empty(S, B, nsplit + 1, **i32),
                               "xv": torch.empty(S, B, Kx, dtype=torch.float32, device=device),
                               "gkey": torch.empty(S, B, dtype=torch.int64, device=device),
                               "gp": torch.empty(S, B, Kg, dtype=torch.float32, device=device)}
        elif self._tused == b["cnt"].shape[0]:
            self.tile_flush()
        _sio_cuda.pflow_tile_stash(b, self._tused, sv_in, sv_out, sv_flow_out, max_val, B,
                                   self.H_in, self.H_out, self._tbytes)
        self._tused += 1
        self._tnegate = negate

    def tile_flush(self) -> None:
        """Accumulate the stored steps of ``_WS_SPARSE_PFLOW = "tile"`` into the deferred accumulator."""
        if self._tused:
            _sio_cuda.pflow_tile_flush(self._s, self._tbufs, self._tused, self._tbufs["B"], self.H_in, self.H_out,
                                       -1.0 if self._tnegate else 1.0, _WS_SPARSE_PF_TILE_NWARP,
                                       slab_bytes=self._tbytes)
            self._s_dirty = True
            self._tused = 0
            self._tnegate = None

    def begin_pass(self) -> None:
        """Params may have changed since the last pass (EM step): rebuild the
        slab-order copy on next use. Unflushed factors of an abandoned pass
        are discarded. Reserved slots are zeroed before reuse."""
        self._slab_valid = False
        if self._s_dirty:            # an abandoned pass left flushes in the accumulator
            self._s.zero_()
            self._s_dirty = False
        self._used = 0
        self._negate = None
        self._tused = 0
        self._tnegate = None

    def slab_copy(self, params: torch.Tensor, TMS: int, slab_bytes: int, BS: int,
                  bwd: bool = False, w16: bool = False) -> torch.Tensor:
        """W in the sparse forward's (``bwd``: backward's) slab order
        (:func:`.sparse_io_cuda.slab_wcopy`), a process-wide buffer rebuilt once per pass
        (or when another group, direction or slab shape used it); ``w16``: the copy in bf16
        (:data:`_WS_SPARSE_SLAB_W16`)."""
        device = params.device
        n = _sio_cuda.slab_copy_numel(self.H_in, self.H_out, TMS, slab_bytes, bwd)
        buf = _ws_scratch("w_slab16" if w16 else "w_slab", (n,), device,
                          torch.bfloat16 if w16 else torch.float32)
        tag = (self, buf.data_ptr(), TMS, slab_bytes, bwd, w16)
        if not (self._slab_valid and _WS_SLAB_OWNER.get(device) == tag):
            _sio_cuda.slab_wcopy(buf, params, self.pid_start, self.H_in, self.H_out, BS, TMS,
                                 slab_bytes, bwd, w16)
            _WS_SLAB_OWNER[device] = tag
            self._slab_valid = True
        return buf

    def ensure_ready(self, params: torch.Tensor, B: int, need_stash: bool) -> bool:
        """Allocate the deferred param-flow buffers up front when ``need_stash``:
        the accumulator ``S`` (``H_out * H_in`` floats) and the factor stash
        (``(H_out + H_in) * chunk_slots * B`` floats). Returns ``False`` after
        disabling the group if they do not fit, so that the sparse pass can
        never turn a run that used to fit into an OOM crash; the blocks then
        stay on the gather kernels for the rest of the session.
        """
        try:
            if need_stash:
                self._ensure_stash(B, params.device)
                if self._s is None:
                    self._s = torch.zeros(self.H_out, self.H_in, dtype=torch.float32, device=params.device)
        except torch.cuda.OutOfMemoryError:
            self.eligible = False
            self._gt = self._xt = self._s = None
            extra = (self.H_out * self.H_in
                     + (self.H_out + self.H_in) * self._stash_slots(B) * B) * 4 / 2 ** 30
            warnings.warn(
                "Not enough GPU memory for the deferred parameter flows of the "
                f"sparse-IO pass (needs up to ~{extra:.1f} GiB beyond the circuit: "
                "one fp32 accumulator the size of the transition matrix plus the "
                "factor stash); falling back to the per-sample gather kernels, "
                "which are several times slower at this batch size. Lower the "
                "batch size or PYJUICE_SPARSE_IO_WS_STASH_MIB.",
                RuntimeWarning,
            )
            return False
        return True

    def _stash_slots(self, B: int) -> int:
        per_slot = (self.H_out + self.H_in) * B * 4
        return (self.num_slots if _WS_STASH_BYTES <= 0 else
                min(self.num_slots, max(1, _WS_STASH_BYTES // per_slot)))

    def _ensure_stash(self, B: int, device: torch.device) -> None:
        capacity = self._stash_slots(B)
        if (self._gt is None or self._B != B or self._gt.device != device
                or self._capacity != capacity):
            self._gt = self._xt = None
            self._gt = torch.empty(self.H_out, capacity * B, dtype=torch.float32, device=device)
            self._xt = torch.empty(self.H_in, capacity * B, dtype=torch.float32, device=device)
            self._B, self._capacity = B, capacity
            self._used = 0

    def stash_slot(self, slot: int, B: int, device: torch.device,
                   negate: bool, params: torch.Tensor,
                   param_flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Reserve a zeroed physical slot, flushing a full chunk first.

        Slots are assigned in traversal order. A gather-path member does not
        consume one; partial chunks and abandoned passes never read stale data.
        """
        assert 0 <= slot < self.num_slots
        self._ensure_stash(B, device)
        if self._used == self._capacity:
            self.flush(params, param_flows)
        assert self._negate is None or self._negate == negate, (
            "negate_pflows changed within one backward chunk."
        )
        self._negate = negate
        col0 = self._used * B
        self._gt[:, col0:col0 + B].zero_()
        self._xt[:, col0:col0 + B].zero_()
        self._used += 1
        return self._gt, self._xt, col0

    def flush(self, params: torch.Tensor, param_flows: torch.Tensor, final: bool = False) -> None:
        """Accumulate the stashed steps into ``S``: the stored packed steps
        on the accumulator tiles (:meth:`tile_flush`) and this chunk's
        ``Gt @ Xt.T`` by cuBLAS (``beta = 1``). At the pass end (``final``)
        ``param_flows += W * S`` by :func:`_ws_pflow_epilogue_kernel`, which
        also zeroes ``S`` for the next pass. The last chunk can be smaller
        than capacity; cuBLAS receives its true reduction extent."""
        self.tile_flush()
        if self._used:
            columns = self._used * self._B
            prev = torch.backends.cuda.matmul.allow_tf32
            # Gt / Xt were rounded to nearest tf32 at densify time, so cuBLAS's
            # tf32 truncation is exact on them.
            torch.backends.cuda.matmul.allow_tf32 = _WS_PRECISION == "tf32"
            try:
                self._s.addmm_(self._gt[:, :columns], self._xt[:, :columns].t(), beta=1.0,
                               alpha=-1.0 if self._negate else 1.0)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
            self._s_dirty = True
            self._used = 0
            self._negate = None
        if final and self._s_dirty:
            TMe = min(64, self.BS)
            TKe = min(64, self.CBS)
            _ws_pflow_epilogue_kernel[(self.H_out // TMe, self.H_in // TKe)](
                param_flows, params, self._s, self.pid_start, self.pfid_start,
                self.H_in, NB_ch=self.NB_ch, BS=self.BS, CBS=self.CBS, TM=TMe, TK=TKe,
            )
            self._s_dirty = False


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
    BLOCK_B: tl.constexpr,
):
    """Batched forward for one SparseIOSumLayer block. Grid =
    ``(cdiv(K_out, TILE_M), cdiv(B, BLOCK_B))``. Per-program (K_out tile,
    batch lanes) math:

      v_k = exp(log_in_values[b, k] - max_val[b])              [per K_in chunk]
      h_m   = out_indices[out_col_starts[b] + m]               [TILE_M gather]
      pblock_m, within_m = h_m // BS, h_m % BS
      c_k = in_indices[in_col_starts[b] + k]
      cblock_k, cslot_k  = c_k // CBS, c_k % CBS
      W[m, k] = mparams[pid_start + (pblock_m*NB_ch + cblock_k)*CBS*BS
                                   + cslot_k*BS + within_m]
      acc_sum[m] += Σ_{k ∈ chunk} W[m, k] · v_k                [TILE_M]
      values_out[b, m] = log(acc_sum[m]) + max_val[b]

    Each batch lane reads its own in/out columns (``k < in_nnz[b]``,
    ``m < out_nnz[b]``); ``K_in`` / ``K_out`` are only upper bounds (the
    per-batch max, or the column-size bound of a CUDA-graph forward) — the
    grid covers ``K_out`` and the loop runs to the lanes' own ``in_nnz`` —
    per-sample ``max_val[b]`` is clamped to ``-1e30``, and lanes with
    ``in_nnz[b] == 0`` store exact ``-inf``. The single-sample case has its
    own kernel, :func:`_sparse_io_sum_forward_single_kernel`.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)              # [TILE_M]

    pid_b = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)            # [BLOCK_B]
    mask_b = offs_b < batch_size
    cs_in_b = tl.load(in_col_starts_ptr + offs_b, mask=mask_b, other=0)
    k_in_b = tl.load(in_nnz_ptr + offs_b, mask=mask_b, other=0)
    cs_out_b = tl.load(out_col_starts_ptr + offs_b, mask=mask_b, other=0)
    k_out_b = tl.load(out_nnz_ptr + offs_b, mask=mask_b, other=0)
    # Tile past every lane's output column (the grid only bounds them):
    # nothing to write. An early return, not a zero trip count, which
    # measurably slows the loop below.
    if pid_m * TILE_M >= tl.max(k_out_b, axis=0):
        return

    mask_mb = mask_b[None, :] & (offs_m[:, None] < k_out_b[None, :])
    h_m = tl.load(
        out_indices_ptr + cs_out_b[None, :] + offs_m[:, None],
        mask=mask_mb, other=0,
    )                                                           # [TILE_M, BLOCK_B]
    pblock_m = h_m // BS
    within_m = h_m % BS

    max_b = tl.load(max_val_ptr + offs_b, mask=mask_b, other=0.0)
    max_b = tl.maximum(max_b, -1e30)                            # empty col guard

    # Loop bound from the device sizes, not the host ``K_in``, which only
    # bounds the batch (the CUDA-graph forward passes the column-size
    # bound): masked iterations add exact zeros, so this only skips work.
    acc_sum = tl.zeros([TILE_M, BLOCK_B], dtype=tl.float32)
    for k0 in tl.range(0, tl.max(k_in_b, axis=0), BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                     # [BLOCK_K]
        mask_kb = mask_b[:, None] & (offs_k[None, :] < k_in_b[:, None])

        child_ids = tl.load(
            in_indices_ptr + cs_in_b[:, None] + offs_k[None, :],
            mask=mask_kb, other=0,
        )                                                       # [BLOCK_B, BLOCK_K]
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
        )                                                       # [TILE_M, BLOCK_B, BLOCK_K]
        combined_mask = mask_mb[:, :, None] & mask_kb[None, :, :]
        W = tl.load(mparams_ptr + W_ptr_off, mask=combined_mask,
                    other=0.0).to(tl.float32)

        acc_sum += tl.sum(W * v_k[None, :, :], axis=2)          # [TILE_M, BLOCK_B]

    result = tl.log(acc_sum) + max_b[None, :]
    result = tl.where(k_in_b[None, :] == 0, -float("inf"), result)

    tl.store(
        values_out_ptr + offs_b[None, :] * out_stride + offs_m[:, None],
        result, mask=mask_mb,
    )


@triton.jit
def _single_forward_splits_dev(k_in, k_out, n_prog, TILE_M: tl.constexpr,
                               BLOCK_K: tl.constexpr):
    """Device twin of :func:`_single_forward_splits`: ``(parent tiles, splits,
    child tiles per split)`` of the single-sample forward. As many splits as
    fit ``n_prog`` programs next to the parent tiles, never more than there
    are child tiles, each split a whole number of ``BLOCK_K`` chunks."""
    n_m = tl.cdiv(k_out, TILE_M)
    k_tiles = tl.cdiv(k_in, BLOCK_K)
    n_s = tl.maximum(tl.minimum(k_tiles, n_prog // tl.maximum(n_m, 1)), 1)
    ct = tl.cdiv(k_tiles, n_s)
    n_s = tl.cdiv(k_tiles, tl.maximum(ct, 1))
    return n_m, n_s, ct


@triton.jit(
    do_not_specialize=["pid_start", "K_in", "K_out", "n_prog", "dense_start", "n_dense",
                       "sparse_start", "n_sparse", "num_rows"],
    do_not_specialize_on_alignment=[
        "values_out_ptr", "partials_ptr",
        "in_indices_ptr", "in_values_ptr", "max_val_ptr", "out_indices_ptr",
        "in_col_starts_ptr", "in_nnz_ptr", "out_col_starts_ptr", "out_nnz_ptr",
        "dense_tiles_ptr", "sparse_rows_ptr", "desc_ptr",
    ],
)
def _sparse_io_sum_forward_single_kernel(
    values_out_ptr, partials_ptr,
    mparams_ptr,
    in_indices_ptr, in_values_ptr, max_val_ptr,
    out_indices_ptr,
    in_col_starts_ptr, in_nnz_ptr,
    out_col_starts_ptr, out_nnz_ptr,
    dense_tiles_ptr, sparse_rows_ptr, desc_ptr,
    dense_start, n_dense, sparse_start, n_sparse, num_rows,
    pid_start, K_in, K_out, n_prog,
    NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
    DEVICE_SIZES: tl.constexpr, DIRECT: tl.constexpr, HYBRID: tl.constexpr,
    ALIGNED: tl.constexpr, ROWS_MASK: tl.constexpr,
):
    """Single-sample forward of one SparseIOSumLayer block, grid ``(n_prog,)``.
    The work items are the (parent tile, child range) pairs of a split-K
    decomposition sized on the device (:func:`_single_forward_splits_dev`):
    ``cdiv(k_out, TILE_M)`` parent tiles times as many child ranges as fit
    ``n_prog`` programs, so that the gathers of many items are in flight on
    every SM (see ``_FWD_SINGLE_PROGRAMS_PER_SM``). Program ``p`` handles
    items ``p, p + n_prog, ...``; per item (``s`` its child range):

      v_k = exp(log_in_values[k] - max_val)                [per BLOCK_K chunk of the range]
      h_m   = out_indices[m]                               [TILE_M gather]
      pblock_m, within_m = h_m // BS, h_m % BS
      cblock_k, cslot_k  = in_indices[k] // CBS, in_indices[k] % CBS
      W[m, k] = mparams[pid_start + (pblock_m*NB_ch + cblock_k)*CBS*BS
                                   + cslot_k*BS + within_m]
      partial[m] = Σ_{k ∈ range} W[m, k] · v_k
      DIRECT:  values_out[m] = log(partial[m]) + max_val   (one range per parent tile)
      else:    partials[s * n_m * TILE_M + m] = partial[m] (summed by the epilogue)

    ``DEVICE_SIZES``: the static B=1 containers of a CUDA-graph forward, whose
    column offsets / sizes live on the device; ``K_in`` / ``K_out`` then only
    bound the grid and the split adapts to the column lengths at replay.

    ``HYBRID`` (``SparseCategorical.b1_descriptors``): the parent tiles are the
    output column's ``n_dense`` dense tiles (``dense_tiles[dense_start + d]``:
    TILE_M consecutive parents, streamed as contiguous runs and masked after
    the load) followed by ``cdiv(n_sparse, TILE_M)`` tiles of its gathered
    parents (``sparse_rows[sparse_start + r]``); the split rule runs on
    ``n_dense * TILE_M + n_sparse``. Every item writes its partial sums to the
    slot ``m_tile * TILE_M + lane``; the epilogue maps every output position
    to its slot through ``slot[col_start + p]``. Never ``DIRECT``. In
    ``DEVICE_SIZES`` mode the four counts come from ``desc_ptr``.
    """
    pid = tl.program_id(0)
    if DEVICE_SIZES:
        k_in = tl.load(in_nnz_ptr)
        k_out = tl.load(out_nnz_ptr)
        in_idx_ptr = in_indices_ptr + tl.load(in_col_starts_ptr)
        out_idx_ptr = out_indices_ptr + tl.load(out_col_starts_ptr)
    else:
        k_in = K_in
        k_out = K_out
        in_idx_ptr = in_indices_ptr
        out_idx_ptr = out_indices_ptr
    if HYBRID:
        if DEVICE_SIZES:
            d_start = tl.load(desc_ptr + 0)
            nd = tl.load(desc_ptr + 1)
            s_start = tl.load(desc_ptr + 2)
            ns = tl.load(desc_ptr + 3)
        else:
            d_start = dense_start
            nd = n_dense
            s_start = sparse_start
            ns = n_sparse
        k_out_eff = nd * TILE_M + ns
    else:
        d_start = 0
        nd = 0
        s_start = 0
        ns = 0
        k_out_eff = k_out
    n_m, n_s, ct = _single_forward_splits_dev(k_in, k_out_eff, n_prog, TILE_M, BLOCK_K)
    max_val = tl.load(max_val_ptr)                              # scalar
    max_val = tl.where(max_val == -float("inf"), 0.0, max_val)

    n_items = tl.cdiv(n_m * n_s - pid, n_prog)
    for i in range(0, n_items):
        w = pid + i * n_prog
        m_tile = w // n_s
        s = w - m_tile * n_s
        k_lo = s * (ct * BLOCK_K)
        k_hi = tl.minimum(k_lo + ct * BLOCK_K, k_in)
        if HYBRID:
            if m_tile < nd:
                # dense tile: TILE_M consecutive parents, contiguous W rows
                t0 = tl.multiple_of(tl.load(dense_tiles_ptr + d_start + m_tile), TILE_M)
                if ALIGNED:
                    partial = _single_item_accumulate_dense(
                        mparams_ptr, in_idx_ptr, in_values_ptr, t0, k_lo, k_hi, max_val,
                        pid_start, num_rows, NB_ch, BS, CBS, TILE_M, BLOCK_K, ROWS_MASK)
                else:
                    h_m = t0 + tl.arange(0, TILE_M)
                    mask_m = h_m < num_rows
                    partial = _single_item_accumulate(
                        mparams_ptr, in_idx_ptr, in_values_ptr, h_m, mask_m, k_lo, k_hi, max_val,
                        pid_start, NB_ch, BS, CBS, TILE_M, BLOCK_K)
                tl.store(partials_ptr + s * (n_m * TILE_M) + m_tile * TILE_M + tl.arange(0, TILE_M), partial)
            else:
                offs_r = (m_tile - nd) * TILE_M + tl.arange(0, TILE_M)
                mask_m = offs_r < ns
                h_m = tl.load(sparse_rows_ptr + s_start + offs_r, mask=mask_m, other=0)
                partial = _single_item_accumulate(
                    mparams_ptr, in_idx_ptr, in_values_ptr, h_m, mask_m, k_lo, k_hi, max_val,
                    pid_start, NB_ch, BS, CBS, TILE_M, BLOCK_K)
                tl.store(partials_ptr + s * (n_m * TILE_M) + m_tile * TILE_M + tl.arange(0, TILE_M),
                         partial, mask=mask_m)
        else:
            # the original gather path, kept inline so the compiled kernel is unchanged
            offs_m = m_tile * TILE_M + tl.arange(0, TILE_M)         # [TILE_M]
            mask_m = offs_m < k_out
            h_m = tl.load(out_idx_ptr + offs_m, mask=mask_m, other=0)
            pblock_m = h_m // BS
            within_m = h_m % BS

            acc = tl.zeros([TILE_M, BLOCK_K], dtype=tl.float32)
            for k0 in range(k_lo, k_hi, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)                 # [BLOCK_K]
                mask_k = offs_k < k_hi
                child_ids = tl.load(in_idx_ptr + offs_k, mask=mask_k, other=0)
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
                W = tl.load(mparams_ptr + W_ptr_off, mask=mask_m[:, None] & mask_k[None, :],
                            other=0.0).to(tl.float32)
                acc += W * v_k[None, :]
            partial = tl.sum(acc, axis=1)                           # [TILE_M]
            if DIRECT:
                tl.store(values_out_ptr + offs_m, tl.log(partial) + max_val, mask=mask_m)
            else:
                tl.store(partials_ptr + s * (n_m * TILE_M) + offs_m, partial, mask=mask_m)


@triton.jit
def _single_item_accumulate_dense(mparams_ptr, in_idx_ptr, in_values_ptr, t0, k_lo, k_hi, max_val,
                                  pid_start, num_rows, NB_ch: tl.constexpr, BS: tl.constexpr,
                                  CBS: tl.constexpr, TILE_M: tl.constexpr, BLOCK_K: tl.constexpr,
                                  ROWS_MASK: tl.constexpr):
    """:func:`_single_item_accumulate` for a dense tile of TILE_M consecutive
    parents inside one parent block (``BS % TILE_M == 0``): one parent block
    for the whole tile, one column base per child, one lane offset per parent,
    so the loads are contiguous vectors and the address arithmetic is one add
    per element. ``ROWS_MASK``: the last tile may run past ``num_rows``."""
    pblock = t0 // BS
    within = t0 % BS + tl.arange(0, TILE_M)                 # [TILE_M], contiguous
    row_base = mparams_ptr + pid_start + pblock.to(tl.int64) * (NB_ch * CBS * BS)
    acc = tl.zeros([TILE_M, BLOCK_K], dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                 # [BLOCK_K]
        mask_k = offs_k < k_hi
        child_ids = tl.load(in_idx_ptr + offs_k, mask=mask_k, other=0)
        log_vals = tl.load(in_values_ptr + offs_k, mask=mask_k, other=-float("inf"))
        v_k = tl.where(mask_k, tl.exp(log_vals - max_val), 0.0)
        col = (child_ids // CBS) * (CBS * BS) + (child_ids % CBS) * BS     # [BLOCK_K]
        if ROWS_MASK:
            mask = mask_k[None, :] & ((t0 + tl.arange(0, TILE_M)) < num_rows)[:, None]
        else:
            mask = mask_k[None, :] & (tl.arange(0, TILE_M) >= 0)[:, None]
        W = tl.load(row_base + col[None, :] + within[:, None], mask=mask, other=0.0).to(tl.float32)
        acc += W * v_k[None, :]
    return tl.sum(acc, axis=1)                              # [TILE_M]


@triton.jit
def _single_item_accumulate(mparams_ptr, in_idx_ptr, in_values_ptr, h_m, mask_m, k_lo, k_hi, max_val,
                            pid_start, NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                            TILE_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """``partial[m] = sum_{k in [k_lo, k_hi)} W[h_m[m], child_k] * exp(log_in[k] - max_val)``
    for one (parent tile, child range) work item of the single-sample forward."""
    pblock_m = h_m // BS
    within_m = h_m % BS
    acc = tl.zeros([TILE_M, BLOCK_K], dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                 # [BLOCK_K]
        mask_k = offs_k < k_hi
        child_ids = tl.load(in_idx_ptr + offs_k, mask=mask_k, other=0)
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
        W = tl.load(mparams_ptr + W_ptr_off, mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0).to(tl.float32)
        acc += W * v_k[None, :]
    return tl.sum(acc, axis=1)                              # [TILE_M]


@triton.jit(
    do_not_specialize=["K_in", "K_out", "n_prog", "col_start", "n_dense", "n_sparse"],
    do_not_specialize_on_alignment=["values_out_ptr", "partials_ptr", "max_val_ptr",
                                    "in_nnz_ptr", "out_nnz_ptr", "out_col_starts_ptr",
                                    "slot_ptr", "desc_ptr"],
)
def _sparse_io_sum_forward_single_epilogue_kernel(
    values_out_ptr, partials_ptr, max_val_ptr, in_nnz_ptr, out_nnz_ptr,
    out_col_starts_ptr, slot_ptr, desc_ptr, col_start, n_dense, n_sparse,
    K_in, K_out, n_prog,
    TILE_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK: tl.constexpr,
    DEVICE_SIZES: tl.constexpr, HYBRID: tl.constexpr,
):
    """``values_out[m] = log(Σ_s partials[s * n_m * TILE_M + slot(m)]) + max_val``
    over the splits of :func:`_sparse_io_sum_forward_single_kernel` (same
    rule, same sizes), ``-inf`` for an empty input column; ``slot(m) = m``, or
    ``slot[col_start + m]`` of the descriptors when ``HYBRID``. Grid
    ``(cdiv(K_out, BLOCK),)``."""
    if DEVICE_SIZES:
        k_in = tl.load(in_nnz_ptr)
        k_out = tl.load(out_nnz_ptr)
    else:
        k_in = K_in
        k_out = K_out
    if HYBRID:
        if DEVICE_SIZES:
            k_out_eff = tl.load(desc_ptr + 1) * TILE_M + tl.load(desc_ptr + 3)
            c0 = tl.load(out_col_starts_ptr)
        else:
            k_out_eff = n_dense * TILE_M + n_sparse
            c0 = col_start
    else:
        k_out_eff = k_out
        c0 = 0
    n_m, n_s, _ = _single_forward_splits_dev(k_in, k_out_eff, n_prog, TILE_M, BLOCK_K)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < k_out
    if HYBRID:
        slots = tl.load(slot_ptr + c0 + offs, mask=mask, other=0)
    else:
        slots = offs
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in range(0, n_s):
        acc += tl.load(partials_ptr + s * (n_m * TILE_M) + slots, mask=mask, other=0.0)
    max_val = tl.load(max_val_ptr)
    max_val = tl.where(max_val == -float("inf"), 0.0, max_val)
    res = tl.where(k_in == 0, -float("inf"), tl.log(acc) + max_val)
    tl.store(values_out_ptr + offs, res, mask=mask)


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
                chunk = tl.where(mask_jb & (nmars > -float("inf")) & (log_val[None, :] > -float("inf")) & (nflows != 0.0), chunk_full, 0.0)
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
            chunk = tl.where(mask_j & (nmars > -float("inf")) & (log_val > -float("inf")) & (nflows != 0.0), chunk_full, 0.0)
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
# Deferred parameter flows ("gemm"): Triton kernels
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


@triton.jit(do_not_specialize=["v_stride", "dst_ld", "dst_off"])
def _ws_densify_x_kernel(dst_ptr, idx_ptr, val_ptr, max_ptr, cs_ptr, nnz_ptr,
                         v_stride, dst_ld, dst_off,
                         BLOCK: tl.constexpr, ROUND: tl.constexpr):
    """Scatter ``exp(values[b, k] - max_b)`` of a batched sparse column into
    the transposed stash slot ``dst[dst_off + b + row * dst_ld]`` (a pre-zeroed
    ``[H, N]`` slot of the factor stash). Grid ``(cdiv(max nnz, BLOCK), B)``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    row = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    v = tl.load(val_ptr + b * v_stride + k, mask=mask, other=-float("inf"))
    mx = tl.maximum(tl.load(max_ptr + b), -1e30)
    x = tl.exp(v - mx)
    if ROUND:
        x = _ws_round_tf32(x)
    tl.store(dst_ptr + dst_off + b + row.to(tl.int64) * dst_ld, x, mask=mask)


@triton.jit(do_not_specialize=["flow_stride", "mars_stride", "gt_ld", "gt_off"])
def _ws_densify_g_kernel(gt_ptr, idx_ptr, nflow_ptr, nmars_ptr, max_ptr, kin_ptr,
                         cs_ptr, nnz_ptr, flow_stride, mars_stride, gt_ld, gt_off,
                         BLOCK: tl.constexpr, ROUND: tl.constexpr):
    """Scatter ``gs = nflow * exp(max_b - nmars)`` (0 for samples without
    active inputs) of the batched output column into the transposed stash
    slot ``gt[h, gt_off + b]``. ``max_b`` is the input max the forward used,
    so ``gs * x = nflow * exp(in - nmars)`` exactly as the gather kernel's
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
    tl.store(gt_ptr + gt_off + h * gt_ld + b, g, mask=mask)


@triton.jit(do_not_specialize=["pid_start", "pfid_start"])
def _ws_pflow_epilogue_kernel(pf_ptr, w_ptr, s_ptr, pid_start, pfid_start, H_in,
                              NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                              TM: tl.constexpr, TK: tl.constexpr):
    """``param_flows[(h, c)] += W[(h, c)] * S[h, c]`` for the deferred
    accumulator ``S [H_out, H_in]`` (same ``(pb*NB_ch + cb)*CBS*BS + cs*BS +
    w`` layout for params and param flows), then ``S = 0`` for the next
    pass. Grid ``(H_out // TM, H_in // TK)``."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    h0 = pid_m * TM
    c0 = pid_c * TK
    pb = h0 // BS
    w0 = h0 % BS
    cb = c0 // CBS
    cs0 = c0 % CBS
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, TK)
    s_off = (h0 + offs_m).to(tl.int64)[:, None] * H_in + (c0 + offs_k)[None, :]
    s = tl.load(s_ptr + s_off)
    tl.store(s_ptr + s_off, tl.zeros_like(s))
    off = ((pb.to(tl.int64) * NB_ch + cb) * (CBS * BS)
           + (cs0 + offs_k)[None, :] * BS + (w0 + offs_m)[:, None])
    upd = tl.load(w_ptr + pid_start + off) * s
    pf = tl.load(pf_ptr + pfid_start + off)
    tl.store(pf_ptr + pfid_start + off, pf + upd)
