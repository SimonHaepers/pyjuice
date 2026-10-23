from __future__ import annotations


import math
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
from .sparse_prod_layer import SparseProdLayer


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
# Weight-stationary (WS) batched path
# --------------------------------------------------------------------------- #
#
# The per-sample gather kernels above read ``W[h, c]`` once per (sample, active
# pair), each from its own 32B sector. At training batch sizes the union of the
# samples' supports covers essentially every tile of W at every step, so they
# move ~B * K_in * K_out sectors (~2 GB per step at H=16k, B=64) where one
# streaming pass over W is ~1 GB. The WS path instead loads each W tile into
# shared memory once per step and applies it to the whole batch tile:
# programs own a block of rows (forward: parents, backward: children), walk the
# reduction axis block by block, and multiply against the batch's activations
# densified for that block (zeros outside each sample's support) on the tensor
# cores. Cost is one pass over W per step, independent of the support size —
# or one pass over the rows of W^T of the batch's active children only (their
# sorted union, compacted on the device each step), which is what the batch
# can share, when that union is small (:func:`_ws_dispatch`). At training
# batch sizes, where the full pass is tensor-core bound, the forward can
# instead walk every parent tile with only the samples that have an active
# parent in it (the "tiles" pass below, :func:`_ws_fwd_tiles_kernel`).
#
# Parameter flows of tied transitions (the homogeneous HMM chain) are deferred:
# each step stashes its densified factors and bounded chunks feed GEMMs,
# ``pflow += W * sum_t G_t^T X_t``, instead of B * K_in * K_out atomics per step.
#
# This is ON BY DEFAULT (``"auto"``): every batched block takes the cheapest of
# gather / full W pass / active-children pass / tiles pass by the cost model of
# :func:`_ws_dispatch`, calibrated on the trained wikitext chains; B=1
# inference keeps the gather kernels, which are faster there. It costs one fp32
# copy of W plus the stash, and :meth:`_WSGroup.ensure_ready` falls back to the
# gather kernels rather than OOM when those do not fit. Results differ from the
# gather path only by tf32 rounding (~1e-3 nats per sequence on a T=128 chain);
# ``_WS_PRECISION="ieee"`` is bit-comparable but gives up most of the speedup.

_WS_MODE = os.environ.get("PYJUICE_SPARSE_IO_WS", "auto")
"""``"auto"`` (the default) — each batched block takes the cheapest of its
kernel sets by the cost model of :func:`_ws_dispatch` (gather / full W pass /
active-children pass / tiles pass; the block-diagonal layers go dense from
``_WS_MIN_BATCH`` samples up); ``"always"`` — WS whenever eligible (tests /
A-B runs; ``_WS_ACTIVE_MODE`` picks the pass); ``"never"`` — gather kernels
only."""

_WS_ACTIVE_MODE = os.environ.get("PYJUICE_SPARSE_IO_WS_ACTIVE", "auto")
"""Which W pass the WS kernels take: ``"auto"`` — the cost model decides
between the full pass and the pass over the batch's active children;
``"always"`` / ``"never"`` — force one (tests / A-B runs)."""

_WS_MIN_BATCH = int(os.environ.get("PYJUICE_SPARSE_IO_WS_MIN_BATCH", "16"))
"""Smallest batch the block-diagonal (Monarch) layers run their dense batched
path at in ``"auto"`` mode (``_bd_dense_batched`` in
``sparse_io_block_diagonal_sum_layer``); below it their per-sample kernels
win. The sparse-IO layer does not use it any more: its cost model
(:func:`_ws_dispatch`) covers every batch from B=2 up, and on the trained
H=16k chains a WS pass already wins from B=2 (the batched gather kernel is
latency-bound below ~16 samples)."""

_WS_PRECISION = os.environ.get("PYJUICE_SPARSE_IO_WS_PRECISION", "tf32")
"""``"tf32"`` — tensor-core tf32 on operands pre-rounded to nearest (a per-pass
rounded copy of W plus rounding at densify time), so the error is zero-mean
(~1e-4 relative) rather than the ~-6e-4 bias of tf32 truncation; ``"ieee"`` —
exact fp32 FMA, ~3x slower than tf32 but still memory-bound at B=64."""

_WS_MIN_W = 1 << 24
"""Elements of W (``H_out * H_in``) below which a block stays on the gather
kernels at any batch size: a smaller W (64 MB fp32 at 2^24) sits in the L2
cache, where the gather kernels read it as cheaply as a streaming pass. On the
trained H=2048 wikitext chains gather beats every WS variant up to B=64
(forward 1.0 vs 1.7 ms, EM step 15 vs 26 ms at B=64)."""

# Cost model of :func:`_ws_dispatch`, in units of one streamed W element of the
# full WS pass (~2.9 TB/s on an H100 NVL). Calibrated 2026-10-19 on the trained
# wikitext chains (H=8k..32k, emission density 0.01..0.1, T=128): per-sample
# gather kernels vs the two WS passes, CUDA-graph forward and eager EM backward.
_WS_GATHER_PAIR_COST = 6.0
"""Forward cost of one gathered ``(sample, in, out)`` pair from B=16 up (the
gather kernel moves ~1.2e11 pairs/s: one 32B sector per 4B weight)."""
_WS_GATHER_PAIR_COST_SMALL = 14.0
"""The same below B=16, where the batched gather kernel's grid (one program per
32 output rows and sample) leaves most of the SMs idle (~5e10 pairs/s)."""
_WS_GATHER_PAIR_COST_BWD = 15.0
"""Backward with parameter flows: the gather kernel adds one atomic per pair
(~4e10 pairs/s at H=16k, worse when the samples' supports overlap)."""
_WS_ACTIVE_COST = 1.7
"""Forward cost of one element of the active children's rows of W^T relative
to a streamed element of the full pass (gathered rows, index loads), up to
``_WS_ACTIVE_MAX_BATCH`` samples; above that the forward kernel loses its
advantage (2x the full pass per element at B=16, 4.5x at B=64, H=16k)."""
_WS_ACTIVE_MAX_BATCH = 8
"""Largest batch the forward takes the active-children pass at."""
_WS_ACTIVE_COST_BWD = 2.0
"""Backward counterpart of ``_WS_ACTIVE_COST``; the backward kernel keeps its
advantage at every batch size (0.8x the full pass at B=16, 1.0x at B=64)."""
_WS_UNION_EXP = 0.7
"""The expected union of ``B`` samples' input supports is modelled as that of
``B ** _WS_UNION_EXP`` independent columns of the expected length: the
frequent tokens' supports overlap heavily (fits 0.56..0.84 on the chains)."""
_WS_MMA_COST_PER_SAMPLE = 0.012
"""Tensor-core work of a W element per sample relative to streaming it: the
WS passes stop being memory-bound from ``1 / _WS_MMA_COST_PER_SAMPLE`` (~80)
samples on and their cost then grows with B (105 TFLOP/s tf32 measured at
B=128). Without it a 99 %-sparse H=4096 chain went to WS from B~300 and
trained 2-3x slower than on the gather kernels at B=8192 (2026-10-16)."""

_WS_FLUSH_PANEL_ELEMS = 1 << 26
"""Max elements of the dense ``[rows, H_in]`` panel the deferred param-flow
GEMM materialises at a time (256 MB fp32)."""

# Pass over the per-sample active parent tiles ("tiles", :func:`_ws_fwd_tiles_kernel`).
#
# At training batch sizes the full W pass is tensor-core bound (its cost grows
# with B) although a sample's active pairs are 0.2-3 % of the dense
# multiply-adds: with the sorted state order a sample touches 5-20 % of the
# 32x32 tiles of W, but a batch's samples rarely share a bigram, so any block of
# 16+ samples touches nearly every child tile (2026-10-23 analysis,
# prune-pc ``scripts/bench/analyze_tile_groups.py``). What survives batching is
# the parent side: for every parent tile the pass gathers exactly the samples
# that have an active parent in it (lists built on the device each step) and
# walks the children with that block, loading an activation row only for the
# child tiles the sample touches (its child-tile bits) and skipping a child
# tile no sample of the block touches. MMA work ~ B * (parent tiles per sample)
# * H_in, 4-6x below the full pass on the 8k/16k d0.01 chains.
_WS_TILES_MODE = os.environ.get("PYJUICE_SPARSE_IO_WS_TILES", "auto")
"""``"auto"`` — the cost model may pick the tiles passes (forward over the
parent tiles, backward element flows over the child tiles); ``"always"`` —
WS blocks take them whenever eligible (tests / A-B runs; it beats
``_WS_ACTIVE_MODE == "always"``); ``"never"`` — excluded."""
# Tile shape, chosen on the 2026-10-23 sweep (8k d0.01, B=64..2048, 13 shapes,
# prune-pc results/sparse_io_paths/2026-10-23_ws-tiles): the kernel's cost per
# multiply-add falls with TM and BB (load elements per MAC ~ 1/TM + 1/BB)
# while the MMA work grows with TM (parent tiles per sample: 32-row 27 %,
# 64-row 36 %, 128-row 48 % of the tiles on that chain) and with the padding
# of a tile's list to BB. 64 x 64 x 128 ran at ~100 TFLOP/s on 34 % of the
# dense work (72 ms per T=128 batch at B=1024 vs 131 for the full pass and 69
# for cuBLAS dense TF32; 138 vs 261 / 133 at B=2048); 32 x 32 x 64 took 87,
# 128 x 32 x 64 109, and 64 x 64 x 64 (4 warps) 173 ms (register pressure).
# The row mask cost 10 % instead of saving bytes: the activation panel sits in
# L2 at these sizes.
_WS_TILES_TM = int(os.environ.get("PYJUICE_SPARSE_IO_WS_TILES_TM", "64"))
"""Parent tile (rows of W per program; the granularity of the per-sample
parent lists of the forward and of the parent walk of the backward)."""
_WS_TILES_TK = int(os.environ.get("PYJUICE_SPARSE_IO_WS_TILES_TK", "64"))
"""Child tile (the child walk of the forward, the per-sample child lists of
the backward; one activation row chunk of ``TK * 4`` bytes per sample and
tile). 128 needs more shared memory than an H100 has at 3 stages."""
_WS_TILES_BB = int(os.environ.get("PYJUICE_SPARSE_IO_WS_TILES_BB", "128"))
"""Samples per program (the MMA batch tile; >= 16). A tile's list is padded
to a multiple of it."""
_WS_TILES_ROW_MASK = os.environ.get("PYJUICE_SPARSE_IO_WS_TILES_ROW_MASK", "0") != "0"
"""Load a sample's activation row of a child tile only when the sample has
an active child in it (its tile bits, held in registers for the program's
samples); off (the default), every listed sample's row is loaded (zeros
outside its support), which measured 10 % faster at 8k with the panel in
L2. Worth trying when ``B * H * 4`` bytes exceed the L2 by far."""
_WS_TILES_SKIP_K = os.environ.get("PYJUICE_SPARSE_IO_WS_TILES_SKIP_K", "0") != "0"
"""Skip a child tile (its W tile load and the MMA) when no sample of the block
touches it. Off by default: the data-dependent branch stops Triton from
pipelining the loop's loads (8k d0.01: 3.4x slower per iteration, 2026-10-23
kernel map), and with 64-sample blocks a block touches ~95 % of the tiles."""
_WS_TILES_MMA_COST_PER_SAMPLE = 0.02
"""Cost model (:func:`_ws_dispatch`): tensor-core work of one W element of a
visited tile row per (padded) sample of the block, relative to streaming the
element in the full pass — the full pass's ``_WS_MMA_COST_PER_SAMPLE`` with
the gathered rows' lower efficiency and the L2 re-reads of the W tiles by
the blocks of a tile folded in. Calibrated 2026-10-23 on 8k d0.01 with the
default shape: measured tiles / full forward 1.27 / 1.21 / 0.73 / 0.55 /
0.53 at B=64 / 256 / 512 / 1024 / 2048, modelled 2.6 / 1.1 / 0.88 / 0.79 /
0.75 (the decisions agree: full up to 256, tiles from 512; the backward
mirrors them)."""
_WS_TILES_LIST_SPREAD = (0.4, 2.4)
"""The lengths of the per-tile sample lists spread around their mean (a tile
is touched by 10-50 % of the batch on the trained chains when the mean is
20 %); the padding to ``BB`` is averaged over a geometric range of this
spread rather than taken at the mean."""

def _ws_tiles_cost(B: int, n_w: float, f_tiles: float, BB: int) -> float:
    """Cost model of the tiles passes (:func:`_ws_fwd_tiles_kernel`,
    :func:`_ws_bwd_flow_tiles_kernel`) in streamed W elements: the larger of
    one HBM pass over the tiles the batch touches (the pass is memory-bound
    at small batches like the full one) and the tensor-core work of the
    padded tile lists — per tile ``ceil(rows / BB) * BB`` rows of ``H``
    elements, with the list lengths spread around their mean ``f_tiles *
    B`` (:data:`_WS_TILES_LIST_SPREAD`)."""
    if f_tiles <= 0.0:
        return math.inf
    mean = B * f_tiles                                     # expected samples per tile
    lo, hi = _WS_TILES_LIST_SPREAD
    n = 9
    padded = 0.0
    for i in range(n):
        rows = mean * lo * (hi / lo) ** (i / (n - 1))
        padded += math.ceil(rows / BB) * BB
    padded /= n
    u = 1.0 - (1.0 - min(f_tiles, 1.0)) ** (B ** _WS_UNION_EXP)
    return n_w * max(u, padded * _WS_TILES_MMA_COST_PER_SAMPLE)


def _ws_dispatch(B: int, H_in: int, H_out: int, pairs: float, k_in: float,
                 backward: bool, f_tiles: Optional[float] = None,
                 BB: Optional[int] = None) -> Optional[str]:
    """Kernel set of a batched sparse-IO block: ``None`` (per-sample gather
    kernels), ``"full"`` (WS, one pass over W), ``"active"`` (WS, one pass
    over the rows of W^T of the batch's active children, compacted on the
    device each step by :func:`_ws_active_children`) or ``"tiles"`` (WS over
    every sample's active parent tiles in the forward,
    :func:`_ws_fwd_tiles_kernel`, active child tiles in the backward,
    :func:`_ws_bwd_flow_tiles_kernel`, when ``f_tiles``, the expected fraction
    of those tiles a sample touches, is given) — the
    cheapest by the cost model above for ``B`` samples with ``pairs`` active
    ``(sample, in, out)`` pairs per step and ``k_in`` active children per
    sample (the batch's own counts, or the model's expectation for a CUDA
    graph), under the ``_WS_MODE`` / ``_WS_ACTIVE_MODE`` / ``_WS_TILES_MODE``
    overrides."""
    if _WS_MODE == "never":
        return None
    # A W pass streams W (memory-bound) until the tensor-core work of the batch
    # tile takes over; the active-children pass only reads the expected union.
    n_w = float(H_in) * float(H_out)
    W = n_w * max(1.0, _WS_MMA_COST_PER_SAMPLE * B)
    u = 1.0 - (1.0 - min(k_in, H_in) / H_in) ** (B ** _WS_UNION_EXP)
    if backward:
        c_gather = pairs * _WS_GATHER_PAIR_COST_BWD
        c_active = u * W * _WS_ACTIVE_COST_BWD
    else:
        c_gather = pairs * (_WS_GATHER_PAIR_COST if B >= 16 else _WS_GATHER_PAIR_COST_SMALL)
        c_active = u * W * _WS_ACTIVE_COST if B <= _WS_ACTIVE_MAX_BATCH else math.inf
    c_full = W
    tiles_ok = f_tiles is not None and _WS_TILES_MODE != "never"
    c_tiles = _ws_tiles_cost(B, n_w, f_tiles, BB or _WS_TILES_BB) if tiles_ok else math.inf
    if tiles_ok and _WS_TILES_MODE == "always":
        ws, c_ws = "tiles", c_tiles
    elif _WS_ACTIVE_MODE == "always":       # forced, whatever the batch
        ws, c_ws = "active", u * W * (_WS_ACTIVE_COST_BWD if backward else _WS_ACTIVE_COST)
    elif _WS_ACTIVE_MODE == "never" or c_active >= c_full:
        ws, c_ws = "full", c_full
    else:
        ws, c_ws = "active", c_active
    if c_tiles < c_ws:
        ws, c_ws = "tiles", c_tiles
    if _WS_MODE == "always":
        return ws
    if B < 2 or n_w < _WS_MIN_W or c_gather <= c_ws:
        return None
    return ws


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

        # ``(group, slot)`` per block for the weight-stationary batched path,
        # assigned by :func:`build_ws_groups` (TensorCircuit compile time).
        # ``None`` keeps the block on the gather kernels.
        self._ws_slots: List[Optional[Tuple["_WSGroup", int]]] = [None] * len(self.nodes)
        # Set by :meth:`backward` while it dispatches its blocks: the tiles
        # pass of a backward is priced on the child tiles, not the parent ones.
        self._ws_in_backward = False

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

            ws = self._ws_use(blk_idx, block, sv_in, sv_out, params) if sv_in.is_batched else None
            if ws:
                self._ws_forward_block(blk_idx, block, sv_in, sv_out, max_val, params,
                                       active_children=ws == "active", tiles=ws == "tiles")
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

            ws = self._ws_use_static(blk_idx, sparse_prod, ns_idx, params, static_ctx)
            if ws:
                self._ws_forward_block(blk_idx, block, sv_in, sv_out, sv_in.max_val, params,
                                       active_children=ws == "active", tiles=ws == "tiles")
            elif B == 1:
                self._launch_single_forward(blk_idx, block, sv_in, sv_out, sv_in.max_val,
                                            params, static_ctx)
            else:
                TILE_M = 32
                self._launch_batched_forward(block, sv_in, sv_out, sv_in.max_val,
                                             params, B, TILE_M)

    def _ws_use_static(self, blk_idx: int, sparse_prod: SparseProdLayer, ns_idx: int,
                       params: torch.Tensor, static_ctx) -> Optional[str]:
        """:meth:`_ws_use` for the CUDA-graph forward, which has no host-side
        column sizes: the cost model runs on the model's expected supports
        (:meth:`_ws_static_support`). The answer is memoised for consistent
        warmup and capture dispatch."""
        key = (id(self), blk_idx)
        if key in static_ctx.ws_decisions:
            return static_ctx.ws_decisions[key]
        use = None
        ws = self._ws_slots[blk_idx]
        if ws is not None and _WS_MODE != "never" and ws[0].eligible:
            group = ws[0]
            B = static_ctx.B
            avg_in, avg_out = self._ws_static_support(blk_idx, static_ctx)
            use = _ws_dispatch(B, group.H_in, group.H_out, B * avg_in * avg_out, avg_in,
                               backward=False,
                               f_tiles=self._ws_tiles_fraction(blk_idx, static_ctx),
                               BB=self._ws_tiles_shapes(blk_idx, B, params.device)[2])
            if use is not None and not group.ensure_ready(params, B, need_stash=False):
                use = None
        static_ctx.ws_decisions[key] = use
        return use

    def _ws_tiles_fraction(self, blk_idx: int, static_ctx=None,
                           backward: bool = False) -> Optional[float]:
        """Expected fraction of the tiles of the tiles pass a sample touches
        (:meth:`SparseCategorical.expected_col_tiles`): the ``TM``-row parent
        tiles of the output column in the forward, the ``TK``-row child tiles
        of the input column in the backward; every tile for a variable the
        profile masks; ``None`` when the block cannot take the pass."""
        TM, TK = self._ws_tiles_shapes(blk_idx, 1, None)[:2]
        if TM is None:
            return None
        prod, ns_idx = self._sparse_input_refs[blk_idx]
        in_ns = prod.nodes[ns_idx]
        emit_ns = in_ns.sparse_input_ns
        base = emit_ns._param_range[0]
        in_values = prod._sparse_input_layers[ns_idx].params[base:base + emit_ns.dist._nnz]
        if backward:
            if static_ctx is not None and static_ctx.is_masked(in_ns.var_id):
                return 1.0
            return emit_ns.dist.expected_col_tiles(TK, in_values)
        out_var = self._output_sparsity_var_ids[blk_idx]
        if static_ctx is not None and static_ctx.is_masked(out_var):
            return 1.0
        out_dist = self._output_sparsity_dists[blk_idx]
        # weigh the tokens by their emission mass when the output column's
        # distribution is this block's input one (the tied emission of a
        # homogeneous chain), as :meth:`_ws_static_support` does for the input
        # side; the length-weighted estimate is 20-30 % low on the trained
        # wikitext chains (0.17 vs 0.21 at 8k d0.01, 0.32 vs 0.45 at 8k d0.1)
        values = in_values if emit_ns.dist is out_dist else None
        return out_dist.expected_col_tiles(TM, values)

    def _ws_tiles_shapes(self, blk_idx: int, B: int, device) -> tuple:
        """``(TM, TK, BB, num_warps, num_stages)`` of the tiles pass for this
        block: the configured tiles clipped to the PC block sizes (powers of
        two >= 16 that divide them); ``(None, ...)`` when the block is not
        eligible (a block size below 16)."""
        group, _ = self._ws_slots[blk_idx]
        BS, CBS = group.BS, group.CBS
        if BS % 16 or CBS % 16:
            return (None, None, None, None, None)
        TM = min(_WS_TILES_TM, BS)
        while BS % TM:
            TM //= 2
        TK = min(_WS_TILES_TK, CBS)
        while CBS % TK:
            TK //= 2
        BB = max(16, _WS_TILES_BB)
        # 8 warps from a 64 x 64 accumulator up (64 x 64 x 64 on 4 warps ran 2x
        # slower than on 8: register pressure); 3 stages when a stage's two
        # operand tiles reach 32 KB of shared memory
        warps = 8 if BB * max(TM, TK) >= 64 * 64 else 4
        stages = 3 if (TK * TM + BB * max(TM, TK)) * 4 >= 32 * 1024 else 4
        return (TM, TK, BB, warps, stages)

    def _ws_static_pairs(self, blk_idx: int, B: int, static_ctx) -> float:
        """Expected active ``(sample, in, out)`` pairs per step of a batch of
        ``B`` (:meth:`_ws_static_support`)."""
        avg_in, avg_out = self._ws_static_support(blk_idx, static_ctx)
        return B * avg_in * avg_out

    def _ws_static_support(self, blk_idx: int, static_ctx) -> Tuple[float, float]:
        """Expected active children and parents per sample of a batch whose
        column lengths live on the device (CUDA-graph forward, backward after
        a replay), from the model rather than from the tokens of any
        particular batch, so a graph's dispatch does not depend on what it
        was captured on: the expected column length of each side's
        distribution (:meth:`SparseCategorical.expected_col_nnz`, weighted by
        the input side's emission probabilities, which this layer can reach),
        all rows for a variable the profile masks."""
        prod, ns_idx = self._sparse_input_refs[blk_idx]
        in_ns = prod.nodes[ns_idx]
        emit_ns = in_ns.sparse_input_ns
        in_dist = emit_ns.dist
        out_dist = self._output_sparsity_dists[blk_idx]
        out_var = self._output_sparsity_var_ids[blk_idx]
        if static_ctx is not None and static_ctx.is_masked(in_ns.var_id):
            avg_in = in_dist._num_nodes
        else:
            base = emit_ns._param_range[0]
            values = prod._sparse_input_layers[ns_idx].params[base:base + in_dist._nnz]
            avg_in = in_dist.expected_col_nnz(values)
        if static_ctx is not None and static_ctx.is_masked(out_var):
            avg_out = out_dist._num_nodes
        else:
            avg_out = out_dist.expected_col_nnz()
        return avg_in, avg_out

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

            self._ws_in_backward = True
            try:
                ws = (self._ws_use(blk_idx, block, sv_in, sv_out, params, need_stash=compute_pflows,
                                   static_ctx=kwargs.get("static_ctx"))
                      if sv_in.is_batched else None)
            finally:
                self._ws_in_backward = False
            if ws:
                self._ws_backward_block(
                    blk_idx, block, sv_in, sv_out, sv_flow_out, sv_flow_in,
                    params, param_flows if compute_pflows else None, negate_pflows,
                    active_children=ws == "active", tiles=ws == "tiles",
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
    # Weight-stationary batched path
    # ------------------------------------------------------------------ #

    def _ws_use(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                sv_out: SparseNodeValues, params: torch.Tensor,
                need_stash: bool = False, static_ctx=None) -> Optional[str]:
        """Kernel set for this batched block: ``None`` for the gather kernels,
        ``"full"`` / ``"active"`` for the WS kernels and their W pass (see
        ``_WS_MODE``, :func:`_ws_dispatch`).

        ``"auto"`` runs the cost model on the batch's own pairs and support,
        or on the model's expected ones (:meth:`_ws_static_support`) for
        static containers, whose column lengths are on the device; a backward
        that accumulates parameter flows (``need_stash``) uses the backward
        costs. Then :meth:`_WSGroup.ensure_ready` has the last word, so a
        group whose buffers do not fit stays on the gather kernels instead of
        failing the run.
        """
        ws = self._ws_slots[blk_idx]
        if ws is None or _WS_MODE == "never":
            return None
        group = ws[0]
        if not group.eligible:
            return None
        B = sv_in.batch_size
        if sv_in.static or sv_out.static:
            avg_in, avg_out = self._ws_static_support(blk_idx, static_ctx)
            pairs, k_in = B * avg_in * avg_out, avg_in
        else:
            pairs = sum(a * b for a, b in zip(sv_in.nnz_list, sv_out.nnz_list))
            k_in = sum(sv_in.nnz_list) / max(B, 1)    # total_nnz is the longest column
        # the tiles passes' expected tile fraction (parents of the forward,
        # children of the backward) comes from the model: the batch's tile
        # counts live on the device
        f_tiles = self._ws_tiles_fraction(blk_idx, static_ctx, backward=self._ws_in_backward)
        use = _ws_dispatch(B, group.H_in, group.H_out, pairs, k_in, backward=need_stash,
                           f_tiles=f_tiles, BB=self._ws_tiles_shapes(blk_idx, B, params.device)[2])
        if use is not None and not group.ensure_ready(params, B, need_stash):
            use = None
        return use

    def _ws_forward_block(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                          sv_out: SparseNodeValues, max_val: torch.Tensor,
                          params: torch.Tensor, active_children: bool = False,
                          tiles: bool = False) -> None:
        """``sv_out = log(W @ exp(sv_in - max)) + max`` for the whole batch:
        densify the inputs, stream W once through :func:`_ws_fwd_kernel` (all
        of it, or with ``active_children`` only the rows of W^T of the batch's
        active children) — or with ``tiles`` walk every parent tile with the
        samples that have an active parent in it
        (:func:`_ws_fwd_tiles_kernel`) — then gather the active parents."""
        group, _ = self._ws_slots[blk_idx]
        _, _, _, _, _NB, NB_ch, BS, CBS = block
        B = sv_in.batch_size
        device = params.device
        H_in, H_out = group.H_in, group.H_out
        rnd = _WS_PRECISION == "tf32"
        w, w_start = group.weights(params)

        xd = _ws_scratch("x", (B, H_in), device)
        xd.zero_()
        _ws_densify_x_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), B)](
            xd, sv_in.indices, sv_in.values, max_val, sv_in.col_starts, sv_in.nnz,
            sv_in.values.stride(0), H_in, 1, 0,
            BLOCK=_WS_VEC, ROUND=rnd,
        )

        yd = _ws_scratch("y", (B, H_out), device)
        if tiles:
            TM, TK, BB, warps, stages = self._ws_tiles_shapes(blk_idx, B, device)
            assert TM is not None
            lists, counts = _ws_tile_lists(sv_out, H_out, TM, device)
            bits, n_words = _ws_tile_bits(sv_in, H_in, TK, device)
            _ws_fwd_tiles_kernel[(triton.cdiv(B, BB), H_out // TM)](
                yd, w, xd, max_val, sv_in.nnz, lists, counts, bits, w_start, B, H_in, H_out,
                NB_ch=NB_ch, BS=BS, CBS=CBS, TM=TM, TK=TK, BB=BB, NW=n_words,
                PREC="tf32" if rnd else "ieee", ROW_MASK=_WS_TILES_ROW_MASK,
                SKIP_K=_WS_TILES_SKIP_K,
                num_warps=warps, num_stages=stages,
            )
        else:
            TM, BB, warps, stages = _ws_tiles(H_out, B, BS, 128, device)
            TK = min(32, CBS)
            if active_children:
                active, n_active = _ws_active_children(sv_in, H_in, device)
            else:
                active = n_active = sv_in.nnz                  # unused
            _ws_fwd_kernel[(triton.cdiv(B, BB), H_out // TM)](
                yd, w, xd, max_val, sv_in.nnz, active, n_active, w_start, B, H_in, H_out,
                NB_ch=NB_ch, BS=BS, CBS=CBS, TM=TM, TK=TK, BB=BB,
                PREC="tf32" if rnd else "ieee", GATHER_K=active_children,
                num_warps=warps, num_stages=stages,
            )

        _ws_gather_kernel[(triton.cdiv(sv_out.total_nnz, _WS_VEC), B)](
            sv_out.values, yd, sv_out.indices, sv_out.col_starts, sv_out.nnz,
            sv_out.values.stride(0), H_out, BLOCK=_WS_VEC,
        )

    def _ws_backward_block(self, blk_idx: int, block: tuple, sv_in: SparseNodeValues,
                           sv_out: SparseNodeValues, sv_flow_out: SparseNodeValues,
                           sv_flow_in: SparseNodeValues, params: torch.Tensor,
                           param_flows: Optional[torch.Tensor],
                           negate_pflows: bool, active_children: bool = False,
                           tiles: bool = False) -> None:
        """Element flows through :func:`_ws_bwd_flow_kernel` (over all
        children, or with ``active_children`` only the batch's) or, with
        ``tiles``, :func:`_ws_bwd_flow_tiles_kernel` (every child tile with
        the samples that have an active child in it); the parameter flows are
        only *stashed* here (densified ``G`` / ``X`` factors in this block's
        slot) and accumulated by :meth:`_WSGroup.flush` in bounded chunks.

        With ``gs[b, h] = nflow[b, h] * exp(max_b - nmars[b, h])`` and
        ``x[b, c] = exp(in[b, c] - max_b)`` (zero off-support):
          flow_in[b, c] = x[b, c] * sum_h gs[b, h] W[h, c]
          pflow[h, c]  += W[h, c] * sum_b gs[b, h] x[b, c]
        """
        group, slot = self._ws_slots[blk_idx]
        _, _, _, _, _NB, NB_ch, BS, CBS = block
        B = sv_in.batch_size
        device = params.device
        H_in, H_out = group.H_in, group.H_out
        rnd = _WS_PRECISION == "tf32"
        max_val = sv_in.max_val
        assert max_val is not None, (
            "WS backward needs the fused per-sample sv_in.max_val from the "
            "upstream prod layer's re-forward."
        )
        w, w_start = group.weights(params)

        gd = _ws_scratch("g", (B, H_out), device)
        gd.zero_()
        if param_flows is not None:
            gt, xt, col0 = group.stash_slot(slot, B, device, negate_pflows, params, param_flows)
        else:
            gt, xt, col0 = gd, None, 0
        _ws_densify_g_kernel[(triton.cdiv(sv_out.total_nnz, _WS_VEC), B)](
            gd, gt, sv_out.indices, sv_flow_out.values, sv_out.values, max_val,
            sv_in.nnz, sv_out.col_starts, sv_out.nnz,
            sv_flow_out.values.stride(0), sv_out.values.stride(0), H_out,
            gt.stride(0), col0,
            BLOCK=_WS_VEC, ROUND=rnd, WRITE_T=param_flows is not None,
        )
        if param_flows is not None:
            _ws_densify_x_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), B)](
                xt, sv_in.indices, sv_in.values, max_val, sv_in.col_starts, sv_in.nnz,
                sv_in.values.stride(0), 1, xt.stride(0), col0,
                BLOCK=_WS_VEC, ROUND=rnd,
            )

        fd = _ws_scratch("f", (B, H_in), device)
        if tiles:
            TM, TK, BB, warps, stages = self._ws_tiles_shapes(blk_idx, B, device)
            assert TM is not None
            lists, counts = _ws_tile_lists(sv_in, H_in, TK, device)
            bits, n_words = _ws_tile_bits(sv_out, H_out, TM, device)
            _ws_bwd_flow_tiles_kernel[(triton.cdiv(B, BB), H_in // TK)](
                fd, w, gd, lists, counts, bits, w_start, B, H_in, H_out,
                NB_ch=NB_ch, BS=BS, CBS=CBS, TK=TK, TM=TM, BB=BB, NW=n_words,
                PREC="tf32" if rnd else "ieee", ROW_MASK=_WS_TILES_ROW_MASK,
                SKIP_K=_WS_TILES_SKIP_K,
                num_warps=warps, num_stages=stages,
            )
        else:
            TK, BB, warps, stages = _ws_tiles(H_in, B, CBS, 128, device)
            TM = min(32, BS)
            if active_children:
                active, n_active = _ws_active_children(sv_in, H_in, device)
            else:
                active = n_active = sv_in.nnz                  # unused
            _ws_bwd_flow_kernel[(triton.cdiv(B, BB), H_in // TK)](
                fd, w, gd, active, n_active, w_start, B, H_in, H_out,
                NB_ch=NB_ch, BS=BS, CBS=CBS, TK=TK, TM=TM, BB=BB,
                PREC="tf32" if rnd else "ieee", GATHER_K=active_children,
                num_warps=warps, num_stages=stages,
            )

        _ws_gather_flow_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), B)](
            sv_flow_in.values, fd, sv_in.indices, sv_in.values, max_val,
            sv_in.col_starts, sv_in.nnz, sv_in.values.stride(0),
            sv_flow_in.values.stride(0), H_in, BLOCK=_WS_VEC,
        )


# =====================================================================
# Weight-stationary path: shared state
# =====================================================================

_WS_STASH_BYTES = int(os.environ.get("PYJUICE_SPARSE_IO_WS_STASH_MIB", "256")) * 2**20
"""Factor-buffer budget per tied matrix; at least one batch slot is retained.
Set to 0 to retain all slots (reference behavior). Chunking changes only the
order of parameter-flow accumulation, not the batch or the EM update.
"""

_WS_VEC = 256
"""Element tile of the densify / gather kernels."""

_WS_SCRATCH: Dict[Tuple[str, torch.device], torch.Tensor] = {}
_WS_W_OWNER: Dict[torch.device, Tuple["_WSGroup", int]] = {}
_WS_NUM_SMS: Dict[torch.device, int] = {}


def _ws_scratch(name: str, shape: Tuple[int, ...], device: torch.device,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Process-wide grow-only scratch (one dtype per name). Layers run one at
    a time on one stream, so every WS call can reuse the same buffers."""
    n = 1
    for s in shape:
        n *= s
    buf = _WS_SCRATCH.get((name, device))
    if buf is None or buf.numel() < n:
        _WS_SCRATCH.pop((name, device), None)
        buf = torch.empty(n, dtype=dtype, device=device)
        _WS_SCRATCH[(name, device)] = buf
    return buf[:n].view(shape)


def _ws_active_children(sv_in: SparseNodeValues, H_in: int,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(active, count)``: the sorted union of the batch's active input rows,
    its first ``count[0]`` entries (``count`` a one-element device view, read
    by the kernels as their loop bound). Flags, inclusive prefix sum,
    compaction: four launches, all graph-capturable, no host sync."""
    flags = _ws_scratch("u_flags", (H_in,), device, torch.int32)
    pos = _ws_scratch("u_pos", (H_in,), device, torch.int32)
    active = _ws_scratch("u_active", (H_in,), device, torch.int32)
    flags.zero_()
    _ws_union_mark_kernel[(triton.cdiv(sv_in.total_nnz, _WS_VEC), sv_in.batch_size)](
        flags, sv_in.indices, sv_in.col_starts, sv_in.nnz, BLOCK=_WS_VEC,
    )
    torch.cumsum(flags, 0, dtype=torch.int32, out=pos)
    _ws_union_compact_kernel[(triton.cdiv(H_in, _WS_VEC),)](
        active, flags, pos, H_in, BLOCK=_WS_VEC,
    )
    return active, pos[H_in - 1:]


def _ws_tile_lists(sv: SparseNodeValues, H: int, T: int,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(lists, counts)`` of the tiles passes: for every ``T``-row tile ``m``
    of the ``H`` rows of the batched column ``sv`` the samples with an active
    row in it, ``lists[m, 0:counts[m]]`` (int32, batch order, ``lists`` is
    ``[n_tiles, B]``): the parents of the forward, the children of the
    backward. Per-sample flags, then one program per tile scans the batch
    (prefix sum) and compacts; graph-capturable, no host sync."""
    B = sv.batch_size
    n_tiles = H // T
    flags = _ws_scratch("tl_flags", (B, n_tiles), device, torch.int32)
    lists = _ws_scratch("tl_lists", (n_tiles, B), device, torch.int32)
    counts = _ws_scratch("tl_counts", (n_tiles,), device, torch.int32)
    flags.zero_()
    _ws_tile_mark_kernel[(triton.cdiv(sv.total_nnz, _WS_VEC), B)](
        flags, sv.indices, sv.col_starts, sv.nnz, n_tiles, T, BLOCK=_WS_VEC,
    )
    _ws_tile_lists_kernel[(n_tiles,)](
        lists, counts, flags, B, n_tiles, BLOCK_B=max(16, triton.next_power_of_2(B)),
        num_warps=8 if B > 2048 else 4,
    )
    return lists, counts


def _ws_tile_bits(sv: SparseNodeValues, H: int, T: int,
                  device: torch.device) -> Tuple[torch.Tensor, int]:
    """``(bits, n_words)``: per sample the ``T``-row tiles of ``sv`` it has an
    active row in, as bits of ``n_words`` (a power of two) int32 words
    (``bits[b, t // 32] >> (t % 32) & 1``): the children of the forward, the
    parents of the backward. Graph-capturable."""
    B = sv.batch_size
    n_tiles = H // T
    n_words = triton.next_power_of_2(triton.cdiv(n_tiles, 32))   # a tl.arange extent in the kernels
    bits = _ws_scratch("tl_bits", (B, n_words), device, torch.int32)
    bits.zero_()
    _ws_tile_bits_kernel[(triton.cdiv(sv.total_nnz, _WS_VEC), B)](
        bits, sv.indices, sv.col_starts, sv.nnz, n_words, T, BLOCK=_WS_VEC,
    )
    return bits, n_words


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


def _ws_tiles(n_rows: int, B: int, blk: int, t_max: int,
              device: torch.device) -> Tuple[int, int, int, int]:
    """``(row tile, batch tile, num_warps, num_stages)``: largest tiles that
    still give one program per SM. Row tiles divide ``blk`` (a power of two
    >= 16); batch slices of the same row block are launched adjacently so
    they share W through L2. From B=128 on the in-tile MMA starts to bind
    (H=16k, B=256: 128-wide batch tiles 0.90 ms/step vs 1.35 with 64-wide)."""
    n_sm = _num_sms(device)
    BB = 128 if B >= 128 else (64 if B >= 64 else max(16, triton.next_power_of_2(B)))
    T = min(t_max, blk)
    while T > 16 and (n_rows // T) * triton.cdiv(B, BB) < n_sm:
        T //= 2
    while BB > 16 and (n_rows // T) * triton.cdiv(B, BB) < n_sm:
        BB //= 2
    return (T, BB, 8, 3) if BB >= 128 else (T, BB, 4, 4)


class _WSGroup:
    """WS state shared by every :class:`SparseIOSumLayer` block that reads
    the same (tied) transition matrix, i.e. the same ``pid_start`` /
    ``pfid_start`` / shape — all 127 steps of a homogeneous HMM chain:

    * the per-pass tf32-rounded copy of W (``_WS_PRECISION == "tf32"``);
    * the stash of densified backward factors, up to the configured number of
      ``B``-column slots: ``Gt[H_out, slots*B]`` / ``Xt[H_in, slots*B]``.
      Full chunks flush during traversal; the remainder flushes at pass end.

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
        # tl.dot needs every tile dim >= 16; tiles must divide the PC blocks.
        self.eligible = BS % 16 == 0 and CBS % 16 == 0
        self._w_valid = False
        self._gt: Optional[torch.Tensor] = None
        self._xt: Optional[torch.Tensor] = None
        self._B = 0
        self._capacity = 0
        self._used = 0
        self._negate: Optional[bool] = None

    def begin_pass(self) -> None:
        """Params may have changed since the last pass (EM step): rebuild the
        rounded copy on next use. Unflushed factors of an abandoned pass are
        discarded. Reserved slots are zeroed before reuse."""
        self._w_valid = False
        self._used = 0
        self._negate = None

    def weights(self, params: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """``(tensor, offset of W in it)`` for the WS GEMMs. In tf32 mode
        this is a process-wide buffer holding W rounded to nearest tf32,
        rebuilt once per pass (or whenever another group used the buffer)."""
        if _WS_PRECISION != "tf32":
            return params, self.pid_start
        device = params.device
        n = self.H_out * self.H_in
        buf = _ws_scratch("w_rn", (n,), device)
        owner = _WS_W_OWNER.get(device)
        if not (self._w_valid and owner is not None and owner[0] is self
                and owner[1] == buf.data_ptr()):
            BLOCK = 4096
            _ws_round_copy_kernel[(triton.cdiv(n, BLOCK),)](
                buf, params, self.pid_start, n, BLOCK=BLOCK,
            )
            _WS_W_OWNER[device] = (self, buf.data_ptr())
            self._w_valid = True
        return buf, 0

    def ensure_ready(self, params: torch.Tensor, B: int, need_stash: bool) -> bool:
        """Allocate this group's WS buffers up front — the per-pass rounded W
        copy (``H^2`` floats) and, when ``need_stash``, the deferred param-flow
        stash (``(H_out + H_in) * chunk_slots * B`` floats). Returns ``False``
        after disabling the group if they do not fit, so that enabling WS by
        default can never turn a run that used to fit into an OOM crash; the
        blocks then stay on the gather kernels for the rest of the session.
        """
        try:
            if _WS_PRECISION == "tf32":
                _ws_scratch("w_rn", (self.H_out * self.H_in,), params.device)
            if need_stash:
                self._ensure_stash(B, params.device)
        except torch.cuda.OutOfMemoryError:
            self.eligible = False
            self._gt = self._xt = None
            _WS_SCRATCH.pop(("w_rn", params.device), None)
            extra = (self.H_out * self.H_in
                     + (self.H_out + self.H_in) * self._stash_slots(B) * B) * 4 / 2 ** 30
            warnings.warn(
                "Not enough GPU memory for the weight-stationary sparse-IO "
                f"path (needs up to ~{extra:.1f} GiB beyond the circuit: one "
                "fp32 copy of the transition matrix plus the deferred "
                "param-flow stash); falling back to the per-sample gather "
                "kernels, which are several times slower at this batch size. "
                "Lower the batch size, or set PYJUICE_SPARSE_IO_WS_PRECISION="
                "ieee to drop the W copy (exact, but its backward is slower "
                "than the gather path).",
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

    def flush(self, params: torch.Tensor, param_flows: torch.Tensor) -> None:
        """Accumulate this chunk's Gt @ Xt.T into parameter flows.

        Row panels also bound the GEMM output temporary. The last chunk can
        be smaller than capacity; cuBLAS receives its true reduction extent.
        """
        if not self._used:
            return
        columns = self._used * self._B
        device = params.device
        TMe = min(64, self.BS)
        TKe = min(64, self.CBS)
        rows = min(self.H_out, max(TMe, _WS_FLUSH_PANEL_ELEMS // self.H_in // TMe * TMe))
        panel = _ws_scratch("pflow_panel", (rows, self.H_in), device)
        prev = torch.backends.cuda.matmul.allow_tf32
        # Gt / Xt were rounded to nearest tf32 at densify time, so cuBLAS's
        # tf32 truncation is exact on them.
        torch.backends.cuda.matmul.allow_tf32 = _WS_PRECISION == "tf32"
        try:
            for r0 in range(0, self.H_out, rows):
                r = min(rows, self.H_out - r0)
                out = panel[:r]
                torch.mm(self._gt[r0:r0 + r, :columns], self._xt[:, :columns].t(), out=out)
                _ws_pflow_epilogue_kernel[(r // TMe, self.H_in // TKe)](
                    param_flows, params, out, r0, self.pid_start, self.pfid_start,
                    self.H_in, NB_ch=self.NB_ch, BS=self.BS, CBS=self.CBS,
                    TM=TMe, TK=TKe, NEGATE=bool(self._negate),
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        self._used = 0
        self._negate = None


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
# Weight-stationary path: Triton kernels
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


@triton.jit(do_not_specialize=["src_start", "n"])
def _ws_round_copy_kernel(dst_ptr, src_ptr, src_start, n, BLOCK: tl.constexpr):
    """``dst[i] = round_tf32(src[src_start + i])`` — the per-pass W copy."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src_ptr + src_start + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, _ws_round_tf32(x), mask=mask)


@triton.jit(do_not_specialize=["v_stride", "dst_stride_b", "dst_stride_r", "dst_off"])
def _ws_densify_x_kernel(dst_ptr, idx_ptr, val_ptr, max_ptr, cs_ptr, nnz_ptr,
                         v_stride, dst_stride_b, dst_stride_r, dst_off,
                         BLOCK: tl.constexpr, ROUND: tl.constexpr):
    """Scatter ``exp(values[b, k] - max_b)`` of a batched sparse column to
    ``dst[dst_off + b*dst_stride_b + row*dst_stride_r]`` (a pre-zeroed dense
    ``[B, H]`` (strides ``H, 1``) or a transposed stash ``[H, N]`` slot
    (strides ``1, N``)). Grid ``(cdiv(max nnz, BLOCK), B)``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    row = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    v = tl.load(val_ptr + b * v_stride + k, mask=mask, other=-float("inf"))
    mx = tl.maximum(tl.load(max_ptr + b), -1e30)
    x = tl.exp(v - mx)
    if ROUND:
        x = _ws_round_tf32(x)
    tl.store(dst_ptr + dst_off + b * dst_stride_b + row.to(tl.int64) * dst_stride_r, x,
             mask=mask)


@triton.jit(do_not_specialize=["out_stride"])
def _ws_gather_kernel(out_ptr, src_ptr, idx_ptr, cs_ptr, nnz_ptr, out_stride, H,
                      BLOCK: tl.constexpr):
    """``out[b, k] = src[b, indices[col_starts[b] + k]]`` for ``k < nnz[b]``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    row = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    v = tl.load(src_ptr + b * H + row, mask=mask, other=0.0)
    tl.store(out_ptr + b * out_stride + k, v, mask=mask)


@triton.jit
def _ws_union_mark_kernel(flags_ptr, idx_ptr, cs_ptr, nnz_ptr, BLOCK: tl.constexpr):
    """``flags[indices[col_starts[b] + k]] = 1`` for ``k < nnz[b]``. Grid
    ``(cdiv(max nnz, BLOCK), B)``; samples sharing a row store the same 1."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    row = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    tl.store(flags_ptr + row, tl.full([BLOCK], 1, tl.int32), mask=mask)


@triton.jit
def _ws_union_compact_kernel(active_ptr, flags_ptr, pos_ptr, H, BLOCK: tl.constexpr):
    """``active[pos[c] - 1] = c`` for every flagged row ``c``, ``pos`` the
    inclusive prefix sum of the flags: the sorted active rows, ``pos[H - 1]``
    of them. Grid ``(cdiv(H, BLOCK),)``."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = c < H
    f = tl.load(flags_ptr + c, mask=mask, other=0)
    p = tl.load(pos_ptr + c, mask=mask, other=0)
    tl.store(active_ptr + p - 1, c, mask=mask & (f != 0))


@triton.jit(do_not_specialize=["w_start", "B"])
def _ws_fwd_kernel(yd_ptr, w_ptr, xd_ptr, max_ptr, kin_ptr, act_ptr, nact_ptr, w_start, B,
                   H_in, H_out,
                   NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                   TM: tl.constexpr, TK: tl.constexpr, BB: tl.constexpr,
                   PREC: tl.constexpr, GATHER_K: tl.constexpr):
    """Weight-stationary forward of one SparseIOSumLayer block. Grid
    ``(cdiv(B, BB), H_out // TM)``: program = (batch slice, block of TM
    parents). It walks the children — the softmax / reduction axis — TK at a
    time; every W tile is loaded once and applied to all BB samples:

        acc[b, m]      = sum_c xd[b, c] * W[h0 + m, c]
        yd[b, h0 + m]  = log(acc[b, m] + 1e-24) + max_b   (-inf if nnz_in[b] == 0)

    W[h, c] sits at ``w_start + (pb*NB_ch + cb)*CBS*BS + cs*BS + w`` with
    ``h = pb*BS + w`` and ``c = cb*CBS + cs`` (parents contiguous), so the
    loaded ``[TK, TM]`` tile is the dot's ``[K, N]`` operand as is. With
    ``GATHER_K`` the children walked are the batch's active ones only,
    ``act[0:nact]`` (sorted, :func:`_ws_active_children`), each a contiguous
    run of TM parents at ``w_rows + c*BS``: W traffic ``nact * H_out``.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    h0 = pid_m * TM
    pb = h0 // BS
    w0 = h0 % BS
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, TK)
    offs_b = pid_b * BB + tl.arange(0, BB)
    mask_b = offs_b < B

    w_rows = w_ptr + w_start + pb.to(tl.int64) * (NB_ch * CBS * BS) + w0
    acc = tl.zeros([BB, TM], dtype=tl.float32)
    if GATHER_K:
        n_act = tl.load(nact_ptr)
        for k0 in range(0, n_act, TK):
            offs = k0 + offs_k
            mask_k = offs < n_act
            c = tl.load(act_ptr + offs, mask=mask_k, other=0)
            wt = tl.load(w_rows + c.to(tl.int64)[:, None] * BS + offs_m[None, :],
                         mask=mask_k[:, None], other=0.0)
            x = tl.load(xd_ptr + offs_b[:, None] * H_in + c[None, :],
                        mask=mask_b[:, None] & mask_k[None, :], other=0.0)
            acc += tl.dot(x, wt, input_precision=PREC)
    else:
        for k0 in range(0, H_in, TK):
            cb = k0 // CBS
            cs0 = k0 % CBS
            wt = tl.load(w_rows + cb * (CBS * BS) + (cs0 + offs_k)[:, None] * BS + offs_m[None, :])
            x = tl.load(xd_ptr + offs_b[:, None] * H_in + (k0 + offs_k)[None, :],
                        mask=mask_b[:, None], other=0.0)
            acc += tl.dot(x, wt, input_precision=PREC)

    mx = tl.maximum(tl.load(max_ptr + offs_b, mask=mask_b, other=0.0), -1e30)
    kin = tl.load(kin_ptr + offs_b, mask=mask_b, other=0)
    res = tl.log(acc) + mx[:, None]
    res = tl.where(kin[:, None] == 0, -float("inf"), res)
    tl.store(yd_ptr + offs_b[:, None] * H_out + (h0 + offs_m)[None, :], res,
             mask=mask_b[:, None])


@triton.jit(do_not_specialize=["n_tiles", "TM"])
def _ws_tile_mark_kernel(flags_ptr, idx_ptr, cs_ptr, nnz_ptr, n_tiles, TM, BLOCK: tl.constexpr):
    """``flags[b, indices[col_starts[b] + k] // TM] = 1`` for ``k < nnz[b]``.
    Grid ``(cdiv(max nnz, BLOCK), B)``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    row = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    tl.store(flags_ptr + b.to(tl.int64) * n_tiles + row // TM, tl.full([BLOCK], 1, tl.int32), mask=mask)


@triton.jit(do_not_specialize=["B", "n_tiles"])
def _ws_tile_lists_kernel(lists_ptr, counts_ptr, flags_ptr, B, n_tiles, BLOCK_B: tl.constexpr):
    """One program per parent tile ``m``: ``lists[m, j] = `` the ``j``-th
    sample with ``flags[b, m]`` set (ascending ``b``), ``counts[m]`` their
    number. Grid ``(n_tiles,)``."""
    m = tl.program_id(0)
    b = tl.arange(0, BLOCK_B)
    mask = b < B
    f = tl.load(flags_ptr + b.to(tl.int64) * n_tiles + m, mask=mask, other=0)
    pos = tl.cumsum(f, 0)
    tl.store(lists_ptr + m.to(tl.int64) * B + pos - 1, b, mask=mask & (f != 0))
    tl.store(counts_ptr + m, tl.sum(f, 0))


@triton.jit(do_not_specialize=["n_words", "TK"])
def _ws_tile_bits_kernel(bits_ptr, idx_ptr, cs_ptr, nnz_ptr, n_words, TK, BLOCK: tl.constexpr):
    """``bits[b, t // 32] |= 1 << (t % 32)`` for the child tile ``t =
    indices[col_starts[b] + k] // TK`` of every ``k < nnz[b]``. Grid
    ``(cdiv(max nnz, BLOCK), B)``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    t = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0) // TK
    one = tl.full([BLOCK], 1, tl.int32)
    tl.atomic_or(bits_ptr + b.to(tl.int64) * n_words + t // 32, one << (t % 32), mask=mask)


@triton.jit(do_not_specialize=["w_start", "B"])
def _ws_fwd_tiles_kernel(yd_ptr, w_ptr, xd_ptr, max_ptr, kin_ptr, lists_ptr, counts_ptr, bits_ptr,
                         w_start, B, H_in, H_out,
                         NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                         TM: tl.constexpr, TK: tl.constexpr, BB: tl.constexpr, NW: tl.constexpr,
                         PREC: tl.constexpr, ROW_MASK: tl.constexpr, SKIP_K: tl.constexpr):
    """Weight-stationary forward over every sample's active parent tiles.
    Grid ``(cdiv(B, BB), H_out // TM)``: program = (block of BB samples of
    the parent tile's list, parent tile ``m``); programs past the list's
    length return at once. As :func:`_ws_fwd_kernel`, the program walks the
    children TK at a time against the W tile ``[TK, TM]``, but its batch rows
    are the gathered samples ``lists[m, blk*BB : ...]``, and with ``ROW_MASK``
    an activation row is loaded only when the sample has an active child in
    the tile (its child-tile bits, ``NW`` words per sample held in registers,
    so the mask never waits for a load; the other rows are zero anyway). The
    children are walked word by word of the bits, 32 tiles per word. With
    ``SKIP_K`` a child tile no sample of the block touches is skipped (a
    branch that costs the loop its pipelining). Writes ``yd[b, m-tile]`` for
    the block's samples only — every active parent of a sample lies in a
    tile whose list holds the sample, which is all the gather that follows
    reads."""
    blk = tl.program_id(0)
    pid_m = tl.program_id(1)
    n = tl.load(counts_ptr + pid_m)
    if blk * BB >= n:
        return
    offs_b = blk * BB + tl.arange(0, BB)
    mask_b = offs_b < n
    ids = tl.load(lists_ptr + pid_m.to(tl.int64) * B + offs_b, mask=mask_b, other=0)
    ids64 = ids.to(tl.int64)
    h0 = pid_m * TM
    pb = h0 // BS
    w0 = h0 % BS
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, TK)
    offs_w = tl.arange(0, NW)
    w_rows = w_ptr + w_start + pb.to(tl.int64) * (NB_ch * CBS * BS) + w0
    x_rows = xd_ptr + ids64[:, None] * H_in
    words = tl.load(bits_ptr + ids64[:, None] * NW + offs_w[None, :], mask=mask_b[:, None], other=0)
    n_tiles = H_in // TK

    acc = tl.zeros([BB, TM], dtype=tl.float32)
    for wi in range(0, NW):
        # the block's bits of the 32 child tiles of word wi, one int per sample
        w_sel = tl.sum(tl.where(offs_w[None, :] == wi, words, 0), axis=1)
        any_bit = tl.sum(w_sel, 0)
        n_j = tl.minimum(32, n_tiles - wi * 32)
        if SKIP_K:
            if any_bit != 0:
                for j in range(0, n_j):
                    touch = mask_b & (((w_sel >> j) & 1) != 0)
                    k0 = (wi * 32 + j) * TK
                    cb = k0 // CBS
                    cs0 = k0 % CBS
                    wt = tl.load(w_rows + cb * (CBS * BS) + (cs0 + offs_k)[:, None] * BS + offs_m[None, :])
                    x = tl.load(x_rows + (k0 + offs_k)[None, :], mask=touch[:, None], other=0.0)
                    acc += tl.dot(x, wt, input_precision=PREC)
        else:
            for j in range(0, n_j):
                if ROW_MASK:
                    touch = mask_b & (((w_sel >> j) & 1) != 0)
                else:
                    touch = mask_b
                k0 = (wi * 32 + j) * TK
                cb = k0 // CBS
                cs0 = k0 % CBS
                wt = tl.load(w_rows + cb * (CBS * BS) + (cs0 + offs_k)[:, None] * BS + offs_m[None, :])
                x = tl.load(x_rows + (k0 + offs_k)[None, :], mask=touch[:, None], other=0.0)
                acc += tl.dot(x, wt, input_precision=PREC)

    mx = tl.maximum(tl.load(max_ptr + ids, mask=mask_b, other=0.0), -1e30)
    kin = tl.load(kin_ptr + ids, mask=mask_b, other=0)
    res = tl.log(acc) + mx[:, None]
    res = tl.where(kin[:, None] == 0, -float("inf"), res)
    tl.store(yd_ptr + ids64[:, None] * H_out + (h0 + offs_m)[None, :], res, mask=mask_b[:, None])


@triton.jit(do_not_specialize=["flow_stride", "mars_stride", "gt_ld", "gt_off"])
def _ws_densify_g_kernel(gd_ptr, gt_ptr, idx_ptr, nflow_ptr, nmars_ptr, max_ptr, kin_ptr,
                         cs_ptr, nnz_ptr, flow_stride, mars_stride, H_out, gt_ld, gt_off,
                         BLOCK: tl.constexpr, ROUND: tl.constexpr, WRITE_T: tl.constexpr):
    """Scatter ``gs = nflow * exp(max_b - nmars)`` (0 for samples without
    active inputs) of the batched output column into the pre-zeroed dense
    ``gd[B, H_out]`` and, with ``WRITE_T``, into the transposed stash slot
    ``gt[h, gt_off + b]``. ``max_b`` is the input max the forward used, so
    ``gs * x = nflow * exp(in - nmars)`` exactly as the gather kernel's
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
    tl.store(gd_ptr + b * H_out + h, g, mask=mask)
    if WRITE_T:
        tl.store(gt_ptr + gt_off + h * gt_ld + b, g, mask=mask)


@triton.jit(do_not_specialize=["w_start", "B"])
def _ws_bwd_flow_kernel(fd_ptr, w_ptr, gd_ptr, act_ptr, nact_ptr, w_start, B, H_in, H_out,
                        NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                        TK: tl.constexpr, TM: tl.constexpr, BB: tl.constexpr,
                        PREC: tl.constexpr, GATHER_K: tl.constexpr):
    """Weight-stationary element-flow backward. Grid ``(cdiv(B, BB),
    H_in // TK)``: program = (batch slice, block of TK children); walks the
    parents TM at a time, ``fd[b, c] = sum_h gd[b, h] * W[h, c]``. The
    ``[TM, TK]`` W tile is parents-contiguous, i.e. K-major for the dot.
    With ``GATHER_K`` the children are the batch's active ones only,
    ``act[0:nact]`` (sorted, :func:`_ws_active_children`), programs past
    them return; ``fd`` is written at those children only, which is all the
    gather that follows reads."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_k = tl.arange(0, TK)
    offs_m = tl.arange(0, TM)
    offs_b = pid_b * BB + tl.arange(0, BB)
    mask_b = offs_b < B
    if GATHER_K:
        n_act = tl.load(nact_ptr)
        if pid_c * TK >= n_act:
            return
        c = pid_c * TK + offs_k
        mask_k = c < n_act
        c = tl.load(act_ptr + c, mask=mask_k, other=0)
    else:
        c = pid_c * TK + offs_k
        mask_k = offs_k < TK

    w_cols = w_ptr + w_start + c.to(tl.int64)[None, :] * BS
    acc = tl.zeros([BB, TK], dtype=tl.float32)
    for h0 in range(0, H_out, TM):
        pb = h0 // BS
        w0 = h0 % BS
        w = tl.load(w_cols + pb.to(tl.int64) * (NB_ch * CBS * BS) + (w0 + offs_m)[:, None],
                    mask=mask_k[None, :], other=0.0)
        g = tl.load(gd_ptr + offs_b[:, None] * H_out + (h0 + offs_m)[None, :],
                    mask=mask_b[:, None], other=0.0)
        acc += tl.dot(g, w, input_precision=PREC)
    tl.store(fd_ptr + offs_b[:, None] * H_in + c[None, :], acc,
             mask=mask_b[:, None] & mask_k[None, :])


@triton.jit(do_not_specialize=["w_start", "B"])
def _ws_bwd_flow_tiles_kernel(fd_ptr, w_ptr, gd_ptr, lists_ptr, counts_ptr, bits_ptr,
                              w_start, B, H_in, H_out,
                              NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                              TK: tl.constexpr, TM: tl.constexpr, BB: tl.constexpr, NW: tl.constexpr,
                              PREC: tl.constexpr, ROW_MASK: tl.constexpr, SKIP_K: tl.constexpr):
    """Weight-stationary element-flow backward over every sample's active
    child tiles — the mirror of :func:`_ws_fwd_tiles_kernel`. Grid
    ``(cdiv(B, BB), H_in // TK)``: program = (block of BB samples of the
    child tile's list, child tile ``k``); it walks the parents TM at a time,
    word by word of the parent-tile bits, ``fd[b, c] = sum_h gd[b, h] * W[h,
    c]``, loading a sample's ``gd`` row only for the parent tiles it has an
    active parent in. ``fd`` is written at the block's samples and this
    tile's children only, which is all the gather that follows reads."""
    blk = tl.program_id(0)
    pid_c = tl.program_id(1)
    n = tl.load(counts_ptr + pid_c)
    if blk * BB >= n:
        return
    offs_b = blk * BB + tl.arange(0, BB)
    mask_b = offs_b < n
    ids = tl.load(lists_ptr + pid_c.to(tl.int64) * B + offs_b, mask=mask_b, other=0)
    ids64 = ids.to(tl.int64)
    offs_k = tl.arange(0, TK)
    offs_m = tl.arange(0, TM)
    offs_w = tl.arange(0, NW)
    c = pid_c * TK + offs_k
    w_cols = w_ptr + w_start + c.to(tl.int64)[None, :] * BS
    g_rows = gd_ptr + ids64[:, None] * H_out
    words = tl.load(bits_ptr + ids64[:, None] * NW + offs_w[None, :], mask=mask_b[:, None], other=0)
    n_tiles = H_out // TM

    acc = tl.zeros([BB, TK], dtype=tl.float32)
    for wi in range(0, NW):
        w_sel = tl.sum(tl.where(offs_w[None, :] == wi, words, 0), axis=1)
        any_bit = tl.sum(w_sel, 0)
        n_j = tl.minimum(32, n_tiles - wi * 32)
        if SKIP_K:
            if any_bit != 0:
                for j in range(0, n_j):
                    touch = mask_b & (((w_sel >> j) & 1) != 0)
                    h0 = (wi * 32 + j) * TM
                    pb = h0 // BS
                    w0 = h0 % BS
                    w = tl.load(w_cols + pb.to(tl.int64) * (NB_ch * CBS * BS) + (w0 + offs_m)[:, None])
                    g = tl.load(g_rows + (h0 + offs_m)[None, :], mask=touch[:, None], other=0.0)
                    acc += tl.dot(g, w, input_precision=PREC)
        else:
            for j in range(0, n_j):
                if ROW_MASK:
                    touch = mask_b & (((w_sel >> j) & 1) != 0)
                else:
                    touch = mask_b
                h0 = (wi * 32 + j) * TM
                pb = h0 // BS
                w0 = h0 % BS
                w = tl.load(w_cols + pb.to(tl.int64) * (NB_ch * CBS * BS) + (w0 + offs_m)[:, None])
                g = tl.load(g_rows + (h0 + offs_m)[None, :], mask=touch[:, None], other=0.0)
                acc += tl.dot(g, w, input_precision=PREC)
    tl.store(fd_ptr + ids64[:, None] * H_in + c[None, :], acc, mask=mask_b[:, None])


@triton.jit(do_not_specialize=["v_stride", "out_stride"])
def _ws_gather_flow_kernel(out_ptr, fd_ptr, idx_ptr, val_ptr, max_ptr, cs_ptr, nnz_ptr,
                           v_stride, out_stride, H_in, BLOCK: tl.constexpr):
    """``flow_in[b, k] = exp(in[b, k] - max_b) * fd[b, indices[cs[b] + k]]``."""
    b = tl.program_id(1)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < tl.load(nnz_ptr + b)
    c = tl.load(idx_ptr + tl.load(cs_ptr + b) + k, mask=mask, other=0)
    v = tl.load(val_ptr + b * v_stride + k, mask=mask, other=-float("inf"))
    mx = tl.maximum(tl.load(max_ptr + b), -1e30)
    f = tl.load(fd_ptr + b * H_in + c, mask=mask, other=0.0)
    tl.store(out_ptr + b * out_stride + k, tl.exp(v - mx) * f, mask=mask)


@triton.jit(do_not_specialize=["r0", "pid_start", "pfid_start"])
def _ws_pflow_epilogue_kernel(pf_ptr, w_ptr, s_ptr, r0, pid_start, pfid_start, H_in,
                              NB_ch: tl.constexpr, BS: tl.constexpr, CBS: tl.constexpr,
                              TM: tl.constexpr, TK: tl.constexpr, NEGATE: tl.constexpr):
    """``param_flows[(h, c)] += W[(h, c)] * S[h - r0, c]`` for one
    ``[rows, H_in]`` panel ``S = Gt[r0:r0+rows] @ Xt^T`` of the deferred
    param-flow GEMM (same ``(pb*NB_ch + cb)*CBS*BS + cs*BS + w`` layout for
    params and param flows). Grid ``(rows // TM, H_in // TK)``."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    h0 = r0 + pid_m * TM
    c0 = pid_c * TK
    pb = h0 // BS
    w0 = h0 % BS
    cb = c0 // CBS
    cs0 = c0 % CBS
    offs_m = tl.arange(0, TM)
    offs_k = tl.arange(0, TK)
    s = tl.load(s_ptr + (pid_m * TM + offs_m).to(tl.int64)[:, None] * H_in
                + (c0 + offs_k)[None, :])
    off = ((pb.to(tl.int64) * NB_ch + cb) * (CBS * BS)
           + (cs0 + offs_k)[None, :] * BS + (w0 + offs_m)[:, None])
    upd = tl.load(w_ptr + pid_start + off) * s
    if NEGATE:
        upd = -upd
    pf = tl.load(pf_ptr + pfid_start + off)
    tl.store(pf_ptr + pfid_start + off, pf + upd)
