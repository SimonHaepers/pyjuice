"""CUDA-graph forward for circuits with data-dependent (sparse-IO) layers.

Layers such as :class:`SparseProdLayer` / :class:`SparseIOSumLayer` only touch
the rows of the observed token's CSC column, so their work depends on the
batch. Their eager forward looks the columns up on the host and sizes every
kernel grid to them, which a graph replay would skip. The graph forward moves
all of that to the device, whatever the circuit's structure (HMM chain, tree,
mixed with dense layers):

* the batch is copied into a fixed device buffer, and one small kernel per
  :class:`SparseCategorical` resolves every ``(dist, var)`` column any layer
  declared through ``Layer.graph_columns`` (:func:`_static_pattern_kernel`);
* those layers run their ``static_ctx`` forward on *static*
  :class:`SparseNodeValues` (``static=True``): the batched layout at any B
  (B=1 included), with the device ``col_starts`` / ``nnz`` and the
  column-size bound ``dist._max_nnz_per_col`` as ``total_nnz``. Grids are
  sized by the bound; the kernels mask by the per-sample sizes and return
  early past them;
* every workspace the captured layers write lives in this state, at addresses
  that stay fixed across replays and are never shared with the eager path.

Layers that never read the data on the host (dense, block-diagonal, blocked)
take part unchanged; a layer that still needs the host patterns opts out via
``Layer.graph_unsupported_reason`` and the circuit runs eagerly. See the
protocol on :class:`pyjuice.layer.Layer`.

One :class:`SparseGraphState` per batch size and execution profile. Observed
profiles reserve the largest CSC column; masked-query profiles reserve all
latent rows and append an all-row column to each distribution's indices.
Tokens and masks are runtime device data. ``device_sparse=True`` uses this
state without capture; ``record_cudagraph=True`` captures the inner forward.
Node-flow backward reuses the same patterns and supports separate capture.
Parameter-flow backward remains eagerly launched. Workspaces and graph
bindings are retained per profile, so switching batches does not rebuild
host-side patterns.
"""
from __future__ import annotations

from pyjuice.utils.kernel_launcher import runtime_triton_jit

from typing import Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.layer import SparseNodeValues
from pyjuice.layer.sparse_io_sum_layer import _WS_SCRATCH


_PATTERN_BLOCK_B = 128

_NUM_WARMUP = 2
"""Eager passes before the capture: the first compiles the Triton kernels and
allocates every workspace, the second runs the exact captured sequence."""


def graph_unsupported_reason(pc) -> Optional[str]:
    """``None`` if every inner layer of ``pc`` can run inside a CUDA graph,
    else the first layer's reason (used in the eager-fallback warning)."""
    for lg in pc.inner_layer_groups:
        for layer in lg:
            reason = layer.graph_unsupported_reason()
            if reason is not None:
                return reason
    return None


@runtime_triton_jit
def _static_pattern_kernel(cs_ptr, nnz_ptr, data_ptr, indptr_ptr, var_ptr, slot_ptr,
                           B, num_cats, missing_ptr, nnz_total, num_rows, BLOCK_B: tl.constexpr):
    """``col_starts[slot, b] = indptr[tok]``, ``nnz[slot, b] = indptr[tok + 1] -
    indptr[tok]`` with ``tok = data[var, b]``, for the ``(var, slot)`` pairs of
    one dist. Grid ``(num pairs, cdiv(B, BLOCK_B))``. An out-of-range token
    gets an empty column (LL ``-inf``) rather than an out-of-bounds read."""
    p = tl.program_id(0)
    offs_b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs_b < B
    var = tl.load(var_ptr + p).to(tl.int64)
    slot = tl.load(slot_ptr + p).to(tl.int64)
    tok = tl.load(data_ptr + var * B + offs_b, mask=mask, other=0).to(tl.int64)
    ok = (tok >= 0) & (tok < num_cats)
    tok = tl.where(ok, tok, 0)
    c0 = tl.load(indptr_ptr + tok, mask=mask, other=0)
    c1 = tl.load(indptr_ptr + tok + 1, mask=mask, other=0)
    missing = tl.load(missing_ptr + var * B + offs_b, mask=mask, other=False)
    n = tl.where(missing, num_rows, tl.where(ok, c1 - c0, 0))
    c0 = tl.where(missing, nnz_total, c0)
    tl.store(cs_ptr + slot * B + offs_b, c0.to(tl.int32), mask=mask)
    tl.store(nnz_ptr + slot * B + offs_b, n.to(tl.int32), mask=mask)


class SparseGraphState:
    """Device-side state of the captured sparse forward of one batch size:
    the input buffer, the per-``(dist, var)`` column table, every workspace
    the captured layers write, and the graph itself."""

    def __init__(self, pc, B: int, device: torch.device, data_dtype: torch.dtype = torch.long, query_masks: bool = False):
        self.B = B
        self.query_masks = query_masks
        self.device = device

        # One slot per (dist, var) column some layer declared it reads, and
        # one row range per layer that accumulates running maxima.
        self._layers = [layer for lg in pc.inner_layer_groups for layer in lg]
        self._slot_of: Dict[Tuple[int, int], int] = {}
        per_dist: Dict[int, Tuple[object, List[Tuple[int, int]]]] = {}
        self._max_rows: Dict[int, Tuple[int, int]] = {}
        n_max = 0
        for layer in self._layers:
            rows = layer.graph_max_rows()
            if rows > 0:
                self._max_rows[id(layer)] = (n_max, rows)
                n_max += rows
            for dist, var in layer.graph_columns():
                key = (id(dist), var)
                if key in self._slot_of:
                    continue
                self._slot_of[key] = len(self._slot_of)
                per_dist.setdefault(id(dist), (dist, []))[1].append(
                    (var, self._slot_of[key]))

        self._indices = {id(dist): torch.cat((dist._csc_indices,
                          torch.arange(dist._num_nodes, device=device, dtype=dist._csc_indices.dtype)))
                         if query_masks else dist._csc_indices
                         for dist, _ in per_dist.values()}
        self.forward_bindings = None
        self.backward_bindings = {}
        self.missing = torch.zeros(pc.num_vars, B, dtype=torch.bool, device=device)
        n_slots = max(len(self._slot_of), 1)
        self.col_starts = torch.zeros(n_slots, B, dtype=torch.int32, device=device)
        self.nnz = torch.zeros(n_slots, B, dtype=torch.int32, device=device)
        # Cached row views: one tensor object per slot, so consumers that
        # compare patterns by identity (CoSparseProdLayer) see equal objects.
        self._cs_rows = [self.col_starts[i] for i in range(n_slots)]
        self._nnz_rows = [self.nnz[i] for i in range(n_slots)]
        self._launches = []
        for dist, cols in per_dist.values():
            assert dist._csc_indptr.device == device, (
                "SparseCategorical CSC pattern is not on the circuit's device.")
            assert dist._nnz < 2 ** 31, "CSC nnz exceeds the int32 col_starts range."
            self._launches.append((
                dist,
                torch.tensor([v for v, _ in cols], dtype=torch.int32, device=device),
                torch.tensor([s for _, s in cols], dtype=torch.int32, device=device),
            ))

        # The running maxima of all layers (``[rows, B]`` each, filled by
        # atomic max) share one buffer, so a single fill per replay resets
        # them all instead of one fill node per layer.
        self._max_all = torch.empty(max(n_max, 1), B, dtype=torch.float32, device=device)

        # The batch, ``[num_vars, B]`` in the input's dtype: read by the pattern
        # kernel and handed to every inner layer as ``data`` (some read the
        # tokens on device, e.g. the blocked layers).
        self.data = torch.zeros(pc.num_vars, B, dtype=data_dtype, device=device)
        self._buffers: Dict[tuple, torch.Tensor] = {}
        self.ws_decisions: Dict[tuple, bool] = {}

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.node_mars: Optional[torch.Tensor] = None
        self.element_mars: Optional[torch.Tensor] = None
        self.node_flows: Optional[torch.Tensor] = None
        self.element_flows: Optional[torch.Tensor] = None
        self._param_ptrs = self._param_signature(pc)
        self._keepalive: List[torch.Tensor] = []
        self._pinned: List[Optional[torch.Tensor]] = [None, None]
        self._pinned_events: List[Optional[torch.cuda.Event]] = [None, None]
        self._pinned_idx = 0

    # ------------------------------------------------------------------ #
    # Used by the layers while capturing
    # ------------------------------------------------------------------ #

    def buffer(self, key: tuple, shape: Tuple[int, ...]) -> torch.Tensor:
        """Fixed-address fp32 workspace owned by this state (allocated by the
        first warm-up pass, reused by the capture and every replay)."""
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=torch.float32, device=self.device)
            self._buffers[key] = buf
        assert tuple(buf.shape) == tuple(shape), (key, tuple(buf.shape), shape)
        return buf

    def max_buffer(self, layer) -> torch.Tensor:
        """``[layer.graph_max_rows(), B]`` running-max rows of ``layer``, reset
        to ``-inf`` by :meth:`begin_pass`."""
        off, n = self._max_rows[id(layer)]
        return self._max_all[off:off + n]

    def slot(self, dist, var_id: int) -> int:
        return self._slot_of[(id(dist), var_id)]

    def pattern(self, dist, var_id: int, num_rows: int,
                values_ws: torch.Tensor) -> SparseNodeValues:
        """Device-described observed or all-row columns in a fixed workspace."""
        s = self.slot(dist, var_id)
        K = self.capacity(dist)
        return SparseNodeValues(
            col_start=0, total_nnz=K, indices=self._indices[id(dist)],
            values=values_ws[:self.B * K].view(self.B, K), num_rows=num_rows,
            batch_size=self.B, col_starts=self._cs_rows[s], nnz=self._nnz_rows[s],
            static=True,
        )

    def capacity(self, dist) -> int:
        return dist._num_nodes if self.query_masks else dist._max_nnz_per_col

    def snapshot_bindings(self):
        # Graph replay updates tensors but does not repeat Python assignments.
        # Restore the containers/caches belonging to this execution profile.
        names = ("_sparse_outputs", "_sparse_flows", "_cached_in_offsets",
                 "_cached_out_offsets", "_was_missing")
        return [(layer, name, list(getattr(layer, name)))
                for layer in self._layers for name in names if hasattr(layer, name)]

    @staticmethod
    def restore_bindings(bindings):
        for layer, name, values in bindings:
            setattr(layer, name, list(values))

    def begin_pass(self) -> None:
        """First nodes of the graph: reset the running maxima and resolve the
        batch's columns."""
        self._max_all.fill_(float("-inf"))
        self.compute_patterns()

    def compute_patterns(self) -> None:
        for dist, var_ids, slots in self._launches:
            grid = (var_ids.numel(), triton.cdiv(self.B, _PATTERN_BLOCK_B))
            _static_pattern_kernel[grid](
                self.col_starts, self.nnz, self.data, dist._csc_indptr, var_ids, slots,
                self.B, dist.num_cats, self.missing, dist._nnz, dist._num_nodes,
                BLOCK_B=_PATTERN_BLOCK_B,
            )

    # ------------------------------------------------------------------ #
    # Used by TensorCircuit
    # ------------------------------------------------------------------ #

    @staticmethod
    def _param_signature(pc) -> tuple:
        ptrs = [pc.params.data_ptr()]
        for layer in pc.input_layer_group:
            if isinstance(getattr(layer, "params", None), torch.Tensor):
                ptrs.append(layer.params.data_ptr())
        return tuple(ptrs)

    def matches(self, pc) -> bool:
        """The graph baked in the parameter addresses: still the same tensors?
        (In-place EM updates keep them; re-initialising or moving does not.)"""
        return self._param_ptrs == self._param_signature(pc)

    def load_inputs(self, run_data: torch.Tensor, missing_mask=None) -> None:
        """Copy the ``[num_vars, B]`` batch into the graph's input buffer."""
        if missing_mask is None:
            self.missing.zero_()
        else:
            if not self.query_masks:
                raise ValueError("A masked query requires a query workspace")
            mask = missing_mask.to(device=self.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask[:, None].expand_as(self.missing)
            elif mask.shape == (self.B, self.data.size(0)):
                mask = mask.T
            else:
                raise ValueError("missing_mask must have shape [num_vars] or [B, num_vars]")
            self.missing.copy_(mask)
        assert tuple(run_data.shape) == tuple(self.data.shape), (
            f"batch of shape {tuple(run_data.shape)} for a graph captured with "
            f"{tuple(self.data.shape)}")
        if run_data.device.type != "cpu":
            self.data.copy_(run_data)
            return
        # Host batch: stage through pinned memory so the copy is asynchronous
        # (a pageable H2D would block until the queue drains). Two staging
        # buffers, each reused only once its previous copy has completed.
        i = self._pinned_idx
        self._pinned_idx ^= 1
        if self._pinned_events[i] is not None:
            self._pinned_events[i].synchronize()
        if self._pinned[i] is None:
            self._pinned[i] = torch.empty(self.data.shape, dtype=self.data.dtype, pin_memory=True)
        self._pinned[i].copy_(run_data)
        self.data.copy_(self._pinned[i], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._pinned_events[i] = ev

    def capture(self, body: Callable[[], None]) -> None:
        """Warm up and capture ``body`` (the inner-layer pass). The inputs must
        already be loaded. WS dispatch uses model-level support statistics,
        independent of the particular capture tokens."""
        self.compute_patterns()

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(_NUM_WARMUP):
                body()
        torch.cuda.current_stream(self.device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        self.graph = graph
        self.forward_bindings = self.snapshot_bindings()
        # The WS kernels use process-wide grow-only scratch (`_ws_scratch`). A
        # later, larger eager pass may replace those buffers; keep the ones
        # the graph baked in alive so the replay never writes freed memory.
        self.retain_ws_buffers()

    def retain_ws_buffers(self):
        """Keep scratch addresses captured by either direction alive."""
        retained = {id(t) for t in self._keepalive}
        self._keepalive.extend(t for (_, dev), t in _WS_SCRATCH.items()
                               if dev == self.device and id(t) not in retained)

    def rebind_outputs(self, data_cpu: torch.Tensor, data_list: Optional[list],
                       pattern_cache: Optional[dict]) -> None:
        """After a replay, before an eager backward: let every layer rebuild
        the host-side state the eager backward reads from its forward."""
        for layer in self._layers:
            layer.graph_rebind_outputs(self, data_cpu, data_list, pattern_cache)
