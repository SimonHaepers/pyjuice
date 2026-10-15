from __future__ import annotations


from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .block_diagonal_sum_layer import BlockDiagonalSumLayer
from .sparse_node_values import SparseNodeValues
from .sparse_io_block_diagonal_sum_layer import (
    _DENSE_BATCHED, _SharedTensor, _bd_dense_batched, _build_block_offsets_table,
    _row_mapped_csc, _sv_dense_rows, _sv_to_dense_rows,
)


class SparseOutputBlockDiagonalSumLayer(BlockDiagonalSumLayer):
    """**Dense**-in, sparse-out variant of :class:`BlockDiagonalSumLayer`.

    The BD₂ half of a Monarch factorisation over sparse emissions: its
    child is a plain (dense) product layer — typically the Monarch
    permutation — but its sole consumer is a :class:`CoSparseProdLayer`
    whose SparseCategorical input's CSC column defines which of this sum's
    outputs are ever read. Forward therefore computes only the ``K_out``
    active rows (``K_out_j × cbs`` MACs per block instead of
    ``bs × cbs``) and writes a packed :class:`SparseNodeValues`;
    ``node_mars`` is **never** written — the consumer reads
    ``self._sparse_outputs`` directly (same contract as
    :class:`SparseIOSumLayer` / :class:`SparseIOBlockDiagonalSumLayer`).

    Backward takes ``sv_flow_out`` from the downstream
    :class:`CoSparseProdLayer` (stashed in ``self._sparse_flows``),
    writes the dense child's ``element_flows`` (plain stores — each child
    slot belongs to exactly one parent block; blocks with no active
    output get exact zeros), and optionally accumulates param flows for
    the active ``K_out_j × cbs`` weight entries via ``atomic_add`` (the
    atomics double as the cross-sample reduction at B>1) — EM-ready,
    unlike the inference-only dense BD backward.

    ``allow_modify_flows`` is irrelevant here (parent flows arrive packed
    from the consumer, never through ``node_flows``) and is ignored, the
    same way the sparse-IO sum layers ignore it.

    Batching follows the batched sparse-IO chain: B=1 runs single-column
    kernels; B>1 runs ``IS_BATCHED`` variants with one program per
    (parent_block, sample), per-sample CSC columns and block-offset table
    rows. ``missing_mask`` handling (all-rows sv_out) stays B=1-only.

    Row-mapped consumers: when the consumer's edge to this sum maps its
    row ``h`` to node ``row_map[h]`` (``SparseProdNodes.dense_row_map`` —
    e.g. the second Monarch permutation, ``P · BD2 · P · BD1``), packed
    slot ``j`` of sv_out holds node ``row_map[csc_row_j]``. The kernels
    then walk a compile-time row-mapped copy of the CSC pattern (mapped
    rows sorted per column, block-offset table over them) and scatter /
    gather each value through its original slot index
    (``HAS_SLOT_MAP``), so the consumer still sees its own slot order.

    From ``_WS_MIN_BATCH`` samples up (the sparse-IO weight-stationary knobs)
    a batched block takes the dense path instead: the dense
    :class:`BlockDiagonalSumLayer` forward fills this layer's own
    ``node_mars`` rows and sv_out gathers its active rows from them; backward
    writes sv_out / sv_flow_out back into this layer's ``node_mars`` /
    ``node_flows`` rows (``-inf`` / 0 off the support — nothing else reads
    them) and runs the dense backward, which writes each param flow once
    instead of one atomic per (sample, active row, child).

    Constraints (inherited or asserted):
      * Block-diagonal pattern (``NB == NB_ch``, ``bs == cbs``,
        ``edge_ids = arange(NB)[None, :].repeat(2, 1)``).
      * Single child group per sum node.
      * ``propagation_alg == 'LL'``.
    """

    output_storage = "scratch"
    graph_host_state = ("_sparse_outputs", "_sparse_flows", "_cached_out_offsets")

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int,
                 global_pid_start: int, global_pfid_start: int,
                 node2tiednodes: dict,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_block: int = 8,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False,
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
        )

        output_sparsity_dists = kwargs.pop("output_sparsity_dists", None)
        output_sparsity_num_rows = kwargs.pop("output_sparsity_num_rows", None)
        output_row_maps = kwargs.pop("output_row_maps", None)
        for name, val in (
            ("output_sparsity_var_ids", output_sparsity_var_ids),
            ("output_sparsity_dists", output_sparsity_dists),
            ("output_sparsity_num_rows", output_sparsity_num_rows),
        ):
            assert val is not None and len(val) == len(self.nodes), (
                f"SparseOutputBlockDiagonalSumLayer requires {name} (one "
                "per sum node)."
            )
        self._output_sparsity_var_ids: List[int] = list(output_sparsity_var_ids)
        self._output_sparsity_dists = list(output_sparsity_dists)
        self._output_sparsity_num_rows: List[int] = list(output_sparsity_num_rows)
        if output_row_maps is None:
            output_row_maps = [None] * len(self.nodes)
        assert len(output_row_maps) == len(self.nodes)

        # Forward-cached sv_out per ns (read by downstream CoSparseProdLayer).
        self._sparse_outputs: List[Optional[SparseNodeValues]] = [None] * len(self.nodes)
        # Backward flow container written by downstream CoSparseProdLayer.
        self._sparse_flows: List[Optional[SparseNodeValues]] = [None] * len(self.nodes)
        # Per-ns (offsets, tile-bound) cache populated at forward time so
        # backward reuses the same partition + specialisation.
        self._cached_out_offsets: List[Optional[tuple]] = [None] * len(self.nodes)

        # Per-block GPU workspaces for sv_out values — sized lazily.
        self._fwd_values_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)
        # Lazily-allocated arange indices for the missing-mask all-rows
        # sv_out path, keyed by ``out_num_rows``.
        self._missing_indices_cache: dict = {}

        # Compile-time per-column block-offsets tables (output side). Two
        # tile-width bounds — same policy as the sparse-input sibling:
        # ``_MAX_K_OUT_ACTIVE`` (worst per-block nnz across all CSC
        # columns) for observed-token calls, ``_MAX_K_OUT_MISSING``
        # (additionally ≥ BS, the all-rows case) for B=1 missing calls.
        # Row-mapped ns (see class docstring) read their offsets / rows /
        # slots from a :class:`_RowMappedCSC` instead (``_out_row_maps``).
        self._out_block_offsets_tables: List[_SharedTensor] = []
        self._missing_out_offsets: List[_SharedTensor] = []
        self._out_row_maps: list = []
        # Device row maps for the dense batched path (``None`` = identity).
        self._out_row_map_ids: List[Optional[_SharedTensor]] = []
        max_k_out_across_ns = 0
        max_k_missing_across_ns = 0
        for blk_idx, (block, out_dist, row_map) in enumerate(zip(
            self._bd_blocks, self._output_sparsity_dists, output_row_maps,
        )):
            _, _, _, _, NB, BS, _CBS = block
            if row_map is not None and torch.equal(
                row_map.cpu(), torch.arange(row_map.numel(), dtype=row_map.dtype),
            ):
                row_map = None
            if row_map is None:
                out_table, max_k_out = _build_block_offsets_table(out_dist, NB, BS)
                missing_offs = _SharedTensor(torch.arange(NB + 1, dtype=torch.int32) * BS)
                max_k_missing = BS
                mapped = None
            else:
                mapped = _row_mapped_csc(out_dist, row_map, NB, BS)
                out_table, max_k_out = mapped.offsets, mapped.max_k
                missing_offs, max_k_missing = mapped.missing_offsets, mapped.missing_max_k
            self._out_block_offsets_tables.append(out_table)
            self._missing_out_offsets.append(missing_offs)
            self._out_row_maps.append(mapped)
            self._out_row_map_ids.append(
                None if row_map is None else _SharedTensor(row_map.detach().cpu().long()))
            max_k_out_across_ns = max(max_k_out_across_ns, max_k_out)
            max_k_missing_across_ns = max(max_k_missing_across_ns, max_k_missing)
        self._MAX_K_OUT_ACTIVE: int = max(
            triton.next_power_of_2(max_k_out_across_ns), 1,
        )
        self._MAX_K_OUT_MISSING: int = max(
            triton.next_power_of_2(max(max_k_out_across_ns, max_k_missing_across_ns)),
            1,
        )

    def graph_columns(self):
        return list(zip(self._output_sparsity_dists, self._output_sparsity_var_ids))

    def graph_rebind_outputs(self, static_ctx, data_cpu: torch.Tensor,
                             data_list: Optional[list],
                             pattern_cache: Optional[dict]) -> None:
        """After a CUDA-graph replay, before the eager backward: rebuild
        ``self._sparse_outputs`` and the backward caches the way the eager
        forward would have (host-described patterns, B=1 column views), with
        ``values`` aliasing the ``static_ctx`` buffers the replay wrote — the
        same memory layout at every B (see :class:`SparseIOSumLayer`)."""
        B = static_ctx.B
        for blk_idx in range(len(self._bd_blocks)):
            out_dist = self._output_sparsity_dists[blk_idx]
            ws = static_ctx.buffer((id(self), "fwd", blk_idx), (B * out_dist._max_nnz_per_col,))
            sv_out = out_dist.build_sparse_pattern(
                data=data_cpu, var_id=self._output_sparsity_var_ids[blk_idx],
                num_rows=self._output_sparsity_num_rows[blk_idx], device=ws.device,
                values_out=ws, data_list=data_list, pattern_cache=pattern_cache,
            )
            self._sparse_outputs[blk_idx] = sv_out
            if sv_out.total_nnz == 0:
                self._cached_out_offsets[blk_idx] = None
            elif _bd_dense_batched(B):
                self._cached_out_offsets[blk_idx] = _DENSE_BATCHED
            else:
                out_offs = self._lookup_out_block_offsets(
                    blk_idx, is_missing=False, data=static_ctx.data,
                    data_list=data_list, device=ws.device, batch_size=B,
                )
                out_rows, out_slots = self._out_rows_and_slots(blk_idx, sv_out, False, ws.device)
                self._cached_out_offsets[blk_idx] = (
                    out_offs, self._MAX_K_OUT_ACTIVE, out_rows, out_slots)

    def __repr__(self) -> str:
        return (
            f"SparseOutputBlockDiagonalSumLayer("
            f"nid_range=({self._layer_nid_range[0]}, {self._layer_nid_range[1]}), "
            f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"out_vars={self._output_sparsity_var_ids})"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _lookup_out_block_offsets(self, blk_idx: int, is_missing: bool,
                                  data: torch.Tensor,
                                  data_list: Optional[list],
                                  device: torch.device,
                                  batch_size: int = 1,
                                  batched: bool = False) -> torch.Tensor:
        """Per-block start offsets into the output column(s) — mirror of
        the sparse-input sibling's ``_lookup_in_block_offsets`` on the
        BS-aligned output table (batched layout for B>1 or *batched*)."""
        # Shared device copies (plain attributes, not buffers).
        table = self._out_block_offsets_tables[blk_idx].on(device)
        missing_offs = self._missing_out_offsets[blk_idx].on(device)

        var_id = self._output_sparsity_var_ids[blk_idx]
        if batch_size > 1 or batched:
            return table.index_select(0, data[var_id, :]).contiguous()
        if is_missing:
            return missing_offs
        if data_list is not None:
            return table[data_list[var_id]]
        v_1d = data[var_id:var_id + 1, 0]
        return table.index_select(0, v_1d).squeeze(0)

    def _out_rows_and_slots(self, blk_idx: int, sv_out: SparseNodeValues,
                            is_missing: bool, device: torch.device):
        """``(row indices, slot map or None)`` the kernels walk for ns
        ``blk_idx``: the sv_out pattern itself, or — for a row-mapped ns —
        its row-mapped CSC copy, laid out like ``sv_out.indices`` (a view
        at the column start at B=1, the full array when batched)."""
        mapped = self._out_row_maps[blk_idx]
        if mapped is None:
            return sv_out.indices, None
        if is_missing:
            return mapped.missing_rows.on(device), mapped.missing_slots.on(device)
        rows, slots = mapped.rows.on(device), mapped.slots.on(device)
        if not sv_out.is_batched:
            rows, slots = rows[sv_out.col_start:], slots[sv_out.col_start:]
        return rows, slots

    def _row_map_on(self, blk_idx: int, device: torch.device) -> Optional[torch.Tensor]:
        row_map = self._out_row_map_ids[blk_idx]
        return None if row_map is None else row_map.on(device)

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
        self._bd_forward_precision = "ieee" if force_use_fp32 else "tf32x3"
        self._bd_forward_bf16 = force_use_bf16 and not force_use_fp32
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL", (
            "SparseOutputBlockDiagonalSumLayer requires propagation_alg == 'LL'."
        )
        assert params.dim() == 1
        if static_ctx is not None:
            self._forward_static(node_mars, element_mars, params, data, static_ctx)
            return None
        assert data is not None, (
            "SparseOutputBlockDiagonalSumLayer.forward requires `data` to "
            "build the output-sparsity pattern for each ns."
        )
        assert missing_mask is None or batch_size == 1, (
            "missing_mask on the sparse fast path is B=1 only."
        )

        data_for_pattern = data_cpu if data_cpu is not None else data

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
                        "SparseOutputBlockDiagonalSumLayer.forward got a 2D "
                        "missing_mask with neither dim == 1."
                    )
            missing_mask_cpu = mm.cpu() if mm.device.type != "cpu" else mm

        for blk_idx, block in enumerate(self._bd_blocks):
            nid_start, cid_start, pid_start, _pfid_start, NB, BS, CBS = block

            out_dist = self._output_sparsity_dists[blk_idx]
            out_var_id = self._output_sparsity_var_ids[blk_idx]
            out_num_rows = self._output_sparsity_num_rows[blk_idx]
            out_is_missing = bool(missing_mask_cpu[out_var_id].item()) \
                if missing_mask_cpu is not None else False

            # --- Build sv_out (same pattern as the sparse-IO layers) ---- #
            if out_is_missing:
                indices = self._missing_indices_cache.get(out_num_rows)
                if indices is None or indices.device != node_mars.device:
                    indices = torch.arange(
                        out_num_rows, dtype=torch.long, device=node_mars.device,
                    ).contiguous()
                    self._missing_indices_cache[out_num_rows] = indices
                ws_out = self._fwd_values_workspaces[blk_idx]
                if (ws_out is None or ws_out.device != node_mars.device
                        or ws_out.numel() < out_num_rows):
                    ws_out = torch.empty(
                        out_num_rows, dtype=torch.float32,
                        device=node_mars.device,
                    )
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
                self._cached_out_offsets[blk_idx] = None
                continue

            if _bd_dense_batched(sv_out.batch_size):
                self._dense_block_forward(node_mars, element_mars, params, blk_idx, block, sv_out)
                self._cached_out_offsets[blk_idx] = _DENSE_BATCHED
                continue

            out_offs = self._lookup_out_block_offsets(
                blk_idx, is_missing=out_is_missing, data=data,
                data_list=data_list, device=node_mars.device,
                batch_size=batch_size,
            )
            max_k_out = self._MAX_K_OUT_MISSING if out_is_missing \
                else self._MAX_K_OUT_ACTIVE
            out_rows, out_slots = self._out_rows_and_slots(
                blk_idx, sv_out, out_is_missing, node_mars.device,
            )
            # Cache for backward — the backward signature doesn't carry
            # ``missing_mask`` / ``data`` (same protocol as the siblings).
            self._cached_out_offsets[blk_idx] = (out_offs, max_k_out, out_rows, out_slots)
            assert sv_out.is_batched or batch_size == 1, (
                "batch_size > 1 but build_sparse_pattern produced a "
                "non-batched sv_out — inconsistent chain state."
            )
            self._launch_forward(node_mars, element_mars, params, block, sv_out, out_offs, max_k_out,
                                 out_rows, out_slots, batch_size)

        return None

    def _forward_static(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                        params: torch.Tensor, data: torch.Tensor, static_ctx) -> None:
        """CUDA-graph forward (see ``pyjuice.model.sparse_cudagraph``): sv_out
        is the device-resolved pattern of the consumer's column — the same
        static slot the consumer reads — in a ``static_ctx`` workspace; block
        offsets are gathered on device from the graph's batch; the batched
        kernels run at any B (B=1 included), or the dense path. A sample with
        an empty column just gets no outputs."""
        B = static_ctx.B
        for blk_idx, block in enumerate(self._bd_blocks):
            out_dist = self._output_sparsity_dists[blk_idx]
            out_var = self._output_sparsity_var_ids[blk_idx]
            ws = static_ctx.buffer((id(self), "fwd", blk_idx), (B * static_ctx.capacity(out_dist, out_var),))
            sv_out = static_ctx.pattern(
                out_dist, out_var, self._output_sparsity_num_rows[blk_idx], ws,
            )
            self._sparse_outputs[blk_idx] = sv_out
            # A masked output variable needs all rows: the dense path.
            if static_ctx.is_masked(out_var) or _bd_dense_batched(B):
                self._cached_out_offsets[blk_idx] = _DENSE_BATCHED
                self._dense_block_forward(node_mars, element_mars, params, blk_idx, block, sv_out)
                continue
            out_offs = self._lookup_out_block_offsets(
                blk_idx, is_missing=False, data=data, data_list=None,
                device=node_mars.device, batch_size=B, batched=True,
            )
            out_rows, out_slots = self._out_rows_and_slots(blk_idx, sv_out, False, node_mars.device)
            self._cached_out_offsets[blk_idx] = (out_offs, self._MAX_K_OUT_ACTIVE, out_rows, out_slots)
            self._launch_forward(node_mars, element_mars, params, block, sv_out, out_offs,
                                 self._MAX_K_OUT_ACTIVE, out_rows, out_slots, B)

    def _dense_block_forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                             params: torch.Tensor, blk_idx: int, block: tuple,
                             sv_out: SparseNodeValues) -> None:
        """Dense batched path: the dense block-diagonal forward into this
        layer's own ``node_mars`` rows, then sv_out gathers its active rows."""
        self._dense_forward_block(node_mars, element_mars, params, block, sv_out.batch_size)
        _sv_dense_rows(node_mars, block[0], sv_out, sv_out.values, False,
                       self._row_map_on(blk_idx, node_mars.device))

    def _launch_forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                        block: tuple, sv_out: SparseNodeValues, out_offs: torch.Tensor,
                        max_k_out: int, out_rows: torch.Tensor,
                        out_slots: Optional[torch.Tensor], batch_size: int) -> None:
        """:func:`_fw_bd_sparse_out_kernel` over grid ``(NB, B)``: the batched
        layout whenever *sv_out* is batched — B>1 or a static container —
        else the B=1 column view."""
        _, cid_start, pid_start, _, NB, BS, CBS = block
        row_map = self._input_row_map(block)
        batched = sv_out.is_batched
        _fw_bd_sparse_out_kernel[(NB, batch_size)](
            out_values_ptr=sv_out.values,
            element_mars_ptr=node_mars if row_map is not None else element_mars,
            input_map=params if row_map is None else row_map, MAP_INPUT=row_map is not None,
            mparams_ptr=params,
            out_indices_ptr=out_rows,
            out_slots_ptr=out_rows if out_slots is None else out_slots,
            out_block_offsets_ptr=out_offs,
            out_col_starts_ptr=sv_out.col_starts if batched else sv_out.values,  # unused at B=1
            cid_start=cid_start,
            pid_start=pid_start,
            batch_size=batch_size,
            v_stride=sv_out.values.stride(0) if batched else 0,
            offs_stride=out_offs.stride(0) if batched else 0,
            BS=BS,
            CBS=CBS,
            CBS_PADDED=triton.next_power_of_2(CBS),
            MAX_K_OUT=max_k_out,
            IS_BATCHED=batched,
            HAS_SLOT_MAP=out_slots is not None,
        )

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
        assert propagation_alg == "LL" and not logspace_flows \
               and not allow_neg_flows, (
            "SparseOutputBlockDiagonalSumLayer.backward requires "
            "propagation_alg='LL', logspace_flows=False, "
            "allow_neg_flows=False."
        )
        assert not accumulate_ch_flows, (
            "SparseOutputBlockDiagonalSumLayer.backward overwrites its "
            "child's element_flows slots; accumulate_ch_flows is not "
            "supported."
        )
        # ``allow_modify_flows`` is ignored: parent flows arrive as the
        # packed sv_flow_out from the consumer, never via node_flows.

        compute_pflows = param_flows is not None

        for blk_idx, block in enumerate(self._bd_blocks):
            nid_start, cid_start, pid_start, pfid_start, NB, BS, CBS = block
            row_map = self._input_row_map(block)

            sv_flow_out = self._sparse_flows[blk_idx]
            sv_out = self._sparse_outputs[blk_idx]
            K_out = sv_out.total_nnz if sv_out is not None else 0

            if K_out == 0:
                # No active output (any sample): zero flow into every child
                # slot — the dense child's backward reads all of them.
                if row_map is None:
                    element_flows[cid_start:cid_start + NB * CBS, :].fill_(0.0)
                else:
                    node_flows.index_fill_(0, row_map, 0.0)
                self._sparse_flows[blk_idx] = None
                self._cached_out_offsets[blk_idx] = None
                continue

            assert sv_flow_out is not None, (
                "SparseOutputBlockDiagonalSumLayer.backward expected "
                "sv_flow_out from the downstream CoSparseProdLayer."
            )
            self._sparse_flows[blk_idx] = None  # consume-and-clear

            cached = self._cached_out_offsets[blk_idx]
            assert cached is not None, (
                "SparseOutputBlockDiagonalSumLayer.backward expected cached "
                "block offsets from forward; call forward before backward."
            )
            if cached is _DENSE_BATCHED:
                # Parent side back to dense rows: marginals -inf and flows 0 off
                # the support, so those rows contribute exact zeros.
                out_row_map = self._row_map_on(blk_idx, node_mars.device)
                _sv_to_dense_rows(node_mars, nid_start, NB * BS, float("-inf"),
                                  sv_out, sv_out.values, out_row_map)
                _sv_to_dense_rows(node_flows, nid_start, NB * BS, 0.0,
                                  sv_out, sv_flow_out.values, out_row_map)
                self._dense_backward_block(
                    node_flows, element_flows, node_mars, element_mars, params,
                    param_flows if compute_pflows else None, block, batch_size,
                    negate_pflows=negate_pflows, force_use_fp32=force_use_fp32,
                )
                continue
            out_offs, max_k_out, out_rows, out_slots = cached

            CBS_PADDED = triton.next_power_of_2(CBS)
            if sv_out.is_batched:
                grid = (NB, batch_size)
                _bk_bd_sparse_out_kernel[grid](
                    element_flows_ptr=node_flows if row_map is not None else element_flows,
                    element_mars_ptr=node_mars if row_map is not None else element_mars,
                    input_map=params if row_map is None else row_map, MAP_INPUT=row_map is not None,
                    sv_flow_out_ptr=sv_flow_out.values,
                    sv_out_values_ptr=sv_out.values,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows
                                else sv_out.values),
                    out_indices_ptr=out_rows,
                    out_slots_ptr=out_rows if out_slots is None else out_slots,
                    out_block_offsets_ptr=out_offs,
                    out_col_starts_ptr=sv_out.col_starts,
                    cid_start=cid_start,
                    pid_start=pid_start,
                    pfid_start=pfid_start,
                    batch_size=batch_size,
                    v_stride=sv_out.values.stride(0),
                    f_stride=sv_flow_out.values.stride(0),
                    offs_stride=out_offs.stride(0),
                    BS=BS,
                    CBS=CBS,
                    CBS_PADDED=CBS_PADDED,
                    MAX_K_OUT=max_k_out,
                    IS_BATCHED=True,
                    HAS_SLOT_MAP=out_slots is not None,
                    COMPUTE_PFLOWS=1 if compute_pflows else 0,
                    NEGATE_PFLOWS=1 if negate_pflows else 0,
                )
            else:
                grid = (NB, 1)
                _bk_bd_sparse_out_kernel[grid](
                    element_flows_ptr=node_flows if row_map is not None else element_flows,
                    element_mars_ptr=node_mars if row_map is not None else element_mars,
                    input_map=params if row_map is None else row_map, MAP_INPUT=row_map is not None,
                    sv_flow_out_ptr=sv_flow_out.values,
                    sv_out_values_ptr=sv_out.values,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows
                                else sv_out.values),
                    out_indices_ptr=out_rows,
                    out_slots_ptr=out_rows if out_slots is None else out_slots,
                    out_block_offsets_ptr=out_offs,
                    out_col_starts_ptr=sv_out.values,  # unused at B=1
                    cid_start=cid_start,
                    pid_start=pid_start,
                    pfid_start=pfid_start,
                    batch_size=batch_size,
                    v_stride=0,
                    f_stride=0,
                    offs_stride=0,
                    BS=BS,
                    CBS=CBS,
                    CBS_PADDED=CBS_PADDED,
                    MAX_K_OUT=max_k_out,
                    IS_BATCHED=False,
                    HAS_SLOT_MAP=out_slots is not None,
                    COMPUTE_PFLOWS=1 if compute_pflows else 0,
                    NEGATE_PFLOWS=1 if negate_pflows else 0,
                )

        return None


# ======================================================================= #
# Triton kernels
# ======================================================================= #


@triton.jit(
    do_not_specialize=["cid_start", "pid_start", "batch_size", "BS", "CBS",
                       "v_stride", "offs_stride"],
    do_not_specialize_on_alignment=[
        "out_values_ptr", "out_indices_ptr", "out_slots_ptr",
        "out_block_offsets_ptr", "out_col_starts_ptr",
    ],
)
def _fw_bd_sparse_out_kernel(
    out_values_ptr,
    element_mars_ptr, mparams_ptr,
    out_indices_ptr, out_slots_ptr, out_block_offsets_ptr, out_col_starts_ptr,
    cid_start, pid_start,
    batch_size,
    v_stride, offs_stride,
    BS, CBS, input_map, MAP_INPUT: tl.constexpr,
    CBS_PADDED: tl.constexpr,
    MAX_K_OUT: tl.constexpr,
    IS_BATCHED: tl.constexpr,
    HAS_SLOT_MAP: tl.constexpr,
):
    """Dense-in / sparse-out forward for one parent block. Grid
    ``(NB, B)`` — one program per (parent_block, sample); axis 1 is
    degenerate at B=1.

    Per-program math (block ``pid_nb``, B=1 form):

      [out_s, out_e) := out_block_offsets[pid_nb : pid_nb+2]  # K_out_j slots
      pslot[m] := out_indices[out_s + m] - pid_nb * BS         # [0, BS)
      emars[c] := element_mars[cid_start + pid_nb*CBS + c, b]  # dense CBS
      max := max_c emars[c] ; lin[c] := exp(emars[c] - max)
      W[m, c] := mparams[pid_start + pid_nb*BS*CBS + c*BS + pslot[m]]
      acc[m] := Σ_c W[m, c] · lin[c]
      out_values[out_s + m] := log(acc[m] + 1e-24) + max
          (exact -inf passthrough when every emars[c] is -inf)

    ``IS_BATCHED == 1``: sample ``b`` reads its own table row (rebased on
    ``out_col_starts[b]`` for the index gather), reads ``element_mars``
    column ``b``, and stores to ``out_values[b*v_stride + out_s + m]``.

    ``HAS_SLOT_MAP == 1`` (row-mapped consumer): ``out_indices`` holds the
    mapped rows sorted per column and the store goes to the entry's
    original slot, ``out_values[b*v_stride + out_slots[idx_base + m]]``.

    Tile is ``[MAX_K_OUT, CBS_PADDED]`` — at low density MAX_K_OUT is far
    below BS, which is the whole point (``K_out_j × cbs`` MACs per block).
    """
    pid_nb = tl.program_id(0)
    pid_b = tl.program_id(1)

    if IS_BATCHED:
        out_s = tl.load(out_block_offsets_ptr + pid_b * offs_stride + pid_nb)
        out_e = tl.load(out_block_offsets_ptr + pid_b * offs_stride + pid_nb + 1)
        cs_b = tl.load(out_col_starts_ptr + pid_b)
        idx_base = cs_b + out_s
        val_col = pid_b * v_stride
    else:
        out_s = tl.load(out_block_offsets_ptr + pid_nb)
        out_e = tl.load(out_block_offsets_ptr + pid_nb + 1)
        idx_base = out_s
        val_col = 0
    K_out_j = out_e - out_s

    offs_m = tl.arange(0, MAX_K_OUT)
    mask_m = offs_m < K_out_j

    offs_c = tl.arange(0, CBS_PADDED)
    mask_c = offs_c < CBS

    out_idx = tl.load(out_indices_ptr + idx_base + offs_m, mask=mask_m, other=0)
    pslot = out_idx - pid_nb * BS                                                # [MAX_K_OUT]

    # Dense child marginals for this block (int64: cid * B crosses 2^31 at
    # large H·T·B).
    child_rows = (cid_start + pid_nb * CBS + offs_c).to(tl.int64)
    if MAP_INPUT:
        child_rows = tl.load(input_map + pid_nb * CBS + offs_c, mask=mask_c, other=0).to(tl.int64)
    emars_ptr = child_rows * batch_size + pid_b
    emars = tl.load(element_mars_ptr + emars_ptr, mask=mask_c,
                    other=-float("inf"))                                         # [CBS_PADDED]

    emars_max = tl.max(emars, axis=0)
    emars_max_safe = tl.where(emars_max == -float("inf"), 0.0, emars_max)
    emars_lin = tl.where(mask_c, tl.exp(emars - emars_max_safe), 0.0)

    block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
    weight_addr = (
        block_base
        + offs_c[None, :] * BS
        + pslot[:, None]
    )                                                                            # [MAX_K_OUT, CBS_PADDED]
    weight = tl.load(
        mparams_ptr + weight_addr,
        mask=mask_m[:, None] & mask_c[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(weight * emars_lin[None, :], axis=1)                            # [MAX_K_OUT]

    result = tl.where(
        emars_max == -float("inf"),
        -float("inf"),
        tl.log(acc) + emars_max,
    )
    if HAS_SLOT_MAP:
        slot = tl.load(out_slots_ptr + idx_base + offs_m, mask=mask_m, other=0)
    else:
        slot = out_s + offs_m
    tl.store(out_values_ptr + val_col + slot, result, mask=mask_m)


@triton.jit(
    do_not_specialize=["cid_start", "pid_start", "pfid_start", "batch_size",
                       "BS", "CBS", "v_stride", "f_stride", "offs_stride"],
    do_not_specialize_on_alignment=[
        "sv_flow_out_ptr", "sv_out_values_ptr", "out_indices_ptr",
        "out_slots_ptr", "out_block_offsets_ptr", "out_col_starts_ptr",
        "pflows_ptr",
    ],
)
def _bk_bd_sparse_out_kernel(
    element_flows_ptr, element_mars_ptr,
    sv_flow_out_ptr, sv_out_values_ptr,
    mparams_ptr,
    pflows_ptr,
    out_indices_ptr, out_slots_ptr, out_block_offsets_ptr, out_col_starts_ptr,
    cid_start, pid_start, pfid_start,
    batch_size,
    v_stride, f_stride, offs_stride,
    BS, CBS, input_map, MAP_INPUT: tl.constexpr,
    CBS_PADDED: tl.constexpr,
    MAX_K_OUT: tl.constexpr,
    IS_BATCHED: tl.constexpr,
    HAS_SLOT_MAP: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr,
    NEGATE_PFLOWS: tl.constexpr,
):
    """Element-flow + (optional) param-flow backward for one parent block.
    Grid ``(NB, B)`` — axis 1 degenerate at B=1.

    Per-program math (block ``pid_nb``, sparse parent side / dense child
    side, B=1 form):

      pslot[m] := out_indices[out_s + m] - pid_nb * BS
      W[m, c] := mparams[pid_start + pid_nb*BS*CBS + c*BS + pslot[m]]
      contrib[m, c] := sv_flow_out[m] · W[m, c]
                       · exp(emars[c] - sv_out_values[m])
      element_flows[cid_start + pid_nb*CBS + c, b] := Σ_m contrib[m, c]
      # When COMPUTE_PFLOWS == 1:
      param_flows[pfid_start + pid_nb*BS*CBS + c*BS + pslot[m]] += contrib[m, c]

    ``HAS_SLOT_MAP == 1``: ``sv_flow_out`` / ``sv_out_values`` are read at
    each entry's original slot (see the forward kernel).

    Element-flow stores are plain (each child slot belongs to exactly one
    parent block; each (slot, sample) is owned by one program). Blocks
    with ``K_out_j == 0`` store exact zeros — the dense child's backward
    reads every slot. Param-flow scatter uses ``atomic_add``: tied
    SumNodes accumulate across calls, and at B>1 programs of different
    samples share addresses — the atomic IS the batch reduction.
    """
    pid_nb = tl.program_id(0)
    pid_b = tl.program_id(1)

    if IS_BATCHED:
        out_s = tl.load(out_block_offsets_ptr + pid_b * offs_stride + pid_nb)
        out_e = tl.load(out_block_offsets_ptr + pid_b * offs_stride + pid_nb + 1)
        cs_b = tl.load(out_col_starts_ptr + pid_b)
        idx_base = cs_b + out_s
        val_col = pid_b * v_stride
        flow_col = pid_b * f_stride
    else:
        out_s = tl.load(out_block_offsets_ptr + pid_nb)
        out_e = tl.load(out_block_offsets_ptr + pid_nb + 1)
        idx_base = out_s
        val_col = 0
        flow_col = 0
    K_out_j = out_e - out_s

    offs_m = tl.arange(0, MAX_K_OUT)
    mask_m = offs_m < K_out_j

    offs_c = tl.arange(0, CBS_PADDED)
    mask_c = offs_c < CBS

    out_idx = tl.load(out_indices_ptr + idx_base + offs_m, mask=mask_m, other=0)
    pslot = out_idx - pid_nb * BS

    if HAS_SLOT_MAP:
        slot = tl.load(out_slots_ptr + idx_base + offs_m, mask=mask_m, other=0)
    else:
        slot = out_s + offs_m
    nflows = tl.load(sv_flow_out_ptr + flow_col + slot, mask=mask_m,
                     other=0.0)                                                  # [MAX_K_OUT]
    nmars = tl.load(sv_out_values_ptr + val_col + slot, mask=mask_m,
                    other=-float("inf"))

    child_rows = (cid_start + pid_nb * CBS + offs_c).to(tl.int64)
    if MAP_INPUT:
        child_rows = tl.load(input_map + pid_nb * CBS + offs_c, mask=mask_c, other=0).to(tl.int64)
    emars_ptr = child_rows * batch_size + pid_b
    emars = tl.load(element_mars_ptr + emars_ptr, mask=mask_c,
                    other=-float("inf"))                                         # [CBS_PADDED]

    block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
    weight_addr = (
        block_base
        + offs_c[None, :] * BS
        + pslot[:, None]
    )                                                                            # [MAX_K_OUT, CBS_PADDED]
    weight = tl.load(
        mparams_ptr + weight_addr,
        mask=mask_m[:, None] & mask_c[None, :],
        other=0.0,
    ).to(tl.float32)

    # contrib[m, c] = nflows[m] · W[m, c] · exp(emars[c] - nmars[m]).
    # nmars == -inf (zero-probability parent ⇒ zero flow) or emars == -inf
    # (impossible child) would drive the exp to inf/NaN — force those
    # terms to exact 0.
    contrib_factor = tl.where(
        (nmars[:, None] == -float("inf"))
        | (emars[None, :] == -float("inf")),
        0.0,
        tl.exp(emars[None, :] - nmars[:, None]),
    )
    contrib = nflows[:, None] * weight * contrib_factor
    contrib = tl.where(mask_m[:, None] & mask_c[None, :], contrib, 0.0)

    # eflows[c] = Σ_m contrib[m, c]; K_out_j == 0 ⇒ all-zero store (the
    # desired "no flow" result for an inactive block).
    eflows = tl.sum(contrib, axis=0)                                             # [CBS_PADDED]
    tl.store(element_flows_ptr + emars_ptr, eflows, mask=mask_c)

    if COMPUTE_PFLOWS == 1:
        pf_base = pfid_start + pid_nb.to(tl.int64) * BS * CBS
        pf_addr = (
            pf_base
            + offs_c[None, :] * BS
            + pslot[:, None]
        )
        pf_val = -contrib if NEGATE_PFLOWS == 1 else contrib
        tl.atomic_add(
            pflows_ptr + pf_addr,
            pf_val,
            mask=mask_m[:, None] & mask_c[None, :],
        )
