from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .block_diagonal_sum_layer import BlockDiagonalSumLayer
from .sparse_node_values import SparseNodeValues
from .sparse_io_block_diagonal_sum_layer import _build_block_offsets_table


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

    Constraints (inherited or asserted):
      * Block-diagonal pattern (``NB == NB_ch``, ``bs == cbs``,
        ``edge_ids = arange(NB)[None, :].repeat(2, 1)``).
      * Single child group per sum node.
      * ``propagation_alg == 'LL'``.
    """

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
        self._out_block_offsets_tables: List[torch.Tensor] = []
        self._missing_out_offsets: List[torch.Tensor] = []
        max_k_out_across_ns = 0
        max_bs_across_ns = 0
        for blk_idx, (block, out_dist) in enumerate(zip(
            self._bd_blocks, self._output_sparsity_dists,
        )):
            _, _, _, _, NB, BS, _CBS = block
            out_table, max_k_out = _build_block_offsets_table(out_dist, NB, BS)
            self._out_block_offsets_tables.append(out_table)
            self._missing_out_offsets.append(
                torch.arange(NB + 1, dtype=torch.int32) * BS
            )
            max_k_out_across_ns = max(max_k_out_across_ns, max_k_out)
            max_bs_across_ns = max(max_bs_across_ns, BS)
        self._MAX_K_OUT_ACTIVE: int = max(
            triton.next_power_of_2(max_k_out_across_ns), 1,
        )
        self._MAX_K_OUT_MISSING: int = max(
            triton.next_power_of_2(max(max_k_out_across_ns, max_bs_across_ns)),
            1,
        )

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
                                  batch_size: int = 1) -> torch.Tensor:
        """Per-block start offsets into the output column(s) — mirror of
        the sparse-input sibling's ``_lookup_in_block_offsets`` on the
        BS-aligned output table."""
        table = self._out_block_offsets_tables[blk_idx]
        missing_offs = self._missing_out_offsets[blk_idx]

        if table.device != device:
            table = table.to(device)
            self._out_block_offsets_tables[blk_idx] = table
        if missing_offs.device != device:
            missing_offs = missing_offs.to(device)
            self._missing_out_offsets[blk_idx] = missing_offs

        var_id = self._output_sparsity_var_ids[blk_idx]
        if batch_size > 1:
            return table.index_select(0, data[var_id, :]).contiguous()
        if is_missing:
            return missing_offs
        if data_list is not None:
            return table[data_list[var_id]]
        v_1d = data[var_id:var_id + 1, 0]
        return table.index_select(0, v_1d).squeeze(0)

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
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL", (
            "SparseOutputBlockDiagonalSumLayer requires propagation_alg == 'LL'."
        )
        assert params.dim() == 1
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
            _nid_start, cid_start, pid_start, _pfid_start, NB, BS, CBS = block

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

            out_offs = self._lookup_out_block_offsets(
                blk_idx, is_missing=out_is_missing, data=data,
                data_list=data_list, device=node_mars.device,
                batch_size=batch_size,
            )
            max_k_out = self._MAX_K_OUT_MISSING if out_is_missing \
                else self._MAX_K_OUT_ACTIVE
            # Cache for backward — the backward signature doesn't carry
            # ``missing_mask`` / ``data`` (same protocol as the siblings).
            self._cached_out_offsets[blk_idx] = (out_offs, max_k_out)

            CBS_PADDED = triton.next_power_of_2(CBS)
            if sv_out.is_batched:
                grid = (NB, batch_size)
                _fw_bd_sparse_out_kernel[grid](
                    out_values_ptr=sv_out.values,
                    element_mars_ptr=element_mars,
                    mparams_ptr=params,
                    out_indices_ptr=sv_out.indices,
                    out_block_offsets_ptr=out_offs,
                    out_col_starts_ptr=sv_out.col_starts,
                    cid_start=cid_start,
                    pid_start=pid_start,
                    batch_size=batch_size,
                    v_stride=sv_out.values.stride(0),
                    offs_stride=out_offs.stride(0),
                    BS=BS,
                    CBS=CBS,
                    CBS_PADDED=CBS_PADDED,
                    MAX_K_OUT=max_k_out,
                    IS_BATCHED=True,
                )
            else:
                assert batch_size == 1, (
                    "batch_size > 1 but build_sparse_pattern produced a "
                    "non-batched sv_out — inconsistent chain state."
                )
                grid = (NB, 1)
                _fw_bd_sparse_out_kernel[grid](
                    out_values_ptr=sv_out.values,
                    element_mars_ptr=element_mars,
                    mparams_ptr=params,
                    out_indices_ptr=sv_out.indices,
                    out_block_offsets_ptr=out_offs,
                    out_col_starts_ptr=sv_out.values,  # unused at B=1
                    cid_start=cid_start,
                    pid_start=pid_start,
                    batch_size=batch_size,
                    v_stride=0,
                    offs_stride=0,
                    BS=BS,
                    CBS=CBS,
                    CBS_PADDED=CBS_PADDED,
                    MAX_K_OUT=max_k_out,
                    IS_BATCHED=False,
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
            _nid_start, cid_start, pid_start, pfid_start, NB, BS, CBS = block

            sv_flow_out = self._sparse_flows[blk_idx]
            sv_out = self._sparse_outputs[blk_idx]
            K_out = sv_out.total_nnz if sv_out is not None else 0

            if K_out == 0:
                # No active output (any sample): zero flow into every child
                # slot — the dense child's backward reads all of them.
                element_flows[cid_start:cid_start + NB * CBS, :].fill_(0.0)
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
            out_offs, max_k_out = cached
            self._cached_out_offsets[blk_idx] = None

            CBS_PADDED = triton.next_power_of_2(CBS)
            if sv_out.is_batched:
                grid = (NB, batch_size)
                _bk_bd_sparse_out_kernel[grid](
                    element_flows_ptr=element_flows,
                    element_mars_ptr=element_mars,
                    sv_flow_out_ptr=sv_flow_out.values,
                    sv_out_values_ptr=sv_out.values,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows
                                else sv_out.values),
                    out_indices_ptr=sv_out.indices,
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
                    COMPUTE_PFLOWS=1 if compute_pflows else 0,
                    NEGATE_PFLOWS=1 if negate_pflows else 0,
                )
            else:
                grid = (NB, 1)
                _bk_bd_sparse_out_kernel[grid](
                    element_flows_ptr=element_flows,
                    element_mars_ptr=element_mars,
                    sv_flow_out_ptr=sv_flow_out.values,
                    sv_out_values_ptr=sv_out.values,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows
                                else sv_out.values),
                    out_indices_ptr=sv_out.indices,
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
        "out_values_ptr", "out_indices_ptr", "out_block_offsets_ptr",
        "out_col_starts_ptr",
    ],
)
def _fw_bd_sparse_out_kernel(
    out_values_ptr,
    element_mars_ptr, mparams_ptr,
    out_indices_ptr, out_block_offsets_ptr, out_col_starts_ptr,
    cid_start, pid_start,
    batch_size,
    v_stride, offs_stride,
    BS, CBS,
    CBS_PADDED: tl.constexpr,
    MAX_K_OUT: tl.constexpr,
    IS_BATCHED: tl.constexpr,
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
        val_base = pid_b * v_stride + out_s
    else:
        out_s = tl.load(out_block_offsets_ptr + pid_nb)
        out_e = tl.load(out_block_offsets_ptr + pid_nb + 1)
        idx_base = out_s
        val_base = out_s
    K_out_j = out_e - out_s

    offs_m = tl.arange(0, MAX_K_OUT)
    mask_m = offs_m < K_out_j

    offs_c = tl.arange(0, CBS_PADDED)
    mask_c = offs_c < CBS

    out_idx = tl.load(out_indices_ptr + idx_base + offs_m, mask=mask_m, other=0)
    pslot = out_idx - pid_nb * BS                                                # [MAX_K_OUT]

    # Dense child marginals for this block (int64: cid * B crosses 2^31 at
    # large H·T·B).
    emars_ptr = (cid_start + pid_nb * CBS + offs_c).to(tl.int64) * batch_size \
        + pid_b
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
        tl.log(acc + 1e-24) + emars_max,
    )
    tl.store(out_values_ptr + val_base + offs_m, result, mask=mask_m)


@triton.jit(
    do_not_specialize=["cid_start", "pid_start", "pfid_start", "batch_size",
                       "BS", "CBS", "v_stride", "f_stride", "offs_stride"],
    do_not_specialize_on_alignment=[
        "sv_flow_out_ptr", "sv_out_values_ptr", "out_indices_ptr",
        "out_block_offsets_ptr", "out_col_starts_ptr", "pflows_ptr",
    ],
)
def _bk_bd_sparse_out_kernel(
    element_flows_ptr, element_mars_ptr,
    sv_flow_out_ptr, sv_out_values_ptr,
    mparams_ptr,
    pflows_ptr,
    out_indices_ptr, out_block_offsets_ptr, out_col_starts_ptr,
    cid_start, pid_start, pfid_start,
    batch_size,
    v_stride, f_stride, offs_stride,
    BS, CBS,
    CBS_PADDED: tl.constexpr,
    MAX_K_OUT: tl.constexpr,
    IS_BATCHED: tl.constexpr,
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
        val_base = pid_b * v_stride + out_s
        flow_base = pid_b * f_stride + out_s
    else:
        out_s = tl.load(out_block_offsets_ptr + pid_nb)
        out_e = tl.load(out_block_offsets_ptr + pid_nb + 1)
        idx_base = out_s
        val_base = out_s
        flow_base = out_s
    K_out_j = out_e - out_s

    offs_m = tl.arange(0, MAX_K_OUT)
    mask_m = offs_m < K_out_j

    offs_c = tl.arange(0, CBS_PADDED)
    mask_c = offs_c < CBS

    out_idx = tl.load(out_indices_ptr + idx_base + offs_m, mask=mask_m, other=0)
    pslot = out_idx - pid_nb * BS

    nflows = tl.load(sv_flow_out_ptr + flow_base + offs_m, mask=mask_m,
                     other=0.0)                                                  # [MAX_K_OUT]
    nmars = tl.load(sv_out_values_ptr + val_base + offs_m, mask=mask_m,
                    other=-float("inf"))

    emars_ptr = (cid_start + pid_nb * CBS + offs_c).to(tl.int64) * batch_size \
        + pid_b
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
