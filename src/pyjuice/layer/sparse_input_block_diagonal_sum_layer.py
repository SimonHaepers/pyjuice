from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from .block_diagonal_sum_layer import BlockDiagonalSumLayer
from .dense_sum_layer import DenseSumLayer
from .sparse_node_values import SparseNodeValues
from .sparse_prod_layer import SparseProdLayer
from .sparse_io_block_diagonal_sum_layer import _build_block_offsets_table


class SparseInputBlockDiagonalSumLayer(BlockDiagonalSumLayer):
    """Sparse-in, **dense**-out variant of :class:`BlockDiagonalSumLayer`.

    This is the BD₁ half of a Monarch factorisation over sparse emissions:
    the sum's child is a :class:`SparseProdLayer` (active rows restricted to
    the observed token's CSC column), but its output is consumed densely —
    e.g. by the Monarch permutation product — so forward writes the full
    ``NB * bs`` rows of ``node_mars``.

    Per parent block ``j`` the forward does ``K_in_j × bs`` MACs instead of
    the dense BD kernel's ``cbs × bs``: only the active child slots that fall
    inside block ``j`` participate, and a block with no active input gets an
    exact ``-inf`` output (log of an empty sum). The per-block partition of
    ``sv_in.indices`` reuses the compile-time block-offsets tables of
    :class:`SparseIOBlockDiagonalSumLayer` (CSC columns are sorted, so block
    ``j`` occupies a contiguous slice of the column).

    Backward mirrors :class:`SparseInputSumLayer`'s sparse fast path: when
    every upstream :class:`SparseProdLayer` has ``_skip_scatter=True`` the
    element flow is written straight into a ``sv_flow`` mirroring the input
    pattern (plain stores — each active slot belongs to exactly one parent
    block), and param flows are accumulated into the BD param-flow buffer
    via ``atomic_add``. Mixed-consumer topologies (scatter still on) fall
    back to the inherited dense BD backward over ``element_flows``
    (inference-only — ``param_flows`` is rejected there).

    Batching follows the batched sparse-IO chain: B=1 runs the classic
    single-column kernels; B>1 runs the ``IS_BATCHED`` variants where each
    (parent_block, sample) program reads the sample's own CSC column via
    ``col_starts`` / a per-sample block-offsets table row, and param-flow
    atomics double as the cross-sample reduction. The batched path requires
    the pure sparse chain (``_skip_scatter=True`` upstream); the dense
    fallback (mixed-consumer topologies) and ``missing_mask`` handling
    remain B=1-only — same constraints as :class:`SparseInputSumLayer`.

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
                 inner_layer_groups: Optional[list] = None,
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

        assert inner_layer_groups is not None, (
            "SparseInputBlockDiagonalSumLayer needs the already-compiled "
            "inner_layer_groups to resolve the upstream SparseProdLayer "
            "that owns each sum's child."
        )

        # Resolve the upstream SparseProdLayer producing each sum's child —
        # same lookup as :class:`SparseIOBlockDiagonalSumLayer`, plus the
        # input-side ``(dist, var_id)`` snapshot for the block-offsets
        # table lookup at forward time.
        self._sparse_input_refs: List[Tuple[SparseProdLayer, int]] = []
        self._input_sparsity_dists: list = []
        self._input_sparsity_var_ids: List[int] = []
        for ns in self.nodes:
            cs = ns.chs[0]
            found = None
            for lg in inner_layer_groups:
                if not lg.is_prod():
                    continue
                for layer in lg:
                    if not isinstance(layer, SparseProdLayer):
                        continue
                    for idx, prod_ns in enumerate(layer.nodes):
                        if prod_ns is cs:
                            found = (layer, idx)
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            assert found is not None, (
                "SparseInputBlockDiagonalSumLayer: child ProdNodes is not "
                "owned by any SparseProdLayer in the compiled "
                "inner_layer_groups."
            )
            self._sparse_input_refs.append(found)
            sparse_prod_layer, prod_ns_idx = found
            prod_ns = sparse_prod_layer.nodes[prod_ns_idx]
            self._input_sparsity_dists.append(prod_ns.sparse_input_ns.dist)
            self._input_sparsity_var_ids.append(prod_ns.var_id)

        # Per-ns block-offset cache populated at forward so backward reuses
        # the same partition without re-threading ``missing_mask``.
        self._cached_in_offsets: List[Optional[torch.Tensor]] = [None] * len(self.nodes)

        # Backward flow workspaces (mirrors sv_in's pattern) — sized lazily.
        self._bwd_flow_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)

        # Compile-time per-column block-offsets tables (input side only —
        # the output side is dense). Two tile-width bounds:
        #   * ``_MAX_K_IN_ACTIVE`` — worst per-block nnz across all CSC
        #     columns, padded to a power of two. This is the bound for
        #     every observed-token call; at low density it is far below
        #     ``CBS``, which directly shrinks the per-program weight-gather
        #     tile (the dominant cost of the per-sample batched kernels).
        #   * ``_MAX_K_IN_MISSING`` — additionally covers the missing-mask
        #     all-rows case (``K_j == CBS``). Only ever needed at B=1
        #     (missing_mask is B=1-only on the sparse chain).
        # One Triton specialisation per bound actually used.
        self._in_block_offsets_tables: List[torch.Tensor] = []
        self._missing_in_offsets: List[torch.Tensor] = []
        max_k_in_across_ns = 0
        max_cbs_across_ns = 0
        for blk_idx, (block, in_dist) in enumerate(zip(
            self._bd_blocks, self._input_sparsity_dists,
        )):
            _, _, _, _, NB, _BS, CBS = block
            in_table, max_k_in = _build_block_offsets_table(in_dist, NB, CBS)
            self._in_block_offsets_tables.append(in_table)
            self._missing_in_offsets.append(
                torch.arange(NB + 1, dtype=torch.int32) * CBS
            )
            max_k_in_across_ns = max(max_k_in_across_ns, max_k_in)
            max_cbs_across_ns = max(max_cbs_across_ns, CBS)
        self._MAX_K_IN_ACTIVE: int = max(
            triton.next_power_of_2(max_k_in_across_ns), 1,
        )
        self._MAX_K_IN_MISSING: int = max(
            triton.next_power_of_2(max(max_k_in_across_ns, max_cbs_across_ns)),
            1,
        )

    def __repr__(self) -> str:
        return (
            f"SparseInputBlockDiagonalSumLayer("
            f"nid_range=({self._layer_nid_range[0]}, {self._layer_nid_range[1]}), "
            f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"num_sum_ns={len(self._sparse_input_refs)})"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _lookup_in_block_offsets(self, blk_idx: int, is_missing: bool,
                                 data: torch.Tensor,
                                 data_list: Optional[list],
                                 device: torch.device,
                                 batch_size: int = 1) -> torch.Tensor:
        """Per-block start offsets into the active column(s) — the
        input-side half of
        :meth:`SparseIOBlockDiagonalSumLayer._lookup_block_offsets` (same
        table layout, same sync-free indexing paths).

        B=1 returns ``[NB+1]`` (offsets are relative to ``sv_in.indices``,
        which is a view of the column slice). B>1 returns ``[B, NB+1]`` —
        one table row per sample's observed token, gathered with a single
        ``index_select`` on the device data; the kernel rebases row ``b``
        on ``col_starts[b]`` (``sv_in.indices`` is the full
        ``_csc_indices`` there)."""
        table = self._in_block_offsets_tables[blk_idx]
        missing_offs = self._missing_in_offsets[blk_idx]

        # Lazy device migration — plain attributes, so ``.to(device)``
        # doesn't move them.
        if table.device != device:
            table = table.to(device)
            self._in_block_offsets_tables[blk_idx] = table
        if missing_offs.device != device:
            missing_offs = missing_offs.to(device)
            self._missing_in_offsets[blk_idx] = missing_offs

        var_id = self._input_sparsity_var_ids[blk_idx]
        if batch_size > 1:
            # ``is_missing`` is excluded up front at B>1 (family-wide
            # constraint, see forward).
            return table.index_select(0, data[var_id, :]).contiguous()
        if is_missing:
            return missing_offs
        if data_list is not None:
            return table[data_list[var_id]]
        v_1d = data[var_id:var_id + 1, 0]
        return table.index_select(0, v_1d).squeeze(0)

    def _all_skip_scatter(self) -> bool:
        return all(sp._skip_scatter for sp, _ in self._sparse_input_refs)

    def _run_modify_flows_prepass(self, node_flows: torch.Tensor,
                                  node_mars: torch.Tensor,
                                  batch_size: int,
                                  propagation_alg: str, **kwargs) -> None:
        """In-place ``log(flow) - log_marg`` transform on this layer's own
        parent rows — the pre-pass the ``ALLOW_MODIFY_FLOWS`` kernel branch
        expects. Same kernel + tiling as ``DenseSumLayer.backward`` (and the
        replica in ``SparseInputSumLayer.backward``)."""
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(
            propagation_alg, **kwargs,
        )
        alpha = float(propagation_alg_kwargs.get("alpha", 0.0))
        for block in self._bd_blocks:
            nid_start, _cid_start, _pid_start, _pfid_start, NB, bs, _cbs = block
            layer_n_nodes = NB * bs
            BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
            BLOCK_B = min(2048, BATCH_SIZE_NP2)
            BLOCK_M = min(max(2048 // BLOCK_B, 1), bs)
            if BLOCK_M < 1:
                BLOCK_M = 1
            grid = (triton.cdiv(batch_size, BLOCK_B),
                    triton.cdiv(layer_n_nodes, BLOCK_M))
            DenseSumLayer._bk_triton_dense_modify_flow_kernel[grid](
                node_flows=node_flows,
                node_mars=node_mars,
                nid_start=nid_start,
                batch_size=batch_size,
                num_parents=layer_n_nodes,
                BLOCK_B=BLOCK_B,
                BLOCK_M=BLOCK_M,
                propagation_alg_id=propagation_alg_id,
                alpha=alpha,
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
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert propagation_alg == "LL", (
            "SparseInputBlockDiagonalSumLayer requires propagation_alg == 'LL'."
        )
        assert params.dim() == 1
        if batch_size > 1:
            assert missing_mask is None, (
                "SparseInputBlockDiagonalSumLayer: missing_mask handling is "
                "B=1 only (same constraint as the rest of the sparse chain)."
            )
            assert self._all_skip_scatter(), (
                "SparseInputBlockDiagonalSumLayer at B>1 requires the pure "
                "sparse chain (every upstream SparseProdLayer must have "
                "_skip_scatter=True); the dense scatter bridge is B=1 only."
            )

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
                        "SparseInputBlockDiagonalSumLayer.forward got a 2D "
                        "missing_mask with neither dim == 1."
                    )
            missing_mask_cpu = mm.cpu() if mm.device.type != "cpu" else mm

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._bd_blocks, self._sparse_input_refs,
        )):
            nid_start, _cid_start, pid_start, _pfid_start, NB, BS, CBS = block
            sv_in = sparse_prod._sparse_outputs[ns_idx]
            K_in = sv_in.total_nnz

            if K_in == 0:
                # Empty active column: every parent's mixture is a sum over
                # zero terms ⇒ log(0) = -inf.
                node_mars[nid_start:nid_start + NB * BS, :].fill_(float("-inf"))
                self._cached_in_offsets[blk_idx] = None
                continue

            in_var_id = self._input_sparsity_var_ids[blk_idx]
            in_is_missing = bool(missing_mask_cpu[in_var_id].item()) \
                if missing_mask_cpu is not None else False

            in_offs = self._lookup_in_block_offsets(
                blk_idx, is_missing=in_is_missing, data=data,
                data_list=data_list, device=node_mars.device,
                batch_size=batch_size,
            )
            max_k_in = self._MAX_K_IN_MISSING if in_is_missing \
                else self._MAX_K_IN_ACTIVE
            # Cache for backward — the standard sum-layer backward signature
            # doesn't carry ``missing_mask`` (same protocol as
            # SparseIOBlockDiagonalSumLayer); the tile bound rides along so
            # backward compiles against the same specialisation.
            self._cached_in_offsets[blk_idx] = (in_offs, max_k_in)

            BS_PADDED = triton.next_power_of_2(BS)
            if sv_in.is_batched:
                # Per-sample columns: each program handles one
                # (parent_block, sample). ``sv_in.max_val`` is the fused
                # per-sample [B] max — mandatory on the batched chain (same
                # contract as SparseInputSumLayer.forward).
                assert sv_in.max_val is not None, (
                    "batched SparseInputBlockDiagonalSumLayer.forward "
                    "requires the fused per-sample sv.max_val from the "
                    "upstream prod layer."
                )
                grid = (NB, batch_size)
                _fw_sparse_in_bd_kernel[grid](
                    node_mars_ptr=node_mars,
                    mparams_ptr=params,
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    in_block_offsets_ptr=in_offs,
                    max_val_ptr=sv_in.max_val,
                    col_starts_ptr=sv_in.col_starts,
                    nid_start=nid_start,
                    pid_start=pid_start,
                    batch_size=batch_size,
                    v_stride=sv_in.values.stride(0),
                    offs_stride=in_offs.stride(0),
                    BS=BS,
                    CBS=CBS,
                    BS_PADDED=BS_PADDED,
                    MAX_K_IN=max_k_in,
                    IS_BATCHED=True,
                )
            else:
                assert batch_size == 1, (
                    "batch_size > 1 but the upstream SparseProdLayer "
                    "produced a non-batched sv — inconsistent chain state."
                )
                max_val = sv_in.max_val if sv_in.max_val is not None \
                    else sv_in.values.max()
                grid = (NB, 1)
                _fw_sparse_in_bd_kernel[grid](
                    node_mars_ptr=node_mars,
                    mparams_ptr=params,
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    in_block_offsets_ptr=in_offs,
                    max_val_ptr=max_val,
                    col_starts_ptr=sv_in.values,  # unused at B=1
                    nid_start=nid_start,
                    pid_start=pid_start,
                    batch_size=batch_size,
                    v_stride=0,
                    offs_stride=0,
                    BS=BS,
                    CBS=CBS,
                    BS_PADDED=BS_PADDED,
                    MAX_K_IN=max_k_in,
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
            "SparseInputBlockDiagonalSumLayer.backward requires "
            "propagation_alg='LL', logspace_flows=False, "
            "allow_neg_flows=False."
        )

        compute_pflows = param_flows is not None

        if not self._all_skip_scatter():
            assert batch_size == 1, (
                "SparseInputBlockDiagonalSumLayer dense-fallback backward "
                "(mixed-consumer topology) is B=1 only; the batched chain "
                "requires _skip_scatter=True on every upstream "
                "SparseProdLayer."
            )
            # Mixed-consumer topology: the upstream prod scattered to
            # element_mars for its dense consumers, and its backward gathers
            # element_flows — so the inherited dense BD backward is the
            # correct (and only) protocol. It has no param-flow support.
            assert not compute_pflows, (
                "SparseInputBlockDiagonalSumLayer.backward with param_flows "
                "requires the sparse fast path (every upstream "
                "SparseProdLayer must have _skip_scatter=True)."
            )
            # No modify pre-pass here — the inherited dense BD backward
            # runs it itself.
            return super().backward(
                node_flows, element_flows, node_mars, element_mars, params,
                param_flows=None, allow_modify_flows=allow_modify_flows,
                propagation_alg=propagation_alg,
                logspace_flows=logspace_flows, negate_pflows=negate_pflows,
                accumulate_ch_flows=accumulate_ch_flows,
                allow_neg_flows=allow_neg_flows,
                force_use_fp32=force_use_fp32, **kwargs,
            )

        assert not accumulate_ch_flows, (
            "SparseInputBlockDiagonalSumLayer (skip_scatter) backward writes "
            "sv_flow straight into the upstream prod layer; "
            "accumulate_ch_flows is not supported."
        )

        # Same in-place transform DenseSumLayer/SparseInputSumLayer do for
        # their own parent rows — required before the ALLOW_MODIFY_FLOWS
        # kernel branch reads ``node_flows``.
        if allow_modify_flows:
            self._run_modify_flows_prepass(
                node_flows, node_mars, batch_size, propagation_alg, **kwargs,
            )

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._bd_blocks, self._sparse_input_refs,
        )):
            nid_start, _cid_start, pid_start, pfid_start, NB, BS, CBS = block
            sv_in = sparse_prod._sparse_outputs[ns_idx]
            K_in = sv_in.total_nnz

            # Mirror sv_in's pattern so SparseProdLayer.backward (sparse
            # path) can find the flow. B=1 is workspace-backed to keep
            # per-call allocations off the hot path (same pattern as
            # SparseIOBlockDiagonalSumLayer); the batched chain allocates a
            # fresh [B, K_stride] mirror per call (same as
            # SparseInputSumLayer's batched path).
            if sv_in.is_batched:
                sv_flow_in = sv_in.like_pattern(torch.empty_like(sv_in.values))
            else:
                ws_flow = self._bwd_flow_workspaces[blk_idx]
                in_dist = self._input_sparsity_dists[blk_idx]
                in_max_nnz = max(in_dist._max_nnz_per_col, in_dist._num_nodes)
                if (ws_flow is None or ws_flow.device != node_mars.device
                        or ws_flow.numel() < max(K_in, in_max_nnz)):
                    ws_flow = torch.empty(
                        max(K_in, in_max_nnz),
                        dtype=torch.float32, device=node_mars.device,
                    )
                    self._bwd_flow_workspaces[blk_idx] = ws_flow
                sv_flow_in = sv_in.like_pattern(ws_flow.narrow(0, 0, K_in))
            sparse_prod._sparse_flows[ns_idx] = sv_flow_in

            if K_in == 0:
                self._cached_in_offsets[blk_idx] = None
                continue

            cached = self._cached_in_offsets[blk_idx]
            assert cached is not None, (
                "SparseInputBlockDiagonalSumLayer.backward expected cached "
                "block offsets from forward; call forward before backward."
            )
            in_offs, max_k_in = cached

            if sv_in.is_batched:
                grid = (NB, batch_size)
                _bk_sparse_in_bd_kernel[grid](
                    flow_in_ptr=sv_flow_in.values,
                    node_flows_ptr=node_flows,
                    node_mars_ptr=node_mars,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows
                                else sv_flow_in.values),
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    in_block_offsets_ptr=in_offs,
                    col_starts_ptr=sv_in.col_starts,
                    nid_start=nid_start,
                    pid_start=pid_start,
                    pfid_start=pfid_start,
                    batch_size=batch_size,
                    v_stride=sv_in.values.stride(0),
                    f_stride=sv_flow_in.values.stride(0),
                    offs_stride=in_offs.stride(0),
                    BS=BS,
                    CBS=CBS,
                    BS_PADDED=triton.next_power_of_2(BS),
                    MAX_K_IN=max_k_in,
                    IS_BATCHED=True,
                    ALLOW_MODIFY_FLOWS=1 if allow_modify_flows else 0,
                    COMPUTE_PFLOWS=1 if compute_pflows else 0,
                    NEGATE_PFLOWS=1 if negate_pflows else 0,
                )
            else:
                grid = (NB, 1)
                _bk_sparse_in_bd_kernel[grid](
                    flow_in_ptr=sv_flow_in.values,
                    node_flows_ptr=node_flows,
                    node_mars_ptr=node_mars,
                    mparams_ptr=params,
                    pflows_ptr=(param_flows if compute_pflows
                                else sv_flow_in.values),
                    in_indices_ptr=sv_in.indices,
                    in_values_ptr=sv_in.values,
                    in_block_offsets_ptr=in_offs,
                    col_starts_ptr=sv_in.values,  # unused at B=1
                    nid_start=nid_start,
                    pid_start=pid_start,
                    pfid_start=pfid_start,
                    batch_size=batch_size,
                    v_stride=0,
                    f_stride=0,
                    offs_stride=0,
                    BS=BS,
                    CBS=CBS,
                    BS_PADDED=triton.next_power_of_2(BS),
                    MAX_K_IN=max_k_in,
                    IS_BATCHED=False,
                    ALLOW_MODIFY_FLOWS=1 if allow_modify_flows else 0,
                    COMPUTE_PFLOWS=1 if compute_pflows else 0,
                    NEGATE_PFLOWS=1 if negate_pflows else 0,
                )
            # Consume-and-clear to surface any stale-pointer bug.
            self._cached_in_offsets[blk_idx] = None

        return None


# ======================================================================= #
# Triton kernels
# ======================================================================= #


@triton.jit(
    do_not_specialize=["nid_start", "pid_start", "batch_size", "BS", "CBS",
                       "v_stride", "offs_stride"],
    do_not_specialize_on_alignment=[
        "in_indices_ptr", "in_values_ptr", "in_block_offsets_ptr",
        "max_val_ptr", "col_starts_ptr",
    ],
)
def _fw_sparse_in_bd_kernel(
    node_mars_ptr, mparams_ptr,
    in_indices_ptr, in_values_ptr, in_block_offsets_ptr, max_val_ptr,
    col_starts_ptr,
    nid_start, pid_start,
    batch_size,
    v_stride, offs_stride,
    BS, CBS,
    BS_PADDED: tl.constexpr,
    MAX_K_IN: tl.constexpr,
    IS_BATCHED: tl.constexpr,
):
    """Sparse-in / dense-out forward for one parent block. Grid
    ``(NB, B)`` — one program per (parent_block, sample); axis 1 is
    degenerate at B=1.

    Per-program math (block ``pid_nb``, B=1 form):

      [in_s, in_e) := in_block_offsets[pid_nb : pid_nb+2]     # K_in_j slots
      cslot[k] := in_indices[in_s + k] - pid_nb * CBS          # [0, CBS)
      W[s, k] := mparams[pid_start + pid_nb*BS*CBS + cslot[k]*BS + s]
      acc[s] := Σ_k W[s, k] · exp(in_values[k] - max_val)
      node_mars[nid_start + pid_nb*BS + s, 0]
          := log(acc[s] + 1e-24) + max_val    (exact -inf if K_in_j == 0)

    ``IS_BATCHED == 1``: sample ``b`` reads its own table row
    (``in_block_offsets[b*offs_stride + ...]`` — offsets are relative to
    the sample's column, rebased on ``col_starts[b]`` for the index
    gather), its own values row (``in_values[b*v_stride + in_s + k]``) and
    its own fused ``max_val[b]`` (clamped to ``-1e30`` so an empty column
    cannot poison the exp); the store lands at ``node_mars[nid, b]``.

    ``BS`` / ``CBS`` are runtime ints (one compiled binary serves every
    tied/untied block); register pressure is bounded by
    ``BS_PADDED × MAX_K_IN``, both powers of two — the same scale as the
    dense BD forward tile.
    """
    pid_nb = tl.program_id(0)
    pid_b = tl.program_id(1)

    if IS_BATCHED:
        in_s = tl.load(in_block_offsets_ptr + pid_b * offs_stride + pid_nb)
        in_e = tl.load(in_block_offsets_ptr + pid_b * offs_stride + pid_nb + 1)
        cs_b = tl.load(col_starts_ptr + pid_b)
        idx_base = cs_b + in_s
        val_base = pid_b * v_stride + in_s
        max_val = tl.load(max_val_ptr + pid_b)
        max_val = tl.maximum(max_val, -1e30)                    # empty col guard
    else:
        in_s = tl.load(in_block_offsets_ptr + pid_nb)
        in_e = tl.load(in_block_offsets_ptr + pid_nb + 1)
        idx_base = in_s
        val_base = in_s
        max_val = tl.load(max_val_ptr)
    K_in_j = in_e - in_s

    offs_k = tl.arange(0, MAX_K_IN)
    mask_k = offs_k < K_in_j

    offs_s = tl.arange(0, BS_PADDED)
    mask_s = offs_s < BS

    in_idx = tl.load(in_indices_ptr + idx_base + offs_k, mask=mask_k, other=0)
    in_val = tl.load(in_values_ptr + val_base + offs_k, mask=mask_k,
                     other=-float("inf"))
    cslot = in_idx - pid_nb * CBS                                                # [MAX_K_IN]

    in_val_lin = tl.where(mask_k, tl.exp(in_val - max_val), 0.0)

    # int64 cast on pid_nb: worst-case range analysis on ``pid_nb*BS*CBS``
    # can trip Triton's int32 overflow check at large NB — same fix as the
    # dense BD kernels.
    block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
    weight_addr = (
        block_base
        + cslot[None, :] * BS
        + offs_s[:, None]
    )                                                                            # [BS_PADDED, MAX_K_IN]
    weight = tl.load(
        mparams_ptr + weight_addr,
        mask=mask_s[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(weight * in_val_lin[None, :], axis=1)                           # [BS_PADDED]

    # ``K_in_j == 0`` ⇒ acc is 0 ⇒ log(1e-24) + max_val is a large negative
    # number but the right answer is exact -inf (empty sum).
    result = tl.where(
        K_in_j == 0,
        -float("inf"),
        tl.log(acc + 1e-24) + max_val,
    )

    out_ptr = (nid_start + pid_nb * BS + offs_s).to(tl.int64) * batch_size \
        + pid_b
    tl.store(node_mars_ptr + out_ptr, result, mask=mask_s)


@triton.jit(
    do_not_specialize=["nid_start", "pid_start", "pfid_start", "batch_size",
                       "BS", "CBS", "v_stride", "f_stride", "offs_stride"],
    do_not_specialize_on_alignment=[
        "flow_in_ptr", "in_indices_ptr", "in_values_ptr",
        "in_block_offsets_ptr", "pflows_ptr", "col_starts_ptr",
    ],
)
def _bk_sparse_in_bd_kernel(
    flow_in_ptr,
    node_flows_ptr, node_mars_ptr,
    mparams_ptr,
    pflows_ptr,
    in_indices_ptr, in_values_ptr, in_block_offsets_ptr,
    col_starts_ptr,
    nid_start, pid_start, pfid_start,
    batch_size,
    v_stride, f_stride, offs_stride,
    BS, CBS,
    BS_PADDED: tl.constexpr,
    MAX_K_IN: tl.constexpr,
    IS_BATCHED: tl.constexpr,
    ALLOW_MODIFY_FLOWS: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr,
    NEGATE_PFLOWS: tl.constexpr,
):
    """Element-flow + (optional) param-flow backward for one parent block.
    Grid ``(NB, B)`` — one program per (parent_block, sample); axis 1 is
    degenerate at B=1.

    Per-program math (block ``pid_nb``, dense parent side / sparse child
    side, B=1 form):

      cslot[k] := in_indices[in_s + k] - pid_nb * CBS
      W[s, k] := mparams[pid_start + pid_nb*BS*CBS + cslot[k]*BS + s]
      contrib[s, k] := nflow[s] · W[s, k] · exp(in_values[k] - nmars[s])
      flow_in[in_s + k] := Σ_s contrib[s, k]        (plain store — each
                            active slot belongs to exactly one parent block)
      # When COMPUTE_PFLOWS == 1:
      param_flows[pfid_start + pid_nb*BS*CBS + cslot[k]*BS + s] += contrib[s, k]

    With ``ALLOW_MODIFY_FLOWS == 1``, ``nflow[s]`` was pre-transformed to
    ``log(flow) - nmars`` (see ``_run_modify_flows_prepass``) and the
    contribution is rewritten to ``exp(nflow[s] + in_values[k]) · W[s, k]``.

    ``IS_BATCHED == 1``: sample ``b`` reads its own table row / values row
    / parent column (same rebasing as the forward kernel) and stores its
    flow at ``flow_in[b*f_stride + in_s + k]`` — still a plain store, each
    ``(sample, slot)`` pair is owned by exactly one program.

    Param-flow scatter uses ``atomic_add``: within one launch at B=1 each
    address is touched once (zero contention), tied SumNodes across kernel
    calls accumulate into the same ``pfid`` slice, and at B>1 programs of
    *different samples* hit the same addresses — the atomic IS the batch
    reduction (EM pflows sum over samples), same contract as the batched
    SparseInputSumLayer kernel.
    """
    pid_nb = tl.program_id(0)
    pid_b = tl.program_id(1)

    if IS_BATCHED:
        in_s = tl.load(in_block_offsets_ptr + pid_b * offs_stride + pid_nb)
        in_e = tl.load(in_block_offsets_ptr + pid_b * offs_stride + pid_nb + 1)
        cs_b = tl.load(col_starts_ptr + pid_b)
        idx_base = cs_b + in_s
        val_base = pid_b * v_stride + in_s
        flow_base = pid_b * f_stride + in_s
    else:
        in_s = tl.load(in_block_offsets_ptr + pid_nb)
        in_e = tl.load(in_block_offsets_ptr + pid_nb + 1)
        idx_base = in_s
        val_base = in_s
        flow_base = in_s
    K_in_j = in_e - in_s

    offs_k = tl.arange(0, MAX_K_IN)
    mask_k = offs_k < K_in_j

    offs_s = tl.arange(0, BS_PADDED)
    mask_s = offs_s < BS

    in_idx = tl.load(in_indices_ptr + idx_base + offs_k, mask=mask_k, other=0)
    in_val = tl.load(in_values_ptr + val_base + offs_k, mask=mask_k,
                     other=-float("inf"))
    cslot = in_idx - pid_nb * CBS

    p_addr = (nid_start + pid_nb * BS + offs_s).to(tl.int64) * batch_size \
        + pid_b                                                                  # [BS_PADDED]

    block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
    weight_addr = (
        block_base
        + cslot[None, :] * BS
        + offs_s[:, None]
    )                                                                            # [BS_PADDED, MAX_K_IN]
    weight = tl.load(
        mparams_ptr + weight_addr,
        mask=mask_s[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)

    if ALLOW_MODIFY_FLOWS == 1:
        # ``nflows`` is already ``log(flow) - nmars`` (-inf where the flow
        # is 0 or nmars is -inf) ⇒ exp(-inf + anything) = 0, no NaN paths.
        nflows = tl.load(node_flows_ptr + p_addr, mask=mask_s,
                         other=-float("inf"))
        contrib = tl.exp(nflows[:, None] + in_val[None, :]) * weight
    else:
        nflows = tl.load(node_flows_ptr + p_addr, mask=mask_s, other=0.0)
        nmars = tl.load(node_mars_ptr + p_addr, mask=mask_s,
                        other=-float("inf"))
        # nmars == -inf (empty block ⇒ zero flow) or in_val == -inf (padded
        # slot) would produce inf/NaN through the exp — force those terms
        # to exact 0 instead.
        contrib_factor = tl.where(
            (nmars[:, None] == -float("inf"))
            | (in_val[None, :] == -float("inf")),
            0.0,
            tl.exp(in_val[None, :] - nmars[:, None]),
        )
        contrib = nflows[:, None] * weight * contrib_factor

    contrib = tl.where(mask_s[:, None] & mask_k[None, :], contrib, 0.0)

    flow_k = tl.sum(contrib, axis=0)                                             # [MAX_K_IN]
    tl.store(flow_in_ptr + flow_base + offs_k, flow_k, mask=mask_k)

    if COMPUTE_PFLOWS == 1:
        pf_base = pfid_start + pid_nb.to(tl.int64) * BS * CBS
        pf_addr = (
            pf_base
            + cslot[None, :] * BS
            + offs_s[:, None]
        )                                                                        # [BS_PADDED, MAX_K_IN]
        pf_val = -contrib if NEGATE_PFLOWS == 1 else contrib
        tl.atomic_add(
            pflows_ptr + pf_addr,
            pf_val,
            mask=mask_s[:, None] & mask_k[None, :],
        )
