from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import SumNodes
from pyjuice.utils.kernel_launcher import triton_jit
from .block_diagonal_sum_layer import BlockDiagonalSumLayer
from .sparse_node_values import SparseNodeValues
from .sparse_prod_layer import SparseProdLayer


def _build_block_offsets_table(dist, NB: int, BS: int) -> Tuple[torch.Tensor, int]:
    """Precompute ``[V, NB+1]`` int32 block-offset table for a
    :class:`SparseCategorical` dist.

    For each CSC column ``v``, ``table[v, b]`` is the position WITHIN
    ``csc_indices[indptr[v]:indptr[v+1]]`` where block ``b`` starts —
    equivalent to ``searchsorted(col_indices, b * BS)``.

    Done once at layer compile time so per-call forward / backward is a
    plain ``table[v]`` device gather (no ``searchsorted``, no host sync).

    Also returns ``max_k_per_block`` (the worst per-block nnz across the
    entire table), used by the caller to size the kernel's MAX_K_IN /
    MAX_K_OUT constexpr.
    """
    assert dist._csc_indptr is not None and dist._csc_indices is not None, (
        "SparseCategorical dist must have CSC pattern set "
        "(call set_meta_parameters / set_params first)."
    )
    indptr = dist._csc_indptr.cpu()
    indices = dist._csc_indices.cpu()
    V = indptr.numel() - 1
    boundary = torch.arange(NB + 1, dtype=indices.dtype) * BS
    table = torch.empty(V, NB + 1, dtype=torch.int32)
    max_k_per_block = 0
    for v in range(V):
        col_s = int(indptr[v])
        col_e = int(indptr[v + 1])
        col = indices[col_s:col_e]
        offs = torch.searchsorted(col, boundary)
        table[v] = offs.to(torch.int32)
        if col_e > col_s:
            block_sizes = offs[1:] - offs[:-1]
            cur = int(block_sizes.max().item())
            if cur > max_k_per_block:
                max_k_per_block = cur
    return table, max_k_per_block


class SparseIOBlockDiagonalSumLayer(BlockDiagonalSumLayer):
    """Sparse-in, sparse-out variant of :class:`BlockDiagonalSumLayer`.

    Sits in the same chain slot as :class:`SparseIOSumLayer`, but uses the
    block-diagonal parameter layout (one ``[bs, cbs]`` block per parent
    block, ``NB`` blocks total, no inter-block edges). Because every input
    in block ``j`` only feeds outputs in block ``j``, the per-block work
    is ``K_in_j × K_out_j`` MACs — strictly smaller than the dense
    sparse-IO kernel's ``K_in × K_out`` whenever ``NB > 1``.

    Forward / backward read upstream :class:`SparseNodeValues` (from the
    feeding :class:`CoSparseProdLayer`) and write a new
    :class:`SparseNodeValues` for the downstream :class:`CoSparseProdLayer`
    to consume. ``node_mars`` is **not** written.

    Constraints (inherited or asserted):
      * Block-diagonal pattern (``NB == NB_ch``, ``bs == cbs``,
        ``edge_ids = arange(NB)[None, :].repeat(2, 1)``).
      * Single child group per sum node.
      * ``batch_size == 1`` (matches the rest of the sparse fast path).
      * ``propagation_alg == 'LL'``; no param-flow accumulation
        (inference-only — same scope as :class:`SparseIOSumLayer`).

    Block-offsets contract:
      Both ``sv_in.indices`` and ``sv_out.indices`` come from CSC columns
      and are therefore **already sorted ascending**. We exploit that to
      compute per-block start offsets in O(NB) per call via
      :func:`torch.searchsorted` — no host-side sort, no preprocessing
      kernel.
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

        assert inner_layer_groups is not None, (
            "SparseIOBlockDiagonalSumLayer needs the already-compiled "
            "inner_layer_groups to resolve the upstream SparseProdLayer "
            "that owns each sum's child."
        )

        output_sparsity_dists = kwargs.pop("output_sparsity_dists", None)
        output_sparsity_num_rows = kwargs.pop("output_sparsity_num_rows", None)
        for name, val in (
            ("output_sparsity_var_ids", output_sparsity_var_ids),
            ("output_sparsity_dists", output_sparsity_dists),
            ("output_sparsity_num_rows", output_sparsity_num_rows),
        ):
            assert val is not None and len(val) == len(self.nodes), (
                f"SparseIOBlockDiagonalSumLayer requires {name} (one per "
                "sum node)."
            )
        self._output_sparsity_var_ids: List[int] = list(output_sparsity_var_ids)
        self._output_sparsity_dists = list(output_sparsity_dists)
        self._output_sparsity_num_rows: List[int] = list(output_sparsity_num_rows)

        # Resolve the upstream SparseProdLayer that produced each sum's
        # child SparseProdNodes — same lookup as
        # :meth:`SparseInputSumLayer._build_sparse_input_refs`. We also
        # snapshot the input-side ``(dist, var_id)`` so the per-call
        # block-offsets lookup at forward time is a plain tensor gather.
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
                "SparseIOBlockDiagonalSumLayer: child ProdNodes is not "
                "owned by any SparseProdLayer in the compiled "
                "inner_layer_groups."
            )
            self._sparse_input_refs.append(found)
            sparse_prod_layer, prod_ns_idx = found
            prod_ns = sparse_prod_layer.nodes[prod_ns_idx]
            # ``sparse_input_ns`` is the SparseCategorical input child of
            # the prod ns (see :class:`SparseProdNodes`). Its CSC pattern
            # determines ``sv_in.indices`` at every forward.
            self._input_sparsity_dists.append(prod_ns.sparse_input_ns.dist)
            self._input_sparsity_var_ids.append(prod_ns.var_id)

        # Forward-cached sv_out per ns (read by downstream CoSparseProdLayer).
        self._sparse_outputs: List[Optional[SparseNodeValues]] = [None] * len(self.nodes)
        # Backward flow container written by downstream CoSparseProdLayer.
        self._sparse_flows: List[Optional[SparseNodeValues]] = [None] * len(self.nodes)
        # Per-ns block-offset cache populated at forward time so backward
        # reuses the same partition (avoids re-looking-up missing-mask
        # flags / re-gathering the precomputed tables).
        self._cached_in_offsets: List[Optional[torch.Tensor]] = [None] * len(self.nodes)
        self._cached_out_offsets: List[Optional[torch.Tensor]] = [None] * len(self.nodes)

        # Per-block GPU workspaces — same memory-reuse pattern as
        # :class:`SparseIOSumLayer`. Sized lazily on first use.
        self._fwd_values_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)
        self._bwd_flow_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)

        # Lazily-allocated arange indices used for the missing-mask
        # "all-rows" sv_out path. Keyed by ``out_num_rows``.
        self._missing_indices_cache: dict = {}

        # ----- Compile-time per-column block-offsets tables --------------- #
        # For each ns and each CSC column ``v`` of its input / output dist,
        # precompute the ``[NB + 1]`` int32 offsets into the column slice:
        #   table[v, b] = searchsorted(csc_indices[indptr[v]:indptr[v+1]], b*BS_or_CBS)
        # i.e. the position WITHIN sv_in.indices (or sv_out.indices) where
        # block ``b`` starts. The CSC pattern is fixed at compile time so
        # this only depends on ``(dist, BS_or_CBS, NB)``; at forward time
        # we resolve the active column ``v`` from the data tensor and gather
        # ``table[v]`` with a single device-side ``index_select`` (no
        # ``searchsorted`` / ``.item()`` / host sync per call).
        #
        # ``MAX_K_IN`` / ``MAX_K_OUT`` are picked as the worst-case
        # per-block nnz across **all** CSC columns plus the missing-mask
        # all-rows case (``K_j == CBS`` / ``K_j == BS``), padded to the
        # next power of two. Compiling for the global max means a single
        # Triton specialisation serves every (token, missing-mask)
        # combination at runtime.
        self._in_block_offsets_tables: List[Optional[torch.Tensor]] = []
        self._out_block_offsets_tables: List[Optional[torch.Tensor]] = []
        max_k_in_across_ns = 0
        max_k_out_across_ns = 0
        for blk_idx, (block, in_dist, in_var) in enumerate(zip(
            self._bd_blocks, self._input_sparsity_dists,
            self._input_sparsity_var_ids,
        )):
            _, _, _, _, NB, BS, CBS = block
            in_table, max_k_in = _build_block_offsets_table(in_dist, NB, CBS)
            out_table, max_k_out = _build_block_offsets_table(
                self._output_sparsity_dists[blk_idx], NB, BS,
            )
            self._in_block_offsets_tables.append(in_table)
            self._out_block_offsets_tables.append(out_table)
            # Missing-mask case = arange(num_rows) ⇒ K_j == block_size.
            max_k_in_across_ns = max(max_k_in_across_ns, max_k_in, CBS)
            max_k_out_across_ns = max(max_k_out_across_ns, max_k_out, BS)

        self._MAX_K_IN: int = max(triton.next_power_of_2(max_k_in_across_ns), 1)
        self._MAX_K_OUT: int = max(triton.next_power_of_2(max_k_out_across_ns), 1)

        # Reusable missing-case offsets ``arange(NB+1) * BS_or_CBS`` —
        # built once on the same device as the tables, indexed by NB.
        # (NB / BS / CBS are uniform across this layer's blocks for typical
        # Monarch / HMM construction, but we key by ns to be safe.)
        self._missing_in_offsets: List[Optional[torch.Tensor]] = [None] * len(self.nodes)
        self._missing_out_offsets: List[Optional[torch.Tensor]] = [None] * len(self.nodes)
        for blk_idx, block in enumerate(self._bd_blocks):
            _, _, _, _, NB, BS, CBS = block
            in_table = self._in_block_offsets_tables[blk_idx]
            device = in_table.device if in_table is not None else torch.device("cpu")
            self._missing_in_offsets[blk_idx] = (
                torch.arange(NB + 1, dtype=torch.int32, device=device) * CBS
            )
            self._missing_out_offsets[blk_idx] = (
                torch.arange(NB + 1, dtype=torch.int32, device=device) * BS
            )

    def __repr__(self) -> str:
        return (
            f"SparseIOBlockDiagonalSumLayer("
            f"nid_range=({self._layer_nid_range[0]}, {self._layer_nid_range[1]}), "
            f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"num_sum_ns={len(self._sparse_input_refs)}, "
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
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        batch_size = node_mars.size(1)
        assert batch_size == 1, "SparseIOBlockDiagonalSumLayer is B=1 only."
        assert propagation_alg == "LL", (
            "SparseIOBlockDiagonalSumLayer requires propagation_alg == 'LL'."
        )
        assert params.dim() == 1
        assert data is not None, (
            "SparseIOBlockDiagonalSumLayer.forward requires `data` to build "
            "the output-sparsity pattern for each ns."
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
                        "SparseIOBlockDiagonalSumLayer.forward got a 2D "
                        "missing_mask with neither dim == 1."
                    )
            missing_mask_cpu = mm.cpu() if mm.device.type != "cpu" else mm

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._bd_blocks, self._sparse_input_refs,
        )):
            nid_start, cid_start, pid_start, _pfid_start, NB, BS, CBS = block
            sv_in = sparse_prod._sparse_outputs[ns_idx]
            K_in = sv_in.total_nnz

            out_dist = self._output_sparsity_dists[blk_idx]
            out_var_id = self._output_sparsity_var_ids[blk_idx]
            out_num_rows = self._output_sparsity_num_rows[blk_idx]
            in_var_id = self._input_sparsity_var_ids[blk_idx]
            out_is_missing = bool(missing_mask_cpu[out_var_id].item()) \
                if missing_mask_cpu is not None else False
            in_is_missing = bool(missing_mask_cpu[in_var_id].item()) \
                if missing_mask_cpu is not None else False

            # --- Build sv_out (same pattern as SparseIOSumLayer) -------- #
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
                        out_num_rows, dtype=torch.float32, device=node_mars.device,
                    )
                    self._fwd_values_workspaces[blk_idx] = ws_out
                values = ws_out.narrow(0, 0, out_num_rows)
                sv_out = SparseNodeValues(
                    col_start=0, total_nnz=out_num_rows,
                    indices=indices, values=values, num_rows=out_num_rows,
                )
            else:
                ws_out = self._fwd_values_workspaces[blk_idx]
                needed = max(out_dist._max_nnz_per_col, out_num_rows)
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
                )

            self._sparse_outputs[blk_idx] = sv_out
            K_out = sv_out.total_nnz

            if K_out == 0:
                continue

            if K_in == 0:
                sv_out.values.fill_(float("-inf"))
                continue

            # --- Per-block offsets via precomputed table ----------------- #
            in_offs = self._lookup_block_offsets(
                blk_idx, side="in", var_id=in_var_id,
                is_missing=in_is_missing, data=data,
                data_list=data_list, device=node_mars.device,
            )
            out_offs = self._lookup_block_offsets(
                blk_idx, side="out", var_id=out_var_id,
                is_missing=out_is_missing, data=data,
                data_list=data_list, device=node_mars.device,
            )
            # Cache for backward — re-deriving these from missing flags
            # would require threading ``missing_mask`` into ``backward``,
            # which the standard sum-layer backward signature doesn't
            # carry. The cached pointers stay valid because the upstream
            # forward → downstream forward → downstream backward → upstream
            # backward dataflow is strictly serial per pass.
            self._cached_in_offsets[blk_idx] = in_offs
            self._cached_out_offsets[blk_idx] = out_offs

            max_val = sv_in.max_val if sv_in.max_val is not None else sv_in.values.max()

            grid = (NB,)
            _fw_bd_sparse_io_kernel[grid](
                out_values_ptr=sv_out.values,
                mparams_ptr=params,
                in_indices_ptr=sv_in.indices,
                in_values_ptr=sv_in.values,
                in_block_offsets_ptr=in_offs,
                max_val_ptr=max_val,
                out_indices_ptr=sv_out.indices,
                out_block_offsets_ptr=out_offs,
                pid_start=pid_start,
                BS=BS,
                CBS=CBS,
                MAX_K_IN=self._MAX_K_IN,
                MAX_K_OUT=self._MAX_K_OUT,
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
        batch_size = node_mars.size(1)
        assert batch_size == 1, "SparseIOBlockDiagonalSumLayer is B=1 only."
        assert propagation_alg == "LL" and not logspace_flows \
               and not allow_neg_flows, (
            "SparseIOBlockDiagonalSumLayer.backward requires "
            "propagation_alg='LL', logspace_flows=False, "
            "allow_neg_flows=False."
        )
        assert not accumulate_ch_flows, (
            "SparseIOBlockDiagonalSumLayer writes sv_flow_in straight into "
            "the upstream prod layer; accumulate_ch_flows is not supported."
        )

        compute_pflows = param_flows is not None

        for blk_idx, (block, (sparse_prod, ns_idx)) in enumerate(zip(
            self._bd_blocks, self._sparse_input_refs,
        )):
            _nid_start, _cid_start, pid_start, pfid_start, NB, BS, CBS = block

            sv_flow_out = self._sparse_flows[blk_idx]
            assert sv_flow_out is not None, (
                "SparseIOBlockDiagonalSumLayer.backward expected "
                "sv_flow_out from the downstream CoSparseProdLayer."
            )
            self._sparse_flows[blk_idx] = None  # consume-and-clear

            sv_in = sparse_prod._sparse_outputs[ns_idx]
            sv_out = self._sparse_outputs[blk_idx]
            K_in = sv_in.total_nnz
            K_out = sv_out.total_nnz

            # Allocate flow_in workspace mirroring sv_in's pattern.
            ws_flow = self._bwd_flow_workspaces[blk_idx]
            in_dist = sparse_prod.nodes[ns_idx].sparse_input_ns.dist
            in_max_nnz = max(in_dist._max_nnz_per_col, in_dist._num_nodes)
            if (ws_flow is None or ws_flow.device != node_mars.device
                    or ws_flow.numel() < max(K_in, in_max_nnz)):
                ws_flow = torch.empty(
                    max(K_in, in_max_nnz),
                    dtype=torch.float32, device=node_mars.device,
                )
                self._bwd_flow_workspaces[blk_idx] = ws_flow
            sv_flow_in = SparseNodeValues(
                col_start=sv_in.col_start, total_nnz=K_in,
                indices=sv_in.indices,
                values=ws_flow.narrow(0, 0, K_in),
                num_rows=sv_in.num_rows,
            )
            sparse_prod._sparse_flows[ns_idx] = sv_flow_in

            if K_in == 0 or K_out == 0:
                if K_in > 0:
                    sv_flow_in.values.fill_(0.0)
                # Clear the cached offsets for this pass — keeps the cache
                # honest if a later forward happens to skip the kernel.
                self._cached_in_offsets[blk_idx] = None
                self._cached_out_offsets[blk_idx] = None
                continue

            in_offs = self._cached_in_offsets[blk_idx]
            out_offs = self._cached_out_offsets[blk_idx]
            assert in_offs is not None and out_offs is not None, (
                "SparseIOBlockDiagonalSumLayer.backward expected cached "
                "block offsets from forward; call forward before backward."
            )

            grid = (NB,)
            _bk_bd_sparse_io_kernel[grid](
                flow_in_ptr=sv_flow_in.values,
                sv_flow_out_ptr=sv_flow_out.values,
                sv_out_values_ptr=sv_out.values,
                mparams_ptr=params,
                pflows_ptr=(param_flows if compute_pflows else sv_flow_in.values),
                in_indices_ptr=sv_in.indices,
                in_values_ptr=sv_in.values,
                in_block_offsets_ptr=in_offs,
                out_indices_ptr=sv_out.indices,
                out_block_offsets_ptr=out_offs,
                pid_start=pid_start,
                pfid_start=pfid_start,
                BS=BS,
                CBS=CBS,
                MAX_K_IN=self._MAX_K_IN,
                MAX_K_OUT=self._MAX_K_OUT,
                COMPUTE_PFLOWS=1 if compute_pflows else 0,
                NEGATE_PFLOWS=1 if negate_pflows else 0,
            )
            # Consume-and-clear the cache to surface any stale-pointer bug.
            self._cached_in_offsets[blk_idx] = None
            self._cached_out_offsets[blk_idx] = None

        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _lookup_block_offsets(self, blk_idx: int, side: str, var_id: int,
                                is_missing: bool, data: torch.Tensor,
                                data_list: Optional[list],
                                device: torch.device) -> torch.Tensor:
        """Look up per-block start offsets via the compile-time table.

        Two paths:
        * Missing position (``is_missing=True``): the upstream / consumer
          ``build_sparse_pattern`` emits an all-rows sv whose indices are
          ``arange(num_rows)``. Block ``b`` starts at ``b * BS`` (resp.
          ``b * CBS``). Returns the precomputed missing-case tensor.
        * Active position: index ``table[v]`` where ``v`` is the observed
          token. If ``data_list`` (Python-list cache of ``data_cpu[:, 0]``)
          is available, ``v`` is a Python int and the gather is a plain
          view — zero CUDA dispatch. Otherwise we ``index_select`` with a
          1-D GPU slice to avoid the 0-d-CUDA-index sync.

        ``side`` selects between input (``"in"``, uses CBS-aligned table)
        and output (``"out"``, uses BS-aligned table).
        """
        if side == "in":
            table = self._in_block_offsets_tables[blk_idx]
            missing_offs = self._missing_in_offsets[blk_idx]
        else:
            table = self._out_block_offsets_tables[blk_idx]
            missing_offs = self._missing_out_offsets[blk_idx]

        # Lazy device migration — the layer's ``.to(device)`` doesn't see
        # these tensors (they're not registered buffers, just plain
        # attributes on a list). Cheap one-time copy on first forward; the
        # tables are tiny (``V × (NB+1)`` int32, typically <1 MB).
        if table.device != device:
            table = table.to(device)
            if side == "in":
                self._in_block_offsets_tables[blk_idx] = table
            else:
                self._out_block_offsets_tables[blk_idx] = table
        if missing_offs.device != device:
            missing_offs = missing_offs.to(device)
            if side == "in":
                self._missing_in_offsets[blk_idx] = missing_offs
            else:
                self._missing_out_offsets[blk_idx] = missing_offs

        if is_missing:
            return missing_offs
        # Sync-free index. ``table[scalar]`` with a 0-d CUDA index forces
        # a host-device sync (pytorch needs the scalar value to determine
        # the output shape). Two faster paths:
        #   * ``data_list`` available ⇒ ``v`` is a Python int and
        #     ``table[v]`` is a plain row view (no kernel, no dispatch).
        #   * Else fall back to ``index_select`` with a 1-D GPU slice —
        #     also sync-free but does a launch.
        if data_list is not None:
            return table[data_list[var_id]]
        v_1d = data[var_id:var_id + 1, 0]
        return table.index_select(0, v_1d).squeeze(0)


# ======================================================================= #
# Triton kernels
# ======================================================================= #


@triton.jit(
    do_not_specialize=["pid_start"],
    do_not_specialize_on_alignment=[
        "in_indices_ptr", "in_values_ptr", "max_val_ptr",
        "in_block_offsets_ptr", "out_indices_ptr",
        "out_block_offsets_ptr", "out_values_ptr",
    ],
)
def _fw_bd_sparse_io_kernel(
    out_values_ptr,
    mparams_ptr,
    in_indices_ptr, in_values_ptr, in_block_offsets_ptr, max_val_ptr,
    out_indices_ptr, out_block_offsets_ptr,
    pid_start,
    BS: tl.constexpr, CBS: tl.constexpr,
    MAX_K_IN: tl.constexpr, MAX_K_OUT: tl.constexpr,
):
    """Forward for one parent block of a SparseIOBlockDiagonalSumLayer.

    Grid = ``(NB,)``: program ``pid_nb`` handles block ``pid_nb`` only,
    independently of every other block (block-diagonal structure ⇒ no
    cross-block reads or writes).

    Per-program math:

      [in_s, in_e) := in_block_offsets[pid_nb : pid_nb+2]   # K_in_j slots
      [out_s, out_e) := out_block_offsets[pid_nb : pid_nb+2]
      cslot[k] := in_indices[in_s + k] - pid_nb * CBS       # [0, CBS)
      pslot[m] := out_indices[out_s + m] - pid_nb * BS      # [0, BS)
      W[m, k] := mparams[pid_start + pid_nb*BS*CBS + cslot[k]*BS + pslot[m]]
      acc[m] := Σ_k W[m, k] · exp(in_values[k] - max_val)
      out_values[out_s + m] := log(acc[m] + 1e-32) + max_val

    Tiling: K_in_j is padded to ``MAX_K_IN``, K_out_j to ``MAX_K_OUT``
    (host computes the per-call max-over-blocks before launch). The
    weight tile lives in registers at ``[MAX_K_OUT, MAX_K_IN]``.
    """
    pid_nb = tl.program_id(0)

    in_s = tl.load(in_block_offsets_ptr + pid_nb)
    in_e = tl.load(in_block_offsets_ptr + pid_nb + 1)
    K_in_j = in_e - in_s

    out_s = tl.load(out_block_offsets_ptr + pid_nb)
    out_e = tl.load(out_block_offsets_ptr + pid_nb + 1)
    K_out_j = out_e - out_s

    offs_m = tl.arange(0, MAX_K_OUT)
    mask_m = offs_m < K_out_j

    # If this block has no active output slots, nothing to write. We must
    # still execute the matching ``tl.store`` shape (Triton has no
    # per-program early return), but ``mask_m`` is all-False below so the
    # store is a no-op.
    offs_k = tl.arange(0, MAX_K_IN)
    mask_k = offs_k < K_in_j

    in_idx = tl.load(in_indices_ptr + in_s + offs_k, mask=mask_k, other=0)
    in_val = tl.load(in_values_ptr + in_s + offs_k, mask=mask_k,
                     other=-float("inf"))
    cslot = in_idx - pid_nb * CBS                                                # [MAX_K_IN]

    out_idx = tl.load(out_indices_ptr + out_s + offs_m, mask=mask_m, other=0)
    pslot = out_idx - pid_nb * BS                                                # [MAX_K_OUT]

    max_val = tl.load(max_val_ptr)
    in_val_lin = tl.where(mask_k, tl.exp(in_val - max_val), 0.0)

    # Weight gather: W[m, k] = mparams[block_base + cslot[k]*BS + pslot[m]].
    # int64 cast on pid_nb mirrors the same fix in BD's forward kernel —
    # large NB·BS·CBS can trip Triton's int32 worst-case range analysis.
    block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
    weight_addr = (
        block_base
        + cslot[None, :] * BS
        + pslot[:, None]
    )                                                                            # [MAX_K_OUT, MAX_K_IN]
    weight = tl.load(
        mparams_ptr + weight_addr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(weight * in_val_lin[None, :], axis=1)                           # [MAX_K_OUT]

    # ``K_in_j == 0`` ⇒ acc is 0 everywhere ⇒ log(1e-32) + max_val is a
    # large negative number, which is the wrong answer (should be -inf).
    # Gate the store to -inf for the empty-input case.
    result = tl.where(
        K_in_j == 0,
        -float("inf"),
        tl.log(acc + 1e-32) + max_val,
    )
    tl.store(out_values_ptr + out_s + offs_m, result, mask=mask_m)


@triton.jit(
    do_not_specialize=["pid_start", "pfid_start"],
    do_not_specialize_on_alignment=[
        "flow_in_ptr", "sv_flow_out_ptr", "sv_out_values_ptr",
        "in_indices_ptr", "in_values_ptr", "in_block_offsets_ptr",
        "out_indices_ptr", "out_block_offsets_ptr",
        "pflows_ptr",
    ],
)
def _bk_bd_sparse_io_kernel(
    flow_in_ptr,
    sv_flow_out_ptr, sv_out_values_ptr,
    mparams_ptr,
    pflows_ptr,
    in_indices_ptr, in_values_ptr, in_block_offsets_ptr,
    out_indices_ptr, out_block_offsets_ptr,
    pid_start, pfid_start,
    BS: tl.constexpr, CBS: tl.constexpr,
    MAX_K_IN: tl.constexpr, MAX_K_OUT: tl.constexpr,
    COMPUTE_PFLOWS: tl.constexpr, NEGATE_PFLOWS: tl.constexpr,
):
    """Element-flow + (optional) param-flow backward for one parent block.

    Grid = ``(NB,)``. Per-program math (one parent block ``pid_nb``):

      cslot[k] := in_indices[in_s + k] - pid_nb * CBS
      pslot[m] := out_indices[out_s + m] - pid_nb * BS
      W[m, k] := mparams[pid_start + pid_nb*BS*CBS + cslot[k]*BS + pslot[m]]
      contrib[m, k] := W[m, k] · sv_flow_out[m]
                       · exp(in_values[k] - sv_out_values[m])
      flow_in[in_s + k] := Σ_m contrib[m, k]
      # When COMPUTE_PFLOWS == 1:
      param_flows[pfid_start + pid_nb*BS*CBS + cslot[k]*BS + pslot[m]] += contrib[m, k]

    Element-flow scatter: block-diagonal structure guarantees each
    ``flow_in`` slot is written by exactly one program — plain store.

    Param-flow scatter: programs in this launch write to disjoint blocks
    (``pblock``-th block ⇒ slice ``pfid_start + pblock*BS*CBS …``), so
    no intra-launch collisions. But tied SumNodes across blocks (e.g. an
    HMM chain whose every time step ties to the same source's pfid_start)
    accumulate across kernel calls — those calls are sequential on the
    stream so they don't race either, but the semantics of
    ``param_flows`` is accumulation (``flows_memory * existing + new``)
    so we use ``atomic_add`` to read-modify-write atomically. Each address
    is touched by only one (pid_nb, m, k) triple per launch, so atomic
    contention is zero.
    """
    pid_nb = tl.program_id(0)

    in_s = tl.load(in_block_offsets_ptr + pid_nb)
    in_e = tl.load(in_block_offsets_ptr + pid_nb + 1)
    K_in_j = in_e - in_s

    out_s = tl.load(out_block_offsets_ptr + pid_nb)
    out_e = tl.load(out_block_offsets_ptr + pid_nb + 1)
    K_out_j = out_e - out_s

    offs_k = tl.arange(0, MAX_K_IN)
    mask_k = offs_k < K_in_j

    offs_m = tl.arange(0, MAX_K_OUT)
    mask_m = offs_m < K_out_j

    in_idx = tl.load(in_indices_ptr + in_s + offs_k, mask=mask_k, other=0)
    in_val = tl.load(in_values_ptr + in_s + offs_k, mask=mask_k,
                     other=-float("inf"))
    cslot = in_idx - pid_nb * CBS

    out_idx = tl.load(out_indices_ptr + out_s + offs_m, mask=mask_m, other=0)
    pslot = out_idx - pid_nb * BS

    nflows = tl.load(sv_flow_out_ptr + out_s + offs_m, mask=mask_m, other=0.0)
    nmars = tl.load(sv_out_values_ptr + out_s + offs_m, mask=mask_m,
                    other=-float("inf"))

    block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
    weight_addr = (
        block_base
        + cslot[None, :] * BS
        + pslot[:, None]
    )                                                                            # [MAX_K_OUT, MAX_K_IN]
    weight = tl.load(
        mparams_ptr + weight_addr,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)

    # contrib[m, k] = W[m, k] · nflows[m] · exp(in_val[k] - nmars[m])
    # nmars == -inf or padded slot ⇒ zero contribution.
    log_factor = in_val[None, :] - nmars[:, None]                                # [MAX_K_OUT, MAX_K_IN]
    contrib_factor = tl.where(
        (nmars[:, None] == -float("inf")) | (in_val[None, :] == -float("inf")),
        0.0,
        tl.exp(log_factor),
    )
    contrib = weight * nflows[:, None] * contrib_factor
    contrib = tl.where(mask_m[:, None] & mask_k[None, :], contrib, 0.0)

    flow_k = tl.sum(contrib, axis=0)                                             # [MAX_K_IN]

    # K_out_j == 0 ⇒ flow_k stays at 0, which is the desired output for
    # an "empty output" block (no parents to take flow from).
    tl.store(flow_in_ptr + in_s + offs_k, flow_k, mask=mask_k)

    if COMPUTE_PFLOWS == 1:
        # Scatter contrib[m, k] into param_flows. Same address arithmetic
        # as the weight gather but rebased on pfid_start; the BD layout
        # guarantees param_flows and params share the per-block stride
        # ``BS * CBS`` with the same intra-block ``(cslot, pslot)`` order.
        pf_base = pfid_start + pid_nb.to(tl.int64) * BS * CBS
        pf_addr = (
            pf_base
            + cslot[None, :] * BS
            + pslot[:, None]
        )                                                                        # [MAX_K_OUT, MAX_K_IN]
        pf_val = -contrib if NEGATE_PFLOWS == 1 else contrib
        tl.atomic_add(
            pflows_ptr + pf_addr,
            pf_val,
            mask=mask_m[:, None] & mask_k[None, :],
        )
