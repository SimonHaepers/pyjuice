from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.nodes import InputNodes, ProdNodes, SparseProdNodes
from pyjuice.nodes.distributions import SparseCategorical
from .prod_layer import ProdLayer
from .input_layer import InputLayer
from .layer_group import LayerGroup
from .sparse_node_values import SparseNodeValues, LOG_EPS


_FWD_BLOCK_K = 256
"""Slot-axis tile size for :func:`_sparse_prod_forward_kernel` at B=1
(1-D tiles, matches the historical BLOCK=256)."""

_FWD_BATCHED_BLOCK_K = 64
_FWD_BATCHED_BLOCK_B = 8
"""[BLOCK_B, BLOCK_K] tile shape for the batched (B>1) variant of
:func:`_sparse_prod_forward_kernel`. All loads are gathers (per-sample
columns), so the tile shape only trades program count against register
pressure — both knobs are safe to retune."""


class SparseProdLayer(ProdLayer):
    """
    Sparsity-propagating product layer. Each owned ``ns`` is a
    :class:`SparseProdNodes` (the DAG-stage subclass that validated the sparse
    pattern at build time: exactly one :class:`SparseCategorical` input child
    with identity block-sparse edges + one or more non-input children).

    Forward: produces a jagged :class:`SparseNodeValues` output — per-batch
    ``(row, value)`` pairs for rows active in the observed CSC column —
    instead of materialising a dense ``H``-vector. A scatter-to-dense step
    bridges the output to the downstream ``SumLayer``/``DenseSumLayer``
    (which still expect dense ``element_mars``) by filling inactive rows with
    ``LOG_EPS``. When *every* consumer is a :class:`SparseInputSumLayer` the
    scatter is skipped (``_skip_scatter=True``).

    Backward mirrors the sparsity: ``element_flows`` is gathered at the active
    row ids into a new :class:`SparseNodeValues`, then handed to
    :meth:`SparseCategorical.custom_backward_sparse` for direct param-flow
    accumulation. The ``InputLayer`` for the sparse child is gated off via
    ``_skip_input_forward`` / ``_skip_input_backward`` on the ``InputNodes`` so
    no duplicate work happens.

    Forward requires ``data`` (per-variable observed tokens) — pass it through
    as a ``data=...`` kwarg on ``layer_group(...)``.
    """

    def __init__(self, nodes: Sequence[ProdNodes],
                 global_nid_start: Optional[int] = None,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False,
                 input_layer_group: Optional[LayerGroup] = None,
                 **kwargs) -> None:

        super().__init__(
            nodes=nodes,
            global_nid_start=global_nid_start,
            layer_sparsity_tol=layer_sparsity_tol,
            max_num_partitions=max_num_partitions,
            disable_gpu_compilation=disable_gpu_compilation,
            force_gpu_compilation=force_gpu_compilation,
        )

        assert input_layer_group is not None, \
            "SparseProdLayer needs the compiled input_layer_group to resolve " \
            "the InputLayer that owns each SparseCategorical child."
        for ns in self.nodes:
            assert isinstance(ns, SparseProdNodes), (
                f"SparseProdLayer expects SparseProdNodes; got {type(ns).__name__}. "
                "Build via `juice.multiply` (auto-detect) or `juice.sparse_multiply`."
            )

        # Parallel to `self.nodes`: per-ns resolved InputLayer + the name of
        # the dense-child-lookup buffer registered on this module.
        self._sparse_input_layers: List[InputLayer] = []
        self._dense_ch_lookups: List[str] = []
        for ns_idx, ns in enumerate(self.nodes):
            sparse_cs = ns.sparse_input_ns

            sparse_input_layer = None
            for lyr in input_layer_group:
                if sparse_cs in lyr.nodes:
                    sparse_input_layer = lyr
                    break
            assert sparse_input_layer is not None, (
                "SparseProdLayer could not locate the InputLayer holding the "
                "SparseCategorical input child."
            )
            self._sparse_input_layers.append(sparse_input_layer)

            # Per-row lookup for each dense child: the global ``node_mars`` nid
            # for the h-th row of ns's output. Registered as a buffer so the
            # Module's ``.to(device)`` moves it.
            H = ns.num_nodes
            bs = ns.block_size
            h_range = torch.arange(H, dtype=torch.long)
            h_block = h_range // bs
            h_within = h_range % bs
            dense_lookups = []
            for ch_idx in ns.dense_ch_idxs:
                cs = ns.chs[ch_idx]
                eids = ns.edge_ids[:, ch_idx].to(torch.long)
                dense_lookups.append(cs._output_ind_range[0] + eids[h_block] * bs + h_within)
            if dense_lookups:
                dense_lookup = torch.stack(dense_lookups, dim=0)  # [num_dense_chs, H]
            else:
                # num_dense_chs == 0: the forward kernel's static_range over
                # NUM_DENSE_CHS = 0 compiles to no-op, but Triton still wants a
                # valid pointer for ``dense_ch_lookup_ptr``. Register a
                # 1-element stub that never gets dereferenced.
                dense_lookup = torch.zeros(1, dtype=torch.long)
            buf_name = f"_dense_ch_lookup_{ns_idx}"
            self.register_buffer(buf_name, dense_lookup)
            self._dense_ch_lookups.append(buf_name)

            # Gate InputLayer from populating this ns's node_mars / node_flows.
            sparse_cs._skip_input_forward = True
            sparse_cs._skip_input_backward = True

            # Back-reference: lets downstream consumers (e.g. the sparse
            # conditional-query kernel) locate the owning layer's
            # ``_sparse_flows[ns_idx]`` slot from a bare InputNodes ns,
            # without re-walking ``inner_layer_groups``. Same convention as
            # the gate flags above and the ``_param_range`` /
            # ``_output_ind_range`` annotations set elsewhere on ns objects.
            sparse_cs._sparse_flow_owner = (self, ns_idx)

        # One slot per ns — forward populates, backward (sum + prod) reads.
        # Lists rather than dicts because ns_idx is contiguous over
        # ``range(len(self.nodes))`` and the buffers are always fully populated
        # after a forward call.
        self._sparse_outputs: List[Optional[SparseNodeValues]] = \
            [None] * len(self.nodes)
        # Backward companion: sv_flow containers written by the downstream
        # :class:`SparseInputSumLayer` fast path and consumed by this layer's
        # sparse backward. Only populated when ``_skip_scatter`` is True.
        self._sparse_flows: List[Optional[SparseNodeValues]] = \
            [None] * len(self.nodes)

        # Per-ns flag set by ``forward`` when ``missing_mask[var_id]`` is True
        # at that timestep. Read by ``backward`` to skip emission-flow
        # accumulation (a marginalised position contributes nothing to the
        # emission gradient). ``False`` whenever no missing_mask is supplied.
        self._was_missing: List[bool] = [False] * len(self.nodes)

        # Lazily populated arange tensors used as ``indices`` for the
        # missing-position SparseNodeValues (one per H value). Allocated on
        # the layer's device the first time we hit a missing position.
        self._missing_indices_cache: dict = {}
        # Pre-allocated ``values`` buffers (length H) for missing-position
        # forward outputs, to avoid per-call cudaMalloc on long sequences.
        self._missing_values_cache: dict = {}

        # Per-ns forward ``values`` workspaces (B * _max_nnz_per_col floats at
        # B>1, max(_max_nnz_per_col, H) at B=1) and the shared
        # ``[len(nodes), B]`` per-sample running-max buffer filled inline by
        # the forward kernel (row ns_idx becomes ``sv.max_val``).
        self._fwd_values_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)
        self._fwd_max_workspace: Optional[torch.Tensor] = None

        # Set to True by ``TensorCircuit._mark_sparse_prod_scatter_skip`` after
        # all layers compile when *every* consumer of this prod's outputs is a
        # :class:`SparseInputSumLayer` (which reads the sparse output directly).
        # In that case the forward pass produces only ``SparseNodeValues`` and
        # skips the O(H·B) ``scatter_to_dense``; on a B>1 fallback the
        # downstream ``SparseInputSumLayer`` materialises element_mars on
        # demand before calling ``super().forward/backward``.
        self._skip_scatter: bool = False

    def __repr__(self) -> str:
        return (
            f"SparseProdLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sparse_ns={len(self.nodes)})"
        )

    # ---------------- Forward ---------------- #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, data: Optional[torch.Tensor] = None,
                data_cpu: Optional[torch.Tensor] = None,
                data_list: Optional[list] = None,
                pattern_cache: Optional[dict] = None,
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert data is not None, (
            "SparseProdLayer.forward requires `data` "
            "(per-var observed tokens, shape [num_vars, 1]) as a kwarg."
        )
        assert not self.provided("fw_partition_local_ids"), \
            "SparseProdLayer does not support partial evaluation yet."

        batch_size = element_mars.size(1)
        assert missing_mask is None or batch_size == 1, (
            "missing_mask on the sparse fast path is only supported at "
            "batch_size == 1 (conditional queries stay B=1 for now)."
        )
        assert self._skip_scatter or batch_size == 1, (
            "SparseProdLayer with a dense consumer (scatter_to_dense bridge) "
            "is B=1 only; the batched chain requires every consumer to be a "
            "SparseInputSumLayer / SparseIOSumLayer (_skip_scatter=True)."
        )

        # Prefer the CPU mirror cached by ``TensorCircuit.forward`` so the
        # per-ns ``build_sparse_pattern`` lookup is a host read, not a
        # device<->host sync. Falls back to ``data`` when called outside
        # TensorCircuit (e.g. unit tests that invoke the layer directly).
        data_for_pattern = data_cpu if data_cpu is not None else data

        # ``missing_mask`` semantics: for each variable, True ⇒ marginalise the
        # input distribution (treat as if the token at that position were
        # unobserved). The standard input-layer ``_fw_missing_mask_kernel``
        # operates on ``node_mars`` at the input layer, but the sparse-emission
        # path bypasses ``node_mars`` for sparse-input ns's (they go directly
        # into ``element_mars`` via ``_compute_sparse_output``), so the
        # correction has to be applied here instead. Without this, missing
        # positions would silently treat ``data[var_id]`` as the observed
        # token — see issue triggered by ``juice.queries.conditional`` on
        # SparseCategorical-emission HMMs.
        missing_mask_cpu = None
        if missing_mask is not None:
            mm = missing_mask
            # missing_mask path is B=1-only (asserted above); tolerate
            # [num_vars] or [B, num_vars] / [num_vars, B] with the B dim == 1.
            if mm.dim() == 2:
                if mm.size(0) == 1:
                    mm = mm[0]
                elif mm.size(1) == 1:
                    mm = mm[:, 0]
                else:
                    raise AssertionError(
                        "SparseProdLayer.forward got a 2D missing_mask with "
                        "neither dim == 1; missing_mask on the sparse path "
                        "is B=1 only."
                    )
            missing_mask_cpu = mm.cpu() if mm.device.type != "cpu" else mm

        # Shared per-sample running-max buffer, filled inline by the forward
        # kernel's fused atomic_max. One memset per forward; empty-column
        # ns'es leave their row at -inf (consumers short-circuit on nnz == 0).
        if (self._fwd_max_workspace is None
                or self._fwd_max_workspace.device != node_mars.device
                or self._fwd_max_workspace.shape != (len(self.nodes), batch_size)):
            self._fwd_max_workspace = torch.empty(
                len(self.nodes), batch_size,
                dtype=torch.float32, device=node_mars.device,
            )
        self._fwd_max_workspace.fill_(float("-inf"))

        for ns_idx, ns in enumerate(self.nodes):
            is_missing = bool(missing_mask_cpu[ns.var_id].item()) if missing_mask_cpu is not None else False
            self._was_missing[ns_idx] = is_missing
            sv = self._compute_sparse_output(
                ns_idx=ns_idx, ns=ns,
                data=data_for_pattern, node_mars=node_mars,
                data_list=data_list,
                pattern_cache=pattern_cache,
                is_missing=is_missing,
                batch_size=batch_size,
            )
            self._sparse_outputs[ns_idx] = sv
            if not self._skip_scatter:
                sv.scatter_to_dense(
                    element_mars, ns._output_ind_range[0], fill_value=LOG_EPS,
                )

        return None

    def _compute_sparse_output(self, ns_idx: int, ns: SparseProdNodes,
                                data: torch.Tensor,
                                node_mars: torch.Tensor,
                                data_list: Optional[list] = None,
                                pattern_cache: Optional[dict] = None,
                                is_missing: bool = False,
                                batch_size: int = 1) -> SparseNodeValues:
        """
        Build the sparse ``(row, value)`` output for one
        :class:`SparseProdNodes`. Pattern construction (column-bounds lookup
        + ``indices`` view / batched ``col_starts``) lives on
        :meth:`SparseCategorical.build_sparse_pattern`; this method only
        launches the triton kernel that fills
        ``values[b, j] = log(params[csc_values_base + col_starts[b] + j]) +
                         Σ_ch node_mars[dense_ch_lookup[ch, row], b]``
        and its per-sample running max (``sv.max_val``).

        ``is_missing=True`` (B=1 only) ⇒ this position is marginalised. The
        sparsity pattern is replaced with all H rows (``indices =
        arange(H)``, ``total_nnz = H``) and the kernel skips the params load
        so each latent's value reduces to just the transition-children sum.
        ``log P(x|h) = log(Σ_v P(v|h)) = log(1) = 0`` is the correct
        marginal contribution from a row-normalised emission.
        """
        device = node_mars.device
        sparse_cs = ns.sparse_input_ns
        sparse_input_layer = self._sparse_input_layers[ns_idx]
        H = ns.num_nodes
        max_out = self._fwd_max_workspace[ns_idx]

        if is_missing:
            indices = self._missing_indices_cache.get(H)
            if indices is None or indices.device != device:
                indices = torch.arange(H, dtype=torch.long, device=device).contiguous()
                self._missing_indices_cache[H] = indices
            values = self._missing_values_cache.get(H)
            if values is None or values.device != device:
                values = torch.empty(H, dtype=torch.float32, device=device)
                self._missing_values_cache[H] = values
            sv = SparseNodeValues(
                col_start=0, total_nnz=H,
                indices=indices, values=values, num_rows=H,
                max_val=max_out,
            )
            dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])
            BLOCK = _FWD_BLOCK_K
            grid = (triton.cdiv(H, BLOCK), 1)
            _sparse_prod_forward_kernel[grid](
                params_ptr=sparse_input_layer.params,
                node_mars_ptr=node_mars,
                dense_ch_lookup_ptr=dense_ch_lookup,
                indices_ptr=indices,
                values_out_ptr=values,
                col_starts_ptr=values, nnz_ptr=values,  # unused at B=1
                max_out_ptr=max_out,
                param_base=0,
                num_rows=H,
                total_nnz=H,
                batch_size=1,
                v_stride=0,
                NUM_DENSE_CHS=ns.num_dense_chs,
                BLOCK=BLOCK,
                BLOCK_B=1,
                IS_MISSING=True,
                IS_BATCHED=False,
            )
            return sv

        # Per-ns values workspace: build_sparse_pattern narrows/views it, so
        # no per-call cudaMalloc on either path. The H term keeps the B=1
        # missing-position capacity contract of the old fresh-alloc path out
        # of the picture (missing has its own cache above).
        ws = self._fwd_values_workspaces[ns_idx]
        needed = (batch_size * sparse_cs.dist._max_nnz_per_col
                  if batch_size > 1 else sparse_cs.dist._max_nnz_per_col)
        if ws is None or ws.device != device or ws.numel() < needed:
            ws = torch.empty(needed, dtype=torch.float32, device=device)
            self._fwd_values_workspaces[ns_idx] = ws

        sv = sparse_cs.dist.build_sparse_pattern(
            data=data, var_id=ns.var_id, num_rows=H, device=device,
            values_out=ws, data_list=data_list, pattern_cache=pattern_cache,
        )
        sv.max_val = max_out

        if sv.total_nnz == 0:
            return sv

        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])

        if sv.is_batched:
            BLOCK = _FWD_BATCHED_BLOCK_K
            BLOCK_B = _FWD_BATCHED_BLOCK_B
            grid = (triton.cdiv(sv.total_nnz, BLOCK),
                    triton.cdiv(batch_size, BLOCK_B))
            _sparse_prod_forward_kernel[grid](
                params_ptr=sparse_input_layer.params,
                node_mars_ptr=node_mars,
                dense_ch_lookup_ptr=dense_ch_lookup,
                indices_ptr=sv.indices,
                values_out_ptr=sv.values,
                col_starts_ptr=sv.col_starts, nnz_ptr=sv.nnz,
                max_out_ptr=max_out,
                param_base=sparse_cs._param_range[0],
                num_rows=sv.num_rows,
                total_nnz=sv.total_nnz,
                batch_size=batch_size,
                v_stride=sv.values.stride(0),
                NUM_DENSE_CHS=ns.num_dense_chs,
                BLOCK=BLOCK,
                BLOCK_B=BLOCK_B,
                IS_MISSING=False,
                IS_BATCHED=True,
            )
        else:
            BLOCK = _FWD_BLOCK_K
            grid = (triton.cdiv(sv.total_nnz, BLOCK), 1)
            _sparse_prod_forward_kernel[grid](
                params_ptr=sparse_input_layer.params,
                node_mars_ptr=node_mars,
                dense_ch_lookup_ptr=dense_ch_lookup,
                indices_ptr=sv.indices,
                values_out_ptr=sv.values,
                col_starts_ptr=sv.values, nnz_ptr=sv.values,  # unused at B=1
                max_out_ptr=max_out,
                param_base=sparse_cs._param_range[0] + sv.col_start,
                num_rows=sv.num_rows,
                total_nnz=sv.total_nnz,
                batch_size=1,
                v_stride=0,
                NUM_DENSE_CHS=ns.num_dense_chs,
                BLOCK=BLOCK,
                BLOCK_B=1,
                IS_MISSING=False,
                IS_BATCHED=False,
            )

        return sv

    # ---------------- Backward ---------------- #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False,
                 data: Optional[torch.Tensor] = None, **kwargs) -> None:
        if self._skip_scatter:
            # Sparse fast path: downstream SparseInputSumLayer wrote sv_flow
            # straight into ``self._sparse_flows``. Route active-row flows to
            # each dense child's node_flows slice directly — no element_flows
            # round-trip, no super().backward() over the full H range.
            for ns_idx, ns in enumerate(self.nodes):
                sv_flow = self._sparse_flows[ns_idx]
                assert sv_flow is not None, (
                    "SparseProdLayer.backward (skip_scatter) expected a "
                    "sv_flow from the downstream SparseInputSumLayer; none "
                    "was stashed. Did the sum layer's backward run?"
                )
                # NOTE: do NOT consume-and-clear ``self._sparse_flows[ns_idx]``
                # — leaving it populated post-backward is what lets external
                # readers (e.g. the sparse conditional-query kernel via
                # ``_sparse_flow_owner``) recover the per-active-row flow
                # without rerunning backward. The next backward overwrites
                # the slot anyway. Same contract as ``_sparse_outputs[ns_idx]``
                # on the forward side, which also persists across passes.

                self._scatter_flow_to_children(ns_idx, ns, sv_flow, node_flows)

                if self._was_missing[ns_idx]:
                    # Position was marginalised in the forward pass: the
                    # emission ``log P(x|h)`` was treated as ``log(1) = 0``
                    # for every row (no specific token observed), so its
                    # gradient w.r.t. the emission params is identically
                    # zero. Skip the param-flow accumulation that would
                    # otherwise spuriously credit this column.
                    continue

                sparse_cs = ns.sparse_input_ns
                sparse_cs.dist.custom_backward_sparse(
                    input_layer=self._sparse_input_layers[ns_idx],
                    sparse_flow=sv_flow,
                    csc_pflows_base=sparse_cs._param_flow_range[0],
                    logspace_flows=logspace_flows,
                )

            return None

        # Dense fallback: reuse the inherited plain-prod backward. It will
        # also scatter into the sparse input's node_flows slice, which is
        # harmless — InputLayer.backward skips that ns (gated via
        # ``_skip_input_backward``), so no accidental second accumulation.
        super().backward(
            node_flows=node_flows, element_flows=element_flows,
            logspace_flows=logspace_flows, **kwargs,
        )

        # Sparse-input flows: gather element_flows → SparseNodeValues →
        # SparseCategorical.custom_backward_sparse.
        for ns_idx, ns in enumerate(self.nodes):
            sv = self._sparse_outputs[ns_idx]
            assert sv is not None, (
                "SparseProdLayer.backward called before forward "
                "(no cached sparse output)."
            )
            if self._was_missing[ns_idx]:
                # See note above: marginalised positions contribute no
                # gradient to the emission params.
                continue
            sparse_cs = ns.sparse_input_ns
            sv_flow = sv.gather_from_dense(element_flows, ns._output_ind_range[0])

            sparse_cs.dist.custom_backward_sparse(
                input_layer=self._sparse_input_layers[ns_idx],
                sparse_flow=sv_flow,
                csc_pflows_base=sparse_cs._param_flow_range[0],
                logspace_flows=logspace_flows,
            )

        return None

    def _scatter_flow_to_children(self, ns_idx: int, ns: SparseProdNodes,
                                    sv_flow: SparseNodeValues,
                                    node_flows: torch.Tensor) -> None:
        """Write ``sv_flow.values`` into each dense child's ``node_flows``.

        Each active row ``h`` of the prod output has exactly one slot in
        each dense child (same rule as ``ProdLayer.backward``:
        ``node_flows[u_cid] = element_flows[parid]``). Scatters via
        ``_dense_ch_lookup[ch, h]`` with one kernel launch covering every
        (sample, slot) pair — plain stores, since each ``(row, b)`` target
        is unique per child.

        Sparse-input flow is **not** scattered into ``node_flows`` — it
        stays in ``self._sparse_flows[ns_idx]``, which is the canonical
        sparse-flow exposure (mirrors the forward's ``_sparse_outputs``
        convention). Consumers like ``pyjuice.queries.conditional`` reach it
        via the ``ns.sparse_input_ns._sparse_flow_owner`` back-reference.

        ``node_flows`` is zero-initialised at the top of
        ``TensorCircuit.backward`` and no other layer writes to these rows
        on the sparse path, so inactive rows correctly stay at 0 (they
        contributed no LL path).
        """
        if sv_flow.total_nnz == 0 or ns.num_dense_chs == 0:
            return
        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])
        batch_size = node_flows.size(1)

        if sv_flow.is_batched:
            BLOCK = _FWD_BATCHED_BLOCK_K
            BLOCK_B = _FWD_BATCHED_BLOCK_B
            grid = (triton.cdiv(sv_flow.total_nnz, BLOCK),
                    triton.cdiv(batch_size, BLOCK_B))
            _sparse_prod_scatter_flow_kernel[grid](
                node_flows_ptr=node_flows,
                dense_ch_lookup_ptr=dense_ch_lookup,
                indices_ptr=sv_flow.indices,
                values_ptr=sv_flow.values,
                col_starts_ptr=sv_flow.col_starts, nnz_ptr=sv_flow.nnz,
                num_rows=sv_flow.num_rows,
                total_nnz=sv_flow.total_nnz,
                batch_size=batch_size,
                v_stride=sv_flow.values.stride(0),
                NUM_DENSE_CHS=ns.num_dense_chs,
                BLOCK=BLOCK,
                BLOCK_B=BLOCK_B,
                IS_BATCHED=True,
            )
        else:
            BLOCK = _FWD_BLOCK_K
            grid = (triton.cdiv(sv_flow.total_nnz, BLOCK), 1)
            _sparse_prod_scatter_flow_kernel[grid](
                node_flows_ptr=node_flows,
                dense_ch_lookup_ptr=dense_ch_lookup,
                indices_ptr=sv_flow.indices,
                values_ptr=sv_flow.values,
                col_starts_ptr=sv_flow.values, nnz_ptr=sv_flow.values,  # unused
                num_rows=sv_flow.num_rows,
                total_nnz=sv_flow.total_nnz,
                batch_size=batch_size,
                v_stride=0,
                NUM_DENSE_CHS=ns.num_dense_chs,
                BLOCK=BLOCK,
                BLOCK_B=1,
                IS_BATCHED=False,
            )


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit(
    do_not_specialize=["num_rows", "total_nnz", "batch_size", "v_stride"],
    do_not_specialize_on_alignment=["indices_ptr", "values_ptr",
                                     "dense_ch_lookup_ptr", "col_starts_ptr",
                                     "nnz_ptr"],
)
def _sparse_prod_scatter_flow_kernel(
    node_flows_ptr, dense_ch_lookup_ptr,
    indices_ptr, values_ptr,
    col_starts_ptr, nnz_ptr,
    num_rows, total_nnz, batch_size, v_stride,
    NUM_DENSE_CHS: tl.constexpr,
    BLOCK: tl.constexpr, BLOCK_B: tl.constexpr,
    IS_BATCHED: tl.constexpr,
):
    """Scatter per-active-row flows into each dense child's ``node_flows``:
    ``node_flows[dense_ch_lookup[ch, row], b] = flow[b, j]``. Grid
    ``(cdiv(K, BLOCK), cdiv(B, BLOCK_B))``. Plain stores — each ``(row, b)``
    target is unique per child. At B=1 replaces the former
    ``index_select`` + advanced-indexing pair (two torch dispatches per ns
    per timestep) with one launch."""
    pid = tl.program_id(0)
    offs_j = pid * BLOCK + tl.arange(0, BLOCK)

    if IS_BATCHED:
        pid_b = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < batch_size
        cs_b = tl.load(col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_b = tl.load(nnz_ptr + offs_b, mask=mask_b, other=0)
        mask = mask_b[:, None] & (offs_j[None, :] < k_b[:, None])

        row = tl.load(indices_ptr + cs_b[:, None] + offs_j[None, :],
                      mask=mask, other=0)
        v = tl.load(values_ptr + offs_b[:, None] * v_stride + offs_j[None, :],
                    mask=mask, other=0.0)
        for ch in tl.static_range(NUM_DENSE_CHS):
            nid = tl.load(dense_ch_lookup_ptr + ch * num_rows + row,
                          mask=mask, other=0)
            tl.store(node_flows_ptr + nid * batch_size + offs_b[:, None],
                     v, mask=mask)
    else:
        mask = offs_j < total_nnz
        row = tl.load(indices_ptr + offs_j, mask=mask, other=0)
        v = tl.load(values_ptr + offs_j, mask=mask, other=0.0)
        for ch in tl.static_range(NUM_DENSE_CHS):
            nid = tl.load(dense_ch_lookup_ptr + ch * num_rows + row,
                          mask=mask, other=0)
            tl.store(node_flows_ptr + nid * batch_size, v, mask=mask)


@triton.jit(
    do_not_specialize=["param_base", "num_rows", "total_nnz", "batch_size",
                       "v_stride"],
    do_not_specialize_on_alignment=["indices_ptr", "values_out_ptr",
                                     "dense_ch_lookup_ptr", "col_starts_ptr",
                                     "nnz_ptr", "max_out_ptr"],
)
def _sparse_prod_forward_kernel(
    params_ptr, node_mars_ptr, dense_ch_lookup_ptr,
    indices_ptr,
    values_out_ptr,
    col_starts_ptr, nnz_ptr, max_out_ptr,
    param_base,
    num_rows,
    total_nnz,
    batch_size,
    v_stride,
    NUM_DENSE_CHS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    IS_MISSING: tl.constexpr,
    IS_BATCHED: tl.constexpr,
):
    """
    Grid = ``(cdiv(K, BLOCK), cdiv(B, BLOCK_B))`` — one program per
    ``[BLOCK_B, BLOCK]`` (batch, slot) tile; ``(cdiv(total_nnz, BLOCK), 1)``
    at B=1.

    ``IS_BATCHED == 0`` (B=1): ``param_base`` is the caller-computed sum of
    the ns's CSC param base and the observed column's ``col_start`` — slots
    within the active column are contiguous, so slot ``j`` lives at
    ``params[param_base + j]``. For each active slot ``j``:
      - ``row      = indices[j]``
      - ``log_emit = log(params[param_base + j])``
      - ``dense_sum = Σ_ch node_mars[dense_ch_lookup[ch, row], 0]``
      - ``values[j] = log_emit + dense_sum``

    ``IS_BATCHED == 1``: each batch lane reads its own column —
    ``slot = col_starts[b] + j`` (masked ``j < nnz[b]``), ``param_base`` is
    the CSC base only, ``indices_ptr`` is the FULL ``_csc_indices``, the
    dense-sibling load moves to ``node_mars[ch_nid * B + b]``, and stores go
    to ``values[b * v_stride + j]``.

    Both variants fuse a per-sample running max: ``atomic_max(max_out[b],
    max_j out[b, j])`` — so every ``sv.max_val`` is produced inline and no
    consumer needs a ``values.max()`` torch dispatch.

    When ``IS_MISSING`` is set (B=1 only) the variable is marginalised:
    ``log P(x|h) = log(Σ_v P(v|h)) = log(1) = 0`` for every row, so the
    kernel skips the ``params`` load and uses ``offs_j`` directly as ``row``
    — the caller pre-sets ``total_nnz = num_rows``.
    """
    pid = tl.program_id(0)
    offs_j = pid * BLOCK + tl.arange(0, BLOCK)

    if IS_BATCHED:
        pid_b = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < batch_size
        cs_b = tl.load(col_starts_ptr + offs_b, mask=mask_b, other=0)
        k_b = tl.load(nnz_ptr + offs_b, mask=mask_b, other=0)
        mask = mask_b[:, None] & (offs_j[None, :] < k_b[:, None])

        slot = cs_b[:, None] + offs_j[None, :]
        row = tl.load(indices_ptr + slot, mask=mask, other=0)
        val = tl.load(params_ptr + param_base + slot, mask=mask, other=1.0)
        log_emit = tl.log(val)

        dense_sum = tl.zeros([BLOCK_B, BLOCK], dtype=tl.float32)
        for ch in tl.static_range(NUM_DENSE_CHS):
            ch_nid = tl.load(
                dense_ch_lookup_ptr + ch * num_rows + row,
                mask=mask, other=0,
            )
            dense_sum += tl.load(
                node_mars_ptr + ch_nid * batch_size + offs_b[:, None],
                mask=mask, other=0.0,
            )

        out = log_emit + dense_sum
        tl.store(
            values_out_ptr + offs_b[:, None] * v_stride + offs_j[None, :],
            out, mask=mask,
        )
        tile_max = tl.max(tl.where(mask, out, float("-inf")), axis=1)
        tl.atomic_max(max_out_ptr + offs_b, tile_max, mask=mask_b)
    else:
        mask = offs_j < total_nnz

        if IS_MISSING:
            row = offs_j
            log_emit = tl.zeros([BLOCK], dtype=tl.float32)
        else:
            row = tl.load(indices_ptr + offs_j, mask=mask, other=0)
            val = tl.load(params_ptr + param_base + offs_j, mask=mask, other=1.0)
            log_emit = tl.log(val)

        dense_sum = tl.zeros([BLOCK], dtype=tl.float32)
        for ch in tl.static_range(NUM_DENSE_CHS):
            ch_nid = tl.load(
                dense_ch_lookup_ptr + ch * num_rows + row,
                mask=mask, other=0,
            )
            # B=1: node_mars stride-1 along batch, so addr == ch_nid.
            dense_sum += tl.load(node_mars_ptr + ch_nid, mask=mask, other=0.0)

        out = log_emit + dense_sum
        tl.store(values_out_ptr + offs_j, out, mask=mask)
        tile_max = tl.max(tl.where(mask, out, float("-inf")), axis=0)
        tl.atomic_max(max_out_ptr, tile_max)
