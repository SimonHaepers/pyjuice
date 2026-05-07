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
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert data is not None, (
            "SparseProdLayer.forward requires `data` "
            "(per-var observed tokens, shape [num_vars, 1]) as a kwarg."
        )
        assert not self.provided("fw_partition_local_ids"), \
            "SparseProdLayer does not support partial evaluation yet."

        batch_size = element_mars.size(1)
        assert batch_size == 1, (
            "SparseProdLayer is B=1 only (inference-only fast path matching "
            "SparseInputSumLayer). For B>1, build the DAG with plain "
            "`multiply` / `summate` (or `_force_plain=True`)."
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
            # Sparse path is B=1; tolerate [num_vars] or [B, num_vars] / [num_vars, B].
            if mm.dim() == 2:
                if mm.size(0) == 1:
                    mm = mm[0]
                elif mm.size(1) == 1:
                    mm = mm[:, 0]
                else:
                    raise AssertionError(
                        "SparseProdLayer.forward got a 2D missing_mask with "
                        "neither dim == 1; sparse path is B=1 only."
                    )
            missing_mask_cpu = mm.cpu() if mm.device.type != "cpu" else mm

        for ns_idx, ns in enumerate(self.nodes):
            is_missing = bool(missing_mask_cpu[ns.var_id].item()) if missing_mask_cpu is not None else False
            self._was_missing[ns_idx] = is_missing
            sv = self._compute_sparse_output(
                ns_idx=ns_idx, ns=ns,
                data=data_for_pattern, node_mars=node_mars,
                data_list=data_list,
                is_missing=is_missing,
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
                                is_missing: bool = False) -> SparseNodeValues:
        """
        Build the sparse ``(row, value)`` output for one
        :class:`SparseProdNodes` at B=1. Pattern construction
        (column-bounds lookup + ``indices`` view) lives on
        :meth:`SparseCategorical.build_sparse_pattern`; this method only
        launches the triton kernel that fills
        ``values[j] = log(params[csc_values_base + col_start + j]) +
                      Σ_ch node_mars[dense_ch_lookup[ch, row_j], 0]``.

        ``is_missing=True`` ⇒ this position is marginalised. The sparsity
        pattern is replaced with all H rows (``indices = arange(H)``,
        ``total_nnz = H``) and the kernel skips the params load so each
        latent's value reduces to just the transition-children sum.
        ``log P(x|h) = log(Σ_v P(v|h)) = log(1) = 0`` is the correct
        marginal contribution from a row-normalised emission.
        """
        device = node_mars.device
        sparse_cs = ns.sparse_input_ns
        sparse_input_layer = self._sparse_input_layers[ns_idx]
        H = ns.num_nodes

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
            )
            dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])
            BLOCK = 256
            grid = (triton.cdiv(H, BLOCK),)
            _sparse_prod_forward_kernel[grid](
                params_ptr=sparse_input_layer.params,
                node_mars_ptr=node_mars,
                dense_ch_lookup_ptr=dense_ch_lookup,
                indices_ptr=indices,
                values_out_ptr=values,
                param_base=0,
                num_rows=H,
                total_nnz=H,
                NUM_DENSE_CHS=ns.num_dense_chs,
                BLOCK=BLOCK,
                IS_MISSING=True,
            )
            return sv

        sv = sparse_cs.dist.build_sparse_pattern(
            data=data, var_id=ns.var_id, num_rows=H, device=device,
            data_list=data_list,
        )

        if sv.total_nnz == 0:
            return sv

        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])

        BLOCK = 256
        grid = (triton.cdiv(sv.total_nnz, BLOCK),)
        _sparse_prod_forward_kernel[grid](
            params_ptr=sparse_input_layer.params,
            node_mars_ptr=node_mars,
            dense_ch_lookup_ptr=dense_ch_lookup,
            indices_ptr=sv.indices,
            values_out_ptr=sv.values,
            param_base=sparse_cs._param_range[0] + sv.col_start,
            num_rows=sv.num_rows,
            total_nnz=sv.total_nnz,
            NUM_DENSE_CHS=ns.num_dense_chs,
            BLOCK=BLOCK,
            IS_MISSING=False,
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

        Each active row ``h = sv_flow.indices[j]`` of the prod output has
        exactly one slot in each dense child (same rule as
        ``ProdLayer.backward``: ``node_flows[u_cid] = element_flows[parid]``).
        Scatters via ``_dense_ch_lookup[ch, h]``.

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
        if sv_flow.total_nnz == 0:
            return
        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])
        # dense_ch_lookup: [num_dense_chs, H]. Per ch, fetch the active rows'
        # global nid indices and scatter values. B=1: only col 0 is used.
        for ch in range(ns.num_dense_chs):
            target_ids = dense_ch_lookup[ch].index_select(0, sv_flow.indices)
            node_flows[target_ids, 0] = sv_flow.values


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit(
    do_not_specialize=["param_base", "num_rows", "total_nnz"],
    do_not_specialize_on_alignment=["indices_ptr", "values_out_ptr",
                                     "dense_ch_lookup_ptr"],
)
def _sparse_prod_forward_kernel(
    params_ptr, node_mars_ptr, dense_ch_lookup_ptr,
    indices_ptr,
    values_out_ptr,
    param_base,
    num_rows,
    total_nnz,
    NUM_DENSE_CHS: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_MISSING: tl.constexpr,
):
    """
    Per-entry 1D kernel (B=1). Grid = ``(cdiv(total_nnz, BLOCK),)``.

    ``param_base`` is the caller-computed sum of the ns's CSC param base and
    the observed column's ``col_start`` — slots within the active column are
    contiguous, so slot ``j`` lives at ``params[param_base + j]``.

    For each active slot ``j``:
      - ``row      = indices[j]``
      - ``log_emit = log(params[param_base + j])``
      - ``dense_sum = Σ_ch node_mars[dense_ch_lookup[ch, row], 0]``
      - ``values[j] = log_emit + dense_sum``

    When ``IS_MISSING`` is set the variable at this ns is marginalised
    (``missing_mask[var_id] == True``). Then ``log P(x|h) = log(Σ_v
    P(v|h)) = log(1) = 0`` for every row, so the kernel skips the
    ``params`` load and uses ``offs_j`` directly as ``row`` — the caller
    pre-sets ``total_nnz = num_rows`` so every latent gets a value.
    """
    pid = tl.program_id(0)
    offs_j = pid * BLOCK + tl.arange(0, BLOCK)
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

    tl.store(values_out_ptr + offs_j, log_emit + dense_sum, mask=mask)
