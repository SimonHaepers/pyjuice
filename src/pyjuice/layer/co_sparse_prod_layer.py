from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import ProdNodes, SparseProdNodes
from .sparse_prod_layer import SparseProdLayer
from .sparse_node_values import SparseNodeValues
from .layer_group import LayerGroup


@triton.jit
def _co_sparse_log_add_kernel(
    out_ptr, params_ptr, dense_values_ptr,
    param_offset, n,
    BLOCK: tl.constexpr,
):
    """Fused ``out[i] = log(params[param_offset + i]) + dense_values[i]``.

    Replaces the three-kernel ``torch.log(emit_params) + sv_dense.values``
    + ``sv.values.copy_(...)`` chain that otherwise launches one tiny
    element-wise kernel each (and allocates two intermediate temps) per
    timestep on the sparse HMM path.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(params_ptr + param_offset + offs, mask=mask, other=1.0)
    d = tl.load(dense_values_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.log(p) + d, mask=mask)


class CoSparseProdLayer(SparseProdLayer):
    """Co-sparse product layer. Both inputs are :class:`SparseNodeValues`
    with **identical** indices — the sparse ``SparseCategorical`` input's
    CSC column at ``var_id`` and the upstream :class:`SparseIOSumLayer`
    output for the same ``var_id``. The log-space product reduces to an
    element-wise add of the two packed ``.values`` tensors; ``indices`` is
    passed through untouched.

    Requires ``num_dense_chs == 1`` (the HMM chain case we support).
    Unconditionally ``_skip_scatter=True`` — never writes ``element_mars``.
    """

    def __init__(self, nodes: Sequence[ProdNodes],
                 global_nid_start: Optional[int] = None,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False,
                 input_layer_group: Optional[LayerGroup] = None,
                 inner_layer_groups: Optional[Sequence[LayerGroup]] = None,
                 **kwargs) -> None:
        super().__init__(
            nodes=nodes,
            global_nid_start=global_nid_start,
            layer_sparsity_tol=layer_sparsity_tol,
            max_num_partitions=max_num_partitions,
            disable_gpu_compilation=disable_gpu_compilation,
            force_gpu_compilation=force_gpu_compilation,
            input_layer_group=input_layer_group,
        )

        for ns in self.nodes:
            assert isinstance(ns, SparseProdNodes), (
                f"CoSparseProdLayer expects SparseProdNodes; got {type(ns).__name__}."
            )
            assert ns.num_dense_chs == 1, (
                f"CoSparseProdLayer requires exactly 1 dense (sum) child; "
                f"got {ns.num_dense_chs}."
            )

        assert inner_layer_groups is not None, (
            "CoSparseProdLayer needs inner_layer_groups to resolve the "
            "upstream SparseIOSumLayer that owns each prod's dense child."
        )
        self._dense_sum_refs: List[Tuple] = []
        # Imported lazily to avoid the circular dep
        # CoSparseProdLayer → SparseIOSumLayer → SparseInputSumLayer.
        from .sparse_io_sum_layer import SparseIOSumLayer

        for ns in self.nodes:
            dense_ch_ns = ns.chs[ns.dense_ch_idxs[0]]
            found = None
            for lg in inner_layer_groups:
                if lg.is_prod():
                    continue
                for layer in lg:
                    if not isinstance(layer, SparseIOSumLayer):
                        continue
                    for idx, sum_ns in enumerate(layer.nodes):
                        if sum_ns is dense_ch_ns:
                            found = (layer, idx)
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            assert found is not None, (
                "CoSparseProdLayer: dense child sum ns is not owned by any "
                "SparseIOSumLayer in inner_layer_groups. The DAG pre-pass "
                "classified the sum as sparse_io but the layer dispatch did "
                "not compile it as SparseIOSumLayer — check TensorCircuit "
                "compilation order."
            )
            self._dense_sum_refs.append(found)

        # Never scatter to element_mars — downstream is always another
        # sparse consumer (upstream sum or the final SparseInputSumLayer-via-
        # consumer-of-our-consumer). The chain invariant is guaranteed by the
        # eligibility check.
        self._skip_scatter = True

        # Per-ns GPU workspace for the forward sv.values buffer. Sized to
        # ``dist._max_nnz_per_col`` and re-used every call so the per-step
        # ``cudaMalloc`` in ``build_sparse_pattern`` becomes a free slice.
        # Allocated lazily on first forward (device unknown at __init__).
        self._fwd_values_workspaces: List[Optional[torch.Tensor]] = \
            [None] * len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"CoSparseProdLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_sparse_ns={len(self.nodes)})"
        )

    # ---------------- Forward ---------------- #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, data: Optional[torch.Tensor] = None,
                data_cpu: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert data is not None, (
            "CoSparseProdLayer.forward requires `data` (per-var observed tokens)."
        )
        assert not self.provided("fw_partition_local_ids"), \
            "CoSparseProdLayer does not support partial evaluation."

        batch_size = element_mars.size(1)
        assert batch_size == 1, "CoSparseProdLayer is B=1 only."

        data_for_pattern = data_cpu if data_cpu is not None else data

        torch.cuda.nvtx.range_push(
            f"CoSparseProdLayer.fwd(n_ns={len(self.nodes)})"
        )
        for ns_idx, ns in enumerate(self.nodes):
            torch.cuda.nvtx.range_push(f"ns[{ns_idx}]")
            sparse_cs = ns.sparse_input_ns
            sparse_input_layer = self._sparse_input_layers[ns_idx]

            ws = self._fwd_values_workspaces[ns_idx]
            if ws is None or ws.device != node_mars.device:
                ws = torch.empty(
                    sparse_cs.dist._max_nnz_per_col,
                    dtype=torch.float32, device=node_mars.device,
                )
                self._fwd_values_workspaces[ns_idx] = ws

            torch.cuda.nvtx.range_push("build_sparse_pattern")
            sv = sparse_cs.dist.build_sparse_pattern(
                data=data_for_pattern, var_id=ns.var_id,
                num_rows=ns.num_nodes, device=node_mars.device,
                values_out=ws,
            )
            torch.cuda.nvtx.range_pop()
            self._sparse_outputs[ns_idx] = sv

            if sv.total_nnz == 0:
                torch.cuda.nvtx.range_pop()
                continue

            sum_layer, sum_ns_idx = self._dense_sum_refs[ns_idx]
            sv_dense = sum_layer._sparse_outputs[sum_ns_idx]

            # Invariant: indices coincide (both built from
            # ``build_sparse_pattern(var_id=ns.var_id)`` — views of the same
            # slice of ``dist._csc_indices``). total_nnz + col_start equality
            # certifies the views alias the same memory.
            assert sv_dense.col_start == sv.col_start \
                   and sv_dense.total_nnz == sv.total_nnz, (
                "CoSparseProdLayer: dense sum output sparsity does not match "
                "this prod's sparse input sparsity. Expected identical views "
                "of dist._csc_indices (same var_id)."
            )

            torch.cuda.nvtx.range_push(f"values_add(nnz={sv.total_nnz})")
            # Emission params live flat-packed in sparse_input_layer.params,
            # indexed by ``_param_range[0] + col_start + j`` for slot j — same
            # addressing as ``_sparse_prod_forward_kernel``.
            param_base = sparse_cs._param_range[0] + sv.col_start
            BLOCK = 256
            grid = (triton.cdiv(sv.total_nnz, BLOCK),)
            _co_sparse_log_add_kernel[grid](
                out_ptr=sv.values,
                params_ptr=sparse_input_layer.params,
                dense_values_ptr=sv_dense.values,
                param_offset=param_base,
                n=sv.total_nnz,
                BLOCK=BLOCK,
            )
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return None

    # ---------------- Backward ---------------- #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False,
                 data: Optional[torch.Tensor] = None, **kwargs) -> None:
        torch.cuda.nvtx.range_push(
            f"CoSparseProdLayer.bwd(n_ns={len(self.nodes)})"
        )
        # Log-space product forward: ``out = log_emit + sv_dense``, so
        # ∂out/∂log_emit = ∂out/∂sv_dense = 1. The incoming sv_flow becomes
        # the outgoing flow on both inputs unchanged.
        for ns_idx, ns in enumerate(self.nodes):
            sv_flow = self._sparse_flows[ns_idx]
            assert sv_flow is not None, (
                "CoSparseProdLayer.backward expected sv_flow from the "
                "downstream sum layer; none was stashed."
            )
            # NOTE: do NOT clear ``self._sparse_flows[ns_idx]`` — leaving it
            # populated post-backward is what lets the sparse conditional
            # query read the per-active-row flow without re-running backward
            # (matches the contract on plain ``SparseProdLayer``).

            sparse_cs = ns.sparse_input_ns

            # Hand the SAME container to the upstream SparseIOSumLayer —
            # its backward reads sv_flow.values at K_out positions.
            sum_layer, sum_ns_idx = self._dense_sum_refs[ns_idx]
            sum_layer._sparse_flows[sum_ns_idx] = sv_flow

            # Emission param flow accumulation (unchanged from SparseProdLayer).
            torch.cuda.nvtx.range_push(f"ns[{ns_idx}]/custom_backward_sparse")
            sparse_cs.dist.custom_backward_sparse(
                input_layer=self._sparse_input_layers[ns_idx],
                sparse_flow=sv_flow,
                csc_pflows_base=sparse_cs._param_flow_range[0],
                logspace_flows=logspace_flows,
            )
            torch.cuda.nvtx.range_pop()

            # Note: we deliberately do NOT scatter into ``node_flows`` here.
            # The dense child is a :class:`SparseIOSumLayer` whose backward
            # reads ``self._sparse_flows[blk_idx]`` (set above), not
            # ``node_flows``; and the sparse-input flow is exposed via
            # :attr:`SparseProdLayer._sparse_flows` for downstream consumers
            # (e.g. :func:`pyjuice.queries.conditional`) — see ``_sparse_flow_owner``
            # for the input-ns → owning-layer back-reference. ``pc.node_flows``
            # at chain-interior and SparseCategorical-input rows stays zero on
            # the sparse path, mirroring how forward leaves ``node_mars``
            # untouched there.

        torch.cuda.nvtx.range_pop()
        return None
