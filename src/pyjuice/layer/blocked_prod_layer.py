from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

from pyjuice.nodes import ProdNodes, BlockedProdNodes
from pyjuice.nodes.distributions import BlockedCategorical
from .prod_layer import ProdLayer
from .input_layer import InputLayer
from .layer_group import LayerGroup
from .blocked_node_values import BlockedNodeValues, LOG_EPS


_FWD_BLOCK_K = 128
_FWD_BLOCK_B = 8
"""[BLOCK_B, BLOCK_K] tile of :func:`_blocked_prod_forward_kernel` /
:func:`_blocked_prod_scatter_flow_kernel`. Pure streaming kernels; both
knobs only trade program count against register pressure."""


class BlockedProdLayer(ProdLayer):
    """
    Blocked-emission product layer — the blocked twin of
    :class:`SparseProdLayer`. Each owned ``ns`` is a :class:`BlockedProdNodes`
    (one :class:`BlockedCategorical` input child + zero or more dense sum
    children).

    Forward produces a :class:`BlockedNodeValues` per ns holding the ``k``
    values of the emission block selected by the observed token::

        values[b, j] = log P(x_b | z = m_b·k + j) + Σ_ch node_mars[lookup[ch, m_b·k + j], b]
        m_b = token_block[data[var_id, b]]

    with a fused per-sample max. No index arrays, no host work: the kernel
    reads ``data`` and ``token_block`` itself. When every consumer is a
    blocked sum layer (``_skip_scatter=True``) nothing is written to
    ``element_mars``; otherwise the values are scattered into the dense
    ``element_mars`` slice (inactive rows ``LOG_EPS``) for plain consumers.

    Backward mirrors this: the consumer writes the ``[B, k]`` flow into
    ``self._blocked_flows[ns_idx]``; this layer scatters it to the dense
    children's ``node_flows`` and accumulates emission parameter flows via
    :meth:`BlockedCategorical.custom_backward_blocked`.

    ``missing_mask`` (marginalised positions) is not supported on the blocked
    chain yet — it would need an "all H rows" mode of the container.
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
            "BlockedProdLayer needs the compiled input_layer_group to resolve " \
            "the InputLayer that owns each BlockedCategorical child."
        for ns in self.nodes:
            assert isinstance(ns, BlockedProdNodes), (
                f"BlockedProdLayer expects BlockedProdNodes; got {type(ns).__name__}. "
                "Build via `juice.multiply` (auto-detect) or `juice.blocked_multiply`."
            )

        self._blocked_input_layers: List[InputLayer] = []
        self._dense_ch_lookups: List[str] = []
        for ns_idx, ns in enumerate(self.nodes):
            blocked_cs = ns.blocked_input_ns

            blocked_input_layer = None
            for lyr in input_layer_group:
                if blocked_cs in lyr.nodes:
                    blocked_input_layer = lyr
                    break
            assert blocked_input_layer is not None, (
                "BlockedProdLayer could not locate the InputLayer holding the "
                "BlockedCategorical input child."
            )
            self._blocked_input_layers.append(blocked_input_layer)

            # Per-row lookup for each dense child: global ``node_mars`` nid of
            # the h-th row of ns's output (same construction as SparseProdLayer).
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
                dense_lookup = torch.zeros(1, dtype=torch.long)
            buf_name = f"_dense_ch_lookup_{ns_idx}"
            self.register_buffer(buf_name, dense_lookup)
            self._dense_ch_lookups.append(buf_name)

            # Gate the InputLayer from populating this ns's node_mars / flows.
            blocked_cs._skip_input_forward = True
            blocked_cs._skip_input_backward = True
            blocked_cs._blocked_flow_owner = (self, ns_idx)

        self._blocked_outputs: List[Optional[BlockedNodeValues]] = [None] * len(self.nodes)
        self._blocked_flows: List[Optional[BlockedNodeValues]] = [None] * len(self.nodes)

        # Per-ns ``[B, k]`` forward workspaces + the shared ``[len(nodes), B]``
        # per-sample max buffer filled inline by the forward kernel.
        self._fwd_values_workspaces: List[Optional[torch.Tensor]] = [None] * len(self.nodes)
        self._fwd_max_workspace: Optional[torch.Tensor] = None

        # Set by ``TensorCircuit._mark_sparse_prod_scatter_skip`` when every
        # consumer reads the packed container directly.
        self._skip_scatter: bool = False

    def __repr__(self) -> str:
        return (
            f"BlockedProdLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_blocked_ns={len(self.nodes)})"
        )

    # ---------------- helpers ---------------- #

    def _values_workspace(self, ns_idx: int, batch_size: int, k: int,
                          device: torch.device) -> torch.Tensor:
        ws = self._fwd_values_workspaces[ns_idx]
        if ws is None or ws.device != device or ws.shape != (batch_size, k):
            ws = torch.empty(batch_size, k, dtype=torch.float32, device=device)
            self._fwd_values_workspaces[ns_idx] = ws
        return ws

    def _reset_max_workspace(self, batch_size: int, device: torch.device) -> None:
        if (self._fwd_max_workspace is None
                or self._fwd_max_workspace.device != device
                or self._fwd_max_workspace.shape != (len(self.nodes), batch_size)):
            self._fwd_max_workspace = torch.empty(
                len(self.nodes), batch_size, dtype=torch.float32, device=device,
            )
        self._fwd_max_workspace.fill_(float("-inf"))

    @staticmethod
    def _check_data(data: torch.Tensor, batch_size: int) -> torch.Tensor:
        assert data is not None, (
            "Blocked layers require `data` (the [num_vars, B] long token tensor "
            "on the device) as a kwarg."
        )
        assert data.dim() == 2 and data.size(1) == batch_size, (
            f"data must be [num_vars, B={batch_size}]; got {tuple(data.shape)}."
        )
        assert data.is_cuda, "Blocked layers read `data` on the device."
        if not data.is_contiguous():
            data = data.contiguous()
        return data

    # ---------------- Forward ---------------- #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, data: Optional[torch.Tensor] = None,
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert not self.provided("fw_partition_local_ids"), \
            "BlockedProdLayer does not support partial evaluation."
        assert missing_mask is None, (
            "missing_mask (marginalised positions) is not supported on the "
            "blocked chain yet."
        )
        batch_size = element_mars.size(1)
        data = self._check_data(data, batch_size)
        device = node_mars.device

        self._reset_max_workspace(batch_size, device)

        for ns_idx, ns in enumerate(self.nodes):
            bv = self._compute_blocked_output(ns_idx, ns, data, node_mars, batch_size)
            self._blocked_outputs[ns_idx] = bv
            if not self._skip_scatter:
                bv.scatter_to_dense(element_mars, ns._output_ind_range[0], fill_value=LOG_EPS)

        return None

    def _compute_blocked_output(self, ns_idx: int, ns: BlockedProdNodes,
                                data: torch.Tensor, node_mars: torch.Tensor,
                                batch_size: int) -> BlockedNodeValues:
        device = node_mars.device
        blocked_cs = ns.blocked_input_ns
        dist: BlockedCategorical = blocked_cs.dist
        input_layer = self._blocked_input_layers[ns_idx]
        H = ns.num_nodes
        k = dist.k
        values = self._values_workspace(ns_idx, batch_size, k, device)
        max_out = self._fwd_max_workspace[ns_idx]

        bv = BlockedNodeValues(
            values=values, k=k, num_rows=H, data=data, var_id=ns.var_id,
            token_block=dist._token_block, batch_size=batch_size, max_val=max_out,
        )

        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])
        BLOCK_K = min(triton.next_power_of_2(k), _FWD_BLOCK_K)
        BLOCK_B = min(triton.next_power_of_2(batch_size), _FWD_BLOCK_B)
        grid = (triton.cdiv(k, BLOCK_K), triton.cdiv(batch_size, BLOCK_B))
        _blocked_prod_forward_kernel[grid](
            params_ptr=input_layer.params,
            node_mars_ptr=node_mars,
            dense_ch_lookup_ptr=dense_ch_lookup,
            data_ptr=data,
            token_block_ptr=dist._token_block,
            values_out_ptr=values,
            max_out_ptr=max_out,
            var_id=ns.var_id,
            param_base=blocked_cs._param_range[0],
            num_rows=H,
            batch_size=batch_size,
            K=k,
            NUM_DENSE_CHS=ns.num_dense_chs,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
        )
        return bv

    # ---------------- Backward ---------------- #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False,
                 data: Optional[torch.Tensor] = None, **kwargs) -> None:
        if self._skip_scatter:
            for ns_idx, ns in enumerate(self.nodes):
                bv_flow = self._blocked_flows[ns_idx]
                assert bv_flow is not None, (
                    "BlockedProdLayer.backward (skip_scatter) expected a flow "
                    "container from the downstream blocked sum layer; none was "
                    "stashed. Did the sum layer's backward run?"
                )
                # Kept populated post-backward (same contract as SparseProdLayer).
                self._scatter_flow_to_children(ns_idx, ns, bv_flow, node_flows)
                ns.blocked_input_ns.dist.custom_backward_blocked(
                    input_layer=self._blocked_input_layers[ns_idx],
                    blocked_flow=bv_flow,
                    pflows_base=ns.blocked_input_ns._param_flow_range[0],
                    logspace_flows=logspace_flows,
                )
            return None

        # Dense fallback: plain-prod backward, then gather the emission flows.
        super().backward(
            node_flows=node_flows, element_flows=element_flows,
            logspace_flows=logspace_flows, **kwargs,
        )
        for ns_idx, ns in enumerate(self.nodes):
            bv = self._blocked_outputs[ns_idx]
            assert bv is not None, "BlockedProdLayer.backward called before forward."
            bv_flow = bv.gather_from_dense(element_flows, ns._output_ind_range[0])
            self._blocked_flows[ns_idx] = bv_flow
            ns.blocked_input_ns.dist.custom_backward_blocked(
                input_layer=self._blocked_input_layers[ns_idx],
                blocked_flow=bv_flow,
                pflows_base=ns.blocked_input_ns._param_flow_range[0],
                logspace_flows=logspace_flows,
            )
        return None

    def _scatter_flow_to_children(self, ns_idx: int, ns: BlockedProdNodes,
                                  bv_flow: BlockedNodeValues,
                                  node_flows: torch.Tensor) -> None:
        """``node_flows[lookup[ch, m_b·k + j], b] = flow[b, j]`` for every dense
        child (plain stores — each target is unique per child). ``node_flows``
        is zero-initialised by ``TensorCircuit.backward`` so inactive rows
        stay 0."""
        if ns.num_dense_chs == 0:
            return
        dense_ch_lookup = getattr(self, self._dense_ch_lookups[ns_idx])
        batch_size = node_flows.size(1)
        k = bv_flow.k
        BLOCK_K = min(triton.next_power_of_2(k), _FWD_BLOCK_K)
        BLOCK_B = min(triton.next_power_of_2(batch_size), _FWD_BLOCK_B)
        grid = (triton.cdiv(k, BLOCK_K), triton.cdiv(batch_size, BLOCK_B))
        _blocked_prod_scatter_flow_kernel[grid](
            node_flows_ptr=node_flows,
            dense_ch_lookup_ptr=dense_ch_lookup,
            data_ptr=bv_flow.data,
            token_block_ptr=bv_flow.token_block,
            values_ptr=bv_flow.values,
            var_id=bv_flow.var_id,
            num_rows=bv_flow.num_rows,
            batch_size=batch_size,
            K=k,
            NUM_DENSE_CHS=ns.num_dense_chs,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
        )


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit(
    do_not_specialize=["var_id", "param_base", "num_rows", "batch_size", "K"],
)
def _blocked_prod_forward_kernel(
    params_ptr, node_mars_ptr, dense_ch_lookup_ptr,
    data_ptr, token_block_ptr,
    values_out_ptr, max_out_ptr,
    var_id, param_base, num_rows, batch_size, K,
    NUM_DENSE_CHS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """Grid ``(cdiv(K, BLOCK_K), cdiv(B, BLOCK_B))``. For sample ``b`` and slot
    ``j``: ``v = data[var_id, b]``, ``m = token_block[v]``, ``row = m·K + j``,
    ``out = log(params[param_base + v·K + j]) + Σ_ch node_mars[lookup[ch, row], b]``,
    stored at ``values[b·K + j]``; ``atomic_max(max_out[b], max_j out)``."""
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_j = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size
    mask = mask_b[:, None] & (offs_j[None, :] < K)

    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask=mask_b, other=0)
    m = tl.load(token_block_ptr + v, mask=mask_b, other=0)

    val = tl.load(params_ptr + param_base + v[:, None] * K + offs_j[None, :],
                  mask=mask, other=1.0)
    out = tl.log(val)

    row = m[:, None] * K + offs_j[None, :]
    for ch in tl.static_range(NUM_DENSE_CHS):
        ch_nid = tl.load(dense_ch_lookup_ptr + ch * num_rows + row, mask=mask, other=0)
        out += tl.load(node_mars_ptr + ch_nid * batch_size + offs_b[:, None],
                       mask=mask, other=0.0)

    tl.store(values_out_ptr + offs_b[:, None] * K + offs_j[None, :], out, mask=mask)
    tile_max = tl.max(tl.where(mask, out, float("-inf")), axis=1)
    tl.atomic_max(max_out_ptr + offs_b, tile_max, mask=mask_b)


@triton.jit(
    do_not_specialize=["var_id", "num_rows", "batch_size", "K"],
)
def _blocked_prod_scatter_flow_kernel(
    node_flows_ptr, dense_ch_lookup_ptr,
    data_ptr, token_block_ptr, values_ptr,
    var_id, num_rows, batch_size, K,
    NUM_DENSE_CHS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_j = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size
    mask = mask_b[:, None] & (offs_j[None, :] < K)

    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask=mask_b, other=0)
    m = tl.load(token_block_ptr + v, mask=mask_b, other=0)
    row = m[:, None] * K + offs_j[None, :]

    flow = tl.load(values_ptr + offs_b[:, None] * K + offs_j[None, :], mask=mask, other=0.0)
    for ch in tl.static_range(NUM_DENSE_CHS):
        ch_nid = tl.load(dense_ch_lookup_ptr + ch * num_rows + row, mask=mask, other=0)
        tl.store(node_flows_ptr + ch_nid * batch_size + offs_b[:, None], flow, mask=mask)
