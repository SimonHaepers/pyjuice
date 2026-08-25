from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.nodes import ProdNodes, BlockedProdNodes
from .blocked_prod_layer import BlockedProdLayer, _FWD_BLOCK_K, _FWD_BLOCK_B
from .blocked_node_values import BlockedNodeValues
from .layer_group import LayerGroup


class CoBlockedProdLayer(BlockedProdLayer):
    """Co-blocked product layer (twin of :class:`CoSparseProdLayer`): the
    emission input at ``var_id`` and the upstream :class:`BlockedIOSumLayer`
    output (whose output block was selected by the *same* ``var_id``) share
    one active block, so the log-space product is an element-wise add of the
    two packed ``[B, k]`` value tensors::

        values[b, j] = log P(x_b | z = m_b·k + j) + upstream[b, j]

    Requires ``num_dense_chs == 1``. Never writes ``element_mars``
    (``_skip_scatter=True`` unconditionally).
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
            assert isinstance(ns, BlockedProdNodes)
            assert ns.num_dense_chs == 1, (
                f"CoBlockedProdLayer requires exactly 1 dense (sum) child; got {ns.num_dense_chs}."
            )
        assert inner_layer_groups is not None, (
            "CoBlockedProdLayer needs inner_layer_groups to resolve the upstream "
            "BlockedIOSumLayer that owns each prod's dense child."
        )
        from .blocked_io_sum_layer import BlockedIOSumLayer

        self._blocked_sum_refs: List[Tuple] = []
        for ns in self.nodes:
            dense_ch_ns = ns.chs[ns.dense_ch_idxs[0]]
            found = None
            for lg in inner_layer_groups:
                if lg.is_prod():
                    continue
                for layer in lg:
                    if not isinstance(layer, BlockedIOSumLayer):
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
                "CoBlockedProdLayer: dense child sum ns is not owned by any "
                "BlockedIOSumLayer in inner_layer_groups — check TensorCircuit's "
                "blocked-chain classification."
            )
            self._blocked_sum_refs.append(found)

        self._skip_scatter = True

    def __repr__(self) -> str:
        return (
            f"CoBlockedProdLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_blocked_ns={len(self.nodes)})"
        )

    # ---------------- Forward ---------------- #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, data: Optional[torch.Tensor] = None,
                missing_mask: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        assert not self.provided("fw_partition_local_ids"), \
            "CoBlockedProdLayer does not support partial evaluation."
        assert missing_mask is None, (
            "missing_mask (marginalised positions) is not supported on the "
            "blocked chain yet."
        )
        batch_size = element_mars.size(1)
        data = self._check_data(data, batch_size)
        device = node_mars.device
        self._reset_max_workspace(batch_size, device)

        for ns_idx, ns in enumerate(self.nodes):
            blocked_cs = ns.blocked_input_ns
            dist = blocked_cs.dist
            input_layer = self._blocked_input_layers[ns_idx]
            k = dist.k
            H = ns.num_nodes

            sum_layer, sum_ns_idx = self._blocked_sum_refs[ns_idx]
            bv_up = sum_layer._blocked_outputs[sum_ns_idx]
            assert bv_up is not None, "upstream BlockedIOSumLayer has no cached output."
            assert bv_up.var_id == ns.var_id and bv_up.k == k, (
                "CoBlockedProdLayer: upstream sum output block pattern does not "
                "match this prod's emission pattern (expected the same var_id / k)."
            )

            values = self._values_workspace(ns_idx, batch_size, k, device)
            max_out = self._fwd_max_workspace[ns_idx]
            bv = BlockedNodeValues(
                values=values, k=k, num_rows=H, data=data, var_id=ns.var_id,
                token_block=dist._token_block, batch_size=batch_size, max_val=max_out,
            )
            self._blocked_outputs[ns_idx] = bv

            BLOCK_K = min(triton.next_power_of_2(k), _FWD_BLOCK_K)
            BLOCK_B = min(triton.next_power_of_2(batch_size), _FWD_BLOCK_B)
            grid = (triton.cdiv(k, BLOCK_K), triton.cdiv(batch_size, BLOCK_B))
            _co_blocked_log_add_kernel[grid](
                out_ptr=values,
                params_ptr=input_layer.params,
                up_values_ptr=bv_up.values,
                max_out_ptr=max_out,
                data_ptr=data,
                var_id=ns.var_id,
                param_base=blocked_cs._param_range[0],
                batch_size=batch_size,
                K=k,
                BLOCK_K=BLOCK_K,
                BLOCK_B=BLOCK_B,
            )
        return None

    # ---------------- Backward ---------------- #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False,
                 data: Optional[torch.Tensor] = None, **kwargs) -> None:
        # ∂out/∂log_emit = ∂out/∂upstream = 1: the incoming packed flow is the
        # outgoing flow on both inputs unchanged.
        for ns_idx, ns in enumerate(self.nodes):
            bv_flow = self._blocked_flows[ns_idx]
            assert bv_flow is not None, (
                "CoBlockedProdLayer.backward expected a packed flow from the "
                "downstream blocked sum layer."
            )
            sum_layer, sum_ns_idx = self._blocked_sum_refs[ns_idx]
            sum_layer._blocked_flows[sum_ns_idx] = bv_flow

            ns.blocked_input_ns.dist.custom_backward_blocked(
                input_layer=self._blocked_input_layers[ns_idx],
                blocked_flow=bv_flow,
                pflows_base=ns.blocked_input_ns._param_flow_range[0],
                logspace_flows=logspace_flows,
            )
        return None


@triton.jit(
    do_not_specialize=["var_id", "param_base", "batch_size", "K"],
)
def _co_blocked_log_add_kernel(
    out_ptr, params_ptr, up_values_ptr, max_out_ptr,
    data_ptr, var_id, param_base, batch_size, K,
    BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_j = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size
    mask = mask_b[:, None] & (offs_j[None, :] < K)

    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask=mask_b, other=0)
    val = tl.load(params_ptr + param_base + v[:, None] * K + offs_j[None, :],
                  mask=mask, other=1.0)
    up = tl.load(up_values_ptr + offs_b[:, None] * K + offs_j[None, :], mask=mask, other=0.0)
    out = tl.log(val) + up
    tl.store(out_ptr + offs_b[:, None] * K + offs_j[None, :], out, mask=mask)
    tile_max = tl.max(tl.where(mask, out, float("-inf")), axis=1)
    tl.atomic_max(max_out_ptr + offs_b, tile_max, mask=mask_b)
