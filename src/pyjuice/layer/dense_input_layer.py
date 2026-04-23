from __future__ import annotations

from typing import Optional, Sequence

import torch

from pyjuice.nodes import InputNodes
from .input_layer import InputLayer


class DenseCategoricalInputLayer(InputLayer):
    """Inference-only fast path for ``Categorical`` input layers whose ``InputNodes``
    form a dense ``[V, K, C]`` layout: every group has scope size 1, the same
    ``num_nodes = K``, and the same ``num_cats = C``. Callers opt in by passing
    ``use_dense_categorical_input_layer=True`` to ``TensorCircuit``; the class
    asserts the layout at construction and caches the view metadata that
    :func:`pyjuice.queries.conditional` consumes on the backward pass.

    Forward and training paths are inherited unchanged from ``InputLayer``.
    Parameter flow accumulation is unaffected; the only behavior this class
    overrides is the conditional backward, which replaces the atomic-add
    scatter kernel with a single ``torch.bmm`` / ``torch.matmul``.
    """

    def __init__(self, nodes: Sequence[InputNodes], cum_nodes: int = 0,
                 pc_num_vars: int = 0, max_tied_ns_per_parflow_block: int = 4) -> None:
        super().__init__(
            nodes = nodes, cum_nodes = cum_nodes, pc_num_vars = pc_num_vars,
            max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block,
        )

        if self.dist_signature != "Categorical":
            raise ValueError(
                "DenseCategoricalInputLayer requires Categorical distributions; "
                f"got {self.dist_signature}."
            )

        # All groups must share one num_cats.
        ncats = self.metadata[self.s_mids]
        if ncats.numel() == 0 or not torch.all(ncats == ncats[0]).item():
            raise ValueError(
                "DenseCategoricalInputLayer requires uniform num_cats across "
                "all InputNodes; got varying metadata."
            )
        C = int(ncats[0].item())

        if self.vids.dim() != 2 or self.vids.size(1) != 1:
            raise ValueError(
                "DenseCategoricalInputLayer requires scope size 1 per node "
                "(num_vars_per_node == 1)."
            )

        # Nodes must be grouped contiguously by variable with uniform K.
        nv = self.vids[:, 0]
        unique_nv, counts = torch.unique_consecutive(nv, return_counts = True)
        V = unique_nv.numel()
        N = self.num_nodes
        if V == 0 or N % V != 0:
            raise ValueError(
                "DenseCategoricalInputLayer requires num_nodes divisible by "
                f"num_variables; got {N} nodes over {V} variables."
            )
        K = N // V
        if not torch.all(counts == K).item():
            raise ValueError(
                "DenseCategoricalInputLayer requires every variable to have "
                f"the same number of nodes K={K}."
            )
        if unique_nv.unique().numel() != V:
            raise ValueError(
                "DenseCategoricalInputLayer requires each variable to appear "
                "as a single contiguous block of nodes."
            )

        # s_pids must follow Categorical's default per-node stride of C.
        s_pids = self.s_pids
        cat_off = torch.arange(K, device = s_pids.device, dtype = s_pids.dtype) * C
        bases = s_pids[::K].contiguous()  # [V]
        expected = bases.unsqueeze(1) + cat_off.unsqueeze(0)
        if not torch.equal(s_pids.view(V, K), expected):
            raise ValueError(
                "DenseCategoricalInputLayer requires s_pids to follow the "
                "default Categorical offset pattern (base_v + k*C)."
            )

        tied = bool(torch.all(bases == bases[0]).item())
        contiguous = False
        if not tied:
            expected_bases = bases[0] + torch.arange(
                V, device = bases.device, dtype = bases.dtype
            ) * (K * C)
            contiguous = bool(torch.equal(bases, expected_bases))

        self._dense_V = V
        self._dense_K = K
        self._dense_C = C
        self._dense_tied = tied
        self._dense_contiguous = contiguous
        self._dense_base0 = int(bases[0].cpu().item())
        self.register_buffer("_dense_bases", bases.clone())
        self.register_buffer("_dense_vids_order", unique_nv.clone().to(torch.long))

    def dense_conditional_backward(self, node_flows: torch.Tensor,
                                   params: torch.Tensor,
                                   target_vars: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Matmul-based replacement for the atomic-add categorical backward.

        Returns ``cat_probs`` with shape ``[B, num_target_vars, C]`` (normalized
        across the category axis) matching the contract of
        :func:`_categorical_backward`.
        """
        V, K, C = self._dense_V, self._dense_K, self._dense_C
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]
        B = node_flows.size(1)
        device = node_flows.device

        flows = node_flows[sid:eid].view(V, K, B)

        if self._dense_tied:
            base = self._dense_base0
            params_vw = params[base : base + K * C].view(K, C)
            # cat_probs[v, c, b] = sum_k params[k, c] * flows[v, k, b]
            cat_probs_layer = torch.matmul(flows.transpose(1, 2), params_vw).transpose(1, 2).contiguous()
        elif self._dense_contiguous:
            base = self._dense_base0
            params_vw = params[base : base + V * K * C].view(V, K, C)
            cat_probs_layer = torch.bmm(params_vw.transpose(1, 2), flows)
        else:
            bases = self._dense_bases
            cat_ids = torch.arange(C, device = device, dtype = bases.dtype)
            k_ids = torch.arange(K, device = device, dtype = bases.dtype)
            param_idx = (
                bases.view(V, 1, 1)
                + k_ids.view(1, K, 1) * C
                + cat_ids.view(1, 1, C)
            )
            params_vw = params[param_idx]
            cat_probs_layer = torch.bmm(params_vw.transpose(1, 2), flows)

        vids_order = self._dense_vids_order
        num_vars = int(self.vids.max().item()) + 1

        if target_vars is None:
            cat_probs = torch.zeros(num_vars, C, B, dtype = cat_probs_layer.dtype, device = device)
            cat_probs.index_copy_(0, vids_order, cat_probs_layer)
        else:
            rev = torch.full((num_vars,), -1, dtype = torch.long, device = device)
            target_t = torch.as_tensor(target_vars, dtype = torch.long, device = device)
            rev[target_t] = torch.arange(len(target_vars), device = device)
            output_idx = rev[vids_order]
            keep = output_idx >= 0
            num_target_vars = len(target_vars)
            cat_probs = torch.zeros(num_target_vars, C, B, dtype = cat_probs_layer.dtype, device = device)
            if keep.any():
                cat_probs.index_copy_(0, output_idx[keep], cat_probs_layer[keep])

        cat_probs = cat_probs / (cat_probs.sum(dim = 1, keepdim = True) + 1e-12)
        return cat_probs.permute(2, 0, 1)
