from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence, List, Tuple, Optional

from pyjuice.nodes import SumNodes
from pyjuice.utils.parameter_list import FastParamList
from .bitonic_topk import BitonicScratch, bitonic_topk
from .layer import Layer
from .prod_layer import ProdLayer


class TopKLayer(ProdLayer):
    """
    Dynamic block-shared top-K reduction. Sits between a dense ``ProdLayer``
    and a :class:`TopKSumLayer`. For each annotated :class:`SumNodes`, this
    layer reads the contiguous slice of dense product activations in
    ``element_mars`` belonging to that sum's child block, picks the K
    log-values with the largest magnitude (one set of K shared across all
    parent rows in the summate, recomputed every forward), and writes the
    selected ``(index, value)`` pairs into the circuit-level
    ``topk_indices`` / ``topk_values`` buffers shared with
    :class:`TopKSumLayer`.

    Subclasses :class:`ProdLayer` so ``is_prod()`` returns ``True``: that
    lets the existing ``layer_group.is_prod()`` dispatch in
    :meth:`TensorCircuit.forward` / :meth:`TensorCircuit.backward` carry
    this layer through without introducing a fourth layer category (a
    refactor that would touch a dozen branch sites).

    The opt-in path is per-:func:`pyjuice.summate` annotation
    (``summate(..., topk=K)`` sets ``ns._topk_k``); the compiler routes
    eligible sum nodes through this layer in
    :meth:`TensorCircuit._init_layers`.

    The selection runs :func:`bitonic_topk` (Triton, two-stage chunk
    sort + grouped merge). An earlier custom Triton kernel fixed
    ``H_PADDED`` as ``tl.constexpr``; at H=32k that materialised a
    single ~128 KB register tile and ptxas spent minutes planning
    spills. The bitonic version avoids that failure mode by tiling H
    into power-of-two chunks (each program holds at most
    ``next_po2(chunk_size) = 128`` elements in registers) and merging
    the per-chunk results hierarchically. Indices are produced as
    int32 directly, matching the circuit-level ``topk_indices``
    buffer dtype — no int64 scratch / narrowing copy needed.
    """

    def __init__(self, nodes: Sequence[SumNodes], slot_start: int = 0) -> None:
        # Bypass ``ProdLayer.__init__`` entirely — we have no block-sparse
        # ``cids`` / ``parids`` bookkeeping to compile.
        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."
        assert len(nodes) == len(set(nodes)), "Input node list contains duplicates."

        for ns in nodes:
            assert getattr(ns, "_topk_k", None) is not None, (
                "TopKLayer: every SumNodes must have ``_topk_k`` set "
                "(via summate(..., topk=K))."
            )
            assert ns.is_block_dense, (
                "TopKLayer: every SumNodes must be block-dense."
            )
            assert len(ns.chs) == 1, (
                "TopKLayer: every SumNodes must have a single child group."
            )
            cs = ns.chs[0]
            assert cs.provided("_output_ind_range"), (
                "TopKLayer: child product layer must be compiled before "
                "this TopKLayer (TensorCircuit topo order)."
            )

        # Per-node forward metadata: (cid_start, H_total, slot_start, K).
        # ``H_total = num_ch_node_blocks * ch_block_size`` — total candidate
        # children in the dense product block feeding this sum. ``slot_start``
        # is the first row in the circuit-level ``topk_indices`` /
        # ``topk_values`` buffer this sum writes to; consecutive K rows
        # belong to this sum.
        groups: List[Tuple[int, int, int, int]] = []
        curr_slot = slot_start
        for ns in nodes:
            cs = ns.chs[0]
            cid_start = cs._output_ind_range[0]
            H_total = cs.num_node_blocks * cs.block_size
            K = ns._topk_k
            assert K < H_total, (
                f"TopKLayer: K={K} must be strictly less than H={H_total}; "
                "callers must fall back to plain SumLayer when K >= H."
            )
            groups.append((cid_start, H_total, curr_slot, K))
            # Stash the slot range on the SumNodes so the matching
            # :class:`TopKSumLayer` reads K log-values from the same place
            # without an extra book-keeping pass.
            ns._topk_slot_range = (curr_slot, curr_slot + K)
            curr_slot += K

        self._topk_groups: List[Tuple[int, int, int, int]] = groups
        self._slot_start = slot_start
        self._slot_end = curr_slot
        self.num_topk_slots = curr_slot - slot_start

        # Stubs for downstream introspection. TopKLayer adds no
        # ``node_mars`` rows and no parameters — it just rewrites the
        # ``topk_*`` side buffers each forward.
        self.num_nodes = 0
        self.num_edges = 0
        self._layer_nid_range = (0, 0)

        # Stub-out the ProdLayer book-keeping that partial-evaluation /
        # scope wiring would otherwise touch. This layer doesn't support
        # partial eval (selection is dynamic per-forward).
        self.num_fw_partitions = 0
        self.num_bk_partitions = 0
        self.use_block_sparse_edges = False
        self.partitioned_nids = FastParamList([])
        self.partitioned_cids = FastParamList([])
        self.partitioned_u_cids = FastParamList([])
        self.partitioned_parids = FastParamList([])

        # Ping-pong scratch for :func:`bitonic_topk`'s multi-level merge,
        # shared across topk groups so a single allocation covers the
        # largest (H_total, B, K) triple seen in this layer.
        self._bitonic_scratch: BitonicScratch = BitonicScratch()

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                _for_backward: bool = False, *,
                topk_indices: Optional[torch.Tensor] = None,
                topk_values: Optional[torch.Tensor] = None,
                **kwargs) -> None:
        # The recompute-during-backward dispatch in
        # ``TensorCircuit.backward`` calls ``inner_layer_groups[layer_id-1]
        # .forward(_for_backward=True)`` to restore ``element_mars`` (which
        # ``DenseSumLayer.forward`` overwrites in-place). TopKSumLayer
        # never modifies ``element_mars`` *or* the topk side buffers, so
        # the original forward's selection is still valid at backward
        # time — re-running selection per timestep is pure waste. Skip it.
        if _for_backward:
            return None

        assert topk_indices is not None and topk_values is not None, (
            "TopKLayer.forward requires `topk_indices` and `topk_values` "
            "buffers (allocated by TensorCircuit at __init__)."
        )

        for cid_start, H_total, slot_start, K in self._topk_groups:
            sl = element_mars[cid_start : cid_start + H_total, :]
            vals_out = topk_values[slot_start : slot_start + K, :]
            idxs_out = topk_indices[slot_start : slot_start + K, :]
            bitonic_topk(sl, K, out_vals = vals_out, out_idx = idxs_out,
                         scratch = self._bitonic_scratch)

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 logspace_flows: bool = False, **kwargs) -> None:
        # No-op. :class:`TopKSumLayer`'s backward atomic-adds the K selected
        # children's flows directly into ``element_flows`` via the stored
        # ``topk_indices`` from the most recent forward. Children outside
        # the top-K set get exactly zero gradient (``element_flows`` is
        # zero-initialised at the start of every backward), which is the
        # straight-through-w.r.t.-selection semantic.
        return None

    def is_prod(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            f"TopKLayer(num_topk_groups={len(self._topk_groups)}, "
            f"num_topk_slots={self.num_topk_slots})"
        )

    # --- partial-eval / scope plumbing --------------------------------- #

    def enable_partial_evaluation(self, *args, **kwargs):
        # No-op: the per-forward dynamic selection means a static
        # scope2localids map doesn't apply. Callers that enable partial
        # evaluation on a circuit also containing TopK summates get the
        # full TopK kernel cost (which is small) plus partial evaluation
        # on the rest of the layers.
        return None

    def _prepare_scope2nids(self, *args, **kwargs):
        # No node_mars-addressable outputs => nothing to register.
        if not (hasattr(self, "fw_scope2localids") and hasattr(self, "bk_scope2localids")):
            self.fw_scope2localids = dict()
            self.bk_scope2localids = dict()
        return []
