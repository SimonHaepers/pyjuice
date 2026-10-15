from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from .nodes import CircuitNodes
from .prod_nodes import ProdNodes
from .input_nodes import InputNodes
from .distributions import SparseCategorical


Tensor = Union[np.ndarray, torch.Tensor]


class SparseProdNodes(ProdNodes):
    """
    A :class:`ProdNodes` whose children satisfy the sparsity-propagating
    pattern: exactly one :class:`InputNodes` child with a
    :class:`SparseCategorical` distribution (identity edges on that slot),
    plus zero or more non-input (sum) children.

    Edges may be block-sparse (the usual case) or node-level
    (``sparse_edges=True``, ``edge_ids`` of shape ``[num_nodes, num_chs]``).
    Either way the sparse slot must be the identity; a dense slot may map
    output row ``h`` to any child node, which :meth:`dense_row_map` exposes
    as a per-row ``[num_nodes]`` index (e.g. the second permutation of a
    Monarch transition, ``P · BD2 · P · BD1``, folded into the emission
    product). Node-level dense maps must be injective so the backward can
    scatter flows with plain stores.

    The zero-dense-child case (``num_dense_chs == 0``) is the 1-child
    wrapper used by the compiler to bridge an :class:`InputNodes` at one
    depth to a sum at a deeper depth — mathematically an identity
    pass-through whose output is just ``log P(x | h)``. It compiles to a
    :class:`SparseProdLayer` whose consumer sum can pick
    :class:`SparseInputSumLayer`, avoiding the dense-sum-over-H fallback
    at the innermost HMM sum.

    Created automatically by :func:`pyjuice.multiply` when the pattern is
    detected, or explicitly via :func:`pyjuice.sparse_multiply`. The
    compilation step picks :class:`SparseProdLayer` for this subclass.

    All structural invariants are checked in ``__init__``; downstream code
    can rely on the attributes (``var_id``, ``sparse_input_ns``, etc.)
    without re-validating.
    """

    def __init__(self, num_node_blocks: int, chs: Sequence[CircuitNodes],
                 edge_ids: Optional[Tensor] = None, block_size: int = 0,
                 **kwargs) -> None:
        super().__init__(num_node_blocks, chs, edge_ids, block_size=block_size, **kwargs)

        sparse_ch_idxs = [
            i for i, cs in enumerate(self.chs)
            if isinstance(cs, InputNodes) and isinstance(cs.dist, SparseCategorical)
        ]
        assert len(sparse_ch_idxs) == 1, (
            f"SparseProdNodes requires exactly 1 SparseCategorical input child; "
            f"got {len(sparse_ch_idxs)}."
        )
        self.sparse_ch_idx: int = sparse_ch_idxs[0]
        self.sparse_input_ns: InputNodes = self.chs[self.sparse_ch_idx]
        self.dense_ch_idxs: List[int] = [
            i for i in range(len(self.chs)) if i != self.sparse_ch_idx
        ]

        for i in self.dense_ch_idxs:
            assert not isinstance(self.chs[i], InputNodes), (
                f"SparseProdNodes: dense children must be non-input; "
                f"chs[{i}] is {type(self.chs[i]).__name__}."
            )

        assert self.block_size == self.sparse_input_ns.block_size, (
            f"SparseProdNodes: block_size ({self.block_size}) must match sparse "
            f"child's block_size ({self.sparse_input_ns.block_size})."
        )
        assert self.num_node_blocks == self.sparse_input_ns.num_node_blocks, (
            f"SparseProdNodes: num_node_blocks ({self.num_node_blocks}) must "
            f"match sparse child's num_node_blocks "
            f"({self.sparse_input_ns.num_node_blocks})."
        )
        assert torch.equal(
            self.edge_ids[:, self.sparse_ch_idx],
            torch.arange(self.edge_ids.size(0), dtype=self.edge_ids.dtype),
        ), "SparseProdNodes requires identity edges on the sparse slot."
        if self.is_sparse():
            for i in self.dense_ch_idxs:
                col = self.edge_ids[:, i]
                assert torch.unique(col).numel() == col.numel(), (
                    f"SparseProdNodes: node-level edges to dense child {i} "
                    "must be injective."
                )

        self.var_id: int = self.sparse_input_ns.scope.to_list()[0]

    def dense_row_map(self, ch_idx: int) -> torch.Tensor:
        """``[num_nodes]`` long tensor: output row ``h`` of this product
        reads node ``dense_row_map(ch_idx)[h]`` of dense child ``ch_idx``
        (an index local to that child)."""
        eids = self.edge_ids[:, ch_idx].to(torch.long)
        if self.is_sparse():
            return eids
        h = torch.arange(self.num_nodes, dtype=torch.long)
        return eids[h // self.block_size] * self.block_size + h % self.block_size

    def has_identity_dense_maps(self) -> bool:
        """Whether every dense child is read at the product's own row ids."""
        ident = torch.arange(self.num_nodes, dtype=torch.long)
        return all(torch.equal(self.dense_row_map(i), ident) for i in self.dense_ch_idxs)

    @property
    def num_dense_chs(self) -> int:
        return len(self.dense_ch_idxs)

    @property
    def max_nnz_per_col(self) -> int:
        return self.sparse_input_ns.dist._max_nnz_per_col
