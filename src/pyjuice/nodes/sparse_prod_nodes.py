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
    :class:`SparseCategorical` distribution (identity block-sparse edges on
    that slot), plus zero or more non-input (sum) children.

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

        assert self.is_block_sparse(), \
            "SparseProdNodes requires block-sparse edges."
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
            torch.arange(self.num_node_blocks, dtype=self.edge_ids.dtype),
        ), "SparseProdNodes requires identity edges on the sparse slot."

        self.var_id: int = self.sparse_input_ns.scope.to_list()[0]

    @property
    def num_dense_chs(self) -> int:
        return len(self.dense_ch_idxs)

    @property
    def max_nnz_per_col(self) -> int:
        return self.sparse_input_ns.dist._max_nnz_per_col
