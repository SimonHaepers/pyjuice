from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from .nodes import CircuitNodes
from .prod_nodes import ProdNodes
from .input_nodes import InputNodes
from .distributions import BlockedCategorical


Tensor = Union[np.ndarray, torch.Tensor]


class BlockedProdNodes(ProdNodes):
    """
    A :class:`ProdNodes` whose children satisfy the blocked-emission pattern:
    exactly one :class:`InputNodes` child with a :class:`BlockedCategorical`
    distribution (identity block-sparse edges on that slot), plus zero or
    more non-input (sum) children. The blocked twin of
    :class:`SparseProdNodes`: an observed token activates one contiguous,
    block-aligned run of ``k`` latents instead of an arbitrary CSC column.

    The zero-dense-child case (``num_dense_chs == 0``) is the 1-child
    identity wrapper used at the start of an HMM chain.

    Created automatically by :func:`pyjuice.multiply` when the pattern is
    detected, or explicitly via :func:`pyjuice.blocked_multiply`. The
    compiler picks :class:`BlockedProdLayer` / :class:`CoBlockedProdLayer`
    for this subclass.
    """

    def __init__(self, num_node_blocks: int, chs: Sequence[CircuitNodes],
                 edge_ids: Optional[Tensor] = None, block_size: int = 0,
                 **kwargs) -> None:
        super().__init__(num_node_blocks, chs, edge_ids, block_size=block_size, **kwargs)

        blocked_ch_idxs = [
            i for i, cs in enumerate(self.chs)
            if isinstance(cs, InputNodes) and isinstance(cs.dist, BlockedCategorical)
        ]
        assert len(blocked_ch_idxs) == 1, (
            f"BlockedProdNodes requires exactly 1 BlockedCategorical input child; "
            f"got {len(blocked_ch_idxs)}."
        )
        self.blocked_ch_idx: int = blocked_ch_idxs[0]
        self.blocked_input_ns: InputNodes = self.chs[self.blocked_ch_idx]
        self.dense_ch_idxs: List[int] = [
            i for i in range(len(self.chs)) if i != self.blocked_ch_idx
        ]

        for i in self.dense_ch_idxs:
            assert not isinstance(self.chs[i], InputNodes), (
                f"BlockedProdNodes: dense children must be non-input; "
                f"chs[{i}] is {type(self.chs[i]).__name__}."
            )

        assert self.is_block_sparse(), \
            "BlockedProdNodes requires block-sparse edges."
        assert self.block_size == self.blocked_input_ns.block_size, (
            f"BlockedProdNodes: block_size ({self.block_size}) must match the "
            f"blocked child's block_size ({self.blocked_input_ns.block_size})."
        )
        assert self.num_node_blocks == self.blocked_input_ns.num_node_blocks, (
            f"BlockedProdNodes: num_node_blocks ({self.num_node_blocks}) must "
            f"match the blocked child's num_node_blocks "
            f"({self.blocked_input_ns.num_node_blocks})."
        )
        assert torch.equal(
            self.edge_ids[:, self.blocked_ch_idx],
            torch.arange(self.num_node_blocks, dtype=self.edge_ids.dtype),
        ), "BlockedProdNodes requires identity edges on the blocked slot."

        dist = self.blocked_input_ns.dist
        assert dist._states_per_block is not None, (
            "BlockedProdNodes: the BlockedCategorical child has no meta-parameters "
            "(pass token_block= / num_blocks= to `inputs`)."
        )
        assert dist.k % self.block_size == 0, (
            f"BlockedProdNodes: states per emission block (k={dist.k}) must be a "
            f"multiple of block_size ({self.block_size}) so every emission block "
            f"is a whole number of node blocks."
        )

        self.var_id: int = self.blocked_input_ns.scope.to_list()[0]

    @property
    def num_dense_chs(self) -> int:
        return len(self.dense_ch_idxs)

    @property
    def k(self) -> int:
        return self.blocked_input_ns.dist.k

    @property
    def num_blocks(self) -> int:
        return self.blocked_input_ns.dist.num_blocks
