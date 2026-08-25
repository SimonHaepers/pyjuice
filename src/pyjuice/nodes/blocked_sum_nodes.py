from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from .nodes import CircuitNodes
from .sum_nodes import SumNodes
from .blocked_prod_nodes import BlockedProdNodes


Tensor = Union[np.ndarray, torch.Tensor]


class BlockedSumNodes(SumNodes):
    """
    A :class:`SumNodes` whose single child is a :class:`BlockedProdNodes`,
    with block-dense edges — the dispatch marker for the blocked sum fast
    paths (:class:`BlockedInputSumLayer` / :class:`BlockedIOSumLayer`).
    All sum semantics are inherited.
    """

    def __init__(self, num_node_blocks: int, chs: Sequence[CircuitNodes],
                 edge_ids: Optional[Union[Tensor, Sequence[Tensor]]] = None,
                 params: Optional[Tensor] = None,
                 zero_param_mask: Optional[Tensor] = None,
                 block_size: int = 0,
                 _presanitised_edge_ids: Optional[Tensor] = None,
                 **kwargs) -> None:
        super().__init__(
            num_node_blocks, chs, edge_ids=edge_ids, params=params,
            zero_param_mask=zero_param_mask, block_size=block_size,
            _presanitised_edge_ids=_presanitised_edge_ids, **kwargs,
        )
        assert len(self.chs) == 1, (
            f"BlockedSumNodes requires exactly 1 child (the BlockedProdNodes); "
            f"got {len(self.chs)}."
        )
        assert isinstance(self.chs[0], BlockedProdNodes), (
            f"BlockedSumNodes requires a BlockedProdNodes child; "
            f"got {type(self.chs[0]).__name__}."
        )
        assert self.is_block_dense, (
            "BlockedSumNodes requires block-dense edges (the blocked sum layers "
            "read contiguous [k, k] / [H, k] tiles of a block-dense parameter "
            "matrix)."
        )
        k = self.chs[0].k
        assert k % self.ch_block_size == 0, (
            f"BlockedSumNodes: states per emission block (k={k}) must be a "
            f"multiple of ch_block_size ({self.ch_block_size})."
        )

    @property
    def blocked_prod_child(self) -> BlockedProdNodes:
        return self.chs[0]
