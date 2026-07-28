from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from .nodes import CircuitNodes
from .sum_nodes import SumNodes
from .sparse_prod_nodes import SparseProdNodes


Tensor = Union[np.ndarray, torch.Tensor]


class SparseSumNodes(SumNodes):
    """
    A :class:`SumNodes` whose single child is a :class:`SparseProdNodes`,
    enabling the column-selecting sparse fast paths at compile time
    (:class:`SparseInputSumLayer` for block-dense edges;
    :class:`SparseInputBlockDiagonalSumLayer` /
    :class:`SparseIOBlockDiagonalSumLayer` for the block-diagonal Monarch
    pattern). The subclass exists purely as a dispatch marker; all sum
    semantics are inherited.
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
            f"SparseSumNodes requires exactly 1 child (the SparseProdNodes); "
            f"got {len(self.chs)}."
        )
        assert isinstance(self.chs[0], SparseProdNodes), (
            f"SparseSumNodes requires a SparseProdNodes child; "
            f"got {type(self.chs[0]).__name__}."
        )
        assert self.is_block_dense or self._is_block_diagonal_edges(), (
            "SparseSumNodes requires block-dense edges (SparseInputSumLayer "
            "reads a block-dense parameter tile) or the block-diagonal "
            "pattern arange(NB)[None,:].repeat(2,1) (routed to the BD "
            "sparse fast paths)."
        )

    def _is_block_diagonal_edges(self) -> bool:
        """Structural replica of the compiler's
        ``_is_block_diagonal_pattern`` (kept local — ``nodes`` must not
        import ``model``), minus the NB >= 2 gate: at NB == 1 the pattern
        coincides with block-dense, which the assert above accepts anyway.
        """
        if self.num_node_blocks != self.num_ch_node_blocks:
            return False
        if self.block_size != self.ch_block_size:
            return False
        NB = self.num_node_blocks
        edge_ids = self.edge_ids
        if edge_ids.size(1) != NB:
            return False
        expected = torch.arange(NB, dtype=edge_ids.dtype,
                                device=edge_ids.device)
        return (torch.equal(edge_ids[0], expected)
                and torch.equal(edge_ids[1], expected))

    @property
    def sparse_prod_child(self) -> SparseProdNodes:
        return self.chs[0]
