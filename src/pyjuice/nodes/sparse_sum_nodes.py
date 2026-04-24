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
    enabling the :class:`SparseInputSumLayer` column-selecting fast path at
    compile time. The subclass exists purely as a dispatch marker; all sum
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
        assert self.is_block_dense, (
            "SparseSumNodes requires block-dense edges (the SparseInputSumLayer "
            "fast path reads a block-dense parameter tile)."
        )

    @property
    def sparse_prod_child(self) -> SparseProdNodes:
        return self.chs[0]
