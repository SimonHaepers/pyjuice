from __future__ import annotations

import torch
import numpy as np
from typing import Union, Sequence
from copy import deepcopy

from pyjuice.utils.context_manager import _DecoratorContextManager
from pyjuice.utils import BitSet
from .nodes import CircuitNodes
from .input_nodes import InputNodes
from .prod_nodes import ProdNodes
from .sum_nodes import SumNodes
from .sparse_prod_nodes import SparseProdNodes
from .sparse_sum_nodes import SparseSumNodes
from .distributions import Distribution, SparseCategorical
from pyjuice.graph import RegionGraph

Tensor = Union[np.ndarray,torch.Tensor]
ProdNodesChs = Union[SumNodes,InputNodes]
SumNodesChs = Union[ProdNodes,InputNodes]


def inputs(var: Union[int,Sequence[int]], num_node_blocks: int = 0, dist: Distribution = Distribution(), 
           params: Optional[Tensor] = None, num_nodes: int = 0, block_size: int = 0, **kwargs) -> InputNodes:
    """
    Construct a vector of input nodes defined on the same variable and have the same input distribution.

    :param var: the variable ID (or a list of variable IDs if the parameter `dist` is a multivariate distribution)
    :type var: Union[int,Sequence[int]]

    :param num_node_blocks: number of node blocks each of size `block_size`. I.e., the total number of nodes is `num_node_blocks` * `block_size`
    :type num_node_blocks: int

    :param dist: the input distribution from `pyjuice.nodes.distributions`
    :type dist: Distribution

    :param params: an optional tensor that defines the parameters of the nodes
    :type params: Optional[Tensor]

    :param num_nodes: an alternative of `num_node_blocks`; if set, it should be divicible by `block_size`
    :type num_nodes: int

    :param block_size: block size of the nodes; it does not change the semantics of the nodes, but will affect the speed of the compiled PC
    :type block_size: int

    :returns: an `InputNodes` object (a subclass of `CircuitNodes`)
    """

    assert block_size == 0 or block_size & (block_size - 1) == 0, "`block_size` must be a power of 2."

    if num_nodes > 0:
        assert num_node_blocks == 0, "Only one of `num_nodes` and `num_node_blocks` can be set at the same time."
        if block_size == 0:
            block_size = CircuitNodes.DEFAULT_BLOCK_SIZE

        assert num_nodes % block_size == 0

        num_node_blocks = num_nodes // block_size

    return InputNodes(
        num_node_blocks = num_node_blocks,
        scope = [var] if isinstance(var, int) else var,
        dist = dist,
        params = params,
        block_size = block_size,
        **kwargs
    )


def multiply(nodes1: ProdNodesChs, *args, edge_ids: Optional[Tensor] = None, sparse_edges: bool = False,
             _force_plain: bool = False, **kwargs) -> ProdNodes:
    """
    Construct a vector of product nodes given a list of children PCs defined on disjoint sets of variables.

    :note: It requires all children to have the same block size.

    :note: If all children have the same `num_node_blocks`, if `edge_ids` is not specified, `multiply` outputs a vector of `num_node_blocks` * `block_size` nodes, where the ith (product) node is connected to the ith node in every child.

    :note: By default every edge specified by `edge_ids` denotes a block of `block_size` edges, unless `sparse_edge` is set to `True`, where the block size is assumed to be 1.

    :param nodes1: the first child node
    :type nodes1: Union[SumNodes,InputNodes]

    :param args: the remaining child nodes
    :type args: Union[SumNodes,InputNodes]

    :param edge_ids: a matrix of size [# product node blocks, # children] - the ith product node block is connected to the `edge_ids[i,j]`th node block in the jth child
    :type edge_ids: Optional[Tensor]

    :param sparse_edges: if set to `True`, the size of `edge_ids` becomes [# product nodes, # children] (i.e., we assume the block size of every child is 1)
    :type sparse_edges: bool

    :returns: an `ProdNodes` object (a subclass of `CircuitNodes`)
    """

    assert isinstance(nodes1, SumNodes) or isinstance(nodes1, InputNodes), "Children of product nodes must be input or sum nodes." 

    chs = [nodes1]
    num_node_blocks = nodes1.num_node_blocks
    block_size = nodes1.block_size
    scope = deepcopy(nodes1.scope)

    for nodes in args:
        assert nodes.is_input() or nodes.is_sum(), f"Children of product nodes must be input or sum nodes, but found input of type {type(nodes)}."
        if edge_ids is None:
            assert nodes.num_node_blocks == num_node_blocks, f"Input nodes should have the same `num_node_blocks`, but got {nodes.num_node_blocks} and {num_node_blocks}."
        assert nodes.block_size == block_size, "Input nodes should have the same `num_node_blocks`."
        if not RegionGraph.ALLOW_NONDECOMPOSABLE:
            assert len(nodes.scope & scope) == 0, "Children of a `ProdNodes` should have disjoint scopes."
        chs.append(nodes)
        scope |= nodes.scope

    if edge_ids is not None:
        if sparse_edges:
            assert edge_ids.shape[0] % block_size == 0
            num_node_blocks = edge_ids.shape[0] // block_size
        else:
            num_node_blocks = edge_ids.shape[0]
        assert edge_ids.shape[0] == num_node_blocks or edge_ids.shape[0] == num_node_blocks * block_size

    cls = SparseProdNodes if (not _force_plain and _is_sparse_prod_pattern(chs)) else ProdNodes
    ns = cls(num_node_blocks, chs, edge_ids, block_size = block_size, **kwargs)
    if _force_plain:
        ns._force_plain_layer = True
    return ns


def summate(nodes1: SumNodesChs, *args, num_node_blocks: int = 0, num_nodes: int = 0,
            edge_ids: Optional[Tensor] = None, block_size: int = 0,
            sum_edge_ids_constructor: Optional[Callable] = None,
            _force_plain: bool = False, **kwargs) -> SumNodes:
    """
    Construct a vector of sum nodes given a list of children PCs defined on the same sets of variables.

    :note: It requires all children to have the same block size.

    :note: If `edge_ids` is not set, `summate` defines a set of sum nodes fully connected to the input nodes.

    :param nodes1: the first child node
    :type nodes1: Union[ProdNodes,InputNodes]

    :param args: the remaining child nodes
    :type args: Union[ProdNodes,InputNodes]

    :param num_node_blocks: number of node blocks each of size `block_size`. I.e., the total number of nodes is `num_node_blocks` * `block_size`
    :type num_node_blocks: int

    :param num_nodes: an alternative of `num_node_blocks`; if set, it should be divicible by `block_size`
    :type num_nodes: int

    :param edge_ids: a matrix of size [2, # edges] - every size-2 column vector [i,j] defines a set of edges that fully connect the ith sum node block and the jth child node block
    :type edge_ids: Optional[Tensor]

    :param block_size: block size of the nodes; it does not change the semantics of the nodes, but will affect the speed of the compiled PC
    :type block_size: int

    :param sum_edge_ids_constructor: optional helper functions to create special edge patterns (e.g., block-sparse)
    :type sum_edge_ids_constructor: Callable

    :returns: an `SumNodes` object (a subclass of `CircuitNodes`)
    """

    assert block_size == 0 or block_size & (block_size - 1) == 0, "`block_size` must be a power of 2."

    if num_nodes > 0:
        assert num_node_blocks == 0, "Only one of `num_nodes` and `num_node_blocks` can be set at the same time."
        if block_size == 0:
            block_size = CircuitNodes.DEFAULT_BLOCK_SIZE

        assert num_nodes % block_size == 0

        num_node_blocks = num_nodes // block_size

    assert isinstance(nodes1, ProdNodes) or isinstance(nodes1, InputNodes), f"Children of sum nodes must be input or product nodes, but found input of type {type(nodes1)}." 

    if sum_edge_ids_constructor is not None:
        edge_ids = sum_edge_ids_constructor(
            nodes1, *args, 
            num_node_blocks = num_node_blocks, 
            block_size = block_size if block_size > 0 else CircuitNodes.DEFAULT_BLOCK_SIZE, 
            **kwargs
        )

    if edge_ids is not None and num_node_blocks == 0:
        num_node_blocks = edge_ids[0,:].max().item() + 1

    assert num_node_blocks > 0, "Number of node blocks should be greater than 0."

    chs = [nodes1]
    scope = deepcopy(nodes1.scope)
    for nodes in args:
        assert isinstance(nodes, ProdNodes) or isinstance(nodes, InputNodes), f"Children of sum nodes must be input or product nodes, but found input of type {type(nodes)}."
        if not RegionGraph.ALLOW_NONSMOOTH:
            assert nodes.scope == scope, "Children of a `SumNodes` should have the same scope."
        chs.append(nodes)

    cls = SumNodes
    if (not _force_plain) and len(chs) == 1 and isinstance(chs[0], SparseProdNodes):
        cls = SparseSumNodes
    ns = cls(num_node_blocks, chs, edge_ids, block_size = block_size, **kwargs)
    if _force_plain:
        # Pin this node so the compile-time structural fallback in
        # ``TensorCircuit._sparse_sum_eligible`` does not promote it to the
        # :class:`SparseInputSumLayer` fast path. Used by tests that need
        # param-flow accumulation through the sums (the sparse sum layer is
        # inference-only).
        ns._force_plain_layer = True
    return ns


def _is_sparse_prod_pattern(chs) -> bool:
    """Structural detector for the sparse-prod pattern used by auto-dispatch
    in :func:`multiply`. Full invariant checks happen in
    :class:`SparseProdNodes.__init__` — this is just the cheap gate so we
    don't build a ``SparseProdNodes`` over children that obviously don't
    match.

    Two acceptable shapes:
      * ``len(chs) >= 2`` — one :class:`SparseCategorical` input + one or
        more non-input children (the usual HMM emission × transition prod).
      * ``len(chs) == 1`` — a single :class:`SparseCategorical` input,
        used as the identity pass-through wrap the compiler inserts
        between an input and a deeper-depth sum."""
    if len(chs) == 0:
        return False
    sparse_ch_idxs = [
        i for i, cs in enumerate(chs)
        if isinstance(cs, InputNodes) and isinstance(cs.dist, SparseCategorical)
    ]
    if len(sparse_ch_idxs) != 1:
        return False
    return all(not isinstance(cs, InputNodes)
               for i, cs in enumerate(chs) if i != sparse_ch_idxs[0])


def sparse_multiply(nodes1: ProdNodesChs, *args, **kwargs) -> SparseProdNodes:
    """
    Explicit sparse counterpart of :func:`multiply` — asserts that the resulting
    :class:`ProdNodes` is a :class:`SparseProdNodes`. Useful when you want to
    fail loudly if the children don't satisfy the sparse-propagation pattern
    (one :class:`SparseCategorical` input + ≥1 non-input sibling, matching
    block sizes, identity edges on the sparse slot).

    Identical to ``multiply`` for eligible children; raises a clear error
    otherwise.
    """
    ns = multiply(nodes1, *args, **kwargs)
    assert isinstance(ns, SparseProdNodes), (
        "sparse_multiply: children do not match the sparse-prod pattern "
        "(need exactly one SparseCategorical input child + ≥1 non-input "
        "sibling). Use `multiply` instead, or adjust the children."
    )
    return ns


def sparse_summate(nodes1: SumNodesChs, *args, **kwargs) -> SparseSumNodes:
    """
    Explicit sparse counterpart of :func:`summate` — asserts that the resulting
    :class:`SumNodes` is a :class:`SparseSumNodes`. A ``SparseSumNodes`` is
    only produced when the single child is a :class:`SparseProdNodes` with
    block-dense edges.
    """
    ns = summate(nodes1, *args, **kwargs)
    assert isinstance(ns, SparseSumNodes), (
        "sparse_summate: the single child must be a `SparseProdNodes` "
        "(built via `multiply` / `sparse_multiply` over a SparseCategorical "
        "input)."
    )
    return ns


class set_block_size(_DecoratorContextManager):
    """
    Context-manager that sets the block size of PC nodes constructed by `inputs`, `multiply`, and `summate`.

    :param block_size: the target block size
    :type block_size: int

    Example::
        >>> with pyjuice.set_block_size(16):
        ...     nis = pyjuice.inputs(var = 0, num_node_blocks = 4, dist = Categorical(num_cats = 20))
        >>> print(nis.num_nodes) # Should be 16 * 4
        64
    """

    def __init__(self, block_size: int = 1):

        assert block_size & (block_size - 1) == 0, "`block_size` must be a power of 2."

        self.block_size = block_size
        
        self.original_block_size = None

    def __enter__(self) -> None:
        self.original_block_size = CircuitNodes.DEFAULT_BLOCK_SIZE
        CircuitNodes.DEFAULT_BLOCK_SIZE = self.block_size

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        CircuitNodes.DEFAULT_BLOCK_SIZE = self.original_block_size


class structural_properties(_DecoratorContextManager):
    """
    Context-manager that controls the assertions of circuit structural properties, including smoothness and decomposability.

    :param allow_nonsmooth: whether to allow non-smooth circuits
    :type allow_nonsmooth: bool

    :param allow_nondecomposable: whether to allow non-decomposable circuits
    :type allow_nondecomposable: bool

    Example::
        >>> with pyjuice.structural_properties(allow_nonsmooth = True):
        ...     nis = pyjuice.inputs(var = 0, num_node_blocks = 4, dist = Categorical(num_cats = 20))
        ...     ....
    """

    def __init__(self, allow_nonsmooth = False, allow_nondecomposable = False):

        self.allow_nonsmooth = allow_nonsmooth
        self.allow_nondecomposable = allow_nondecomposable

    def __enter__(self) -> None:
        RegionGraph.ALLOW_NONSMOOTH = self.allow_nonsmooth
        RegionGraph.ALLOW_NONDECOMPOSABLE = self.allow_nondecomposable

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        RegionGraph.ALLOW_NONSMOOTH = False
        RegionGraph.ALLOW_NONDECOMPOSABLE = False
