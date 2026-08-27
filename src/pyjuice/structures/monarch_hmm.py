from __future__ import annotations

import math
from typing import Optional

import torch

from pyjuice.nodes import multiply, summate, inputs, set_block_size
from pyjuice.nodes.distributions import Categorical


def monarch_block_size(num_latents: int) -> int:
    """Natural Monarch factorisation: ``num_blocks = block_size = sqrt(H)``
    square blocks. :func:`set_block_size` additionally needs a power of two."""
    bs = math.isqrt(num_latents)
    if bs * bs != num_latents:
        raise ValueError(f"num_latents={num_latents} must be a perfect square")
    if bs & (bs - 1):
        raise ValueError(f"sqrt(num_latents)={bs} must be a power of two")
    return bs


def block_diagonal_edge_ids(num_node_blocks: int) -> torch.Tensor:
    """``[[0..NB-1], [0..NB-1]]`` — the block-diagonal edge pattern picked up by
    :class:`BlockDiagonalSumLayer` at compile time."""
    return torch.arange(0, num_node_blocks)[None, :].repeat(2, 1)


def monarch_permutation(num_latents: int, permute_block_size: int) -> torch.Tensor:
    """Reshape-transpose-flatten permutation as a ``[H, 1]`` sparse product
    edge list (``multiply(..., edge_ids=..., sparse_edges=True)``), so the
    re-pairing happens per node — the Monarch transpose crosses block
    boundaries."""
    return (
        torch.arange(0, num_latents)
        .reshape(num_latents // permute_block_size, permute_block_size)
        .permute(1, 0)
        .reshape(num_latents)[:, None]
    )


def MonarchHMM(seq_length: int, num_latents: int, num_emits: int, homogeneous: bool = True,
               block_size: Optional[int] = None, permute_block_size: Optional[int] = None,
               bd1: Optional[torch.Tensor] = None, bd2: Optional[torch.Tensor] = None,
               beta: Optional[torch.Tensor] = None, gamma: Optional[torch.Tensor] = None):
    """
    Hidden Markov Model whose ``[H, H]`` transition is a single **Monarch**
    matrix: ``BD1 -> permutation -> BD2``, two block-diagonal sums with
    ``H // block_size`` square blocks re-paired by a reshape-transpose-flatten
    permutation. Parameter count is ``2 * H * block_size`` instead of
    ``H ** 2`` (``2 * H ** 1.5`` for the natural ``block_size = sqrt(H)``).
    Both block-diagonal sums compile to :class:`BlockDiagonalSumLayer`, which
    supports param-flow accumulation, so the chain trains with EM like any
    other HMM.

    :param seq_length: sequence length
    :param num_latents: size of the latent space ``H``
    :param num_emits: size of the emission space
    :param homogeneous: tie the transition / emission parameters across time
    :param block_size: block size of the two block-diagonal sums (power of
        two, must divide ``H``). Defaults to ``sqrt(H)``, which requires
        ``H`` to be a perfect square with a power-of-two root.
    :param permute_block_size: block size of the permutation; defaults to
        ``block_size`` (the natural Monarch permutation). Must be a multiple
        of ``block_size``.
    :param bd1: optional ``[H // block_size, block_size, block_size]``
        parameters of the first block-diagonal sum (rows = output nodes)
    :param bd2: optional parameters of the second block-diagonal sum, same shape
    :param beta: optional ``[num_latents, num_emits]`` emission parameters
    :param gamma: optional ``[num_latents]`` initial-state distribution
    """
    if block_size is None:
        block_size = monarch_block_size(num_latents)
    assert block_size > 0 and (block_size & (block_size - 1)) == 0, "block_size must be a power of two"
    assert num_latents % block_size == 0, f"block_size ({block_size}) must divide num_latents ({num_latents})"
    if permute_block_size is None:
        permute_block_size = block_size
    assert permute_block_size % block_size == 0, "permute_block_size must be a multiple of block_size"
    assert num_latents % permute_block_size == 0, "permute_block_size must divide num_latents"

    num_node_blocks = num_latents // block_size
    bd_edges = block_diagonal_edge_ids(num_node_blocks)
    perm = monarch_permutation(num_latents, permute_block_size)

    with set_block_size(block_size = block_size):

        ns_input = inputs(
            seq_length - 1, num_node_blocks = num_node_blocks,
            dist = Categorical(num_cats = num_emits)
        )
        if beta is not None:
            assert beta.size(0) == num_latents and beta.size(1) == num_emits
            ns_input.set_params(beta)

        ns_bd1 = None
        ns_bd2 = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = homogeneous)

            if ns_bd1 is None:
                ns1 = summate(curr_zs, edge_ids = bd_edges, block_size = block_size)
                if bd1 is not None:
                    assert bd1.shape == (num_node_blocks, block_size, block_size)
                    ns1.set_params(bd1)
                ns_bd1 = ns1
            else:
                ns1 = ns_bd1.duplicate(curr_zs, tie_params = homogeneous)

            np_perm = multiply(ns1, edge_ids = perm, sparse_edges = True)

            if ns_bd2 is None:
                ns2 = summate(np_perm, edge_ids = bd_edges, block_size = block_size)
                if bd2 is not None:
                    assert bd2.shape == (num_node_blocks, block_size, block_size)
                    ns2.set_params(bd2)
                ns_bd2 = ns2
            else:
                ns2 = ns_bd2.duplicate(np_perm, tie_params = homogeneous)

            curr_zs = multiply(curr_xs, ns2)

        ns = summate(curr_zs, num_node_blocks = 1, block_size = 1)

        if gamma is not None:
            assert gamma.dim() == 1 and gamma.size(0) == num_latents
            ns.set_params(gamma.unsqueeze(0))

    return ns
