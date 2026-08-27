from __future__ import annotations

from typing import Optional, Tuple

import torch

from pyjuice.nodes import summate, inputs, set_block_size, sparse_multiply, sparse_summate
from pyjuice.nodes.distributions import SparseCategorical
from pyjuice.utils.util import max_cdf_power_of_2


def _cover_rows_and_cols(support: torch.Tensor, scores: torch.Tensor) -> None:
    """In-place: make sure every row (latent) and every column (token) of the
    boolean ``[H, V]`` ``support`` keeps at least one entry, re-adding the
    highest-``scores`` entry of any empty row / column. A token outside the
    support has probability exactly zero, so one occurrence in the data would
    make a whole sequence's LL ``-inf``."""
    empty_rows = ~support.any(dim = 1)
    if empty_rows.any():
        rows = empty_rows.nonzero(as_tuple = True)[0]
        support[rows, scores[rows].argmax(dim = 1)] = True
    empty_cols = ~support.any(dim = 0)
    if empty_cols.any():
        cols = empty_cols.nonzero(as_tuple = True)[0]
        support[scores[:, cols].argmax(dim = 0), cols] = True


def dense_to_csc(beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense ``[H, V]`` probability matrix (exact zeros off-support) to the
    CSC arrays consumed by :class:`SparseCategorical` / :func:`SparseHMM`:
    ``(csc_indptr [V+1] long, csc_indices [nnz] long, csc_values [nnz] float32)``,
    all on CPU."""
    csc = beta.detach().cpu().to_sparse().to_sparse_csc()
    return (
        csc.ccol_indices().to(torch.long).contiguous(),
        csc.row_indices().to(torch.long).contiguous(),
        csc.values().to(torch.float32).contiguous(),
    )


def prune_emissions_to_csc(beta: torch.Tensor, density: float,
                           tokens_present: Optional[torch.Tensor] = None):
    """Magnitude-prune a dense ``[H, V]`` emission matrix (probability space)
    to the target ``density`` (fraction of entries kept), row-normalise the
    survivors and return them in CSC form.

    Every latent keeps at least one token and every token keeps at least one
    latent (see :func:`_cover_rows_and_cols`), so the actual density can be
    marginally above the target.

    :param tokens_present: optional long tensor of token ids; if given, only
        these columns are guaranteed coverage (the others may be empty)
    :returns: ``(csc_indptr, csc_indices, csc_values, actual_density)``
    """
    beta = beta.detach().float().cpu()
    H, V = beta.shape
    assert 0.0 < density <= 1.0
    k = max(int(round(density * H * V)), 1)
    if k >= H * V:
        support = torch.ones(H, V, dtype = torch.bool)
    else:
        threshold = beta.reshape(-1).kthvalue(H * V - k + 1).values
        support = beta >= threshold
    if tokens_present is None:
        _cover_rows_and_cols(support, beta)
    else:
        # Rows first (all columns eligible), then only the requested columns.
        empty_rows = ~support.any(dim = 1)
        if empty_rows.any():
            rows = empty_rows.nonzero(as_tuple = True)[0]
            support[rows, beta[rows].argmax(dim = 1)] = True
        cols = torch.as_tensor(tokens_present, dtype = torch.long).reshape(-1)
        cols = cols[~support[:, cols].any(dim = 0)]
        if cols.numel() > 0:
            support[beta[:, cols].argmax(dim = 0), cols] = True

    pruned = beta * support
    pruned = pruned / pruned.sum(dim = 1, keepdim = True).clamp(min = 1e-30)
    actual_density = support.float().mean().item()
    return (*dense_to_csc(pruned), actual_density)


def random_emission_csc(num_latents: int, num_emits: int, density: float, seed: int = 0):
    """Random emission support at the target ``density`` (each entry kept
    i.i.d. with probability ``density``, plus row / column coverage), with
    row-normalised uniform-random values. Useful as a structure-only
    benchmark when no trained circuit is available.

    :returns: ``(csc_indptr, csc_indices, csc_values, actual_density)``
    """
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(num_latents, num_emits, generator = g)
    support = torch.rand(num_latents, num_emits, generator = g) < density
    _cover_rows_and_cols(support, raw)
    beta = raw * support
    beta = beta / beta.sum(dim = 1, keepdim = True).clamp(min = 1e-30)
    actual_density = support.float().mean().item()
    return (*dense_to_csc(beta), actual_density)


def SparseHMM(seq_length: int, num_latents: int, num_emits: int,
              csc_indptr: torch.Tensor, csc_indices: torch.Tensor,
              csc_values: Optional[torch.Tensor] = None,
              homogeneous: bool = True, block_size: Optional[int] = None,
              alpha: Optional[torch.Tensor] = None, gamma: Optional[torch.Tensor] = None):
    """
    Hidden Markov Model with a **fixed sparse emission support** (sparse-IO
    HMM): the ``[num_latents, num_emits]`` emission matrix is stored in CSC
    form by :class:`SparseCategorical` and the chain uses
    :func:`sparse_multiply` / :func:`sparse_summate`, so the forward /
    backward cost scales with the number of non-zeros rather than
    ``num_latents * num_emits``. EM only ever re-estimates the non-zero
    entries — the support never regrows.

    Obtain the CSC pattern from a trained dense circuit with
    :func:`prune_emissions_to_csc`, or from :func:`random_emission_csc`.

    :param csc_indptr: long ``[num_emits + 1]`` column pointers
    :param csc_indices: long ``[nnz]`` latent (row) ids, column-major order
    :param csc_values: optional ``[nnz]`` emission probabilities in CSC order;
        random (row-normalised) when omitted
    :param block_size: block size of the PC (power of two dividing
        ``num_latents``); defaults to ``min(max_pow2(num_latents), 1024)``
    :param alpha: optional ``[num_latents, num_latents]`` transition matrix
    :param gamma: optional ``[num_latents]`` initial-state distribution
    """
    if block_size is None:
        block_size = min(max_cdf_power_of_2(num_latents), 1024)
    assert num_latents % block_size == 0, f"block_size ({block_size}) must divide num_latents ({num_latents})"
    num_node_blocks = num_latents // block_size

    csc_indptr = torch.as_tensor(csc_indptr, dtype = torch.long).cpu().contiguous()
    csc_indices = torch.as_tensor(csc_indices, dtype = torch.long).cpu().contiguous()

    with set_block_size(block_size = block_size):

        ns_input = inputs(
            seq_length - 1, num_node_blocks = num_node_blocks,
            dist = SparseCategorical(num_cats = num_emits),
            csc_indptr = csc_indptr, csc_indices = csc_indices,
        )
        if csc_values is not None:
            csc_values = torch.as_tensor(csc_values, dtype = torch.float32).cpu().contiguous()
            assert csc_values.numel() == csc_indices.numel()
            ns_input.set_params(csc_values, normalize = False)

        ns_sum = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = homogeneous)

            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks = num_node_blocks)
                if alpha is not None:
                    assert alpha.size(0) == num_latents and alpha.size(1) == num_latents
                    ns.set_params(alpha)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params = homogeneous)

            curr_zs = sparse_multiply(curr_xs, ns)

        ns = sparse_summate(curr_zs, num_node_blocks = 1, block_size = 1)

        if gamma is not None:
            assert gamma.dim() == 1 and gamma.size(0) == num_latents
            ns.set_params(gamma.unsqueeze(0))

    return ns
