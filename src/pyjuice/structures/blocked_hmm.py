from __future__ import annotations

from typing import Optional

import torch

from pyjuice.nodes import summate, inputs, set_block_size, blocked_multiply, blocked_summate
from pyjuice.nodes.distributions import BlockedCategorical
from pyjuice.utils.util import max_cdf_power_of_2


def frequency_balanced_partition(token_counts: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Greedy token → block assignment that balances the total token
    *frequency* per block (largest-first, least-loaded-block). A cheap stand-in
    for the Brown-cluster partition of Chiu & Rush (2020) when no clustering
    is available; ``token_counts`` of all ones gives a size-balanced split.

    :returns: long tensor ``[V]`` of block ids.
    """
    counts = torch.as_tensor(token_counts, dtype=torch.float64).reshape(-1)
    V = counts.numel()
    order = torch.argsort(counts, descending=True)
    load = torch.zeros(num_blocks, dtype=torch.float64)
    token_block = torch.empty(V, dtype=torch.long)
    for v in order.tolist():
        m = int(torch.argmin(load).item())
        token_block[v] = m
        load[m] += counts[v]
    return token_block


def context_cluster_partition(sequences: torch.Tensor, num_emits: int, num_blocks: int,
                              dim: int = 64, num_context: int = 2048, iters: int = 30,
                              seed: int = 0, device=None) -> torch.Tensor:
    """Cheap stand-in for the Brown-cluster token partition of Chiu & Rush
    (2020): spherical k-means over SVD-reduced PPMI vectors of each token's
    left/right bigram contexts (restricted to the ``num_context`` most
    frequent context tokens). Tokens with similar contexts land in the same
    block, which is what lets a block's ``k`` states specialise. Every block
    is guaranteed at least one token.

    :param sequences: long tensor ``[N, T]`` (or flat) of training tokens.
    :returns: long tensor ``[num_emits]`` of block ids (on CPU).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(sequences).reshape(-1).to(torch.long)
    if device is not None:
        x = x.to(device)
    V, M = num_emits, num_blocks
    counts = torch.bincount(x, minlength=V).float()
    ctx = torch.argsort(counts, descending=True)[:min(num_context, V)]
    ctx_id = torch.full((V,), -1, dtype=torch.long, device=x.device)
    ctx_id[ctx] = torch.arange(ctx.numel(), device=x.device)
    C = ctx.numel()
    left, right = x[:-1], x[1:]
    co = torch.zeros(V, 2 * C, device=x.device)
    m = ctx_id[left] >= 0
    co.index_put_((right[m], ctx_id[left[m]]), torch.ones(int(m.sum()), device=x.device), accumulate=True)
    m = ctx_id[right] >= 0
    co.index_put_((left[m], C + ctx_id[right[m]]), torch.ones(int(m.sum()), device=x.device), accumulate=True)
    total = co.sum()
    pw = co.sum(1, keepdim=True) / total
    pc = co.sum(0, keepdim=True) / total
    ppmi = torch.log((co / total) / (pw * pc + 1e-12) + 1e-12).clamp_min(0.0)
    U, S, _ = torch.svd_lowrank(ppmi, q=min(dim, min(ppmi.shape) - 1), niter=4)
    emb = torch.nn.functional.normalize(U * S, dim=1)
    cent = emb[torch.randperm(V, generator=g)[:M].to(emb.device)].clone()
    for _ in range(iters):
        assign = (emb @ cent.T).argmax(1)
        for c in range(M):
            sel = assign == c
            if sel.any():
                cent[c] = torch.nn.functional.normalize(emb[sel].mean(0), dim=0)
            else:   # re-seed empty cluster with the worst-fitting token
                worst = (emb * cent[assign]).sum(1).argmin()
                cent[c] = emb[worst]
                assign[worst] = c
    assign = (emb @ cent.T).argmax(1)
    for c in range(M):
        if not (assign == c).any():
            worst = (emb * cent[assign]).sum(1).argmin()
            assign[worst] = c
    return assign.cpu()


def BlockedHMM(seq_length: int, num_latents: int, num_emits: int, num_blocks: int,
               token_block: Optional[torch.Tensor] = None, homogeneous: bool = True,
               block_size: Optional[int] = None,
               alpha: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None,
               gamma: Optional[torch.Tensor] = None):
    """
    Hidden Markov Model with **blocked emissions** (Chiu & Rush, 2020): the
    ``num_latents`` states are split into ``num_blocks`` groups of
    ``k = num_latents // num_blocks`` states and every token is emitted by
    exactly one group (``token_block[v]``). The chain compiles to the blocked
    fast path (:class:`BlockedProdLayer` / :class:`BlockedIOSumLayer` /
    :class:`CoBlockedProdLayer` / :class:`BlockedInputSumLayer`), which does
    ``O(T k^2)`` work per sequence instead of ``O(T num_latents^2)``.

    :param token_block: long ``[num_emits]`` block id per token; defaults to a
        size-balanced round-robin partition (see
        :func:`frequency_balanced_partition` for a frequency-aware one, or
        pass Brown-cluster ids).
    :param block_size: PC block size; must divide ``k``. Defaults to
        ``min(max_pow2(k), 1024)``.
    :param alpha: optional ``[num_latents, num_latents]`` transition matrix.
    :param beta: optional emission parameters, either token-major ``[num_emits, k]``
        (``beta[v, j] = P(v | token_block[v]*k + j)``) or a dense
        ``[num_latents, num_emits]`` matrix (entries outside the block pattern
        are dropped, rows re-normalised).
    :param gamma: optional ``[num_latents]`` initial-state distribution.
    """
    assert num_latents % num_blocks == 0, "num_blocks must divide num_latents"
    k = num_latents // num_blocks
    if token_block is None:
        token_block = torch.arange(num_emits, dtype=torch.long) % num_blocks
    token_block = torch.as_tensor(token_block, dtype=torch.long).reshape(-1)
    assert token_block.numel() == num_emits

    if block_size is None:
        block_size = min(max_cdf_power_of_2(k), 1024)
    assert k % block_size == 0, f"block_size ({block_size}) must divide k = {k}"
    num_node_blocks = num_latents // block_size

    with set_block_size(block_size=block_size):
        ns_input = inputs(
            seq_length - 1, num_node_blocks=num_node_blocks,
            dist=BlockedCategorical(num_cats=num_emits),
            token_block=token_block, num_blocks=num_blocks,
        )
        if beta is not None:
            if beta.dim() == 2 and beta.shape == (num_latents, num_emits):
                rows = token_block[:, None] * k + torch.arange(k)[None, :]
                cols = torch.arange(num_emits)[:, None].expand(num_emits, k)
                beta = beta[rows, cols]                                # [V, k]
            assert beta.numel() == num_emits * k
            ns_input.set_params(beta.reshape(-1))

        ns_sum = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=homogeneous)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=num_node_blocks)
                if alpha is not None:
                    assert alpha.shape == (num_latents, num_latents)
                    ns.set_params(alpha)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=homogeneous)
            curr_zs = blocked_multiply(curr_xs, ns)

        ns = blocked_summate(curr_zs, num_node_blocks=1, block_size=1)
        if gamma is not None:
            assert gamma.dim() == 1 and gamma.size(0) == num_latents
            ns.set_params(gamma.unsqueeze(0))

    return ns
