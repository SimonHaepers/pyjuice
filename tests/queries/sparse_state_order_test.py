"""Latent-state reordering (``pyjuice.transformations.reorder``) is
model-equivalent: a sparse-IO HMM chain and a small tree circuit with sparse
emissions give the same log-likelihoods after ``permute_latents`` with a
random permutation and after ``reorder_states`` (the emission-based order),
and the emission-based order is a permutation."""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
from pyjuice.nodes import inputs, sparse_multiply, sparse_summate, summate, set_block_size
from pyjuice.nodes.distributions import SparseCategorical
from pyjuice.structures import SparseHMM
from pyjuice.transformations import emission_state_order, permute_latents, reorder_states, source_nodes


def _random_csc(H: int, V: int, density: float, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(density * H * V) + H
    rows = torch.cat([torch.randint(0, H, (n,), generator=g), torch.arange(H)])
    cols = torch.cat([torch.randint(0, V, (n,), generator=g), torch.randint(0, V, (H,), generator=g)])
    linear = torch.unique(rows * V + cols)
    rows_d, cols_d = linear // V, linear % V
    order = torch.argsort(cols_d * H + rows_d)
    indices = rows_d[order].contiguous()
    indptr = torch.zeros(V + 1, dtype=torch.long)
    indptr[1:] = torch.cumsum(torch.bincount(cols_d[order], minlength=V), 0)
    values = torch.rand(indices.numel(), generator=g) + 0.05
    return indptr, indices, values


def _tokens(indptr: torch.Tensor, B: int, T: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    nonempty = torch.nonzero(torch.diff(indptr) > 0).reshape(-1)
    return nonempty[torch.randint(0, nonempty.numel(), (B, T), generator=g)]


def _chain(T: int, H: int, V: int, bs: int, seed: int):
    indptr, indices, values = _random_csc(H, V, 0.08, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    alpha = torch.rand(H, H, generator=g) + 0.01
    alpha = alpha / alpha.sum(1, keepdim=True)
    gamma = torch.rand(H, generator=g) + 0.01
    gamma = gamma / gamma.sum()
    root = SparseHMM(T, H, V, indptr, indices, values, block_size=bs, alpha=alpha, gamma=gamma)
    return root, indptr


def _tree(H: int, V: int, bs: int, seed: int):
    """Three variables: an untied sparse input (own dist copy) under a sum
    block, multiplied element-wise with a second input, a second sum block,
    and a root over the product with a tied copy of the second input."""
    indptr, indices, values = _random_csc(H, V, 0.08, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 2)
    with set_block_size(block_size=bs):
        x0 = inputs(0, num_node_blocks=H // bs, dist=SparseCategorical(num_cats=V),
                    csc_indptr=indptr, csc_indices=indices)
        x0.set_params(values, normalize=False)
        x1 = x0.duplicate(1, tie_params=False)
        x1.set_params(torch.rand(indices.numel(), generator=g) + 0.05, normalize=False)
        s1 = summate(x1, num_node_blocks=H // bs)
        W1 = torch.rand(H, H, generator=g) + 0.01
        s1.set_params(W1 / W1.sum(1, keepdim=True))
        p = sparse_multiply(x0, s1)
        s = summate(p, num_node_blocks=H // bs)
        W = torch.rand(H, H, generator=g) + 0.01
        s.set_params(W / W.sum(1, keepdim=True))
        x2 = x0.duplicate(2, tie_params=True)
        root = sparse_summate(sparse_multiply(x2, s), num_node_blocks=1, block_size=1)
        gamma = torch.rand(H, generator=g) + 0.01
        root.set_params((gamma / gamma.sum()).unsqueeze(0))
    return root, indptr


def _ll(root, data: torch.Tensor, device) -> torch.Tensor:
    pc = juice.TensorCircuit(root, verbose=False).to(device)
    return pc(data.to(device)).detach().cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("H,V,bs", [(64, 40, 32), (256, 120, 64)])
def test_chain_permutation_invariance(H, V, bs):
    device = torch.device("cuda:0")
    T, B = 6, 5
    root, indptr = _chain(T, H, V, bs, seed=3)
    data = _tokens(indptr, B, T, seed=4)
    ref = _ll(root, data, device)

    root2, _ = _chain(T, H, V, bs, seed=3)
    perm = torch.randperm(H, generator=torch.Generator().manual_seed(5))
    permute_latents(root2, perm)
    assert torch.allclose(_ll(root2, data, device), ref, rtol=1e-4, atol=1e-4)

    root3, _ = _chain(T, H, V, bs, seed=3)
    order = reorder_states(root3)
    assert torch.equal(torch.sort(order).values, torch.arange(H))
    emit = next(ns for ns in source_nodes(root3) if ns.is_input())
    assert torch.equal(emit.dist._state_perm, order)
    assert torch.allclose(_ll(root3, data, device), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tree_permutation_invariance():
    device = torch.device("cuda:0")
    H, V, bs = 64, 50, 32
    root, indptr = _tree(H, V, bs, seed=7)
    data = _tokens(indptr, 4, 3, seed=8)
    ref = _ll(root, data, device)
    root2, _ = _tree(H, V, bs, seed=7)
    permute_latents(root2, torch.randperm(H, generator=torch.Generator().manual_seed(9)))
    assert torch.allclose(_ll(root2, data, device), ref, rtol=1e-4, atol=1e-4)


def test_emission_state_order_groups_states_by_token():
    """States emitting the same most frequent token end up adjacent."""
    H, V = 32, 6
    # state h emits token h % V (weight by frequency) and a rare token
    rows = torch.cat([torch.arange(H), torch.arange(H)])
    cols = torch.cat([torch.arange(H) % V, torch.full((H,), V - 1)])
    linear = torch.unique(rows * V + cols)
    rows_d, cols_d = linear // V, linear % V
    order = torch.argsort(cols_d * H + rows_d)
    indices = rows_d[order].contiguous()
    indptr = torch.zeros(V + 1, dtype=torch.long)
    indptr[1:] = torch.cumsum(torch.bincount(cols_d[order], minlength=V), 0)
    with set_block_size(block_size=16):
        x = inputs(0, num_node_blocks=H // 16, dist=SparseCategorical(num_cats=V),
                   csc_indptr=indptr, csc_indices=indices)
    weights = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    perm = emission_state_order(x, token_weights=weights)
    assert torch.equal(torch.sort(perm).values, torch.arange(H))
    dominant = (perm % V)                      # old state's most frequent token
    # sorted by decreasing weight of the dominant token: 0,0,...,1,1,...,5
    assert torch.equal(dominant, torch.sort(dominant).values)
