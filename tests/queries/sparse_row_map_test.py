"""
Row-mapped sparse emission products: a :class:`SparseProdNodes` whose edge to
its dense child maps output row ``h`` to child node ``row_map[h]``
(node-level edges, :meth:`SparseProdNodes.dense_row_map`).

The motivating case is the standard Monarch transition ``P · BD2 · P · BD1``,
whose second permutation is folded into the emission product
(``MonarchHMM(output_permutation=True)`` / ``SparseMonarchHMM``). Every test
compares a sparse build against the same model with dense ``Categorical``
emissions (exact zeros off the CSC support), which runs through the plain
product / block-diagonal kernels:

* Monarch chain — BD2 compiles to :class:`SparseOutputBlockDiagonalSumLayer`
  with a row map, its consumers to :class:`CoSparseProdLayer`: LLs, BD1 / BD2
  param flows, batched-vs-loop at non-pow2 B, marginalised positions (B=1)
  and the conditional query.
* Dense-transition chain with a permuted emission product — a sparse-IO sum
  cannot serve a row-mapped consumer, so the chain falls back to
  :class:`SparseInputSumLayer` + :class:`SparseProdLayer` reading
  ``node_mars`` through the row map: LLs must still match.
* The per-timestep BD layers share one compile-time table / row-mapped CSC.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size
from pyjuice.layer import (
    BlockDiagonalSumLayer, CoSparseProdLayer, SparseInputBlockDiagonalSumLayer,
    SparseInputSumLayer, SparseIOSumLayer, SparseOutputBlockDiagonalSumLayer,
    SparseProdLayer,
)
from pyjuice.structures import MonarchHMM, SparseMonarchHMM, random_emission_csc


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


def _dense_beta(indptr, indices, values, H, V):
    return torch.sparse_csc_tensor(indptr, indices, values, size=(H, V)).to_dense()


def _rand_rows(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*shape, generator=g) + 0.05
    return x / x.sum(dim=-1, keepdim=True)


def _monarch_pair(T, H, V, bs, density, seed, output_permutation=True):
    """(dense-emission root, sparse-emission root) of one Monarch model."""
    indptr, indices, values, _ = random_emission_csc(H, V, density, seed=seed)
    NB = H // bs
    bd1 = _rand_rows(NB, bs, bs, seed=seed + 1)
    bd2 = _rand_rows(NB, bs, bs, seed=seed + 2)
    gamma = _rand_rows(H, seed=seed + 3)
    dense = MonarchHMM(T, H, V, block_size=bs, bd1=bd1, bd2=bd2, gamma=gamma,
                       beta=_dense_beta(indptr, indices, values, H, V),
                       output_permutation=output_permutation)
    sparse = SparseMonarchHMM(T, H, V, indptr, indices, values, block_size=bs,
                              bd1=bd1, bd2=bd2, gamma=gamma,
                              output_permutation=output_permutation)
    return dense, sparse


def _bd_sources(root):
    """(BD1, BD2) source sum nodes of a Monarch chain."""
    emit_prod = root.chs[0]
    bd2 = next(c for c in emit_prod.chs if c.is_sum())
    bd1 = bd2.chs[0].chs[0]
    return bd1.get_source_ns(), bd2.get_source_ns()


def _layers(pc, cls):
    return [layer for lg in pc.inner_layer_groups for layer in lg if type(layer) is cls]


# ---------------------------------------------------------------------- #
# Monarch chain with the second permutation (P · BD2 · P · BD1)
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("T,H,V,bs,B", [
    (5, 16, 8, 4, 1),
    (6, 64, 12, 8, 1),
    (5, 16, 8, 4, 8),
    (6, 64, 12, 8, 4),
    (4, 256, 20, 16, 8),
])
def test_monarch_forward_and_pflows(T, H, V, bs, B):
    device = _cuda()
    dense_root, sparse_root = _monarch_pair(T, H, V, bs, density=0.3, seed=11)
    pc_d = juice.TensorCircuit(dense_root, verbose=False).to(device)
    pc_s = juice.TensorCircuit(sparse_root, verbose=False).to(device)

    out_bds = _layers(pc_s, SparseOutputBlockDiagonalSumLayer)
    assert len(out_bds) == T - 1
    assert all(layer._out_row_maps[0] is not None for layer in out_bds)
    assert len(_layers(pc_s, SparseInputBlockDiagonalSumLayer)) == T - 1
    assert len(_layers(pc_s, CoSparseProdLayer)) == T - 1
    assert not _layers(pc_s, BlockDiagonalSumLayer)

    torch.manual_seed(0)
    data = torch.randint(0, V, (B, T), device=device)
    lls_d = pc_d(data)
    lls_s = pc_s(data)
    assert torch.isfinite(lls_d).all()
    torch.testing.assert_close(lls_s, lls_d, rtol=1e-4, atol=1e-3)

    pc_d.backward(data, compute_param_flows=True, flows_memory=0.0,
                  allow_modify_flows=False)
    pc_s.backward(data, compute_param_flows=True, flows_memory=0.0,
                  allow_modify_flows=False)
    for name, ns_d, ns_s in zip(("BD1", "BD2"), _bd_sources(dense_root), _bd_sources(sparse_root)):
        ns_d.update_param_flows(pc_d.param_flows, origin_ns_only=True)
        ns_s.update_param_flows(pc_s.param_flows, origin_ns_only=True)
        pf_d, pf_s = ns_d._param_flows.detach(), ns_s._param_flows.detach()
        scale = max(pf_d.abs().max().item(), 1e-6)
        err = (pf_s - pf_d).abs().max().item() / scale
        assert err < 1e-3, f"{name} param flows differ: rel err {err:.2e}"


def test_monarch_permutation_matters():
    """The permuted and unpermuted models must differ (guards against the
    row map being silently ignored)."""
    device = _cuda()
    T, H, V, bs = 6, 64, 12, 8
    _, perm_root = _monarch_pair(T, H, V, bs, density=0.3, seed=5)
    _, plain_root = _monarch_pair(T, H, V, bs, density=0.3, seed=5, output_permutation=False)
    pc_perm = juice.TensorCircuit(perm_root, verbose=False).to(device)
    pc_plain = juice.TensorCircuit(plain_root, verbose=False).to(device)
    data = torch.randint(0, V, (4, T), device=device)
    assert (pc_perm(data) - pc_plain(data)).abs().max().item() > 1e-2


@pytest.mark.parametrize("T,H,V,bs,B", [(5, 16, 8, 4, 7), (6, 64, 12, 8, 9)])
def test_monarch_batched_matches_loop(T, H, V, bs, B):
    """One batched pass equals B single-sample passes (LLs and BD2 param
    flows) — covers non-pow2 B without a plain reference."""
    device = _cuda()
    _, root = _monarch_pair(T, H, V, bs, density=0.3, seed=23)
    pc = juice.TensorCircuit(root, verbose=False).to(device)
    _, bd2 = _bd_sources(root)

    torch.manual_seed(1)
    data = torch.randint(0, V, (B, T), device=device)
    lls_loop, pf_loop = [], None
    for b in range(B):
        lls_loop.append(pc(data[b:b + 1]).detach().clone())
        pc.backward(data[b:b + 1], compute_param_flows=True, flows_memory=0.0,
                    allow_modify_flows=False)
        bd2.update_param_flows(pc.param_flows, origin_ns_only=True)
        pf = bd2._param_flows.detach().clone()
        pf_loop = pf if pf_loop is None else pf_loop + pf

    lls = pc(data)
    torch.testing.assert_close(lls.flatten(), torch.cat(lls_loop).flatten(), rtol=1e-4, atol=1e-4)
    pc.backward(data, compute_param_flows=True, flows_memory=0.0, allow_modify_flows=False)
    bd2.update_param_flows(pc.param_flows, origin_ns_only=True)
    torch.testing.assert_close(bd2._param_flows.detach(), pf_loop, rtol=1e-4, atol=1e-5)


def test_monarch_missing_and_conditional():
    """B=1 marginalised positions use the row-mapped all-rows pattern;
    the conditional query also runs the backward through it."""
    device = _cuda()
    T, H, V, bs = 6, 64, 12, 8
    dense_root, sparse_root = _monarch_pair(T, H, V, bs, density=0.3, seed=31)
    pc_d = juice.TensorCircuit(dense_root, verbose=False).to(device)
    pc_s = juice.TensorCircuit(sparse_root, verbose=False).to(device)

    torch.manual_seed(2)
    data = torch.randint(0, V, (1, T), device=device)
    missing = torch.zeros(T, dtype=torch.bool, device=device)
    missing[2] = True
    missing[4] = True
    torch.testing.assert_close(pc_s(data, missing_mask=missing),
                               pc_d(data, missing_mask=missing), rtol=1e-4, atol=1e-3)

    target_mask = torch.zeros(T, dtype=torch.bool, device=device)
    target_mask[3] = True
    out_d = juice.queries.conditional(pc_d, data=data, missing_mask=target_mask, target_vars=[3])
    out_s = juice.queries.conditional(pc_s, data=data, missing_mask=target_mask, target_vars=[3])
    torch.testing.assert_close(out_s, out_d, rtol=1e-3, atol=1e-4)


def test_monarch_tables_shared_across_timesteps():
    device = _cuda()
    T, H, V, bs = 6, 64, 12, 8
    _, root = _monarch_pair(T, H, V, bs, density=0.3, seed=3)
    pc = juice.TensorCircuit(root, verbose=False).to(device)
    pc(torch.randint(0, V, (2, T), device=device))

    out_bds = _layers(pc, SparseOutputBlockDiagonalSumLayer)
    first = out_bds[0]._out_row_maps[0]
    assert all(layer._out_row_maps[0] is first for layer in out_bds)
    in_bds = _layers(pc, SparseInputBlockDiagonalSumLayer)
    table = in_bds[0]._in_block_offsets_tables[0]
    assert all(layer._in_block_offsets_tables[0] is table for layer in in_bds)
    assert table.on(device) is in_bds[-1]._in_block_offsets_tables[0].on(device)


# ---------------------------------------------------------------------- #
# Fallback: row-mapped consumer of a dense-transition (non-BD) sum
# ---------------------------------------------------------------------- #


def _permuted_hmm(T, H, V, bs, sparse, csc, alpha, gamma, perm):
    indptr, indices, values = csc
    edges = torch.stack([torch.arange(H), perm], dim=1)
    with set_block_size(block_size=bs):
        if sparse:
            ns_input = inputs(T - 1, num_node_blocks=H // bs,
                              dist=dists.SparseCategorical(num_cats=V),
                              csc_indptr=indptr, csc_indices=indices)
            ns_input.set_params(values, normalize=False)
        else:
            ns_input = inputs(T - 1, num_node_blocks=H // bs,
                              dist=dists.Categorical(num_cats=V))
            ns_input.set_params(_dense_beta(indptr, indices, values, H, V))
        ns_sum, curr_zs = None, ns_input
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                ns_sum = summate(curr_zs, num_node_blocks=H // bs)
                ns_sum.set_params(alpha)
                ns = ns_sum
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = multiply(curr_xs, ns, edge_ids=edges, sparse_edges=True)
        root = summate(curr_zs, num_node_blocks=1, block_size=1)
        root.set_params(gamma.unsqueeze(0))
    return root


@pytest.mark.parametrize("B", [1, 4])
def test_dense_transition_row_map_fallback(B):
    device = _cuda()
    T, H, V, bs = 5, 32, 10, 8
    indptr, indices, values, _ = random_emission_csc(H, V, 0.3, seed=7)
    alpha = _rand_rows(H, H, seed=8)
    gamma = _rand_rows(H, seed=9)
    perm = torch.randperm(H, generator=torch.Generator().manual_seed(10))
    csc = (indptr, indices, values)
    pc_d = juice.TensorCircuit(_permuted_hmm(T, H, V, bs, False, csc, alpha, gamma, perm),
                               verbose=False).to(device)
    pc_s = juice.TensorCircuit(_permuted_hmm(T, H, V, bs, True, csc, alpha, gamma, perm),
                               verbose=False).to(device)

    assert not _layers(pc_s, SparseIOSumLayer)
    assert not _layers(pc_s, CoSparseProdLayer)
    assert _layers(pc_s, SparseInputSumLayer)
    assert _layers(pc_s, SparseProdLayer)

    torch.manual_seed(4)
    data = torch.randint(0, V, (B, T), device=device)
    torch.testing.assert_close(pc_s(data), pc_d(data), rtol=1e-4, atol=1e-3)
