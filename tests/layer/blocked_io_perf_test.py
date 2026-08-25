"""Perf harness: blocked-emission chain vs the *same model* on the generic
sparse-IO chain (CSC with k nonzeros per column) vs the dense fast path.

Per configuration prints a table of ``fwd`` / ``bwd_ele`` / ``bwd_pflow``
milliseconds per call (B samples) for each build. Run with::

    pytest tests/layer/blocked_io_perf_test.py -s --runslow -k perf

The dense build is skipped above ``_DENSE_MAX_H`` (H×H transition params +
flows are 8·H² bytes).
"""
from __future__ import annotations

import sys
import os

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size, \
    sparse_multiply, sparse_summate, blocked_multiply, blocked_summate
from pyjuice.layer import BlockedIOSumLayer, SparseIOSumLayer, DenseSumLayer

sys.path.insert(0, os.path.dirname(__file__))
from _sparse_io_perf_helpers import time_phase  # noqa: E402


_DENSE_MAX_H = 16384


def _chain(T, H, bs, ns_input, mul, summ, force_plain=False):
    kw = {"_force_plain": True} if force_plain else {}
    with set_block_size(block_size=bs):
        ns_sum = None
        curr_zs = ns_input if not force_plain else multiply(ns_input, **kw)
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=H // bs, **kw)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = mul(curr_xs, ns, **kw)
        return summ(curr_zs, num_node_blocks=1, block_size=1, **kw)


def build_blocked(T, H, V, M, bs, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = H // M
    token_block = torch.cat([torch.arange(M), torch.randint(0, M, (V - M,), generator=g)])
    token_block = token_block[torch.randperm(V, generator=g)]
    with set_block_size(block_size=bs):
        ns_input = inputs(T - 1, num_node_blocks=H // bs,
                          dist=dists.BlockedCategorical(num_cats=V),
                          token_block=token_block, num_blocks=M)
        ns_input.set_params(torch.rand(V * k, generator=g) + 0.05)
    return _chain(T, H, bs, ns_input, blocked_multiply, blocked_summate), ns_input


def build_sparse_equiv(T, H, V, bs, ns_blocked):
    indptr, indices, values = ns_blocked.dist.to_csc(ns_blocked.get_params())
    with set_block_size(block_size=bs):
        ns_input = inputs(T - 1, num_node_blocks=H // bs,
                          dist=dists.SparseCategorical(num_cats=V),
                          csc_indptr=indptr, csc_indices=indices)
        ns_input.set_params(values, normalize=False)
    return _chain(T, H, bs, ns_input, sparse_multiply, sparse_summate)


def build_dense(T, H, V, bs, ns_blocked):
    dense = ns_blocked.dist.to_dense(ns_blocked.get_params(), fill_value=1e-10)
    with set_block_size(block_size=bs):
        ns_input = inputs(T - 1, num_node_blocks=H // bs, dist=dists.Categorical(num_cats=V))
        ns_input.set_params(dense, normalize=False)
    return _chain(T, H, bs, ns_input, multiply, summate, force_plain=True)


CONFIGS = [
    # (H, M, bs, V, T)
    (4096, 16, 256, 10000, 32),     # k = 256, paper-like ratio at small H
    (16384, 64, 256, 10000, 32),    # k = 256
    (32768, 128, 256, 10000, 32),   # paper: |Z| = 2^15, M = 128, k = 256
]


@pytest.mark.slow
@pytest.mark.parametrize("H,M,bs,V,T", CONFIGS)
@pytest.mark.parametrize("B", [1, 64])
def test_blocked_vs_sparse_vs_dense_perf(H, M, bs, V, T, B):
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    root_b, ns_b = build_blocked(T, H, V, M, bs)
    builds = {"blocked": (root_b, {})}
    builds["sparse-io"] = (build_sparse_equiv(T, H, V, bs, ns_b), {})
    if H <= _DENSE_MAX_H:
        builds["dense"] = (build_dense(T, H, V, bs, ns_b), {"use_dense_sum_layer": True})

    data = torch.randint(0, V, (B, T), device=dev)
    rows = []
    ref_ll = None
    ref_params = None
    for name, (root, kw) in builds.items():
        pc = juice.TensorCircuit(root, verbose=False, **kw).to(dev)
        # Transition / root params are randomly initialised per compile; all
        # three builds share the DenseSumLayer flat layout, so copy them over.
        if ref_params is None:
            ref_params = pc.params.detach().clone()
        else:
            assert pc.params.numel() == ref_params.numel()
            pc.params.data.copy_(ref_params)
        layers = [l for lg in pc.inner_layer_groups for l in lg]
        if name == "blocked":
            assert any(isinstance(l, BlockedIOSumLayer) for l in layers)
        elif name == "sparse-io":
            assert any(isinstance(l, SparseIOSumLayer) for l in layers)
        else:
            assert any(type(l) is DenseSumLayer for l in layers)
        ll = pc(data)
        if ref_ll is None:
            ref_ll = ll
        else:
            assert torch.allclose(ll, ref_ll, atol=1e-2, rtol=5e-4), (name, ll[:3], ref_ll[:3])
        t = {ph: time_phase(pc, data, ph, n_warmup=3, n_iter=10)
             for ph in ("fwd", "bwd_ele", "bwd_pflow")}
        rows.append((name, t))
        del pc
        torch.cuda.empty_cache()

    print(f"\n[H={H} M={M} k={H // M} bs={bs} V={V} T={T} B={B}]  ms per call")
    print(f"{'build':<10} {'fwd':>9} {'bwd_ele':>9} {'bwd_pflow':>10}")
    for name, t in rows:
        print(f"{name:<10} {t['fwd']:>9.3f} {t['bwd_ele']:>9.3f} {t['bwd_pflow']:>10.3f}")
