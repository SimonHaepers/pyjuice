"""Masked queries on the device sparse path (``pyjuice.model.sparse_cudagraph``).

A masked query gets one profile per set of masked variables: only those
variables' columns are sized to all latent rows (and only their Monarch
blocks take the dense path); every other column keeps the observed-token
bound. Results are checked against brute force: unmasked forwards over every
completion of the masked variables.
"""
from __future__ import annotations

import contextlib
import itertools

import pytest
import torch

import pyjuice as juice
import pyjuice.layer.sparse_io_sum_layer as sio
from pyjuice.layer import (CoSparseProdLayer, SparseInputBlockDiagonalSumLayer,
                           SparseOutputBlockDiagonalSumLayer, SparseProdLayer)
from pyjuice.layer.sparse_io_block_diagonal_sum_layer import _DENSE_BATCHED
from pyjuice.model import sparse_cudagraph as sc
from pyjuice.queries import conditional
from pyjuice.structures import SparseHMM, SparseMonarchHMM, random_emission_csc


DEV = torch.device("cuda:0")
T, H, V = 6, 64, 17


def _build(kind, compact=False):
    cptr, rid, vals, _ = random_emission_csc(H, V, 0.25, seed=11)
    torch.manual_seed(11)
    builder = SparseMonarchHMM if kind == "monarch" else SparseHMM
    root = builder(T, H, V, cptr, rid, vals, block_size=16)
    root.init_parameters()
    return juice.TensorCircuit(root, verbose=False, compact_sparse_buffers=compact).to(DEV)


@contextlib.contextmanager
def _no_ws():
    """The brute-force batches (V or V^2 completions) are large enough for
    the tf32 weight-stationary path (~1e-4 relative); keep the reference on
    the fp32 gather path."""
    mode, sio._WS_MODE = sio._WS_MODE, "never"
    try:
        yield
    finally:
        sio._WS_MODE = mode


def _completion_lls(pc, x, vars_):
    """Unmasked LL of every completion of sample ``x`` ([T]) over ``vars_``,
    shaped ``[V] * len(vars_)``."""
    combos = torch.tensor(list(itertools.product(range(V), repeat=len(vars_))), device=DEV)
    xs = x.repeat(combos.size(0), 1)
    xs[:, list(vars_)] = combos
    with _no_ws():
        ll = pc(xs, apply_cudagraph=False).reshape(-1)
    return ll.reshape([V] * len(vars_))


def _reference(pc, x, mask, target):
    """Brute-force marginal LL per sample and ``p(x_target | observed)`` for
    the samples that mask ``target``."""
    lls, conds = [], {}
    for b in range(x.size(0)):
        vars_ = torch.nonzero(mask[b]).flatten().tolist()
        ll = _completion_lls(pc, x[b], vars_)
        lls.append(torch.logsumexp(ll.reshape(-1), 0))
        if target in vars_:
            i = vars_.index(target)
            per_v = torch.logsumexp(ll.movedim(i, 0).reshape(V, -1), 1)
            conds[b] = torch.softmax(per_v, 0)
    return torch.stack(lls), conds


def _check(pc, x, mask, target, **kw):
    per_sample = mask if mask.dim() == 2 else mask[None].expand(x.size(0), -1)
    ll_ref, cond_ref = _reference(pc, x, per_sample, target)
    ll = pc(x, missing_mask=mask, **kw).reshape(-1)
    torch.testing.assert_close(ll, ll_ref, rtol=0, atol=2e-4)
    probs = conditional(pc, x, missing_mask=mask, target_vars=[target],
                        allow_modify_flows=False, **kw)
    for b, ref in cond_ref.items():
        torch.testing.assert_close(probs[b, 0], ref, rtol=0, atol=2e-5)


def _check_profile(pc, x, mask, masked):
    """Workspaces and dense switches of a masked forward (a backward
    consumes the block-offset caches, so run a fresh forward first)."""
    pc(x, missing_mask=mask)
    B = x.size(0)
    state = pc._sparse_graph_fwd
    assert state.masked_vars == set(masked)
    narrow = 0
    for lg in pc.inner_layer_groups:
        for layer in lg:
            if isinstance(layer, (SparseProdLayer, CoSparseProdLayer)):
                for i, ns in enumerate(layer.nodes):
                    dist = ns.sparse_input_ns.dist
                    K = H if ns.var_id in masked else dist._max_nnz_per_col
                    assert state._buffers[(id(layer), "fwd", i)].numel() == B * K
                    narrow += K < H
            elif isinstance(layer, SparseInputBlockDiagonalSumLayer) and B == 1:
                for i, var in enumerate(layer._input_sparsity_var_ids):
                    assert (layer._cached_in_offsets[i] is _DENSE_BATCHED) == (var in masked)
            elif isinstance(layer, SparseOutputBlockDiagonalSumLayer) and B == 1:
                for i, var in enumerate(layer._output_sparsity_var_ids):
                    assert (layer._cached_out_offsets[i] is _DENSE_BATCHED) == (var in masked)
    assert narrow > 0


@pytest.mark.parametrize("kind", ["io", "monarch"])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("masked", [(2,), (1, 4)])
def test_masked_query_matches_brute_force(kind, compact, B, masked):
    pc = _build(kind, compact)
    torch.manual_seed(B)
    x = torch.randint(0, V, (B, T), device=DEV)
    mask = torch.zeros(T, dtype=torch.bool, device=DEV)
    mask[list(masked)] = True
    _check(pc, x, mask[None].expand(B, T), masked[0])
    _check_profile(pc, x, mask, masked)
    # 1-D mask: the same profile
    n = len(pc._sparse_graphs)
    _check(pc, x, mask, masked[-1])
    assert len(pc._sparse_graphs) == n


@pytest.mark.parametrize("kind", ["io", "monarch"])
def test_per_sample_masks_use_the_union(kind):
    # Var 1 is masked in samples 0 and 2 and observed in sample 1 — its
    # columns hold all rows for some samples and the token's rows for others.
    pc = _build(kind)
    torch.manual_seed(3)
    x = torch.randint(0, V, (3, T), device=DEV)
    m = torch.zeros(3, T, dtype=torch.bool, device=DEV)
    m[0, 1] = m[1, 4] = m[2, 1] = m[2, 4] = True
    _check(pc, x, m, 1, record_cudagraph=True)
    _check_profile(pc, x, m, (1, 4))
    # Same union, rows swapped: replays the same graphs.
    m2 = m[[1, 0, 2]]
    n_graphs = len(pc._recorded_cuda_graphs)
    _check(pc, x, m2, 4, apply_cudagraph=True)
    assert len(pc._recorded_cuda_graphs) == n_graphs
    assert sum(1 for s in pc._sparse_graphs.values() if s.masked_vars) == 1


@pytest.mark.parametrize("kind", ["io", "monarch"])
def test_evicted_profiles_never_replay_stale_graphs(kind):
    # More masked sets than the profile cache holds, captured and replayed
    # (conditional captures forward and backward); evicted states must take
    # their backward graphs with them.
    pc = _build(kind)
    torch.manual_seed(4)
    x = torch.randint(0, V, (1, T), device=DEV)
    sets = [(t,) for t in range(T)] + [(t, t + 1) for t in range(T - 1)]
    assert len(sets) > sc.MAX_MASKED_PROFILES
    for _ in range(2):
        for vars_ in sets:
            mask = torch.zeros(T, dtype=torch.bool, device=DEV)
            mask[list(vars_)] = True
            _check(pc, x, mask, vars_[0], record_cudagraph=True)
            masked_states = [s for s in pc._sparse_graphs.values() if s.masked_vars]
            assert len(masked_states) <= sc.MAX_MASKED_PROFILES
            live = {s.uid for s in pc._sparse_graphs.values()}
            assert all(sig[-1] is None or sig[-1] in live
                       for sig in pc._recorded_cuda_graphs if sig[0] == 1)
