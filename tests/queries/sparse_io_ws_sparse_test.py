"""The CUDA-core forward of :class:`SparseIOSumLayer` over the nonzero
products only (``"sparse"``, :mod:`pyjuice.layer.sparse_io_cuda`): for every
parent tile of 4 or 8 rows the samples with an active parent in it walk their
packed input columns against the tile's W slab in shared memory and write the
packed output. It must give the gather path's log-likelihoods (and, through
them, flows) at every batch size, eagerly and under a CUDA graph, on state
counts that are not powers of two, with the slab copied by TMA and by 4-byte
``cp.async``, with empty columns and masked variables, on a circuit that is
not a chain, for every kernel configuration; blocks with unsorted columns must
not take it.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.layer.sparse_io_cuda as sio_cuda
import pyjuice.layer.sparse_io_sum_layer as sio
from pyjuice.nodes import inputs, multiply, summate, set_block_size
from pyjuice.nodes import distributions as dists
from pyjuice.structures import SparseHMM, random_emission_csc

DEV = torch.device("cuda:0")
_TOL = {"ieee": 1e-4, "tf32": 5e-3}      # atol on the per-sequence LL


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _build(H, V, T, bs, density, seed=0, indptr=None, indices=None):
    if indptr is None:
        indptr, indices, _, _ = random_emission_csc(H, V, density, seed=seed)
    torch.manual_seed(seed)
    ns = SparseHMM(T, H, V, indptr, indices, block_size=bs)
    ns.init_parameters()
    return juice.TensorCircuit(ns, verbose=False).to(DEV)


def _build_forest(H=64, V=30, T=5, bs=16, density=0.3, seed=0):
    """Not a chain: two sparse HMM chains over disjoint variables, each with
    its own emission support and transition matrix, joined by a product and a
    root sum. Every SparseIOSumLayer holds two nodes (one per chain) with
    different output columns and different W."""
    from pyjuice.nodes import sparse_multiply, sparse_summate
    nb = H // bs
    tops = []
    with set_block_size(bs):
        for c in range(2):
            indptr, indices, values, _ = random_emission_csc(H, V, density, seed=seed + c)
            emit = inputs(c * T + T - 1, num_node_blocks=nb, dist=dists.SparseCategorical(num_cats=V),
                          csc_indptr=indptr, csc_indices=indices)
            emit.set_params(values, normalize=False)
            zs, ns_sum = emit, None
            for var in range(T - 2, -1, -1):
                xs = emit.duplicate(c * T + var, tie_params=True)
                if ns_sum is None:
                    ns_sum = ns = summate(zs, num_node_blocks=nb)
                else:
                    ns = ns_sum.duplicate(zs, tie_params=True)
                zs = sparse_multiply(xs, ns)
            tops.append(sparse_summate(zs, num_node_blocks=1, block_size=1))
        root = summate(multiply(*tops), num_node_blocks=1, block_size=1)
    torch.manual_seed(seed)
    root.init_parameters()
    pc = juice.TensorCircuit(root, verbose=False).to(DEV)
    io_layers = [l for lg in pc.inner_layer_groups for l in lg if type(l) is sio.SparseIOSumLayer]
    assert io_layers and all(len(l.nodes) == 2 for l in io_layers)
    return pc, V, 2 * T


def _set(monkeypatch, mode, prec="ieee", sparse=None, **knobs):
    """``sparse``: force the sparse forward on (True) / off (False) for the
    WS forwards (``_WS_SPARSE_MODE``); ``knobs``: ``_WS_SPARSE_<name>``
    (``SLAB_TM`` / ``SLAB_BYTES`` / ``SLAB_NWARP`` / ``SLAB_MINB``)."""
    monkeypatch.setattr(sio, "_WS_MODE", mode)
    monkeypatch.setattr(sio, "_WS_PRECISION", prec)
    if sparse is not None:
        monkeypatch.setattr(sio, "_WS_SPARSE_MODE", "always" if sparse else "never")
    for k, v in knobs.items():
        if k == "SLAB_NWARP":
            v = {t: v for t in (1, 2, 4, 8, 16)}
        monkeypatch.setattr(sio, f"_WS_SPARSE_{k}", v)


def _count(monkeypatch):
    calls = {"fwd": 0, "sparse": 0}
    fwd = sio.SparseIOSumLayer._ws_forward_block

    def counted_fwd(self, *a, **k):
        calls["fwd"] += 1
        calls["sparse"] += bool(k.get("sparse"))
        return fwd(self, *a, **k)

    monkeypatch.setattr(sio.SparseIOSumLayer, "_ws_forward_block", counted_fwd)
    return calls


def _emit_flows(pc):
    return torch.cat([layer.param_flows.detach().reshape(-1).clone() for layer in pc.input_layer_group])


def _pass(pc, data):
    """LL, transition param flows and emission param flows (the backward
    reads the forward's packed node values)."""
    ll = pc(data, apply_cudagraph=False).detach().clone()
    pc.backward(data, compute_param_flows=True, flows_memory=0.0)
    return ll, pc.param_flows.detach().clone(), _emit_flows(pc)


def _assert_flows_close(got, ref, rtol, what):
    scale = ref.abs().max().clamp_min(1e-12)
    err = ((got - ref).abs().max() / scale).item()
    assert err < rtol, f"{what}: max |diff| / max |ref| = {err:.2e} (tol {rtol:.0e})"


@pytest.mark.parametrize("tm", ["auto", 4, 8])
@pytest.mark.parametrize("prec", ["ieee", "tf32"])
@pytest.mark.parametrize("H,V,T,bs,density,B", [
    (64, 20, 6, 16, 0.3, 3),              # few samples, PC blocks of 16
    (96, 30, 5, 32, 0.25, 9),             # 3 PC blocks, H not a power of two
    (80, 30, 5, 16, 0.25, 9),             # 5 PC blocks of 16, H not a power of two
    (128, 40, 8, 32, 0.2, 16),
    (128, 40, 7, 16, 0.2, 70),            # lists longer than one CTA
    (4096, 50, 4, 1024, 0.05, 130),       # several PC blocks of 1024, sparse columns
    (2048, 300, 4, 1024, 0.02, 300),
])
def test_sparse_matches_gather(monkeypatch, prec, H, V, T, bs, density, B, tm):
    """The forward is exact fp32 in both precisions (it reads W itself, not
    the tf32 copy) with either tile height; the backward takes its own
    dispatch."""
    pc = _build(H, V, T, bs, density)
    data = torch.randint(0, V, (B, T), device=DEV)

    _set(monkeypatch, "never")
    ll_ref, pf_ref, emit_ref = _pass(pc, data)

    calls = _count(monkeypatch)
    _set(monkeypatch, "always", prec, sparse=True, SLAB_TM=tm)
    ll, pf, emit = _pass(pc, data)
    assert calls["fwd"] == T - 1 and calls["sparse"] == T - 1
    torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=_TOL[prec])
    rtol = 1e-4 if prec == "ieee" else 3e-3
    _assert_flows_close(pf, pf_ref, rtol, "transition param flows")
    _assert_flows_close(emit, emit_ref, rtol, "emission param flows")


@pytest.mark.parametrize("loader", [None, 1])
@pytest.mark.parametrize("knobs", [
    dict(),                                                   # the default: 4 or 8-row tiles by cost
    dict(SLAB_TM=4, SLAB_BYTES=512, SLAB_NWARP=2),            # 32 children per CTA: 8 ranges
    dict(SLAB_TM=2, SLAB_BYTES=1024),                         # 2-row tiles, 4 ranges
    dict(SLAB_TM=1, SLAB_NWARP=4, SLAB_MINB=1),
    dict(SLAB_TM=8, SLAB_BYTES=2048),                         # 8-row tiles, 4 ranges
    dict(SLAB_TM=16, SLAB_BYTES=6144, SLAB_NWARP=12, SLAB_MINB=2),   # 16 rows, 3 ranges
])
def test_sparse_knobs(monkeypatch, knobs, loader):
    """Every slab configuration gives the same result: tile heights 1 to 16
    (only the 4-row quarters with one of the sample's parents read), the
    children split into ranges whose partial sums add up in the output, TMA
    and 4-byte ``cp.async`` copies of the slab; density 0.3 at H=256 puts up
    to all rows of a tile in a sample."""
    pc = _build(256, 60, 5, 128, 0.3, seed=2)
    data = torch.randint(0, 60, (40, 5), device=DEV)
    _set(monkeypatch, "never")
    ref = pc(data, apply_cudagraph=False).detach().clone()
    monkeypatch.setattr(sio_cuda, "FORCE_LOADER", loader)
    _set(monkeypatch, "always", "ieee", sparse=True, **knobs)
    got = pc(data, apply_cudagraph=False).detach().clone()
    torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-4)


def test_sparse_empty_columns(monkeypatch):
    """A sample whose token has an empty emission column (impossible
    sequence) gets ``-inf``-like values like the gather path; the others are
    unaffected."""
    H, V, T, bs = 64, 20, 6, 16
    indptr, indices, _, _ = random_emission_csc(H, V, 0.3, seed=3)
    indices = indices[indptr[1]:]                    # token 0 loses its column
    indptr = torch.cat([indptr[:1], indptr[1:] - indptr[1]])
    pc = _build(H, V, T, bs, 0.3, seed=3, indptr=indptr, indices=indices)
    x = torch.randint(1, V, (5, T), device=DEV)
    x[1, 2] = 0
    x[3, 0] = 0
    _set(monkeypatch, "never")
    ref = pc(x, apply_cudagraph=False).detach().clone()
    _set(monkeypatch, "always", "ieee", sparse=True)
    got = pc(x, apply_cudagraph=False).detach().clone()
    impossible = torch.tensor([False, True, False, True, False], device=DEV)
    assert (got[impossible] < -1e29).all()
    torch.testing.assert_close(got[~impossible], ref[~impossible], rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("H,V,T,bs,density,B", [
    (64, 20, 6, 16, 0.3, 1),            # static B=1 container
    (96, 30, 5, 32, 0.25, 9),
    (128, 40, 7, 16, 0.2, 70),
])
def test_sparse_graph_replay_matches_eager(monkeypatch, H, V, T, bs, density, B):
    """The per-step lists and ranges are built on the device inside the
    captured graph: replays on new batches give the eager gather path, and
    the eager backward after a replay agrees."""
    pc = _build(H, V, T, bs, density, seed=1)
    batches = [torch.randint(0, V, (B, T), device=DEV) for _ in range(3)]
    _set(monkeypatch, "never")
    refs = [pc(x, apply_cudagraph=False).detach().clone() for x in batches]
    _set(monkeypatch, "always", "ieee", sparse=True)
    calls = _count(monkeypatch)
    pc(batches[0], record_cudagraph=True)
    assert pc._sparse_graph_fwd is not None and pc._sparse_graph_fwd.graph is not None
    assert calls["sparse"] > 0
    for x, ref in zip(batches, refs):
        got = pc(x).detach().clone()
        torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-4)
    _set(monkeypatch, "never")
    _, pf_ref, emit_ref = _pass(pc, batches[1])
    _set(monkeypatch, "always", "ieee", sparse=True)
    pc(batches[1])
    pc.backward(batches[1], compute_param_flows=True, flows_memory=0.0)
    _assert_flows_close(pc.param_flows.detach(), pf_ref, 1e-4, "transition param flows")
    _assert_flows_close(_emit_flows(pc), emit_ref, 1e-4, "emission param flows")


def test_sparse_masked_variables(monkeypatch):
    """A marginalised variable gives an all-rows output column (every parent
    of every tile active) and an all-rows input column at the next step."""
    H, V, T, bs = 64, 20, 6, 16
    pc = _build(H, V, T, bs, 0.3, seed=6)
    x = torch.randint(0, V, (3, T), device=DEV)
    mm = torch.zeros(3, T, dtype=torch.bool, device=DEV)
    mm[0, 2] = True
    mm[2, 4] = True
    _set(monkeypatch, "never")
    ref = pc(x, missing_mask=mm, apply_cudagraph=False).detach().clone()
    _set(monkeypatch, "always", "ieee", sparse=True)
    got = pc(x, missing_mask=mm, record_cudagraph=True).detach().clone()
    torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-5)
    mm[0, 2], mm[1, 2] = False, True
    got = pc(x, missing_mask=mm).detach().clone()
    _set(monkeypatch, "never")
    ref = pc(x, missing_mask=mm, apply_cudagraph=False).detach().clone()
    torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("B", [5, 40])
def test_sparse_on_forest(monkeypatch, B):
    """Not a chain: two SparseIOSumLayer nodes per layer with different
    output columns and transition matrices, eagerly and replayed."""
    pc, V, n_vars = _build_forest()
    batches = [torch.randint(0, V, (B, n_vars), device=DEV) for _ in range(2)]
    _set(monkeypatch, "never")
    refs = [_pass(pc, x) for x in batches]
    calls = _count(monkeypatch)
    _set(monkeypatch, "always", "ieee", sparse=True, SLAB_TM=8)
    for x, (ll_ref, pf_ref, emit_ref) in zip(batches, refs):
        ll, pf, emit = _pass(pc, x)
        torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=1e-4)
        _assert_flows_close(pf, pf_ref, 1e-4, "sum param flows")
        _assert_flows_close(emit, emit_ref, 1e-4, "emission param flows")
    assert calls["sparse"] > 0
    pc(batches[0], record_cudagraph=True)
    for x, (ll_ref, _, _) in zip(batches, refs):
        torch.testing.assert_close(pc(x).detach(), ll_ref, rtol=0.0, atol=1e-4)


def test_sparse_needs_sorted_columns(monkeypatch):
    """A column whose rows do not ascend makes the block ineligible: forced
    WS forwards take another pass, and the result is still right."""
    H, V, T, bs = 64, 20, 5, 16
    indptr, indices, _, _ = random_emission_csc(H, V, 0.3, seed=4)
    a, b = int(indptr[2]), int(indptr[3])
    indices = indices.clone()
    indices[a:b] = indices[a:b].flip(0)
    pc = _build(H, V, T, bs, 0.3, seed=4, indptr=indptr, indices=indices)
    x = torch.randint(0, V, (6, T), device=DEV)
    _set(monkeypatch, "never")
    ref = pc(x, apply_cudagraph=False).detach().clone()
    calls = _count(monkeypatch)
    _set(monkeypatch, "always", "ieee", sparse=True)
    got = pc(x, apply_cudagraph=False).detach().clone()
    assert calls["fwd"] == T - 1 and calls["sparse"] == 0
    torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-4)


def test_columns_sorted():
    """``SparseCategorical.columns_sorted`` (host only)."""
    dist = dists.SparseCategorical(num_cats=3)
    dist.set_meta_parameters(64, torch.tensor([0, 2, 5, 5]), torch.tensor([0, 1, 3, 40, 63]))
    assert dist.columns_sorted()
    dist = dists.SparseCategorical(num_cats=3)
    dist.set_meta_parameters(64, torch.tensor([0, 2, 5, 5]), torch.tensor([0, 1, 40, 3, 63]))
    assert not dist.columns_sorted()
    dist = dists.SparseCategorical(num_cats=3)                          # descending across columns is fine
    dist.set_meta_parameters(64, torch.tensor([0, 2, 2, 4]), torch.tensor([10, 20, 1, 2]))
    assert dist.columns_sorted()


def test_sparse_dispatch_cost_model():
    """The cost model (:func:`_ws_sparse_plan`, fitted 2026-10-24 on the
    forced kernel maps of the four 8k / 16k chains) takes the sparse forward
    on the density-0.01 chains from B=64 on — 8-row tiles up to B=512 and
    4-row tiles above on 8k d0.01 (where they measured faster), 8-row tiles on
    16k d0.01 — and leaves the density-0.1 chains to the full and tiles
    passes, which were faster there; never in a backward, never without the
    tile fractions. Model inputs are the trained chains' expected supports
    (mass-weighted, sorted states). Pure host logic, no GPU."""
    d = sio._ws_dispatch
    cases = [   # (H, k_in, f(4), f(8), f(64), expected (decision, sparse tile height) per B)
        (8192, 508, 0.107, 0.143, 0.361, {16: (None, 0), 64: ("sparse", 8), 256: ("sparse", 8),
                                          512: ("sparse", 8), 1024: ("sparse", 4), 2048: ("sparse", 4)}),
        (16384, 1592, 0.158, 0.203, 0.425, {64: ("sparse", 8), 256: ("sparse", 8), 512: ("sparse", 8)}),
        (8192, 1382, 0.267, 0.327, 0.611, {64: ("full", 0), 256: ("full", 0), 1024: ("tiles", 0),
                                           2048: ("tiles", 0)}),
        (16384, 3196, 0.301, 0.360, 0.606, {64: ("full", 0), 256: ("full", 0), 512: ("tiles", 0)}),
    ]
    for H, k, f4, f8, f64, expect in cases:
        fs = {4: f4, 8: f8}
        for B, (use, tm) in expect.items():
            got = d(B, H, H, B * k * k, k, False, f_tiles=f64, BB=128, n_tiles=H // 64, f_sparse=fs)
            got_tm = sio._ws_sparse_plan(B, H, H, k, fs)[1] if got == "sparse" else 0
            assert (got, got_tm) == (use, tm), (H, k, B, got, got_tm)
    fs = {4: 0.107, 8: 0.143}
    assert d(1024, 8192, 8192, 1024 * 508 * 508, 508, True, f_tiles=0.361, BB=128, n_tiles=128,
             f_sparse=fs) == "tiles"                                       # backward: not a candidate
    assert d(1024, 8192, 8192, 1024 * 508 * 508, 508, False, f_tiles=0.361, BB=128,
             n_tiles=128) == "tiles"                                       # no tile fractions
