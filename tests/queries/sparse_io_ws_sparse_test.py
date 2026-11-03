"""The batched sparse pass of :class:`SparseIOSumLayer`
(:mod:`pyjuice.layer.sparse_io_cuda`): for every parent tile of 4 / 8 / 16
rows the samples with an active parent in it walk their packed input columns
against the tile's W slab in shared memory and write the packed output; the
element-flow backward is its mirror. It must give the gather kernels'
log-likelihoods (and, through them, flows) at every batch size from 2 on,
eagerly and under a CUDA graph, on state counts that are not powers of two,
with fp32 and bf16 slabs, with the children split into several ranges, with
empty columns and masked variables, on a circuit that is not a chain, on
both parameter-flow paths (accumulator tiles, deferred dense GEMM) and
across accumulated / negated backward calls; B=1 and blocks with unsorted
columns must stay on the gather kernels.
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


def _set(monkeypatch, mode, prec="ieee", **knobs):
    """``mode``: ``_WS_MODE`` (``"never"`` = the gather kernels, the reference;
    ``"auto"`` = the sparse pass from B=2 on); ``prec``: the deferred GEMM's
    precision; ``knobs``: ``_WS_SPARSE_<name>`` (``SLAB_TM`` / ``SLAB_BYTES`` /
    ``SLAB_NWARP`` / ``SLAB_MINB`` / ``SLAB_W16`` / ``PFLOW`` / ``PF_TILE_*``)."""
    monkeypatch.setattr(sio, "_WS_MODE", mode)
    monkeypatch.setattr(sio, "_WS_PRECISION", prec)
    knobs.setdefault("SLAB_W16", False)       # the exactness tests read W in fp32; test_sparse_w16 covers bf16
    for k, v in knobs.items():
        if k == "SLAB_NWARP":
            v = {t: v for t in sio_cuda.TILE_HEIGHTS}
        monkeypatch.setattr(sio, f"_WS_SPARSE_{k}", v)


def _count(monkeypatch):
    """Calls of the sparse forward / backward block methods."""
    calls = {"fwd": 0, "bwd": 0}
    fwd, bwd = sio.SparseIOSumLayer._ws_forward_block, sio.SparseIOSumLayer._ws_backward_block

    def counted_fwd(self, *a, **k):
        calls["fwd"] += 1
        return fwd(self, *a, **k)

    def counted_bwd(self, *a, **k):
        calls["bwd"] += 1
        return bwd(self, *a, **k)

    monkeypatch.setattr(sio.SparseIOSumLayer, "_ws_forward_block", counted_fwd)
    monkeypatch.setattr(sio.SparseIOSumLayer, "_ws_backward_block", counted_bwd)
    return calls


def _emit_flows(pc):
    return torch.cat([layer.param_flows.detach().reshape(-1).clone() for layer in pc.input_layer_group])


def _pass(pc, data, flows_memory=0.0, negate=False):
    """LL, transition param flows and emission param flows (the backward
    reads the forward's packed node values)."""
    ll = pc(data, apply_cudagraph=False).detach().clone()
    pc.backward(data, compute_param_flows=True, flows_memory=flows_memory, negate_pflows=negate)
    return ll, pc.param_flows.detach().clone(), _emit_flows(pc)


def _assert_flows_close(got, ref, rtol, what):
    scale = ref.abs().max().clamp_min(1e-12)
    err = ((got - ref).abs().max() / scale).item()
    assert err < rtol, f"{what}: max |diff| / max |ref| = {err:.2e} (tol {rtol:.0e})"


@pytest.mark.parametrize("tm", [4, 8, 16])
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
    """The forward and the element-flow backward are exact fp32 with either
    tile height; the parameter flows take the deferred (tf32 or ieee) GEMM
    (W below the tile threshold)."""
    pc = _build(H, V, T, bs, density)
    data = torch.randint(0, V, (B, T), device=DEV)

    _set(monkeypatch, "never")
    ll_ref, pf_ref, emit_ref = _pass(pc, data)

    calls = _count(monkeypatch)
    _set(monkeypatch, "auto", prec, SLAB_TM=tm)
    ll, pf, emit = _pass(pc, data)
    assert calls["fwd"] == T - 1 and calls["bwd"] == T - 1
    torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=_TOL[prec])
    rtol = 1e-4 if prec == "ieee" else 3e-3
    _assert_flows_close(pf, pf_ref, rtol, "transition param flows")
    _assert_flows_close(emit, emit_ref, rtol, "emission param flows")


@pytest.mark.parametrize("tm", [4, 8])
@pytest.mark.parametrize("H,V,T,bs,density,B", [
    (96, 30, 5, 32, 0.25, 9),
    (128, 40, 7, 16, 0.2, 70),
    (2048, 300, 4, 1024, 0.02, 300),
])
def test_sparse_w16(monkeypatch, H, V, T, bs, density, B, tm):
    """bf16 slab-order copies of W (``_WS_SPARSE_SLAB_W16``): the forward and the element-flow
    backward round W to 8 bits of mantissa, so the LLs agree with the gather kernels to a few
    1e-3 per sequence and the flows to 2 % (relative to the largest), and differ from the exact
    pass."""
    pc = _build(H, V, T, bs, density)
    data = torch.randint(0, V, (B, T), device=DEV)
    _set(monkeypatch, "never")
    ll_ref, pf_ref, emit_ref = _pass(pc, data)
    calls = _count(monkeypatch)
    _set(monkeypatch, "auto", "ieee", SLAB_TM=tm, SLAB_W16=True)
    ll, pf, emit = _pass(pc, data)
    assert calls["fwd"] == T - 1 and calls["bwd"] == T - 1
    torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=4e-3 * T)
    _assert_flows_close(pf, pf_ref, 2e-2, "transition param flows")
    _assert_flows_close(emit, emit_ref, 2e-2, "emission param flows")
    assert not torch.equal(ll, ll_ref), "bf16 slabs gave bit-identical LLs: the fp32 copy was read"


def test_sparse_w16_forest(monkeypatch):
    """bf16 slabs on a circuit that is not a chain (two transition matrices per layer, each with its
    own slab-order copies)."""
    pc, V, n_vars = _build_forest()
    x = torch.randint(0, V, (24, n_vars), device=DEV)
    _set(monkeypatch, "never")
    ll_ref, pf_ref, emit_ref = _pass(pc, x)
    _set(monkeypatch, "auto", "ieee", SLAB_TM=4, SLAB_W16=True)
    ll, pf, emit = _pass(pc, x)
    torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=4e-3 * n_vars)
    _assert_flows_close(pf, pf_ref, 2e-2, "sum param flows")
    _assert_flows_close(emit, emit_ref, 2e-2, "emission param flows")


@pytest.mark.parametrize("knobs", [
    dict(SLAB_TM=4, SLAB_BYTES=512, SLAB_NWARP=2),            # 32 children per CTA: 8 ranges
    dict(SLAB_TM=8, SLAB_BYTES=2048),                         # 8-row tiles, 4 ranges
    dict(SLAB_TM=16, SLAB_BYTES=6144, SLAB_NWARP=12, SLAB_MINB=2),   # 16 rows, 3 ranges
    dict(SLAB_TM=4, SLAB_BYTES=512, SLAB_NWARP=2, SLAB_W16=True),    # 8 ranges of bf16 slabs
])
def test_sparse_ranges(monkeypatch, knobs):
    """The children split into several ranges whose partial sums add up in
    the output (and the parents likewise in the backward), with tile heights
    4 to 16 (only the 4-row quarters with one of the sample's parents read);
    density 0.3 at H=256 puts up to all rows of a tile in a sample."""
    pc = _build(256, 60, 5, 128, 0.3, seed=2)
    data = torch.randint(0, 60, (40, 5), device=DEV)
    _set(monkeypatch, "never")
    ref = _pass(pc, data)
    calls = _count(monkeypatch)
    _set(monkeypatch, "auto", "ieee", **knobs)
    got = _pass(pc, data)
    assert calls["fwd"] == 4 and calls["bwd"] == 4
    tol = 2e-2 if knobs.get("SLAB_W16") else 1e-4
    torch.testing.assert_close(got[0], ref[0], rtol=0.0, atol=tol)
    _assert_flows_close(got[1], ref[1], tol, "transition param flows")
    _assert_flows_close(got[2], ref[2], tol, "emission param flows")


@pytest.mark.parametrize("opts", [
    dict(PFLOW="gemm"),                                                   # the deferred dense GEMM
    dict(PFLOW="tile"),                                                   # param flows on accumulator tiles
    dict(PFLOW="tile", PF_TILE_SLOTS=1),                                  # ... accumulated after every step
    dict(PFLOW="tile", PF_TILE_SLOTS=3, PF_TILE_NWARP=2),                 # ... every 3 steps, 2 warps per tile
    dict(PFLOW="tile", PF_TILE_STASH_BYTES=1),                            # ... the stash budget: one step at a time
    dict(PFLOW="tile", PF_TILE_BYTES=1024, PF_TILE_NWARP=4),              # ... 4 ranges of 64 children
])
def test_sparse_pflow_paths(monkeypatch, opts):
    """Both parameter-flow paths give the gather kernels' flows, and a second
    pass (which reuses the slab copies and the accumulator) again."""
    pc = _build(256, 60, 5, 128, 0.3, seed=3)
    data = torch.randint(0, 60, (40, 5), device=DEV)
    _set(monkeypatch, "never")
    ref = _pass(pc, data)
    calls = _count(monkeypatch)
    _set(monkeypatch, "auto", "ieee", **opts)
    got = _pass(pc, data)
    assert calls["fwd"] == 4 and calls["bwd"] == 4
    torch.testing.assert_close(got[0], ref[0], rtol=0.0, atol=1e-4)
    _assert_flows_close(got[1], ref[1], 1e-4, "transition param flows")
    _assert_flows_close(got[2], ref[2], 1e-4, "emission param flows")
    got2 = _pass(pc, data)
    _assert_flows_close(got2[1], ref[1], 1e-4, "transition param flows, second pass")


def test_pflow_mode_auto(monkeypatch):
    """``"auto"`` param flows: the tile kernel for a large W from the batch threshold on (a larger W from
    proportionally fewer samples), the dense GEMM below; an explicit mode is kept."""
    monkeypatch.setattr(sio, "_WS_SPARSE_PFLOW", "auto")
    monkeypatch.setattr(sio, "_WS_SPARSE_PF_TILE_MIN_W", 1 << 26)
    monkeypatch.setattr(sio, "_WS_SPARSE_PF_TILE_MIN_BATCH", 256)
    assert sio._pflow_mode(8192, 8192, 1024) == "tile"
    assert sio._pflow_mode(8192, 8192, 256) == "tile"
    assert sio._pflow_mode(8192, 8192, 128) == "gemm"
    assert sio._pflow_mode(16384, 16384, 64) == "tile"
    assert sio._pflow_mode(16384, 16384, 32) == "gemm"
    assert sio._pflow_mode(2048, 2048, 4096) == "gemm"
    assert sio._pflow_mode(16384, 4096, 256) == "tile"
    monkeypatch.setattr(sio, "_WS_SPARSE_PFLOW", "gemm")
    assert sio._pflow_mode(8192, 8192, 1024) == "gemm"


@pytest.mark.parametrize("H,pad", [(256, 0), (250, 0), (256, 1)])
def test_pflow_tile_corner(H, pad):
    """``pflow_tile_flush`` with a corner leaves exactly the pairs (parent < P, child < C) out: the corner
    run plus a dense outer product over the corner equals the run without one (fp32, random samples; several
    samples share their child list, as samples with the same next token do). The tiles are added to S by 16-byte
    groups (H = 256) or one value at a time: H not a multiple of 4 (250), or S's rows not 16-byte aligned (pad 1)."""
    B, S = 24, 3
    gen = torch.Generator(device="cpu").manual_seed(7)
    lists = [torch.sort(torch.randperm(H, generator=gen)[:k]).values for k in (37, 60, 1, 90)]
    rows = torch.cat(lists).to(DEV)
    starts = torch.tensor([0, 37, 97, 98], dtype=torch.int32)

    class SV:
        def __init__(self, idx, cs, nnz, vals):
            self.indices, self.col_starts, self.nnz, self.values = idx, cs, nnz, vals
            self.total_nnz = vals.shape[1]

    KB = 1024
    KS, nsplit, n_m = sio_cuda.pflow_tile_shape(H, H, KB)
    bufs = None
    ref = torch.zeros(H, H, dtype=torch.float64, device=DEV)
    for s in range(S):
        ci = torch.randint(0, 4, (B,), generator=gen)                         # child list per sample (shared)
        pi = torch.randint(0, 4, (B,), generator=gen)
        cn = torch.tensor([len(lists[i]) for i in ci], dtype=torch.int32)
        pn = torch.tensor([len(lists[i]) for i in pi], dtype=torch.int32)
        K = 90
        v = torch.rand(B, K, generator=gen) * -3.0
        g = torch.rand(B, K, generator=gen) + 0.1
        sv_in = SV(rows, starts[ci].to(DEV), cn.to(DEV), v.to(DEV))
        sv_out = SV(rows, starts[pi].to(DEV), pn.to(DEV), torch.zeros(B, K, device=DEV))
        sv_flow = SV(rows, starts[pi].to(DEV), pn.to(DEV), g.to(DEV))
        if bufs is None:
            i32 = dict(dtype=torch.int32, device=DEV)
            bufs = {"ent": torch.empty(S, n_m, B, 2, **i32), "cnt": torch.zeros(S, n_m, **i32),
                    "xo": torch.empty(S, B, K, **i32), "xv": torch.empty(S, B, K, device=DEV),
                    "cofs": torch.empty(S, B, nsplit + 1, **i32),
                    "gkey": torch.empty(S, B, dtype=torch.int64, device=DEV),
                    "gp": torch.empty(S, B, K, dtype=torch.float32, device=DEV)}
        sio_cuda.pflow_tile_stash(bufs, s, sv_in, sv_out, sv_flow, torch.zeros(B, device=DEV), B, H, H, KB)
        for b in range(B):
            p = lists[pi[b]].to(DEV)
            c = lists[ci[b]].to(DEV)
            ref[p[:, None], c[None, :]] += (g[b, :len(p)].double().to(DEV)[:, None]
                                            * torch.exp(v[b, :len(c)].double()).to(DEV)[None, :])
    full = torch.zeros(H, H + pad, device=DEV)[:, pad:]
    sio_cuda.pflow_tile_flush(full, bufs, S, B, H, H, 1.0, NW=2, slab_bytes=KB)
    torch.testing.assert_close(full.double(), ref, rtol=1e-5, atol=1e-6)
    P, C = 128, 2 * KS
    part = torch.zeros(H, H + pad, device=DEV)[:, pad:]
    sio_cuda.pflow_tile_flush(part, bufs, S, B, H, H, -1.0, NW=2, slab_bytes=KB, corner=(P, C))
    assert torch.all(part[:P, :C] == 0)
    torch.testing.assert_close(-part[P:].double(), ref[P:], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(-part[:P, C:].double(), ref[:P, C:], rtol=1e-5, atol=1e-6)


def test_sparse_empty_columns(monkeypatch):
    """A sample whose token has an empty emission column (impossible
    sequence) gets ``-inf``-like values like the gather kernels; the others
    are unaffected."""
    H, V, T, bs = 64, 20, 6, 16
    indptr, indices, _, _ = random_emission_csc(H, V, 0.3, seed=3)
    indices = indices[indptr[1]:]                    # token 0 loses its column
    indptr = torch.cat([indptr[:1], indptr[1:] - indptr[1]])
    pc = _build(H, V, T, bs, 0.3, seed=3, indptr=indptr, indices=indices)
    x = torch.randint(1, V, (5, T), device=DEV)
    x[1, 2] = 0
    x[3, 0] = 0
    x[0, 2] = x[4, 2] = x[2, 0] = 1                  # token 1 next to the empty columns
    _set(monkeypatch, "never")
    ref = pc(x, apply_cudagraph=False).detach().clone()
    _set(monkeypatch, "auto", "ieee", SLAB_TM=4)
    got = pc(x, apply_cudagraph=False).detach().clone()
    impossible = torch.tensor([False, True, False, True, False], device=DEV)
    assert (got[impossible] < -1e29).all()
    torch.testing.assert_close(got[~impossible], ref[~impossible], rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("H,V,T,bs,density,B", [
    (64, 20, 6, 16, 0.3, 1),            # static B=1 container: the gather kernels
    (96, 30, 5, 32, 0.25, 9),
    (128, 40, 7, 16, 0.2, 70),
])
def test_sparse_graph_replay_matches_eager(monkeypatch, H, V, T, bs, density, B):
    """The per-step lists and ranges are built on the device inside the
    captured graph: replays on new batches give the eager gather kernels,
    and the eager backward after a replay agrees."""
    pc = _build(H, V, T, bs, density, seed=1)
    batches = [torch.randint(0, V, (B, T), device=DEV) for _ in range(3)]
    _set(monkeypatch, "never")
    refs = [pc(x, apply_cudagraph=False).detach().clone() for x in batches]
    _set(monkeypatch, "auto", "ieee")
    calls = _count(monkeypatch)
    pc(batches[0], record_cudagraph=True)
    assert pc._sparse_graph_fwd is not None and pc._sparse_graph_fwd.graph is not None
    assert (calls["fwd"] > 0) == (B > 1)
    for x, ref in zip(batches, refs):
        got = pc(x).detach().clone()
        torch.testing.assert_close(got, ref, rtol=0.0, atol=1e-4)
    _set(monkeypatch, "never")
    _, pf_ref, emit_ref = _pass(pc, batches[1])
    _set(monkeypatch, "auto", "ieee")
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
    _set(monkeypatch, "auto", "ieee")
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
    _set(monkeypatch, "auto", "ieee", SLAB_TM=8)
    for x, (ll_ref, pf_ref, emit_ref) in zip(batches, refs):
        ll, pf, emit = _pass(pc, x)
        torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=1e-4)
        _assert_flows_close(pf, pf_ref, 1e-4, "sum param flows")
        _assert_flows_close(emit, emit_ref, 1e-4, "emission param flows")
    assert calls["fwd"] > 0 and calls["bwd"] > 0
    pc(batches[0], record_cudagraph=True)
    for x, (ll_ref, _, _) in zip(batches, refs):
        torch.testing.assert_close(pc(x).detach(), ll_ref, rtol=0.0, atol=1e-4)


def test_sparse_needs_sorted_columns(monkeypatch):
    """A column whose rows do not ascend makes the block ineligible: it stays
    on the gather kernels, and the result is still right."""
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
    _set(monkeypatch, "auto", "ieee")
    got = pc(x, apply_cudagraph=False).detach().clone()
    assert calls["fwd"] == 0
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


def test_rule_on_batch_size(monkeypatch):
    """The dispatch is a rule on the batch size: B=1 runs the gather kernels,
    every B from 2 on the sparse pass (forward and backward), and ``"never"``
    keeps the gather kernels at any B."""
    H, V, T, bs = 128, 40, 6, 32
    pc = _build(H, V, T, bs, 0.2, seed=5)
    for B, mode, expect in ((1, "auto", 0), (2, "auto", T - 1), (3, "auto", T - 1), (64, "auto", T - 1),
                            (64, "never", 0)):
        calls = _count(monkeypatch)
        _set(monkeypatch, mode)
        _pass(pc, torch.randint(0, V, (B, T), device=DEV))
        assert (calls["fwd"], calls["bwd"]) == (expect, expect), (B, mode, calls)
        monkeypatch.undo()


def test_flow_accumulation_and_negate(monkeypatch):
    """``flows_memory=1`` accumulates across backward calls and
    ``negate_pflows`` flips the deferred flows, as on the gather kernels."""
    H, V, T, bs = 64, 24, 6, 16
    pc = _build(H, V, T, bs, 0.3, seed=2)
    d1 = torch.randint(0, V, (8, T), device=DEV)
    d2 = torch.randint(0, V, (8, T), device=DEV)
    results = {}
    for mode in ("never", "auto"):
        _set(monkeypatch, mode)
        _pass(pc, d1)
        _, pf_acc, _ = _pass(pc, d2, flows_memory=1.0)
        _, pf_neg, _ = _pass(pc, d1, negate=True)
        results[mode] = (pf_acc, pf_neg)
    _assert_flows_close(results["auto"][0], results["never"][0], 1e-4, "accumulated flows")
    _assert_flows_close(results["auto"][1], results["never"][1], 1e-4, "negated flows")
    assert results["auto"][1].sum().item() < 0


def test_em_step_matches_gather(monkeypatch):
    """One mini-batch EM step (forward, backward with parameter flows, update)
    on the sparse pass gives the gather kernels' parameters and LLs."""
    H, V, T, bs = 128, 40, 8, 32
    data = torch.randint(0, V, (20, T), device=DEV)
    params = {}
    for mode in ("never", "auto"):
        _set(monkeypatch, mode)
        pc = _build(H, V, T, bs, 0.2, seed=3)
        _pass(pc, data)
        pc.mini_batch_em(step_size=0.5, pseudocount=1e-3, use_cudagraph=False)
        params[mode] = pc.params.detach().clone()
        params[mode + "_ll"] = pc(data).detach().clone()
    assert (params["auto"] - params["never"]).abs().max().item() < 1e-5
    torch.testing.assert_close(params["auto_ll"], params["never_ll"], rtol=0.0, atol=1e-4)


def test_graph_backward_replay_after_em(monkeypatch):
    """A captured node-flow backward (a conditional query with
    ``record_cudagraph``) replayed after an EM step and a forward replay: the
    sparse pass reads W from a per-pass copy in slab order that the forward
    and the backward share, so the backward's copy must be captured with it
    (gpuq job 623 on snapshot 2026-11-02_lean: the second turn's conditional
    probabilities were off by up to 0.35 relative on the compact layout)."""
    from pyjuice.queries import conditional
    H, V, T, bs, B = 128, 23, 7, 32, 17
    torch.manual_seed(2)
    xs = [torch.randint(0, V, (B, T), device=DEV) for _ in range(2)]
    mask = torch.zeros(T, dtype=torch.bool, device=DEV)
    mask[1] = mask[3] = True                   # the same masked set both turns: one graph, replayed
    results = {}
    for mode in ("never", "auto"):
        _set(monkeypatch, mode, "ieee")
        pc = _build(H, V, T, bs, 0.3, seed=5)
        kw = dict(apply_cudagraph=False) if mode == "never" else dict(record_cudagraph=True)
        out = []
        for x in xs:
            pc(x, **kw)
            pc.backward(x, flows_memory=0.0, allow_modify_flows=False, apply_cudagraph=False)
            pc.mini_batch_em(step_size=0.2, pseudocount=0.1)
            out.append((pc(x, **kw).detach().clone(),
                        conditional(pc, x, missing_mask=mask, allow_modify_flows=False, **kw).clone()))
        results[mode] = out
    for (ll, q), (ll_ref, q_ref) in zip(results["auto"], results["never"]):
        torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=1e-4)
        torch.testing.assert_close(q, q_ref, rtol=0.0, atol=1e-4)
