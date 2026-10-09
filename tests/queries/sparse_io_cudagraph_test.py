"""CUDA-graph forward of sparse-IO circuits (``pyjuice.model.sparse_cudagraph``).

``pc(x, record_cudagraph=True)`` captures the sparse chain once per batch size
with every column resolved on device; later calls with that batch size replay
it. A replay must agree with the eager forward on *any* batch (not only the
captured one), an eager backward after a replay must produce the flows of an
eager forward + backward, and replays must follow in-place EM updates.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.layer.sparse_io_sum_layer as sio
import pyjuice.nodes.distributions as dists
from pyjuice.layer import ProdLayer, SparseInputSumLayer, SparseProdLayer, SumLayer
from pyjuice.nodes import inputs, multiply, set_block_size, summate
from pyjuice.structures import BlockedHMM, SparseHMM, random_emission_csc


DEV = torch.device("cuda:0")


def _build(H, V, T, bs, density, seed=0, drop_col0=False):
    indptr, indices, _, _ = random_emission_csc(H, V, density, seed=seed)
    if drop_col0:
        # Token 0 loses its whole emission column: P(x_t = 0) = 0.
        indices = indices[indptr[1]:]
        indptr = torch.cat([indptr[:1], indptr[1:] - indptr[1]])
    torch.manual_seed(seed)
    ns = SparseHMM(T, H, V, indptr, indices, block_size=bs)
    ns.init_parameters()
    return juice.TensorCircuit(ns, verbose=False).to(DEV)


def _build_tree(plain_subtree_roots=False, H=64, V=30, bs=16, density=0.3, seed=0):
    """Not a chain: two binary trees over 7 variables each, every leaf
    SparseCategorical and every inner product ``leaf (x) sum (x) sum``
    (SparseProdLayer with two dense children feeding SparseInputSumLayers),
    joined by a plain product + sum on top. ``plain_subtree_roots`` pins the
    two subtree roots to plain sum layers, so their sparse children have to
    scatter into ``element_mars`` (``_skip_scatter=False``)."""
    nb = H // bs

    def leaf(var):
        indptr, indices, values, _ = random_emission_csc(H, V, density, seed=seed + var)
        ns = inputs(var, num_node_blocks=nb, dist=dists.SparseCategorical(num_cats=V),
                    csc_indptr=indptr, csc_indices=indices)
        ns.set_params(values, normalize=False)
        return ns

    def subtree(v0):
        leaves = {v: leaf(v) for v in range(v0, v0 + 7)}
        s = {v: summate(leaves[v], num_node_blocks=nb) for v in range(v0 + 3, v0 + 7)}
        s1 = summate(multiply(leaves[v0 + 1], s[v0 + 3], s[v0 + 4]), num_node_blocks=nb)
        s2 = summate(multiply(leaves[v0 + 2], s[v0 + 5], s[v0 + 6]), num_node_blocks=nb)
        return summate(multiply(leaves[v0], s1, s2), num_node_blocks=nb,
                       _force_plain=plain_subtree_roots)

    with set_block_size(bs):
        top = multiply(subtree(0), subtree(7))
    root = summate(top, num_node_blocks=1, block_size=1)
    torch.manual_seed(seed)
    root.init_parameters()
    pc = juice.TensorCircuit(root, verbose=False).to(DEV)
    layers = [layer for lg in pc.inner_layer_groups for layer in lg]
    assert any(type(l) is ProdLayer for l in layers) and any(type(l) is SumLayer for l in layers)
    assert any(type(l) is SparseInputSumLayer for l in layers)
    sparse_prods = [l for l in layers if type(l) is SparseProdLayer]
    assert any(max(ns.num_dense_chs for ns in l.nodes) == 2 for l in sparse_prods)
    assert any(not l._skip_scatter for l in sparse_prods) == plain_subtree_roots
    return pc, V


def _set_ws(monkeypatch, mode, prec="ieee"):
    monkeypatch.setattr(sio, "_WS_MODE", mode)
    monkeypatch.setattr(sio, "_WS_PRECISION", prec)


def _eager(pc, x):
    return pc(x, apply_cudagraph=False).detach().clone()


def _replay(pc, x):
    out = pc(x).detach().clone()
    assert pc._sparse_graph_fwd is not None, "expected a graph replay"
    return out


def _pass(pc, x, graph):
    ll = _replay(pc, x) if graph else _eager(pc, x)
    pc.backward(x, compute_param_flows=True, flows_memory=0.0)
    emit = torch.cat([layer.param_flows.detach().reshape(-1).clone()
                      for layer in pc.input_layer_group])
    return ll, pc.param_flows.detach().clone(), emit


def _assert_flows_close(got, ref, rtol, what):
    scale = ref.abs().max().clamp_min(1e-12)
    err = ((got - ref).abs().max() / scale).item()
    assert err < rtol, f"{what}: max |diff| / max |ref| = {err:.2e} (tol {rtol:.0e})"


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize("ws_mode", ["never", "always"])
@pytest.mark.parametrize("H,V,T,bs,density,B", [
    (64, 20, 6, 16, 0.3, 1),
    (64, 20, 6, 16, 0.3, 3),
    (96, 30, 5, 32, 0.25, 9),      # 3 PC blocks (non power-of-2 row count)
    (128, 40, 8, 32, 0.2, 16),
    (128, 40, 7, 16, 0.2, 70),
])
def test_replay_matches_eager(monkeypatch, ws_mode, H, V, T, bs, density, B):
    _set_ws(monkeypatch, ws_mode)
    pc = _build(H, V, T, bs, density)
    batches = [torch.randint(0, V, (B, T), device=DEV) for _ in range(3)]
    refs = [_eager(pc, x) for x in batches]

    torch.testing.assert_close(pc(batches[0], record_cudagraph=True).detach(), refs[0],
                               rtol=0.0, atol=1e-4)
    assert pc._sparse_graphs[B].graph is not None
    # Replays on other batches (in both orders) and on host batches, which
    # go through the pinned staging copy.
    for x, ref in zip(batches + batches[::-1], refs + refs[::-1]):
        torch.testing.assert_close(_replay(pc, x), ref, rtol=0.0, atol=1e-4)
    for x, ref in zip(batches, refs):
        torch.testing.assert_close(_replay(pc, x.cpu()), ref, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("ws_mode", ["never", "always"])
@pytest.mark.parametrize("B", [1, 12])
def test_backward_after_replay_matches_eager(monkeypatch, ws_mode, B):
    """The eager backward rebinds the sum outputs the replay wrote; its flows
    must equal those after an eager forward — including on the capture call
    itself, which is how pc-arena records (forward + backward of batch 0)."""
    _set_ws(monkeypatch, ws_mode)
    H, V, T, bs = 128, 40, 8, 32
    pc = _build(H, V, T, bs, 0.2, seed=1)
    d0, d1 = (torch.randint(0, V, (B, T), device=DEV) for _ in range(2))
    ref0, ref1 = _pass(pc, d0, graph=False), _pass(pc, d1, graph=False)

    pc(d0, record_cudagraph=True)
    pc.backward(d0, compute_param_flows=True, flows_memory=0.0)
    emit0 = torch.cat([l.param_flows.detach().reshape(-1) for l in pc.input_layer_group])
    got0 = (None, pc.param_flows.detach().clone(), emit0)
    got1 = _pass(pc, d1, graph=True)
    got1b = _pass(pc, d1, graph=True)        # second replay + backward, same batch

    for got, ref in ((got0, ref0), (got1, ref1), (got1b, ref1)):
        if got[0] is not None:
            torch.testing.assert_close(got[0], ref[0], rtol=0.0, atol=1e-4)
        _assert_flows_close(got[1], ref[1], 1e-4, "transition param flows")
        _assert_flows_close(got[2], ref[2], 1e-4, "emission param flows")


@pytest.mark.parametrize("prec", ["ieee", "tf32"])
def test_replay_follows_em_updates(monkeypatch, prec):
    """EM rewrites the parameters in place: a replay must read the new values,
    including the tf32 copy of W that the WS path re-rounds every pass. (The
    param flows accumulate with atomics, so even two eager runs drift apart by
    ~1e-4 nats here once tf32 rounds the updated W — hence the tf32 atol.)"""
    atol = {"ieee": 1e-4, "tf32": 5e-3}[prec]
    _set_ws(monkeypatch, "always", prec)
    H, V, T, bs = 128, 40, 8, 32
    data = [torch.randint(0, V, (16, T), device=DEV) for _ in range(3)]
    results = {}
    for graph in (False, True):
        pc = _build(H, V, T, bs, 0.2, seed=3)
        if graph:
            pc(data[0], record_cudagraph=True)
        lls = []
        for x in data:
            lls.append(_pass(pc, x, graph)[0])
            pc.mini_batch_em(step_size=0.5, pseudocount=1e-3, use_cudagraph=False)
        lls.append(_replay(pc, data[0]) if graph else _eager(pc, data[0]))
        results[graph] = (lls, pc.params.detach().clone())

    for got, ref in zip(results[True][0], results[False][0]):
        torch.testing.assert_close(got, ref, rtol=0.0, atol=atol)
    assert (results[True][1] - results[False][1]).abs().max().item() < 1e-4


def test_batch_size_switching(monkeypatch):
    """One graph per batch size, each replaying into its own buffers; an
    eager pass that grows the WS scratch in between must not disturb them."""
    _set_ws(monkeypatch, "always")
    H, V, T, bs = 128, 40, 6, 32
    pc = _build(H, V, T, bs, 0.2, seed=4)
    xs = {B: torch.randint(0, V, (B, T), device=DEV) for B in (4, 2, 40)}
    refs = {B: _eager(pc, x) for B, x in xs.items()}

    pc(xs[4], record_cudagraph=True)
    pc(xs[2], record_cudagraph=True)
    for B in (4, 2, 4):
        torch.testing.assert_close(_replay(pc, xs[B]), refs[B], rtol=0.0, atol=1e-4)

    torch.testing.assert_close(_eager(pc, xs[40]), refs[40], rtol=0.0, atol=1e-4)
    for B in (2, 4):
        torch.testing.assert_close(_replay(pc, xs[B]), refs[B], rtol=0.0, atol=1e-4)
    # No graph for B=40: the default ``apply_cudagraph`` runs it eagerly.
    torch.testing.assert_close(pc(xs[40]).detach(), refs[40], rtol=0.0, atol=1e-4)
    assert pc._sparse_graph_fwd is None


@pytest.mark.parametrize("B", [1, 5])
def test_empty_columns(B):
    """A token outside the emission support makes the sequence impossible.
    The graph runs the batched kernels at every B, which clamp an all-``-inf``
    step max to ``-1e30``: such sequences score ``-inf`` or ``~-1e30``, as on
    the eager batched path (the eager B=1 kernels give NaN there)."""
    H, V, T, bs = 64, 20, 6, 16
    pc = _build(H, V, T, bs, 0.3, seed=5, drop_col0=True)
    x = torch.randint(1, V, (B, T), device=DEV)
    x[0, 2] = 0
    if B > 1:
        x[3, T - 1] = 0
        x[4, 0] = 0
    impossible = (x == 0).any(dim=1)
    ref = _eager(pc, x)
    assert torch.isfinite(ref[~impossible]).all()

    pc(x, record_cudagraph=True)
    got = _replay(pc, x)
    assert (got[impossible] < -1e29).all()
    torch.testing.assert_close(got[~impossible], ref[~impossible], rtol=0.0, atol=1e-4)
    if B > 1:
        torch.testing.assert_close(got[impossible], ref[impossible], rtol=1e-6, atol=0.0)
    # The same graph on a batch without the token is finite again.
    x2 = torch.randint(1, V, (B, T), device=DEV)
    torch.testing.assert_close(_replay(pc, x2), _eager(pc, x2), rtol=0.0, atol=1e-4)


def test_missing_mask_falls_back_to_eager():
    """Marginalised positions change the patterns' shape on the host: no
    graph, a one-time warning, and the eager result."""
    H, V, T, bs = 64, 20, 6, 16
    pc = _build(H, V, T, bs, 0.3, seed=6)
    x = torch.randint(0, V, (1, T), device=DEV)
    mm = torch.zeros(1, T, dtype=torch.bool, device=DEV)
    mm[0, 2] = True
    ref = pc(x, missing_mask=mm, apply_cudagraph=False).detach().clone()
    with pytest.warns(UserWarning, match="missing_mask"):
        got = pc(x, missing_mask=mm, record_cudagraph=True).detach().clone()
    assert not pc._sparse_graphs and pc._sparse_graph_fwd is None
    torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("B", [1, 5])
def test_tree_replay_matches_eager(B):
    """The graph forward is not tied to the HMM chain: sparse prods with two
    dense children, sparse-input sums and plain dense layers, several nodes
    per layer, in one captured circuit; the eager backward after a replay
    gives the eager flows."""
    pc, V = _build_tree()
    assert pc._graph_reason is None
    batches = [torch.randint(0, V, (B, 14), device=DEV) for _ in range(3)]
    refs = [_pass(pc, x, graph=False) for x in batches]
    pc(batches[0], record_cudagraph=True)
    for x, ref in zip(batches, refs):
        got = _pass(pc, x, graph=True)
        torch.testing.assert_close(got[0], ref[0], rtol=0.0, atol=1e-4)
        _assert_flows_close(got[1], ref[1], 1e-4, "sum param flows")
        _assert_flows_close(got[2], ref[2], 1e-4, "input param flows")


def test_tree_dense_consumer_replay():
    """A SparseProdLayer whose consumer is a plain sum layer scatters into
    ``element_mars`` inside the graph. Eager supports that topology at B=1
    only; the graph scatter works at any B (checked per sample)."""
    pc, V = _build_tree(plain_subtree_roots=True)
    xs = torch.randint(0, V, (4, 14), device=DEV)
    refs = [_pass(pc, xs[i:i + 1], graph=False) for i in range(4)]

    pc(xs[:1], record_cudagraph=True)
    for i, ref in enumerate(refs):
        got = _pass(pc, xs[i:i + 1], graph=True)
        torch.testing.assert_close(got[0], ref[0], rtol=0.0, atol=1e-4)
        _assert_flows_close(got[1], ref[1], 1e-4, "sum param flows")
        _assert_flows_close(got[2], ref[2], 1e-4, "input param flows")

    pc(xs, record_cudagraph=True)
    torch.testing.assert_close(_replay(pc, xs), torch.cat([r[0] for r in refs]),
                               rtol=0.0, atol=1e-4)


def test_layer_without_graph_forward_falls_back(monkeypatch):
    """A layer that opts out through ``graph_unsupported_reason`` keeps the
    whole circuit eager, with a one-time warning naming the reason."""
    pc, V = _build_tree()
    layer = next(l for lg in pc.inner_layer_groups for l in lg if type(l) is SparseInputSumLayer)
    monkeypatch.setattr(layer, "graph_unsupported_reason", lambda: "test opt-out")
    pc._graph_reason = juice.model.sparse_cudagraph.graph_unsupported_reason(pc)
    x = torch.randint(0, V, (3, 14), device=DEV)
    ref = _eager(pc, x)
    with pytest.warns(UserWarning, match="test opt-out"):
        got = pc(x, record_cudagraph=True).detach().clone()
    assert not pc._sparse_graphs
    torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("B", [1, 8])
def test_blocked_hmm_replay_matches_eager(B):
    """The blocked layers read the tokens on device inside the graph: replays
    must see each call's batch (the generic graph path feeds the inner layers
    a fixed copy of it), not the recorded one."""
    T, H, V, M = 16, 256, 64, 8
    torch.manual_seed(0)
    root = BlockedHMM(seq_length=T, num_latents=H, num_emits=V, num_blocks=M,
                      token_block=torch.arange(V) % M)
    root.init_parameters()
    pc = juice.TensorCircuit(root, verbose=False).to(DEV)
    batches = [torch.randint(0, V, (B, T), device=DEV) for _ in range(3)]
    refs = [_eager(pc, x) for x in batches]
    pc(batches[0], record_cudagraph=True)
    for x, ref in zip(batches + batches[::-1], refs + refs[::-1]):
        torch.testing.assert_close(pc(x).detach(), ref, rtol=0.0, atol=1e-5)


def test_sparse_backward_graph_refused():
    """The sparse-IO backward reads host-side column sizes: recording its
    graph is refused with a warning and the backward runs eagerly."""
    pc = _build(64, 20, 6, 16, 0.3, seed=7)
    x = torch.randint(0, 20, (4, 6), device=DEV)
    ref = _pass(pc, x, graph=False)
    pc(x, apply_cudagraph=False)
    with pytest.warns(UserWarning, match="backward"):
        pc.backward(x, compute_param_flows=True, flows_memory=0.0, record_cudagraph=True)
    assert not any(k[0] == 1 for k in pc._recorded_cuda_graphs)
    _assert_flows_close(pc.param_flows.detach().clone(), ref[1], 1e-4, "transition param flows")
