"""CUDA-graph forward of sparse-emission Monarch chains: the sparse
block-diagonal layers (:class:`SparseInputBlockDiagonalSumLayer` /
:class:`SparseOutputBlockDiagonalSumLayer`) take part in
``pyjuice.model.sparse_cudagraph`` like the sparse-IO layers.

Same contract as ``sparse_io_cudagraph_test.py``: a replay agrees with the
eager forward on any batch, an eager backward after a replay (or after the
capture call) gives the flows of an eager forward + backward, and replays
follow in-place EM updates — per-sample kernels (``"never"``) and the dense
batched path (``"always"``), with and without the second Monarch permutation,
and with a token whose emission column is empty.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.layer.sparse_io_sum_layer as sio
from pyjuice.layer import SparseInputBlockDiagonalSumLayer, SparseOutputBlockDiagonalSumLayer
from pyjuice.structures import SparseMonarchHMM, random_emission_csc


DEV = torch.device("cuda:0")


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _rand_rows(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*shape, generator=g) + 0.05
    return x / x.sum(dim=-1, keepdim=True)


def _build(T, H, V, bs, density, output_permutation, seed=0, drop_col0=False):
    indptr, indices, values, _ = random_emission_csc(H, V, density, seed=seed)
    if drop_col0:
        # Token 0 loses its whole emission column: P(x_t = 0) = 0.
        n0 = int(indptr[1])
        indices, values = indices[n0:], values[n0:]
        indptr = torch.cat([indptr[:1], indptr[1:] - n0])
    NB = H // bs
    root = SparseMonarchHMM(T, H, V, indptr, indices, values, block_size=bs,
                            bd1=_rand_rows(NB, bs, bs, seed=seed + 1),
                            bd2=_rand_rows(NB, bs, bs, seed=seed + 2),
                            gamma=_rand_rows(H, seed=seed + 3),
                            output_permutation=output_permutation)
    pc = juice.TensorCircuit(root, verbose=False).to(DEV)
    layers = [layer for lg in pc.inner_layer_groups for layer in lg]
    assert any(type(l) is SparseInputBlockDiagonalSumLayer for l in layers)
    assert any(type(l) is SparseOutputBlockDiagonalSumLayer for l in layers)
    assert pc._graph_reason is None
    return pc


def _eager(pc, x):
    return pc(x, apply_cudagraph=False).detach().clone()


def _replay(pc, x):
    out = pc(x).detach().clone()
    assert pc._sparse_graph_fwd is not None, "expected a graph replay"
    return out


def _flows(pc, x):
    pc.backward(x, compute_param_flows=True, flows_memory=0.0)
    emit = torch.cat([layer.param_flows.detach().reshape(-1).clone()
                      for layer in pc.input_layer_group])
    return pc.param_flows.detach().clone(), emit


def _assert_flows_close(got, ref, rtol, what):
    scale = ref.abs().max().clamp_min(1e-12)
    err = ((got - ref).abs().max() / scale).item()
    assert err < rtol, f"{what}: max |diff| / max |ref| = {err:.2e} (tol {rtol:.0e})"


@pytest.mark.parametrize("ws_mode", ["never", "always"])
@pytest.mark.parametrize("output_permutation", [True, False])
@pytest.mark.parametrize("T,H,V,bs,density,B,drop_col0", [
    (6, 64, 12, 8, 0.3, 1, False),
    (6, 64, 12, 8, 0.3, 3, True),
    (5, 256, 40, 16, 0.2, 16, False),
    (5, 256, 40, 16, 0.2, 33, True),
])
def test_replay_matches_eager(monkeypatch, ws_mode, output_permutation, T, H, V, bs,
                              density, B, drop_col0):
    monkeypatch.setattr(sio, "_WS_MODE", ws_mode)
    pc = _build(T, H, V, bs, density, output_permutation, drop_col0=drop_col0)
    batches = [torch.randint(0, V, (B, T), device=DEV) for _ in range(3)]
    if drop_col0:
        batches[1][0, 2] = 0
    refs = [_eager(pc, x) for x in batches]
    if drop_col0:
        assert refs[1][0].item() == float("-inf")

    torch.testing.assert_close(pc(batches[0], record_cudagraph=True).detach(), refs[0],
                               rtol=0.0, atol=1e-4)
    assert pc._sparse_graphs[B].graph is not None
    for x, ref in zip(batches + batches[::-1], refs + refs[::-1]):
        torch.testing.assert_close(_replay(pc, x), ref, rtol=0.0, atol=1e-4)
    for x, ref in zip(batches, refs):
        torch.testing.assert_close(_replay(pc, x.cpu()), ref, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("ws_mode", ["never", "always"])
@pytest.mark.parametrize("output_permutation", [True, False])
@pytest.mark.parametrize("B", [1, 12, 20])
def test_backward_after_replay_matches_eager(monkeypatch, ws_mode, output_permutation, B):
    """The eager backward rebinds what the replay wrote (BD2's sparse outputs,
    both layers' block offsets / dense-path markers); its flows must equal
    those after an eager forward — on the capture call and on later replays."""
    monkeypatch.setattr(sio, "_WS_MODE", ws_mode)
    T, H, V, bs = 5, 256, 40, 16
    pc = _build(T, H, V, bs, 0.2, output_permutation)
    xs = [torch.randint(0, V, (B, T), device=DEV) for _ in range(2)]

    refs = []
    for x in xs:
        ll = _eager(pc, x)
        refs.append((ll, *_flows(pc, x)))

    for i, x in enumerate([xs[0], xs[1], xs[0]]):
        ll = pc(x, record_cudagraph=True).detach().clone() if i == 0 else _replay(pc, x)
        pf, emit = _flows(pc, x)
        ref_ll, ref_pf, ref_emit = refs[0 if i != 1 else 1]
        torch.testing.assert_close(ll, ref_ll, rtol=0.0, atol=1e-4)
        _assert_flows_close(pf, ref_pf, 1e-4, f"transition param flows (pass {i})")
        _assert_flows_close(emit, ref_emit, 1e-4, f"emission param flows (pass {i})")


@pytest.mark.parametrize("ws_mode", ["never", "always"])
def test_replay_follows_em_updates(monkeypatch, ws_mode):
    """EM updates the parameters in place, so the captured graph must see them."""
    monkeypatch.setattr(sio, "_WS_MODE", ws_mode)
    T, H, V, bs, B = 5, 256, 40, 16, 16
    pc = _build(T, H, V, bs, 0.2, output_permutation=True)
    x = torch.randint(0, V, (B, T), device=DEV)
    pc(x, record_cudagraph=True)
    for _ in range(2):
        pc.init_param_flows(flows_memory=0.0)
        _replay(pc, x)
        pc.backward(x, flows_memory=1.0, allow_modify_flows=False, logspace_flows=False)
        pc.mini_batch_em(step_size=0.5, pseudocount=1e-6)
        y = torch.randint(0, V, (B, T), device=DEV)
        ref = _eager(pc, y)
        torch.testing.assert_close(_replay(pc, y), ref, rtol=0.0, atol=1e-4)
