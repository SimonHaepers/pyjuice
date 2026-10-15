"""
Dense batched path of the sparse block-diagonal layers
(:class:`SparseInputBlockDiagonalSumLayer` / BD1 and
:class:`SparseOutputBlockDiagonalSumLayer` / BD2 of a sparse-emission Monarch
chain): from ``_WS_MIN_BATCH`` samples up they densify their sparse side and
run the dense :class:`BlockDiagonalSumLayer` kernels instead of the per-sample
gather / atomic kernels.

Every test runs one compiled circuit twice, with the sparse-IO WS knob at
``"never"`` (per-sample kernels, the reference) and ``"always"`` / ``"auto"``,
and compares LLs, all transition param flows and the emission param flows,
with and without the second Monarch permutation (row-mapped BD2), with and
without ``allow_modify_flows``, and over a full EM step.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.layer.sparse_io_sum_layer as sio
from pyjuice.layer import SparseInputBlockDiagonalSumLayer, SparseOutputBlockDiagonalSumLayer
from pyjuice.layer.sparse_io_block_diagonal_sum_layer import _DENSE_BATCHED
from pyjuice.structures import SparseMonarchHMM, random_emission_csc


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


def _rand_rows(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*shape, generator=g) + 0.05
    return x / x.sum(dim=-1, keepdim=True)


def _sparse_monarch(T, H, V, bs, density, seed, output_permutation):
    indptr, indices, values, _ = random_emission_csc(H, V, density, seed=seed)
    NB = H // bs
    return SparseMonarchHMM(T, H, V, indptr, indices, values, block_size=bs,
                            bd1=_rand_rows(NB, bs, bs, seed=seed + 1),
                            bd2=_rand_rows(NB, bs, bs, seed=seed + 2),
                            gamma=_rand_rows(H, seed=seed + 3),
                            output_permutation=output_permutation)


def _bd_layers(pc):
    return [layer for lg in pc.inner_layer_groups for layer in lg
            if isinstance(layer, (SparseInputBlockDiagonalSumLayer,
                                  SparseOutputBlockDiagonalSumLayer))]


def _took_dense(pc) -> bool:
    """Whether the last forward ran every sparse BD block on the dense path."""
    cached = []
    for layer in _bd_layers(pc):
        cached += getattr(layer, "_cached_in_offsets", []) + getattr(layer, "_cached_out_offsets", [])
    flags = [c is _DENSE_BATCHED for c in cached if c is not None]
    assert flags, "no sparse BD block ran"
    assert all(flags) or not any(flags), "mixed dense / per-sample blocks"
    return flags[0]


def _run(pc, data, allow_modify_flows):
    lls = pc(data)
    dense = _took_dense(pc)
    pc.backward(data, compute_param_flows=True, flows_memory=0.0,
                allow_modify_flows=allow_modify_flows)
    in_pf = torch.cat([layer.param_flows.reshape(-1) for layer in pc.input_layer_group])
    return lls.detach().clone(), pc.param_flows.detach().clone(), in_pf.detach().clone(), dense


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


CASES = [
    # T, H, V, bs, density
    (6, 64, 12, 8, 0.3),
    (5, 256, 40, 16, 0.2),
]


@pytest.mark.parametrize("T,H,V,bs,density", CASES)
@pytest.mark.parametrize("output_permutation", [True, False])
@pytest.mark.parametrize("B", [16, 33])
@pytest.mark.parametrize("allow_modify_flows", [False, True])
def test_dense_batched_matches_per_sample(monkeypatch, T, H, V, bs, density,
                                          output_permutation, B, allow_modify_flows):
    device = _cuda()
    root = _sparse_monarch(T, H, V, bs, density, seed=7, output_permutation=output_permutation)
    pc = juice.TensorCircuit(root, verbose=False).to(device)
    assert len(_bd_layers(pc)) == 2 * (T - 1)
    torch.manual_seed(1)
    data = torch.randint(0, V, (B, T), device=device)

    monkeypatch.setattr(sio, "_WS_MODE", "never")
    lls_r, pf_r, in_pf_r, dense_r = _run(pc, data, allow_modify_flows)
    monkeypatch.setattr(sio, "_WS_MODE", "always")
    lls_d, pf_d, in_pf_d, dense_d = _run(pc, data, allow_modify_flows)

    assert not dense_r and dense_d
    assert torch.isfinite(lls_r).all()
    torch.testing.assert_close(lls_d, lls_r, rtol=1e-5, atol=1e-4)
    # The dense backward reduces on the tensor cores (tf32).
    assert _rel(pf_d, pf_r) < 5e-3, f"transition param flows: rel err {_rel(pf_d, pf_r):.2e}"
    assert _rel(in_pf_d, in_pf_r) < 5e-3, f"emission param flows: rel err {_rel(in_pf_d, in_pf_r):.2e}"


def test_dense_batched_auto_threshold(monkeypatch):
    """``"auto"``: per-sample kernels below ``_WS_MIN_BATCH``, dense from it."""
    device = _cuda()
    T, H, V, bs = 5, 64, 12, 8
    pc = juice.TensorCircuit(_sparse_monarch(T, H, V, bs, 0.3, seed=3, output_permutation=True),
                             verbose=False).to(device)
    monkeypatch.setattr(sio, "_WS_MODE", "auto")
    monkeypatch.setattr(sio, "_WS_MIN_BATCH", 16)
    for B, dense in ((1, False), (8, False), (15, False), (16, True), (40, True)):
        pc(torch.randint(0, V, (B, T), device=device))
        if B == 1:
            assert not any(c is _DENSE_BATCHED for layer in _bd_layers(pc)
                           for c in getattr(layer, "_cached_in_offsets", [])
                           + getattr(layer, "_cached_out_offsets", []))
        else:
            assert _took_dense(pc) == dense, f"B={B}"


@pytest.mark.parametrize("output_permutation", [True, False])
def test_dense_batched_em_step(monkeypatch, output_permutation):
    """A full pc-arena style EM step (log-space flows off, as for every sparse
    circuit) lands on the same parameters either way."""
    device = _cuda()
    T, H, V, bs, B = 6, 256, 40, 16, 32
    torch.manual_seed(2)
    data = torch.randint(0, V, (B, T), device=device)
    params = {}
    for mode in ("never", "always"):
        monkeypatch.setattr(sio, "_WS_MODE", mode)
        pc = juice.TensorCircuit(
            _sparse_monarch(T, H, V, bs, 0.2, seed=9, output_permutation=output_permutation),
            verbose=False).to(device)
        pc.init_param_flows(flows_memory=0.0)
        pc(data)
        pc.backward(data, flows_memory=1.0, allow_modify_flows=False, logspace_flows=False)
        pc.mini_batch_em(step_size=0.5, pseudocount=1e-6)
        params[mode] = (pc.params.detach().clone(),
                        torch.cat([layer.params.reshape(-1) for layer in pc.input_layer_group]))
    for name, a, b in zip(("transition", "emission"), params["always"], params["never"]):
        assert _rel(a, b) < 2e-3, f"{name} params after EM: rel err {_rel(a, b):.2e}"
