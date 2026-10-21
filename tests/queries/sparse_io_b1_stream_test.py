"""The hybrid stream / gather single-sample forward of ``SparseIOSumLayer``
(``SparseCategorical.b1_descriptors``, ``PYJUICE_SPARSE_IO_B1_STREAM_FRAC``)
gives the log-likelihoods of the pure gather path at every setting of the
stream fraction, eagerly and under a CUDA graph, on tile-aligned and
unaligned state counts, before and after a latent-state reordering."""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions.sparse_categorical as scat
from pyjuice.structures import SparseHMM
from pyjuice.transformations import reorder_states

from sparse_state_order_test import _random_csc, _tokens


def _chain(T, H, V, bs, seed, density=0.12):
    indptr, indices, values = _random_csc(H, V, density, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    alpha = torch.rand(H, H, generator=g) + 0.01
    gamma = torch.rand(H, generator=g) + 0.01
    root = SparseHMM(T, H, V, indptr, indices, values, block_size=bs,
                     alpha=alpha / alpha.sum(1, keepdim=True), gamma=gamma / gamma.sum())
    return root, indptr


def _lls(root, data, device, frac, graph):
    old = scat.B1_STREAM_FRAC
    scat.B1_STREAM_FRAC = frac
    try:
        pc = juice.TensorCircuit(root, verbose=False).to(device)
        out = []
        for b in range(data.size(0)):          # B=1 forwards, one sequence at a time
            x = data[b:b + 1].to(device)
            if graph:
                pc(x, record_cudagraph=True)
                ll = pc(x)
            else:
                ll = pc(x)
            out.append(ll.detach().cpu())
        return torch.cat(out)
    finally:
        scat.B1_STREAM_FRAC = old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("H,V,bs", [(64, 40, 32), (48, 40, 16), (320, 200, 64)])
@pytest.mark.parametrize("graph", [False, True])
def test_b1_stream_fractions_match_gather(H, V, bs, graph):
    device = torch.device("cuda:0")
    T, B = 5, 4
    root, indptr = _chain(T, H, V, bs, seed=21)
    data = _tokens(indptr, B, T, seed=22)
    ref = _lls(root, data, device, 2.0, graph)          # pure gather (the original path)
    assert torch.isfinite(ref).all()
    for frac in (0.0, 0.125, 0.25, 0.5):
        got = _lls(root, data, device, frac, graph)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (frac, got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_b1_stream_after_reordering():
    device = torch.device("cuda:0")
    T, B, H, V, bs = 5, 3, 256, 150, 64
    root, indptr = _chain(T, H, V, bs, seed=31)
    data = _tokens(indptr, B, T, seed=32)
    ref = _lls(root, data, device, 2.0, False)
    root2, _ = _chain(T, H, V, bs, seed=31)
    reorder_states(root2)
    for graph in (False, True):
        got = _lls(root2, data, device, 0.25, graph)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (graph, got, ref)


def test_descriptor_rules():
    """Dense tiles hold at least the threshold's active rows; 0 streams every
    touched tile; above 1 streams nothing."""
    H, V = 96, 30
    indptr, indices, _ = _random_csc(H, V, 0.15, seed=41)
    d = scat.SparseCategorical(V)
    d.set_meta_parameters(H, indptr, indices)
    cpu = torch.device("cpu")
    all_dense = d.b1_descriptors(cpu, stream_frac=0.0)
    none = d.b1_descriptors(cpu, stream_frac=2.0)
    quarter = d.b1_descriptors(cpu, stream_frac=0.25)
    assert int(none["dense_indptr"][V]) == 0 and int(none["sparse_indptr"][V]) == indices.numel()
    assert int(all_dense["sparse_indptr"][V]) == 0
    tiles = quarter["dense_tiles"][:int(quarter["dense_indptr"][V])]
    assert torch.all(tiles % quarter["tile"] == 0)
    # the all-rows column of a marginalised variable streams every tile
    assert int(quarter["dense_indptr"][V + 1] - quarter["dense_indptr"][V]) == -(-H // quarter["tile"])
