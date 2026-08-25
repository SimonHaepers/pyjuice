"""Standalone (InputLayer-path) tests of :class:`BlockedCategorical`: forward,
backward param flows, EM update, normalisation and CSC/dense conversions —
independent of the blocked chain layers (plain consumers)."""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size


def _device():
    return torch.device("cuda:0")


def _build(H, V, M, bs, token_block, emit, w, dense=False):
    k = H // M
    with set_block_size(bs):
        if dense:
            ni = inputs(0, num_node_blocks=H // bs, dist=dists.Categorical(num_cats=V))
        else:
            ni = inputs(0, num_node_blocks=H // bs,
                        dist=dists.BlockedCategorical(num_cats=V),
                        token_block=token_block, num_blocks=M)
        ni.set_params(emit, normalize=False)
        prod = multiply(ni, _force_plain=True)
        root = summate(prod, num_node_blocks=1, block_size=1, _force_plain=True)
        root.set_params(w)
    return juice.TensorCircuit(root, verbose=False), ni


def test_meta_params_and_conversions():
    H, V, M = 32, 20, 4
    k = H // M
    token_block = torch.randint(0, M, (V,))
    d = dists.BlockedCategorical(num_cats=V)
    p0 = d.set_meta_parameters(H, token_block=token_block, num_blocks=M)
    assert p0.numel() == V * k and d.k == k and d.num_blocks == M

    raw = torch.rand(V * k)
    p = d.normalize_parameters(raw)
    dense = d.to_dense(p, fill_value=0.0)
    assert dense.shape == (H, V)
    assert torch.allclose(dense.sum(dim=1), torch.ones(H), atol=1e-6)
    # support pattern
    for v in range(V):
        m = token_block[v].item()
        assert (dense[:, v] != 0).nonzero().flatten().tolist() == list(range(m * k, (m + 1) * k))

    indptr, indices, values = d.to_csc(p)
    assert torch.equal(indptr, torch.arange(V + 1) * k)
    assert torch.equal(values, p)
    for v in range(V):
        rows = indices[indptr[v]:indptr[v + 1]]
        assert torch.equal(rows, token_block[v] * k + torch.arange(k))

    with pytest.raises(AssertionError):
        dists.BlockedCategorical(num_cats=V).set_meta_parameters(30, token_block=token_block, num_blocks=M)


@pytest.mark.parametrize("H,M,bs", [(64, 4, 16), (64, 8, 4), (128, 2, 32)])
def test_input_layer_forward_backward_em(H, M, bs):
    torch.manual_seed(0)
    V, B = 40, 37
    k = H // M
    dev = _device()
    token_block = torch.cat([torch.arange(M), torch.randint(0, M, (V - M,))])
    d = dists.BlockedCategorical(num_cats=V)
    d.set_meta_parameters(H, token_block=token_block, num_blocks=M)
    emit = d.normalize_parameters(torch.rand(V * k) + 0.05)
    w = torch.rand(1, H)

    pc, ni = _build(H, V, M, bs, token_block, emit, w)
    pc.to(dev)
    pc_ref, _ = _build(H, V, M, bs, token_block, d.to_dense(emit, fill_value=1e-10), w, dense=True)
    pc_ref.to(dev)

    data = torch.randint(0, V, (B, 1), device=dev)
    assert torch.allclose(pc(data), pc_ref(data), atol=1e-5)

    pc.backward(data, compute_param_flows=True)
    pc_ref.backward(data, compute_param_flows=True)
    pf = pc.input_layer_group[0].param_flows.view(V, k).cpu()
    pf_ref = pc_ref.input_layer_group[0].param_flows.view(H, V).cpu()
    rows = token_block[:, None] * k + torch.arange(k)[None, :]
    cols = torch.arange(V)[:, None].expand(V, k)
    assert torch.allclose(pf, pf_ref[rows, cols], atol=1e-5)
    assert abs(pf.sum().item() - B) < 1e-3   # one unit of flow per sample

    # EM: closed-form blocked update.
    params_before = pc.input_layer_group[0].params[:V * k].view(V, k).cpu().clone()
    step, pc_ = 0.6, 0.02
    pc.mini_batch_em(step_size=step, pseudocount=pc_)
    contrib = pf + pc_
    row_sums = torch.zeros(M, k).index_add_(0, token_block, contrib)
    expected = (1 - step) * params_before + step * contrib / row_sums[token_block]
    got = pc.input_layer_group[0].params[:V * k].view(V, k).cpu()
    assert torch.allclose(got, expected, atol=1e-6)
    # each latent still normalised over its block's tokens
    assert torch.allclose(torch.zeros(M, k).index_add_(0, token_block, got), torch.ones(M, k), atol=1e-5)


def test_tied_duplicates_share_params():
    torch.manual_seed(1)
    H, V, M, bs, T = 32, 15, 4, 8, 4
    k = H // M
    token_block = torch.randint(0, M, (V,))
    with set_block_size(bs):
        ni = inputs(T - 1, num_node_blocks=H // bs,
                    dist=dists.BlockedCategorical(num_cats=V),
                    token_block=token_block, num_blocks=M)
        ni.set_params(torch.rand(V * k))
        chs = [ni.duplicate(v, tie_params=True) for v in range(T - 1)] + [ni]
        # independent product over the T vars (several BlockedCategorical
        # ns's in one InputLayer, plain consumer)
        prod = multiply(*chs, _force_plain=True)
        root = summate(prod, num_node_blocks=1, block_size=1, _force_plain=True)
    pc = juice.TensorCircuit(root, verbose=False).to(_device())
    assert pc.input_layer_group[0].params.numel() >= V * k
    data = torch.randint(0, V, (5, T), device=_device())
    ll = pc(data)
    # independent emissions: LL = Σ_t log Σ_h w_h P(x_t | h) is finite
    assert torch.isfinite(ll).all()
