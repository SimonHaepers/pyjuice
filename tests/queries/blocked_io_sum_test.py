"""
Blocked-emission HMM chain (Chiu & Rush 2020) — parity of the blocked fast
path (BlockedProdLayer → BlockedIOSumLayer → CoBlockedProdLayer → ... →
BlockedInputSumLayer) against

  (a) the *same model* expressed as a SparseCategorical CSC pattern on the
      sparse-IO chain (identical math, generic CSC kernels), and
  (b) a dense masked Categorical + DenseSumLayer build.

Covers LL (B=1 and B>1), element flows, transition + emission param flows,
and one EM step.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size, \
    sparse_multiply, sparse_summate, blocked_multiply, blocked_summate
from pyjuice.layer import (
    BlockedProdLayer, CoBlockedProdLayer, BlockedInputSumLayer, BlockedIOSumLayer,
    SparseIOSumLayer, CoSparseProdLayer, SparseInputSumLayer, DenseSumLayer,
)


def _device():
    return torch.device("cuda:0")


def _random_blocked_model(H, V, M, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = H // M
    # every block owns >= 1 token
    token_block = torch.cat([torch.arange(M), torch.randint(0, M, (V - M,), generator=g)])
    token_block = token_block[torch.randperm(V, generator=g)]
    emit = torch.rand(V, k, generator=g) + 0.05                      # [V, k], token-major
    alpha = torch.rand(H, H, generator=g) + 0.05
    alpha = alpha / alpha.sum(dim=1, keepdim=True)
    gamma = torch.rand(H, generator=g) + 0.05
    gamma = gamma / gamma.sum()
    return token_block, emit, alpha, gamma


def _chain(T, H, bs, ns_input, alpha, gamma, mul, summ, force_plain=False):
    num_node_blocks = H // bs
    kw = {"_force_plain": True} if force_plain else {}
    with set_block_size(block_size=bs):
        ns_sum = None
        curr_zs = ns_input if not force_plain else multiply(ns_input, **kw)
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                # First transition sums over the raw InputNodes; the marker
                # summate variants only apply once a prod child exists.
                ns = summate(curr_zs, num_node_blocks=num_node_blocks, **kw)
                ns.set_params(alpha)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = mul(curr_xs, ns, **kw)
        root = summ(curr_zs, num_node_blocks=1, block_size=1, **kw)
        root.set_params(gamma.unsqueeze(0))
    return root


def build_blocked(T, H, V, bs, token_block, M, emit, alpha, gamma):
    with set_block_size(block_size=bs):
        ns_input = inputs(T - 1, num_node_blocks=H // bs,
                          dist=dists.BlockedCategorical(num_cats=V),
                          token_block=token_block, num_blocks=M)
        ns_input.set_params(emit.reshape(-1))
    root = _chain(T, H, bs, ns_input, alpha, gamma, blocked_multiply, blocked_summate)
    return root, ns_input


def build_sparse(T, H, V, bs, ns_blocked_input, alpha, gamma):
    dist = ns_blocked_input.dist
    csc_indptr, csc_indices, csc_values = dist.to_csc(ns_blocked_input.get_params())
    with set_block_size(block_size=bs):
        ns_input = inputs(T - 1, num_node_blocks=H // bs,
                          dist=dists.SparseCategorical(num_cats=V),
                          csc_indptr=csc_indptr, csc_indices=csc_indices)
        ns_input.set_params(csc_values, normalize=False)
    return _chain(T, H, bs, ns_input, alpha, gamma, sparse_multiply, sparse_summate)


def build_dense(T, H, V, bs, ns_blocked_input, alpha, gamma):
    dense = ns_blocked_input.dist.to_dense(ns_blocked_input.get_params(), fill_value=1e-10)
    with set_block_size(block_size=bs):
        ns_input = inputs(T - 1, num_node_blocks=H // bs, dist=dists.Categorical(num_cats=V))
        ns_input.set_params(dense, normalize=False)
    return _chain(T, H, bs, ns_input, alpha, gamma, multiply, summate, force_plain=True)


def _layers(pc):
    return [layer for lg in pc.inner_layer_groups for layer in lg]


@pytest.mark.parametrize("H,M,bs", [(64, 4, 16), (64, 4, 8), (256, 8, 16), (128, 2, 4)])
@pytest.mark.parametrize("B", [1, 5, 16])
def test_blocked_chain_forward_backward_parity(H, M, bs, B):
    torch.manual_seed(0)
    T, V = 6, 50
    dev = _device()
    token_block, emit, alpha, gamma = _random_blocked_model(H, V, M, seed=H + M + bs)

    root_b, ns_in = build_blocked(T, H, V, bs, token_block, M, emit, alpha, gamma)
    root_s = build_sparse(T, H, V, bs, ns_in, alpha, gamma)
    root_d = build_dense(T, H, V, bs, ns_in, alpha, gamma)

    pc_b = juice.TensorCircuit(root_b, verbose=False).to(dev)
    pc_s = juice.TensorCircuit(root_s, verbose=False).to(dev)
    pc_d = juice.TensorCircuit(root_d, use_dense_sum_layer=True, verbose=False).to(dev)

    lb = _layers(pc_b)
    # T vars → T-1 transitions (all blocked-IO: even the innermost one's child
    # is the 0-dense-child wrapper BlockedProdNodes) + the blocked-in root.
    assert sum(isinstance(l, BlockedIOSumLayer) for l in lb) == T - 1
    assert sum(isinstance(l, CoBlockedProdLayer) for l in lb) == T - 1
    assert sum(isinstance(l, BlockedProdLayer) and not isinstance(l, CoBlockedProdLayer) for l in lb) == 1
    assert sum(isinstance(l, BlockedInputSumLayer) and not isinstance(l, BlockedIOSumLayer) for l in lb) == 1
    assert all(l._skip_scatter for l in lb if isinstance(l, BlockedProdLayer))
    ls = _layers(pc_s)
    assert sum(isinstance(l, SparseIOSumLayer) for l in ls) == T - 1
    assert all(type(l) is DenseSumLayer
               for lg in pc_d.inner_layer_groups if lg.is_sum() for l in lg)

    data = torch.randint(0, V, (B, T), device=dev)

    ll_b = pc_b(data)
    ll_s = pc_s(data)
    ll_d = pc_d(data)
    assert torch.allclose(ll_b, ll_s, atol=1e-4, rtol=1e-5), (ll_b, ll_s)
    # Dense/plain sum kernels switch to tl.dot (TF32) tiles at B>=16: ~2e-3
    # abs on LL≈-23 (blocked ≡ dense bit-exactly at B=1, see the loop test).
    assert torch.allclose(ll_b, ll_d, atol=1e-5, rtol=2e-4), (ll_b, ll_d)

    pc_b.backward(data, compute_param_flows=True)
    pc_s.backward(data, compute_param_flows=True)
    pc_d.backward(data, compute_param_flows=True)

    # Transition param flows share the DenseSumLayer flat layout in all three builds.
    pf_b, pf_s, pf_d = pc_b.param_flows, pc_s.param_flows, pc_d.param_flows
    assert pf_b.shape == pf_s.shape == pf_d.shape
    assert torch.allclose(pf_b, pf_s, atol=1e-4, rtol=1e-4)
    assert torch.allclose(pf_b, pf_d, atol=1e-3, rtol=1e-2)   # TF32 dense path at B>=16
    assert pf_b.abs().sum() > 0

    # Emission param flows: blocked [V, k] slots coincide with the CSC slots.
    epf_b = pc_b.input_layer_group[0].param_flows
    epf_s = pc_s.input_layer_group[0].param_flows
    assert epf_b.shape == epf_s.shape
    assert torch.allclose(epf_b, epf_s, atol=1e-4, rtol=1e-4)
    assert epf_b.abs().sum() > 0

    # Element flows of the first (dense) transition child: the blocked prod's
    # dense child is the sum feeding the first product; both chains scatter
    # its flow into node_flows.
    assert torch.allclose(pc_b.node_flows[pc_b._root_node_range[0]:pc_b._root_node_range[1]],
                          pc_s.node_flows[pc_s._root_node_range[0]:pc_s._root_node_range[1]])

    # One EM step: transitions + emissions (blocked vs sparse: identical math).
    for pc in (pc_b, pc_s, pc_d):
        pc.mini_batch_em(step_size=0.7, pseudocount=0.05)
    assert torch.allclose(pc_b.params, pc_s.params, atol=1e-5, rtol=1e-4)
    # Dense sums over all NB_ch*CBS children (incl. the 1e-10 cells) in a
    # different order → fp32 accumulation noise only.
    assert torch.allclose(pc_b.params, pc_d.params, atol=1e-4, rtol=1e-3)
    ep_b = pc_b.input_layer_group[0].params
    ep_s = pc_s.input_layer_group[0].params
    assert torch.allclose(ep_b, ep_s, atol=1e-5, rtol=1e-4)

    # Post-EM forward still agrees.
    ll_b2, ll_s2 = pc_b(data), pc_s(data)
    assert torch.allclose(ll_b2, ll_s2, atol=1e-4, rtol=1e-5)


def test_blocked_chain_batched_matches_per_sample_loop():
    torch.manual_seed(1)
    T, H, V, M, bs, B = 7, 128, 60, 4, 16, 9
    dev = _device()
    token_block, emit, alpha, gamma = _random_blocked_model(H, V, M, seed=3)
    root_b, ns_in = build_blocked(T, H, V, bs, token_block, M, emit, alpha, gamma)
    pc = juice.TensorCircuit(root_b, verbose=False).to(dev)
    data = torch.randint(0, V, (B, T), device=dev)
    ll = pc(data)
    ll_loop = torch.cat([pc(data[i:i + 1]) for i in range(B)])
    assert torch.allclose(ll, ll_loop, atol=1e-4)

    pc(data)   # buffers must match the batch of the backward call
    pc.backward(data, compute_param_flows=True, flows_memory=0.0)
    pf = pc.param_flows.clone()
    epf = pc.input_layer_group[0].param_flows.clone()
    pf_acc = torch.zeros_like(pf)
    epf_acc = torch.zeros_like(epf)
    for i in range(B):
        pc(data[i:i + 1])
        pc.backward(data[i:i + 1], compute_param_flows=True, flows_memory=0.0)
        pf_acc += pc.param_flows
        epf_acc += pc.input_layer_group[0].param_flows
    assert torch.allclose(pf, pf_acc, atol=1e-4, rtol=1e-4)
    assert torch.allclose(epf, epf_acc, atol=1e-4, rtol=1e-4)


def test_blocked_dense_bridge_mixed_consumer():
    """Blocked prods whose consumers are plain (force_plain) sums fall back to
    the scatter-to-dense bridge (``_skip_scatter=False``) and stay correct,
    forward and backward (B=1)."""
    torch.manual_seed(2)
    T, H, V, M, bs = 4, 64, 30, 4, 16
    dev = _device()
    token_block, emit, alpha, gamma = _random_blocked_model(H, V, M, seed=5)
    with set_block_size(block_size=bs):
        ns_input = inputs(T - 1, num_node_blocks=H // bs,
                          dist=dists.BlockedCategorical(num_cats=V),
                          token_block=token_block, num_blocks=M)
        ns_input.set_params(emit.reshape(-1))
        ns_sum = None
        curr_zs = ns_input
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=H // bs, _force_plain=True)
                ns.set_params(alpha)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = blocked_multiply(curr_xs, ns)
        root = summate(curr_zs, num_node_blocks=1, block_size=1, _force_plain=True)
        root.set_params(gamma.unsqueeze(0))
    root_ref = build_dense(T, H, V, bs, ns_input, alpha, gamma)
    pc = juice.TensorCircuit(root, use_dense_sum_layer=True, verbose=False).to(dev)
    pc_ref = juice.TensorCircuit(root_ref, use_dense_sum_layer=True, verbose=False).to(dev)
    prods = [l for l in _layers(pc) if isinstance(l, BlockedProdLayer)]
    assert len(prods) == T and not any(isinstance(l, CoBlockedProdLayer) for l in prods)  # incl. the input wrapper
    assert not any(l._skip_scatter for l in prods)
    assert all(type(l) is DenseSumLayer for lg in pc.inner_layer_groups if lg.is_sum() for l in lg)

    data = torch.randint(0, V, (1, T), device=dev)
    assert torch.allclose(pc(data), pc_ref(data), atol=1e-4)
    pc.backward(data, compute_param_flows=True, flows_memory=0.0)
    pc_ref.backward(data, compute_param_flows=True, flows_memory=0.0)
    assert torch.allclose(pc.param_flows, pc_ref.param_flows, atol=1e-4, rtol=1e-4)
    epf = pc.input_layer_group[0].param_flows.view(-1, H // M)
    epf_ref = pc_ref.input_layer_group[0].param_flows.view(-1, H, V)
    # emission flows at active cells agree (tied duplicates keep separate
    # accumulators before the EM reduction; compare their totals)
    assert abs(epf.sum().item() - epf_ref.sum().item()) < 1e-3


def test_blocked_hmm_builder_compiles_to_blocked_chain():
    from pyjuice.structures import BlockedHMM, frequency_balanced_partition
    torch.manual_seed(3)
    T, H, V, M = 8, 256, 100, 8
    dev = _device()
    counts = torch.randint(1, 50, (V,))
    token_block = frequency_balanced_partition(counts, M)
    assert token_block.shape == (V,) and token_block.max() < M
    root = BlockedHMM(seq_length=T, num_latents=H, num_emits=V, num_blocks=M, token_block=token_block)
    pc = juice.TensorCircuit(root, verbose=False).to(dev)
    layers = _layers(pc)
    assert sum(isinstance(l, BlockedIOSumLayer) for l in layers) == T - 1
    assert sum(isinstance(l, CoBlockedProdLayer) for l in layers) == T - 1
    data = torch.randint(0, V, (4, T), device=dev)
    ll = pc(data)
    assert torch.isfinite(ll).all()
    pc.backward(data, compute_param_flows=True, flows_memory=0.0)
    pc.mini_batch_em(step_size=1.0, pseudocount=0.01)
    assert torch.isfinite(pc(data)).all()
