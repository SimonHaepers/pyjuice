import pytest
import torch

import pyjuice as juice
from pyjuice.layer import DenseSumLayer, SumLayer


def _build_pair(seed, K, T, V, homogeneous=False, device="cuda:0"):
    """Compile the same HMM twice (sparse + dense) with identical params."""
    torch.manual_seed(seed)
    ns_a = juice.structures.HMM(seq_length=T, num_latents=K, num_emits=V,
                                homogeneous=homogeneous)
    pc_sparse = juice.TensorCircuit(ns_a, use_dense_sum_layer=False, verbose=False).to(device)

    torch.manual_seed(seed)
    ns_b = juice.structures.HMM(seq_length=T, num_latents=K, num_emits=V,
                                homogeneous=homogeneous)
    pc_dense = juice.TensorCircuit(ns_b, use_dense_sum_layer=True, verbose=False).to(device)
    pc_dense.params.data.copy_(pc_sparse.params.data)
    return pc_sparse, pc_dense


def _counts(pc):
    dense = 0
    sparse = 0
    for lg in pc.inner_layer_groups:
        for l in lg.layers:
            if isinstance(l, DenseSumLayer):
                dense += 1
            elif type(l) is SumLayer:
                sparse += 1
    return dense, sparse


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


def test_dense_layer_is_used(device):
    _, pc_dense = _build_pair(seed=0, K=64, T=4, V=8, device=device)
    dense_n, sparse_n = _counts(pc_dense)
    assert dense_n > 0, f"Expected DenseSumLayer to be used; got dense={dense_n} sparse={sparse_n}"


def test_flag_off_falls_back(device):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length=4, num_latents=64, num_emits=8, homogeneous=False)
    pc = juice.TensorCircuit(ns, use_dense_sum_layer=False, verbose=False).to(device)
    dense_n, _sparse_n = _counts(pc)
    assert dense_n == 0


def test_tied_params_use_dense_path(device):
    """Homogeneous (tied) HMMs must land on ``DenseSumLayer`` too — it's the
    dispatch target that makes realistic H×H transitions affordable (a single
    shared param range instead of T-1 CPU-cloned copies). ``DenseSumLayer``
    reuses the tied source's ``_param_range`` so the flat params tensor
    stays at the source's size."""
    pc_plain, pc_dense = _build_pair(
        seed=3, K=64, T=6, V=8, homogeneous=True, device=device,
    )
    dense_n, _ = _counts(pc_dense)
    assert dense_n > 0, "homogeneous HMM should still dispatch to DenseSumLayer"

    tied_seen = False
    for lg in pc_dense.inner_layer_groups:
        for l in lg.layers:
            if isinstance(l, DenseSumLayer):
                for n in l.nodes:
                    if n.is_tied():
                        tied_seen = True
    assert tied_seen, "expected tied duplicates on DenseSumLayer for a homogeneous HMM"

    # Forward parity with the plain SumLayer baseline — tied alias logic
    # must produce identical LLs.
    torch.manual_seed(11)
    data = torch.randint(0, 8, (1, 6), device=device)
    ll_plain = pc_plain(data).detach().cpu()
    ll_dense = pc_dense(data).detach().cpu()
    assert torch.allclose(ll_plain, ll_dense, atol=1e-5, rtol=1e-5), \
        f"tied DenseSumLayer LL mismatch: {(ll_plain - ll_dense).abs().max().item():.3e}"


@pytest.mark.parametrize("K", [32, 256, 1024])
@pytest.mark.parametrize("T", [4, 8])
def test_forward_matches_sparse(device, K, T):
    B, V = 16, min(K // 2, 32)
    pc_sparse, pc_dense = _build_pair(seed=K * T, K=K, T=T, V=V, device=device)
    data = torch.randint(0, V, (B, T), device=device)

    lls_sparse = pc_sparse(data)
    lls_dense = pc_dense(data)
    diff = (lls_sparse - lls_dense).abs().max().item()
    assert diff < 1e-3, f"forward LL diff={diff}"


@pytest.mark.parametrize("propagation_alg,kwargs", [
    ("LL", {}),
    ("MPE", {}),
    ("GeneralLL", {"alpha": 0.7}),
    ("GeneralLL", {"alpha": 1.3}),
])
def test_forward_propagation_modes(device, propagation_alg, kwargs):
    K, T, V, B = 64, 5, 8, 8
    pc_sparse, pc_dense = _build_pair(seed=17, K=K, T=T, V=V, device=device)
    pc_sparse.set_propagation_alg(propagation_alg, **kwargs)
    pc_dense.set_propagation_alg(propagation_alg, **kwargs)

    data = torch.randint(0, V, (B, T), device=device)
    lls_sparse = pc_sparse(data)
    lls_dense = pc_dense(data)
    diff = (lls_sparse - lls_dense).abs().max().item()
    assert diff < 1e-3, f"{propagation_alg} {kwargs} diff={diff}"


@pytest.mark.parametrize("propagation_alg,kwargs", [
    ("LL", {}),
    ("MPE", {}),
    ("GeneralLL", {"alpha": 0.7}),
])
@pytest.mark.parametrize("B", [8, 32])
def test_conditional_query_matches_sparse(device, propagation_alg, kwargs, B):
    # B=32 is a regression guard for the Triton middle-axis-reduction
    # miscompile: the bs=1 root's ele kernel used to emit 8x element flows
    # at BLOCK_B >= 16 (normalisation hid it in conditionals, but raw flows
    # and param flows were wrong).
    K, T, V = 64, 5, 8
    pc_sparse, pc_dense = _build_pair(seed=99, K=K, T=T, V=V, device=device)
    pc_sparse.set_propagation_alg(propagation_alg, **kwargs)
    pc_dense.set_propagation_alg(propagation_alg, **kwargs)

    data = torch.randint(0, V, (B, T), device=device)
    cond_sparse = juice.queries.conditional(pc_sparse, data, target_vars=[1, 3])
    cond_dense = juice.queries.conditional(pc_dense, data, target_vars=[1, 3])
    diff = (cond_sparse - cond_dense).abs().max().item()
    assert diff < 1e-4, f"{propagation_alg} {kwargs} conditional diff={diff}"


def _matched_sum_nodes(pc_a, pc_b):
    """Pairs of corresponding (non-tied) sum nodes from two same-structure
    DAGs, in traversal order."""
    pairs = []
    for ns_a, ns_b in zip(pc_a.root_ns, pc_b.root_ns):
        assert type(ns_a) is type(ns_b)
        if ns_a.is_sum() and not ns_a.is_tied():
            pairs.append((ns_a, ns_b))
    return pairs


# NOTE: plain-referenced rows keep B a power of 2 — the plain SumLayer's own
# B>1 pflow path writes at wrong offsets for non-pow2 B at bs>=8
# (pre-existing bug, present on main). Non-pow2 B coverage for the dense
# path lives in test_param_flows_batched_matches_loop, which needs no plain
# reference.
@pytest.mark.parametrize("B", [1, 8, 16, 128])
@pytest.mark.parametrize("allow_modify_flows", [True, False])
def test_param_flows_match_plain(device, B, allow_modify_flows):
    K, T, V = 64, 5, 8
    pc_plain, pc_dense = _build_pair(seed=7, K=K, T=T, V=V, device=device)

    torch.manual_seed(21)
    data = torch.randint(0, V, (B, T), device=device)

    pc_plain(data)
    pc_dense(data)
    pc_plain.backward(data, compute_param_flows=True, flows_memory=0.0,
                      allow_modify_flows=allow_modify_flows)
    pc_dense.backward(data, compute_param_flows=True, flows_memory=0.0,
                      allow_modify_flows=allow_modify_flows)

    for ns_p, ns_d in _matched_sum_nodes(pc_plain, pc_dense):
        ns_p.update_param_flows(pc_plain.param_flows, origin_ns_only=True)
        ns_d.update_param_flows(pc_dense.param_flows, origin_ns_only=True)
        pf_p = ns_p._param_flows.detach()
        pf_d = ns_d._param_flows.detach()
        abs_diff = (pf_d - pf_p).abs().max().item()
        rel_scale = max(pf_p.abs().max().item(), 1e-6)
        assert abs_diff / rel_scale < 5e-3, (
            f"param_flow mismatch (B={B}, modify={allow_modify_flows}, "
            f"scope={ns_p.scope}): max abs diff = {abs_diff:.3e}, "
            f"scale = {rel_scale:.3e}"
        )


@pytest.mark.parametrize("propagation_alg,kwargs", [
    ("MPE", {}),
    ("GeneralLL", {"alpha": 0.7}),
])
def test_param_flows_other_algs_match_plain(device, propagation_alg, kwargs):
    K, T, V, B = 64, 5, 8, 8
    pc_plain, pc_dense = _build_pair(seed=23, K=K, T=T, V=V, device=device)

    torch.manual_seed(31)
    data = torch.randint(0, V, (B, T), device=device)

    pc_plain(data, propagation_alg=propagation_alg, **kwargs)
    pc_dense(data, propagation_alg=propagation_alg, **kwargs)
    pc_plain.backward(data, compute_param_flows=True, flows_memory=0.0,
                      propagation_alg=propagation_alg, **kwargs)
    pc_dense.backward(data, compute_param_flows=True, flows_memory=0.0,
                      propagation_alg=propagation_alg, **kwargs)

    for ns_p, ns_d in _matched_sum_nodes(pc_plain, pc_dense):
        ns_p.update_param_flows(pc_plain.param_flows, origin_ns_only=True)
        ns_d.update_param_flows(pc_dense.param_flows, origin_ns_only=True)
        pf_p = ns_p._param_flows.detach()
        pf_d = ns_d._param_flows.detach()
        abs_diff = (pf_d - pf_p).abs().max().item()
        rel_scale = max(pf_p.abs().max().item(), 1e-6)
        assert abs_diff / rel_scale < 5e-3, (
            f"{propagation_alg} {kwargs} param_flow mismatch: "
            f"max abs diff = {abs_diff:.3e}, scale = {rel_scale:.3e}"
        )


@pytest.mark.parametrize("B", [7, 13])
def test_param_flows_batched_matches_loop(device, B):
    """One batched backward must accumulate the same param flows as B
    single-sample backward calls summed on host. Covers non-power-of-2 B
    (batch-tile tail masking) without relying on the plain reference."""
    K, T, V = 64, 5, 8
    _, pc = _build_pair(seed=13, K=K, T=T, V=V, device=device)

    torch.manual_seed(41)
    data = torch.randint(0, V, (B, T), device=device)

    sum_nodes = [ns for ns in pc.root_ns if ns.is_sum() and not ns.is_tied()]

    pf_loop = {}
    for b in range(B):
        pc(data[b:b + 1])
        pc.backward(data[b:b + 1], compute_param_flows=True, flows_memory=0.0)
        for ns in sum_nodes:
            ns.update_param_flows(pc.param_flows, origin_ns_only=True)
            pf_b = ns._param_flows.detach().clone()
            key = id(ns)
            pf_loop[key] = pf_b if key not in pf_loop else pf_loop[key] + pf_b

    pc(data)
    pc.backward(data, compute_param_flows=True, flows_memory=0.0)
    for ns in sum_nodes:
        ns.update_param_flows(pc.param_flows, origin_ns_only=True)
        pf_batched = ns._param_flows.detach()
        # rtol accommodates TF32: the batched pass takes the tl.dot path
        # (TILE_B=16 at B=13) while the B=1 loop passes use the fp32
        # non-dot path.
        torch.testing.assert_close(
            pf_batched, pf_loop[id(ns)], rtol=5e-3, atol=1e-4,
        )


@pytest.mark.parametrize("homogeneous", [False, True])
def test_em_step_matches_plain(device, homogeneous):
    """One full EM update lands on the same params as the plain path. The
    homogeneous case exercises tied duplicates accumulating into the
    source's aliased pflow region (vs. the plain path's separate parflow
    blocks + compute_cum_par_flows fusing)."""
    K, T, V, B = 64, 6, 8, 8
    pc_plain, pc_dense = _build_pair(
        seed=37, K=K, T=T, V=V, homogeneous=homogeneous, device=device,
    )

    torch.manual_seed(43)
    data = torch.randint(0, V, (B, T), device=device)

    pc_plain(data)
    pc_dense(data)
    pc_plain.backward(data, compute_param_flows=True, flows_memory=0.0)
    pc_dense.backward(data, compute_param_flows=True, flows_memory=0.0)

    pc_plain.mini_batch_em(step_size=1.0, pseudocount=0.01)
    pc_dense.mini_batch_em(step_size=1.0, pseudocount=0.01)

    diff = (pc_plain.params - pc_dense.params).abs().max().item()
    assert diff < 5e-3, (
        f"post-EM params mismatch (homogeneous={homogeneous}): "
        f"max abs diff = {diff:.3e}"
    )

    torch.manual_seed(44)
    test_seq = torch.randint(0, V, (B, T), device=device)
    ll_p = pc_plain(test_seq).detach()
    ll_d = pc_dense(test_seq).detach()
    ll_diff = (ll_d - ll_p).abs().max().item()
    assert ll_diff < 0.1, f"post-EM LL mismatch: {ll_diff:.3e}"


def test_em_training_improves_ll(device):
    """A few EM iterations on the dense path alone must increase the
    training LL (end-to-end sanity, incl. the homogeneous/tied case)."""
    K, T, V, B = 64, 6, 8, 32
    torch.manual_seed(51)
    ns = juice.structures.HMM(seq_length=T, num_latents=K, num_emits=V,
                              homogeneous=True)
    pc = juice.TensorCircuit(ns, use_dense_sum_layer=True, verbose=False).to(device)

    torch.manual_seed(52)
    data = torch.randint(0, V, (B, T), device=device)

    ll_before = pc(data).mean().item()
    for _ in range(5):
        pc(data)
        pc.backward(data, compute_param_flows=True, flows_memory=0.0)
        pc.mini_batch_em(step_size=1.0, pseudocount=0.01)
    ll_after = pc(data).mean().item()

    assert ll_after > ll_before + 0.1, (
        f"EM on the dense path did not improve LL: "
        f"{ll_before:.4f} -> {ll_after:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
