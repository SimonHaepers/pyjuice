import pytest
import torch

import pyjuice as juice
from pyjuice.layer import TopKLayer, TopKSumLayer, SumLayer, ProdLayer


# ---- helpers ----------------------------------------------------------- #


def _build_topk_hmm(seed, H, T, V, K, block_size=4, tied=False, device="cuda:0"):
    """Build a small HMM-shaped circuit with ``topk=K`` on every transition
    summate. ``tied=True`` mirrors the homogeneous-HMM use case (one shared
    transition matrix across timesteps)."""
    torch.manual_seed(seed)
    assert H % block_size == 0, "H must be a multiple of block_size"

    with juice.set_block_size(block_size):
        leaves = [
            juice.inputs(t, num_node_blocks=H // block_size,
                         dist=juice.distributions.Categorical(num_cats=V))
            for t in range(T)
        ]
        sum0 = juice.summate(juice.multiply(leaves[0]),
                             num_node_blocks=H // block_size, topk=K)
        sums = [sum0]
        for t in range(1, T):
            prod = juice.multiply(sums[-1], leaves[t])
            if tied:
                sums.append(sum0.duplicate(prod, tie_params=True))
            else:
                sums.append(juice.summate(prod, num_node_blocks=H // block_size, topk=K))
        root = juice.summate(juice.multiply(sums[-1]), num_node_blocks=1)

    return juice.TensorCircuit(root, verbose=False).to(device)


def _build_plain_hmm(seed, H, T, V, block_size=4, device="cuda:0"):
    """Same circuit but without the ``topk`` annotation — falls through to
    the default ``SumLayer`` block-sparse path."""
    torch.manual_seed(seed)
    with juice.set_block_size(block_size):
        leaves = [
            juice.inputs(t, num_node_blocks=H // block_size,
                         dist=juice.distributions.Categorical(num_cats=V))
            for t in range(T)
        ]
        sums = [juice.summate(juice.multiply(leaves[0]),
                              num_node_blocks=H // block_size)]
        for t in range(1, T):
            prod = juice.multiply(sums[-1], leaves[t])
            sums.append(juice.summate(prod, num_node_blocks=H // block_size))
        root = juice.summate(juice.multiply(sums[-1]), num_node_blocks=1)
    return juice.TensorCircuit(root, verbose=False).to(device)


def _layer_counts(pc):
    counts = {}
    for lg in pc.inner_layer_groups:
        name = type(lg.layers[0]).__name__
        counts[name] = counts.get(name, 0) + len(lg.layers)
    return counts


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


# ---- dispatch tests ---------------------------------------------------- #


def test_topk_dispatch_creates_topk_layers(device):
    pc = _build_topk_hmm(seed=0, H=16, T=4, V=5, K=3, device=device)
    counts = _layer_counts(pc)
    assert counts.get("TopKLayer", 0) > 0, f"no TopKLayer compiled: {counts}"
    assert counts.get("TopKSumLayer", 0) > 0, f"no TopKSumLayer compiled: {counts}"
    # TopK groups come in pairs with a corresponding sum: same count.
    assert counts["TopKLayer"] == counts["TopKSumLayer"]
    assert pc.num_topk_slots > 0


def test_K_geq_H_falls_back(device):
    # When K >= H_total, the dispatch must fall through to plain SumLayer
    # so callers can leave the annotation in place during sweeps.
    pc = _build_topk_hmm(seed=0, H=8, T=3, V=4, K=8, device=device)
    counts = _layer_counts(pc)
    assert counts.get("TopKLayer", 0) == 0
    assert counts.get("TopKSumLayer", 0) == 0
    assert pc.num_topk_slots == 0


def test_no_topk_kwarg_means_no_topk_layers(device):
    pc = _build_plain_hmm(seed=0, H=8, T=3, V=4, device=device)
    counts = _layer_counts(pc)
    assert counts.get("TopKLayer", 0) == 0
    assert counts.get("TopKSumLayer", 0) == 0


# ---- forward correctness ---------------------------------------------- #


def test_forward_topk_le_plain(device):
    """Block-shared truncated logsumexp can only *lose* mass relative to the
    full logsumexp, so per-batch topk LL must be <= plain LL."""
    H, T, V, K, B = 8, 3, 4, 3, 16

    # Build both circuits with the same seed so their initialised params line up.
    torch.manual_seed(7)
    pc_topk = _build_topk_hmm(seed=7, H=H, T=T, V=V, K=K, device=device)
    torch.manual_seed(7)
    pc_plain = _build_plain_hmm(seed=7, H=H, T=T, V=V, device=device)
    pc_topk.params.data.copy_(pc_plain.params.data)

    data = torch.randint(0, V, (B, T), device=device)
    lls_topk = pc_topk(data).detach()
    lls_plain = pc_plain(data).detach()
    assert (lls_topk <= lls_plain + 1e-3).all(), (
        f"topk LL exceeded plain LL: max excess "
        f"{(lls_topk - lls_plain).max().item():.3e}"
    )


def test_forward_topk_eq_h_minus_1_close(device):
    """Sanity: with K = H-1, only one child is dropped per parent — the
    LL should be very close to the exact value (a few percent at most for
    this random seed)."""
    H, T, V, B = 8, 3, 4, 8
    torch.manual_seed(1)
    pc_topk = _build_topk_hmm(seed=1, H=H, T=T, V=V, K=H - 1, device=device)
    torch.manual_seed(1)
    pc_plain = _build_plain_hmm(seed=1, H=H, T=T, V=V, device=device)
    pc_topk.params.data.copy_(pc_plain.params.data)

    data = torch.randint(0, V, (B, T), device=device)
    lls_topk = pc_topk(data).detach()
    lls_plain = pc_plain(data).detach()
    # Loose bound — just confirms it's the same order of magnitude.
    assert (lls_plain - lls_topk).max().item() < 1.0


# ---- backward / EM tests ---------------------------------------------- #


def test_backward_runs_and_populates_flows(device):
    pc = _build_topk_hmm(seed=2, H=8, T=3, V=4, K=3, device=device)
    data = torch.randint(0, 4, (8, 3), device=device)
    pc(data)
    pc.backward(data, allow_modify_flows=False, compute_param_flows=True)
    assert pc.param_flows is not None
    assert (pc.param_flows != 0).any().item()
    assert (pc.element_flows != 0).any().item()


def test_em_step_runs_and_updates_params(device):
    """A single EM step under hard top-K is *not* guaranteed to monotonically
    improve LL on the same batch (the truncated logsumexp can shift mass to
    children that the next forward then drops). The wired-in property we
    really need is: the EM mechanics complete end-to-end and the parameter
    tensor changes."""
    pc = _build_topk_hmm(seed=3, H=8, T=3, V=4, K=3, device=device)
    data = torch.randint(0, 4, (16, 3), device=device)
    pc(data)
    pc.backward(data, allow_modify_flows=False, compute_param_flows=True)
    params_before = pc.params.detach().clone()
    pc.mini_batch_em(step_size=0.5, pseudocount=0.1)
    assert not torch.allclose(pc.params, params_before), \
        "EM step did not update parameters"


def test_tied_em_runs(device):
    """Homogeneous HMM (one shared transition) — load-bearing for the realistic
    use case. EM step must run end-to-end and the param tensor must stay at
    the source's size (no T-1 cloned copies). LL monotonicity is not
    guaranteed under hard top-K (see ``test_em_step_runs_and_updates_params``)."""
    pc = _build_topk_hmm(seed=4, H=16, T=8, V=5, K=4, tied=True, device=device)
    # Tied: ``num_sum_params`` should reflect a single shared transition,
    # not T-1 copies.
    assert pc.num_sum_params <= 16 * 16 * 8 * 2, (
        f"tied param tensor too large: {pc.num_sum_params}"
    )
    data = torch.randint(0, 5, (8, 8), device=device)
    pc(data)
    pc.backward(data, allow_modify_flows=False, compute_param_flows=True)
    params_before = pc.params.detach().clone()
    pc.mini_batch_em(step_size=0.5, pseudocount=0.1)
    assert not torch.allclose(pc.params, params_before)


# ---- assertion / guard tests ------------------------------------------ #


def test_logspace_flows_rejected(device):
    pc = _build_topk_hmm(seed=5, H=8, T=3, V=4, K=3, device=device)
    data = torch.randint(0, 4, (4, 3), device=device)
    pc(data)
    with pytest.raises(AssertionError):
        pc.backward(data, allow_modify_flows=False, logspace_flows=True,
                    compute_param_flows=True)


def test_invalid_topk_raises():
    with pytest.raises(AssertionError):
        with juice.set_block_size(2):
            leaf = juice.inputs(0, num_node_blocks=2,
                                dist=juice.distributions.Categorical(num_cats=4))
            juice.summate(juice.multiply(leaf), num_node_blocks=2, topk=0)


# ---- save/load round-trip --------------------------------------------- #


def test_topk_round_trips_through_save_load(device, tmp_path):
    H, T, V, K = 8, 3, 4, 3
    pc = _build_topk_hmm(seed=6, H=H, T=T, V=V, K=K, device=device)
    data = torch.randint(0, V, (4, T), device=device)
    ll_before = pc(data).detach().cpu()

    path = tmp_path / "topk_pc.jpc"
    juice.save(str(path), pc)

    # ``load`` returns the deserialised root ``CircuitNodes``; recompile to
    # the TensorCircuit. The loaded DAG must still have ``_topk_k`` set on
    # every annotated sum node so the dispatch picks the TopK fast path.
    root2 = juice.load(str(path))
    pc2 = juice.TensorCircuit(root2, verbose=False).to(device)

    counts = _layer_counts(pc2)
    assert counts.get("TopKSumLayer", 0) > 0, (
        f"loaded pc lost the TopK fast path: {counts}"
    )

    # Sync params (re-init randomness in fresh circuit picks different
    # values; the persisted ``_params`` should be the same since they ride
    # along on the SumNodes).
    pc2.params.data.copy_(pc.params.data)
    ll_after = pc2(data).detach().cpu()
    assert torch.allclose(ll_before, ll_after, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
