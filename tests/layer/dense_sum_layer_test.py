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


def test_tied_params_stay_on_sparse_path(device):
    _, pc = _build_pair(seed=3, K=64, T=6, V=8, homogeneous=True, device=device)
    for lg in pc.inner_layer_groups:
        for l in lg.layers:
            if isinstance(l, DenseSumLayer):
                for n in l.nodes:
                    assert not n.is_tied(), (
                        f"tied node {n} ended up on DenseSumLayer"
                    )


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
def test_conditional_query_matches_sparse(device, propagation_alg, kwargs):
    K, T, V, B = 64, 5, 8, 8
    pc_sparse, pc_dense = _build_pair(seed=99, K=K, T=T, V=V, device=device)
    pc_sparse.set_propagation_alg(propagation_alg, **kwargs)
    pc_dense.set_propagation_alg(propagation_alg, **kwargs)

    data = torch.randint(0, V, (B, T), device=device)
    cond_sparse = juice.queries.conditional(pc_sparse, data, target_vars=[1, 3])
    cond_dense = juice.queries.conditional(pc_dense, data, target_vars=[1, 3])
    diff = (cond_sparse - cond_dense).abs().max().item()
    assert diff < 1e-4, f"{propagation_alg} {kwargs} conditional diff={diff}"


def test_learning_raises(device):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length=4, num_latents=64, num_emits=8, homogeneous=False)
    pc = juice.TensorCircuit(ns, use_dense_sum_layer=True, verbose=False).to(device)
    data = torch.randint(0, 8, (8, 4), device=device)
    pc(data)
    with pytest.raises(NotImplementedError):
        pc.backward(data, compute_param_flows=True, allow_modify_flows=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
