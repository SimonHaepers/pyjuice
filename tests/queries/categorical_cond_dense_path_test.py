"""Correctness test: ``DenseCategoricalInputLayer`` conditional backward must
match the base-``InputLayer`` atomic-add kernel output on HMM circuits.

The class is opt-in via ``TensorCircuit(..., use_dense_categorical_input_layer=True)``.
We build the same HMM twice with the same seed (so parameters match), once with
each InputLayer implementation, and compare ``juice.queries.conditional(...)``.
"""
import torch
import pytest

import pyjuice as juice
from pyjuice.layer import DenseCategoricalInputLayer, InputLayer


def _build_pair(T, K, V, homogeneous, device, seed):
    torch.manual_seed(seed)
    ns_a = juice.structures.HMM(
        seq_length=T, num_latents=K, num_emits=V, homogeneous=homogeneous,
    )
    pc_ref = juice.TensorCircuit(ns_a, verbose=False).to(device)

    torch.manual_seed(seed)
    ns_b = juice.structures.HMM(
        seq_length=T, num_latents=K, num_emits=V, homogeneous=homogeneous,
    )
    pc_dense = juice.TensorCircuit(
        ns_b, use_dense_categorical_input_layer=True, verbose=False,
    ).to(device)
    pc_dense.params.data.copy_(pc_ref.params.data)

    return pc_ref, pc_dense


@pytest.mark.parametrize("homogeneous", [True, False])
@pytest.mark.parametrize("target_subset", [None, "half", "all"])
def test_dense_categorical_input_layer_matches_base(homogeneous, target_subset):
    device = torch.device("cuda:0")

    T, K, V = 8, 32, 16
    B = 4

    pc_ref, pc_dense = _build_pair(T, K, V, homogeneous, device, seed=42)
    data = torch.randint(0, V, (B, T), device=device)

    # Layer-type invariants
    assert isinstance(pc_ref.input_layer_group[0], InputLayer)
    assert not isinstance(pc_ref.input_layer_group[0], DenseCategoricalInputLayer)
    assert isinstance(pc_dense.input_layer_group[0], DenseCategoricalInputLayer)

    if target_subset is None:
        target_vars = None
    elif target_subset == "all":
        target_vars = list(range(T))
    else:
        target_vars = [0, 2, 4, 6]

    out_ref   = juice.queries.conditional(pc_ref,   data=data, target_vars=target_vars)
    out_dense = juice.queries.conditional(pc_dense, data=data, target_vars=target_vars)

    assert out_ref.shape == out_dense.shape
    torch.testing.assert_close(out_dense, out_ref, rtol=1e-4, atol=1e-5)


def test_dense_categorical_input_layer_rejects_heterogeneous_num_cats():
    """The class asserts uniform ``num_cats`` at construction. Building a PC
    with two Categorical InputNodes of different ``num_cats`` and requesting
    ``use_dense_categorical_input_layer=True`` must raise.
    """
    import pyjuice.nodes.distributions as dists
    from pyjuice.nodes import inputs, multiply, summate

    ni0 = inputs(0, num_nodes=2, dist=dists.Categorical(num_cats=2))
    ni1 = inputs(1, num_nodes=2, dist=dists.Categorical(num_cats=4))
    m   = multiply(ni0, ni1)
    n   = summate(m, num_nodes=1)

    with pytest.raises(ValueError, match="uniform num_cats"):
        juice.TensorCircuit(n, use_dense_categorical_input_layer=True, verbose=False)


if __name__ == "__main__":
    for homo in [True, False]:
        for sub in [None, "half", "all"]:
            test_dense_categorical_input_layer_matches_base(homo, sub)
            print(f"ok: homogeneous={homo} target={sub}")
    test_dense_categorical_input_layer_rejects_heterogeneous_num_cats()
    print("ok: rejects heterogeneous num_cats")
