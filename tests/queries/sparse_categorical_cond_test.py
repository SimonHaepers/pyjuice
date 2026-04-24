"""Correctness test: the ``conditional`` query on a ``SparseCategorical`` input
layer must match the reference ``Categorical`` path on equivalent circuits.

We build the same HMM twice: once with ``Categorical`` leaves (reference) and
once with ``SparseCategorical`` leaves using a **dense CSC pattern** — every
``(latent, category)`` pair marked active — so the two paths encode the same
distribution bit-for-bit up to storage layout. Parameters are copied across
after compilation (transcribed from the per-latent row-major layout used by
``Categorical`` into the column-major CSC layout used by ``SparseCategorical``),
and inner sum-layer params are shared verbatim.

``juice.queries.conditional(...)`` should produce the same posterior per
target variable on both circuits, up to atomic-add rounding from the CSC
backward scatter.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size


def _dense_csc_pattern(H: int, V: int):
    """CSC pattern covering every ``(row, col)`` in an H x V emission matrix,
    in CSC order (each column lists all H rows in ascending order)."""
    csc_indptr = torch.arange(0, V * H + 1, H, dtype=torch.long)
    csc_indices = torch.arange(H, dtype=torch.long).repeat(V)
    return csc_indptr, csc_indices


def _build_hmm(T: int, H: int, V: int, bs: int, homogeneous: bool,
               dist_cls, **dist_kwargs):
    """Standard HMM DAG. Uses ``_force_plain=True`` on every ``multiply`` /
    ``summate`` so the compiler always picks plain :class:`ProdLayer` /
    :class:`SumLayer`, even when ``dist_cls`` is :class:`SparseCategorical`
    (which normally auto-opts into the sparse inner-layer fast path). The
    conditional query needs standard dense-propagation flows through the
    inner layers; the SparseProdLayer / SparseInputSumLayer fast path is a
    separate, inference-only B=1 build that doesn't populate input
    ``node_flows``.

    ``SumNodes._standardize_chs`` unconditionally wraps a bare
    ``SparseCategorical`` input child in a ``SparseProdNodes``, so we
    pre-wrap the innermost emission with a plain ``multiply(..., _force_plain=True)``
    before handing it to the first ``summate`` — otherwise the compiler
    would still pick ``SparseProdLayer`` at the deepest level regardless of
    the ``_force_plain`` flags higher up.
    """
    num_node_blocks = H // bs
    with set_block_size(block_size=bs):
        ns_input = inputs(
            T - 1, num_node_blocks=num_node_blocks,
            dist=dist_cls(num_cats=V),
            **dist_kwargs,
        )
        ns_sum = None
        curr_zs = multiply(ns_input, _force_plain=True)
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=homogeneous)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=num_node_blocks, _force_plain=True)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=homogeneous)
            curr_zs = multiply(curr_xs, ns, _force_plain=True)
        return summate(curr_zs, num_node_blocks=1, block_size=1, _force_plain=True)


def _build_pair(T: int, H: int, V: int, bs: int, homogeneous: bool,
                device: torch.device, seed: int):
    """Build Categorical and SparseCategorical HMMs, then transcribe the
    reference input-layer params into CSC order so both circuits encode the
    same joint distribution."""
    torch.manual_seed(seed)
    root_ref = _build_hmm(T, H, V, bs, homogeneous, dists.Categorical)
    pc_ref = juice.TensorCircuit(root_ref, verbose=False).to(device)

    csc_indptr, csc_indices = _dense_csc_pattern(H, V)
    torch.manual_seed(seed)
    root_sparse = _build_hmm(
        T, H, V, bs, homogeneous, dists.SparseCategorical,
        csc_indptr=csc_indptr, csc_indices=csc_indices,
    )
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)

    # Inner sum-layer params share identical layout across the two builds
    # (same multiply/summate topology), so we can copy verbatim.
    pc_sparse.params.data.copy_(pc_ref.params.data)

    # Transcribe input-layer params per source ns:
    #   ref:    [H, V] row-major   -> params[n*V + c] = P(c | n)
    #   sparse: [V, H] column-major -> params[c*H + n] = P(c | n)   (dense CSC)
    layer_ref = pc_ref.input_layer_group[0]
    layer_sparse = pc_sparse.input_layer_group[0]
    assert layer_ref.num_parameters == layer_sparse.num_parameters, (
        f"dense CSC pattern should match Categorical param count "
        f"({layer_ref.num_parameters} vs {layer_sparse.num_parameters})"
    )
    sources_ref = [ns for ns in layer_ref.nodes if not ns.is_tied()]
    sources_sparse = [ns for ns in layer_sparse.nodes if not ns.is_tied()]
    assert len(sources_ref) == len(sources_sparse)
    for ns_r, ns_s in zip(sources_ref, sources_sparse):
        lo_r, hi_r = ns_r._param_range
        lo_s, hi_s = ns_s._param_range
        p_cat = layer_ref.params.data[lo_r:hi_r].view(H, V)
        layer_sparse.params.data[lo_s:hi_s] = p_cat.t().contiguous().view(-1)

    return pc_ref, pc_sparse


@pytest.mark.parametrize("homogeneous", [True, False])
@pytest.mark.parametrize("target_subset", [None, "half", "all"])
def test_sparse_categorical_conditional_matches_reference(homogeneous, target_subset):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    T, H, V, bs = 8, 32, 16, 8
    B = 4

    pc_ref, pc_sparse = _build_pair(T, H, V, bs, homogeneous, device, seed=42)

    assert pc_ref.input_layer_group[0].dist_signature == "Categorical"
    assert pc_sparse.input_layer_group[0].dist_signature == "SparseCategorical"

    data = torch.randint(0, V, (B, T), device=device)

    if target_subset is None:
        target_vars = None
    elif target_subset == "all":
        target_vars = list(range(T))
    else:
        target_vars = [0, 2, 4, 6]

    out_ref = juice.queries.conditional(pc_ref, data=data, target_vars=target_vars)
    out_sparse = juice.queries.conditional(pc_sparse, data=data, target_vars=target_vars)

    assert out_ref.shape == out_sparse.shape
    torch.testing.assert_close(out_sparse, out_ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    for homo in [True, False]:
        for sub in [None, "half", "all"]:
            test_sparse_categorical_conditional_matches_reference(homo, sub)
            print(f"ok: homogeneous={homo} target={sub}")
