"""Correctness test for the **generative** ``conditional`` query: given a
sequence with a single position ``t*`` masked out, ``p(x_{t*} | x_{!=t*})``
must agree across the reference ``Categorical`` pipeline and the
``SparseCategorical`` pipeline (with a fully-dense CSC pattern + transcribed
params, so the two circuits encode the same distribution bit-for-bit up to
storage layout).

Mirrors the structure of ``sparse_categorical_cond_test.py`` (the smoothing
correctness test). The only differences are (1) we pass ``missing_mask`` to
mark a single position unobserved, and (2) we ask for ``target_vars=[t*]``
so only that one column is returned.

The third test below also asserts a self-consistency property: the
``[:, t*, :]`` slice of a full-output (``target_vars=None``) call with the
same ``missing_mask`` must equal the ``target_vars=[t*]`` output. This
guards against the partial-eval block-id bug from
``project_partial_eval_block_bug``: if a future change rewires the dispatch
into the buggy atomic-add fallback, this assertion will trip.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size


def _dense_csc_pattern(H: int, V: int):
    """CSC pattern covering every ``(row, col)`` slot — one full column per
    category, rows in ascending order."""
    csc_indptr = torch.arange(0, V * H + 1, H, dtype=torch.long)
    csc_indices = torch.arange(H, dtype=torch.long).repeat(V)
    return csc_indptr, csc_indices


def _build_hmm(T: int, H: int, V: int, bs: int, homogeneous: bool,
               dist_cls, **dist_kwargs):
    """Standard HMM DAG with ``_force_plain=True`` on every multiply/summate
    so the conditional query goes through dense-propagation inner layers
    even with a ``SparseCategorical`` input. Same shape as the smoothing
    correctness test — see that file for the full rationale."""
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
    """Build matched Categorical and SparseCategorical HMMs and transcribe
    the input-layer params so both circuits encode the same joint."""
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

    pc_sparse.params.data.copy_(pc_ref.params.data)

    layer_ref = pc_ref.input_layer_group[0]
    layer_sparse = pc_sparse.input_layer_group[0]
    assert layer_ref.num_parameters == layer_sparse.num_parameters
    sources_ref = [ns for ns in layer_ref.nodes if not ns.is_tied()]
    sources_sparse = [ns for ns in layer_sparse.nodes if not ns.is_tied()]
    assert len(sources_ref) == len(sources_sparse)
    for ns_r, ns_s in zip(sources_ref, sources_sparse):
        lo_r, hi_r = ns_r._param_range
        lo_s, hi_s = ns_s._param_range
        # ref: [H, V] row-major  ->  sparse (dense CSC): [V, H] col-major
        p_cat = layer_ref.params.data[lo_r:hi_r].view(H, V)
        layer_sparse.params.data[lo_s:hi_s] = p_cat.t().contiguous().view(-1)

    return pc_ref, pc_sparse


@pytest.mark.parametrize("homogeneous", [True, False])
@pytest.mark.parametrize("target_t", [0, 4, 7])
def test_sparse_categorical_generative_matches_reference(homogeneous, target_t):
    """Dense vs sparse: the generative posterior at one masked-out position
    must agree across the two pipelines on a fully-dense CSC pattern."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    T, H, V, bs = 8, 32, 16, 8
    B = 4

    pc_ref, pc_sparse = _build_pair(T, H, V, bs, homogeneous, device, seed=42)

    assert pc_ref.input_layer_group[0].dist_signature == "Categorical"
    assert pc_sparse.input_layer_group[0].dist_signature == "SparseCategorical"

    data = torch.randint(0, V, (B, T), device=device)
    missing_mask = torch.zeros(T, dtype=torch.bool, device=device)
    missing_mask[target_t] = True

    out_ref = juice.queries.conditional(
        pc_ref, data=data, missing_mask=missing_mask, target_vars=[target_t],
    )
    out_sparse = juice.queries.conditional(
        pc_sparse, data=data, missing_mask=missing_mask, target_vars=[target_t],
    )

    assert out_ref.shape == (B, 1, V)
    assert out_sparse.shape == (B, 1, V)
    torch.testing.assert_close(out_sparse, out_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("target_t", [0, 4, 7])
def test_generative_posterior_is_a_distribution(target_t):
    """The returned posterior must sum to 1 over V on each batch element."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    T, H, V, bs = 8, 32, 16, 8
    B = 4
    pc_ref, _ = _build_pair(T, H, V, bs, True, device, seed=7)

    data = torch.randint(0, V, (B, T), device=device)
    missing_mask = torch.zeros(T, dtype=torch.bool, device=device)
    missing_mask[target_t] = True

    out = juice.queries.conditional(
        pc_ref, data=data, missing_mask=missing_mask, target_vars=[target_t],
    )

    assert out.shape == (B, 1, V)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("target_t", [0, 4, 7])
def test_generative_matches_smoothing_slice(target_t):
    """Self-consistency: ``target_vars=[t*]`` must equal the ``[:, t*, :]``
    slice of a full ``target_vars=None`` call with the same ``missing_mask``.
    Sliced full-output is a known-good reference for the partial-eval path
    (see memory ``project_partial_eval_block_bug``: the atomic-add fallback
    is silently wrong for ``block_size > 1``, but the full path isn't —
    so any future dispatch regression that routes us into that fallback
    will fail this test instead of returning subtly wrong probabilities)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    T, H, V, bs = 8, 32, 16, 8
    B = 4
    pc_ref, _ = _build_pair(T, H, V, bs, True, device, seed=13)

    data = torch.randint(0, V, (B, T), device=device)
    missing_mask = torch.zeros(T, dtype=torch.bool, device=device)
    missing_mask[target_t] = True

    out_full = juice.queries.conditional(
        pc_ref, data=data, missing_mask=missing_mask, target_vars=None,
    )
    out_target = juice.queries.conditional(
        pc_ref, data=data, missing_mask=missing_mask, target_vars=[target_t],
    )

    assert out_full.shape == (B, T, V)
    assert out_target.shape == (B, 1, V)
    torch.testing.assert_close(
        out_target[:, 0, :], out_full[:, target_t, :],
        rtol=1e-4, atol=1e-5,
    )


if __name__ == "__main__":
    for homo in [True, False]:
        for tt in [0, 4, 7]:
            test_sparse_categorical_generative_matches_reference(homo, tt)
            print(f"ok dense-vs-sparse: homogeneous={homo} target_t={tt}")
    for tt in [0, 4, 7]:
        test_generative_posterior_is_a_distribution(tt)
        print(f"ok normalised: target_t={tt}")
    for tt in [0, 4, 7]:
        test_generative_matches_smoothing_slice(tt)
        print(f"ok smoothing-slice: target_t={tt}")
