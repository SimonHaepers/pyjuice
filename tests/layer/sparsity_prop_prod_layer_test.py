"""
Equivalence + EM tests for ``SparseProdLayer``. Sparse/plain dispatch is driven
by the DAG class (:class:`SparseProdNodes` vs :class:`ProdNodes`); the reference
DAG is built with ``_force_plain=True`` on the construction helpers to compile
to a plain :class:`ProdLayer`, and the comparison DAG picks up the sparse path
via the default auto-detection.

Inactive-row semantics differ by design: the sparse path fills inactive rows
of ``element_mars`` with plain ``LOG_EPS`` instead of ``LOG_EPS +
Σ log_trans``. Tests therefore use ``atol=1e-4``, which is comfortably above
the per-inactive-latent contribution ``exp(LOG_EPS) ≈ 1e-10``.
"""

from __future__ import annotations

import math

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size
from pyjuice.model import TensorCircuit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csc_pattern(H, V, density=0.5, seed=42):
    g = torch.Generator().manual_seed(seed)
    mask_hv = (torch.rand(H, V, generator=g) < density)
    for h in range(H):
        if not mask_hv[h].any():
            mask_hv[h, torch.randint(0, V, (1,), generator=g).item()] = True

    dense_probs = torch.rand(H, V, generator=g) * mask_hv.float()
    dense_probs = dense_probs / dense_probs.sum(dim=1, keepdim=True)

    csc_indptr = torch.zeros(V + 1, dtype=torch.long)
    csc_indices_list = []
    csc_values_list = []
    for v in range(V):
        col_rows = torch.where(mask_hv[:, v])[0]
        csc_indices_list.extend(col_rows.tolist())
        csc_values_list.extend(dense_probs[col_rows, v].tolist())
        csc_indptr[v + 1] = len(csc_indices_list)

    csc_indices = torch.tensor(csc_indices_list, dtype=torch.long)
    csc_values = torch.tensor(csc_values_list, dtype=torch.float32)
    return csc_indptr, csc_indices, csc_values


def _build_sparse_hmm_dag(T, H, V, block_size,
                           csc_indptr, csc_indices, csc_values,
                           homogeneous=True, force_plain=False,
                           force_plain_sum_only=False):
    """Construct an HMM DAG with a SparseCategorical input. Returns the root.

    ``force_plain=True`` threads ``_force_plain`` through both ``multiply``
    and ``summate`` so the whole DAG compiles with plain :class:`ProdLayer` +
    :class:`SumLayer` — the baseline reference for the sparse-path tests.
    ``force_plain_sum_only=True`` keeps the multiplies auto-detecting as
    :class:`SparseProdNodes` (exercising :class:`SparseProdLayer`) but keeps
    the sums plain so the structural fallback does NOT pick
    :class:`SparseInputSumLayer` — used by tests that need param-flow
    accumulation through the sums (``SparseInputSumLayer`` is inference-only).
    """
    num_node_blocks = H // block_size
    fp_mul = force_plain
    fp_sum = force_plain or force_plain_sum_only
    with set_block_size(block_size=block_size):
        ns_input = inputs(
            T - 1, num_node_blocks=num_node_blocks,
            dist=dists.SparseCategorical(num_cats=V),
            csc_indptr=csc_indptr, csc_indices=csc_indices,
        )
        ns_input.set_params(csc_values, normalize=False)

        ns_sum = None
        curr_zs = ns_input
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=homogeneous)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=num_node_blocks, _force_plain=fp_sum)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=homogeneous)
            curr_zs = multiply(curr_xs, ns, _force_plain=fp_mul)

        root = summate(curr_zs, num_node_blocks=1, block_size=1, _force_plain=fp_sum)
    return root


def _copy_params(dst_pc: TensorCircuit, src_pc: TensorCircuit) -> None:
    """Byte-level param copy so two compilations start from identical state."""
    dst_pc.params.data.copy_(src_pc.params.data)
    for dst, src in zip(dst_pc.input_layer_group, src_pc.input_layer_group):
        dst.params.data.copy_(src.params.data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("homogeneous", [True, False])
def test_sparse_prod_forward_ll_equivalence(homogeneous):
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    B = 1

    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=17)

    root_dense = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=homogeneous,
        force_plain=True,
    )
    root_sparse = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=homogeneous,
    )

    pc_dense = TensorCircuit(root_dense, verbose=False).to(device)
    pc_sparse = TensorCircuit(root_sparse, verbose=False).to(device)
    _copy_params(pc_sparse, pc_dense)

    # Verify the sparse compilation actually picked the sparse path.
    from pyjuice.layer import SparseProdLayer
    assert any(
        isinstance(layer, SparseProdLayer)
        for lg in pc_sparse.inner_layer_groups
        for layer in lg
    ), "Expected at least one SparseProdLayer in the sparse compilation."

    torch.manual_seed(0)
    data = torch.randint(0, V, (B, T), device=device)

    ll_d = pc_dense(data).detach()
    ll_s = pc_sparse(data).detach()

    max_diff = (ll_d - ll_s).abs().max().item()
    assert torch.allclose(ll_d, ll_s, atol=1e-4), \
        f"LL mismatch (homogeneous={homogeneous}): max diff = {max_diff:.3e}"


@pytest.mark.parametrize("homogeneous", [True, False])
def test_sparse_prod_backward_param_flow_equivalence(homogeneous):
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    B = 1

    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=23)

    root_dense = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=homogeneous,
        force_plain=True,
    )
    # Keep sums plain so SparseInputSumLayer (inference-only) is not picked;
    # the prods stay on the SparseProdLayer path, which is what this test checks.
    root_sparse = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=homogeneous,
        force_plain_sum_only=True,
    )

    pc_dense = TensorCircuit(root_dense, verbose=False).to(device)
    pc_sparse = TensorCircuit(root_sparse, verbose=False).to(device)
    _copy_params(pc_sparse, pc_dense)

    torch.manual_seed(1)
    data = torch.randint(0, V, (B, T), device=device)

    pc_dense(data)
    pc_sparse(data)
    pc_dense.backward(data)
    pc_sparse.backward(data)

    # LOG_EPS-fill divergence at inactive rows accumulates through the sum/prod
    # stack; allow ~1e-3 tolerance (inactive contributions per-node are
    # ~exp(LOG_EPS)=1e-10, amplified by the logsumexp + chain-rule path into
    # the mid-1e-3 range for the inhomogeneous case).
    pf_d = pc_dense.param_flows.detach().cpu()
    pf_s = pc_sparse.param_flows.detach().cpu()
    assert torch.allclose(pf_d, pf_s, atol=1e-3, rtol=1e-2), \
        f"sum param_flows diff max = {(pf_d - pf_s).abs().max().item():.3e}"

    # Input-layer (SparseCategorical) param_flows.
    assert len(pc_dense.input_layer_group.layers) == len(pc_sparse.input_layer_group.layers)
    for il_d, il_s in zip(pc_dense.input_layer_group, pc_sparse.input_layer_group):
        if il_d.param_flows is None or il_s.param_flows is None:
            continue
        pfd = il_d.param_flows.detach().cpu()
        pfs = il_s.param_flows.detach().cpu()
        assert torch.allclose(pfd, pfs, atol=1e-3, rtol=1e-2), \
            f"input param_flows diff max = {(pfd - pfs).abs().max().item():.3e}"


def test_sparse_prod_em_step_equivalence():
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    B = 1

    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=31)

    root_dense = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=True,
        force_plain=True,
    )
    # EM needs param_flow accumulation through sums — SparseInputSumLayer
    # rejects that, so keep sums plain.
    root_sparse = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=True,
        force_plain_sum_only=True,
    )

    pc_dense = TensorCircuit(root_dense, verbose=False).to(device)
    pc_sparse = TensorCircuit(root_sparse, verbose=False).to(device)
    _copy_params(pc_sparse, pc_dense)

    torch.manual_seed(2)
    data = torch.randint(0, V, (B, T), device=device)

    pc_dense(data); pc_dense.backward(data)
    pc_sparse(data); pc_sparse.backward(data)

    pc_dense.mini_batch_em(step_size=0.3, pseudocount=0.1, keep_zero_params=False)
    pc_sparse.mini_batch_em(step_size=0.3, pseudocount=0.1, keep_zero_params=False)

    # Sum-layer params.
    p_d = pc_dense.params.data.detach().cpu()
    p_s = pc_sparse.params.data.detach().cpu()
    assert torch.allclose(p_d, p_s, atol=1e-3, rtol=1e-2), \
        f"post-EM sum params diff max = {(p_d - p_s).abs().max().item():.3e}"

    # Input-layer params (CSC values).
    for il_d, il_s in zip(pc_dense.input_layer_group, pc_sparse.input_layer_group):
        ip_d = il_d.params.data.detach().cpu()
        ip_s = il_s.params.data.detach().cpu()
        assert torch.allclose(ip_d, ip_s, atol=1e-3, rtol=1e-2), \
            f"post-EM input params diff max = {(ip_d - ip_s).abs().max().item():.3e}"


def test_sparse_prod_skip_flag_spot_check():
    """After compile: duplicated sparse inputs fed into a SparseProdLayer are
    marked ``_skip_input_forward`` / ``_skip_input_backward``; the source
    ``ns_input`` (consumed by the initial summate) is NOT marked."""
    device = torch.device("cuda:0")
    T, H, V, bs = 3, 4, 8, 4
    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.5, seed=7)

    root = _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values)
    pc = TensorCircuit(root, verbose=False).to(device)

    # Walk the DAG to find SparseCategorical input nodes.
    from pyjuice.nodes.distributions import SparseCategorical
    sparse_inputs = []
    def _walk(ns, seen):
        if ns in seen:
            return
        seen.add(ns)
        if hasattr(ns, "dist") and isinstance(ns.dist, SparseCategorical):
            sparse_inputs.append(ns)
        for cs in getattr(ns, "chs", []):
            _walk(cs, seen)
    _walk(root, set())

    # Expect T sparse inputs: the source + (T-1) duplicates.
    assert len(sparse_inputs) == T

    # The source is consumed directly by the initial summate — must NOT be skipped.
    source = next(ns for ns in sparse_inputs if not ns.is_tied())
    assert not getattr(source, "_skip_input_forward", False), \
        "ns_input (consumed by summate) should NOT have _skip_input_forward."
    assert not getattr(source, "_skip_input_backward", False), \
        "ns_input (consumed by summate) should NOT have _skip_input_backward."

    # All duplicates are consumed by a multiply only — should be skipped.
    duplicates = [ns for ns in sparse_inputs if ns.is_tied()]
    assert len(duplicates) == T - 1
    for ns in duplicates:
        assert getattr(ns, "_skip_input_forward", False), \
            "duplicated curr_xs should have _skip_input_forward=True."
        assert getattr(ns, "_skip_input_backward", False), \
            "duplicated curr_xs should have _skip_input_backward=True."


if __name__ == "__main__":
    test_sparse_prod_forward_ll_equivalence(True)
    test_sparse_prod_forward_ll_equivalence(False)
    test_sparse_prod_backward_param_flow_equivalence(True)
    test_sparse_prod_backward_param_flow_equivalence(False)
    test_sparse_prod_em_step_equivalence()
    test_sparse_prod_skip_flag_spot_check()
    print("all sparsity_prop_prod_layer tests passed")
