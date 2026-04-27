"""
Tests for ``SparseInputSumLayer``. Sparse-sum dispatch is driven by the DAG
class (:class:`SparseSumNodes` vs :class:`SumNodes`); to exercise the
non-sparse-sum reference path we pass ``_force_plain=True`` through ``summate``
so the comparison DAG stays on :class:`DenseSumLayer`. Validates the B=1
sparse fast path against the dense-sum reference, plus the B>1 fallback via
``super().forward/backward``.
"""

from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size
from pyjuice.model import TensorCircuit


# ---------------------------------------------------------------------------
# Helpers (mirrors sparsity_prop_prod_layer_test.py)
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
    csc_indices_list, csc_values_list = [], []
    for v in range(V):
        col_rows = torch.where(mask_hv[:, v])[0]
        csc_indices_list.extend(col_rows.tolist())
        csc_values_list.extend(dense_probs[col_rows, v].tolist())
        csc_indptr[v + 1] = len(csc_indices_list)
    return (csc_indptr,
            torch.tensor(csc_indices_list, dtype=torch.long),
            torch.tensor(csc_values_list, dtype=torch.float32))


def _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values,
                           homogeneous=True, force_plain=False):
    """Build the HMM DAG. ``force_plain=True`` threads ``_force_plain`` through
    both ``multiply`` and ``summate`` so the compilation stays on
    :class:`ProdLayer` + :class:`DenseSumLayer` — the reference for the
    sparse-path equivalence tests below. (Structural detection at compile
    time picks :class:`SparseInputSumLayer` whenever a sum's child is a
    :class:`SparseProdLayer`; keeping the prod plain is how we force the
    dense-sum reference.)"""
    num_node_blocks = H // bs
    with set_block_size(block_size=bs):
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
                ns = summate(curr_zs, num_node_blocks=num_node_blocks, _force_plain=force_plain)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=homogeneous)
            curr_zs = multiply(curr_xs, ns, _force_plain=force_plain)
        root = summate(curr_zs, num_node_blocks=1, block_size=1, _force_plain=force_plain)
    return root


def _copy_params(dst_pc, src_pc):
    dst_pc.params.data.copy_(src_pc.params.data)
    for dst, src in zip(dst_pc.input_layer_group, src_pc.input_layer_group):
        dst.params.data.copy_(src.params.data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sparse_input_sum_layer_is_used_b1():
    """Compile-time dispatch: every dense-eligible sum whose child is a
    SparseProdLayer gets a SparseInputSumLayer.

    We use an INHOMOGENEOUS HMM (no tied params) so the duplicated sum
    nodes also satisfy `_dense_eligible` (which requires ``not is_tied``).
    """
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=7)

    root = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False,
    )
    pc = TensorCircuit(
        root, use_dense_sum_layer=True, verbose=False,
    ).to(device)

    from pyjuice.layer import SparseInputSumLayer
    n_sparse_sum = sum(
        isinstance(layer, SparseInputSumLayer)
        for lg in pc.inner_layer_groups
        for layer in lg
    )
    # For T=4 inhomogeneous: ns_1 (d=2), ns_0 (d=3), root (d=4) all qualify.
    # ns_sum at d=1 consumes the 1-child wrapper around ns_input (plain
    # ProdLayer), so it stays on the regular DenseSumLayer path.
    assert n_sparse_sum >= T - 1, \
        f"expected at least {T-1} SparseInputSumLayer instances, got {n_sparse_sum}"


def test_sparse_input_sum_forward_b1_equivalence():
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=19)

    root_dense = _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False, force_plain=True)
    root_sparse = _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False)

    # Both circuits use the sparse prod path; only the sum differs.
    pc_dense_sum = TensorCircuit(
        root_dense, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    pc_sparse_sum = TensorCircuit(
        root_sparse, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    _copy_params(pc_sparse_sum, pc_dense_sum)

    torch.manual_seed(0)
    # B=1 exercises the sparse sum path.
    data = torch.randint(0, V, (1, T), device=device)
    ll_d = pc_dense_sum(data).detach()
    ll_s = pc_sparse_sum(data).detach()

    assert torch.allclose(ll_d, ll_s, atol=1e-4), \
        f"B=1 forward LL diff max = {(ll_d - ll_s).abs().max().item():.3e}"


def test_sparse_input_sum_backward_b1_equivalence():
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=23)

    root_dense = _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False, force_plain=True)
    root_sparse = _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False)

    pc_dense_sum = TensorCircuit(
        root_dense, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    pc_sparse_sum = TensorCircuit(
        root_sparse, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    _copy_params(pc_sparse_sum, pc_dense_sum)

    torch.manual_seed(1)
    data = torch.randint(0, V, (1, T), device=device)
    pc_dense_sum(data)
    pc_sparse_sum(data)

    # DenseSumLayer is inference-only — skip param_flow accumulation. Also
    # pin ``allow_modify_flows=False`` so neither path rewrites its own
    # node_flows in-place (the dense modify kernel transforms flows to
    # ``log(flow) - nmars`` at sum NIDs; the sparse fast path never needs to
    # modify, and we want comparable post-backward tensors).
    pc_dense_sum.backward(data, compute_param_flows=False, allow_modify_flows=False)
    pc_sparse_sum.backward(data, compute_param_flows=False, allow_modify_flows=False)

    # The sparse path no longer populates ``pc.node_flows`` at SparseCategorical
    # input rows — flow is exposed via the layer's ``_sparse_flows[ns_idx]``
    # slot, reachable through the ``_sparse_flow_owner`` back-reference on
    # each input ns. Compare those values to the dense path's ``node_flows``
    # at active CSC rows only (inactive rows are LOG_EPS-tiny on the dense
    # path and absent from the sparse container by construction).
    for in_ns_d, in_ns_s in zip(
        pc_dense_sum.input_layer_group[0].nodes,
        pc_sparse_sum.input_layer_group[0].nodes,
    ):
        owner = getattr(in_ns_s, "_sparse_flow_owner", None)
        assert owner is not None, "expected _sparse_flow_owner on each sparse input ns"
        sparse_layer, ns_idx = owner
        sv_flow = sparse_layer._sparse_flows[ns_idx]
        assert sv_flow is not None, "sparse backward should leave sv_flow populated"
        lo_d, _ = in_ns_d._output_ind_range
        torch.testing.assert_close(
            sv_flow.values.cpu(),
            pc_dense_sum.node_flows[lo_d + sv_flow.indices, 0].cpu(),
            rtol=1e-2, atol=1e-3,
        )


def test_sparse_input_sum_b_gt_1_rejected():
    """The sparse fast path is B=1-only; B>1 must raise a clear error so
    users build with plain ``summate`` / ``multiply`` instead."""
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    B = 4
    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=29)

    root = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False,
    )
    pc = TensorCircuit(root, use_dense_sum_layer=True, verbose=False).to(device)

    data = torch.randint(0, V, (B, T), device=device)
    with pytest.raises(AssertionError, match="B=1 only"):
        pc(data)


def test_sparse_prod_skip_scatter_when_all_consumers_sparse():
    """When every SparseProdLayer consumer is a SparseInputSumLayer, its
    ``_skip_scatter`` flag is flipped on by the post-compile pass. When some
    consumer is not, the flag stays off so element_mars is still populated."""
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=41)

    from pyjuice.layer import SparseProdLayer

    # Case A: sparse-sum enabled → all SparseProdLayers should have _skip_scatter=True.
    root_a = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False,
    )
    pc_a = TensorCircuit(
        root_a, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    sparse_prods_a = [l for lg in pc_a.inner_layer_groups
                      for l in lg if isinstance(l, SparseProdLayer)]
    assert len(sparse_prods_a) >= 1
    assert all(l._skip_scatter for l in sparse_prods_a), \
        "expected all SparseProdLayers to skip scatter when every consumer is sparse"

    # Case B: force the summates to plain SumNodes → consumer is DenseSumLayer → scatter still needed.
    root_b = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False,
        force_plain=True,
    )
    pc_b = TensorCircuit(
        root_b, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    sparse_prods_b = [l for lg in pc_b.inner_layer_groups
                      for l in lg if isinstance(l, SparseProdLayer)]
    assert not any(l._skip_scatter for l in sparse_prods_b), \
        "expected SparseProdLayer.scatter NOT skipped when consumer is dense-sum"


def test_sparse_prod_skip_scatter_forward_correctness():
    """With scatter-skip on, SparseProdLayer does NOT populate element_mars
    but the final LL still matches a compilation that does."""
    device = torch.device("cuda:0")
    T, H, V, bs = 4, 8, 16, 4
    csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, density=0.4, seed=47)

    root_dense = _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False, force_plain=True)
    root_skip = _build_sparse_hmm_dag(T, H, V, bs, csc_indptr, csc_indices, csc_values, homogeneous=False)

    pc_ref = TensorCircuit(
        root_dense, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    pc_skip = TensorCircuit(
        root_skip, use_dense_sum_layer=True, verbose=False,
    ).to(device)
    _copy_params(pc_skip, pc_ref)

    torch.manual_seed(0)
    data = torch.randint(0, V, (1, T), device=device)
    ll_ref = pc_ref(data).detach()
    ll_skip = pc_skip(data).detach()
    assert torch.allclose(ll_ref, ll_skip, atol=1e-4), \
        f"LL diff with scatter-skip: {(ll_ref - ll_skip).abs().max().item():.3e}"


if __name__ == "__main__":
    test_sparse_input_sum_layer_is_used_b1()
    test_sparse_input_sum_forward_b1_equivalence()
    test_sparse_input_sum_backward_b1_equivalence()
    test_sparse_input_sum_b_gt_1_rejected()
    test_sparse_prod_skip_scatter_when_all_consumers_sparse()
    test_sparse_prod_skip_scatter_forward_correctness()
    print("all sparse_input_sum_layer tests passed")
