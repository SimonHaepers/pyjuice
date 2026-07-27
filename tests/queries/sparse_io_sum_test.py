"""Correctness test for the sparse-IO sum + co-sparse prod fast path.

Builds the same HMM twice:
  * **plain**: all ``multiply`` / ``summate`` with ``_force_plain=True``, so the
    compiler picks :class:`ProdLayer` + :class:`SumLayer`. Uses a
    :class:`SparseCategorical` input to match storage, but no inner-layer
    sparsity propagation.
  * **sparse_io_chain**: ``sparse_multiply`` + plain ``summate`` on intermediate
    sums so the structural fallback picks up the chain. The DAG pre-pass in
    :class:`TensorCircuit` detects the ``SparseSumNodes → SparseProdNodes``
    chain and upgrades the interior sum/prod pairs to
    :class:`SparseIOSumLayer` + :class:`CoSparseProdLayer`.

Emissions and transition parameters are copied across the two circuits after
compilation. The forward LL must match at float32 tolerance and every chain-
interior layer must compile to the expected sparse class on the fast-path
circuit.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import (
    inputs, multiply, summate, sparse_multiply, sparse_summate, set_block_size,
)
from pyjuice.layer import (
    CoSparseProdLayer, SparseIOSumLayer, SparseInputSumLayer, SparseProdLayer,
)


def _random_csc_pattern(H: int, V: int, density: float, seed: int):
    """Random CSC emission pattern with per-row coverage, row-normalised."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    target_nnz = int(density * H * V * 1.05) + H
    rows = torch.randint(0, H, (target_nnz,), generator=g)
    cols = torch.randint(0, V, (target_nnz,), generator=g)
    # Coverage: every row gets at least one column.
    all_rows = torch.arange(H)
    cov_cols = torch.randint(0, V, (H,), generator=g)
    rows = torch.cat([rows, all_rows])
    cols = torch.cat([cols, cov_cols])
    linear = rows.to(torch.long) * V + cols.to(torch.long)
    linear = torch.unique(linear)
    rows_d = linear // V
    cols_d = linear % V
    sort_key = cols_d * H + rows_d
    order = torch.argsort(sort_key)
    csc_indices = rows_d[order].contiguous()
    cols_sorted = cols_d[order]
    col_counts = torch.bincount(cols_sorted, minlength=V)
    csc_indptr = torch.zeros(V + 1, dtype=torch.long)
    csc_indptr[1:] = torch.cumsum(col_counts, dim=0)
    raw = torch.rand(csc_indices.numel(), generator=g)
    row_sums = torch.zeros(H)
    row_sums.scatter_add_(0, csc_indices, raw)
    csc_values = (raw / row_sums[csc_indices]).to(torch.float32)
    return csc_indptr, csc_indices, csc_values


def _build_plain(T: int, H: int, V: int, bs: int,
                  csc_indptr: torch.Tensor, csc_indices: torch.Tensor,
                  csc_values: torch.Tensor):
    """Reference build — every multiply/summate pinned to plain ProdLayer/SumLayer."""
    num_node_blocks = H // bs
    with set_block_size(block_size=bs):
        ns_input = inputs(
            T - 1, num_node_blocks=num_node_blocks,
            dist=dists.SparseCategorical(num_cats=V),
            csc_indptr=csc_indptr, csc_indices=csc_indices,
        )
        ns_input.set_params(csc_values, normalize=False)
        ns_sum = None
        curr_zs = multiply(ns_input, _force_plain=True)
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=num_node_blocks,
                             _force_plain=True)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = multiply(curr_xs, ns, _force_plain=True)
        return summate(curr_zs, num_node_blocks=1, block_size=1,
                        _force_plain=True)


def _build_sparse_chain(T: int, H: int, V: int, bs: int,
                         csc_indptr: torch.Tensor, csc_indices: torch.Tensor,
                         csc_values: torch.Tensor):
    """Sparse-chain build — matches ``_build_sparse_hmm_dag`` in
    ``sparse_categorical_cond_perf_test.py``; the chain auto-upgrades at
    compile time."""
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
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=num_node_blocks)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = sparse_multiply(curr_xs, ns)
        return sparse_summate(curr_zs, num_node_blocks=1, block_size=1)


@pytest.mark.parametrize("T,H,V,bs,B", [
    (3, 8, 4, 4, 1), (5, 16, 8, 8, 1), (6, 16, 12, 4, 1),
    (3, 8, 4, 4, 4), (5, 16, 8, 8, 7), (6, 16, 12, 4, 16),
])
def test_sparse_io_chain_matches_plain(T, H, V, bs, B):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.6, seed=17,
    )

    torch.manual_seed(123)
    root_plain = _build_plain(T, H, V, bs, csc_indptr, csc_indices, csc_values)
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(123)
    root_sparse = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)

    # Inner-layer params have identical layout across both builds (same
    # multiply/summate topology, same ns order). Copy verbatim so the two
    # circuits encode the same distribution exactly.
    pc_sparse.params.data.copy_(pc_plain.params.data)

    # Sanity-check the compile actually picked the sparse-chain classes.
    has_io_sum = any(
        isinstance(layer, SparseIOSumLayer)
        for lg in pc_sparse.inner_layer_groups for layer in lg
    )
    has_cosparse_prod = any(
        isinstance(layer, CoSparseProdLayer)
        for lg in pc_sparse.inner_layer_groups for layer in lg
    )
    # For T >= 3 the chain has at least one interior sum+prod pair eligible
    # for the upgrade.
    assert has_io_sum, "expected SparseIOSumLayer on the sparse-chain build"
    assert has_cosparse_prod, "expected CoSparseProdLayer on the sparse-chain build"

    # Root sum stays as the dense-output fast path.
    root_sum = None
    for lg in pc_sparse.inner_layer_groups:
        if not lg.is_prod():
            for layer in lg:
                root_sum = layer
    assert root_sum is not None
    # Root is the LAST sum layer in the topological order; it should be a
    # plain SparseInputSumLayer (and NOT a SparseIOSumLayer) because it has no
    # downstream sparse consumer.
    assert isinstance(root_sum, SparseInputSumLayer) \
           and not isinstance(root_sum, SparseIOSumLayer), (
        "root sum layer should stay as dense-output SparseInputSumLayer"
    )

    # Forward comparison — exercise the chain on a handful of token batches
    # (B=1 keeps the classic single-column path; B>1 runs the batched chain).
    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        data = torch.randint(0, V, (B, T), device=device)
        lls_plain = pc_plain(data)
        lls_sparse = pc_sparse(data)
        torch.testing.assert_close(lls_sparse, lls_plain, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("T,H,V,bs,B", [(5, 16, 8, 8, 6), (6, 16, 12, 4, 3)])
def test_sparse_io_chain_batched_matches_loop(T, H, V, bs, B):
    """Batched sparse forward == concatenation of per-sample B=1 calls
    (tight tolerance — catches cross-sample leakage without needing the
    plain reference build)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.6, seed=23,
    )
    torch.manual_seed(7)
    root_sparse = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)

    torch.manual_seed(11)
    data = torch.randint(0, V, (B, T), device=device)
    lls = pc_sparse(data)
    lls_loop = torch.cat([pc_sparse(data[b:b + 1]) for b in range(B)], dim=0)
    torch.testing.assert_close(lls, lls_loop, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("allow_modify_flows", [False, True])
@pytest.mark.parametrize("T,H,V,bs,B", [
    (4, 8, 4, 4, 1), (6, 16, 8, 8, 1),
    (4, 8, 4, 4, 5), (6, 16, 8, 8, 8),
])
def test_sparse_io_chain_backward_matches_plain(T, H, V, bs, B, allow_modify_flows):
    """``pc.backward()`` should drive the same root LL gradient through both
    paths, so the sparse flow containers match the plain ``pc.node_flows``
    at the active rows, per sample."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.5, seed=31,
    )

    torch.manual_seed(7)
    root_plain = _build_plain(T, H, V, bs, csc_indptr, csc_indices, csc_values)
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(7)
    root_sparse = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)
    pc_sparse.params.data.copy_(pc_plain.params.data)

    torch.manual_seed(0)
    data = torch.randint(0, V, (B, T), device=device)

    lls_plain = pc_plain(data)
    lls_sparse = pc_sparse(data)
    torch.testing.assert_close(lls_sparse, lls_plain, rtol=1e-4, atol=1e-5)

    pc_plain.backward(data, compute_param_flows=False, flows_memory=0.0,
                      allow_modify_flows=allow_modify_flows)
    pc_sparse.backward(data, compute_param_flows=False, flows_memory=0.0,
                       allow_modify_flows=allow_modify_flows)

    # Sparse chain leaves ``pc.node_flows`` untouched at SparseCategorical
    # input rows — flow lives in the layer's ``_sparse_flows[ns_idx]`` slot,
    # reachable from the input ns via ``_sparse_flow_owner``. Compare those
    # values to the plain path's dense ``node_flows`` at exactly the active
    # CSC rows, per sample (inactive rows on the plain path are LOG_EPS-tiny
    # and absent from the sparse container by construction).
    for in_ns_plain, in_ns_sparse in zip(
        pc_plain.input_layer_group[0].nodes, pc_sparse.input_layer_group[0].nodes,
    ):
        lo_p, _ = in_ns_plain._output_ind_range
        owner = getattr(in_ns_sparse, "_sparse_flow_owner", None)
        assert owner is not None, (
            "expected SparseProdLayer to set _sparse_flow_owner on each "
            "SparseCategorical input ns at compile time"
        )
        sparse_layer, sparse_ns_idx = owner
        sv_flow = sparse_layer._sparse_flows[sparse_ns_idx]
        assert sv_flow is not None, (
            "expected backward to leave sv_flow populated post-call"
        )
        if sv_flow.is_batched:
            col_starts_cpu = sv_flow.col_starts.cpu().tolist()
            for b in range(B):
                k_b = sv_flow.nnz_list[b]
                if k_b == 0:
                    continue
                active_b = sv_flow.indices[
                    col_starts_cpu[b]:col_starts_cpu[b] + k_b].cpu()
                torch.testing.assert_close(
                    sv_flow.values[b, :k_b].cpu(),
                    pc_plain.node_flows[lo_p + active_b, b].cpu(),
                    rtol=1e-3, atol=1e-5,
                )
        else:
            active = sv_flow.indices.cpu()
            torch.testing.assert_close(
                sv_flow.values.cpu(),
                pc_plain.node_flows[lo_p + active, 0].cpu(),
                rtol=1e-3, atol=1e-5,
            )


@pytest.mark.parametrize("T,H,V,bs", [(4, 8, 4, 4), (5, 16, 8, 8)])
def test_sparse_io_chain_conditional_matches_plain(T, H, V, bs):
    """``juice.queries.conditional()`` on the sparse-chain build must match
    the plain build. Exercises the sparse-native conditional backward kernel
    that reads ``SparseNodeValues`` directly via ``_sparse_flow_owner``,
    no dense ``node_flows`` intermediate."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import pyjuice.queries as queries
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.5, seed=29,
    )

    torch.manual_seed(11)
    root_plain = _build_plain(T, H, V, bs, csc_indptr, csc_indices, csc_values)
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(11)
    root_sparse = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)
    pc_sparse.params.data.copy_(pc_plain.params.data)

    torch.manual_seed(0)
    data = torch.randint(0, V, (1, T), device=device)

    out_plain = queries.conditional(pc_plain, data=data, target_vars=list(range(T)))
    out_sparse = queries.conditional(pc_sparse, data=data, target_vars=list(range(T)))

    torch.testing.assert_close(out_sparse, out_plain, rtol=1e-3, atol=1e-4)


def test_sparse_io_chain_conditional_b_gt_1_rejected():
    """Conditional queries on the sparse chain stay B=1-only: the sparse-flow
    conditional backward asserts batch_size == 1, and the batched training
    path must not silently change that."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import pyjuice.queries as queries
    device = torch.device("cuda:0")

    T, H, V, bs = 4, 8, 4, 4
    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.5, seed=29,
    )
    torch.manual_seed(11)
    root_sparse = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)

    torch.manual_seed(0)
    data = torch.randint(0, V, (3, T), device=device)
    with pytest.raises(AssertionError, match="batch_size == 1"):
        queries.conditional(pc_sparse, data=data, target_vars=list(range(T)))


if __name__ == "__main__":
    for params in [(3, 8, 4, 4, 1), (5, 16, 8, 8, 7)]:
        test_sparse_io_chain_matches_plain(*params)
        print(f"ok forward: {params}")
    for params in [(4, 8, 4, 4, 1), (6, 16, 8, 8, 8)]:
        test_sparse_io_chain_backward_matches_plain(*params, allow_modify_flows=True)
        print(f"ok backward: {params}")
    for params in [(4, 8, 4, 4), (5, 16, 8, 8)]:
        test_sparse_io_chain_conditional_matches_plain(*params)
        print(f"ok conditional: {params}")
    test_sparse_io_chain_conditional_b_gt_1_rejected()
    print("ok conditional B>1 rejected")
