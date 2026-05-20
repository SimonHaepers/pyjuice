"""Correctness test for :class:`SparseIOBlockDiagonalSumLayer`.

Builds the same HMM twice:
  * **plain**: SparseCategorical input + `_force_plain=True` everywhere, so
    the compiler picks generic :class:`ProdLayer` / :class:`SumLayer`.
  * **sparse_io_bd_chain**: same emission but each transition is a
    block-diagonal ``summate`` (single BD; no Monarch permutation —
    that's outside the scope of (a)). Combined with sparse emissions,
    the DAG pre-pass classifies the BD sums as sparse-IO-eligible and
    the dispatch routes them to :class:`SparseIOBlockDiagonalSumLayer`.

Forward LL must match across the two builds at fp32 tolerance, and the
chain-interior layers on the fast-path circuit must compile to the
expected sparse-IO BD class.
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
    CoSparseProdLayer, SparseIOBlockDiagonalSumLayer, SparseInputSumLayer,
    SparseProdLayer,
)


def _random_csc_pattern(H: int, V: int, density: float, seed: int):
    """Random CSC emission pattern with per-row coverage, row-normalised.

    Reused verbatim from ``tests/queries/sparse_io_sum_test.py`` — every
    latent row gets at least one active emission column, and per-row sums
    are normalised to 1.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    target_nnz = int(density * H * V * 1.05) + H
    rows = torch.randint(0, H, (target_nnz,), generator=g)
    cols = torch.randint(0, V, (target_nnz,), generator=g)
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


def _bd_edge_ids(NB: int) -> torch.Tensor:
    return torch.arange(0, NB)[None, :].repeat(2, 1)


def _build_plain_bd(T: int, H: int, V: int, bs: int,
                     csc_indptr: torch.Tensor, csc_indices: torch.Tensor,
                     csc_values: torch.Tensor):
    """Plain baseline — BD-shaped transitions but every layer pinned plain."""
    num_node_blocks = H // bs
    bd_edges = _bd_edge_ids(num_node_blocks)
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
                ns = summate(curr_zs, edge_ids=bd_edges, block_size=bs,
                              _force_plain=True)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
                ns._force_plain_layer = True
            curr_zs = multiply(curr_xs, ns, _force_plain=True)
        return summate(curr_zs, num_node_blocks=1, block_size=1,
                        _force_plain=True)


def _build_sparse_io_bd_chain(T: int, H: int, V: int, bs: int,
                                csc_indptr: torch.Tensor,
                                csc_indices: torch.Tensor,
                                csc_values: torch.Tensor):
    """Sparse-chain build with BD transitions. The DAG pre-pass detects
    the SparseProdNodes → BD summate → SparseProdNodes pattern and the
    dispatch upgrades each interior pair to
    :class:`SparseIOBlockDiagonalSumLayer` + :class:`CoSparseProdLayer`."""
    num_node_blocks = H // bs
    bd_edges = _bd_edge_ids(num_node_blocks)
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
                ns = summate(curr_zs, edge_ids=bd_edges, block_size=bs)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = sparse_multiply(curr_xs, ns)
        return sparse_summate(curr_zs, num_node_blocks=1, block_size=1)


@pytest.mark.parametrize("T,H,V,bs", [
    (3, 8, 4, 4),
    (5, 16, 8, 8),
    (6, 32, 12, 8),
])
def test_sparse_io_bd_chain_matches_plain(T, H, V, bs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.6, seed=17,
    )

    torch.manual_seed(123)
    root_plain = _build_plain_bd(T, H, V, bs, csc_indptr, csc_indices, csc_values)
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(123)
    root_chain = _build_sparse_io_bd_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_chain = juice.TensorCircuit(root_chain, verbose=False).to(device)
    pc_chain.params.data.copy_(pc_plain.params.data)

    # Sanity-check the compile actually picked the sparse-IO BD class.
    has_bd_io = any(
        isinstance(layer, SparseIOBlockDiagonalSumLayer)
        for lg in pc_chain.inner_layer_groups for layer in lg
    )
    has_cosparse = any(
        isinstance(layer, CoSparseProdLayer)
        for lg in pc_chain.inner_layer_groups for layer in lg
    )
    assert has_bd_io, (
        "expected SparseIOBlockDiagonalSumLayer on the BD sparse-chain build"
    )
    assert has_cosparse, (
        "expected CoSparseProdLayer on the BD sparse-chain build"
    )

    # Forward LL parity across a handful of seeds.
    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        data = torch.randint(0, V, (1, T), device=device)
        lls_plain = pc_plain(data)
        lls_chain = pc_chain(data)
        torch.testing.assert_close(lls_chain, lls_plain, rtol=1e-3, atol=5e-3)


@pytest.mark.parametrize("T,H,V,bs", [(4, 8, 4, 4), (6, 16, 8, 8)])
def test_sparse_io_bd_chain_backward_matches_plain(T, H, V, bs):
    """``pc.backward()`` must drive the same gradient through both paths.
    Sparse-chain flow lives in ``_sparse_flows[ns_idx]`` on the upstream
    :class:`SparseProdLayer`; we compare it against the dense
    ``node_flows`` slot at the matching active CSC rows."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.5, seed=31,
    )

    torch.manual_seed(7)
    root_plain = _build_plain_bd(T, H, V, bs, csc_indptr, csc_indices, csc_values)
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(7)
    root_chain = _build_sparse_io_bd_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_chain = juice.TensorCircuit(root_chain, verbose=False).to(device)
    pc_chain.params.data.copy_(pc_plain.params.data)

    torch.manual_seed(0)
    data = torch.randint(0, V, (1, T), device=device)

    lls_plain = pc_plain(data)
    lls_chain = pc_chain(data)
    torch.testing.assert_close(lls_chain, lls_plain, rtol=1e-3, atol=5e-3)

    pc_plain.backward(data, compute_param_flows=False, flows_memory=0.0)
    pc_chain.backward(data, compute_param_flows=False, flows_memory=0.0)

    for in_ns_plain, in_ns_chain in zip(
        pc_plain.input_layer_group[0].nodes,
        pc_chain.input_layer_group[0].nodes,
    ):
        lo_p, _ = in_ns_plain._output_ind_range
        owner = getattr(in_ns_chain, "_sparse_flow_owner", None)
        assert owner is not None, (
            "expected SparseProdLayer to set _sparse_flow_owner on each "
            "SparseCategorical input ns at compile time"
        )
        sparse_layer, sparse_ns_idx = owner
        sv_flow = sparse_layer._sparse_flows[sparse_ns_idx]
        assert sv_flow is not None, (
            "expected backward to leave sv_flow populated post-call"
        )
        active = sv_flow.indices.cpu()
        torch.testing.assert_close(
            sv_flow.values.cpu(),
            pc_plain.node_flows[lo_p + active, 0].cpu(),
            rtol=1e-2, atol=5e-3,
        )
