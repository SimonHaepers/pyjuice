"""Parameter-flow backward correctness for :class:`SparseIOSumLayer`
(the non-BD variant used on block-dense ``summate`` chains with sparse-IO
upgrade).

Mirrors :mod:`tests.queries.sparse_io_block_diagonal_param_flow_test` but
the chain interior uses block-dense ``summate(num_node_blocks=...)`` rather
than BD-shaped ``summate(edge_ids=arange(NB)[None,:].repeat(2,1))``. The
DAG pre-pass routes the interior sum + co-sparse prod pairs to
:class:`SparseIOSumLayer` + :class:`CoSparseProdLayer` whenever the
upstream / downstream pattern matches.

Both the per-edge ``_param_flow`` tensor and a full EM update are compared
against a ``_force_plain=True`` baseline using ``SparseCategorical``
emissions; the dense-CSC variant (every ``(latent, category)`` slot
active, à la :mod:`tests.queries.sparse_categorical_cond_test`) is the
direct cousin of that conditional-query test.
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
    CoSparseProdLayer, SparseIOSumLayer, SparseInputSumLayer,
)


# ---------------------------------------------------------------------- #
# Helpers (CSC patterns + builders)
# ---------------------------------------------------------------------- #


def _random_csc_pattern(H: int, V: int, density: float, seed: int):
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


def _dense_csc_pattern(H: int, V: int, seed: int):
    """Dense CSC: every ``(row, col)`` active, row-normalised. Matches the
    pattern used in :mod:`tests.queries.sparse_categorical_cond_test`."""
    csc_indptr = torch.arange(0, V * H + 1, H, dtype=torch.long)
    csc_indices = torch.arange(H, dtype=torch.long).repeat(V)
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.rand(csc_indices.numel(), generator=g)
    row_sums = torch.zeros(H)
    row_sums.scatter_add_(0, csc_indices, raw)
    csc_values = (raw / row_sums[csc_indices]).to(torch.float32)
    return csc_indptr, csc_indices, csc_values


def _build_plain(T: int, H: int, V: int, bs: int,
                  csc_indptr: torch.Tensor, csc_indices: torch.Tensor,
                  csc_values: torch.Tensor):
    """Plain baseline — block-dense ``summate`` chain pinned to plain SumLayer."""
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
                ns._force_plain_layer = True
            curr_zs = multiply(curr_xs, ns, _force_plain=True)
        root = summate(curr_zs, num_node_blocks=1, block_size=1,
                       _force_plain=True)
        return root, ns_sum


def _build_sparse_chain(T: int, H: int, V: int, bs: int,
                         csc_indptr: torch.Tensor, csc_indices: torch.Tensor,
                         csc_values: torch.Tensor):
    """Sparse-IO chain (non-BD). Block-dense ``summate`` interior with
    ``sparse_multiply`` / ``sparse_summate`` wrappers — DAG pre-pass
    upgrades the interior pairs to :class:`SparseIOSumLayer` +
    :class:`CoSparseProdLayer`."""
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
        root = sparse_summate(curr_zs, num_node_blocks=1, block_size=1)
        return root, ns_sum


def _has_sparse_io_sum(pc: juice.TensorCircuit) -> bool:
    for lg in pc.inner_layer_groups:
        for layer in lg:
            if isinstance(layer, SparseIOSumLayer):
                return True
    return False


# ---------------------------------------------------------------------- #
# Per-edge param-flow parity
# ---------------------------------------------------------------------- #


# NOTE on the B>1 rows: the PLAIN reference's general-SumLayer backward
# writes param flows at wrong offsets for non-power-of-2 batches at bs>=8
# shapes (pre-existing bug, present on main — canonical extraction reads
# zeros while the raw buffer holds the mass). So plain-referenced rows use
# B the reference handles (B=8 at bs=8; non-pow2 B=6 works at bs=4 via the
# SPARSE-mode kernel). Non-pow2-B pflow coverage for the sparse chain lives
# in test_param_flow_batched_matches_loop below, which needs no plain build.
@pytest.mark.parametrize("T,H,V,bs,density,B", [
    (4, 8, 4, 4, 0.6, 1),
    (5, 16, 8, 8, 0.5, 1),
    (6, 32, 12, 8, 0.4, 1),
    (4, 8, 4, 4, 0.6, 6),
    (5, 16, 8, 8, 0.5, 8),
])
def test_param_flow_matches_plain(T, H, V, bs, density, B):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=density, seed=29,
    )

    torch.manual_seed(101)
    root_plain, ns_sum_plain = _build_plain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(101)
    root_chain, ns_sum_chain = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_chain = juice.TensorCircuit(root_chain, verbose=False).to(device)
    pc_chain.params.data.copy_(pc_plain.params.data)

    assert _has_sparse_io_sum(pc_chain), (
        "expected SparseIOSumLayer on the block-dense sparse-chain build"
    )

    torch.manual_seed(0)
    data = torch.randint(0, V, (B, T), device=device)

    ll_plain = pc_plain(data)
    ll_chain = pc_chain(data)
    torch.testing.assert_close(ll_chain, ll_plain, rtol=1e-3, atol=5e-3)

    pc_plain.backward(data, compute_param_flows=True, flows_memory=0.0)
    pc_chain.backward(data, compute_param_flows=True, flows_memory=0.0)

    # Canonical per-edge layout via update_param_flows; the dense baseline
    # uses plain SumLayer compile path, the chain uses DenseSumLayer path —
    # both produce the same edge ordering for block-dense ``summate``. At
    # B>1 both sides accumulate the batch sum into the same flat buffer.
    ns_sum_plain.update_param_flows(pc_plain.param_flows, origin_ns_only=True)
    ns_sum_chain.update_param_flows(pc_chain.param_flows, origin_ns_only=True)

    pf_plain = ns_sum_plain._param_flows.detach()
    pf_chain = ns_sum_chain._param_flows.detach()

    abs_diff = (pf_chain - pf_plain).abs().max().item()
    rel_scale = max(pf_plain.abs().max().item(), 1e-6)
    assert abs_diff / rel_scale < 5e-3, (
        f"param_flow mismatch (T={T}, H={H}, V={V}, bs={bs}): "
        f"max abs diff = {abs_diff:.3e}, scale = {rel_scale:.3e}"
    )


# ---------------------------------------------------------------------- #
# Dense-CSC variant — direct cousin of sparse_categorical_cond_test
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("T,H,V,bs,B", [
    (5, 16, 8, 8, 1), (6, 32, 10, 8, 1), (5, 16, 8, 8, 8),
])
def test_param_flow_dense_csc_matches_plain(T, H, V, bs, B):
    """Same shape as ``sparse_categorical_cond_test`` (dense CSC pattern) so
    every ``(latent, category)`` slot is active. Verifies the param-flow
    backward holds when the sparse-IO path degenerates to full coverage."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _dense_csc_pattern(H=H, V=V, seed=11)

    torch.manual_seed(101)
    root_plain, ns_sum_plain = _build_plain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(101)
    root_chain, ns_sum_chain = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_chain = juice.TensorCircuit(root_chain, verbose=False).to(device)
    pc_chain.params.data.copy_(pc_plain.params.data)

    assert _has_sparse_io_sum(pc_chain)

    torch.manual_seed(0)
    data = torch.randint(0, V, (B, T), device=device)

    ll_plain = pc_plain(data)
    ll_chain = pc_chain(data)
    torch.testing.assert_close(ll_chain, ll_plain, rtol=1e-3, atol=5e-3)

    pc_plain.backward(data, compute_param_flows=True, flows_memory=0.0)
    pc_chain.backward(data, compute_param_flows=True, flows_memory=0.0)

    ns_sum_plain.update_param_flows(pc_plain.param_flows, origin_ns_only=True)
    ns_sum_chain.update_param_flows(pc_chain.param_flows, origin_ns_only=True)

    pf_plain = ns_sum_plain._param_flows.detach()
    pf_chain = ns_sum_chain._param_flows.detach()

    abs_diff = (pf_chain - pf_plain).abs().max().item()
    rel_scale = max(pf_plain.abs().max().item(), 1e-6)
    assert abs_diff / rel_scale < 5e-3, (
        f"param_flow mismatch (T={T}, H={H}, V={V}, bs={bs}): "
        f"max abs diff = {abs_diff:.3e}, scale = {rel_scale:.3e}"
    )


# ---------------------------------------------------------------------- #
# Batched-vs-loop pflow self-consistency (no plain reference needed)
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("T,H,V,bs,density,B", [
    (5, 16, 8, 8, 0.5, 7),
    (6, 32, 12, 8, 0.4, 9),
])
def test_param_flow_batched_matches_loop(T, H, V, bs, density, B):
    """One batched backward must accumulate the same param flows as B
    single-sample backward calls summed on host. Covers non-power-of-2 B,
    where the plain reference's own B>1 pflow path is broken (see note on
    test_param_flow_matches_plain)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=density, seed=59,
    )

    torch.manual_seed(202)
    root_chain, ns_sum = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc = juice.TensorCircuit(root_chain, verbose=False).to(device)
    assert _has_sparse_io_sum(pc)

    torch.manual_seed(3)
    data = torch.randint(0, V, (B, T), device=device)

    # Loop of B single-sample passes, canonical pflows summed on host.
    pf_loop = None
    for b in range(B):
        pc(data[b:b + 1])
        pc.backward(data[b:b + 1], compute_param_flows=True, flows_memory=0.0)
        ns_sum.update_param_flows(pc.param_flows, origin_ns_only=True)
        pf_b = ns_sum._param_flows.detach().clone()
        pf_loop = pf_b if pf_loop is None else pf_loop + pf_b

    # One batched pass.
    pc(data)
    pc.backward(data, compute_param_flows=True, flows_memory=0.0)
    ns_sum.update_param_flows(pc.param_flows, origin_ns_only=True)
    pf_batched = ns_sum._param_flows.detach()

    torch.testing.assert_close(pf_batched, pf_loop, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------- #
# One EM step end-to-end
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("T,H,V,bs,density,B", [
    (6, 16, 8, 8, 0.5, 1),
    (8, 32, 10, 8, 0.4, 1),
    (6, 16, 8, 8, 0.5, 8),
])
def test_em_step_matches_plain(T, H, V, bs, density, B):
    """Drive one full EM update and confirm both paths land on virtually
    identical params (and equivalent post-EM LLs on a held-out sequence)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=density, seed=83,
    )

    torch.manual_seed(2024)
    root_plain, _ = _build_plain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(2024)
    root_chain, _ = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_chain = juice.TensorCircuit(root_chain, verbose=False).to(device)
    pc_chain.params.data.copy_(pc_plain.params.data)

    # Pseudocount > 0 — the sparse fast path emits exact-zero pflows for
    # latents that never emit any observed token across T steps, which
    # would divide-by-zero in the normalizer otherwise. See the analogous
    # note in sparse_io_block_diagonal_param_flow_test.
    torch.manual_seed(5)
    train = torch.randint(0, V, (B, T), device=device)

    opt_plain = juice.optim.CircuitOptimizer(
        pc_plain, base_optimizer=None, lr=1.0, pseudocount=0.01,
    )
    opt_chain = juice.optim.CircuitOptimizer(
        pc_chain, base_optimizer=None, lr=1.0, pseudocount=0.01,
    )
    opt_plain.zero_grad()
    opt_chain.zero_grad()

    pc_plain(train)
    pc_chain(train)
    pc_plain.backward(train, compute_param_flows=True, flows_memory=0.0)
    pc_chain.backward(train, compute_param_flows=True, flows_memory=0.0)

    opt_plain.step()
    opt_chain.step()

    diff = (pc_plain.params - pc_chain.params).abs().max().item()
    assert diff < 5e-3, (
        f"post-EM params mismatch (T={T}, H={H}): max abs diff = {diff:.3e}"
    )

    # LL parity on a held-out sequence — loose tolerance because HMM LL
    # amplifies tiny per-param fp32 noise through T transitions.
    torch.manual_seed(6)
    test_seq = torch.randint(0, V, (B, T), device=device)
    ll_p = pc_plain(test_seq).detach()
    ll_c = pc_chain(test_seq).detach()
    ll_diff = (ll_c - ll_p).abs().max().item()
    assert ll_diff < max(1.0, 0.1 * T), (
        f"post-EM LL mismatch (T={T}, H={H}): max abs LL diff = {ll_diff:.3e}"
    )
