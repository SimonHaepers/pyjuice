"""
Correctness tests for the sparse Monarch transition pair —
:class:`SparseInputBlockDiagonalSumLayer` (sparse-in / dense-out BD₁) and
:class:`SparseOutputBlockDiagonalSumLayer` (dense-in / sparse-out BD₂) —
over :class:`SparseCategorical` emissions.

Topology under test (per timestep, built backward like the HMM chain):

    sparse_multiply(emission, BD₂ₜ₊₁)     -> CoSparseProdLayer (packed sv)
      -> summate(edge_ids=BD)             -> SparseInputBlockDiagonalSumLayer
      -> multiply(edge_ids=perm)          -> plain ProdLayer (permutation)
      -> summate(edge_ids=BD)             -> SparseOutputBlockDiagonalSumLayer

BD₁'s child is sparse but its consumer (the permutation product) is
dense → sparse-in/dense-out. BD₂'s child (the permutation) is dense but
its sole consumer is the next emission's SparseProdNodes, whose CSC
column defines which BD₂ outputs are ever read → dense-in/sparse-out,
and the consumer compiles to :class:`CoSparseProdLayer`.

Both layers accumulate their own param flows, so the param-flow tests
compare BD₁ *and* BD₂ against the plain build.
``test_param_flow_batched_matches_loop`` keeps
``allow_modify_flows=False`` so both kernel branches of BD₁ stay
covered; ``test_modify_flows_and_pflows`` exercises BD₁'s flag handling
in isolation.
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
    BlockDiagonalSumLayer, SparseInputBlockDiagonalSumLayer,
    SparseOutputBlockDiagonalSumLayer, CoSparseProdLayer, SparseProdLayer,
)


# ---------------------------------------------------------------------- #
# Helpers (CSC generator mirrors the sibling BD chain tests)
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


def _bd_edge_ids(NB: int) -> torch.Tensor:
    return torch.arange(0, NB)[None, :].repeat(2, 1)


def _monarch_permutation(H: int, permute_block_size: int) -> torch.Tensor:
    """Reshape-transpose-flatten permutation (same as the Monarch perf
    harness) — ``[H, 1]``, consumed with ``sparse_edges=True``."""
    return (
        torch.arange(0, H)
        .reshape(H // permute_block_size, permute_block_size)
        .permute(1, 0)
        .reshape(H)[:, None]
    )


def _build_monarch_chain(T: int, H: int, V: int, bs: int,
                         csc_indptr: torch.Tensor,
                         csc_indices: torch.Tensor,
                         csc_values: torch.Tensor,
                         force_plain: bool):
    """Homogeneous Monarch HMM over sparse emissions. With
    ``force_plain=True`` every layer is pinned to the plain
    SumLayer / ProdLayer baseline; otherwise the emission products build
    as :class:`SparseProdNodes` and each BD₁ compiles to
    :class:`SparseInputBlockDiagonalSumLayer` (BD₂ stays on the dense
    :class:`BlockDiagonalSumLayer`)."""
    NB = H // bs
    bd_edges = _bd_edge_ids(NB)
    perm = _monarch_permutation(H, permute_block_size=bs)
    with set_block_size(block_size=bs):
        ns_input = inputs(
            T - 1, num_node_blocks=NB,
            dist=dists.SparseCategorical(num_cats=V),
            csc_indptr=csc_indptr, csc_indices=csc_indices,
        )
        ns_input.set_params(csc_values, normalize=False)

        ns_bd1 = None
        ns_bd2 = None
        if force_plain:
            curr_zs = multiply(ns_input, _force_plain=True)
        else:
            curr_zs = ns_input
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_bd1 is None:
                bd1 = summate(curr_zs, edge_ids=bd_edges, block_size=bs,
                              _force_plain=force_plain)
                ns_bd1 = bd1
            else:
                bd1 = ns_bd1.duplicate(curr_zs, tie_params=True)
                if force_plain:
                    bd1._force_plain_layer = True
            np_perm = multiply(bd1, edge_ids=perm, sparse_edges=True,
                               _force_plain=force_plain)
            if ns_bd2 is None:
                bd2 = summate(np_perm, edge_ids=bd_edges, block_size=bs,
                              _force_plain=force_plain)
                ns_bd2 = bd2
            else:
                bd2 = ns_bd2.duplicate(np_perm, tie_params=True)
                if force_plain:
                    bd2._force_plain_layer = True
            if force_plain:
                curr_zs = multiply(curr_xs, bd2, _force_plain=True)
            else:
                curr_zs = sparse_multiply(curr_xs, bd2)
        if force_plain:
            root = summate(curr_zs, num_node_blocks=1, block_size=1,
                           _force_plain=True)
        else:
            root = sparse_summate(curr_zs, num_node_blocks=1, block_size=1)
    return root, ns_bd1, ns_bd2


def _sparse_in_bd_layers(pc: juice.TensorCircuit):
    return [
        layer
        for lg in pc.inner_layer_groups for layer in lg
        if isinstance(layer, SparseInputBlockDiagonalSumLayer)
    ]


def _make_pair(T, H, V, bs, density, csc_seed, build_seed, device):
    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=density, seed=csc_seed,
    )
    torch.manual_seed(build_seed)
    root_plain, bd1_plain, bd2_plain = _build_monarch_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, force_plain=True,
    )
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)

    torch.manual_seed(build_seed)
    root_sp, bd1_sp, bd2_sp = _build_monarch_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, force_plain=False,
    )
    pc_sp = juice.TensorCircuit(root_sp, verbose=False).to(device)
    pc_sp.params.data.copy_(pc_plain.params.data)
    return (pc_plain, bd1_plain, bd2_plain), (pc_sp, bd1_sp, bd2_sp)


# ---------------------------------------------------------------------- #
# Layer selection + forward parity
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("T,H,V,bs,B", [
    (3, 16, 6, 4, 1),
    (5, 16, 8, 4, 1),
    (6, 64, 12, 8, 1),
    (5, 16, 8, 4, 6),
    (6, 64, 12, 8, 8),
])
def test_forward_matches_plain(T, H, V, bs, B):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    (pc_plain, _, _), (pc_sp, _, _) = _make_pair(
        T, H, V, bs, density=0.5, csc_seed=17, build_seed=123, device=device,
    )

    sparse_bd_layers = _sparse_in_bd_layers(pc_sp)
    assert len(sparse_bd_layers) > 0, (
        "expected SparseInputBlockDiagonalSumLayer on the Monarch sparse "
        "build (BD₁ over a SparseProdLayer child, dense consumer)"
    )
    # The upstream sparse prods must have been marked skip-scatter (their
    # only consumer is the new layer type).
    for layer in sparse_bd_layers:
        for sp, _ in layer._sparse_input_refs:
            assert sp._skip_scatter, (
                "upstream SparseProdLayer of a "
                "SparseInputBlockDiagonalSumLayer should be skip-scatter"
            )
    # BD₂ must route to the dense-in/sparse-out layer, its consumers to
    # CoSparseProdLayer, and no dense BlockDiagonalSumLayer should remain.
    has_sparse_out_bd = any(
        isinstance(layer, SparseOutputBlockDiagonalSumLayer)
        for lg in pc_sp.inner_layer_groups for layer in lg
    )
    assert has_sparse_out_bd, (
        "expected SparseOutputBlockDiagonalSumLayer for BD₂"
    )
    has_cosparse = any(
        isinstance(layer, CoSparseProdLayer)
        for lg in pc_sp.inner_layer_groups for layer in lg
    )
    assert has_cosparse, "expected CoSparseProdLayer emission products"
    has_dense_bd = any(
        type(layer) is BlockDiagonalSumLayer
        for lg in pc_sp.inner_layer_groups for layer in lg
    )
    assert not has_dense_bd, (
        "no dense BlockDiagonalSumLayer should remain in the sparse "
        "Monarch build"
    )

    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        data = torch.randint(0, V, (B, T), device=device)
        lls_plain = pc_plain(data)
        lls_sp = pc_sp(data)
        torch.testing.assert_close(lls_sp, lls_plain, rtol=1e-3, atol=5e-3)


# ---------------------------------------------------------------------- #
# Backward element-flow parity
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("T,H,V,bs", [(4, 16, 6, 4), (6, 32, 10, 8)])
def test_backward_matches_plain(T, H, V, bs):
    """The sv_flow the new layer writes into each upstream
    :class:`SparseProdLayer` must equal the plain build's dense
    ``node_flows`` at the matching active CSC rows of the emission."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    (pc_plain, _, _), (pc_sp, _, _) = _make_pair(
        T, H, V, bs, density=0.5, csc_seed=31, build_seed=7, device=device,
    )
    assert len(_sparse_in_bd_layers(pc_sp)) > 0

    torch.manual_seed(0)
    data = torch.randint(0, V, (1, T), device=device)

    lls_plain = pc_plain(data)
    lls_sp = pc_sp(data)
    torch.testing.assert_close(lls_sp, lls_plain, rtol=1e-3, atol=5e-3)

    # Default flags (allow_modify_flows=True) — exercises the BD modify
    # pre-pass through both BD₁ (sparse) and BD₂ (dense).
    pc_plain.backward(data, compute_param_flows=False, flows_memory=0.0)
    pc_sp.backward(data, compute_param_flows=False, flows_memory=0.0)

    checked = 0
    for in_ns_plain, in_ns_sp in zip(
        pc_plain.input_layer_group[0].nodes,
        pc_sp.input_layer_group[0].nodes,
    ):
        lo_p, _ = in_ns_plain._output_ind_range
        owner = getattr(in_ns_sp, "_sparse_flow_owner", None)
        if owner is None:
            continue
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
        checked += 1
    assert checked > 0, "no sparse-flow-owning emission ns found to compare"


# ---------------------------------------------------------------------- #
# Param-flow parity (BD₁ + emissions)
# ---------------------------------------------------------------------- #


# NOTE on B>1 rows: the plain reference's general-SumLayer backward writes
# pflows at wrong offsets for non-pow2 B at bs >= 8 (pre-existing bug, see
# the same note in sparse_io_sum_param_flow_test) — so plain-referenced
# rows use B the reference handles. Non-pow2-B coverage lives in
# test_param_flow_batched_matches_loop below (no plain build involved).
@pytest.mark.parametrize("T,H,V,bs,B", [
    (4, 16, 6, 4, 1),
    (6, 32, 10, 8, 1),
    (4, 16, 6, 4, 6),
    (6, 32, 10, 8, 8),
])
def test_param_flow_matches_plain(T, H, V, bs, B):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    (pc_plain, bd1_plain, bd2_plain), (pc_sp, bd1_sp, bd2_sp) = _make_pair(
        T, H, V, bs, density=0.6, csc_seed=29, build_seed=101, device=device,
    )
    assert len(_sparse_in_bd_layers(pc_sp)) > 0

    torch.manual_seed(0)
    data = torch.randint(0, V, (B, T), device=device)

    ll_plain = pc_plain(data)
    ll_sp = pc_sp(data)
    torch.testing.assert_close(ll_sp, ll_plain, rtol=1e-3, atol=5e-3)

    pc_plain.backward(data, compute_param_flows=True, flows_memory=0.0)
    pc_sp.backward(data, compute_param_flows=True, flows_memory=0.0)

    # BD₁'s and BD₂'s own param flows, pulled through each circuit's
    # canonical ``_param_flow_ids`` layout (tied duplicates accumulate
    # into the source's slice on both sides).
    for name, ns_plain, ns_sp in (
        ("BD₁", bd1_plain, bd1_sp),
        ("BD₂", bd2_plain, bd2_sp),
    ):
        ns_plain.update_param_flows(pc_plain.param_flows, origin_ns_only=True)
        ns_sp.update_param_flows(pc_sp.param_flows, origin_ns_only=True)
        pf_plain = ns_plain._param_flows.detach()
        pf_sp = ns_sp._param_flows.detach()
        abs_diff = (pf_sp - pf_plain).abs().max().item()
        rel_scale = max(pf_plain.abs().max().item(), 1e-6)
        assert abs_diff / rel_scale < 5e-3, (
            f"{name} param_flow mismatch (T={T}, H={H}, V={V}, bs={bs}, "
            f"B={B}): max abs diff = {abs_diff:.3e}, scale = {rel_scale:.3e}"
        )


# ---------------------------------------------------------------------- #
# Batched-vs-loop self-consistency (covers non-pow2 B, no plain build)
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("T,H,V,bs,B", [
    (4, 16, 6, 4, 7),
    (6, 32, 10, 8, 9),
])
def test_param_flow_batched_matches_loop(T, H, V, bs, B):
    """One batched backward on the sparse Monarch chain must accumulate the
    same BD₁ param flows (and emission pflows) as B single-sample passes
    summed on host. Covers non-power-of-2 B, where the plain reference's
    own B>1 pflow path is broken (see note on
    test_param_flow_matches_plain)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.5, seed=59,
    )
    torch.manual_seed(202)
    root, bd1, _ = _build_monarch_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values, force_plain=False,
    )
    pc = juice.TensorCircuit(root, verbose=False).to(device)
    assert len(_sparse_in_bd_layers(pc)) > 0

    torch.manual_seed(3)
    data = torch.randint(0, V, (B, T), device=device)

    # Loop of B single-sample passes, canonical BD₁ pflows summed on host.
    pf_loop = None
    lls_loop = []
    for b in range(B):
        lls_loop.append(pc(data[b:b + 1]).detach().clone())
        pc.backward(data[b:b + 1], compute_param_flows=True,
                    flows_memory=0.0, allow_modify_flows=False)
        bd1.update_param_flows(pc.param_flows, origin_ns_only=True)
        pf_b = bd1._param_flows.detach().clone()
        pf_loop = pf_b if pf_loop is None else pf_loop + pf_b

    # One batched pass.
    lls_batched = pc(data)
    torch.testing.assert_close(
        lls_batched.flatten(), torch.cat([l.flatten() for l in lls_loop]),
        rtol=1e-4, atol=1e-5,
    )
    pc.backward(data, compute_param_flows=True, flows_memory=0.0,
                allow_modify_flows=False)
    bd1.update_param_flows(pc.param_flows, origin_ns_only=True)
    pf_batched = bd1._param_flows.detach()

    torch.testing.assert_close(pf_batched, pf_loop, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------- #
# allow_modify_flows=True on a BD₂-free DAG
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("H,V,bs,B", [
    (16, 6, 4, 1),
    (32, 10, 8, 1),
    (16, 6, 4, 6),
    (32, 10, 8, 8),
])
def test_modify_flows_and_pflows(H, V, bs, B):
    """Single sparse-in BD transition, no dense BD₂ anywhere — every layer
    in both builds supports ``allow_modify_flows=True``, so the default
    backward path exercises the new layer's modify-flow pre-pass."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")

    NB = H // bs
    bd_edges = _bd_edge_ids(NB)
    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=0.5, seed=53,
    )

    def _build(force_plain: bool):
        with set_block_size(block_size=bs):
            em = inputs(
                0, num_node_blocks=NB,
                dist=dists.SparseCategorical(num_cats=V),
                csc_indptr=csc_indptr, csc_indices=csc_indices,
            )
            em.set_params(csc_values, normalize=False)
            st = inputs(1, num_node_blocks=NB,
                        dist=dists.Categorical(num_cats=V))
            # ``sparse_multiply`` needs a non-input dense sibling; a plain
            # block-dense mixer keeps every non-BD₁ layer modify-flow-safe.
            st_sum = summate(st, num_node_blocks=NB,
                             _force_plain=force_plain)
            if force_plain:
                prod = multiply(em, st_sum, _force_plain=True)
            else:
                prod = sparse_multiply(em, st_sum)
            bd1 = summate(prod, edge_ids=bd_edges, block_size=bs,
                          _force_plain=force_plain)
            wrap = multiply(bd1, _force_plain=force_plain)
            root = summate(wrap, num_node_blocks=1, block_size=1,
                           _force_plain=force_plain)
        return root, bd1

    torch.manual_seed(11)
    root_plain, bd1_plain = _build(force_plain=True)
    pc_plain = juice.TensorCircuit(root_plain, verbose=False).to(device)
    torch.manual_seed(11)
    root_sp, bd1_sp = _build(force_plain=False)
    pc_sp = juice.TensorCircuit(root_sp, verbose=False).to(device)
    pc_sp.params.data.copy_(pc_plain.params.data)

    assert len(_sparse_in_bd_layers(pc_sp)) == 1, (
        "expected exactly one SparseInputBlockDiagonalSumLayer"
    )

    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        data = torch.randint(0, V, (B, 2), device=device)
        ll_plain = pc_plain(data)
        ll_sp = pc_sp(data)
        torch.testing.assert_close(ll_sp, ll_plain, rtol=1e-3, atol=5e-3)

        # Default backward → allow_modify_flows=True on both sides.
        pc_plain.backward(data, compute_param_flows=True, flows_memory=0.0)
        pc_sp.backward(data, compute_param_flows=True, flows_memory=0.0)

        bd1_plain.update_param_flows(pc_plain.param_flows, origin_ns_only=True)
        bd1_sp.update_param_flows(pc_sp.param_flows, origin_ns_only=True)
        pf_plain = bd1_plain._param_flows.detach()
        pf_sp = bd1_sp._param_flows.detach()
        abs_diff = (pf_sp - pf_plain).abs().max().item()
        rel_scale = max(pf_plain.abs().max().item(), 1e-6)
        assert abs_diff / rel_scale < 5e-3, (
            f"param_flow mismatch under allow_modify_flows=True "
            f"(H={H}, bs={bs}, seed={seed}): max abs diff = {abs_diff:.3e}, "
            f"scale = {rel_scale:.3e}"
        )
