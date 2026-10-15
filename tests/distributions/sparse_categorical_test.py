import math

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, summate


LOG_EPS = -23.0258509299  # log(1e-10) — matches kernel's "position not in sparsity pattern"


def _make_csc_pattern(H, V, density = 0.5, seed = 42):
    """Build a random H x V categorical matrix in CSC form. Each row has >= 1 nonzero."""
    g = torch.Generator().manual_seed(seed)
    mask_hv = (torch.rand(H, V, generator = g) < density)
    for h in range(H):
        if not mask_hv[h].any():
            mask_hv[h, torch.randint(0, V, (1,), generator = g).item()] = True

    dense_probs = torch.rand(H, V, generator = g) * mask_hv.float()
    dense_probs = dense_probs / dense_probs.sum(dim = 1, keepdim = True)

    csc_indptr = torch.zeros(V + 1, dtype = torch.long)
    csc_indices_list = []
    csc_values_list = []
    for v in range(V):
        col_rows = torch.where(mask_hv[:, v])[0]
        csc_indices_list.extend(col_rows.tolist())
        csc_values_list.extend(dense_probs[col_rows, v].tolist())
        csc_indptr[v + 1] = len(csc_indices_list)

    csc_indices = torch.tensor(csc_indices_list, dtype = torch.long)
    csc_values = torch.tensor(csc_values_list, dtype = torch.float32)

    return mask_hv, dense_probs, csc_indptr, csc_indices, csc_values


def _build_sparse_pc(H, V, csc_indptr, csc_indices, csc_values, device):
    """Build a minimal PC: SparseCategorical input on var 0 → summate root."""
    with juice.set_block_size(block_size = H):
        sparse_dist = dists.SparseCategorical(num_cats = V)
        ni = inputs(
            0, num_node_blocks = 1, dist = sparse_dist,
            csc_indptr = csc_indptr, csc_indices = csc_indices,
        )
        # Values are in CSC order (matching the user-supplied csc_indices).
        ni.set_params(csc_values, normalize = False)
        root = summate(ni, num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 0.0)

    pc = juice.TensorCircuit(root)
    pc.to(device)
    return pc, ni, root


def test_sparse_categorical_forward():
    """Per-latent emission log-probs for each observed token.

    The compiled circuit routes the input through the sparse fast path
    (SparseProdLayer wrap with ``_skip_scatter``), which bypasses
    ``node_mars`` — the per-active-row values live in the layer's packed
    ``SparseNodeValues`` container, reachable from the input ns via
    ``_sparse_flow_owner``. With no dense siblings in the 1-child wrap, the
    container values are exactly ``log P(v | h)`` at the active rows.
    """
    device = torch.device("cuda:0")
    H, V = 8, 16
    mask_hv, dense_probs, csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V)

    pc, ni, _ = _build_sparse_pc(H, V, csc_indptr, csc_indices, csc_values, device)

    batch_size = 32
    torch.manual_seed(0)
    data = torch.randint(0, V, (batch_size, 1)).to(device)
    data_cpu = data.cpu()

    pc(data)

    owner = getattr(ni, "_sparse_flow_owner", None)
    assert owner is not None, (
        "expected the input ns to be claimed by a SparseProdLayer wrap"
    )
    layer, ns_idx = owner
    sv = layer._sparse_outputs[ns_idx]
    assert sv is not None and sv.is_batched and sv.batch_size == batch_size

    dense_log = torch.where(mask_hv, dense_probs.log(), torch.full_like(dense_probs, LOG_EPS))
    col_starts_cpu = sv.col_starts.cpu().tolist()
    values_cpu = sv.values.detach().cpu()
    indices_cpu = sv.indices.cpu()
    for b in range(batch_size):
        v = int(data_cpu[b, 0])
        k_b = sv.nnz_list[b]
        active = indices_cpu[col_starts_cpu[b]:col_starts_cpu[b] + k_b]
        expected = dense_log[active, v]
        assert torch.allclose(values_cpu[b, :k_b], expected, atol = 1e-4, rtol = 1e-4), \
            f"sample {b}: max diff = {(values_cpu[b, :k_b] - expected).abs().max().item():.3e}"


def test_sparse_categorical_backward_flows():
    """Backward accumulates expected posterior mass into the CSC slots."""
    device = torch.device("cuda:0")
    H, V = 8, 16
    mask_hv, dense_probs, csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, seed = 7)

    pc, ni, _ = _build_sparse_pc(H, V, csc_indptr, csc_indices, csc_values, device)

    batch_size = 16
    torch.manual_seed(1)
    data = torch.randint(0, V, (batch_size, 1)).to(device)
    data_cpu = data.cpu()

    pc(data)
    pc.backward(data)

    il = pc.input_layer_group[0]
    par_start, par_end = ni._param_range
    pf = il.param_flows[par_start:par_end].detach().cpu()  # [nnz], CSC order

    # Posterior: with uniform root weights, p(n | v) = p(v | n) / sum_n' p(v | n').
    dense_probs_safe = torch.where(mask_hv, dense_probs, torch.full_like(dense_probs, 1e-10))
    priors = dense_probs_safe / H
    posteriors = priors / priors.sum(dim = 0, keepdim = True)     # [H, V]
    expected_dense_flow = torch.zeros(H, V)
    for b in range(batch_size):
        v = int(data_cpu[b, 0].item())
        expected_dense_flow[:, v] += posteriors[:, v]

    # Read expected flow at each CSC slot, in the same order as pf.
    expected_csc = expected_dense_flow[csc_indices, torch.repeat_interleave(
        torch.arange(V), torch.diff(csc_indptr),
    )]

    assert torch.allclose(pf, expected_csc, atol = 1e-4, rtol = 1e-3), \
        f"max flow diff = {(pf - expected_csc).abs().max().item():.3e}"


def test_sparse_categorical_em_renormalizes():
    """After an EM step, values still sum to 1 within each row."""
    device = torch.device("cuda:0")
    H, V = 8, 16
    _, _, csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, seed = 11)

    pc, ni, _ = _build_sparse_pc(H, V, csc_indptr, csc_indices, csc_values, device)

    batch_size = 32
    torch.manual_seed(2)
    data = torch.randint(0, V, (batch_size, 1)).to(device)

    pc(data)
    pc.backward(data)
    pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1, keep_zero_params = False)

    il = pc.input_layer_group[0]
    par_start, par_end = ni._param_range
    csc_vals = il.params.data[par_start:par_end].detach().cpu()

    # Row sums over CSC-ordered values, grouped by row id.
    row_sums = torch.zeros(H)
    row_sums.scatter_add_(0, csc_indices, csc_vals)
    for h in range(H):
        assert math.isclose(row_sums[h].item(), 1.0, abs_tol = 1e-4), \
            f"row {h} sum is {row_sums[h].item()}, expected ~1"


def test_sparse_categorical_partition_fn():
    """Per-latent partition = log(sum_row_values) = log(1) = 0 after row-normalization."""
    device = torch.device("cuda:0")
    H, V = 8, 16
    _, _, csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, seed = 13)

    pc, ni, _ = _build_sparse_pc(H, V, csc_indptr, csc_indices, csc_values, device)

    # Trigger allocation of pc.node_mars via a dummy forward.
    data = torch.zeros(1, 1, dtype = torch.long, device = device)
    pc(data)

    il = pc.input_layer_group[0]
    node_mars = torch.zeros(pc.node_mars.size(0), device = device)
    il.eval_partition_fn(node_mars)
    sid, eid = ni._output_ind_range
    per_latent_log_Z = node_mars[sid:eid].detach().cpu()

    assert torch.allclose(per_latent_log_Z, torch.zeros_like(per_latent_log_Z), atol = 1e-4), \
        f"expected per-latent partition ~0, max abs = {per_latent_log_Z.abs().max().item():.3e}"


def test_build_sparse_pattern_batched():
    """Batched (B>1) pattern build: col_starts / nnz / nnz_list must match
    host indptr math, values must be a [B, K_stride] workspace view, and the
    per-query pattern_cache must dedup to identical tensors. The B=1 branch
    must keep returning the classic single-column container."""
    device = torch.device("cuda:0")
    H, V = 8, 16
    _, _, csc_indptr, csc_indices, csc_values = _make_csc_pattern(H, V, seed = 7)

    pc, ni, _ = _build_sparse_pc(H, V, csc_indptr, csc_indices, csc_values, device)
    dist = ni.dist

    B = 8
    torch.manual_seed(1)
    tokens = torch.randint(0, V, (B,))
    data_cpu = tokens.view(1, B)  # [num_vars, B] layout, var_id = 0

    ws = torch.empty(B * dist._max_nnz_per_col, dtype = torch.float32, device = device)
    cache = dict()
    sv = dist.build_sparse_pattern(
        data = data_cpu, var_id = 0, num_rows = H, device = device,
        values_out = ws, pattern_cache = cache,
    )

    assert sv.is_batched and sv.batch_size == B
    ref_starts = [int(csc_indptr[v]) for v in tokens.tolist()]
    ref_nnz = [int(csc_indptr[v + 1] - csc_indptr[v]) for v in tokens.tolist()]
    assert sv.col_starts.cpu().tolist() == ref_starts
    assert sv.nnz.cpu().tolist() == ref_nnz
    assert sv.nnz_list == ref_nnz
    assert sv.total_nnz == max(ref_nnz)
    assert sv.values.shape == (B, dist._max_nnz_per_col)
    # indices is the FULL csc_indices tensor (base for col_starts addressing).
    assert sv.indices.data_ptr() == dist._csc_indices.data_ptr()
    assert sv.indices.numel() == dist._csc_indices.numel()

    # Cache hit returns the identical device tensors (no rebuild / re-H2D).
    sv2 = dist.build_sparse_pattern(
        data = data_cpu, var_id = 0, num_rows = H, device = device,
        values_out = ws, pattern_cache = cache,
    )
    assert sv2.col_starts is sv.col_starts and sv2.nnz is sv.nnz

    # like_pattern mirrors every batched field.
    flow = sv.like_pattern(torch.empty_like(sv.values))
    assert flow.is_batched and flow.col_starts is sv.col_starts \
        and flow.nnz_list == sv.nnz_list and flow.max_val is None

    # B=1 branch unchanged: single-column view container.
    sv1 = dist.build_sparse_pattern(
        data = data_cpu[:, :1], var_id = 0, num_rows = H, device = device,
    )
    assert not sv1.is_batched
    assert sv1.col_start == ref_starts[0] and sv1.total_nnz == ref_nnz[0]
    assert sv1.values.shape == (ref_nnz[0],)


if __name__ == "__main__":
    test_sparse_categorical_forward()
    test_sparse_categorical_backward_flows()
    test_sparse_categorical_em_renormalizes()
    test_sparse_categorical_partition_fn()
    test_build_sparse_pattern_batched()
    print("all sparse_categorical tests passed")
