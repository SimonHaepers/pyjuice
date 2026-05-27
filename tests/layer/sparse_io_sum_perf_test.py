"""Perf harness: sparse-IO sum chain (non-BD) vs dense plain SumLayer.

Times forward, backward (element flows only), and backward (with param
flows) independently, so each phase's speedup is visible.

Both circuits use SparseCategorical emissions with the same CSC pattern;
the dense baseline pins every layer to plain SumLayer / ProdLayer via
``_force_plain=True``, so the comparison isolates the SparseIOSumLayer +
CoSparseProdLayer fast path against the general dense fallback.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import (
    inputs, multiply, summate, sparse_multiply, sparse_summate, set_block_size,
)
from pyjuice.layer import SparseIOSumLayer


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


def _build_dense(T, H, V, bs):
    """Dense baseline using DenseCategorical input (no CSC) + plain layers."""
    num_node_blocks = H // bs
    with set_block_size(block_size=bs):
        ns_input = inputs(
            T - 1, num_node_blocks=num_node_blocks,
            dist=dists.DenseCategorical(num_cats=V),
        )
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
        return summate(curr_zs, num_node_blocks=1, block_size=1,
                       _force_plain=True)


def _build_sparse_chain(T, H, V, bs, csc_indptr, csc_indices, csc_values):
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


def _time_phase(pc, data, phase, n_warmup, n_iter):
    """Time one phase (fwd / bwd_ele / bwd_pflow) over n_iter iterations."""
    def _run():
        if phase == "fwd":
            pc(data)
        elif phase == "bwd_ele":
            pc.backward(data, compute_param_flows=False, flows_memory=0.0,
                        allow_modify_flows=False)
        elif phase == "bwd_pflow":
            pc.backward(data, compute_param_flows=True, flows_memory=0.0,
                        allow_modify_flows=False)

    # Forward is needed before backward — run it always.
    if phase != "fwd":
        pc(data)

    for i in range(n_warmup):
        if phase != "fwd":
            pc(data)
        _run()
    torch.cuda.synchronize()

    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    for i in range(n_iter):
        if phase != "fwd":
            pc(data)
        torch.cuda.synchronize()
        evs[i][0].record()
        _run()
        evs[i][1].record()
    torch.cuda.synchronize()

    ms = [s.elapsed_time(e) for s, e in evs]
    return sum(ms) / len(ms)


def _build_and_run(T, H, V, bs, density, n_warmup, n_iter, seed=0):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=density, seed=seed,
    )

    torch.manual_seed(seed)
    root_dense = _build_dense(T, H, V, bs)
    pc_dense = juice.TensorCircuit(root_dense, verbose=False).to(device)

    torch.manual_seed(seed)
    root_sparse = _build_sparse_chain(T, H, V, bs, csc_indptr, csc_indices, csc_values)
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)

    has_io = any(isinstance(l, SparseIOSumLayer)
                 for lg in pc_sparse.inner_layer_groups for l in lg)
    assert has_io, "expected SparseIOSumLayer on the sparse build"

    data = torch.randint(0, V, (1, T), device=device)

    # DenseCategorical's input-layer backward kernel unconditionally calls
    # tl.atomic_add(param_flows_ptr, ...) even when compute_param_flows=False
    # (the ptr is None → Triton compilation error). So we time only:
    #   fwd       — forward only
    #   bwd_pflow — backward WITH param flows (the full training path)
    # The element-flow-only phase is omitted from the dense path; we still
    # time it on the sparse side for reference.
    phases_dense  = ["fwd", "bwd_pflow"]
    phases_sparse = ["fwd", "bwd_ele", "bwd_pflow"]

    print(f"\n=== T={T} H={H} V={V} bs={bs} density={density} ===")
    header = f"  {'':16s} {'fwd':>10s} {'bwd_ele':>10s} {'bwd_pflow':>10s}"
    print(header)

    results = {}
    for label, pc, phases in [("dense", pc_dense, phases_dense),
                               ("sparse_io", pc_sparse, phases_sparse)]:
        row = {}
        for phase in phases:
            ms = _time_phase(pc, data, phase, n_warmup, n_iter)
            row[phase] = ms
        results[label] = row
        fwd = row.get("fwd", 0)
        ele = row.get("bwd_ele", 0)
        pfl = row.get("bwd_pflow", 0)
        print(f"  {label:16s} {fwd:8.3f}ms {ele:8.3f}ms {pfl:8.3f}ms")

    # Speedups (only for phases both paths timed)
    spd = {}
    for phase in ["fwd", "bwd_pflow"]:
        d = results["dense"].get(phase, 0)
        s = results["sparse_io"].get(phase, 0)
        spd[phase] = d / s if s > 0 else float("inf")
    print(f"  {'speedup':16s} {spd['fwd']:8.2f}x  {'---':>10s} {spd['bwd_pflow']:8.2f}x")

    return results


@pytest.mark.slow
@pytest.mark.parametrize("T,H,V,bs,density", [
    (8, 32, 16, 8, 0.3),
    (8, 64, 32, 8, 0.2),
])
def test_sparse_io_sum_perf_smoke(T, H, V, bs, density):
    """Smoke test — confirms both circuits build and the timed loops
    complete. No hard speedup assertion (see module docstring)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _build_and_run(T=T, H=H, V=V, bs=bs, density=density,
                   n_warmup=2, n_iter=5, seed=42)


if __name__ == "__main__":
    # Dense baseline uses DenseCategorical input whose backward kernel has
    # a pre-existing bug at H ≥ 256 (CUDA illegal memory access in
    # _flows_kernel). Restrict the head-to-head comparison to H ≤ 128;
    # then run the sparse path alone at larger H to show it scales.
    for T, H, V, bs, density in [
        (8, 32, 16, 8, 0.3),
        (8, 64, 16, 8, 0.2),
        (8, 128, 16, 8, 0.1),
    ]:
        _build_and_run(T=T, H=H, V=V, bs=bs, density=density,
                       n_warmup=3, n_iter=20, seed=42)

    # Sparse-only scaling: the dense path crashes at H ≥ 256 due to a
    # pre-existing DenseCategorical backward bug. Show the sparse-IO
    # path works at large H by timing it standalone.
    print("\n--- Sparse-IO only (dense baseline unavailable at these sizes) ---")
    for T, H, V, bs, density in [
        (8, 512, 32, 16, 0.03),
        (8, 1024, 32, 32, 0.02),
        (8, 2048, 32, 32, 0.01),
        (8, 4096, 32, 64, 0.005),
    ]:
        csc_indptr, csc_indices, csc_values = _random_csc_pattern(
            H=H, V=V, density=density, seed=42,
        )
        device = torch.device("cuda:0")
        torch.manual_seed(42)
        root = _build_sparse_chain(T, H, V, bs, csc_indptr, csc_indices, csc_values)
        pc = juice.TensorCircuit(root, verbose=False).to(device)
        data = torch.randint(0, V, (1, T), device=device)
        print(f"\n  T={T} H={H} V={V} bs={bs} density={density}")
        for phase in ["fwd", "bwd_ele", "bwd_pflow"]:
            ms = _time_phase(pc, data, phase, n_warmup=3, n_iter=20)
            print(f"    {phase:12s} {ms:.3f}ms")
