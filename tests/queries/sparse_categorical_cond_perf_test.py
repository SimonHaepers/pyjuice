"""NVTX-instrumented perf harness comparing ``juice.queries.conditional(...)``
on the **dense** and **sparse** HMM pipelines from
``tests/layer/sparse_hmm_perf_test.py``, but now running end-to-end conditional
posteriors (per-token emission distribution) instead of just the inner
backward.

* Dense path: ``DenseCategorical`` input + ``DenseCategoricalInputLayer`` +
  ``DenseSumLayer``. The conditional backward dispatches to
  :meth:`DenseCategoricalInputLayer.dense_conditional_backward` (a single
  bmm/matmul over ``[V, K, C]`` params).
* Sparse path: ``SparseCategorical`` input with a low-density CSC emission
  pattern + ``SparseProdLayer`` + ``SparseInputSumLayer``. The conditional
  backward dispatches to :func:`_sparse_categorical_backward` — a CSC-per-column
  Triton scatter whose cost scales with ``nnz`` rather than ``H·V``.

Both paths use ``homogeneous=True`` (tied transitions + tied emissions) and
``B=1`` — required for the ``SparseInputSumLayer`` / ``SparseProdLayer`` fast
path.

Emits NVTX ranges around each ``conditional`` call for profiling with ``nsys``
and prints wall-clock per-iteration timings (CUDA events) so speedups are
readable off the console without opening a profiler.

Run standalone for profiling::

    pixi run -e dev nsys profile --trace=cuda,nvtx \\
        -o /tmp/cond_sparse_hmm python tests/queries/sparse_categorical_cond_perf_test.py

The ``test_cond_perf_smoke`` case is marked ``slow`` and is skipped unless
``--run-slow`` is passed.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, sparse_multiply, sparse_summate, set_block_size
from pyjuice.model import TensorCircuit


# ---------------------------------------------------------------------------
# DAG builders (adapted from tests/layer/sparse_hmm_perf_test.py)
# ---------------------------------------------------------------------------


def _make_csc_emissions(H: int, V: int, density: float, seed: int,
                         device: Optional[torch.device] = None):
    """Random CSC emission pattern + row-normalised probabilities.

    Oversampled direct (row, col) draws → dedupe via linearised ids → CSC sort
    by (col, row). Guarantees coverage by appending one random column per row
    so every latent has at least one active emission.
    """
    device = device or torch.device("cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    target_nnz = int(density * H * V * 1.05)
    rand_rows = torch.randint(0, H, (target_nnz,), generator=g, device=device)
    rand_cols = torch.randint(0, V, (target_nnz,), generator=g, device=device)

    all_rows_coverage = torch.arange(H, device=device)
    cov_cols = torch.randint(0, V, (H,), generator=g, device=device)
    rand_rows = torch.cat([rand_rows, all_rows_coverage])
    rand_cols = torch.cat([rand_cols, cov_cols])

    linear = rand_rows.to(torch.long) * V + rand_cols.to(torch.long)
    linear = torch.unique(linear)
    rows_dedup = linear // V
    cols_dedup = linear % V

    sort_key = cols_dedup.to(torch.long) * H + rows_dedup.to(torch.long)
    order = torch.argsort(sort_key)
    csc_indices = rows_dedup[order].contiguous()
    cols = cols_dedup[order]

    col_counts = torch.bincount(cols, minlength=V)
    csc_indptr = torch.zeros(V + 1, dtype=torch.long, device=device)
    csc_indptr[1:] = torch.cumsum(col_counts, dim=0)

    raw = torch.rand(csc_indices.numel(), generator=g, device=device)
    row_sums = torch.zeros(H, device=device)
    row_sums.scatter_add_(0, csc_indices, raw)
    csc_values = raw / row_sums[csc_indices]

    return (csc_indptr.cpu(), csc_indices.cpu(),
            csc_values.to(torch.float32).cpu())


def _build_dense_hmm_dag(T: int, H: int, V: int, bs: int):
    num_node_blocks = H // bs
    with set_block_size(block_size=bs):
        ns_input = inputs(
            T - 1, num_node_blocks=num_node_blocks,
            dist=dists.DenseCategorical(num_cats=V),
        )
        ns_sum = None
        curr_zs = ns_input
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_sum is None:
                ns = summate(curr_zs, num_node_blocks=num_node_blocks)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = multiply(curr_xs, ns)
        root = summate(curr_zs, num_node_blocks=1, block_size=1)
    return root


def _build_sparse_hmm_dag(T: int, H: int, V: int, bs: int,
                           csc_indptr: torch.Tensor,
                           csc_indices: torch.Tensor,
                           csc_values: torch.Tensor):
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
    return root


# ---------------------------------------------------------------------------
# Timing / NVTX helpers
# ---------------------------------------------------------------------------


def _time_conditional(pc: TensorCircuit, data: torch.Tensor,
                      target_vars, label: str,
                      n_warmup: int, n_iter: int,
                      warmup_data: Optional[torch.Tensor] = None) -> dict:
    """Warm up JIT, then run ``n_iter`` conditional() calls under NVTX ranges
    and return per-iter wall-clock stats.

    ``warmup_data`` should be a *different* token sequence than ``data`` so
    we exercise input variability — for sparse paths the per-token CSC
    column slice differs across sequences, and reusing the timing tokens
    during warmup risks priming caches/branches in a way the timed loop
    wouldn't see in practice.
    """
    if warmup_data is None:
        warmup_data = data
    torch.cuda.nvtx.range_push(f"{label}_warmup(n={n_warmup})")
    for i in range(n_warmup):
        torch.cuda.nvtx.range_push(f"{label}_warmup_{i}")
        juice.queries.conditional(pc, data=warmup_data, target_vars=target_vars)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]

    torch.cuda.nvtx.range_push(f"{label}_loop")
    for i in range(n_iter):
        torch.cuda.nvtx.range_push(f"{label}_iter_{i}")
        events[i][0].record()
        juice.queries.conditional(pc, data=data, target_vars=target_vars)
        events[i][1].record()
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    ms = [s.elapsed_time(e) for s, e in events]
    return {
        "ms": ms,
        "mean": sum(ms) / len(ms),
        "min": min(ms),
        "max": max(ms),
    }


def _print_summary(label: str, stats: dict) -> None:
    print(
        f"[{label}] cond: mean {stats['mean']:.3f} ms  "
        f"min {stats['min']:.3f} ms  "
        f"max {stats['max']:.3f} ms"
    )


# ---------------------------------------------------------------------------
# Tests / entry point
# ---------------------------------------------------------------------------


def _build_and_run(T: int, H: int, V: int, bs: int, density: float,
                   n_warmup: int, n_iter: int, seed: int = 0,
                   target_vars=None):
    assert torch.cuda.is_available(), "this perf test requires CUDA"
    device = torch.device("cuda:0")

    torch.cuda.nvtx.range_push("build_dense")
    root_dense = _build_dense_hmm_dag(T, H, V, bs)
    pc_dense = TensorCircuit(
        root_dense,
        use_dense_sum_layer=True,
        device=device,
        verbose=False,
    )
    torch.cuda.nvtx.range_pop()

    csc_indptr, csc_indices, csc_values = _make_csc_emissions(
        H=H, V=V, density=density, seed=seed, device=device,
    )
    torch.cuda.nvtx.range_push("build_sparse")
    root_sparse = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = TensorCircuit(
        root_sparse,
        use_dense_sum_layer=True,
        device=device,
        verbose=False,
    )
    torch.cuda.nvtx.range_pop()

    # Sanity-check the expected layer classes are actually in use.
    from pyjuice.layer import (
        DenseCategoricalInputLayer, DenseSumLayer, SparseProdLayer,
        SparseInputSumLayer,
    )
    dense_input_layers = [l for l in pc_dense.input_layer_group
                          if isinstance(l, DenseCategoricalInputLayer)]
    assert dense_input_layers, "dense path should compile DenseCategoricalInputLayer"
    sparse_prod_layers = [l for lg in pc_sparse.inner_layer_groups for l in lg
                          if isinstance(l, SparseProdLayer)]
    sparse_sum_layers = [l for lg in pc_sparse.inner_layer_groups for l in lg
                         if isinstance(l, SparseInputSumLayer)]
    assert sparse_prod_layers, "sparse path should compile SparseProdLayer"
    assert sparse_sum_layers, "sparse path should compile SparseInputSumLayer"

    print(
        f"\n=== cond perf: T={T}, H={H}, V={V}, bs={bs}, "
        f"density={density:.2%}, target_vars={'all' if target_vars is None else len(target_vars)} ===\n"
        f"  dense:  {len(dense_input_layers)} DenseCategoricalInputLayer(s)\n"
        f"  sparse: {len(sparse_prod_layers)} SparseProdLayer(s), "
        f"{len(sparse_sum_layers)} SparseInputSumLayer(s)"
    )

    # B=1 to exercise the sparse fast path. Distinct warmup vs. timing
    # sequences so warmup doesn't prime per-token caches/branches that the
    # timed loop wouldn't otherwise hit.
    g = torch.Generator(device=device).manual_seed(seed)
    warmup_data = torch.randint(0, V, (1, T), generator=g, device=device)
    data = torch.randint(0, V, (1, T), generator=g, device=device)

    stats_dense = _time_conditional(pc_dense, data, target_vars, "dense",
                                     n_warmup=n_warmup, n_iter=n_iter,
                                     warmup_data=warmup_data)
    stats_sparse = _time_conditional(pc_sparse, data, target_vars, "sparse",
                                      n_warmup=n_warmup, n_iter=n_iter,
                                      warmup_data=warmup_data)

    _print_summary("dense", stats_dense)
    _print_summary("sparse", stats_sparse)

    if stats_sparse["mean"] > 0:
        print(
            f"  speedup cond: {stats_dense['mean']/stats_sparse['mean']:.2f}x"
        )

    return stats_dense, stats_sparse


@pytest.mark.slow
def test_sparse_cond_perf_smoke():
    """Small perf run under pytest — confirms both circuits build and that
    NVTX-instrumented ``conditional(...)`` loops complete on both paths."""
    _build_and_run(T=8, H=32, V=128, bs=8, density=0.1,
                   n_warmup=2, n_iter=3)


if __name__ == "__main__":
    # Larger defaults intended for standalone profiling under nsys. See
    # tests/layer/sparse_hmm_perf_test.py for the bs/H tradeoff discussion
    # (edge-id memory vs kernel tile size).
    _build_and_run(
        T=32,
        H=8192,
        V=32768,
        bs=1024,
        density=0.01,
        n_warmup=1,
        n_iter=2,
    )
