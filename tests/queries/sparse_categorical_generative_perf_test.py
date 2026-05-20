"""Perf harness for the **generative** conditional query on HMMs: given a
sequence with a single position ``t*`` unobserved (``missing_mask`` True
there) and the rest observed, compute ``p(x_{t*} | x_{!=t*})``.

This is the imputation/completion sibling of
``tests/queries/sparse_categorical_cond_perf_test.py`` (smoothing). Both tests
share the same compiled circuits and the same forward over known tokens;
only the backward differs in scope:

* Smoothing benchmark: ``target_vars=None`` materialises ``[B, T, V]``
  posteriors at every timestep simultaneously.
* Generative benchmark: ``target_vars=[t*]`` + ``missing_mask[t*]=True``
  produces ``[B, 1, V]``.

The forward overwrites the unknown's input log-likelihood with 0.0 via
``_fw_missing_mask_kernel`` (the per-var CSC gather still runs on whatever
junk index lives at ``data[b, t*]``, then is thrown away — one negligible
wasted column). The backward hits a single column on each pipeline:

* Dense path: ``DenseCategoricalInputLayer.dense_conditional_backward``
  with ``target_vars=[t*]`` slices a single ``[K, C]`` block out of the bmm.
* Sparse path: ``_sparse_categorical_backward`` filters ``layer.nodes`` by
  variable id (one ``ns`` runs); with ``B=1`` the sparse-native CSR-side
  fast path kicks in via the ``sv_flow`` cached on the parent
  ``SparseProdLayer``.

Both bypass the known partial-eval/block-id bug in the atomic-add fallback
(see memory ``project_partial_eval_block_bug``) — dense doesn't call
``enable_partial_evaluation``, and sparse iterates ``layer.nodes`` directly.

Wall-clock per-iter timings (CUDA events) are printed for at-a-glance
inspection. The DAG builders mirror those in the smoothing perf test so
results can be compared apples-to-apples.

The ``test_generative_perf_smoke`` case is marked ``slow``.
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
# DAG builders (duplicated from sparse_categorical_cond_perf_test.py — kept
# in-file to match the directory's convention of self-contained tests)
# ---------------------------------------------------------------------------


def _make_csc_emissions(H: int, V: int, density: float, seed: int,
                         device: Optional[torch.device] = None):
    """Random CSC emission pattern + row-normalised probabilities. Identical
    to the smoothing perf test's helper — see that file for details."""
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


def _time_generative(pc: TensorCircuit, data: torch.Tensor,
                     missing_mask: torch.Tensor, target_t: int,
                     n_warmup: int, n_iter: int,
                     warmup_data: Optional[torch.Tensor] = None) -> dict:
    """Warm up JIT, then run ``n_iter`` generative ``conditional`` calls and
    return per-iter wall-clock stats.

    The query runs ``conditional(pc, data, missing_mask=missing_mask,
    target_vars=[target_t])`` — produces a ``[B, 1, V]`` posterior over the
    single masked-out token, given the rest of the sequence.
    """
    if warmup_data is None:
        warmup_data = data
    target_vars = [target_t]
    for i in range(n_warmup):
        juice.queries.conditional(pc, data=warmup_data,
                                  missing_mask=missing_mask,
                                  target_vars=target_vars)
    torch.cuda.synchronize()

    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]

    for i in range(n_iter):
        events[i][0].record()
        juice.queries.conditional(pc, data=data, missing_mask=missing_mask,
                                  target_vars=target_vars)
        events[i][1].record()
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
        f"[{label}] generative: mean {stats['mean']:.3f} ms  "
        f"min {stats['min']:.3f} ms  "
        f"max {stats['max']:.3f} ms"
    )


# ---------------------------------------------------------------------------
# Tests / entry point
# ---------------------------------------------------------------------------


def _build_and_run_generative(T: int, H: int, V: int, bs: int, density: float,
                              n_warmup: int, n_iter: int, seed: int = 0,
                              target_t: Optional[int] = None):
    assert torch.cuda.is_available(), "this perf test requires CUDA"
    device = torch.device("cuda:0")

    if target_t is None:
        target_t = T // 2

    root_dense = _build_dense_hmm_dag(T, H, V, bs)
    pc_dense = TensorCircuit(
        root_dense,
        use_dense_sum_layer=True,
        device=device,
        verbose=False,
    )

    csc_indptr, csc_indices, csc_values = _make_csc_emissions(
        H=H, V=V, density=density, seed=seed, device=device,
    )
    root_sparse = _build_sparse_hmm_dag(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = TensorCircuit(
        root_sparse,
        use_dense_sum_layer=True,
        device=device,
        verbose=False,
    )

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
        f"\n=== generative perf: T={T}, H={H}, V={V}, bs={bs}, "
        f"density={density:.2%}, target_t={target_t} ===\n"
        f"  dense:  {len(dense_input_layers)} DenseCategoricalInputLayer(s)\n"
        f"  sparse: {len(sparse_prod_layers)} SparseProdLayer(s), "
        f"{len(sparse_sum_layers)} SparseInputSumLayer(s)"
    )

    # B=1 to exercise the sparse fast path. Distinct warmup vs. timing
    # sequences as in the smoothing perf test. ``data[b, target_t]`` is
    # filled with a valid token id but its forward is overwritten by the
    # missing_mask so the value is irrelevant.
    g = torch.Generator(device=device).manual_seed(seed)
    warmup_data = torch.randint(0, V, (1, T), generator=g, device=device)
    data = torch.randint(0, V, (1, T), generator=g, device=device)

    missing_mask = torch.zeros(T, dtype=torch.bool, device=device)
    missing_mask[target_t] = True

    stats_dense = _time_generative(
        pc_dense, data, missing_mask, target_t,
        n_warmup=n_warmup, n_iter=n_iter, warmup_data=warmup_data,
    )
    stats_sparse = _time_generative(
        pc_sparse, data, missing_mask, target_t,
        n_warmup=n_warmup, n_iter=n_iter, warmup_data=warmup_data,
    )

    _print_summary("dense", stats_dense)
    _print_summary("sparse", stats_sparse)

    if stats_sparse["mean"] > 0:
        print(
            f"  speedup generative: "
            f"{stats_dense['mean']/stats_sparse['mean']:.2f}x"
        )

    return stats_dense, stats_sparse


@pytest.mark.slow
def test_generative_perf_smoke():
    """Small perf run under pytest — confirms both circuits build and that
    NVTX-instrumented generative ``conditional(...)`` loops complete on both
    paths."""
    _build_and_run_generative(T=8, H=32, V=128, bs=8, density=0.1,
                              target_t=4, n_warmup=2, n_iter=3)


if __name__ == "__main__":
    # Larger defaults intended for standalone profiling under nsys, matched
    # to ``sparse_categorical_cond_perf_test.py`` so the smoothing and
    # generative numbers are directly comparable.
    _build_and_run_generative(
        T=32,
        H=8192*4,
        V=32768,
        bs=8192*4,
        density=0.01,
        target_t=16,
        n_warmup=1,
        n_iter=2,
    )
