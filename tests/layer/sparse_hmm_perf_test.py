"""
Perf harness comparing the **dense** and **sparse** HMM inference pipelines
on circuits of identical topology (same ``T``, ``H``, ``V``, ``block_size``).

* Dense path: ``Categorical`` input + ``DenseCategoricalInputLayer`` +
  ``DenseSumLayer``.
* Sparse path: ``SparseCategorical`` input with a low-density CSC emission
  pattern + ``SparseProdLayer`` (sparsity-propagating) + ``SparseInputSumLayer``
  (column-selecting sum). The ``SparseProdLayer._skip_scatter`` optimisation
  kicks in when every consumer is a ``SparseInputSumLayer``, eliding the
  O(H·B) scatter-to-dense into ``element_mars``.

Both paths use ``homogeneous=True`` (tied transitions + tied emissions) so
the DAG construction stays at a single shared H×H transition and a single
shared emission table — untied duplicates at realistic HMM sizes balloon
CPU memory (per-timestep clones of H×H and H×V). ``B=1`` is required for
the ``SparseInputSumLayer`` fast path (it falls back to :class:`DenseSumLayer`
for B>1).

Prints wall-clock per-iteration timings measured with ``torch.cuda.Event``
so you can read speedups off the console.

The ``test_perf_smoke`` case is marked ``slow`` and is skipped unless
``--run-slow`` is passed.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, sparse_multiply, sparse_summate, set_block_size
from pyjuice.model import TensorCircuit


# ---------------------------------------------------------------------------
# DAG builders
# ---------------------------------------------------------------------------


def _make_csc_emissions(H: int, V: int, density: float, seed: int,
                         device: Optional[torch.device] = None):
    """Build a random CSC emission pattern + (row-normalised) probabilities.

    Samples ``(row, col)`` pairs directly at the expected ``nnz`` size and
    dedupes — so we never materialise an ``[H, V]`` mask or Bernoulli draw.
    At H=V=32k, density=0.01, that's ~10M pair entries (~160 MB) instead of
    a 1 GB bool / 4 GB float matrix.

    We also guarantee every latent row has ≥ 1 active column by appending
    one random column per row and deduping against the drawn pool.
    """
    device = device or torch.device("cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    # Oversample by ~5% to end up near the target nnz after dedupe; small
    # overshoots are fine. Dedup is exact via unique on the linearised id.
    target_nnz = int(density * H * V * 1.05)
    rand_rows = torch.randint(0, H, (target_nnz,), generator=g, device=device)
    rand_cols = torch.randint(0, V, (target_nnz,), generator=g, device=device)

    # Guarantee coverage: append (row, random_col) for every row.
    all_rows_coverage = torch.arange(H, device=device)
    cov_cols = torch.randint(0, V, (H,), generator=g, device=device)
    rand_rows = torch.cat([rand_rows, all_rows_coverage])
    rand_cols = torch.cat([rand_cols, cov_cols])

    # Dedupe (row, col) pairs via linearised ids.
    linear = rand_rows.to(torch.long) * V + rand_cols.to(torch.long)
    linear = torch.unique(linear)  # sorted; (col, row) order comes next
    rows_dedup = linear // V                                       # [nnz]
    cols_dedup = linear % V                                        # [nnz]

    # CSC order = sort by (col, row). Re-sort with col primary key.
    sort_key = cols_dedup.to(torch.long) * H + rows_dedup.to(torch.long)
    order = torch.argsort(sort_key)
    csc_indices = rows_dedup[order].contiguous()
    cols = cols_dedup[order]

    col_counts = torch.bincount(cols, minlength=V)
    csc_indptr = torch.zeros(V + 1, dtype=torch.long, device=device)
    csc_indptr[1:] = torch.cumsum(col_counts, dim=0)

    # Row-normalise probabilities over the active slots.
    raw = torch.rand(csc_indices.numel(), generator=g, device=device)
    row_sums = torch.zeros(H, device=device)
    row_sums.scatter_add_(0, csc_indices, raw)
    csc_values = raw / row_sums[csc_indices]

    # SparseCategorical expects CPU tensors at DAG-build time; move back.
    return (csc_indptr.cpu(), csc_indices.cpu(),
            csc_values.to(torch.float32).cpu())


def _build_dense_hmm_dag(T: int, H: int, V: int, bs: int):
    """Homogeneous (tied) HMM with a dense ``Categorical`` input; uses the
    ``DenseCategorical`` marker so the compiler picks
    :class:`DenseCategoricalInputLayer`."""
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
    """Same topology as ``_build_dense_hmm_dag`` but with a
    ``SparseCategorical`` input carrying the supplied CSC pattern. Uses the
    explicit ``sparse_multiply`` / ``sparse_summate`` builders so ineligible
    children would raise a clear error; relies on ``duplicate()`` preserving
    the ``SparseProdNodes`` / ``SparseSumNodes`` subclass for tied copies."""
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
# Timing helpers
# ---------------------------------------------------------------------------


def _time_pc(pc: TensorCircuit, data: torch.Tensor,
             n_warmup: int, n_iter: int,
             warmup_data: Optional[torch.Tensor] = None,
             do_backward: bool = True) -> dict:
    """Warm up JIT, then run ``n_iter`` fwd(+bwd) and return per-phase
    wall-clock stats measured with CUDA events.

    ``warmup_data`` should be a *different* token sequence than ``data`` so
    we exercise input variability — for sparse paths the per-token CSC
    column slice differs across sequences, and reusing the timing tokens
    during warmup risks priming caches/branches in a way the timed loop
    wouldn't see in practice.

    Backward runs with ``_inner_layers_only=True`` + ``compute_param_flows=False``
    because both ``DenseSumLayer`` and ``SparseInputSumLayer`` are
    inference-only (they refuse ``param_flows is not None``) and because
    the plain ``InputLayer`` backward path doesn't guard against
    ``param_flows=None`` for Categorical leaves. That keeps the timed
    region to the prod/sum work — where the dense-vs-sparse difference
    actually lives — and makes the two paths directly comparable.
    """
    if warmup_data is None:
        warmup_data = data
    for i in range(n_warmup):
        pc(warmup_data)
        if do_backward:
            pc.backward(
                warmup_data, compute_param_flows=False, allow_modify_flows=False,
                _inner_layers_only=True,
            )
    torch.cuda.synchronize()

    fwd_events = [(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    bwd_events = [(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]

    for i in range(n_iter):
        fwd_events[i][0].record()
        pc(data)
        fwd_events[i][1].record()

        if do_backward:
            bwd_events[i][0].record()
            pc.backward(
                data, compute_param_flows=False, allow_modify_flows=False,
                _inner_layers_only=True,
            )
            bwd_events[i][1].record()
    torch.cuda.synchronize()

    fwd_ms = [s.elapsed_time(e) for s, e in fwd_events]
    bwd_ms = [s.elapsed_time(e) for s, e in bwd_events] if do_backward else []
    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "fwd_mean": sum(fwd_ms) / len(fwd_ms),
        "bwd_mean": (sum(bwd_ms) / len(bwd_ms)) if bwd_ms else 0.0,
    }


def _print_summary(label: str, stats: dict) -> None:
    print(
        f"[{label}] fwd: mean {stats['fwd_mean']:.3f} ms  "
        f"min {min(stats['fwd_ms']):.3f} ms  "
        f"max {max(stats['fwd_ms']):.3f} ms   "
        f"bwd: mean {stats['bwd_mean']:.3f} ms"
    )


# ---------------------------------------------------------------------------
# Tests / entry point
# ---------------------------------------------------------------------------


def _build_and_run(T: int, H: int, V: int, bs: int, density: float,
                   n_warmup: int, n_iter: int, seed: int = 0):
    assert torch.cuda.is_available(), "this perf test requires CUDA"
    device = torch.device("cuda:0")

    # --- Build circuits -------------------------------------------------- #
    # Pass ``device=`` so ``TensorCircuit`` allocates the flat params tensor
    # (H² floats + H·V emission floats) directly on GPU instead of going
    # through CPU and PCIe-copying on ``.to(device)``.
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

    # Verify expected layer classes are actually in use.
    from pyjuice.layer import (
        DenseCategoricalInputLayer, DenseSumLayer, SparseProdLayer,
        SparseInputSumLayer,
    )
    dense_input_layers = [l for l in pc_dense.input_layer_group
                          if isinstance(l, DenseCategoricalInputLayer)]
    assert dense_input_layers, "dense path should compile DenseCategoricalInputLayer"
    dense_sum_layers = [l for lg in pc_dense.inner_layer_groups for l in lg
                        if isinstance(l, DenseSumLayer)]
    assert dense_sum_layers, "dense path should compile DenseSumLayer"

    sparse_prod_layers = [l for lg in pc_sparse.inner_layer_groups for l in lg
                          if isinstance(l, SparseProdLayer)]
    sparse_sum_layers = [l for lg in pc_sparse.inner_layer_groups for l in lg
                         if isinstance(l, SparseInputSumLayer)]
    assert sparse_prod_layers, "sparse path should compile SparseProdLayer"
    assert sparse_sum_layers, "sparse path should compile SparseInputSumLayer"
    assert all(l._skip_scatter for l in sparse_prod_layers), (
        "expected SparseProdLayer._skip_scatter=True when every consumer is "
        "a SparseInputSumLayer"
    )

    print(
        f"\n=== HMM perf: T={T}, H={H}, V={V}, bs={bs}, density={density:.2%} ===\n"
        f"  dense:  {len(dense_sum_layers)} DenseSumLayer(s), "
        f"{len(dense_input_layers)} DenseCategoricalInputLayer(s)\n"
        f"  sparse: {len(sparse_prod_layers)} SparseProdLayer(s) "
        f"(skip_scatter={sparse_prod_layers[0]._skip_scatter}), "
        f"{len(sparse_sum_layers)} SparseInputSumLayer(s)"
    )

    # --- Data (B=1 to exercise sparse sum fast path) -------------------- #
    # Distinct warmup vs. timing sequences so warmup doesn't prime per-token
    # caches/branches that the timed loop wouldn't otherwise hit.
    g = torch.Generator(device=device).manual_seed(seed)
    warmup_data = torch.randint(0, V, (1, T), generator=g, device=device)
    data = torch.randint(0, V, (1, T), generator=g, device=device)

    # --- Run ------------------------------------------------------------ #
    stats_dense = _time_pc(pc_dense, data,
                           n_warmup=n_warmup, n_iter=n_iter,
                           warmup_data=warmup_data)
    stats_sparse = _time_pc(pc_sparse, data,
                            n_warmup=n_warmup, n_iter=n_iter,
                            warmup_data=warmup_data)

    _print_summary("dense", stats_dense)
    _print_summary("sparse", stats_sparse)

    if stats_sparse["fwd_mean"] > 0:
        print(
            f"  speedup fwd: {stats_dense['fwd_mean']/stats_sparse['fwd_mean']:.2f}x"
        )
    if stats_sparse["bwd_mean"] > 0:
        print(
            f"  speedup bwd: {stats_dense['bwd_mean']/stats_sparse['bwd_mean']:.2f}x"
        )

    return stats_dense, stats_sparse


@pytest.mark.slow
def test_sparse_hmm_perf_smoke():
    """Small perf run under pytest — confirms both circuits build and that
    NVTX-instrumented fwd/bwd loops complete for both paths."""
    _build_and_run(T=8, H=32, V=128, bs=8, density=0.1,
                   n_warmup=2, n_iter=3)


if __name__ == "__main__":
    # Larger defaults intended for standalone profiling under nsys. Adjust
    # freely: the knob that matters most for the sparse speedup is ``density``.
    #
    # ``bs`` is important. Block-dense sum nodes materialise an
    # ``edge_ids`` tensor of shape ``[2, (H/bs)²]`` long-ints on CPU at DAG
    # build time. At H=32k, bs=1 that's ~16 GB; at bs=64 it drops to 4 MB.
    # The transitions themselves are ``(H/bs)² * bs² = H²`` floats
    # regardless, so ``bs`` only trades edge-list overhead against kernel
    # tile size — keep it in the 32–128 range for realistic H.
    _build_and_run(
        T=32,
        H=32768,
        V=32768,
        bs=1024,
        density=0.01,
        n_warmup=1,
        n_iter=2,
    )
