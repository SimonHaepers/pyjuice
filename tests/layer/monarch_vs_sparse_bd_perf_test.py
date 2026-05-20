"""
Perf harness comparing the **dense Monarch** HMM transition against the
**sparse-IO block-diagonal** HMM transition on circuits of matched
``T``, ``H``, ``V``.

The two are *not* the same DAG — there's no sparse-aware permutation
product (item (b) in the design discussion), so a sparse-IO chain
can't host a Monarch ``BD₁ → permutation → BD₂``. The natural
comparison instead is:

* **Monarch (dense)**: ``DenseCategorical`` input + plain ``ProdLayer`` +
  ``BlockDiagonalSumLayer``. Each timestep's transition is
  ``BD₁ → perm-multiply → BD₂`` (one Monarch matrix), tied across T.
  Total transition params ≈ 2·NB·bs² ≈ 2·H·sqrt(H).

* **Sparse-IO BD**: ``SparseCategorical`` input with a low-density CSC
  emission pattern + ``SparseProdLayer`` → ``SparseIOBlockDiagonalSumLayer``
  → ``CoSparseProdLayer`` chain. Each timestep's transition is a *single*
  block-diagonal sum (no Monarch permutation in the chain — the
  block-diagonal layer's per-block work is ``K_in_j · K_out_j`` and only
  the active subset of states is materialised per step).
  Total transition params ≈ NB·bs² ≈ H·sqrt(H).

Both use ``homogeneous=True`` (tied transition + tied emission) and
``B == 1`` (the sparse-IO fast path is B=1 by design). With matched H,
the Monarch path captures the full latent space in dense form every
timestep; the sparse-IO BD path only computes over the active
K_in / K_out columns determined by the observed token's CSC pattern.

Backward runs with ``compute_param_flows=False, _inner_layers_only=True``
to confine the timed region to the sum / prod / sparse-IO kernels.

NVTX ranges (visible in ``nsys-ui``):
  * ``build/monarch_dense``, ``build/sparse_io_bd`` — compile phases
  * ``{label}/warmup``, ``{label}/timed`` — outer phases per path
  * ``{label}/{phase}/iter-{i}/{fwd|bwd}`` — per-iteration

Profile with::

    nsys profile --trace=cuda,nvtx --output mon_vs_sparse \\
        pixi run -e dev python tests/layer/monarch_vs_sparse_bd_perf_test.py
"""
from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import (
    inputs, multiply, summate, sparse_multiply, sparse_summate, set_block_size,
)
from pyjuice.model import TensorCircuit


# ---------------------------------------------------------------------------
# Random CSC emission pattern (same helper as the parity tests).
# ---------------------------------------------------------------------------


def _random_csc_pattern(H: int, V: int, density: float, seed: int):
    """Random CSC emission pattern with per-row coverage, row-normalised.

    Every latent row gets at least one active emission column; per-row
    probs are renormalised to 1.
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


# ---------------------------------------------------------------------------
# DAG builders
# ---------------------------------------------------------------------------


def _bd_edge_ids(NB: int) -> torch.Tensor:
    return torch.arange(0, NB)[None, :].repeat(2, 1)


def _monarch_permutation(H: int, permute_block_size: int) -> torch.Tensor:
    """Standard Monarch transpose, expressed as a per-node ``multiply``
    ``edge_ids`` (consumed with ``sparse_edges=True``)."""
    return (
        torch.arange(0, H)
        .reshape(H // permute_block_size, permute_block_size)
        .permute(1, 0)
        .reshape(H)[:, None]
    )


def _build_monarch_dense_hmm(T: int, H: int, V: int, bs: int):
    """Homogeneous (tied) HMM with a Monarch-factored transition per
    timestep (``BD₁ → perm-multiply → BD₂``), dense ``DenseCategorical``
    emission. Same builder as ``monarch_hmm_perf_test._build_monarch_hmm_dag``."""
    num_node_blocks = H // bs
    bd_edges = _bd_edge_ids(num_node_blocks)
    perm = _monarch_permutation(H, permute_block_size=bs)

    with set_block_size(block_size=bs):
        ns_input = inputs(
            T - 1, num_node_blocks=num_node_blocks,
            dist=dists.DenseCategorical(num_cats=V),
        )
        ns_bd1 = None
        ns_bd2 = None
        curr_zs = ns_input
        for var in range(T - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params=True)
            if ns_bd1 is None:
                bd1 = summate(curr_zs, edge_ids=bd_edges, block_size=bs)
                ns_bd1 = bd1
            else:
                bd1 = ns_bd1.duplicate(curr_zs, tie_params=True)
            np_perm = multiply(bd1, edge_ids=perm, sparse_edges=True)
            if ns_bd2 is None:
                bd2 = summate(np_perm, edge_ids=bd_edges, block_size=bs)
                ns_bd2 = bd2
            else:
                bd2 = ns_bd2.duplicate(np_perm, tie_params=True)
            curr_zs = multiply(curr_xs, bd2)
        root = summate(curr_zs, num_node_blocks=1, block_size=1)
    return root


def _build_sparse_io_bd_hmm(T: int, H: int, V: int, bs: int,
                              csc_indptr: torch.Tensor,
                              csc_indices: torch.Tensor,
                              csc_values: torch.Tensor):
    """Homogeneous (tied) HMM with a *single* block-diagonal transition
    per timestep and ``SparseCategorical`` emissions. The DAG-level chain
    SparseProdLayer → BD summate → CoSparseProdLayer is detected by the
    compiler and the BD sums route to
    :class:`SparseIOBlockDiagonalSumLayer`."""
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


# ---------------------------------------------------------------------------
# Timing helpers (mirrors monarch_hmm_perf_test._time_pc)
# ---------------------------------------------------------------------------


def _time_pc(pc: TensorCircuit, data: torch.Tensor,
             n_warmup: int, n_iter: int,
             warmup_data: Optional[torch.Tensor] = None,
             do_backward: bool = True,
             label: str = "pc",
             use_cuda_graphs: bool = False) -> dict:
    if warmup_data is None:
        warmup_data = data

    fwd_kwargs_record = {"record_cudagraph": True} if use_cuda_graphs else {}
    bwd_kwargs_record = {"record_cudagraph": True} if use_cuda_graphs else {}

    with torch.cuda.nvtx.range(f"{label}/warmup"):
        for i in range(n_warmup):
            extra_fwd = fwd_kwargs_record if i == 0 else {}
            extra_bwd = bwd_kwargs_record if i == 0 else {}
            with torch.cuda.nvtx.range(f"{label}/warmup/iter-{i}"):
                with torch.cuda.nvtx.range(f"{label}/warmup/iter-{i}/fwd"):
                    pc(warmup_data, **extra_fwd)
                if do_backward:
                    with torch.cuda.nvtx.range(f"{label}/warmup/iter-{i}/bwd"):
                        pc.backward(
                            warmup_data, compute_param_flows=False,
                            allow_modify_flows=False, _inner_layers_only=True,
                            **extra_bwd,
                        )
        torch.cuda.synchronize()

    fwd_events = [(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    bwd_events = [(torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]

    with torch.cuda.nvtx.range(f"{label}/timed"):
        for i in range(n_iter):
            with torch.cuda.nvtx.range(f"{label}/timed/iter-{i}"):
                with torch.cuda.nvtx.range(f"{label}/timed/iter-{i}/fwd"):
                    fwd_events[i][0].record()
                    pc(data)
                    fwd_events[i][1].record()

                if do_backward:
                    with torch.cuda.nvtx.range(f"{label}/timed/iter-{i}/bwd"):
                        bwd_events[i][0].record()
                        pc.backward(
                            data, compute_param_flows=False,
                            allow_modify_flows=False, _inner_layers_only=True,
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
# Build + run
# ---------------------------------------------------------------------------


def _build_and_run(T: int, H: int, V: int, bs: int, density: float,
                   n_warmup: int, n_iter: int, seed: int = 0,
                   use_cuda_graphs: bool = False):
    """Build both circuits, sanity-check dispatch, run paired timed loops.

    ``B == 1`` is hard-wired — the sparse-IO BD chain is B=1 by design,
    and Monarch is benched at B=1 too to keep the comparison
    apples-to-apples.
    """
    assert torch.cuda.is_available(), "this perf test requires CUDA"
    assert H % bs == 0, f"H={H} must be divisible by bs={bs}"
    device = torch.device("cuda:0")
    B = 1

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=density, seed=seed,
    )

    # --- Build circuits -------------------------------------------------- #
    with torch.cuda.nvtx.range("build/monarch_dense"):
        root_monarch = _build_monarch_dense_hmm(T, H, V, bs)
        pc_monarch = TensorCircuit(
            root_monarch, use_dense_sum_layer=True,
            device=device, verbose=False,
        )

    with torch.cuda.nvtx.range("build/sparse_io_bd"):
        root_sparse = _build_sparse_io_bd_hmm(
            T, H, V, bs, csc_indptr, csc_indices, csc_values,
        )
        pc_sparse = TensorCircuit(
            root_sparse, use_dense_sum_layer=True,
            device=device, verbose=False,
        )

    # --- Verify expected layer classes are actually in use --------------- #
    from pyjuice.layer import (
        DenseCategoricalInputLayer, BlockDiagonalSumLayer,
        SparseIOBlockDiagonalSumLayer, CoSparseProdLayer,
    )
    monarch_bd = [l for lg in pc_monarch.inner_layer_groups for l in lg
                   if isinstance(l, BlockDiagonalSumLayer)
                   and not isinstance(l, SparseIOBlockDiagonalSumLayer)]
    assert monarch_bd, "Monarch path should compile BlockDiagonalSumLayer"
    sparse_bd_io = [l for lg in pc_sparse.inner_layer_groups for l in lg
                     if isinstance(l, SparseIOBlockDiagonalSumLayer)]
    sparse_cosparse = [l for lg in pc_sparse.inner_layer_groups for l in lg
                        if isinstance(l, CoSparseProdLayer)]
    assert sparse_bd_io, (
        "sparse path should compile SparseIOBlockDiagonalSumLayer"
    )
    assert sparse_cosparse, (
        "sparse path should compile CoSparseProdLayer in the chain interior"
    )

    num_node_blocks = H // bs
    nnz = int(csc_indices.numel())
    print(
        f"\n=== HMM perf: T={T}, H={H}, V={V}, bs={bs}, "
        f"NB={num_node_blocks}, density={density:.2%}"
        f"{' [cuda graphs]' if use_cuda_graphs else ''} ===\n"
        f"  monarch_dense:  {len(monarch_bd)} BD layer(s), "
        f"{pc_monarch.num_sum_params} sum params "
        f"(≈ 2·NB·bs² = {2*num_node_blocks*bs*bs})\n"
        f"  sparse_io_bd:   {len(sparse_bd_io)} sparse-IO BD + "
        f"{len(sparse_cosparse)} CoSparseProdLayer, "
        f"{pc_sparse.num_sum_params} sum params "
        f"(≈ NB·bs² = {num_node_blocks*bs*bs}); "
        f"emission nnz={nnz}/{H*V} ({nnz/(H*V):.2%})"
    )

    # --- Data ----------------------------------------------------------- #
    g = torch.Generator(device=device).manual_seed(seed)
    warmup_data = torch.randint(0, V, (B, T), generator=g, device=device)
    data = torch.randint(0, V, (B, T), generator=g, device=device)

    # --- Run ------------------------------------------------------------ #
    stats_monarch = _time_pc(
        pc_monarch, data, n_warmup=n_warmup, n_iter=n_iter,
        warmup_data=warmup_data, label="monarch_dense",
        use_cuda_graphs=use_cuda_graphs,
    )
    stats_sparse = _time_pc(
        pc_sparse, data, n_warmup=n_warmup, n_iter=n_iter,
        warmup_data=warmup_data, label="sparse_io_bd",
        use_cuda_graphs=use_cuda_graphs,
    )

    _print_summary("monarch_dense", stats_monarch)
    _print_summary("sparse_io_bd ", stats_sparse)

    if stats_sparse["fwd_mean"] > 0:
        print(
            f"  speedup fwd (monarch / sparse): "
            f"{stats_monarch['fwd_mean']/stats_sparse['fwd_mean']:.2f}x"
        )
    if stats_sparse["bwd_mean"] > 0:
        print(
            f"  speedup bwd (monarch / sparse): "
            f"{stats_monarch['bwd_mean']/stats_sparse['bwd_mean']:.2f}x"
        )

    return stats_monarch, stats_sparse


# ---------------------------------------------------------------------------
# Tests / entry point
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_monarch_vs_sparse_bd_perf_smoke():
    """Small perf run under pytest — confirms both circuits build, both
    fast paths are engaged, and fwd/bwd loops complete."""
    _build_and_run(T=8, H=64, V=64, bs=8, density=0.5,
                   n_warmup=2, n_iter=3)


# NOTE: no ``use_cuda_graphs=True`` variant for this comparison — the
# sparse-IO BD chain calls into ``SparseCategorical.build_sparse_pattern``
# every timestep, which does host-side Python work (lookup table indexing,
# ``narrow`` slicing) that CUDA graph capture rejects with
# ``cudaErrorStreamCaptureInvalidated``. The Monarch-only perf test in
# :mod:`monarch_hmm_perf_test` exercises the cudagraph path against the
# dense baseline; here we just compare wall-clock with both circuits on
# the live stream.


if __name__ == "__main__":
    # Realistic-size run for standalone profiling.
    #
    # Density knob: the sparse-IO BD path's compute scales with
    # ``K_in + K_out`` per timestep, where ``K_in`` and ``K_out`` are the
    # active emission rows per CSC column. Lower density → fewer active
    # rows → bigger speedup over dense Monarch.
    #
    # The Monarch path runs over the full ``H`` state space every
    # timestep regardless of token, so it doesn't benefit from emission
    # sparsity at all — the asymmetry is the whole point of this
    # comparison.
    _build_and_run(
        T=32,
        H=4096*4,
        V=4096,
        bs=128,
        density=0.02,
        n_warmup=1,
        n_iter=2,
        use_cuda_graphs=False,
    )
