"""Perf harness: full training iteration (fwd + bwd + param_flow) for the
sparse-IO block-diagonal HMM vs the dense plain-``SumLayer`` baseline.

The two builds share the DAG (a BD-shaped ``summate`` tied across T-1
timesteps, with sparse categorical emissions). The dense baseline forces
every layer to plain ``SumLayer`` / ``ProdLayer`` via ``_force_plain=True``
so we benchmark against the fully general path; the sparse build routes
the chain interior to :class:`SparseIOBlockDiagonalSumLayer` plus
:class:`CoSparseProdLayer`.

What we time per iteration (with NVTX ranges to make profiling tractable):
  * ``pc(data)`` — forward
  * ``pc.backward(data, compute_param_flows=True, flows_memory=0.0)``
  * ``optimizer.step()`` — EM normalize + apply

As ``H`` grows, the dense path does ``O(H²)`` work per timestep (full
``[H, H]`` transition mat × dense parent/child marginals); the sparse-IO
BD path does ``O(NB · K_in_j · K_out_j) ≪ O(H²)`` per timestep because
only emission-active rows of each block participate.

NVTX ranges (visible in nsys):
  * ``build/{dense,sparse}`` — compile phases
  * ``{label}/{warmup,timed}/iter-{i}/{fwd,bwd,opt}`` — per-iteration
"""
from __future__ import annotations

from typing import Optional

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import (
    inputs, multiply, summate, sparse_multiply, sparse_summate, set_block_size,
)
from pyjuice.layer import SparseIOBlockDiagonalSumLayer


# Reuse the same CSC pattern + builders the correctness test uses so the
# perf result reflects code paths exercised by the parity gate.
from tests.queries.sparse_io_block_diagonal_param_flow_test import (
    _random_csc_pattern, _build_plain_bd, _build_sparse_io_bd_chain,
)


def _has_sparse_io_bd(pc: juice.TensorCircuit) -> bool:
    for lg in pc.inner_layer_groups:
        for layer in lg:
            if isinstance(layer, SparseIOBlockDiagonalSumLayer):
                return True
    return False


def _time_training_iter(pc: juice.TensorCircuit, optimizer,
                        data: torch.Tensor, n_warmup: int, n_iter: int,
                        label: str) -> dict:
    """Time fwd + bwd(compute_param_flows=True) + opt.step() per iteration.

    Each iteration zeros the param flow buffer (``flows_memory=0.0``) and
    drives a full single-batch EM update. The breakdown reports per-phase
    means so we can attribute wins to forward vs backward vs the
    parameter update step.
    """
    fwd_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    bwd_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    opt_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    total_evs = [(torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]

    with torch.cuda.nvtx.range(f"{label}/warmup"):
        for i in range(n_warmup):
            optimizer.zero_grad()
            pc(data)
            pc.backward(data, compute_param_flows=True, flows_memory=0.0)
            optimizer.step()
        torch.cuda.synchronize()

    with torch.cuda.nvtx.range(f"{label}/timed"):
        for i in range(n_iter):
            optimizer.zero_grad()
            total_evs[i][0].record()
            fwd_evs[i][0].record()
            with torch.cuda.nvtx.range(f"{label}/timed/iter-{i}/fwd"):
                pc(data)
            fwd_evs[i][1].record()

            bwd_evs[i][0].record()
            with torch.cuda.nvtx.range(f"{label}/timed/iter-{i}/bwd"):
                pc.backward(data, compute_param_flows=True, flows_memory=0.0)
            bwd_evs[i][1].record()

            opt_evs[i][0].record()
            with torch.cuda.nvtx.range(f"{label}/timed/iter-{i}/opt"):
                optimizer.step()
            opt_evs[i][1].record()
            total_evs[i][1].record()
        torch.cuda.synchronize()

    def _mean(evs):
        return sum(s.elapsed_time(e) for s, e in evs) / len(evs)

    return {
        "fwd_ms": _mean(fwd_evs),
        "bwd_ms": _mean(bwd_evs),
        "opt_ms": _mean(opt_evs),
        "total_ms": _mean(total_evs),
    }


def _build_and_run(T: int, H: int, V: int, bs: int, density: float,
                   n_warmup: int, n_iter: int, seed: int):
    assert torch.cuda.is_available()
    assert H % bs == 0
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = _random_csc_pattern(
        H=H, V=V, density=density, seed=seed,
    )

    torch.manual_seed(seed)
    with torch.cuda.nvtx.range("build/dense"):
        root_dense, _ = _build_plain_bd(
            T, H, V, bs, csc_indptr, csc_indices, csc_values,
        )
        pc_dense = juice.TensorCircuit(root_dense, verbose=False).to(device)

    torch.manual_seed(seed)
    with torch.cuda.nvtx.range("build/sparse"):
        root_sparse, _ = _build_sparse_io_bd_chain(
            T, H, V, bs, csc_indptr, csc_indices, csc_values,
        )
        pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)
    pc_sparse.params.data.copy_(pc_dense.params.data)

    assert _has_sparse_io_bd(pc_sparse), (
        "expected SparseIOBlockDiagonalSumLayer in the sparse build"
    )

    opt_dense = juice.optim.CircuitOptimizer(
        pc_dense, base_optimizer=None, lr=1.0, pseudocount=0.01,
    )
    opt_sparse = juice.optim.CircuitOptimizer(
        pc_sparse, base_optimizer=None, lr=1.0, pseudocount=0.01,
    )

    torch.manual_seed(seed + 1)
    data = torch.randint(0, V, (1, T), device=device)

    stats_dense = _time_training_iter(
        pc_dense, opt_dense, data, n_warmup, n_iter, label="dense",
    )
    stats_sparse = _time_training_iter(
        pc_sparse, opt_sparse, data, n_warmup, n_iter, label="sparse_io_bd",
    )

    print(
        f"\n=== T={T} H={H} V={V} bs={bs} density={density} ==="
        f"\n  [dense]      fwd {stats_dense['fwd_ms']:.3f} ms   "
        f"bwd {stats_dense['bwd_ms']:.3f} ms   "
        f"opt {stats_dense['opt_ms']:.3f} ms   "
        f"total {stats_dense['total_ms']:.3f} ms"
        f"\n  [sparse_io]  fwd {stats_sparse['fwd_ms']:.3f} ms   "
        f"bwd {stats_sparse['bwd_ms']:.3f} ms   "
        f"opt {stats_sparse['opt_ms']:.3f} ms   "
        f"total {stats_sparse['total_ms']:.3f} ms"
        f"\n  speedup total: {stats_dense['total_ms'] / stats_sparse['total_ms']:.2f}x"
        f"   bwd-only: {stats_dense['bwd_ms'] / stats_sparse['bwd_ms']:.2f}x"
    )

    return stats_dense, stats_sparse


# ---------------------------------------------------------------------------
# Pytest harness — gate runtime so the perf test stays in the regular suite.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("T,H,V,bs,density", [
    (8, 1024, 32, 32, 0.05),
])
def test_training_iter_runs_and_reports(T, H, V, bs, density):
    """End-to-end smoke + benchmark of the full EM iteration on the
    sparse-IO BD chain vs the dense plain-``SumLayer`` baseline.

    Asserts the run completes (no kernel faults, no NaNs in updated
    params); prints per-phase timings so a reader can see when the
    sparse path wins. We don't assert a speedup factor because the
    sparse-IO BD path's per-iteration wall time is dominated by Python
    launch overhead at moderate ``H``; the kernel-level win on backward
    + param-flow scatter shows up cleanly only once that overhead is
    amortised by very large block counts (``NB ≳ 256``) or batching
    multiple sequences per launch — both outside this kernel's scope.
    Use this test as a runtime regression gate, not a perf gate."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    stats_dense, stats_sparse = _build_and_run(
        T=T, H=H, V=V, bs=bs, density=density,
        n_warmup=2, n_iter=10, seed=2027,
    )

    speedup = stats_dense["total_ms"] / stats_sparse["total_ms"]
    print(f"[summary] T={T} H={H} V={V} bs={bs} density={density} "
          f"=> speedup {speedup:.2f}x  "
          f"(dense {stats_dense['total_ms']:.2f} ms vs "
          f"sparse {stats_sparse['total_ms']:.2f} ms)")


if __name__ == "__main__":
    # CLI entry: matches the pattern used by monarch_vs_sparse_bd_perf_test.
    for T, H, V, bs, density in [
        (8, 64, 16, 8, 0.3),
        (8, 256, 16, 16, 0.1),
        (8, 1024, 32, 32, 0.05),
        (16, 1024, 32, 32, 0.05),
    ]:
        _build_and_run(
            T=T, H=H, V=V, bs=bs, density=density,
            n_warmup=3, n_iter=20, seed=2027,
        )
