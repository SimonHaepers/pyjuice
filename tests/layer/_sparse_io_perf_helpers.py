"""Shared helpers for the sparse-IO perf harnesses.

Both :mod:`tests.layer.sparse_io_sum_perf_test` (non-BD chain) and
:mod:`tests.layer.sparse_io_bd_training_perf_test` (block-diagonal chain)
build the same HMM two ways — a dense plain-``SumLayer`` baseline and a
sparse-IO fast path — and time them head to head. The CSC emission-pattern
generator and the per-phase CUDA-event timer are identical across the two,
so they live here rather than being copy-pasted.
"""
from __future__ import annotations

import torch


def random_csc_pattern(H: int, V: int, density: float, seed: int):
    """Random row-normalised CSC emission pattern with per-row coverage.

    Returns ``(csc_indptr, csc_indices, csc_values)`` describing an ``H x V``
    emission matrix with ``~density`` of its entries active, every row
    guaranteed at least one active column, and each row summing to one.
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


def time_phase(pc, data, phase, n_warmup, n_iter):
    """Mean wall-clock (ms) of one backward/forward phase over ``n_iter`` runs.

    ``phase`` is one of:
      * ``"fwd"``        — forward only (``pc(data)``)
      * ``"bwd_ele"``    — backward, element flows only
                           (``compute_param_flows=False``)
      * ``"bwd_pflow"``  — backward with parameter flows (the full EM step)

    A forward pass precedes every backward (flows need fresh ``node_mars``).
    Timing uses CUDA events and synchronises around each measured region.
    """
    def _run():
        if phase == "fwd":
            pc(data)
        elif phase == "bwd_ele":
            pc.backward(data, compute_param_flows=False, flows_memory=0.0,
                        allow_modify_flows=False)
        elif phase == "bwd_pflow":
            pc.backward(data, compute_param_flows=True, flows_memory=0.0,
                        allow_modify_flows=False)
        else:
            raise ValueError(f"unknown phase {phase!r}")

    if phase != "fwd":
        pc(data)

    for _ in range(n_warmup):
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
