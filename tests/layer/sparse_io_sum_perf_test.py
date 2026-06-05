"""Perf harness: sparse-IO sum chain (non-BD) vs dense GEMV (DenseSumLayer).

Times forward, backward (element flows only), and backward (with param
flows) independently, so each phase's speedup is visible side by side.

The dense baseline uses ``DenseCategorical`` emissions and is compiled with
``use_dense_sum_layer=True``, so the chain interior runs on
:class:`DenseSumLayer` — a genuine dense matrix-vector (GEMV) transition,
the right thing to compare against. (Without this flag the interior falls
back to the general :class:`SumLayer`, whose B=1 per-edge kernel is orders
of magnitude slower and is *not* a dense GEMV.) The sparse build uses
``SparseCategorical`` CSC emissions + ``sparse_multiply`` / ``sparse_summate``
so the DAG pre-pass upgrades the interior to :class:`SparseIOSumLayer` +
:class:`CoSparseProdLayer`.

Batch size is fixed at 1: the sparse fast path (``SparseProdLayer`` /
``SparseIOSumLayer``) is an inference-only, ``B=1``-only path, so the
head-to-head can only be run at ``B=1``.

``bwd_pflow`` on the dense path is reported as ``n/a``: :class:`DenseSumLayer`
is inference-only and raises on parameter-flow accumulation (learning needs
``use_dense_sum_layer=False``, i.e. the general path). The sparse-IO path
supports param flows, so its ``bwd_pflow`` is still timed. The dense GEMV is
memory-bound at large ``H`` (the ``H x H`` transition is ``4*H*H`` bytes per
buffer), not edge-count bound.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import (
    inputs, multiply, summate, sparse_multiply, sparse_summate, set_block_size,
)
from pyjuice.layer import DenseSumLayer, SparseIOSumLayer

# Allow running this file directly (``python tests/layer/sparse_io_sum_perf_test.py``):
# put the repo root on ``sys.path`` so the absolute ``tests.*`` import resolves
# (under pytest the rootdir is already on the path).
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.layer._sparse_io_perf_helpers import random_csc_pattern, time_phase


def _build_dense(T, H, V, bs):
    """Dense baseline using DenseCategorical input (no CSC) + plain layers.

    Compile the returned root with ``use_dense_sum_layer=True`` so the
    block-dense interior summates become :class:`DenseSumLayer` (GEMV)."""
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


def _build_and_run(T, H, V, bs, density, n_warmup, n_iter, seed=0):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    csc_indptr, csc_indices, csc_values = random_csc_pattern(
        H=H, V=V, density=density, seed=seed,
    )

    print(f"\n=== T={T} H={H} V={V} bs={bs} density={density} ===")
    print(f"  {'':16s} {'fwd':>10s} {'bwd_ele':>10s} {'bwd_pflow':>10s}")

    data = torch.randint(0, V, (1, T), device=device)

    # Dense GEMV baseline. DenseSumLayer is inference-only, so it times
    # fwd + bwd_ele only; bwd_pflow is n/a (param-flow not supported).
    dense_phases = ["fwd", "bwd_ele"]
    torch.manual_seed(seed)
    root_dense = _build_dense(T, H, V, bs)
    pc_dense = juice.TensorCircuit(
        root_dense, verbose=False, use_dense_sum_layer=True,
    ).to(device)
    assert any(isinstance(l, DenseSumLayer)
               for lg in pc_dense.inner_layer_groups for l in lg), (
        "expected DenseSumLayer on the dense baseline (use_dense_sum_layer=True)"
    )
    dense_row = {ph: time_phase(pc_dense, data, ph, n_warmup, n_iter)
                 for ph in dense_phases}
    del pc_dense
    torch.cuda.empty_cache()

    # Sparse-IO fast path (times all three phases).
    sparse_phases = ["fwd", "bwd_ele", "bwd_pflow"]
    torch.manual_seed(seed)
    root_sparse = _build_sparse_chain(
        T, H, V, bs, csc_indptr, csc_indices, csc_values,
    )
    pc_sparse = juice.TensorCircuit(root_sparse, verbose=False).to(device)
    assert any(isinstance(l, SparseIOSumLayer)
               for lg in pc_sparse.inner_layer_groups for l in lg), (
        "expected SparseIOSumLayer on the sparse build"
    )
    sparse_row = {ph: time_phase(pc_sparse, data, ph, n_warmup, n_iter)
                  for ph in sparse_phases}

    results = {"dense_gemv": dense_row, "sparse_io": sparse_row}

    def _fmt(row, phase):
        v = row.get(phase)
        return f"{'n/a':>10s}" if v is None else f"{v:8.3f}ms"

    for label in ("dense_gemv", "sparse_io"):
        row = results[label]
        print(f"  {label:16s} {_fmt(row, 'fwd')} {_fmt(row, 'bwd_ele')} "
              f"{_fmt(row, 'bwd_pflow')}")

    def _spd(phase):
        d, s = dense_row.get(phase), sparse_row.get(phase)
        if d is None or s is None or s <= 0:
            return f"{'n/a':>10s}"
        return f"{d / s:8.2f}x"
    print(f"  {'speedup':16s} {_spd('fwd'):>10s} {_spd('bwd_ele'):>10s} "
          f"{_spd('bwd_pflow'):>10s}")

    return results


@pytest.mark.slow
def test_sparse_io_sum_perf_smoke():
    """Smoke test — both circuits build, the dense interior compiles to
    DenseSumLayer (GEMV), and the timed phases complete: fwd + bwd_ele on
    both paths, bwd_pflow on the sparse path (n/a on the dense GEMV path)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    res = _build_and_run(T=8, H=32, V=128, bs=8, density=0.1,
                         n_warmup=2, n_iter=5, seed=42)
    for phase in ("fwd", "bwd_ele"):
        assert res["dense_gemv"][phase] is not None
    for phase in ("fwd", "bwd_ele", "bwd_pflow"):
        assert res["sparse_io"][phase] is not None
    assert "bwd_pflow" not in res["dense_gemv"]  # inference-only GEMV


if __name__ == "__main__":
    # The dense H*H GEMV transition is ~256 MiB per buffer at H=8192; the
    # sparse path is unaffected by H*H. Both run fwd + bwd_ele; only the
    # sparse path runs bwd_pflow.
    _build_and_run(
        T=32, H=8192, V=32768, bs=8192, density=0.01,
        n_warmup=1, n_iter=2, seed=42,
    )
