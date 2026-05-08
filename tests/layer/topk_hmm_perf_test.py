"""
Perf harness comparing the **dense** and **TopK** HMM inference pipelines on
circuits of identical topology (same ``T``, ``H``, ``V``, ``block_size``).

* Dense path: ``DenseCategorical`` input + ``DenseCategoricalInputLayer`` +
  plain ``ProdLayer`` + ``DenseSumLayer`` (the inference-only fast path
  that skips block-sparse partitioning and addresses parameters by direct
  pointer arithmetic).
* TopK path: same input + plain ``ProdLayer`` + ``TopKLayer`` (per-summate
  per-batch top-K selection over the dense product activations) +
  ``TopKSumLayer`` (logsumexp over only the K selected children).
  Approximation knob is ``K``: smaller K → less work, more bias.

Both paths use ``homogeneous=True`` (tied transitions + tied emissions) so
the DAG construction stays at a single shared H×H transition and a single
shared emission table — untied duplicates at realistic HMM sizes balloon
CPU memory.

Backward is run with ``compute_param_flows=False, _inner_layers_only=True``
because ``DenseSumLayer`` is inference-only (refuses ``param_flows is not
None``). Running TopK in the same mode keeps the two paths apples-to-apples
and confines the timed work to the prod / sum / topk-selection kernels —
where the difference actually lives.

The ``test_perf_smoke`` case is marked ``slow``; the ``__main__`` block is
the realistic-size run intended for standalone profiling.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, set_block_size
from pyjuice.model import TensorCircuit


# ---------------------------------------------------------------------------
# DAG builders
# ---------------------------------------------------------------------------


def _build_dense_hmm_dag(T: int, H: int, V: int, bs: int):
    """Homogeneous (tied) HMM with a dense ``DenseCategorical`` input + plain
    ``summate``. ``DenseSumLayer`` is selected by the
    ``use_dense_sum_layer=True`` flag at compile time."""
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


def _build_topk_hmm_dag(T: int, H: int, V: int, bs: int, K: int):
    """Same topology as the dense builder, but every transition ``summate`` is
    annotated with ``topk=K``. The annotation rides through ``duplicate(...,
    tie_params=True)`` so every timestep dispatches to ``TopKSumLayer``."""
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
                ns = summate(curr_zs, num_node_blocks=num_node_blocks, topk=K)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)
            curr_zs = multiply(curr_xs, ns)
        root = summate(curr_zs, num_node_blocks=1, block_size=1)
    return root


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_pc(pc: TensorCircuit, data: torch.Tensor,
             n_warmup: int, n_iter: int,
             warmup_data: Optional[torch.Tensor] = None,
             do_backward: bool = True,
             label: str = "pc",
             use_cuda_graphs: bool = False) -> dict:
    """Warm up JIT, then run ``n_iter`` fwd(+bwd) and return per-phase
    wall-clock stats measured with CUDA events.

    Backward runs with ``_inner_layers_only=True, compute_param_flows=False``
    because :class:`DenseSumLayer` is inference-only — that keeps the timed
    region apples-to-apples between the two paths and confines it to the
    prod / sum / topk-selection kernels (where the difference lives).

    With ``use_cuda_graphs=True`` we pass ``record_cudagraph=True`` on the
    first warmup call so :class:`TensorCircuit` captures the inner-layer
    kernel sequence on a side stream and replays it on every subsequent
    call (signature is keyed by the persistent buffer ids and ``B``, so
    later calls hit the cache and ``apply_cudagraph=True`` — the default —
    triggers ``g.replay()``). Input-layer forward + root-flow init still
    run on the live stream every call, so per-batch data dependence is
    preserved.

    NVTX ranges (visible in ``nsys-ui``):
      * ``{label}/warmup`` and ``{label}/timed`` — outer phases
      * ``{label}/{warmup|timed}/iter-{i}/{fwd|bwd}`` — per-iteration
    Use the ``label`` arg to keep ``dense`` and ``topk`` runs visually
    distinct in the timeline.
    """
    if warmup_data is None:
        warmup_data = data

    fwd_kwargs_record = {"record_cudagraph": True} if use_cuda_graphs else {}
    bwd_kwargs_record = {"record_cudagraph": True} if use_cuda_graphs else {}

    with torch.cuda.nvtx.range(f"{label}/warmup"):
        for i in range(n_warmup):
            # Only the first warmup pass triggers the capture; further
            # calls find the signature in the cache and replay.
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


def _build_and_run(T: int, H: int, V: int, bs: int, K: int, B: int,
                   n_warmup: int, n_iter: int, seed: int = 0,
                   use_cuda_graphs: bool = False):
    assert torch.cuda.is_available(), "this perf test requires CUDA"
    device = torch.device("cuda:0")

    # --- Build circuits -------------------------------------------------- #
    with torch.cuda.nvtx.range("build/dense"):
        root_dense = _build_dense_hmm_dag(T, H, V, bs)
        pc_dense = TensorCircuit(
            root_dense,
            use_dense_sum_layer=True,
            device=device,
            verbose=False,
        )

    with torch.cuda.nvtx.range("build/topk"):
        root_topk = _build_topk_hmm_dag(T, H, V, bs, K)
        pc_topk = TensorCircuit(
            root_topk,
            # ``use_dense_sum_layer`` only matters for sums *without* topk; the
            # root sum (no topk annotation) takes the dense path either way.
            use_dense_sum_layer=True,
            device=device,
            verbose=False,
        )
    # Sync params so any LL comparison is meaningful (we mostly care about
    # wall-clock here, but identical params lets you sanity-check the LL
    # gap matches the topk approximation rather than init noise).
    pc_topk.params.data.copy_(pc_dense.params.data)

    # --- Verify expected layer classes are actually in use --------------- #
    from pyjuice.layer import (
        DenseCategoricalInputLayer, DenseSumLayer, TopKLayer, TopKSumLayer,
    )
    dense_input_layers = [l for l in pc_dense.input_layer_group
                          if isinstance(l, DenseCategoricalInputLayer)]
    assert dense_input_layers, "dense path should compile DenseCategoricalInputLayer"
    dense_sum_layers = [l for lg in pc_dense.inner_layer_groups for l in lg
                        if isinstance(l, DenseSumLayer)]
    assert dense_sum_layers, "dense path should compile DenseSumLayer"

    topk_layers = [l for lg in pc_topk.inner_layer_groups for l in lg
                   if isinstance(l, TopKLayer)]
    topk_sum_layers = [l for lg in pc_topk.inner_layer_groups for l in lg
                       if isinstance(l, TopKSumLayer)]
    assert topk_layers, "topk path should compile TopKLayer"
    assert topk_sum_layers, "topk path should compile TopKSumLayer"
    assert len(topk_layers) == len(topk_sum_layers) == T - 1, (
        f"expected one TopK pair per transition timestep, got "
        f"{len(topk_layers)} / {len(topk_sum_layers)} for T-1 = {T - 1}"
    )

    print(
        f"\n=== HMM perf: T={T}, H={H}, V={V}, bs={bs}, K={K}, B={B}"
        f"{' [cuda graphs]' if use_cuda_graphs else ''} ===\n"
        f"  dense: {len(dense_sum_layers)} DenseSumLayer(s), "
        f"{len(dense_input_layers)} DenseCategoricalInputLayer(s)\n"
        f"  topk:  {len(topk_layers)} TopKLayer(s) + "
        f"{len(topk_sum_layers)} TopKSumLayer(s) (K/H = {K/H:.2%})"
    )

    # --- Data ----------------------------------------------------------- #
    # Distinct warmup vs. timing sequences so warmup doesn't prime per-token
    # caches/branches the timed loop wouldn't otherwise hit. TopK selection
    # is data-dependent, so the indices change every batch.
    g = torch.Generator(device=device).manual_seed(seed)
    warmup_data = torch.randint(0, V, (B, T), generator=g, device=device)
    data = torch.randint(0, V, (B, T), generator=g, device=device)

    # --- Run ------------------------------------------------------------ #
    stats_dense = _time_pc(pc_dense, data,
                           n_warmup=n_warmup, n_iter=n_iter,
                           warmup_data=warmup_data, label="dense",
                           use_cuda_graphs=use_cuda_graphs)
    stats_topk = _time_pc(pc_topk, data,
                          n_warmup=n_warmup, n_iter=n_iter,
                          warmup_data=warmup_data, label=f"topk-K{K}",
                          use_cuda_graphs=use_cuda_graphs)

    _print_summary("dense", stats_dense)
    _print_summary("topk ", stats_topk)

    if stats_topk["fwd_mean"] > 0:
        print(
            f"  speedup fwd: {stats_dense['fwd_mean']/stats_topk['fwd_mean']:.2f}x"
        )
    if stats_topk["bwd_mean"] > 0:
        print(
            f"  speedup bwd: {stats_dense['bwd_mean']/stats_topk['bwd_mean']:.2f}x"
        )

    return stats_dense, stats_topk


@pytest.mark.slow
def test_topk_hmm_perf_smoke():
    """Small perf run under pytest — confirms both circuits build, the TopK
    fast path is engaged, and fwd/bwd loops complete for both paths."""
    _build_and_run(T=8, H=64, V=64, bs=16, K=8, B=4,
                   n_warmup=2, n_iter=3)


@pytest.mark.slow
def test_topk_hmm_perf_smoke_cudagraph():
    """Same smoke run but with CUDA graph capture/replay enabled on the
    inner-layer sequence — exercises the ``record_cudagraph`` /
    ``apply_cudagraph`` path on both the dense and topk circuits."""
    _build_and_run(T=8, H=64, V=64, bs=16, K=8, B=4,
                   n_warmup=2, n_iter=3, use_cuda_graphs=True)


if __name__ == "__main__":
    # Realistic-size run for standalone profiling. Knobs:
    #
    #   * ``K`` is the approximation tightness *and* the speedup knob.
    #     After replacing the per-summate ``torch.topk`` call with the
    #     batched Triton bitonic top-k in :mod:`pyjuice.layer.bitonic_topk`
    #     (chunk-sort + grouped merge with ``tl.sort`` per stage; the
    #     register-level hypercube fast path was dropped — it tripped a
    #     codegen edge in Triton 3.2.0 / dev-cuda124), measured speedups
    #     at H=8192, T=32, B=1 vs DenseSumLayer:
    #
    #         K | fwd speedup | bwd speedup
    #         --+-------------+------------
    #         16|     1.23x   |   43.60x
    #         64|     1.75x   |    8.25x
    #
    #     Backward is dominated by atomic-add work that scales with K, so
    #     small K wins big there. Forward stays a clear improvement over
    #     ``torch.topk`` across the K range we use for beam-style
    #     approximate inference (8–64).
    #
    #     K is also the unroll factor for the iterative-argmax selection
    #     kernel — K >> 128 generates a multi-hundred-stage unrolled
    #     kernel that takes minutes to JIT-compile.
    #
    #   * ``bs`` (block_size) matters: block-dense sum nodes materialise an
    #     ``edge_ids`` tensor of shape ``[2, (H/bs)²]`` long-ints on CPU at
    #     DAG build time. Keep ``bs`` in the 32–128 range for realistic H.
    #
    #   * ``B`` (batch size) is fine to push — neither layer hot path is
    #     B=1-only here. Larger B amortises kernel-launch overhead.
    _build_and_run(
        T=32,
        H=4096*8,
        V=4096,
        bs=4096*8,
        K=64,
        B=1,
        n_warmup=1,
        n_iter=2,
        use_cuda_graphs=True,
    )
