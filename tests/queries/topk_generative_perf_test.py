"""Perf harness for the **generative** conditional query on HMMs across the
**dense** and **TopK** inference pipelines.

Sibling of :mod:`tests.queries.sparse_categorical_generative_perf_test`
(dense vs sparse) and :mod:`tests.layer.topk_hmm_perf_test` (dense vs topk
on a plain forward+backward). Same circuit topology in both paths; the
difference is only that every transition ``summate`` carries ``topk=K`` on
the topk path so the compiler routes it through
:class:`TopKLayer`/:class:`TopKSumLayer` instead of
:class:`DenseSumLayer`.

The query is identical to the sparse generative test: a single position
``t*`` is unobserved (``missing_mask[t*]=True``) and the rest of the
sequence is observed; the result is ``p(x_{t*} | x_{!=t*})`` shaped
``[B, 1, V]``. The forward overwrites the unknown's input log-likelihood
with 0.0 via the dense input layer's missing-mask path, and the backward
hits a single column at the input layer via
``DenseCategoricalInputLayer.dense_conditional_backward`` with
``target_vars=[t*]``.

Both paths use ``homogeneous=True`` (tied transitions + tied emissions)
so the DAG construction stays at one shared H×H transition and one
shared emission table.

The ``test_topk_generative_perf_smoke`` case is marked ``slow``; the
``__main__`` block is the realistic-size run intended for standalone
profiling.
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
# DAG builders (mirror tests/layer/topk_hmm_perf_test.py)
# ---------------------------------------------------------------------------


def _build_dense_hmm_dag(T: int, H: int, V: int, bs: int):
    """Homogeneous (tied) HMM with ``DenseCategorical`` emissions and plain
    ``summate`` transitions — routes to :class:`DenseSumLayer` under
    ``use_dense_sum_layer=True``."""
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
    """Same topology; every transition ``summate`` is annotated with
    ``topk=K``. The annotation rides through ``duplicate(...,
    tie_params=True)`` so every timestep dispatches to
    :class:`TopKSumLayer`."""
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
# Timing / NVTX helpers
# ---------------------------------------------------------------------------


def _time_generative(pc: TensorCircuit, data: torch.Tensor,
                     missing_mask: torch.Tensor, target_t: int,
                     n_warmup: int, n_iter: int,
                     warmup_data: Optional[torch.Tensor] = None,
                     label: str = "pc",
                     use_cuda_graphs: bool = False) -> dict:
    """Warm up JIT, then run ``n_iter`` generative ``conditional`` calls and
    return per-iter wall-clock stats.

    ``conditional`` runs a forward followed by a backward with
    ``input_layer_fn`` rewired to
    :func:`pyjuice.queries.conditional._conditional_bk_input_fn`. The inner
    layer backward (TopK or dense) propagates flows from the root down to
    the input layer's scope, then ``DenseCategoricalInputLayer`` reads
    ``node_flows[..., target_var]`` to materialise the ``[1, V]`` posterior.

    With ``use_cuda_graphs=True`` we pass ``record_cudagraph=True`` on the
    first warmup call. The kwargs flow through
    :func:`pyjuice.queries.conditional` → ``query()`` → ``pc.forward`` /
    ``pc.backward`` (both already accept the flag), so
    :class:`TensorCircuit` captures the inner-layer kernel sequences on a
    side stream and replays them on every subsequent call. The conditional
    input-layer fns (missing-mask forward + ``dense_conditional_backward``)
    run outside the captured region every call, so per-batch data
    dependence and the per-call ``cat_probs`` output allocation are both
    preserved.

    NVTX ranges (visible in ``nsys-ui``):
      * ``{label}/warmup`` and ``{label}/timed`` — outer phases
      * ``{label}/{warmup|timed}/iter-{i}`` — per-iteration
    """
    if warmup_data is None:
        warmup_data = data
    target_vars = [target_t]

    record_kwargs = {"record_cudagraph": True} if use_cuda_graphs else {}

    with torch.cuda.nvtx.range(f"{label}/warmup"):
        for i in range(n_warmup):
            # Only the first warmup call triggers the capture; further calls
            # find the signature in the cache and replay (apply_cudagraph
            # defaults to True on pc.forward/backward).
            extra = record_kwargs if i == 0 else {}
            with torch.cuda.nvtx.range(f"{label}/warmup/iter-{i}"):
                juice.queries.conditional(
                    pc, data=warmup_data,
                    missing_mask=missing_mask,
                    target_vars=target_vars,
                    **extra,
                )
        torch.cuda.synchronize()

    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]

    last_out = None
    with torch.cuda.nvtx.range(f"{label}/timed"):
        for i in range(n_iter):
            with torch.cuda.nvtx.range(f"{label}/timed/iter-{i}"):
                events[i][0].record()
                last_out = juice.queries.conditional(
                    pc, data=data, missing_mask=missing_mask,
                    target_vars=target_vars,
                )
                events[i][1].record()
        torch.cuda.synchronize()

    ms = [s.elapsed_time(e) for s, e in events]
    return {
        "ms": ms,
        "mean": sum(ms) / len(ms),
        "min": min(ms),
        "max": max(ms),
        "out": last_out,
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


def _build_and_run_generative(T: int, H: int, V: int, bs: int, K: int, B: int,
                              n_warmup: int, n_iter: int, seed: int = 0,
                              target_t: Optional[int] = None,
                              use_cuda_graphs: bool = False):
    assert torch.cuda.is_available(), "this perf test requires CUDA"
    device = torch.device("cuda:0")

    if target_t is None:
        target_t = T // 2

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
            # ``use_dense_sum_layer`` only matters for sums *without* topk;
            # the root sum (no topk annotation) takes the dense path either
            # way.
            use_dense_sum_layer=True,
            device=device,
            verbose=False,
        )
    # Sync params so the two posteriors are on the same model and the only
    # difference is the topk approximation in the inner sums.
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

    topk_input_layers = [l for l in pc_topk.input_layer_group
                         if isinstance(l, DenseCategoricalInputLayer)]
    assert topk_input_layers, "topk path should compile DenseCategoricalInputLayer"
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
        f"\n=== topk generative perf: T={T}, H={H}, V={V}, bs={bs}, "
        f"K={K}, B={B}, target_t={target_t}"
        f"{' [cuda graphs]' if use_cuda_graphs else ''} ===\n"
        f"  dense: {len(dense_sum_layers)} DenseSumLayer(s), "
        f"{len(dense_input_layers)} DenseCategoricalInputLayer(s)\n"
        f"  topk:  {len(topk_layers)} TopKLayer(s) + "
        f"{len(topk_sum_layers)} TopKSumLayer(s) (K/H = {K/H:.2%})"
    )

    # --- Data ----------------------------------------------------------- #
    g = torch.Generator(device=device).manual_seed(seed)
    warmup_data = torch.randint(0, V, (B, T), generator=g, device=device)
    data = torch.randint(0, V, (B, T), generator=g, device=device)

    missing_mask = torch.zeros(T, dtype=torch.bool, device=device)
    missing_mask[target_t] = True

    # --- Run ------------------------------------------------------------ #
    stats_dense = _time_generative(
        pc_dense, data, missing_mask, target_t,
        n_warmup=n_warmup, n_iter=n_iter, warmup_data=warmup_data,
        label="dense", use_cuda_graphs=use_cuda_graphs,
    )
    stats_topk = _time_generative(
        pc_topk, data, missing_mask, target_t,
        n_warmup=n_warmup, n_iter=n_iter, warmup_data=warmup_data,
        label=f"topk-K{K}", use_cuda_graphs=use_cuda_graphs,
    )

    # --- Sanity-check the output shape + that it's a valid distribution. --
    # ``conditional`` returns ``[B, num_target_vars, num_cats]``; with
    # target_t a single var the shape is ``[B, 1, V]``. Both paths must
    # produce a per-batch probability distribution; topk introduces bias
    # but still sums to ~1 because the input-layer backward normalises by
    # the marginal at the target var.
    out_dense = stats_dense["out"]
    out_topk = stats_topk["out"]
    assert out_dense.shape == (B, 1, V), (
        f"dense generative output has unexpected shape {tuple(out_dense.shape)}"
    )
    assert out_topk.shape == (B, 1, V), (
        f"topk generative output has unexpected shape {tuple(out_topk.shape)}"
    )
    assert torch.isfinite(out_topk).all(), "topk generative output has non-finite entries"
    sum_dense = out_dense.sum(dim=2)
    sum_topk = out_topk.sum(dim=2)
    assert torch.allclose(sum_dense, torch.ones_like(sum_dense), atol=1e-3), (
        f"dense posterior does not sum to 1: max dev {(sum_dense - 1).abs().max().item():.3e}"
    )
    assert torch.allclose(sum_topk, torch.ones_like(sum_topk), atol=1e-3), (
        f"topk posterior does not sum to 1: max dev {(sum_topk - 1).abs().max().item():.3e}"
    )

    _print_summary("dense", stats_dense)
    _print_summary("topk ", stats_topk)

    if stats_topk["mean"] > 0:
        print(
            f"  speedup generative: "
            f"{stats_dense['mean']/stats_topk['mean']:.2f}x"
        )

    return stats_dense, stats_topk


@pytest.mark.slow
def test_topk_generative_perf_smoke():
    """Small perf run under pytest — confirms both circuits build, the TopK
    fast path is engaged, the conditional generative query completes on
    both paths, and each posterior is a valid probability distribution."""
    _build_and_run_generative(T=8, H=64, V=64, bs=16, K=8, B=4,
                              target_t=4, n_warmup=2, n_iter=3)


@pytest.mark.slow
def test_topk_generative_perf_smoke_cudagraph():
    """Same smoke run but with CUDA graph capture/replay enabled on the
    inner-layer forward + backward sequences — exercises the
    ``record_cudagraph`` / ``apply_cudagraph`` plumbing through
    :func:`pyjuice.queries.conditional` on both the dense and topk
    circuits. The conditional input-layer fns still run outside the
    captured region every call, so the per-call ``cat_probs`` allocation
    + ``dense_conditional_backward`` bmm aren't part of the graph."""
    _build_and_run_generative(T=8, H=64, V=64, bs=16, K=8, B=4,
                              target_t=4, n_warmup=2, n_iter=3,
                              use_cuda_graphs=True)


if __name__ == "__main__":
    # Realistic-size run for standalone profiling. Knobs match
    # ``tests/layer/topk_hmm_perf_test.py``'s realistic config so the plain
    # forward+backward and the generative query numbers are directly
    # comparable.
    _build_and_run_generative(
        T=32,
        H=4096*8,
        V=4096,
        bs=4096*8,
        K=64,
        B=1,
        target_t=16,
        n_warmup=1,
        n_iter=2,
        use_cuda_graphs=True,
    )
