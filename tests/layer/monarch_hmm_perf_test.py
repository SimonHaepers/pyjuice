"""
Perf harness comparing a **dense** H×H HMM transition to a **Monarch**-factored
HMM transition (``BD₁ → permutation → BD₂``).

* Dense path: ``DenseCategorical`` input + plain ``ProdLayer`` +
  ``DenseSumLayer``. The H×H transition is one tied ``summate(num_node_blocks=
  H/bs)`` call duplicated across timesteps.

* Monarch path: same input + plain ``ProdLayer`` + ``BlockDiagonalSumLayer``.
  Each timestep's transition is the chain
  ``summate(BD) → multiply(permutation) → summate(BD)`` (one Monarch matrix).
  Both BDs are tied across timesteps via ``duplicate(..., tie_params=True)``.
  Block factorisation ``H = NB · bs`` with ``NB = bs = sqrt(H)`` — keeps the
  parameter count at ``2·NB²·bs²`` ⁼ ``2·H·sqrt(H)`` instead of ``H²``.

Both paths use ``homogeneous=True`` (tied transitions + tied emissions); the
DAG construction stays at a single shared transition (or BD pair) and a
single shared emission table — untied duplicates at realistic HMM sizes
balloon CPU memory.

Backward is run with ``compute_param_flows=False, _inner_layers_only=True``
so the timed region is the prod / sum kernels (and the BD layer's own
backward), not the input-layer param-flow accumulation.

NVTX ranges (visible in ``nsys-ui``):
  * ``build/dense``, ``build/monarch`` — DAG construction + compile
  * ``{label}/warmup`` and ``{label}/timed`` — outer phases per path
  * ``{label}/{phase}/iter-{i}/{fwd|bwd}`` — per-iteration

Profile with::

    nsys profile --trace=cuda,nvtx --output mon_hmm \\
        pixi run -e dev python tests/layer/monarch_hmm_perf_test.py

The ``test_monarch_hmm_perf_smoke`` case is marked ``slow``; the ``__main__``
block is the realistic-size run intended for standalone profiling.
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


def _bd_edge_ids(NB: int) -> torch.Tensor:
    """``[[0,1,...,NB-1], [0,1,...,NB-1]]`` — the canonical block-diagonal
    pattern picked up by :class:`BlockDiagonalSumLayer`."""
    return torch.arange(0, NB)[None, :].repeat(2, 1)


def _monarch_permutation(H: int, permute_block_size: int) -> torch.Tensor:
    """Reshape-transpose-flatten permutation used by pc-arena's
    ``_create_monarch_layer``. ``[H, 1]`` shape, consumed by
    ``multiply(..., edge_ids=permuted, sparse_edges=True)`` so the
    re-pairing happens at the per-node (not per-block) granularity —
    necessary because the Monarch transpose crosses block boundaries."""
    return (
        torch.arange(0, H)
        .reshape(H // permute_block_size, permute_block_size)
        .permute(1, 0)
        .reshape(H)[:, None]
    )


def _build_dense_hmm_dag(T: int, H: int, V: int, bs: int):
    """Homogeneous (tied) HMM with a single dense H×H transition shared
    across timesteps. ``DenseSumLayer`` is selected by
    ``use_dense_sum_layer=True`` at compile time."""
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


def _build_monarch_hmm_dag(T: int, H: int, V: int, bs: int):
    """Same HMM topology as :func:`_build_dense_hmm_dag`, but each
    timestep's transition is a single Monarch matrix
    (``BD₁ → permutation → BD₂``). Both BDs are tied across timesteps.

    Requires ``H == NB · bs`` with ``NB`` chosen so the BD layer's
    square-block invariant (``bs == cbs == bs``) holds — call sites
    typically pick ``NB = bs = sqrt(H)``.
    """
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
            # First BD ("input mixing"): combines incoming state slots
            # within each block.
            if ns_bd1 is None:
                bd1 = summate(curr_zs, edge_ids=bd_edges, block_size=bs)
                ns_bd1 = bd1
            else:
                bd1 = ns_bd1.duplicate(curr_zs, tie_params=True)
            # Permutation product: re-addresses BD₁'s outputs into the
            # block structure BD₂ expects.
            np_perm = multiply(bd1, edge_ids=perm, sparse_edges=True)
            # Second BD ("output mixing").
            if ns_bd2 is None:
                bd2 = summate(np_perm, edge_ids=bd_edges, block_size=bs)
                ns_bd2 = bd2
            else:
                bd2 = ns_bd2.duplicate(np_perm, tie_params=True)
            curr_zs = multiply(curr_xs, bd2)
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

    Backward runs with ``compute_param_flows=False, _inner_layers_only=True``
    because :class:`BlockDiagonalSumLayer` is inference-only (it accepts
    ``param_flows`` but ignores them — see plan
    ``look-i-m-trying-to-eventual-waterfall.md``). The same call shape
    keeps both paths apples-to-apples and confines the timed region to
    the inner sum / prod kernels.

    With ``use_cuda_graphs=True`` we pass ``record_cudagraph=True`` on the
    first warmup call so :class:`TensorCircuit` captures the inner-layer
    kernel sequence on a side stream and replays it on every subsequent
    call. Input-layer forward + root-flow init still run on the live
    stream every call, so per-batch data dependence is preserved.
    """
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


def _build_and_run(T: int, H: int, V: int, bs: int, B: int,
                   n_warmup: int, n_iter: int, seed: int = 0,
                   use_cuda_graphs: bool = False):
    """Build both circuits, sanity-check dispatch, run paired timed loops.

    ``H`` must equal ``(H // bs) · bs`` — the Monarch builder enforces
    square BD blocks, so ``bs`` should typically be ``sqrt(H)``.
    """
    assert torch.cuda.is_available(), "this perf test requires CUDA"
    assert H % bs == 0, f"H={H} must be divisible by bs={bs}"
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

    with torch.cuda.nvtx.range("build/monarch"):
        root_monarch = _build_monarch_hmm_dag(T, H, V, bs)
        pc_monarch = TensorCircuit(
            root_monarch,
            # Irrelevant for the BD sums (they take their own dispatch
            # branch); kept on so the root summate hits DenseSumLayer.
            use_dense_sum_layer=True,
            device=device,
            verbose=False,
        )

    # --- Verify expected layer classes are actually in use --------------- #
    from pyjuice.layer import (
        DenseCategoricalInputLayer, DenseSumLayer, BlockDiagonalSumLayer,
    )
    dense_input_layers = [l for l in pc_dense.input_layer_group
                          if isinstance(l, DenseCategoricalInputLayer)]
    assert dense_input_layers, "dense path should compile DenseCategoricalInputLayer"
    dense_sum_layers = [l for lg in pc_dense.inner_layer_groups for l in lg
                        if isinstance(l, DenseSumLayer)]
    assert dense_sum_layers, "dense path should compile DenseSumLayer"

    bd_layers = [l for lg in pc_monarch.inner_layer_groups for l in lg
                 if isinstance(l, BlockDiagonalSumLayer)]
    assert bd_layers, "monarch path should compile BlockDiagonalSumLayer"
    # Two BD sums per Monarch transition, one transition per (T-1) timestep,
    # but tied duplicates fuse into the same layer groups — assert at least
    # one BD layer per timestep emerged.
    assert len(bd_layers) >= 2 * (T - 1), (
        f"expected ≥{2 * (T - 1)} BD layers (2 per Monarch transition × "
        f"T-1 timesteps), got {len(bd_layers)}"
    )

    num_node_blocks = H // bs
    dense_params = pc_dense.num_sum_params
    monarch_params = pc_monarch.num_sum_params
    print(
        f"\n=== HMM perf: T={T}, H={H}, V={V}, bs={bs}, NB={num_node_blocks}, B={B}"
        f"{' [cuda graphs]' if use_cuda_graphs else ''} ===\n"
        f"  dense:   {len(dense_sum_layers)} DenseSumLayer(s), "
        f"{dense_params} sum params (≈ H² = {H*H})\n"
        f"  monarch: {len(bd_layers)} BlockDiagonalSumLayer(s), "
        f"{monarch_params} sum params (≈ 2·NB·bs² = {2*num_node_blocks*bs*bs})"
    )

    # --- Data ----------------------------------------------------------- #
    g = torch.Generator(device=device).manual_seed(seed)
    warmup_data = torch.randint(0, V, (B, T), generator=g, device=device)
    data = torch.randint(0, V, (B, T), generator=g, device=device)

    # --- Run ------------------------------------------------------------ #
    stats_dense = _time_pc(pc_dense, data,
                           n_warmup=n_warmup, n_iter=n_iter,
                           warmup_data=warmup_data, label="dense",
                           use_cuda_graphs=use_cuda_graphs)
    stats_monarch = _time_pc(pc_monarch, data,
                             n_warmup=n_warmup, n_iter=n_iter,
                             warmup_data=warmup_data, label="monarch",
                             use_cuda_graphs=use_cuda_graphs)

    _print_summary("dense  ", stats_dense)
    _print_summary("monarch", stats_monarch)

    if stats_monarch["fwd_mean"] > 0:
        print(
            f"  speedup fwd: "
            f"{stats_dense['fwd_mean']/stats_monarch['fwd_mean']:.2f}x"
        )
    if stats_monarch["bwd_mean"] > 0:
        print(
            f"  speedup bwd: "
            f"{stats_dense['bwd_mean']/stats_monarch['bwd_mean']:.2f}x"
        )

    return stats_dense, stats_monarch


@pytest.mark.slow
def test_monarch_hmm_perf_smoke():
    """Small perf run under pytest — confirms both circuits build, the BD
    fast path is engaged on the Monarch side, and fwd/bwd loops complete
    for both paths. ``H=64, bs=8`` gives ``NB=8`` BD blocks per transition."""
    _build_and_run(T=8, H=64, V=64, bs=8, B=4,
                   n_warmup=2, n_iter=3)


@pytest.mark.slow
def test_monarch_hmm_perf_smoke_cudagraph():
    """Same smoke run but with CUDA graph capture/replay enabled on the
    inner-layer sequence — exercises the ``record_cudagraph`` /
    ``apply_cudagraph`` path on both circuits."""
    _build_and_run(T=8, H=64, V=64, bs=8, B=4,
                   n_warmup=2, n_iter=3, use_cuda_graphs=True)


if __name__ == "__main__":
    # Realistic-size run for standalone profiling.
    #
    # Picking ``H = NB · bs`` with NB = bs = sqrt(H) gives the natural
    # Monarch factorisation (parameter count 2·H·sqrt(H)). Power-of-two
    # ``bs`` is required by ``set_block_size``, so pick ``H`` such that
    # ``sqrt(H)`` is a power of two:
    #
    #     H =  4096 → NB = bs =  64   (param count: 524 288 vs 16.8 M)
    #     H = 16384 → NB = bs = 128   (param count: 4.2 M  vs 268 M)
    #     H = 65536 → NB = bs = 256   (param count: 33.6 M vs 4.3 B)
    #
    # The dense H=65536 case won't fit on most GPUs; H ∈ {4096, 16384} are
    # the meaningful comparison points. Bump ``T`` to amortise build time
    # over more inner-loop iterations; bump ``B`` to amortise launch
    # overhead within each iteration.
    _build_and_run(
        T=32,
        H=4096*4,
        V=4096,
        bs=128,
        B=16,
        n_warmup=1,
        n_iter=2,
        use_cuda_graphs=False,
    )
