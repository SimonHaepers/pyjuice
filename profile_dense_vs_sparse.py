"""Back-to-back forward + conditional on sparse vs dense SumLayer for profiling.

Run under a profiler, e.g.:
    nsys profile -t cuda,nvtx -o dense_vs_sparse -f true \\
        pixi run -e dev python profile_dense_vs_sparse.py

    ncu --set full --nvtx --nvtx-include "dense_bf16_forward/" \\
        pixi run -e dev python profile_dense_vs_sparse.py

NVTX ranges emitted per iteration:
    sparse_forward / dense_forward / dense_bf16_forward
    sparse_conditional / dense_conditional / dense_bf16_conditional

Defaults target a mid-size HMM; override via argv.
"""
from __future__ import annotations

import argparse
import time

import torch
import pyjuice as juice


nvtx = torch.cuda.nvtx


def build_triplet(seed: int, K: int, T: int, V: int, device: torch.device):
    """Compile the same HMM three times (sparse, dense-fp32, dense-bf16) with
    identical initial params."""
    torch.manual_seed(seed)
    ns_a = juice.structures.HMM(seq_length=T, num_latents=K, num_emits=V, homogeneous=False)
    pc_sparse = juice.TensorCircuit(ns_a, use_dense_sum_layer=False, verbose=False).to(device)

    torch.manual_seed(seed)
    ns_b = juice.structures.HMM(seq_length=T, num_latents=K, num_emits=V, homogeneous=False)
    pc_dense = juice.TensorCircuit(ns_b, use_dense_sum_layer=True, verbose=False).to(device)
    pc_dense.params.data.copy_(pc_sparse.params.data)

    torch.manual_seed(seed)
    ns_c = juice.structures.HMM(seq_length=T, num_latents=K, num_emits=V, homogeneous=False)
    pc_dense_bf16 = juice.TensorCircuit(
        ns_c, use_dense_sum_layer=True, verbose=False, param_dtype=torch.bfloat16,
    ).to(device)
    pc_dense_bf16.params.data.copy_(pc_sparse.params.data.to(torch.bfloat16))

    return pc_sparse, pc_dense, pc_dense_bf16


def run_forward(pc, data, label: str, n_warmup: int, n_iter: int):
    for _ in range(n_warmup):
        pc(data)
    torch.cuda.synchronize()

    nvtx.range_push(f"{label}_block")
    t0 = time.time()
    for i in range(n_iter):
        nvtx.range_push(f"{label}_iter_{i}")
        pc(data)
        nvtx.range_pop()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_iter * 1000
    nvtx.range_pop()
    print(f"  {label:28s} {ms:7.3f} ms/iter")
    return ms


def run_conditional(pc, data, target_vars, label: str, n_warmup: int, n_iter: int):
    for _ in range(n_warmup):
        juice.queries.conditional(pc, data, target_vars=target_vars)
    torch.cuda.synchronize()

    nvtx.range_push(f"{label}_block")
    t0 = time.time()
    for i in range(n_iter):
        nvtx.range_push(f"{label}_iter_{i}")
        juice.queries.conditional(pc, data, target_vars=target_vars)
        nvtx.range_pop()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n_iter * 1000
    nvtx.range_pop()
    print(f"  {label:28s} {ms:7.3f} ms/iter")
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1024, help="num_latents")
    parser.add_argument("--T", type=int, default=32, help="seq_length")
    parser.add_argument("--V", type=int, default=2000, help="num_emits")
    parser.add_argument("--B", type=int, default=256, help="batch size")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-conditional", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required."
    device = torch.device("cuda:0")

    print(f"HMM K={args.K} T={args.T} V={args.V} B={args.B}  "
          f"warmup={args.warmup} iters={args.iters}")
    pc_sparse, pc_dense, pc_dense_bf16 = build_triplet(
        args.seed, args.K, args.T, args.V, device,
    )
    data = torch.randint(0, args.V, (args.B, args.T), device=device)

    print("Forward:")
    ms_s  = run_forward(pc_sparse,      data, "sparse_forward",     args.warmup, args.iters)
    ms_d  = run_forward(pc_dense,       data, "dense_forward",      args.warmup, args.iters)
    ms_db = run_forward(pc_dense_bf16,  data, "dense_bf16_forward", args.warmup, args.iters)
    print(f"  speedup (sparse/dense):      {ms_s / ms_d:.2f}x")
    print(f"  speedup (sparse/dense_bf16): {ms_s / ms_db:.2f}x")
    print(f"  speedup (dense/dense_bf16):  {ms_d / ms_db:.2f}x")

    if not args.skip_conditional:
        target_vars = list(range(args.T // 2))
        print("Conditional (fwd + bwd):")
        ms_cs  = run_conditional(pc_sparse,     data, target_vars, "sparse_conditional",     args.warmup, args.iters)
        ms_cd  = run_conditional(pc_dense,      data, target_vars, "dense_conditional",      args.warmup, args.iters)
        ms_cdb = run_conditional(pc_dense_bf16, data, target_vars, "dense_bf16_conditional", args.warmup, args.iters)
        print(f"  speedup (sparse/dense):      {ms_cs / ms_cd:.2f}x")
        print(f"  speedup (sparse/dense_bf16): {ms_cs / ms_cdb:.2f}x")
        print(f"  speedup (dense/dense_bf16):  {ms_cd / ms_cdb:.2f}x")


if __name__ == "__main__":
    main()
