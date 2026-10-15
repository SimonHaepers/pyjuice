"""Full training and conditional-query measurements; run through gpuq.

Uses the same public builders as the parity suite. Reports host submission,
end-to-end wall time, CUDA timeline time and allocator peak memory. CUDA
elapsed time includes launch gaps; these are end-to-end measurements, not
an estimate of isolated kernel occupancy.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch
import pyjuice as juice
from pyjuice.structures import MonarchHMM, SparseMonarchHMM, SparseHMM, random_emission_csc
from pyjuice.queries import conditional
from pyjuice.queries.conditional import _sparse_categorical_backward


def measure(fn, repeats):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall, submit, gpu = [], [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn()
        submit.append((time.perf_counter() - t0) * 1000)
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        gpu.append(start.elapsed_time(end))
    return dict(wall_ms=statistics.median(wall), submit_ms=statistics.median(submit),
                cuda_timeline_ms=statistics.median(gpu),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def build(model, H, T, V, bs, fuse):
    torch.manual_seed(90)
    if model == 'monarch':
        root = MonarchHMM(T, H, V, block_size=bs)
    else:
        indptr, indices, values, _ = random_emission_csc(H, V, 0.05, seed=90)
        builder = SparseMonarchHMM if model == 'sparse_monarch' else SparseHMM
        root = builder(T, H, V, indptr, indices, values, block_size=bs)
    root.init_parameters()
    import inspect
    options = {'fuse_permutations': fuse} if 'fuse_permutations' in inspect.signature(juice.TensorCircuit).parameters else {}
    return juice.TensorCircuit(root, verbose=False, **options).to(torch.device('cuda:0'))


def main():
    assert os.environ.get('GPUQ_GPU') == '0', 'Run only through gpuq on physical GPU 0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    rows = []
    def save(row):
        rows.append(row)
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    for model in ('monarch', 'sparse_monarch', 'sparse_io'):
        cases = [(1024, 16, 256, 32, B) for B in (1, 32, 512, 4096)]
        if args.large:
            cases.append((4096, 32, 2048, 64, 8192))
        for H, T, V, bs, B in cases:
            meta = dict(label=args.label, model=model, H=H, T=T, V=V, block_size=bs, B=B,
                        gpu=torch.cuda.get_device_name(), torch=torch.__version__)
            pc = None
            try:
                pc = build(model, H, T, V, bs, True)
                x = torch.randint(0, V, (B, T), device='cuda:0')
                def train():
                    pc(x, apply_cudagraph=False)
                    pc.backward(x, compute_param_flows=True, flows_memory=0.0,
                                allow_modify_flows=False, apply_cudagraph=False)
                    pc.mini_batch_em(step_size=0.1, pseudocount=0.1)
                save(dict(meta, phase='train_forward_backward_em', **measure(train, args.repeats)))
                if B <= 32:
                    save(dict(meta, phase='forward_eager', **measure(lambda: pc(x, apply_cudagraph=False), args.repeats)))
                    pc(x, record_cudagraph=True)
                    save(dict(meta, phase='forward_graph', **measure(lambda: pc(x), args.repeats)))
                    if model != 'monarch':
                        mask = torch.zeros(T, dtype=torch.bool, device='cuda:0')
                        mask[T // 2] = True
                        def query(graph=False):
                            return conditional(pc, x, missing_mask=mask, target_vars=[T // 2],
                                               allow_modify_flows=False, record_cudagraph=graph,
                                               apply_cudagraph=graph)
                        save(dict(meta, phase='conditional_eager', **measure(query, args.repeats)))
                        query(True)
                        save(dict(meta, phase='conditional_graph', **measure(lambda: query(True), args.repeats)))
                        def project():
                            return [_sparse_categorical_backward(layer, x, pc.node_flows, pc.node_mars,
                                                               target_vars=[T // 2])
                                    for layer in pc.input_layer_group]
                        save(dict(meta, phase='conditional_projection', **measure(project, args.repeats)))
            except Exception as exc:
                save(dict(meta, phase='error', error=repr(exc)))
            finally:
                pc = None
                from pyjuice.layer.sparse_io_sum_layer import _WS_SCRATCH, _WS_W_OWNER
                _WS_W_OWNER.clear()
                _WS_SCRATCH.clear()
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
