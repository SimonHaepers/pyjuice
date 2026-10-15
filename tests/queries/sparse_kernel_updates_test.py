"""Regression coverage for device patterns, query backward and Monarch fusion."""
import importlib
import inspect

import pytest
import torch
import triton

import pyjuice as juice
from pyjuice.queries import conditional
from pyjuice.queries.conditional import _conditional_fw_input_fn
from pyjuice.structures import SparseMonarchHMM, random_emission_csc
from pyjuice.utils.kernel_launcher import STABLE_BUFFER_POINTERS

MODULES = (
    'layer.block_diagonal_sum_layer', 'layer.sparse_input_sum_layer',
    'layer.sparse_io_sum_layer', 'layer.sparse_io_block_diagonal_sum_layer',
    'layer.sparse_input_block_diagonal_sum_layer', 'layer.sparse_output_block_diagonal_sum_layer',
    'layer.sparse_prod_layer', 'layer.co_sparse_prod_layer',
    'nodes.distributions.sparse_categorical', 'model.sparse_cudagraph', 'queries.conditional',
)


def test_runtime_specialization_contract():
    """CPU-only: every runtime argument follows the same policy."""
    count = 0
    for name in MODULES:
        module = importlib.import_module('pyjuice.' + name)
        objects = list(vars(module).values())
        for obj in list(objects):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                objects.extend(vars(obj).values())
        for obj in objects:
            if isinstance(obj, staticmethod):
                obj = obj.__func__
            if isinstance(obj, triton.runtime.JITFunction) and obj.__module__ == module.__name__:
                for arg in obj.params:
                    if arg.name in ("batch_size", "B"):
                        assert arg.is_constexpr, (name, obj.__name__, arg.name)
                    if not arg.is_constexpr:
                        dynamic = arg.name not in STABLE_BUFFER_POINTERS
                        assert arg.do_not_specialize == dynamic, (name, obj.__name__, arg.name)
                        assert arg.do_not_specialize_on_alignment == dynamic, (name, obj.__name__, arg.name)
                count += 1
    assert count > 25


def build(fuse=True, H=64, T=4, V=11, bs=8):
    indptr, indices, values, _ = random_emission_csc(H, V, 0.5, seed=718)
    torch.manual_seed(123)
    root = SparseMonarchHMM(T, H, V, indptr, indices, values,
                            block_size=bs, output_permutation=True)
    root.init_parameters()
    return juice.TensorCircuit(root, verbose=False, fuse_permutations=fuse).to(torch.device('cuda:0'))


@pytest.mark.parametrize('B', [1, 7, 33])
def test_fused_permutation_training_matches_unfused(B):
    a, b = build(True), build(False)
    x = torch.randint(0, 11, (B, 4), device='cuda:0')
    assert any(getattr(l, '_fused_permutation', False) for g in a.inner_layer_groups for l in g)
    ya = a(x, apply_cudagraph=False, force_use_fp32=True)
    yb = b(x, apply_cudagraph=False, force_use_fp32=True)
    torch.testing.assert_close(ya, yb, atol=3e-5, rtol=3e-5)
    for pc in (a, b):
        pc.backward(x, flows_memory=0.0, allow_modify_flows=False, force_use_fp32=True)
    torch.testing.assert_close(a.param_flows, b.param_flows, atol=3e-5, rtol=3e-5)
    for la, lb in zip(a.input_layer_group, b.input_layer_group):
        torch.testing.assert_close(la.param_flows, lb.param_flows, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('B', [1, 3, 17])
def test_batched_conditional_masks_and_inference_statistics(B, capture):
    pc, reference = build(), build(False)
    train = torch.randint(0, 11, (B, 4), device='cuda:0')
    pc(train, apply_cudagraph=False)
    pc.backward(train, flows_memory=0.0, allow_modify_flows=False)
    before = [pc.param_flows.clone()] + [l.param_flows.clone() for l in pc.input_layer_group]
    for turn in range(2):
        x = torch.randint(0, 11, (B, 4), device='cuda:0')
        mask = torch.rand(B, 4, device='cuda:0') < 0.5
        mask[:, turn] = True
        expected = torch.cat([
            conditional(reference, x[i:i+1], missing_mask=mask[i],
                        fw_input_fn=_conditional_fw_input_fn,
                        allow_modify_flows=False, apply_cudagraph=False)
            for i in range(B)
        ])
        got = conditional(pc, x, missing_mask=mask,
                          allow_modify_flows=False, record_cudagraph=capture,
                          apply_cudagraph=capture)
        torch.testing.assert_close(got, expected, atol=5e-5, rtol=5e-5)
        assert pc._run_data_cpu is None
        if capture:
            assert pc._sparse_graph_fwd.graph is not None
            assert pc._sparse_graph_fwd.backward_bindings
        after = [pc.param_flows] + [l.param_flows for l in pc.input_layer_group]
        for saved, now in zip(before, after):
            torch.testing.assert_close(saved, now, rtol=0, atol=0)


def test_zero_count_em_preserves_rows():
    from pyjuice.nodes.distributions.sparse_categorical import _sparse_cat_em_update_kernel
    params = torch.tensor([0.4, 0.6], device='cuda:0')
    before = params.clone()
    flows = torch.zeros_like(params)
    rows = torch.zeros(2, dtype=torch.int64, device='cuda:0')
    sums = torch.zeros(1, device='cuda:0')
    _sparse_cat_em_update_kernel[(1,)](params, flows, rows, sums,
                                      0, 0, 2, 0.5, 0.0,
                                      keep_zero_params=False, BLOCK=128)
    torch.testing.assert_close(params, before, rtol=0, atol=0)


@pytest.mark.parametrize('B', [1, 3])
def test_new_tokens_do_not_add_sparse_kernel_variants(monkeypatch, B):
    # Prewarm modes at one batch size, then reject any compilation. Offsets,
    # support sizes and sliced pointer alignments vary across these tokens.
    import triton.runtime.jit as jit
    pc = build()
    x = torch.zeros((B, 4), device='cuda:0', dtype=torch.long)
    mask = torch.zeros((B, 4), device='cuda:0', dtype=torch.bool)
    conditional(pc, x, missing_mask=mask, allow_modify_flows=False, apply_cudagraph=False)
    def reject(*args, **kwargs):
        raise AssertionError('Unexpected compilation after profile warmup')
    monkeypatch.setattr(jit.JITFunction, '_do_compile', reject)
    for token in range(1, 11):
        x.fill_(token)
        mask[:, token % 4] = ~mask[:, token % 4]
        conditional(pc, x, missing_mask=mask, allow_modify_flows=False, apply_cudagraph=False)


def test_bd_sparse_input_uses_block_local_maximum():
    from pyjuice.layer.sparse_input_block_diagonal_sum_layer import _fw_sparse_in_bd_kernel
    device = 'cuda:0'
    result = torch.empty((4, 1), device=device)
    weights = torch.full((8,), 0.5, device=device)
    indices = torch.tensor([0, 2], device=device)
    values = torch.tensor([-100.0, 0.0], device=device)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    maximum = torch.tensor([0.0], device=device)
    def run():
        _fw_sparse_in_bd_kernel[(2, 1)](
            result, weights, indices, values, offsets, maximum, values,
            0, 0, 1, 0, 0, 2, 2,
            BS_PADDED=2, MAX_K_IN=1, IS_BATCHED=False,
        )
    run()
    expected = torch.tensor([-100.0, -100.0, 0.0, 0.0], device=device) - torch.log(torch.tensor(2.0, device=device))
    torch.testing.assert_close(result[:, 0], expected, rtol=1e-6, atol=1e-5)
    values.fill_(float('-inf'))
    run()
    assert torch.isneginf(result).all()


def test_dense_backward_graph_warmup_preserves_accumulation():
    from pyjuice.structures import MonarchHMM
    def dense():
        torch.manual_seed(332)
        root = MonarchHMM(4, 64, 11, block_size=8)
        root.init_parameters()
        return juice.TensorCircuit(root, verbose=False).to(torch.device('cuda:0'))
    a, b = dense(), dense()
    x = torch.randint(0, 11, (7, 4), device='cuda:0')
    for pc in (a, b):
        pc(x, apply_cudagraph=False)
        pc.backward(x, flows_memory=0.0, allow_modify_flows=False, apply_cudagraph=False)
    a(x, record_cudagraph=True)
    a.backward(x, flows_memory=0.5, allow_modify_flows=False, record_cudagraph=True)
    b(x, apply_cudagraph=False)
    b.backward(x, flows_memory=0.5, allow_modify_flows=False, apply_cudagraph=False)
    torch.testing.assert_close(a.param_flows, b.param_flows, atol=1e-5, rtol=1e-5)
    before = a.param_flows.clone()
    a(x)
    a.backward(x, compute_param_flows=False, allow_modify_flows=False, record_cudagraph=True)
    torch.testing.assert_close(a.param_flows, before, atol=0, rtol=0)
    assert len([k for k in a._recorded_cuda_graphs if k[0] == 1]) == 2
