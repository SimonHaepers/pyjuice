"""Bounded training factors and packed dense-buffer layout parity."""
import pytest
import torch
import pyjuice as juice
import pyjuice.layer.sparse_io_sum_layer as sio
from pyjuice.structures import SparseHMM, SparseMonarchHMM, random_emission_csc
from pyjuice.queries import conditional


def build(kind, compact, H=128, T=7, V=23, bs=32):
    cptr, rid, values, _ = random_emission_csc(H, V, 0.3, seed=504)
    torch.manual_seed(504)
    builder = SparseMonarchHMM if kind == 'monarch' else SparseHMM
    root = builder(T, H, V, cptr, rid, values, block_size=bs)
    root.init_parameters()
    return juice.TensorCircuit(root, verbose=False, compact_sparse_buffers=compact).to(torch.device('cuda:0'))


@pytest.mark.parametrize('kind', ['monarch', 'io'])
@pytest.mark.parametrize('B', [1, 17, 129])
def test_compact_training_and_conditional_match_dense_layout(kind, B):
    a, b = build(kind, True), build(kind, False)
    assert a.num_nodes < b.num_nodes
    assert a.num_logical_nodes == b.num_nodes
    for turn in range(2):
        x = torch.randint(0, 23, (B, 7), device='cuda:0')
        ya = a(x, apply_cudagraph=False)
        yb = b(x, apply_cudagraph=False)
        torch.testing.assert_close(ya, yb, rtol=0, atol=3e-5)
        for pc in (a, b):
            pc.backward(x, flows_memory=0., allow_modify_flows=False, apply_cudagraph=False)
        torch.testing.assert_close(a.param_flows, b.param_flows, atol=3e-4, rtol=3e-4)
        for la, lb in zip(a.input_layer_group, b.input_layer_group):
            torch.testing.assert_close(la.param_flows, lb.param_flows, atol=3e-4, rtol=3e-4)
        for pc in (a, b):
            pc.mini_batch_em(step_size=0.2, pseudocount=0.1)
        mask = torch.rand(B, 7, device='cuda:0') < 0.5
        mask[:, turn] = True
        qa = conditional(a, x, missing_mask=mask, record_cudagraph=True, allow_modify_flows=False)
        qb = conditional(b, x, missing_mask=mask, apply_cudagraph=False, allow_modify_flows=False)
        torch.testing.assert_close(qa, qb, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize('precision', ['ieee', 'tf32'])
@pytest.mark.parametrize('slots', [1, 2, 4])
def test_chunked_statistics_match_full_stash(monkeypatch, precision, slots):
    monkeypatch.setattr(sio, '_WS_MODE', 'always')
    monkeypatch.setattr(sio, '_WS_PRECISION', precision)
    full, chunked = build('io', True), build('io', True)
    for B, negate in [(17, False), (9, True), (17, False)]:
        x = torch.randint(0, 23, (B, 7), device='cuda:0')
        for pc, budget in [(full, 0), (chunked, slots * (128 + 128) * B * 4)]:
            monkeypatch.setattr(sio, '_WS_STASH_BYTES', budget)
            pc(x, apply_cudagraph=False)
            pc.backward(x, flows_memory=0., negate_pflows=negate, allow_modify_flows=False)
        for group in chunked._sparse_io_ws_groups:
            assert group._gt.shape[1] <= slots * B
            assert group._used == 0
        torch.testing.assert_close(full.param_flows, chunked.param_flows, atol=3e-4, rtol=3e-4)


def test_chunked_mixed_dispatch_and_accumulation(monkeypatch):
    from tests.queries.sparse_io_ws_test import (
        test_ws_mixed_dispatch_and_stash_reuse, test_ws_flow_accumulation_and_negate)
    monkeypatch.setattr(sio, '_WS_STASH_BYTES', 1)  # floor of one batch slot
    test_ws_mixed_dispatch_and_stash_reuse(monkeypatch)
    # Restore dispatch patched by the first regression before running another.
    monkeypatch.undo()
    monkeypatch.setattr(sio, '_WS_STASH_BYTES', 1)
    test_ws_flow_accumulation_and_negate(monkeypatch)
