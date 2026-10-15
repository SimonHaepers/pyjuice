"""CPU compiler and factor-buffer tests, without a CUDA context."""
from contextlib import nullcontext
from functools import reduce
from math import gcd
import pytest
import torch
import pyjuice as juice
from pyjuice.layer.input_layer import InputLayer
import pyjuice.layer.sparse_io_sum_layer as sio
from pyjuice.structures import SparseHMM, SparseMonarchHMM, random_emission_csc


@pytest.mark.parametrize('builder', [SparseHMM, SparseMonarchHMM])
def test_compact_layout_ranges_on_cpu(monkeypatch, builder):
    # InputLayer computes its metadata GCD on CUDA even when GPU compilation
    # is disabled. Replace only that reduction and the device context here.
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(InputLayer, '_get_num_nodes_gcd',
                        lambda layer: reduce(gcd, [n.num_nodes for n in layer.nodes]))
    cptr, rid, vals, _ = random_emission_csc(64, 17, 0.25, seed=10)
    counts = []
    for compact in (False, True):
        root = builder(5, 64, 17, cptr, rid, vals, block_size=16)
        root.init_parameters()
        pc = juice.compile(root, compact_sparse_buffers=compact,
                                 disable_gpu_compilation=True, verbose=False)
        assert pc._root_node_range == root._output_ind_range
        assert pc.num_logical_nodes >= pc.num_nodes
        counts.append(pc.num_nodes)
        for ns in root:
            if (ns.is_input() or ns.is_sum()) and ns._dense_storage:
                lo, hi = ns._output_ind_range
                assert 0 <= lo < hi <= pc.num_nodes
        if compact:
            for layer in pc.input_layer_group:
                if layer._packed_only:
                    with pytest.raises(NotImplementedError, match='compact_sparse_buffers=False'):
                        layer.eval_partition_fn(torch.empty(pc.num_nodes))
    assert counts[1] < counts[0]


@pytest.mark.parametrize('capacity', [1, 2, 4])
def test_chunk_reduction_and_aborted_pass_on_cpu(monkeypatch, capacity):
    # The GPU epilogue is unchanged. Replace it with a CPU sink to exercise
    # the real buffer reservation, clearing and partial GEMM reduction code.
    group = sio._WSGroup(0, 0, 1, 1, 16, 16)
    group.num_slots = 7
    B = 3
    monkeypatch.setattr(sio, '_WS_STASH_BYTES', capacity * 32 * B * 4)
    accumulator = torch.zeros(16, 16)
    class Sink:
        def __getitem__(self, grid):
            def accumulate(pflows, params, panel, r0, *args, **kwargs):
                accumulator[r0:r0+panel.size(0)].add_(panel, alpha=-1 if kwargs['NEGATE'] else 1)
            return accumulate
    monkeypatch.setattr(sio, '_ws_pflow_epilogue_kernel', Sink())
    params = torch.ones(256)
    pflows = torch.zeros(256)
    # Abandon a partial pass: none of its factors may leak into the next one.
    gt, xt, col = group.stash_slot(0, B, torch.device('cpu'), False, params, pflows)
    gt[:, col:col+B].fill_(100.)
    xt[:, col:col+B].fill_(100.)
    group.begin_pass()
    expected = torch.zeros_like(accumulator)
    for slot in [6, 1, 4, 0, 3]:  # mixed traversal; skipped slots mimic gather dispatch
        g, x = torch.rand(16, B), torch.rand(16, B)
        expected -= g @ x.t()
        gt, xt, col = group.stash_slot(slot, B, torch.device('cpu'), True, params, pflows)
        gt[:, col:col+B].copy_(g)
        xt[:, col:col+B].copy_(x)
    group.flush(params, pflows)
    torch.testing.assert_close(accumulator, expected)
    assert group._used == 0
    assert group._gt.numel() + group._xt.numel() == capacity * 32 * B


@pytest.mark.parametrize('plain', [False, True])
def test_existing_tree_builder_preserves_fallback_rows(monkeypatch, plain):
    from tests.queries.sparse_io_cudagraph_test import _build_tree
    cls = juice.TensorCircuit
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(InputLayer, '_get_num_nodes_gcd',
                        lambda layer: reduce(gcd, [n.num_nodes for n in layer.nodes]))
    monkeypatch.setattr(cls, 'to', lambda self, device: self)
    monkeypatch.setattr(juice, 'TensorCircuit',
                        lambda root, **kw: cls(root, disable_gpu_compilation=True, **kw))
    pc, _ = _build_tree(plain_subtree_roots=plain)
    found = False
    for layer in pc.input_layer_group:
        for ns in layer.nodes:
            owner = getattr(ns, '_sparse_flow_owner', None)
            if owner is not None and not owner[0]._skip_scatter:
                assert ns._dense_storage
                assert ns._output_ind_range[1] <= pc.num_nodes
                found = True
    assert found == plain


@pytest.mark.parametrize('kind', ['single_block', 'sparse_io_bd'])
def test_packed_products_need_no_dense_metadata(monkeypatch, kind):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(InputLayer, '_get_num_nodes_gcd',
                        lambda layer: reduce(gcd, [n.num_nodes for n in layer.nodes]))
    cptr, rid, vals, _ = random_emission_csc(8, 16, 0.5, seed=10)
    if kind == 'sparse_io_bd':
        from tests.queries.sparse_io_block_diagonal_param_flow_test import _build_sparse_io_bd_chain
        root, _ = _build_sparse_io_bd_chain(4, 8, 16, 4, cptr, rid, vals)
    else:
        root = SparseHMM(4, 8, 16, cptr, rid, vals, block_size=8)
    root.init_parameters()
    pc = juice.TensorCircuit(root, disable_gpu_compilation=True, verbose=False)
    from pyjuice.layer.sparse_prod_layer import SparseProdLayer
    packed = [layer for group in pc.inner_layer_groups for layer in group
              if isinstance(layer, SparseProdLayer)]
    assert packed
    assert all(layer._skip_scatter and not layer._dense_compiled for layer in packed)
    assert all(layer.num_fw_partitions == layer.num_bk_partitions == 0 for layer in packed)
