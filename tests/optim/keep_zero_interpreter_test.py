"""CPU checks of the actual update kernels, without initializing CUDA.

Run with CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 pytest -q <this file>.
The separate GPU tests also cover layer dispatch, tied flows, and training.
"""
import os
from types import SimpleNamespace

import pytest
import torch
import triton

from pyjuice.layer import InputLayer
from pyjuice.nodes.distributions import (
    Categorical, ExternProductCategorical, SoftEvidenceCategorical,
    BlockedCategorical,
)
from pyjuice.nodes.distributions.sparse_categorical import (
    _sparse_cat_em_row_sum_kernel, _sparse_cat_em_update_kernel,
)
from pyjuice.model.backend.par_update import em_par_update, sgd_par_update


pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="CPU interpreter checks require TRITON_INTERPRET=1",
)


def _inputs(rows, cols):
    rng = torch.Generator().manual_seed(7)
    before = torch.rand(rows, cols, generator=rng) + 0.1
    before[:, :3] = torch.tensor([0.0, 1e-16, 5e-13])
    before /= before.sum(1, keepdim=True)
    flows = (torch.rand(rows, cols, generator=rng) + 0.1) * torch.arange(1, rows + 1)[:, None]
    # A positive weight may stay tiny after updating, and must survive again.
    flows[:, 1] = 1e-15
    return before, flows


def _target(before, flows, prior, keep):
    counts = flows.double() + prior
    if keep:
        counts[before == 0] = 0
    return counts / counts.sum(1, keepdim=True)


def _check(actual, expected, before, keep):
    torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)
    tiny = (before > 0) & (before < 1e-12)
    torch.testing.assert_close(actual[tiny].double(), expected[tiny], atol=0, rtol=2e-5)
    torch.testing.assert_close(actual.double().sum(1), torch.ones(actual.shape[0], dtype=torch.float64),
                               atol=2e-6, rtol=0)
    if keep:
        assert torch.equal(actual == 0, before == 0)
    else:
        assert (actual[before == 0] > 0).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("cls", [Categorical, ExternProductCategorical, SoftEvidenceCategorical])
@pytest.mark.parametrize("cols", [256, 257, 50262])
@pytest.mark.parametrize("step,prior,zero_flow", [(1., 0., False), (.25, .3, False), (1., 1e-6, True)])
@pytest.mark.parametrize("keep", [False, True])
def test_dense_categorical_kernel(cls, cols, step, prior, zero_flow, keep):
    rows = 11  # unequal totals, full eight-row tile plus masked tail
    before, flows = _inputs(rows, cols)
    if zero_flow:
        flows.zero_()
        # A tiny retained prior mass exposes cancellation from adding and
        # subtracting the full pseudocount around the masked normalizer.
        before[:, 7:] = 0
        before /= before.sum(1, keepdim=True)
    actual = before.clone()
    dist = cls(num_cats=cols)
    kernel = InputLayer._compile_triton_kernel(InputLayer._em_kernel_template, em_fn=dist.get_em_fn())
    block = getattr(dist, "em_block_size", 1024)
    starts = torch.arange(rows, dtype=torch.int64) * cols
    expected = before.double()
    target = _target(before, flows, prior / cols, keep)
    for _ in range(2):
        kernel[(triton.cdiv(rows, block),)](
            actual, flows, starts, starts, torch.tensor([cols], dtype=torch.float32),
            torch.zeros(rows, dtype=torch.int64), torch.arange(rows, dtype=torch.int64),
            torch.tensor([step, prior], dtype=torch.float32), rows, keep, block,
        )
        expected = (1 - step) * expected + step * target
        _check(actual, expected, before, keep)


@pytest.mark.parametrize("step,prior", [(1., 0.), (1., 1e-6), (.25, .3)])
@pytest.mark.parametrize("keep", [False, True])
def test_sparse_categorical_kernels(step, prior, keep):
    rows, cols = 11, 257
    before, flows = _inputs(rows, cols)
    # CSC slot order, with unequal support lengths per row and a partial tile.
    mask = torch.arange(cols)[None, :] < (torch.arange(rows)[:, None] * 7 + 5)
    col, row = mask.T.nonzero(as_tuple=True)
    before[~mask] = 0
    before /= before.sum(1, keepdim=True)
    values = before[row, col].clone()
    packed_flows = flows[row, col].contiguous()
    counts = flows.double() + prior  # packed distributions use a per-slot prior
    counts[~mask] = 0
    if keep:
        counts[before == 0] = 0
    target = counts / counts.sum(1, keepdim=True)
    expected = before.double()
    for _ in range(2):
        totals = torch.zeros(rows)
        grid = (triton.cdiv(len(values), 256),)
        _sparse_cat_em_row_sum_kernel[grid](values, packed_flows, row.contiguous(), totals,
                                           0, 0, len(values), prior, keep, 256)
        _sparse_cat_em_update_kernel[grid](values, packed_flows, row.contiguous(), totals,
                                          0, 0, len(values), step, prior, keep, 256)
        expected = (1 - step) * expected + step * target
        actual = torch.zeros_like(before)
        actual[row, col] = values
        # Absent slots are structural, even with the flag disabled.
        torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)
        tiny = (before > 0) & (before < 1e-12)
        torch.testing.assert_close(actual[tiny].double(), expected[tiny], atol=0, rtol=2e-5)
        torch.testing.assert_close(actual.double().sum(1), torch.ones(rows, dtype=torch.float64), atol=2e-6, rtol=0)
        if keep:
            assert torch.equal(actual > 0, before > 0)
        else:
            assert torch.equal(actual > 0, mask)


@pytest.mark.parametrize("step,prior", [(1., 0.), (1., 1e-6), (.25, .3)])
@pytest.mark.parametrize("keep", [False, True])
def test_blocked_categorical_update(step, prior, keep):
    rows, cols = 8, 257
    dist = BlockedCategorical(num_cats=cols)
    blocks = torch.arange(cols) % 2
    dist.set_meta_parameters(rows, token_block=blocks, num_blocks=2)
    before, flows = _inputs(rows, cols)
    mask = torch.arange(rows)[:, None] // 4 == blocks[None, :]
    before[~mask] = 0
    before /= before.sum(1, keepdim=True)
    row = blocks[:, None] * 4 + torch.arange(4)
    col = torch.arange(cols)[:, None].expand(cols, 4)
    values = before[row, col].reshape(-1).clone()
    node = SimpleNamespace(dist=dist, _param_range=(0, values.numel()),
                           _param_flow_range=(0, values.numel()), is_tied=lambda: False)
    layer = SimpleNamespace(nodes=[node], params=values, param_flows=flows[row, col].reshape(-1))
    counts = flows.double() + prior
    counts[~mask] = 0
    if keep:
        counts[before == 0] = 0
    target = counts / counts.sum(1, keepdim=True)
    expected = before.double()
    for _ in range(2):
        dist.custom_em(layer, step_size=step, pseudocount=prior, keep_zero_params=keep)
        expected = (1 - step) * expected + step * target
        actual = torch.zeros_like(before)
        actual[row, col] = values.reshape(cols, 4)
        torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)
        tiny = (before > 0) & (before < 1e-12)
        torch.testing.assert_close(actual[tiny].double(), expected[tiny], atol=0, rtol=2e-5)
        torch.testing.assert_close(actual.double().sum(1), torch.ones(rows, dtype=torch.float64), atol=2e-6, rtol=0)
        assert torch.equal(actual > 0, before > 0 if keep else mask)


def _sum_args(rows, cols):
    ids = torch.arange(rows, dtype=torch.int64)
    return [ids * cols, ids * cols, torch.full((rows,), cols, dtype=torch.int64),
            torch.ones(rows, dtype=torch.int64), ids, torch.full((rows,), cols, dtype=torch.int64),
            None, {"tot_num_nodes": rows, "BLOCK_SIZE": cols}]


@pytest.mark.parametrize("step,prior", [(1., 0.), (1., .3), (.25, .3)])
@pytest.mark.parametrize("keep", [False, True])
def test_sum_em_kernels(step, prior, keep):
    before, flows = _inputs(11, 8)
    actual, expected = before.clone(), before.double()
    target = _target(before, flows, prior / 8, keep)
    for _ in range(2):
        em_par_update(actual, flows, _sum_args(11, 8), step, prior, keep)
        expected = (1 - step) * expected + step * target
        _check(actual, expected, before, keep)


@pytest.mark.parametrize("keep", [False, True])
def test_sum_sgd_kernel(keep):
    before, _ = _inputs(11, 8)
    actual = before.clone()
    sgd_par_update(actual, torch.ones_like(actual), _sum_args(11, 8), .1, keep)
    expected = before.double() * torch.exp(torch.tensor(.1, dtype=torch.float64))
    torch.testing.assert_close(actual.double(), expected, atol=0, rtol=5e-6)
    assert not torch.cuda.is_initialized()
