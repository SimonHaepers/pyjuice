"""Packed categorical updates preserve exact zeros without a magnitude cutoff."""
import json
import os
import pytest
import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists


def fixture(kind, v):
    h = 8
    rng = torch.Generator().manual_seed(29)
    if kind == 'sparse':
        pattern = torch.rand(h, v, generator=rng) < .23
        pattern[:, :4] = True
        cols, rows = pattern.T.nonzero(as_tuple=True)
        ptr = torch.zeros(v+1, dtype=torch.long)
        ptr[1:] = torch.bincount(cols, minlength=v).cumsum(0)
        ns = juice.inputs(0, num_nodes=h, block_size=1,
                          dist=dists.SparseCategorical(num_cats=v),
                          csc_indptr=ptr, csc_indices=rows)
    else:
        blocks = torch.arange(v) % 2
        rows = (blocks[:, None]*4 + torch.arange(4)[None, :]).flatten()
        cols = torch.arange(v).repeat_interleave(4)
        ns = juice.inputs(0, num_nodes=h, block_size=1,
                          dist=dists.BlockedCategorical(num_cats=v),
                          token_block=blocks, num_blocks=2)
    before = torch.rand(len(rows), generator=rng)+.1
    for row in range(h):
        ids = torch.where(rows == row)[0]
        before[ids[:3]] = torch.tensor([0., 1e-16, 5e-13])
    sums = torch.zeros(h, dtype=torch.float64).scatter_add_(0, rows, before.double())
    before = (before.double()/sums[rows]).float()
    ns.set_params(before, normalize=False)
    root = juice.summate(ns, num_nodes=1, block_size=1)
    root.init_parameters(perturbation=0.)
    pc = juice.compile(root, verbose=False).to(torch.device('cuda:0'))
    pc.init_param_flows(flows_memory=0.)
    layer = pc.input_layer_group[0]
    ps, pe = ns._param_range
    layer.params.data[ps:pe].copy_(before.cuda())
    flows = (torch.rand(len(rows), generator=rng)+.1)*(rows+1)
    return pc, ns, rows, cols, before, flows


@pytest.mark.parametrize('kind', ['sparse', 'blocked'])
@pytest.mark.parametrize('v', [257, 50262])
@pytest.mark.parametrize('step,pseudocount', [(1., 0.), (1., 1e-6), (.25, .3)])
@pytest.mark.parametrize('keep', [False, True])
def test_packed_update_matches_reference(kind, v, step, pseudocount, keep):
    pc, ns, rows, cols, before, flows = fixture(kind, v)
    layer = pc.input_layer_group[0]
    ps, pe = ns._param_range
    fs, fe = ns._param_flow_range
    layer.param_flows[fs:fe].copy_(flows.cuda())
    zero, tiny = before == 0, (before > 0) & (before < 1e-12)
    # Existing packed kernels take pseudocount PER STORED ENTRY. Preserve this
    # convention; dense Categorical takes a per-row amount and divides by V.
    weights = flows.double()+pseudocount
    if keep:
        weights[zero] = 0.
    totals = torch.zeros(8, dtype=torch.float64).scatter_add_(0, rows, weights)
    target = weights/totals[rows]
    expected = before.double()
    for _ in range(2):
        expected = (1-step)*expected+step*target
        layer.mini_batch_em(step_size=step, pseudocount=pseudocount, keep_zero_params=keep)
        actual = layer.params[ps:pe].detach().cpu()
        torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(actual[tiny].double(), expected[tiny], atol=0, rtol=2e-5)
        totals = torch.zeros(8, dtype=torch.float64).scatter_add_(0, rows, actual.double())
        # Large packed rows use FP32 atomic/index_add reductions; check their
        # sums in float64 with a tolerance for accumulated FP32 roundoff.
        torch.testing.assert_close(totals, torch.ones(8, dtype=torch.float64), atol=1e-5, rtol=0)
        if os.environ.get('PYJUICE_REMASK_RESULTS'):
            with open(os.environ['PYJUICE_REMASK_RESULTS'], 'a') as f:
                f.write(json.dumps(dict(kind='packed_reference', layout=kind, V=v,
                         keep=keep, step=step, pseudocount=pseudocount,
                         max_row_sum_error=float((totals-1).abs().max()),
                         max_parameter_error=float((actual.double()-expected).abs().max())))+'\n')
        if keep:
            assert torch.equal(actual == 0, zero)
        else:
            assert bool((actual[zero] > 0).all())


@pytest.mark.parametrize('kind', ['sparse', 'blocked'])
def test_packed_full_em_matches_dense_remasking(kind):
    pc, ns, rows, cols, before, flows = fixture(kind, 257)
    il = pc.input_layer_group[0]
    fs, fe = ns._param_flow_range
    il.param_flows[fs:fe].copy_(flows.cuda())
    # Match the numerical prior per category, not the different API conventions.
    total_pseudocount = .3
    il.mini_batch_em(step_size=1., pseudocount=total_pseudocount/257, keep_zero_params=True)
    got = il.params[slice(*ns._param_range)].detach().cpu()
    dense_before = torch.zeros(8, 257)
    dense_flows = torch.zeros_like(dense_before)
    dense_before[rows, cols] = before
    dense_flows[rows, cols] = flows
    dn = juice.inputs(0, num_nodes=8, block_size=1, dist=dists.Categorical(num_cats=257))
    dl = juice.layer.InputLayer([dn], pc_num_vars=1)
    dl._init_parameters(perturbation=0.)
    dl.to(torch.device('cuda:0'))
    dl.init_param_flows(flows_memory=0.)
    dl.params.data.copy_(dense_before.flatten().cuda())
    dl.param_flows.copy_(dense_flows.flatten().cuda())
    dl.mini_batch_em(step_size=1., pseudocount=total_pseudocount, keep_zero_params=False)
    expected = dl.params.detach().cpu().reshape(8, 257)
    expected[dense_before == 0] = 0.
    expected /= expected.sum(1, keepdim=True)
    torch.testing.assert_close(got, expected[rows, cols], atol=2e-7, rtol=2e-5)
    assert torch.equal(got > 0, before > 0)


def test_sparse_training_preserves_stored_support():
    pc, ns, rows, cols, before, _ = fixture('sparse', 257)
    # Only observe columns that have positive support; this is a support test,
    # not a test of the kernel's separate treatment of structurally absent cells.
    valid = cols[before > 0].unique()
    data = valid[torch.arange(96) % len(valid)][:, None].cuda()
    for _ in range(3):
        pc.init_param_flows(flows_memory=0.)
        assert torch.isfinite(pc(data)).all()
        pc.backward(data, logspace_flows=False, allow_modify_flows=False)
        pc.mini_batch_em(step_size=1., pseudocount=1e-6, keep_zero_params=True)
        actual = pc.input_layer_group[0].params[slice(*ns._param_range)].detach().cpu()
        assert torch.equal(actual > 0, before > 0)
        torch.testing.assert_close(torch.zeros(8, dtype=torch.float64).scatter_add_(0, rows, actual.double()),
                                   torch.ones(8, dtype=torch.float64), atol=4e-6, rtol=0)
