"""Compare zero-preserving EM with the experiment's exact remask procedure.

Full EM agrees for the same support, including tiny positive weights.
Partial-step tests document the different normalization/blending order.
"""
import json
import os
from pathlib import Path

import pytest
import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import InputLayer


def record(result):
    if os.environ.get('PYJUICE_REMASK_RESULTS'):
        with Path(os.environ['PYJUICE_REMASK_RESULTS']).open('a') as f:
            f.write(json.dumps(result) + '\n')


def input_layer(before):
    h, v = before.shape
    nodes = juice.inputs(0, num_nodes=h, block_size=1, dist=dists.Categorical(num_cats=v))
    layer = InputLayer([nodes], pc_num_vars=1)
    layer._init_parameters(perturbation=0.)
    layer.to(torch.device('cuda:0'))
    layer.init_param_flows(flows_memory=0.)
    layer.params.data.copy_(before.flatten().to(torch.device('cuda:0')))
    return layer


def paired_update(before, flows, step_size, pseudocount):
    outputs = []
    for preserve in [True, False]:
        layer = input_layer(before)
        layer.param_flows.copy_(flows.flatten().to(torch.device('cuda:0')))
        layer.mini_batch_em(step_size=step_size, pseudocount=pseudocount,
                            keep_zero_params=preserve)
        result = layer.params.detach().cpu().view_as(before).clone()
        if not preserve:
            result[before == 0] = 0.
            result /= result.sum(1, keepdim=True)
        outputs.append(result)
    return outputs


@pytest.mark.parametrize('num_cats', [256, 257, 512, 50262])
@pytest.mark.parametrize('pseudocount', [0., 1e-6, .3])
def test_full_em_matches_remasking(num_cats, pseudocount):
    rng = torch.Generator().manual_seed(37)
    before = torch.rand(11, num_cats, generator=rng) + .1
    mask = torch.rand(11, num_cats, generator=rng) < .05
    mask[:, 0] = True
    before *= mask
    before /= before.sum(1, keepdim=True)
    flows = (torch.rand(11, num_cats, generator=rng) + .1) * mask
    a, b = paired_update(before, flows, 1., pseudocount)
    expected = (flows.double() + pseudocount / num_cats) * mask
    expected /= expected.sum(1, keepdim=True)
    for result in [a, b]:
        torch.testing.assert_close(result.double(), expected, atol=2e-7, rtol=2e-5)
        assert torch.equal(result > 0, mask)
    torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
    record(dict(kind='same_flows', num_cats=num_cats, pseudocount=pseudocount,
                max_param_difference=float((a-b).abs().max())))


def test_tiny_positive_support_matches_exact_remasking():
    before = torch.zeros(11, 512)
    before[:, :2] = torch.tensor([1., 5e-13])
    flows = torch.zeros_like(before)
    flows[:, :2] = 1.
    a, b = paired_update(before, flows, 1., 1e-6)
    torch.testing.assert_close(a[:, 1], torch.full((11,), .5))
    torch.testing.assert_close(b[:, 1], torch.full((11,), .5))
    torch.testing.assert_close(a.sum(1), torch.ones(11))
    record(dict(kind='tiny_positive_support_equivalence', preserved=float(a[0, 1]),
                remasked=float(b[0, 1])))


def test_partial_em_has_different_projection_order():
    before = torch.zeros(11, 512)
    before[:, :2] = torch.tensor([.8, .2])
    flows = torch.zeros_like(before)
    flows[:, :2] = torch.tensor([1., 9.])
    rate, pseudocount = .25, 100.
    a, b = paired_update(before, flows, rate, pseudocount)
    weights = flows.double() + pseudocount / 512
    retained = weights * (before > 0)
    expected_a = (1-rate)*before + rate*retained/retained.sum(1, keepdim=True)
    expected_b = ((1-rate)*before + rate*weights/weights.sum(1, keepdim=True))*(before > 0)
    expected_b /= expected_b.sum(1, keepdim=True)
    torch.testing.assert_close(a.double(), expected_a, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(b.double(), expected_b, atol=2e-7, rtol=2e-5)
    assert float((a-b).abs().max()) > .1
    record(dict(kind='partial_step_counterexample', max_param_difference=float((a-b).abs().max())))


def hmm_params(pc):
    pc.update_parameters()
    h = pc.root_ns.chs[0].num_nodes
    transition = next(n for n in pc.root_ns if n.is_sum() and not n.is_tied() and n.num_nodes == h)
    emission = next(n for n in pc.root_ns if n.is_input() and not n.is_tied())
    return (transition.get_params(as_matrix=True).detach().cpu().clone(),
            emission.get_params().detach().cpu().reshape(h, -1).clone(),
            pc.root_ns.get_params(as_matrix=True).detach().cpu().reshape(-1).clone())


def exact_ll(a, b, g, data):
    a, b, g = a.double().cuda(), b.double().cuda(), g.double().cuda()
    message = b[:, data[:, -1]].T
    scale = message.sum(1)
    answer = scale.log()
    message /= scale[:, None]
    for pos in range(data.shape[1]-2, -1, -1):
        message = (message @ a.T) * b[:, data[:, pos]].T
        scale = message.sum(1)
        answer += scale.log()
        message /= scale[:, None]
    answer += (message * g[None, :]).sum(1).log()
    return answer.cpu()


@pytest.mark.parametrize('num_cats,pseudocount', [(512, 1e-6), (512, .3), (50262, .3)])
def test_joint_pruned_hmm_matches_remasking(num_cats, pseudocount):
    h, length = 32, 16
    rng = torch.Generator().manual_seed(73)
    a = torch.rand(h, h, generator=rng) + .1
    tmask = torch.rand(h, h, generator=rng) < .5
    tmask[0, :] = True
    tmask[:, 0] = True
    a *= tmask
    a /= a.sum(1, keepdim=True)
    b = torch.rand(h, num_cats, generator=rng) + .1
    emask = torch.rand(h, num_cats, generator=rng) < .05
    emask[0, :] = True  # A complete state guarantees an exact nonzero path.
    emask[:, 0] = True
    b *= emask
    b /= b.sum(1, keepdim=True)
    g = torch.ones(h) / h
    data = torch.randint(num_cats, (32, length), generator=rng).cuda()
    pc = juice.compile(juice.structures.HMM(length, h, num_cats, block_size=16,
                         alpha=a.clone(), beta=b.clone(), gamma=g.clone())).to(torch.device('cuda:0'))
    sum_support = (pc.params > 0).clone()
    layer = pc.input_layer_group[0]
    # Replay the identical E-step statistics through both update procedures.
    # Advance the remasked model between checks; independent training trajectories
    # are a separate numerical experiment, not an identity of floating-point code.
    for update in range(1, 6):
        pc.init_param_flows(flows_memory=0.)
        pc(data, force_use_fp32=True)
        pc.backward(data, flows_memory=1., logspace_flows=True,
                    allow_modify_flows=False, force_use_fp32=True)
        original = [tensor.detach().clone() for tensor in
                    [pc.params, layer.params, pc.param_flows, layer.param_flows]]
        outputs = []
        for preserve in [True, False]:
            for tensor, saved in zip([pc.params, layer.params, pc.param_flows, layer.param_flows], original):
                tensor.data.copy_(saved)
            pc.mini_batch_em(step_size=1., pseudocount=pseudocount, keep_zero_params=preserve)
            if not preserve:
                pc.params.data[~sum_support] = 0.
                pc._normalize_parameters(pc.params.data, pseudocount=0.)
                emission = layer.params.data.reshape_as(b)
                emission[~emask.cuda()] = 0.
                emission /= emission.sum(1, keepdim=True)
            params = hmm_params(pc)
            for param in params:
                assert torch.isfinite(param).all()
                torch.testing.assert_close(param.sum(-1), torch.ones_like(param.sum(-1)), atol=5e-6, rtol=0)
            assert torch.equal(params[0] > 0, tmask)
            assert torch.equal(params[1] > 0, emask)
            outputs.append((params, exact_ll(*params, data)))
        left, right = outputs
        for param, other in zip(left[0], right[0]):
            torch.testing.assert_close(param, other, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(left[1], right[1], atol=1e-4, rtol=0)
        record(dict(kind='hmm_same_flows', update=update, num_cats=num_cats, pseudocount=pseudocount,
                    max_param_difference=max(float((p-q).abs().max()) for p,q in zip(left[0], right[0])),
                    max_sequence_ll_difference=float((left[1]-right[1]).abs().max()),
                    mean_ll_difference=float((left[1]-right[1]).mean())))
