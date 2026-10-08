"""Weight-stationary (WS) batched path of :class:`SparseIOSumLayer`.

At B>1 the transition step can run on the WS kernels (W tile loaded once per
step and applied to the whole batch, deferred param flows flushed by one GEMM
at the end of the backward) instead of the per-sample gather kernels. Both
must agree on the forward LLs, on the flows that reach the emission layer and
on the transition param flows — for both WS precisions, for mixed per-step
dispatch (forward and backward of one step may take different paths), and
across consecutive passes that reuse the deferred-flow stash.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.layer.sparse_io_sum_layer as sio
from pyjuice.structures import SparseHMM, random_emission_csc


# (atol on per-sequence LL, rtol on flows) per WS precision. tf32 runs on
# operands rounded to nearest, so its error is zero-mean ~1e-4 per step.
_TOL = {"ieee": (1e-4, 1e-4), "tf32": (5e-3, 3e-3)}


def _build(H, V, T, bs, density, seed=0):
    indptr, indices, _, _ = random_emission_csc(H, V, density, seed=seed)
    torch.manual_seed(seed)
    ns = SparseHMM(T, H, V, indptr, indices, block_size=bs)
    ns.init_parameters()
    return juice.TensorCircuit(ns, verbose=False).to(torch.device("cuda:0"))


def _pass(pc, data, flows_memory=0.0, negate=False):
    ll = pc(data).detach().clone()
    pc.backward(data, compute_param_flows=True, flows_memory=flows_memory,
                negate_pflows=negate)
    emit = torch.cat([layer.param_flows.detach().reshape(-1).clone()
                      for layer in pc.input_layer_group])
    return ll, pc.param_flows.detach().clone(), emit


def _set(monkeypatch, mode, prec="ieee"):
    monkeypatch.setattr(sio, "_WS_MODE", mode)
    monkeypatch.setattr(sio, "_WS_PRECISION", prec)


def _assert_flows_close(got, ref, rtol, what):
    scale = ref.abs().max().clamp_min(1e-12)
    err = ((got - ref).abs().max() / scale).item()
    assert err < rtol, f"{what}: max |diff| / max |ref| = {err:.2e} (tol {rtol:.0e})"


def _count_ws_calls(monkeypatch):
    calls = {"fwd": 0, "bwd": 0}
    fwd, bwd = sio.SparseIOSumLayer._ws_forward_block, sio.SparseIOSumLayer._ws_backward_block

    def counted_fwd(self, *a, **k):
        calls["fwd"] += 1
        return fwd(self, *a, **k)

    def counted_bwd(self, *a, **k):
        calls["bwd"] += 1
        return bwd(self, *a, **k)

    monkeypatch.setattr(sio.SparseIOSumLayer, "_ws_forward_block", counted_fwd)
    monkeypatch.setattr(sio.SparseIOSumLayer, "_ws_backward_block", counted_bwd)
    return calls


@pytest.mark.parametrize("prec", ["ieee", "tf32"])
@pytest.mark.parametrize("H,V,T,bs,density,B", [
    (64, 20, 6, 16, 0.3, 3),       # B < batch tile
    (96, 30, 5, 32, 0.25, 9),      # 3 PC blocks (non power-of-2 row count)
    (128, 40, 8, 32, 0.2, 16),
    (128, 40, 7, 16, 0.2, 70),     # B > 64: two batch slices per row block
    (4096, 50, 4, 1024, 0.05, 130),  # 128-wide batch tiles (8 warps)
])
def test_ws_matches_gather(monkeypatch, prec, H, V, T, bs, density, B):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pc = _build(H, V, T, bs, density)
    data = torch.randint(0, V, (B, T), device="cuda:0")

    _set(monkeypatch, "never")
    ll_ref, pf_ref, emit_ref = _pass(pc, data)

    calls = _count_ws_calls(monkeypatch)
    _set(monkeypatch, "always", prec)
    ll, pf, emit = _pass(pc, data)
    assert calls["fwd"] == T - 1 and calls["bwd"] == T - 1

    atol, rtol = _TOL[prec]
    torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=atol)
    _assert_flows_close(pf, pf_ref, rtol, "transition param flows")
    _assert_flows_close(emit, emit_ref, rtol, "emission param flows")
    # Total flow mass is conserved exactly up to fp32 rounding.
    assert abs(pf.sum().item() - pf_ref.sum().item()) <= 1e-3 * pf_ref.sum().item()


def test_ws_mixed_dispatch_and_stash_reuse(monkeypatch):
    """Alternate WS / gather per call so a step's forward and backward take
    different paths, then run a second pass where other slots use the gather
    kernels: stale stash columns of the first pass must not leak into the
    second pass's deferred flows."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    H, V, T, bs = 128, 40, 9, 32
    pc = _build(H, V, T, bs, 0.2, seed=1)
    d1 = torch.randint(0, V, (12, T), device="cuda:0")
    d2 = torch.randint(0, V, (12, T), device="cuda:0")

    _set(monkeypatch, "never")
    ref1 = _pass(pc, d1)
    ref2 = _pass(pc, d2)

    _set(monkeypatch, "auto")
    counter = {"n": 0}

    def alternating(self, blk_idx, block, sv_in, sv_out, params, need_stash=False):
        counter["n"] += 1
        return self._ws_slots[blk_idx] is not None and counter["n"] % 2 == 0

    monkeypatch.setattr(sio.SparseIOSumLayer, "_ws_use", alternating)
    got1 = _pass(pc, d1)
    counter["n"] = 1                     # flip the parity for the second pass
    got2 = _pass(pc, d2)

    for got, ref in ((got1, ref1), (got2, ref2)):
        torch.testing.assert_close(got[0], ref[0], rtol=0.0, atol=1e-4)
        _assert_flows_close(got[1], ref[1], 1e-4, "transition param flows")
        _assert_flows_close(got[2], ref[2], 1e-4, "emission param flows")


def test_ws_flow_accumulation_and_negate(monkeypatch):
    """``flows_memory=1`` accumulates across backward calls and
    ``negate_pflows`` flips the deferred flows, as on the gather path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    H, V, T, bs = 64, 24, 6, 16
    pc = _build(H, V, T, bs, 0.3, seed=2)
    d1 = torch.randint(0, V, (8, T), device="cuda:0")
    d2 = torch.randint(0, V, (8, T), device="cuda:0")

    results = {}
    for mode in ("never", "always"):
        _set(monkeypatch, mode)
        _pass(pc, d1)
        _, pf_acc, _ = _pass(pc, d2, flows_memory=1.0)
        _, pf_neg, _ = _pass(pc, d1, negate=True)
        results[mode] = (pf_acc, pf_neg)

    _assert_flows_close(results["always"][0], results["never"][0], 1e-4, "accumulated flows")
    _assert_flows_close(results["always"][1], results["never"][1], 1e-4, "negated flows")
    assert results["always"][1].sum().item() < 0


def test_ws_em_step_matches_gather(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    H, V, T, bs = 128, 40, 8, 32
    data = torch.randint(0, V, (20, T), device="cuda:0")
    params = {}
    for mode in ("never", "always"):
        _set(monkeypatch, mode)
        pc = _build(H, V, T, bs, 0.2, seed=3)
        _pass(pc, data)
        pc.mini_batch_em(step_size=0.5, pseudocount=1e-3, use_cudagraph=False)
        params[mode] = pc.params.detach().clone()
        params[mode + "_ll"] = pc(data).detach().clone()
    assert (params["always"] - params["never"]).abs().max().item() < 1e-5
    torch.testing.assert_close(params["always_ll"], params["never_ll"], rtol=0.0, atol=1e-4)


def test_ws_auto_batch_threshold(monkeypatch):
    """``auto`` (the default) takes WS from ``_WS_MIN_BATCH`` samples up and
    leaves smaller batches — B=1 inference above all — on the gather kernels.
    The emission density here keeps the circuit well clear of the
    ``_WS_COST_RATIO`` guard, so the batch size is what decides."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    H, V, T = 4096, 50, 4
    pc = _build(H, V, T, 1024, 0.2, seed=7)
    assert all(g.eligible for g in pc._sparse_io_ws_groups)

    for B, expect_ws in ((1, False), (2, False), (15, False), (16, True), (24, True)):
        calls = _count_ws_calls(monkeypatch)
        _set(monkeypatch, "auto")
        monkeypatch.setattr(sio, "_WS_MIN_BATCH", 16)
        _pass(pc, torch.randint(0, V, (B, T), device="cuda:0"))
        used = calls["fwd"] > 0
        assert used is expect_ws, f"B={B}: WS used={used}, expected {expect_ws}"
        if expect_ws:
            assert calls["fwd"] == T - 1 and calls["bwd"] == T - 1

    # The threshold is the knob: raising it pushes B=16 back to gather.
    calls = _count_ws_calls(monkeypatch)
    _set(monkeypatch, "auto")
    monkeypatch.setattr(sio, "_WS_MIN_BATCH", 32)
    _pass(pc, torch.randint(0, V, (16, T), device="cuda:0"))
    assert calls == {"fwd": 0, "bwd": 0}


def test_ws_auto_skips_small_circuits(monkeypatch):
    """A circuit too small to amortise a pass over W stays on the gather
    kernels however large the batch (``_WS_COST_RATIO``)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pc = _build(1024, 400, 4, 1024, 0.01, seed=8)
    calls = _count_ws_calls(monkeypatch)
    _set(monkeypatch, "auto")
    _pass(pc, torch.randint(0, 400, (64, 4), device="cuda:0"))
    assert calls == {"fwd": 0, "bwd": 0}


@pytest.mark.parametrize("starved", ["w_copy", "stash"])
def test_ws_falls_back_when_buffers_dont_fit(monkeypatch, starved):
    """WS is on by default, so a circuit whose extra buffers do not fit must
    drop to the gather kernels with a warning — never fail the pass. The W
    copy is only allocated in tf32 mode, the deferred stash only once a
    backward actually asks for param flows, so starve each in turn."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    H, V, T = 4096, 50, 4
    pc = _build(H, V, T, 1024, 0.2, seed=9)
    data = torch.randint(0, V, (16, T), device="cuda:0")

    _set(monkeypatch, "never")
    ll_ref, pf_ref, emit_ref = _pass(pc, data)

    if starved == "w_copy":
        real_scratch = sio._ws_scratch

        def oom_on_w(name, shape, device):
            if name == "w_rn":
                raise torch.cuda.OutOfMemoryError("simulated")
            return real_scratch(name, shape, device)

        monkeypatch.setattr(sio, "_ws_scratch", oom_on_w)
        prec, expect_fwd = "tf32", 0
    else:
        def oom_on_stash(self, B, device):
            raise torch.cuda.OutOfMemoryError("simulated")

        monkeypatch.setattr(sio._WSGroup, "_ensure_stash", oom_on_stash)
        # The forward needs no stash, so it still runs on WS; only the
        # param-flow backward hits the starved allocation.
        prec, expect_fwd = "ieee", T - 1

    calls = _count_ws_calls(monkeypatch)
    _set(monkeypatch, "auto", prec)
    with pytest.warns(RuntimeWarning, match="weight-stationary"):
        ll, pf, emit = _pass(pc, data)

    assert calls["fwd"] == expect_fwd and calls["bwd"] == 0
    assert all(not g.eligible for g in pc._sparse_io_ws_groups)
    torch.testing.assert_close(ll, ll_ref, rtol=0.0, atol=1e-4)
    _assert_flows_close(pf, pf_ref, 1e-4, "transition param flows")
    _assert_flows_close(emit, emit_ref, 1e-4, "emission param flows")


def test_ws_skips_ineligible_blocks(monkeypatch):
    """Block sizes below the 16-wide tl.dot minimum stay on the gather path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pc = _build(32, 12, 5, 8, 0.4)
    assert all(not g.eligible for g in pc._sparse_io_ws_groups)
    calls = _count_ws_calls(monkeypatch)
    _set(monkeypatch, "always")
    _pass(pc, torch.randint(0, 12, (6, 5), device="cuda:0"))
    assert calls == {"fwd": 0, "bwd": 0}
