"""CUDA-graph keys for keyword options: tensors are replayed from graph-owned
copies, so a fresh or updated mask reuses one graph instead of recapturing."""
import subprocess
import sys

import torch

import pyjuice as juice
from pyjuice.model.tensorcircuit import TensorCircuit
from pyjuice.structures import MonarchHMM

DEV = torch.device("cuda:0")


def _dense(T=6, H=64, V=11):
    torch.manual_seed(5)
    root = MonarchHMM(T, H, V, block_size=8)
    root.init_parameters()
    return juice.TensorCircuit(root, verbose=False).to(DEV)


def _n_graphs(pc, kind):
    return len([k for k in pc._recorded_cuda_graphs if k[0] == kind])


def test_fresh_and_updated_masks_reuse_one_graph():
    pc = _dense()
    B, T, V = 5, 6, 11
    x = torch.randint(0, V, (B, T), device=DEV)
    masks = [torch.rand(B, T, device=DEV) < 0.3 for _ in range(3)]

    def eager(m):
        y = pc(x, missing_mask=m, apply_cudagraph=False).detach().clone()
        pc.backward(x, missing_mask=m, compute_param_flows=False, allow_modify_flows=False,
                    apply_cudagraph=False)
        return y, pc.node_flows.detach().clone()

    refs = [eager(m) for m in masks]

    def graph(m, record=False):
        y = pc(x, missing_mask=m, record_cudagraph=record).detach().clone()
        pc.backward(x, missing_mask=m, compute_param_flows=False, allow_modify_flows=False,
                    record_cudagraph=record)
        return y, pc.node_flows.detach().clone()

    def check(got, ref):
        torch.testing.assert_close(got[0], ref[0], rtol=0.0, atol=1e-4)
        torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-6)

    live = masks[0].clone()
    check(graph(live, record=True), refs[0])
    assert (_n_graphs(pc, 0), _n_graphs(pc, 1)) == (1, 1)
    for m, ref in zip(masks + masks[::-1], refs + refs[::-1]):
        check(graph(m.clone()), ref)          # new tensor each call
    live.copy_(masks[2])                      # same tensor, new contents
    check(graph(live), refs[2])
    assert (_n_graphs(pc, 0), _n_graphs(pc, 1)) == (1, 1)

    ref = eager(None)                         # no mask is a different capture
    check(graph(None, record=True), ref)
    assert (_n_graphs(pc, 0), _n_graphs(pc, 1)) == (2, 2)


def test_options_key():
    key = TensorCircuit._graph_options_key
    a = torch.tensor([0.5, 1.0])
    # CPU tensors are read on the host: their contents are baked in.
    assert key(a) == key(a.clone())
    assert key(a) != key(torch.tensor([0.5, 2.0]))
    assert key({"m": torch.tensor([True, False])}, static_tensors=True) != \
        key({"m": torch.tensor([False, False])}, static_tensors=True)
    # Top-level CUDA tensors become graph-owned copies: keyed on shape / dtype.
    m1, m2 = torch.zeros(3, 4, device=DEV), torch.ones(3, 4, device=DEV)
    assert key({"m": m1}, static_tensors=True) == key({"m": m2}, static_tensors=True)
    assert key({"m": m1}, static_tensors=True) != key({"m": m1[:2]}, static_tensors=True)
    assert key({"m": m1}, static_tensors=True) != key({"m": m1.bool()}, static_tensors=True)
    # Nested CUDA tensors are not copied, so their address stays in the key.
    assert key({"m": [m1]}, static_tensors=True) != key({"m": [m2]}, static_tensors=True)


def test_cuda_host_read_aborts_capture():
    # Premise of the shape-only key: captured code cannot bake in the value
    # of a CUDA tensor through a host read. Run in a child process so a
    # failed capture cannot affect the other tests.
    code = (
        "import torch\n"
        "x = torch.ones(1, device='cuda:0')\n"
        "g = torch.cuda.CUDAGraph()\n"
        "try:\n"
        "    with torch.cuda.graph(g):\n"
        "        float(x)\n"
        "except RuntimeError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], timeout=300).returncode == 0
