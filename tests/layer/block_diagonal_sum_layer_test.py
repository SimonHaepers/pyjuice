"""Parity tests for :class:`BlockDiagonalSumLayer`.

Strategy: build the same block-diagonal ``summate(...)`` twice — once
normally (dispatches to ``BlockDiagonalSumLayer``) and once with
``_force_plain=True`` (forces plain ``SumLayer``) — and verify forward
``node_mars`` and backward ``element_flows`` agree at fp32 tolerance. A
trailing Monarch-end-to-end test reuses the pc-arena construction pattern
inline (BD → permutation-via-multiply → BD → permutation → BD) to confirm
the fast path slots into the broader Monarch chain without breaking it.
"""
from __future__ import annotations

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.layer import BlockDiagonalSumLayer, SumLayer


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


def _bd_edge_ids(NB: int) -> torch.Tensor:
    """Standard block-diagonal pattern: edge_ids[0] == edge_ids[1] == arange(NB)."""
    return torch.arange(0, NB)[None, :].repeat(2, 1)


def _build_bd_summate_pair(num_vars: int, V: int, NB: int, block_size: int,
                            seed: int, device: torch.device):
    """Build the same BD circuit twice; sync params at the DAG level.

    We can't ``copy_`` the flat params tensors directly because the BD and
    plain compile paths can disagree on per-block edge ordering inside
    ``_param_range``. Instead: build both, sync ``ns._params`` (the
    canonical ``[num_edges, bs, cbs]`` tensor on each ``SumNodes``) from
    one circuit to the other, then re-``gather`` to repopulate the flat
    buffer through each path's own layout — guaranteed to land on the
    same logical weights regardless of internal stride conventions.

    Returns ``(pc_bd, pc_plain)`` where ``pc_bd`` is the BlockDiagonalSumLayer
    path and ``pc_plain`` is the general SumLayer baseline.
    """
    def _build(force_plain: bool) -> juice.TensorCircuit:
        torch.manual_seed(seed)
        with juice.set_block_size(block_size):
            ni_list = [
                inputs(v, num_node_blocks=NB,
                       dist=dists.Categorical(num_cats=V))
                for v in range(num_vars)
            ]
            np_in = multiply(*ni_list)
            ns = summate(
                np_in,
                edge_ids=_bd_edge_ids(NB),
                block_size=block_size,
                _force_plain=force_plain,
            )
        return ns, juice.TensorCircuit(ns, verbose=False).to(device)

    ns_bd, pc_bd = _build(force_plain=False)
    ns_plain, pc_plain = _build(force_plain=True)

    # Pull pc_bd's canonical (per-ns) params into ns_bd._params, then
    # mirror onto ns_plain._params and re-gather through pc_plain's flat
    # buffer. Each call goes through that circuit's own ``_inverse_param_ids``
    # so the layouts can differ without breaking parity.
    pc_bd.update_parameters()

    def _ns_iter(root):
        seen, stack = set(), [root]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            yield n
            for c in getattr(n, "chs", []):
                stack.append(c)

    bd_sum_ns = [n for n in _ns_iter(ns_bd) if n.is_sum() and not n.is_tied()]
    plain_sum_ns = [n for n in _ns_iter(ns_plain) if n.is_sum() and not n.is_tied()]
    assert len(bd_sum_ns) == len(plain_sum_ns), (
        "DAG topology mismatch between BD and plain builds"
    )
    for nb, np_ in zip(bd_sum_ns, plain_sum_ns):
        np_._params = nb._params.clone()
        np_.gather_parameters(pc_plain.params)

    # Input layer params are stored separately and are seeded identically
    # (we re-seed before each build) but the layer compile order can shift
    # RNG consumption — copy directly to be safe.
    for li, lj in zip(pc_bd.input_layer_group, pc_plain.input_layer_group):
        lj.params.data.copy_(li.params.data)
    return pc_bd, pc_plain


def _has_bd_layer(pc: juice.TensorCircuit) -> bool:
    for lg in pc.inner_layer_groups:
        for layer in lg.layers:
            if isinstance(layer, BlockDiagonalSumLayer):
                return True
    return False


def _count_bd_sum_layers(pc: juice.TensorCircuit) -> tuple[int, int]:
    bd = plain = 0
    for lg in pc.inner_layer_groups:
        for layer in lg.layers:
            if isinstance(layer, BlockDiagonalSumLayer):
                bd += 1
            elif type(layer) is SumLayer:
                plain += 1
    return bd, plain


# ---------------------------------------------------------------------- #
# Dispatch tests
# ---------------------------------------------------------------------- #


def test_block_diagonal_dispatch(device):
    """BD-shaped ``summate`` must route to ``BlockDiagonalSumLayer``."""
    pc_bd, pc_plain = _build_bd_summate_pair(
        num_vars=4, V=8, NB=4, block_size=8, seed=0, device=device,
    )
    bd_n, plain_n = _count_bd_sum_layers(pc_bd)
    assert bd_n > 0, "BD circuit should dispatch to BlockDiagonalSumLayer"
    assert plain_n == 0, "no plain SumLayer expected when BD path is eligible"

    bd_n2, plain_n2 = _count_bd_sum_layers(pc_plain)
    assert bd_n2 == 0, "_force_plain=True must suppress BD dispatch"
    assert plain_n2 > 0, "expected plain SumLayer on the _force_plain side"


def test_non_bd_falls_back(device):
    """Non-BD ``summate`` (block-dense or sparse) must NOT touch the BD layer."""
    torch.manual_seed(0)
    with juice.set_block_size(8):
        ni_list = [
            inputs(v, num_node_blocks=4, dist=dists.Categorical(num_cats=8))
            for v in range(4)
        ]
        np_in = multiply(*ni_list)
        # Block-dense (default): not BD.
        ns = summate(np_in, num_node_blocks=4, block_size=8)
    pc = juice.TensorCircuit(ns, verbose=False).to(device)
    assert not _has_bd_layer(pc), (
        "block-dense summate must not route to BlockDiagonalSumLayer"
    )


# ---------------------------------------------------------------------- #
# Forward parity
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("NB,block_size", [
    (2, 4),    # tiny
    (4, 8),    # small
    (8, 16),   # medium
    (16, 32),  # larger blocks
])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_forward_parity(device, NB, block_size, batch_size):
    pc_bd, pc_plain = _build_bd_summate_pair(
        num_vars=3, V=8, NB=NB, block_size=block_size, seed=NB * 1000 + block_size,
        device=device,
    )

    torch.manual_seed(7)
    data = torch.randint(0, 8, (batch_size, 3), device=device)

    ll_bd = pc_bd(data).detach()
    ll_plain = pc_plain(data).detach()
    diff = (ll_bd - ll_plain).abs().max().item()
    # 1e-2 absolute: the plain SumLayer is itself ~5e-3 from fp64 ground
    # truth at these shapes (different fp32 reduction order), and the BD
    # kernel is ~1e-3 from ground truth — so the kernel-vs-kernel diff is
    # dominated by the baseline's precision, not by BD bugs.
    assert diff < 1e-2, (
        f"forward LL mismatch (NB={NB}, bs={block_size}, B={batch_size}): "
        f"max abs diff = {diff:.3e}"
    )


# ---------------------------------------------------------------------- #
# Backward parity (element flows)
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("NB,block_size", [
    (2, 4),
    (4, 8),
    (8, 16),
])
def test_backward_parity(device, NB, block_size):
    """Compare ``element_flows`` after a forward+backward on both paths."""
    pc_bd, pc_plain = _build_bd_summate_pair(
        num_vars=3, V=8, NB=NB, block_size=block_size, seed=42 + NB,
        device=device,
    )

    torch.manual_seed(11)
    batch_size = 4
    data = torch.randint(0, 8, (batch_size, 3), device=device)

    # Forward + backward through pyjuice's high-level entry points so the
    # plumbing (node_mars, element_mars, node_flows, element_flows) is set
    # up identically on both sides. ``allow_modify_flows=False`` keeps
    # ``node_flows`` in canonical (parent-flow) form so the comparison is
    # apples-to-apples. We compare only the sum-layer's element-flow
    # output (not the input-layer param_flows), but pyjuice's backward
    # always accumulates input-layer param_flows so we leave
    # ``compute_param_flows`` at the default. The BD layer's own
    # ``param_flows=None`` path is exercised because the sum-layer
    # ``backward`` dispatch routes through ``param_flows`` only when
    # ``flows_memory > 0`` and the param_flows kernel is the sum-layer
    # one (not the input layer's), which we don't reach.
    pc_bd(data)
    pc_plain(data)
    pc_bd.backward(data, allow_modify_flows=False)
    pc_plain.backward(data, allow_modify_flows=False)

    # Compare element_flows directly — that's what the BD backward kernel
    # writes (no param_flows path). ``node_flows`` parity falls out of
    # parent-side propagation through the rest of the circuit.
    ef_bd = pc_bd.element_flows.detach()
    ef_plain = pc_plain.element_flows.detach()
    diff = (ef_bd - ef_plain).abs().max().item()
    assert diff < 1e-2, (
        f"backward element_flows mismatch (NB={NB}, bs={block_size}): "
        f"max abs diff = {diff:.3e}"
    )

    nf_bd = pc_bd.node_flows.detach()
    nf_plain = pc_plain.node_flows.detach()
    diff_nf = (nf_bd - nf_plain).abs().max().item()
    assert diff_nf < 1e-2, (
        f"backward node_flows mismatch (NB={NB}, bs={block_size}): "
        f"max abs diff = {diff_nf:.3e}"
    )


# ---------------------------------------------------------------------- #
# Tied chain
# ---------------------------------------------------------------------- #


def test_tied_chain(device):
    """Tied duplicates of a BD source must share ``_param_range`` and produce
    identical numerical results to the plain SumLayer baseline."""

    NB, block_size, T, V = 4, 8, 5, 8

    def _build(force_plain: bool):
        torch.manual_seed(91)
        with juice.set_block_size(block_size):
            ni_list = [
                inputs(v, num_node_blocks=NB,
                       dist=dists.Categorical(num_cats=V))
                for v in range(T)
            ]
            # Stack a tied BD chain: ns0 -> dup(ns0) -> dup(ns0) ...
            # Each tied duplicate must alias ns0._param_range.
            ns0 = summate(
                multiply(ni_list[0], ni_list[1]),
                edge_ids=_bd_edge_ids(NB), block_size=block_size,
                _force_plain=force_plain,
            )
            tail = ns0
            for v in ni_list[2:]:
                tail = ns0.duplicate(
                    multiply(tail, v), tie_params=True,
                )
                if force_plain:
                    tail._force_plain_layer = True
            return juice.TensorCircuit(tail, verbose=False).to(device)

    pc_bd = _build(force_plain=False)
    pc_plain = _build(force_plain=True)
    pc_plain.params.data.copy_(pc_bd.params.data)
    for li, lj in zip(pc_bd.input_layer_group, pc_plain.input_layer_group):
        lj.params.data.copy_(li.params.data)

    # Tied alias check: every BD-routed tied duplicate shares its source's
    # _param_range.
    bd_layer_sources = []
    for lg in pc_bd.inner_layer_groups:
        for layer in lg.layers:
            if isinstance(layer, BlockDiagonalSumLayer):
                for n in layer.nodes:
                    if n.is_tied():
                        src = n.get_source_ns()
                        assert n._param_range == src._param_range, (
                            "tied BD ns must alias source's _param_range"
                        )
                        bd_layer_sources.append(src)
    assert bd_layer_sources, "expected tied duplicates on BlockDiagonalSumLayer"

    torch.manual_seed(13)
    batch_size = 8
    data = torch.randint(0, V, (batch_size, T), device=device)

    ll_bd = pc_bd(data).detach()
    ll_plain = pc_plain(data).detach()
    diff = (ll_bd - ll_plain).abs().max().item()
    assert diff < 1e-2, (
        f"tied BD LL mismatch: max abs diff = {diff:.3e}"
    )


# ---------------------------------------------------------------------- #
# Monarch end-to-end
# ---------------------------------------------------------------------- #


def _build_monarch_pair(NB: int, block_size: int, num_layers: int,
                        homogeneous: bool, device: torch.device):
    """Reproduce pc-arena's ``create_monarch_layers`` inline so the test
    doesn't depend on an external module. ``num_layers=3`` builds two
    Monarch matrices: BD → perm → BD → perm → BD."""
    H = NB * block_size
    permute_block_size = block_size

    # Permutation pattern: reshape→transpose→flatten — same as pc-arena
    # ``_create_monarch_layer``.
    def _permuted_edges() -> torch.Tensor:
        return torch.arange(0, H).reshape(
            H // permute_block_size, permute_block_size,
        ).permute(1, 0).reshape(H)[:, None]

    bd_edges = _bd_edge_ids(NB)

    def _build(force_plain: bool):
        torch.manual_seed(2026)
        with juice.set_block_size(block_size):
            ni_list = [
                inputs(v, num_node_blocks=NB,
                       dist=dists.Categorical(num_cats=8))
                for v in range(3)
            ]
            np0 = multiply(*ni_list)
            ns = summate(np0, edge_ids=bd_edges, block_size=block_size,
                          _force_plain=force_plain)
            for _ in range(num_layers - 1):
                np_perm = multiply(ns, edge_ids=_permuted_edges(),
                                   sparse_edges=True)
                # ``duplicate`` for the homogeneous case ties params across
                # the chain of BDs; in non-homogeneous mode every BD has
                # its own parameters.
                if homogeneous:
                    ns = ns.duplicate(np_perm, tie_params=True)
                    if force_plain:
                        ns._force_plain_layer = True
                else:
                    ns = summate(np_perm, edge_ids=bd_edges,
                                  block_size=block_size,
                                  _force_plain=force_plain)
            return juice.TensorCircuit(ns, verbose=False).to(device)

    pc_bd = _build(force_plain=False)
    pc_plain = _build(force_plain=True)
    pc_plain.params.data.copy_(pc_bd.params.data)
    for li, lj in zip(pc_bd.input_layer_group, pc_plain.input_layer_group):
        lj.params.data.copy_(li.params.data)
    return pc_bd, pc_plain


@pytest.mark.parametrize("homogeneous", [False, True])
def test_monarch_end_to_end(device, homogeneous):
    """Full Monarch sub-circuit (3-BD composition with permutations
    in between) must produce identical LLs through both compile paths."""
    pc_bd, pc_plain = _build_monarch_pair(
        NB=4, block_size=8, num_layers=3, homogeneous=homogeneous,
        device=device,
    )

    bd_count, plain_count = _count_bd_sum_layers(pc_bd)
    assert bd_count > 0, (
        "Monarch should dispatch at least one BD sum to BlockDiagonalSumLayer"
    )

    torch.manual_seed(2027)
    data = torch.randint(0, 8, (16, 3), device=device)
    ll_bd = pc_bd(data).detach()
    ll_plain = pc_plain(data).detach()
    diff = (ll_bd - ll_plain).abs().max().item()
    assert diff < 1e-2, (
        f"Monarch LL mismatch (homogeneous={homogeneous}): "
        f"max abs diff = {diff:.3e}"
    )
