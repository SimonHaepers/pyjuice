"""Model-equivalent reordering of the latent states of a circuit.

A circuit's hidden states are interchangeable labels: relabelling the ``H``
nodes of every block that shares a state space (the sum blocks, the sparse
emission inputs, the element-wise products between them) by the same
permutation leaves every marginal and likelihood unchanged. The order does
matter for the kernels: the sparse-IO forward gathers, for every active child
of a step, the weights of the step's active parents, and a 32-byte sector
delivers eight consecutive fp32 weights. On the trained wikitext chains the
active parents of a token are spread at random over the state index (mean run
of consecutive active parents 1.1-1.3, 18-24 % of every fetched sector used);
:func:`emission_state_order` sorts the states by the most frequent token they
emit, so states that share frequent tokens become neighbours and every
frequent token's parents form runs (43-54 % of the fetched bytes used on the
same chains, 2.3x less traffic). :func:`permute_latents` applies any
permutation in place, before compilation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from pyjuice.nodes import CircuitNodes
from pyjuice.nodes.distributions import SparseCategorical


def source_nodes(root_ns: CircuitNodes) -> list:
    """The nodes that own parameters (tied duplicates resolve to their
    source), in traversal order, each once."""
    out, seen = [], set()
    for ns in root_ns:
        src = ns.get_source_ns()
        if id(src) not in seen:
            seen.add(id(src))
            out.append(src)
    return out


def emission_state_order(ns_input: CircuitNodes, token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Permutation ``perm`` (new position -> old state) of the ``H`` states of a
    :class:`SparseCategorical` input: states sorted by the weight of the most
    frequent token they emit, then of the second, so that states emitting the
    same frequent tokens are adjacent. ``token_weights`` (``[num_cats]``, e.g.
    corpus counts) weighs the tokens; by default each token's emission column
    mass under a uniform latent is used (model-intrinsic, no data needed).
    States that emit nothing go last."""
    src = ns_input.get_source_ns()
    dist = src.dist
    assert isinstance(dist, SparseCategorical), "emission_state_order needs a SparseCategorical input"
    H, V = int(dist._num_nodes), int(dist.num_cats)
    indptr = dist._csc_indptr.cpu()
    indices = dist._csc_indices.cpu()
    nnz = int(indices.numel())
    col_of = torch.repeat_interleave(torch.arange(V), torch.diff(indptr))
    if token_weights is None:
        params = src.get_params()
        assert params is not None, "the emission input has no parameters; pass token_weights"
        w = torch.zeros(V, dtype=torch.float64).scatter_add_(0, col_of, params[:nnz].detach().cpu().double())
    else:
        w = torch.as_tensor(token_weights, dtype=torch.float64).cpu()
        assert w.numel() == V, f"token_weights must have {V} entries"
    w_nz = w[col_of]
    # nonzeros ordered by (state asc, token weight desc)
    o = torch.argsort(w_nz, descending=True, stable=True)
    o = o[torch.argsort(indices[o], stable=True)]
    rows, wv = indices[o], w_nz[o]
    states = torch.arange(H)
    starts = torch.searchsorted(rows, states)
    ends = torch.searchsorted(rows, states, right=True)
    n = ends - starts
    top1 = torch.full((H,), -float("inf"), dtype=torch.float64)
    top2 = torch.zeros(H, dtype=torch.float64)
    has = n > 0
    top1[has] = wv[starts[has]]
    has2 = n > 1
    top2[has2] = wv[starts[has2] + 1]
    perm = torch.argsort(top2, descending=True, stable=True)
    perm = perm[torch.argsort(top1[perm], descending=True, stable=True)]
    return perm


def _is_identity_edges(ns) -> bool:
    e = ns.edge_ids
    return bool(torch.equal(e, torch.arange(e.size(0)).unsqueeze(1).expand_as(e)))


def permute_latents(root_ns: CircuitNodes, perm: torch.Tensor,
                    nodes: Optional[Sequence[CircuitNodes]] = None) -> torch.Tensor:
    """Relabel the latent states in place: new index ``i`` holds old state
    ``perm[i]``. Applied to every source node block of exactly ``H =
    len(perm)`` nodes (or to ``nodes``, resolved to their sources, when given):

    * sum nodes: rows of the parameter matrix when the block has ``H`` nodes,
      and the columns that belong to a child block of ``H`` nodes (the root
      of a chain, one node with ``H`` children, is such a case);
    * input nodes with a :class:`SparseCategorical`: the emission rows, i.e.
      the CSC row ids and values (re-sorted within every column); other
      distributions: their per-node parameter blocks;
    * product nodes: nothing to move, their edges must be element-wise.

    Call before :func:`pyjuice.compile`. Returns ``perm``."""
    perm = torch.as_tensor(perm, dtype=torch.long).cpu()
    H = int(perm.numel())
    assert torch.equal(torch.sort(perm).values, torch.arange(H)), "perm must be a permutation of range(H)"
    inv = torch.empty(H, dtype=torch.long)
    inv[perm] = torch.arange(H)
    if nodes is None:
        targets = [ns for ns in source_nodes(root_ns) if ns.num_nodes == H]
    else:
        targets = []
        for ns in nodes:
            src = ns.get_source_ns()
            assert src.num_nodes == H, "every node block to permute must have len(perm) nodes"
            if all(src is not t for t in targets):
                targets.append(src)
    target_ids = {id(t) for t in targets}
    done_dists = set()

    def col_perm(ns):
        parts, off = [], 0
        for ch in ns.chs:
            n = ch.num_nodes
            parts.append(off + (perm if id(ch.get_source_ns()) in target_ids or id(ch) in target_ids else torch.arange(n)))
            off += n
        return torch.cat(parts)

    for ns in source_nodes(root_ns):
        if ns.is_sum():
            rows = id(ns) in target_ids
            cp = col_perm(ns)
            cols = not torch.equal(cp, torch.arange(cp.numel()))
            if not rows and not cols:
                continue
            assert not ns.provided("_zero_param_mask") or ns._zero_param_mask is None, \
                "permute_latents does not handle zero_param_mask"
            W = ns.get_params(as_matrix=True)
            assert W is not None, "sum node without parameters"
            if rows:
                W = W[perm]
            if cols:
                W = W[:, cp]
            nb, ncb = ns.num_node_blocks, ns.num_ch_node_blocks
            BS, CBS = ns.block_size, ns.ch_block_size
            p4 = W.reshape(nb, BS, ncb, CBS).permute(0, 2, 1, 3).contiguous()
            ns.set_params(p4, normalize=False)
            if ns.provided("_param_flows"):
                ns._param_flows = None
        elif ns.is_prod():
            if id(ns) in target_ids or any(id(ch.get_source_ns()) in target_ids for ch in ns.chs):
                assert _is_identity_edges(ns), "permute_latents: product edges must be element-wise"
        elif ns.is_input() and id(ns) in target_ids:
            dist = ns.dist
            if isinstance(dist, SparseCategorical):
                if id(dist) in done_dists:
                    continue
                done_dists.add(id(dist))
                dev = dist._csc_indices.device
                indptr = dist._csc_indptr.cpu()
                indices = dist._csc_indices.cpu()
                nnz = int(indices.numel())
                V = int(dist.num_cats)
                col_of = torch.repeat_interleave(torch.arange(V), torch.diff(indptr))
                new_rows = inv[indices]
                order = torch.argsort(col_of * H + new_rows)
                for attr in ("_csc_indptr_cpu", "_csc_indptr_list", "_csc_indices_cpu"):
                    if hasattr(dist, attr):
                        setattr(dist, attr, None)
                dist.set_meta_parameters(H, indptr, new_rows[order])
                if dev.type != "cpu":
                    dist.move_to_device(dev)
                if ns.provided("_params"):
                    p = ns._params
                    ns._params = torch.cat([p[:nnz][order.to(p.device)], p[nnz:]])
                dist._state_perm = perm.clone()
            else:
                assert not dist.need_meta_parameters, f"permute_latents: unsupported distribution {type(dist).__name__}"
                if ns.provided("_params"):
                    p = ns._params
                    ns._params = p.reshape(H, -1)[perm.to(p.device)].reshape(-1).contiguous()
    return perm


def reorder_states(root_ns: CircuitNodes, token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """:func:`emission_state_order` of the circuit's sparse emission input
    applied with :func:`permute_latents`. Every :class:`SparseCategorical`
    input must have the same number of nodes; returns the permutation."""
    emits = [ns for ns in source_nodes(root_ns) if ns.is_input() and isinstance(ns.dist, SparseCategorical)]
    assert emits, "reorder_states: no SparseCategorical input in the circuit"
    H = emits[0].num_nodes
    assert all(e.num_nodes == H for e in emits), "reorder_states: sparse inputs of different sizes"
    perm = emission_state_order(emits[0], token_weights)
    return permute_latents(root_ns, perm)
