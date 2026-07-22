from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import List, Optional, Sequence

if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math

from pyjuice.nodes import SumNodes
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.parameter_list import FastParamList
from .layer import Layer
from .sum_layer import SumLayer


class BlockDiagonalSumLayer(SumLayer):
    """
    Inference + element-flow backward fast path for sum layers whose
    transition matrix is block-diagonal at the block level: parent block
    ``i`` is connected to exactly child block ``i`` (and only that one).

    This is the structure produced by a Monarch factorisation
    (``BD₁ → permutation → BD₂``) when each BD is built with
    ``edge_ids = arange(NB)[None, :].repeat(2, 1)``. The natural data-model
    shape is ``NB`` independent ``[bs, cbs]`` weight matrices concatenated;
    a single Triton launch handles all ``NB`` blocks in parallel via the
    grid's parent-block axis.

    Asymptotically the kernel is ``NB``-fold cheaper than the general
    sum kernel: only the diagonal ``(pblock, pblock)`` weight blocks
    exist, and the per-parent-block child marginals live in a disjoint
    contiguous slice of ``element_mars``, so each program handles its
    block independently with no cross-block reduction.

    Constraints (asserted at compile time):
      * Every ``SumNodes`` has a single child group.
      * ``NB == NB_ch`` and ``bs == cbs`` (square blocks — the natural
        Monarch case; relax later if a non-square use case appears).
      * ``edge_ids`` is exactly ``arange(NB)[None, :].repeat(2, 1)``.
      * ``param_flows`` writes are rejected (inference-only in phase 1 —
        see plan ``look-i-m-trying-to-eventual-waterfall.md``).

    Tied ``SumNodes`` are supported: ``_param_range`` / ``_param_flow_range``
    alias the source's, same protocol as :class:`DenseSumLayer`.
    """

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int,
                 global_pid_start: int, global_pfid_start: int,
                 node2tiednodes: dict,
                 layer_sparsity_tol: Optional[float] = None,
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_block: int = 8,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False) -> None:

        # Bypass ``SumLayer.__init__`` — we replace its block-sparse
        # bookkeeping with the direct-layout metadata below (same pattern
        # as ``DenseSumLayer``).
        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."
        assert len(nodes) == len(set(nodes)), "Input node list contains duplicates."

        for ns in nodes:
            assert len(ns.chs) == 1, (
                "BlockDiagonalSumLayer requires num_chs == 1 per SumNodes."
            )
            assert ns.num_node_blocks == ns.num_ch_node_blocks, (
                "BlockDiagonalSumLayer requires NB == NB_ch (square block "
                f"diagonal); got NB={ns.num_node_blocks}, "
                f"NB_ch={ns.num_ch_node_blocks}."
            )
            assert ns.block_size == ns.ch_block_size, (
                "BlockDiagonalSumLayer requires bs == cbs (square per-block "
                f"weight matrix); got bs={ns.block_size}, "
                f"cbs={ns.ch_block_size}."
            )
            assert ns.edge_ids.size(1) == ns.num_node_blocks, (
                "BlockDiagonalSumLayer: edge_ids has wrong number of "
                f"columns ({ns.edge_ids.size(1)} != NB={ns.num_node_blocks})."
            )
            if ns.is_tied():
                source_ns = ns.get_source_ns()
                assert source_ns.provided("_param_range"), (
                    "BlockDiagonalSumLayer: tied sum node encountered "
                    "before its source was compiled."
                )

        layer_nid_start = global_nid_start
        layer_pid_start = global_pid_start
        layer_pfid_start = global_pfid_start

        layer_num_nodes = 0
        layer_num_edges = 0

        # Per-SumNodes metadata: ``(nid_start, cid_start, pid_start,
        # pfid_start, NB, bs, cbs)`` — one CUDA kernel launch per entry per
        # direction.
        blocks: List[tuple] = []

        curr_nid = layer_nid_start
        curr_pid = layer_pid_start
        curr_pfid = layer_pfid_start

        for ns in nodes:
            cs = ns.chs[0]
            assert cs.provided("_output_ind_range"), (
                "BlockDiagonalSumLayer: child has no _output_ind_range; "
                "make sure the product layer is compiled before the sum."
            )

            NB = ns.num_node_blocks
            bs = ns.block_size
            cbs = ns.ch_block_size

            ns._output_ind_range = (curr_nid, curr_nid + ns.num_nodes)

            # Param range — alias-on-tied (same protocol as
            # DenseSumLayer). Block-diagonal needs exactly ``NB * bs * cbs``
            # scalars per ns (NB_ch-fold smaller than the dense case).
            if ns.is_tied():
                source_ns = ns.get_source_ns()
                block_pid_start, _ = source_ns._param_range
                block_pfid_start, _ = source_ns._param_flow_range
                ns._param_range = source_ns._param_range
                ns._param_flow_range = source_ns._param_flow_range
            else:
                pid_end = curr_pid + ns.num_edges
                pfid_end = curr_pfid + ns.num_edges
                ns._param_range = (curr_pid, pid_end)
                ns._param_flow_range = (curr_pfid, pfid_end)
                block_pid_start = curr_pid
                block_pfid_start = curr_pfid
                curr_pid = pid_end
                curr_pfid = pfid_end

            # Validate the block-diagonal pattern AFTER computing num_edges
            # so a stray non-BD ns doesn't silently consume the wrong slice
            # of the param buffer. We need edge_ids[0] == edge_ids[1] ==
            # arange(NB); ``_init_layers`` already gates this layer behind
            # ``_block_diagonal_eligible``, but the assert here keeps the
            # invariant local to the layer for anyone constructing it
            # directly (e.g. tests).
            edge_ids = ns.edge_ids
            expected = torch.arange(NB, dtype = edge_ids.dtype,
                                    device = edge_ids.device)
            assert torch.equal(edge_ids[0], expected) and \
                   torch.equal(edge_ids[1], expected), (
                "BlockDiagonalSumLayer: edge_ids is not the block-diagonal "
                "pattern (expected arange(NB)[None,:].repeat(2,1))."
            )

            # ``_params`` is shape ``[NB, bs, cbs]`` ordered by edge_ids
            # columns. The BD edge ordering is already the natural
            # ``arange(NB)``, so ``_inverse_param_ids`` is identity and
            # ``gather_parameters`` lays out the flat buffer as
            # ``[NB, cbs, bs]`` row-major. The kernel addresses an entry
            # ``(pblock, c, s)`` at
            # ``pid_start + pblock*bs*cbs + c*bs + s`` — see the docstrings
            # on ``_fw_bd_kernel`` / ``_bk_bd_kernel`` for the full layout.
            edge_lin_ids = torch.arange(NB, dtype = torch.long)
            ns._param_ids = block_pid_start + edge_lin_ids * bs * cbs
            ns._inverse_param_ids = torch.argsort(edge_lin_ids)
            # ``_param_flow_ids`` mirrors ``_param_ids`` in the BD layout —
            # the param-flow buffer reuses the params' ``[NB, bs, cbs]`` flat
            # stride exactly (one ``bs*cbs`` block per edge, in arange order).
            # Needed by :meth:`SumNodes.update_param_flows` and
            # :meth:`TensorCircuit.get_node_param_flows` to pull this ns's
            # accumulated pflows back into canonical ``[NB, bs, cbs]`` shape.
            ns._param_flow_ids = block_pfid_start + edge_lin_ids * bs * cbs

            blocks.append((
                curr_nid,                      # nid_start in node_mars
                cs._output_ind_range[0],       # cid_start in element_mars
                block_pid_start,               # pid_start in flat params
                block_pfid_start,              # pfid_start (unused phase 1)
                NB, bs, cbs,
            ))

            curr_nid += ns.num_nodes
            layer_num_nodes += ns.num_nodes
            if not ns.is_tied():
                layer_num_edges += ns.num_edges

        self.num_nodes = layer_num_nodes
        self.num_edges = layer_num_edges
        self._layer_nid_range = (layer_nid_start, layer_nid_start + layer_num_nodes)
        self._layer_pid_range = (layer_pid_start, curr_pid)
        self._layer_pfid_range = (layer_pfid_start, curr_pfid)

        self._bd_blocks = blocks

        # Stub out the block-sparse bookkeeping that some parent
        # introspection paths read — mirrors the same stubs in
        # ``DenseSumLayer``.
        self.num_fw_partitions = 0
        self.num_bk_partitions = 0
        self.partitioned_nids = FastParamList([])
        self.partitioned_cids = FastParamList([])
        self.partitioned_pids = FastParamList([])
        self.partitioned_pfids = FastParamList([])
        self.partitioned_chids = FastParamList([])
        self.partitioned_parids = FastParamList([])
        self.partitioned_parpids = FastParamList([])
        self.cs_block_sizes = []
        self._cached_fw_pcids = dict()
        self._cached_bk_parids = dict()

    def __repr__(self) -> str:
        return (
            f"BlockDiagonalSumLayer(nid_range=({self._layer_nid_range[0]}, "
            f"{self._layer_nid_range[1]}), num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, num_blocks={len(self._bd_blocks)})"
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                params: torch.Tensor, force_use_bf16: bool = False,
                force_use_fp32: bool = False, propagation_alg: str = "LL",
                **kwargs) -> None:

        assert params.dim() == 1, (
            "BlockDiagonalSumLayer only supports flat 1-D params; got "
            f"params.dim()={params.dim()}."
        )
        assert propagation_alg == "LL", (
            "BlockDiagonalSumLayer currently supports only LL propagation."
        )

        batch_size = node_mars.size(1)

        for block in self._bd_blocks:
            nid_start, cid_start, pid_start, _pfid_start, NB, bs, cbs = block

            CBS_PADDED = triton.next_power_of_2(cbs)
            BS_PADDED = triton.next_power_of_2(bs)
            BLOCK_B = min(triton.next_power_of_2(batch_size), 64)
            if BLOCK_B < 1:
                BLOCK_B = 1

            # Grid: one program per (parent_block, batch_tile). The
            # forward kernel handles all ``bs`` parent rows in the block
            # within a single program because per-block compute is small
            # (BS_PADDED * CBS_PADDED weights) and reusing the linearised
            # ``emars_lin`` across all parents in the block is the main
            # arithmetic win over the general path.
            grid = (NB, triton.cdiv(batch_size, BLOCK_B))

            self._fw_bd_kernel[grid](
                node_mars = node_mars,
                element_mars = element_mars,
                mparams = params,
                batch_size = batch_size,
                nid_start = nid_start,
                cid_start = cid_start,
                pid_start = pid_start,
                BS = bs,
                CBS = cbs,
                BS_PADDED = BS_PADDED,
                CBS_PADDED = CBS_PADDED,
                BLOCK_B = BLOCK_B,
            )

        return None

    # ------------------------------------------------------------------ #
    # Backward (element flows only — phase 1)
    # ------------------------------------------------------------------ #

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 allow_modify_flows: bool = False, propagation_alg: str = "LL",
                 logspace_flows: bool = False, negate_pflows: bool = False,
                 accumulate_ch_flows: bool = False, allow_neg_flows: bool = False,
                 force_use_fp32: bool = False, **kwargs) -> None:

        # ``param_flows`` is accepted (so the standard
        # :meth:`TensorCircuit.backward` pipeline runs through cleanly with
        # ``compute_param_flows=True`` for the input layers / other sums)
        # but **silently ignored** here — the BD kernel writes element
        # flows only. See the plan in
        # ``look-i-m-trying-to-eventual-waterfall.md``; EM-style param-flow
        # accumulation is phase 2 work. Callers that need correct
        # ``param_flows`` for this sum's parameters must use the plain
        # :class:`SumLayer` (e.g. via ``summate(_force_plain=True)``).
        del param_flows
        assert propagation_alg == "LL", (
            "BlockDiagonalSumLayer.backward currently supports only LL."
        )
        assert not allow_neg_flows, (
            "BlockDiagonalSumLayer.backward does not support allow_neg_flows."
        )

        batch_size = node_mars.size(1)

        for block in self._bd_blocks:
            nid_start, cid_start, pid_start, _pfid_start, NB, bs, cbs = block

            CBS_PADDED = triton.next_power_of_2(cbs)
            BS_PADDED = triton.next_power_of_2(bs)
            BLOCK_B = min(triton.next_power_of_2(batch_size), 64)
            if BLOCK_B < 1:
                BLOCK_B = 1

            grid = (NB, triton.cdiv(batch_size, BLOCK_B))

            self._bk_bd_kernel[grid](
                node_flows = node_flows,
                element_flows = element_flows,
                node_mars = node_mars,
                element_mars = element_mars,
                mparams = params,
                batch_size = batch_size,
                nid_start = nid_start,
                cid_start = cid_start,
                pid_start = pid_start,
                BS = bs,
                CBS = cbs,
                BS_PADDED = BS_PADDED,
                CBS_PADDED = CBS_PADDED,
                BLOCK_B = BLOCK_B,
                ALLOW_MODIFY_FLOWS = 1 if allow_modify_flows else 0,
                ACCUMULATE_CH_FLOWS = 1 if accumulate_ch_flows else 0,
                LOGSPACE_FLOWS = 1 if logspace_flows else 0,
            )

        return None

    # ------------------------------------------------------------------ #
    # Triton kernels
    # ------------------------------------------------------------------ #

    @staticmethod
    @triton_jit
    def _fw_bd_kernel(node_mars, element_mars, mparams,
                      batch_size,
                      nid_start, cid_start, pid_start,
                      BS, CBS,
                      BS_PADDED: tl.constexpr,
                      CBS_PADDED: tl.constexpr,
                      BLOCK_B: tl.constexpr):
        """
        Forward for one BlockDiagonalSumLayer block, one program per
        ``(parent_block, batch_tile)``.

        Param layout (matches ``gather_parameters``'s flat layout):
          ``mparams[pid_start + pblock * BS * CBS + c * BS + s]``
          = weight from child_offs ``c`` to parent_offs ``s`` within block
          ``pblock``. Outer-block ordering is dense — block ``pblock``
          occupies ``[pid_start + pblock*BS*CBS, pid_start + (pblock+1)*BS*CBS)``,
          there are no gaps (BD has one block per parent block, no missing
          edges).

        Math (LL propagation):
          1. Load all ``CBS`` child marginals from ``element_mars`` at
             ``cid_start + pblock * CBS + c``.
          2. Stabilise: ``max_b = max_c emars[c, b]`` then ``emars_lin =
             exp(emars - max_b)`` (zero for ``c >= CBS`` padding slots).
          3. Load the ``[BS_PADDED, CBS_PADDED]`` weight tile.
          4. ``acc[s, b] = sum_c weight[s, c] * emars_lin[c, b]``.
          5. ``nmars[s, b] = log(acc + 1e-24) + max_b`` (with -inf
             passthrough for fully-masked batches).

        ``BS`` and ``CBS`` are runtime ints so a single binary serves every
        (tied or untied) block regardless of summate-specific shape;
        register pressure is bounded by ``BS_PADDED * CBS_PADDED *
        BLOCK_B``, both padded to powers of two.
        """

        pid_nb = tl.program_id(0)       # parent block id
        pid_b = tl.program_id(1)        # batch tile

        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        offs_c = tl.arange(0, CBS_PADDED)
        mask_c = offs_c < CBS

        offs_s = tl.arange(0, BS_PADDED)
        mask_s = offs_s < BS

        # Load all CBS child marginals for this parent block. Children for
        # block ``pid_nb`` live at ``cid_start + pid_nb*CBS + c`` — disjoint
        # from every other parent block (no cross-block read).
        emars_ptr = (
            (cid_start + pid_nb * CBS + offs_c)[:, None] * batch_size
            + offs_batch[None, :]
        )
        emars = tl.load(
            element_mars + emars_ptr,
            mask = mask_c[:, None] & mask_batch[None, :],
            other = -float("inf"),
        )                                                                       # [CBS_PADDED, BLOCK_B]

        # Stabilise + linearise per-batch.
        emars_max = tl.max(emars, axis = 0)                                     # [BLOCK_B]
        emars_max_safe = tl.where(emars_max == -float("inf"), 0.0, emars_max)
        emars_lin = tl.where(
            mask_c[:, None],
            tl.exp(emars - emars_max_safe[None, :]),
            0.0,
        )                                                                       # [CBS_PADDED, BLOCK_B]

        # Load weights for this block: row-major ``[BS, CBS]`` with strides
        # ``(1, BS)`` — i.e., ``mparams[pid_start + pblock*BS*CBS + c*BS + s]``.
        # Cast pid_nb to int64 so worst-case range analysis on
        # ``pblock * BS * CBS`` doesn't trip Triton's int32 overflow check
        # for large NB.
        block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
        weight_addr = (
            block_base
            + offs_c[None, :] * BS
            + offs_s[:, None]
        )                                                                       # [BS_PADDED, CBS_PADDED]
        weight = tl.load(
            mparams + weight_addr,
            mask = mask_s[:, None] & mask_c[None, :],
            other = 0.0,
        )
        weight = weight.to(tl.float32)

        # acc[s, b] = sum_c weight[s, c] * emars_lin[c, b]
        acc = tl.sum(weight[:, :, None] * emars_lin[None, :, :], axis = 1)      # [BS_PADDED, BLOCK_B]

        # Reassemble log-space output. ``emars_max == -inf`` => every child
        # is -inf => output stays -inf (acc would be 0 anyway, but the
        # explicit branch avoids ``log(1e-24) + (-inf)`` arithmetic on the
        # padded path).
        nmars = tl.where(
            emars_max[None, :] == -float("inf"),
            -float("inf"),
            tl.log(acc + 1e-24) + emars_max[None, :],
        )

        out_ptr = (
            (nid_start + pid_nb * BS + offs_s)[:, None] * batch_size
            + offs_batch[None, :]
        )
        tl.store(
            node_mars + out_ptr,
            nmars,
            mask = mask_s[:, None] & mask_batch[None, :],
        )

    @staticmethod
    @triton_jit
    def _bk_bd_kernel(node_flows, element_flows, node_mars, element_mars,
                      mparams,
                      batch_size,
                      nid_start, cid_start, pid_start,
                      BS, CBS,
                      BS_PADDED: tl.constexpr,
                      CBS_PADDED: tl.constexpr,
                      BLOCK_B: tl.constexpr,
                      ALLOW_MODIFY_FLOWS: tl.constexpr,
                      ACCUMULATE_CH_FLOWS: tl.constexpr,
                      LOGSPACE_FLOWS: tl.constexpr):
        """
        Element-flow backward for one BlockDiagonalSumLayer block, one
        program per ``(parent_block, batch_tile)``.

        Math (LL, no param_flows):

          1. Load all ``BS`` parent flows + marginals for this block.
          2. ``log_n_fdm[s, b] = log(parent_flow[s, b] + 1e-32) - log_marg[s, b]``
             (or read directly if ``ALLOW_MODIFY_FLOWS=1`` because
             ``DenseSumLayer.modify_flows`` already wrote it in place).
          3. Per-batch stabilisation:
             ``m_b = max_s log_n_fdm[s, b]``,
             ``n_fdm_sub[s, b] = exp(log_n_fdm[s, b] - m_b)`` (0 if m_b is
             -inf).
          4. ``partial[c, b] = sum_s weight[s, c] * n_fdm_sub[s, b]``.
          5. ``log_child_marg[c, b]`` from element_mars (block-local slice).
          6. ``eflows[c, b] = partial[c, b] * exp(log_child_marg[c, b] + m_b)``
             (with -inf passthrough).
          7. Store into ``element_flows[cid_start + pblock*CBS + c, batch]``.
             Disjoint child slices per block ⇒ no atomic-add needed even
             across program-grid neighbours (a child slot belongs to
             exactly one parent block).

        ``ACCUMULATE_CH_FLOWS=1`` adds the result onto existing
        ``element_flows`` content rather than overwriting (used when the
        child has multiple sum-layer parents).
        """

        pid_nb = tl.program_id(0)       # parent block id
        pid_b = tl.program_id(1)        # batch tile

        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(tl.multiple_of(offs_batch, BLOCK_B), BLOCK_B)
        mask_batch = offs_batch < batch_size

        offs_s = tl.arange(0, BS_PADDED)
        mask_s = offs_s < BS

        offs_c = tl.arange(0, CBS_PADDED)
        mask_c = offs_c < CBS

        par_ptr = (
            (nid_start + pid_nb * BS + offs_s)[:, None] * batch_size
            + offs_batch[None, :]
        )                                                                       # [BS_PADDED, BLOCK_B]
        nflows = tl.load(
            node_flows + par_ptr,
            mask = mask_s[:, None] & mask_batch[None, :],
            other = 0.0,
        )
        if ALLOW_MODIFY_FLOWS == 1:
            # The TensorCircuit-level ``modify_flows`` pre-pass already
            # wrote ``log(parent_flow) - log_marg`` (or
            # ``log_parent_flow - log_marg`` under logspace_flows) into
            # ``node_flows`` in place — both produce the canonical
            # ``log_n_fdm`` we want.
            log_n_fdm = nflows
        else:
            nmars = tl.load(
                node_mars + par_ptr,
                mask = mask_s[:, None] & mask_batch[None, :],
                other = -float("inf"),
            )
            if LOGSPACE_FLOWS == 1:
                # ``node_flows`` is already in log-space when callers set
                # ``logspace_flows=True`` (typical for large-fan circuits
                # where linear flows would underflow). Subtract log-marg
                # directly without taking another ``log``.
                log_n_fdm = tl.where(
                    nmars == -float("inf"),
                    -float("inf"),
                    nflows - nmars,
                )
            else:
                log_n_fdm = tl.where(
                    nmars == -float("inf"),
                    -float("inf"),
                    tl.log(nflows + 1e-32) - nmars,
                )                                                                   # [BS_PADDED, BLOCK_B]

        # Mask padded parent rows so they don't poison the max / sum.
        log_n_fdm = tl.where(mask_s[:, None], log_n_fdm, -float("inf"))

        # Per-batch stabilisation across BS parent rows.
        log_n_fdm_max = tl.max(log_n_fdm, axis = 0)                             # [BLOCK_B]
        log_n_fdm_max_safe = tl.where(
            log_n_fdm_max == -float("inf"), 0.0, log_n_fdm_max,
        )
        n_fdm_sub = tl.where(
            mask_s[:, None],
            tl.exp(log_n_fdm - log_n_fdm_max_safe[None, :]),
            0.0,
        )                                                                       # [BS_PADDED, BLOCK_B]

        # Load this block's child marginals (log space).
        emars_ptr = (
            (cid_start + pid_nb * CBS + offs_c)[:, None] * batch_size
            + offs_batch[None, :]
        )
        emars = tl.load(
            element_mars + emars_ptr,
            mask = mask_c[:, None] & mask_batch[None, :],
            other = -float("inf"),
        )                                                                       # [CBS_PADDED, BLOCK_B]

        # Load weights ``[BS_PADDED, CBS_PADDED]`` (same layout as forward).
        block_base = pid_start + pid_nb.to(tl.int64) * BS * CBS
        weight_addr = (
            block_base
            + offs_c[None, :] * BS
            + offs_s[:, None]
        )
        weight = tl.load(
            mparams + weight_addr,
            mask = mask_s[:, None] & mask_c[None, :],
            other = 0.0,
        )
        weight = weight.to(tl.float32)

        # partial[c, b] = sum_s weight[s, c] * n_fdm_sub[s, b]
        partial = tl.sum(
            weight[:, :, None] * n_fdm_sub[:, None, :],
            axis = 0,
        )                                                                       # [CBS_PADDED, BLOCK_B]

        # eflows[c, b] = partial * exp(emars + log_n_fdm_max).
        log_factor = emars + log_n_fdm_max[None, :]                             # [CBS_PADDED, BLOCK_B]

        out_ptr = (
            (cid_start + pid_nb * CBS + offs_c)[:, None] * batch_size
            + offs_batch[None, :]
        )
        out_mask = mask_c[:, None] & mask_batch[None, :]

        if LOGSPACE_FLOWS == 1:
            # Logspace path: store ``log(eflows) = log(partial) + emars
            # + log_n_fdm_max``. ``partial <= 0`` (numeric underflow on
            # otherwise-zero rows) gets mapped to -inf.
            log_eflows = tl.where(
                (log_factor == -float("inf")) | (~mask_c[:, None]),
                -float("inf"),
                tl.log(partial + 1e-32) + log_factor,
            )
            if ACCUMULATE_CH_FLOWS == 1:
                ori = tl.load(
                    element_flows + out_ptr,
                    mask = out_mask,
                    other = -float("inf"),
                )
                # logaddexp(a, b) with -inf passthrough.
                hi = tl.maximum(log_eflows, ori)
                lo = tl.minimum(log_eflows, ori)
                log_eflows = tl.where(
                    hi == -float("inf"),
                    -float("inf"),
                    hi + tlmath.log1p(tl.exp(lo - hi)),
                )
            tl.store(element_flows + out_ptr, log_eflows, mask = out_mask)
        else:
            eflows = tl.where(
                (log_factor == -float("inf")) | (~mask_c[:, None]),
                0.0,
                partial * tl.exp(log_factor),
            )
            if ACCUMULATE_CH_FLOWS == 1:
                ori = tl.load(
                    element_flows + out_ptr,
                    mask = out_mask,
                    other = 0.0,
                )
                eflows = eflows + ori
            tl.store(element_flows + out_ptr, eflows, mask = out_mask)
