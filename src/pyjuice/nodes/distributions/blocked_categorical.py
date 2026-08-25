from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Tuple

from .distributions import Distribution


LOG_EPS = -23.0258509299  # log(1e-10), value of inactive (latent, token) cells


class BlockedCategorical(Distribution):
    """
    Block-structured categorical emission distribution (Chiu & Rush, 2020,
    "Scaling Hidden Markov Language Models", arXiv 2011.04640).

    The ``H`` latents are partitioned into ``M`` contiguous groups of
    ``k = H // M`` latents each, and every token ``v`` is owned by exactly one
    group ``token_block[v]``: latent ``h`` can emit ``v`` iff
    ``h // k == token_block[v]``. The ``H x V`` emission matrix is therefore
    block-diagonal (after permuting the vocabulary) and is stored densely as
    ``[V, k]`` — one contiguous ``k``-vector per token::

        params[v * k + j] = P(x = v | z = token_block[v] * k + j)

    Positions outside the pattern have probability ~``1e-10`` (``LOG_EPS`` in
    log space), the same convention as :class:`SparseCategorical` /
    :class:`MaskedCategorical`.

    This is exactly the :class:`SparseCategorical` model with
    ``csc_indptr = arange(V + 1) * k`` and
    ``csc_indices[v * k + j] = token_block[v] * k + j`` (see
    :meth:`to_csc`); the point of a dedicated class is that the active set of
    an observed token is one *implicit, contiguous, block-aligned* row range,
    which the blocked layers (:class:`BlockedProdLayer`,
    :class:`BlockedIOSumLayer`, ...) exploit: no index arrays, no host-side
    pattern build, contiguous parameter tiles.

    Meta-parameters (pass through :func:`pyjuice.inputs` kwargs):

      * ``token_block`` — long tensor ``[V]`` with values in ``[0, M)``.
      * ``num_blocks``  — ``M``; must divide ``num_nodes``.

    Sampling is not supported.
    """

    def __init__(self, num_cats: int):
        super(BlockedCategorical, self).__init__()

        self.num_cats = num_cats

        # Populated by set_meta_parameters
        self._num_nodes = None          # H
        self._num_blocks = None         # M
        self._states_per_block = None   # k = H // M
        self._token_block = None        # [V] long, block id per token
        self._block_token_counts = None # [M] long, tokens owned by each block

    def get_signature(self):
        return "BlockedCategorical"

    @property
    def need_meta_parameters(self):
        return True

    @property
    def k(self) -> int:
        return self._states_per_block

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def set_meta_parameters(self, num_nodes: int, token_block: torch.Tensor,
                            num_blocks: int, **kwargs):
        """
        Attach the token → block assignment of the ``num_nodes x num_cats``
        emission matrix.

        :returns: a random row-normalised ``[V * k]`` flat tensor that becomes
                  the initial ``_params`` of the :class:`InputNodes` (kept
                  unless :meth:`InputNodes.set_params` overwrites it; token-major).
                  ``InputNodes`` treats the returned tensor as *set* params, so
                  a zero init here would silently train from ``log 0``.
        """
        V = self.num_cats
        H = num_nodes
        M = int(num_blocks)
        assert M >= 1 and H % M == 0, (
            f"num_blocks ({M}) must divide num_nodes ({H})."
        )
        k = H // M

        token_block = torch.as_tensor(token_block, dtype = torch.long).reshape(-1).contiguous()
        assert token_block.numel() == V, (
            f"token_block must have shape [num_cats] = [{V}], got {tuple(token_block.shape)}."
        )
        if V > 0:
            assert token_block.min().item() >= 0 and token_block.max().item() < M, \
                "token_block contains block ids outside [0, num_blocks)."

        self._num_nodes = H
        self._num_blocks = M
        self._states_per_block = k
        self._token_block = token_block
        self._block_token_counts = torch.bincount(token_block, minlength = M)

        return self.init_parameters(H, perturbation = 2.0)

    # --- Distribution protocol ----------------------------------------

    def get_metadata(self):
        return []

    def num_parameters(self):
        return 1

    def num_param_flows(self):
        return 1

    def num_parameters_total(self, num_nodes: int) -> int:
        assert self._states_per_block is not None, "Meta-parameters not set."
        assert num_nodes == self._num_nodes
        return max(self.num_cats * self._states_per_block, 1)

    def num_param_flows_total(self, num_nodes: int) -> int:
        return self.num_parameters_total(num_nodes)

    def compute_pid_offsets(self, num_nodes: int) -> torch.Tensor:
        return torch.zeros(num_nodes, dtype = torch.long)

    def compute_pfid_offsets(self, num_nodes: int) -> torch.Tensor:
        return torch.zeros(num_nodes, dtype = torch.long)

    def compute_mid_offsets(self, num_nodes: int) -> torch.Tensor:
        return torch.zeros(num_nodes, dtype = torch.long)

    def _row_sums(self, params_vk: torch.Tensor) -> torch.Tensor:
        """``[M, k]`` per-latent sums of a ``[V, k]`` (token-major) tensor:
        ``out[m, j] = Σ_{v : token_block[v] == m} params_vk[v, j]``."""
        M, k = self._num_blocks, self._states_per_block
        tb = self._token_block.to(params_vk.device)
        out = torch.zeros(M, k, dtype = params_vk.dtype, device = params_vk.device)
        out.index_add_(0, tb, params_vk)
        return out

    def normalize_parameters(self, params: torch.Tensor):
        """Normalise so each latent's probabilities over its block's tokens sum to 1."""
        assert self._token_block is not None, "Meta-parameters not set."
        V, k = self.num_cats, self._states_per_block
        params = params.reshape(-1).clone()
        if V == 0:
            return params
        p = params.view(V, k)
        row_sums = self._row_sums(p)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        tb = self._token_block.to(p.device)
        p = p / row_sums[tb]
        return p.reshape(-1)

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0,
                        params: Optional[torch.Tensor] = None, **kwargs):
        assert self._token_block is not None, "Meta-parameters not set."
        assert num_nodes == self._num_nodes
        n = self.num_parameters_total(num_nodes)
        if params is not None:
            assert isinstance(params, torch.Tensor)
            assert params.numel() == n
            return params.reshape(-1)
        if self.num_cats == 0:
            return torch.zeros(1, dtype = torch.float32)
        vals = torch.exp(torch.rand(n, dtype = torch.float32) * -perturbation)
        return self.normalize_parameters(vals)

    def get_data_dtype(self):
        return torch.long

    def _get_constructor(self):
        return BlockedCategorical, {"num_cats": self.num_cats}

    def _need_2nd_kernel_dim(self):
        return True

    def move_to_device(self, device):
        if self._token_block is not None:
            self._token_block = self._token_block.to(device)

    # --- Conversions ---------------------------------------------------

    def to_csc(self, params: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Equivalent :class:`SparseCategorical` pattern ``(csc_indptr,
        csc_indices, csc_values)``. The CSC slot order coincides with the
        token-major ``[V, k]`` layout, so ``params`` (if given) is returned
        as ``csc_values`` unchanged."""
        V, k = self.num_cats, self._states_per_block
        tb = self._token_block.cpu()
        csc_indptr = torch.arange(V + 1, dtype = torch.long) * k
        csc_indices = (tb[:, None] * k + torch.arange(k, dtype = torch.long)[None, :]).reshape(-1)
        csc_values = None if params is None else params.reshape(-1).cpu().clone()
        return csc_indptr, csc_indices, csc_values

    def to_dense(self, params: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """Dense ``[H, V]`` emission matrix with ``fill_value`` outside the pattern."""
        V, k, H = self.num_cats, self._states_per_block, self._num_nodes
        tb = self._token_block.cpu()
        p = params.reshape(V, k).cpu()
        dense = torch.full((H, V), fill_value, dtype = p.dtype)
        rows = tb[:, None] * k + torch.arange(k)[None, :]          # [V, k]
        cols = torch.arange(V)[:, None].expand(V, k)
        dense[rows.reshape(-1), cols.reshape(-1)] = p.reshape(-1)
        return dense

    # --- Unused template kernels (required to not compile the default path) ---

    @staticmethod
    def fw_mar_fn(*args, **kwargs):
        raise NotImplementedError("BlockedCategorical uses a custom forward kernel.")

    @staticmethod
    def bk_flow_fn(*args, **kwargs):
        raise NotImplementedError("BlockedCategorical uses a custom backward kernel.")

    @staticmethod
    def em_fn(*args, **kwargs):
        raise NotImplementedError("BlockedCategorical uses a custom EM kernel.")

    @staticmethod
    def partition_fn(*args, **kwargs):
        raise NotImplementedError("BlockedCategorical uses a custom partition kernel.")

    # --- Custom dispatch flags ----------------------------------------

    def has_custom_forward(self) -> bool:
        return True

    def has_custom_backward(self) -> bool:
        return True

    def has_custom_em(self) -> bool:
        return True

    def has_custom_partition(self) -> bool:
        return True

    # --- Custom kernel implementations (standalone InputLayer path) ----

    def custom_forward(self, layer, params, node_mars, data, batch_size,
                       fw_local_ids = None):
        """Per-ns LOG_EPS fill + write of the observed token's ``k`` active
        rows, skipping nodes claimed by a downstream :class:`BlockedProdLayer`
        (``_skip_input_forward`` flag)."""
        assert fw_local_ids is None, "BlockedCategorical does not support partial_eval yet."

        BLOCK_B = 64
        k = self._states_per_block
        BLOCK_K = max(min(triton.next_power_of_2(k), 256), 4)

        for ns in layer.nodes:
            if getattr(ns, "_skip_input_forward", False):
                continue

            dist = ns.dist
            assert isinstance(dist, BlockedCategorical)
            sid, eid = ns._output_ind_range
            node_mars[sid:eid].fill_(LOG_EPS)
            if dist.num_cats == 0:
                continue

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(k, BLOCK_K))
            _blocked_cat_forward_kernel[grid](
                data_ptr = data,
                node_mars_ptr = node_mars,
                params_ptr = params,
                token_block_ptr = dist._token_block,
                var_id = ns.scope.to_list()[0],
                node_offset = sid,
                param_base = ns._param_range[0],
                batch_size = batch_size,
                K = k,
                BLOCK_B = BLOCK_B,
                BLOCK_K = BLOCK_K,
            )

    def custom_backward(self, layer, params, param_flows, node_flows, node_mars,
                        data, batch_size, logspace_flows: bool = False):
        if param_flows is None:
            return

        BLOCK_B = 64
        k = self._states_per_block
        BLOCK_K = max(min(triton.next_power_of_2(k), 256), 4)

        for ns in layer.nodes:
            if getattr(ns, "_skip_input_backward", False):
                continue
            dist = ns.dist
            if dist.num_cats == 0:
                continue

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(k, BLOCK_K))
            _blocked_cat_backward_kernel[grid](
                data_ptr = data,
                node_flows_ptr = node_flows,
                param_flows_ptr = param_flows,
                token_block_ptr = dist._token_block,
                var_id = ns.scope.to_list()[0],
                node_offset = ns._output_ind_range[0],
                pflow_base = ns._param_flow_range[0],
                batch_size = batch_size,
                K = k,
                logspace_flows = 1 if logspace_flows else 0,
                BLOCK_B = BLOCK_B,
                BLOCK_K = BLOCK_K,
            )

    def custom_backward_blocked(self, input_layer, blocked_flow, pflows_base: int,
                                logspace_flows: bool = False):
        """Accumulate emission parameter flows from a
        :class:`BlockedNodeValues` produced by the blocked chain:
        ``param_flows[pflows_base + v_b * k + j] += blocked_flow.values[b, j]``
        with ``v_b = data[var_id, b]``. Samples observing the same token
        collide — the atomic is the cross-sample reduction EM needs."""
        if input_layer.param_flows is None or self.num_cats == 0:
            return
        B = blocked_flow.batch_size
        k = blocked_flow.k
        BLOCK_B = 8
        BLOCK_K = max(min(triton.next_power_of_2(k), 256), 4)
        grid = (triton.cdiv(k, BLOCK_K), triton.cdiv(B, BLOCK_B))
        _blocked_cat_backward_blocked_kernel[grid](
            values_ptr = blocked_flow.values,
            param_flows_ptr = input_layer.param_flows,
            data_ptr = blocked_flow.data,
            var_id = blocked_flow.var_id,
            pflow_base = pflows_base,
            batch_size = B,
            K = k,
            logspace_flows = 1 if logspace_flows else 0,
            BLOCK_B = BLOCK_B,
            BLOCK_K = BLOCK_K,
        )

    def custom_em(self, layer, step_size: float, pseudocount: float,
                  keep_zero_params: bool = True):
        # One EM pass per source InputNodes group (tied duplicates share the
        # source's param range; their flows were already fused into the
        # source's parflow range by `_pflow_accum_kernel`).
        for ns in layer.nodes:
            if ns.is_tied():
                continue
            dist = ns.dist
            if dist.num_cats == 0:
                continue
            V, k = dist.num_cats, dist._states_per_block
            psid = ns._param_range[0]
            pfsid = ns._param_flow_range[0]
            params = layer.params[psid:psid + V * k].view(V, k)
            flows = layer.param_flows[pfsid:pfsid + V * k].view(V, k)

            contribution = flows + pseudocount
            if keep_zero_params:
                contribution = torch.where(params < 1e-12, torch.zeros_like(contribution), contribution)
            row_sums = dist._row_sums(contribution)                     # [M, k]
            denom = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
            tb = dist._token_block.to(params.device)
            new_params = (1.0 - step_size) * params + step_size * contribution / denom[tb]
            if keep_zero_params:
                new_params = torch.where(params < 1e-12, torch.zeros_like(new_params), new_params)
            params.copy_(new_params)

    def custom_partition(self, layer, node_mars):
        for ns in layer.nodes:
            dist = ns.dist
            sid, eid = ns._output_ind_range
            if dist.num_cats == 0:
                node_mars[sid:eid] = LOG_EPS
                continue
            V, k = dist.num_cats, dist._states_per_block
            psid = ns._param_range[0]
            params = layer.params[psid:psid + V * k].view(V, k)
            row_sums = dist._row_sums(params).reshape(-1)               # [H]
            node_mars[sid:eid] = torch.log(row_sums).to(node_mars.dtype)


# =====================================================================
# Triton kernels
# =====================================================================


@triton.jit
def _blocked_cat_forward_kernel(
    data_ptr, node_mars_ptr, params_ptr, token_block_ptr,
    var_id, node_offset, param_base,
    batch_size, K,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_j = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_b = offs_b < batch_size
    mask_j = offs_j < K
    mask = mask_b[:, None] & mask_j[None, :]

    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask = mask_b, other = 0)
    m = tl.load(token_block_ptr + v, mask = mask_b, other = 0)

    val = tl.load(params_ptr + param_base + v[:, None] * K + offs_j[None, :],
                  mask = mask, other = 1.0)
    row = m[:, None] * K + offs_j[None, :]
    tl.store(node_mars_ptr + (node_offset + row) * batch_size + offs_b[:, None],
             tl.log(val), mask = mask)


@triton.jit
def _blocked_cat_backward_kernel(
    data_ptr, node_flows_ptr, param_flows_ptr, token_block_ptr,
    var_id, node_offset, pflow_base,
    batch_size, K,
    logspace_flows: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_j = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_b = offs_b < batch_size
    mask_j = offs_j < K
    mask = mask_b[:, None] & mask_j[None, :]

    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask = mask_b, other = 0)
    m = tl.load(token_block_ptr + v, mask = mask_b, other = 0)

    row = m[:, None] * K + offs_j[None, :]
    flow = tl.load(node_flows_ptr + (node_offset + row) * batch_size + offs_b[:, None],
                   mask = mask, other = 0.0)
    if logspace_flows:
        flow = tl.exp(flow)

    tl.atomic_add(param_flows_ptr + pflow_base + v[:, None] * K + offs_j[None, :],
                  flow, mask = mask)


@triton.jit(do_not_specialize = ["pflow_base", "batch_size", "K"])
def _blocked_cat_backward_blocked_kernel(
    values_ptr, param_flows_ptr, data_ptr,
    var_id, pflow_base,
    batch_size, K,
    logspace_flows: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``param_flows[pflow_base + data[var_id, b] * K + j] += values[b, j]``."""
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_j = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size
    mask = mask_b[:, None] & (offs_j[None, :] < K)

    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask = mask_b, other = 0)
    flow = tl.load(values_ptr + offs_b[:, None] * K + offs_j[None, :], mask = mask, other = 0.0)
    if logspace_flows:
        flow = tl.exp(flow)
    tl.atomic_add(param_flows_ptr + pflow_base + v[:, None] * K + offs_j[None, :],
                  flow, mask = mask)
