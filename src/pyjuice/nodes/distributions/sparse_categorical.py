from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Any

from .distributions import Distribution


LOG_EPS = -23.0258509299  # log(1e-10), used for (row, col) positions outside the sparsity pattern


class SparseCategorical(Distribution):
    """
    Sparse categorical distribution. The ``num_nodes x num_cats`` emission matrix is
    stored in CSC (Compressed Sparse Column) form:

      * ``csc_indptr``  [num_cats + 1] — column pointers
      * ``csc_indices`` [nnz]          — row ids (latent ids) per CSC entry
      * ``csc_values``  [nnz]          — probabilities at those positions (learnable)

    All kernels are CSC-native: forward and backward are per-(batch, column-slot)
    scatter/gather kernels that load the active column in one contiguous range and write
    log-probs or accumulate flows into exactly the latents that are active for the
    observed token. EM and partition use per-slot kernels with atomic row reductions.
    Positions outside the sparsity pattern are treated as probability ~``1e-10``
    (``LOG_EPS`` in log-space), matching the convention used by ``MaskedCategorical``.

    The sparsity pattern is a meta-parameter: pass ``csc_indptr`` and ``csc_indices``
    through :func:`pyjuice.inputs` kwargs (or :meth:`InputNodes.set_meta_params`). Then
    provide the nonzero probabilities via :meth:`InputNodes.set_params` — values must be
    in CSC order, matching your supplied ``csc_indices``.

    Sampling is not supported in this iteration (``sample_fn`` raises
    ``NotImplementedError``).
    """

    def __init__(self, num_cats: int):
        super(SparseCategorical, self).__init__()

        self.num_cats = num_cats

        # Populated by set_meta_parameters
        self._num_nodes = None       # H
        self._nnz = None             # total nonzeros in the CSC matrix
        self._csc_indptr = None      # [V+1] long, on dist's "home" device (CPU initially)
        self._csc_indices = None     # [nnz] long
        self._max_nnz_per_col = None # max over columns of (indptr[v+1] - indptr[v])

    def get_signature(self):
        return "SparseCategorical"

    @property
    def need_meta_parameters(self):
        return True

    def set_meta_parameters(self, num_nodes: int, csc_indptr: torch.Tensor,
                            csc_indices: torch.Tensor, **kwargs):
        """
        Attach the CSC sparsity pattern of the ``num_nodes x num_cats`` emission matrix.

        :param num_nodes: number of latent rows (H).
        :param csc_indptr: long tensor of shape ``[num_cats + 1]``, column pointers.
        :param csc_indices: long tensor of shape ``[nnz]``, row ids.
        :returns: a zero-initialized ``[nnz]`` flat tensor that becomes the initial
                  ``_params`` of the :class:`InputNodes` (to be overwritten by
                  :meth:`InputNodes.set_params` with the actual probabilities, in CSC
                  order).
        """
        V = self.num_cats
        H = num_nodes

        csc_indptr = torch.as_tensor(csc_indptr, dtype = torch.long).contiguous()
        csc_indices = torch.as_tensor(csc_indices, dtype = torch.long).contiguous()
        assert csc_indptr.dim() == 1 and csc_indptr.numel() == V + 1, \
            f"csc_indptr must have shape [num_cats + 1] = [{V + 1}], got {tuple(csc_indptr.shape)}."
        assert csc_indptr[0].item() == 0, "csc_indptr[0] must be 0."
        nnz = int(csc_indptr[-1].item())
        assert csc_indices.dim() == 1 and csc_indices.numel() == nnz, \
            f"csc_indices must have shape [nnz] = [{nnz}], got {tuple(csc_indices.shape)}."
        if nnz > 0:
            assert csc_indices.min().item() >= 0 and csc_indices.max().item() < H, \
                "csc_indices contains row ids outside [0, num_nodes)."

        col_counts = torch.diff(csc_indptr)
        max_nnz_per_col = int(col_counts.max().item()) if V > 0 else 0

        self._num_nodes = H
        self._nnz = nnz
        self._csc_indptr = csc_indptr
        self._csc_indices = csc_indices
        self._max_nnz_per_col = max(max_nnz_per_col, 1)  # >= 1 to keep Triton tiles non-degenerate

        return torch.zeros(max(nnz, 1), dtype = torch.float32)

    # --- Distribution protocol ----------------------------------------

    def get_metadata(self):
        # All CSC state lives on `self`, not in the layer's metadata buffer.
        return []

    def num_parameters(self):
        # Per-node parameter count isn't meaningful for variable-nnz CSC; InputLayer
        # routes through num_parameters_total and the compute_*_offsets hooks.
        return 1

    def num_param_flows(self):
        return 1

    def num_parameters_total(self, num_nodes: int) -> int:
        assert self._nnz is not None, "Sparsity pattern not set."
        assert num_nodes == self._num_nodes
        return max(self._nnz, 1)

    def num_param_flows_total(self, num_nodes: int) -> int:
        return self.num_parameters_total(num_nodes)

    def compute_pid_offsets(self, num_nodes: int) -> torch.Tensor:
        # All latents in a group share the same CSC base (the group's _param_range[0]).
        # Per-latent s_pids are unused by the custom kernels, but we set them to 0 so
        # downstream code that still loads s_pids_ptr gets a consistent value.
        return torch.zeros(num_nodes, dtype = torch.long)

    def compute_pfid_offsets(self, num_nodes: int) -> torch.Tensor:
        return torch.zeros(num_nodes, dtype = torch.long)

    def compute_mid_offsets(self, num_nodes: int) -> torch.Tensor:
        return torch.zeros(num_nodes, dtype = torch.long)

    def normalize_parameters(self, params: torch.Tensor):
        """Row-wise normalize values (in CSC order) so each latent's active probabilities sum to 1."""
        assert self._csc_indices is not None, "Sparsity pattern not set."
        params = params.reshape(-1).clone()
        if self._nnz == 0:
            return params
        row_ids = self._csc_indices
        H = self._num_nodes
        row_sums = torch.zeros(H, dtype = params.dtype)
        row_sums.scatter_add_(0, row_ids, params)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        return params / row_sums[row_ids]

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0,
                        params: Optional[torch.Tensor] = None, **kwargs):
        assert self._nnz is not None, "Sparsity pattern not set."
        assert num_nodes == self._num_nodes
        if params is not None:
            assert isinstance(params, torch.Tensor)
            assert params.numel() == max(self._nnz, 1)
            return params.reshape(-1)
        if self._nnz == 0:
            return torch.zeros(1, dtype = torch.float32)
        vals = torch.exp(torch.rand(self._nnz, dtype = torch.float32) * -perturbation)
        return self.normalize_parameters(vals)

    def get_data_dtype(self):
        return torch.long

    def _get_constructor(self):
        return SparseCategorical, {"num_cats": self.num_cats}

    def _need_2nd_kernel_dim(self):
        return True

    def move_to_device(self, device):
        if self._csc_indptr is not None:
            self._csc_indptr = self._csc_indptr.to(device)
            self._csc_indices = self._csc_indices.to(device)

    # --- Unused template kernels (required to not compile the default path) ---

    @staticmethod
    def fw_mar_fn(*args, **kwargs):
        raise NotImplementedError("SparseCategorical uses a custom forward kernel.")

    @staticmethod
    def bk_flow_fn(*args, **kwargs):
        raise NotImplementedError("SparseCategorical uses a custom backward kernel.")

    @staticmethod
    def em_fn(*args, **kwargs):
        raise NotImplementedError("SparseCategorical uses a custom EM kernel.")

    @staticmethod
    def partition_fn(*args, **kwargs):
        raise NotImplementedError("SparseCategorical uses a custom partition kernel.")

    # --- Custom dispatch flags ----------------------------------------

    def has_custom_forward(self) -> bool:
        return True

    def has_custom_backward(self) -> bool:
        return True

    def has_custom_em(self) -> bool:
        return True

    def has_custom_partition(self) -> bool:
        return True

    # --- Custom kernel implementations --------------------------------

    def custom_forward(self, layer, params, node_mars, data, batch_size,
                       fw_local_ids = None):
        """Pre-fill this layer's output rows with LOG_EPS, then column-scatter per group."""
        assert fw_local_ids is None, "SparseCategorical does not support partial_eval yet."
        sid, eid = layer._output_ind_range
        node_mars[sid:eid].fill_(LOG_EPS)

        BLOCK_B = 64
        BLOCK_K = max(triton.next_power_of_2(self._max_nnz_per_col), 4)

        for ns in layer.nodes:
            dist = ns.dist
            assert isinstance(dist, SparseCategorical)
            if dist._nnz == 0:
                continue
            var_id = ns.scope.to_list()[0]
            node_offset = ns._output_ind_range[0]
            param_base = ns._param_range[0]

            grid = (triton.cdiv(batch_size, BLOCK_B),
                    triton.cdiv(dist._max_nnz_per_col, BLOCK_K))

            _sparse_cat_forward_kernel[grid](
                data_ptr = data,
                node_mars_ptr = node_mars,
                params_ptr = params,
                csc_indptr_ptr = dist._csc_indptr,
                csc_indices_ptr = dist._csc_indices,
                var_id = var_id,
                node_offset = node_offset,
                param_base = param_base,
                batch_size = batch_size,
                BLOCK_B = BLOCK_B,
                BLOCK_K = BLOCK_K,
            )

    def custom_backward(self, layer, params, param_flows, node_flows, node_mars,
                        data, batch_size, logspace_flows: bool = False):
        BLOCK_B = 64
        BLOCK_K = max(triton.next_power_of_2(self._max_nnz_per_col), 4)

        for ns in layer.nodes:
            dist = ns.dist
            if dist._nnz == 0:
                continue
            var_id = ns.scope.to_list()[0]
            node_offset = ns._output_ind_range[0]
            # Flows accumulate into the param-flow range. For tied nodes, this range
            # points to their own accumulator (later reduced into the source's range by
            # `_pflow_accum_kernel`).
            pflow_base = ns._param_flow_range[0]

            grid = (triton.cdiv(batch_size, BLOCK_B),
                    triton.cdiv(dist._max_nnz_per_col, BLOCK_K))

            _sparse_cat_backward_kernel[grid](
                data_ptr = data,
                node_flows_ptr = node_flows,
                param_flows_ptr = param_flows,
                csc_indptr_ptr = dist._csc_indptr,
                csc_indices_ptr = dist._csc_indices,
                var_id = var_id,
                node_offset = node_offset,
                pflow_base = pflow_base,
                batch_size = batch_size,
                logspace_flows = 1 if logspace_flows else 0,
                BLOCK_B = BLOCK_B,
                BLOCK_K = BLOCK_K,
            )

    def custom_em(self, layer, step_size: float, pseudocount: float,
                  keep_zero_params: bool = True):
        # One EM pass per source InputNodes group (tied duplicates share the source's
        # param range, and their flows were already accumulated into the source's
        # parflow range by `_pflow_accum_kernel` before this call).
        for ns in layer.nodes:
            if ns.is_tied():
                continue
            dist = ns.dist
            if dist._nnz == 0:
                continue
            param_base = ns._param_range[0]
            pflow_base = ns._param_flow_range[0]
            H = dist._num_nodes
            nnz = dist._nnz

            row_sums = torch.zeros(H, dtype = torch.float32, device = layer.device)

            BLOCK = 1024
            grid = (triton.cdiv(nnz, BLOCK),)

            _sparse_cat_em_row_sum_kernel[grid](
                params_ptr = layer.params,
                param_flows_ptr = layer.param_flows,
                csc_indices_ptr = dist._csc_indices,
                row_sums_ptr = row_sums,
                param_base = param_base,
                pflow_base = pflow_base,
                nnz = nnz,
                pseudocount = pseudocount,
                keep_zero_params = 1 if keep_zero_params else 0,
                BLOCK = BLOCK,
            )

            _sparse_cat_em_update_kernel[grid](
                params_ptr = layer.params,
                param_flows_ptr = layer.param_flows,
                csc_indices_ptr = dist._csc_indices,
                row_sums_ptr = row_sums,
                param_base = param_base,
                pflow_base = pflow_base,
                nnz = nnz,
                step_size = step_size,
                pseudocount = pseudocount,
                keep_zero_params = 1 if keep_zero_params else 0,
                BLOCK = BLOCK,
            )

    def custom_partition(self, layer, node_mars):
        sid, eid = layer._output_ind_range
        # Per-latent partition = log(sum of row values). Compute via atomic row sums.
        for ns in layer.nodes:
            dist = ns.dist
            if dist._nnz == 0:
                node_mars[ns._output_ind_range[0]:ns._output_ind_range[1]] = LOG_EPS
                continue
            param_base = ns._param_range[0]
            node_offset = ns._output_ind_range[0]
            H = dist._num_nodes
            nnz = dist._nnz

            row_sums = torch.zeros(H, dtype = torch.float32, device = layer.device)

            BLOCK = 1024
            grid_nnz = (triton.cdiv(nnz, BLOCK),)
            _sparse_cat_partition_accum_kernel[grid_nnz](
                params_ptr = layer.params,
                csc_indices_ptr = dist._csc_indices,
                row_sums_ptr = row_sums,
                param_base = param_base,
                nnz = nnz,
                BLOCK = BLOCK,
            )

            grid_h = (triton.cdiv(H, BLOCK),)
            _sparse_cat_partition_write_kernel[grid_h](
                node_mars_ptr = node_mars,
                row_sums_ptr = row_sums,
                node_offset = node_offset,
                H = H,
                BLOCK = BLOCK,
            )


# =====================================================================
# Triton kernels (module-level so @triton.jit can find them)
# =====================================================================


@triton.jit
def _sparse_cat_forward_kernel(
    data_ptr, node_mars_ptr, params_ptr,
    csc_indptr_ptr, csc_indices_ptr,
    var_id, node_offset, param_base,
    batch_size,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_b = offs_b < batch_size

    # Observed category per batch item.
    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask = mask_b, other = 0)
    col_start = tl.load(csc_indptr_ptr + v, mask = mask_b, other = 0)
    col_end = tl.load(csc_indptr_ptr + v + 1, mask = mask_b, other = 0)
    k_v = col_end - col_start

    slot_mask = mask_b[:, None] & (offs_k[None, :] < k_v[:, None])
    slot_idx = col_start[:, None] + offs_k[None, :]
    row_id = tl.load(csc_indices_ptr + slot_idx, mask = slot_mask, other = 0)
    val = tl.load(params_ptr + param_base + slot_idx, mask = slot_mask, other = 1.0)
    log_val = tl.log(val)

    mars_offs = (node_offset + row_id) * batch_size + offs_b[:, None]
    tl.store(node_mars_ptr + mars_offs, log_val, mask = slot_mask)


@triton.jit
def _sparse_cat_backward_kernel(
    data_ptr, node_flows_ptr, param_flows_ptr,
    csc_indptr_ptr, csc_indices_ptr,
    var_id, node_offset, pflow_base,
    batch_size,
    logspace_flows: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_b = offs_b < batch_size

    v = tl.load(data_ptr + var_id * batch_size + offs_b, mask = mask_b, other = 0)
    col_start = tl.load(csc_indptr_ptr + v, mask = mask_b, other = 0)
    col_end = tl.load(csc_indptr_ptr + v + 1, mask = mask_b, other = 0)
    k_v = col_end - col_start

    slot_mask = mask_b[:, None] & (offs_k[None, :] < k_v[:, None])
    slot_idx = col_start[:, None] + offs_k[None, :]
    row_id = tl.load(csc_indices_ptr + slot_idx, mask = slot_mask, other = 0)

    flow_offs = (node_offset + row_id) * batch_size + offs_b[:, None]
    flow = tl.load(node_flows_ptr + flow_offs, mask = slot_mask, other = 0)
    if logspace_flows:
        flow = tl.exp(flow)

    tl.atomic_add(param_flows_ptr + pflow_base + slot_idx, flow, mask = slot_mask)


@triton.jit
def _sparse_cat_em_row_sum_kernel(
    params_ptr, param_flows_ptr, csc_indices_ptr, row_sums_ptr,
    param_base, pflow_base, nnz,
    pseudocount,
    keep_zero_params: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Pass 1 of EM: per CSC slot, atomically add (flow + pseudocount) to its row's
    # accumulator. This way row_sums[n] = sum over n's active slots of
    # (flow + pseudocount), which equals the denominator used by the update kernel —
    # so the updated row normalizes back to 1 exactly.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz

    flow = tl.load(param_flows_ptr + pflow_base + offs, mask = mask, other = 0.0)
    row_id = tl.load(csc_indices_ptr + offs, mask = mask, other = 0)
    contribution = flow + pseudocount

    if keep_zero_params:
        param = tl.load(params_ptr + param_base + offs, mask = mask, other = 0.0)
        contribution = tl.where(param < 1e-12, 0.0, contribution)

    tl.atomic_add(row_sums_ptr + row_id, contribution, mask = mask)


@triton.jit
def _sparse_cat_em_update_kernel(
    params_ptr, param_flows_ptr, csc_indices_ptr, row_sums_ptr,
    param_base, pflow_base, nnz,
    step_size, pseudocount,
    keep_zero_params: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz

    param = tl.load(params_ptr + param_base + offs, mask = mask, other = 0.0)
    flow = tl.load(param_flows_ptr + pflow_base + offs, mask = mask, other = 0.0)
    row_id = tl.load(csc_indices_ptr + offs, mask = mask, other = 0)
    row_sum = tl.load(row_sums_ptr + row_id, mask = mask, other = 1.0)

    # Guard against degenerate (pseudocount=0 and no observations) rows — leave params
    # alone in that case.
    denom = tl.where(row_sum > 0, row_sum, 1.0)
    new_param = (1.0 - step_size) * param + step_size * (flow + pseudocount) / denom

    if keep_zero_params:
        new_param = tl.where(param < 1e-12, 0.0, new_param)

    tl.store(params_ptr + param_base + offs, new_param, mask = mask)


@triton.jit
def _sparse_cat_partition_accum_kernel(
    params_ptr, csc_indices_ptr, row_sums_ptr,
    param_base, nnz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz

    val = tl.load(params_ptr + param_base + offs, mask = mask, other = 0.0)
    row_id = tl.load(csc_indices_ptr + offs, mask = mask, other = 0)
    tl.atomic_add(row_sums_ptr + row_id, val, mask = mask)


@triton.jit
def _sparse_cat_partition_write_kernel(
    node_mars_ptr, row_sums_ptr,
    node_offset, H,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    s = tl.load(row_sums_ptr + offs, mask = mask, other = 1.0)
    tl.store(node_mars_ptr + node_offset + offs, tl.log(s), mask = mask)
