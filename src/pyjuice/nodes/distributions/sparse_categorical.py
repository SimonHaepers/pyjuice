from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Any

from .distributions import Distribution


LOG_EPS = -23.0258509299  # log(1e-10), matches MaskedCategorical convention


class SparseCategorical(Distribution):
    """
    A categorical distribution over `num_cats` values whose emission matrix of shape
    `[num_nodes, num_cats]` is sparse and stored in CSC (Compressed Sparse Column) form.

    The sparsity pattern is a meta-parameter: it must be provided at construction time via
    :meth:`pyjuice.nodes.InputNodes.set_meta_params` (or passed through
    :func:`pyjuice.inputs` kwargs) and is fixed for the lifetime of the node. The
    probability values at the nonzero positions are the learnable parameters.

    User-facing CSC shape: ``[num_nodes, num_cats]``. ``csc_indptr`` has length
    ``num_cats + 1``; ``csc_indices`` has length ``nnz`` and holds row ids (node ids).

    Internally, values are stored in row-major order (latents contiguous) so each
    per-node Triton thread loads its row with contiguous accesses. A permutation from
    the user's CSC order to this internal row-major order is cached so that callers can
    convert CSC-ordered values via :meth:`csc_values_to_row_major`.

    Forward/backward/EM/sample kernels treat positions outside the sparsity pattern as
    if they had probability ``1e-10`` (``LOG_EPS`` in log-space) — they are not updated
    by EM.
    """

    def __init__(self, num_cats: int):
        super(SparseCategorical, self).__init__()

        self.num_cats = num_cats

        # Populated by set_meta_parameters
        self._num_nodes = None
        self._nnz = None
        self._row_ptr = None           # [H+1] long, cumsum0 of nnz_per_row
        self._col_ids_row_major = None # [nnz] long, column ids in row-major order (sorted per row)
        self._csc_to_row_perm = None   # [nnz] long, permutation from CSC order to row-major
        self._metadata_list = None     # flat list for InputLayer: [k_0, col_0_0, ..., k_1, col_1_0, ...]
        self._mid_offsets = None       # [H] long, per-node offsets into the group metadata slice
        self._pid_offsets = None       # [H] long, per-node offsets into the group params slice

    def get_signature(self):
        return "SparseCategorical"

    @property
    def need_meta_parameters(self):
        return True

    def set_meta_parameters(self, num_nodes: int, csc_indptr: torch.Tensor,
                            csc_indices: torch.Tensor, **kwargs):
        """
        Attach the CSC sparsity pattern of the `num_nodes x num_cats` emission matrix.

        :param num_nodes: number of rows (latent nodes) — must equal ``InputNodes.num_nodes``.
        :param csc_indptr: long tensor of shape ``[num_cats + 1]``; column pointers.
        :param csc_indices: long tensor of shape ``[nnz]``; row indices.
        :returns: a zero-initialized flat tensor of shape ``[nnz]`` (the storage for
                  learnable values, in row-major order). Users should then provide the
                  actual values via :meth:`pyjuice.nodes.InputNodes.set_params` either
                  in row-major order or after converting via
                  :meth:`csc_values_to_row_major`.
        """
        V = self.num_cats
        H = num_nodes

        csc_indptr = torch.as_tensor(csc_indptr, dtype = torch.long)
        csc_indices = torch.as_tensor(csc_indices, dtype = torch.long)
        assert csc_indptr.dim() == 1 and csc_indptr.numel() == V + 1, \
            f"csc_indptr must have shape [num_cats + 1] = [{V + 1}], got {tuple(csc_indptr.shape)}."
        assert csc_indptr[0].item() == 0, "csc_indptr[0] must be 0."
        nnz = int(csc_indptr[-1].item())
        assert csc_indices.dim() == 1 and csc_indices.numel() == nnz, \
            f"csc_indices must have shape [nnz] = [{nnz}], got {tuple(csc_indices.shape)}."
        if nnz > 0:
            assert csc_indices.min().item() >= 0 and csc_indices.max().item() < H, \
                "csc_indices contains row ids outside [0, num_nodes)."

        # Expand CSC to COO
        col_counts = torch.diff(csc_indptr)  # [V]
        col_ids_coo = torch.repeat_interleave(torch.arange(V, dtype = torch.long), col_counts)  # [nnz]
        row_ids_coo = csc_indices  # [nnz]

        # Stable sort by (row, col) → row-major order with cols sorted within each row
        sort_key = row_ids_coo * V + col_ids_coo
        perm = torch.argsort(sort_key, stable = True)
        col_ids_row = col_ids_coo[perm]
        row_ids_row = row_ids_coo[perm]

        # Per-row structure
        nnz_per_row = torch.bincount(row_ids_row, minlength = H)          # [H]
        row_ptr = torch.cat([torch.zeros(1, dtype = torch.long), nnz_per_row.cumsum(0)])  # [H+1]

        # Per-node offsets
        pid_offsets = row_ptr[:-1].clone()                                # [H]
        # Metadata layout per row: [k_n, col_0, ..., col_{k_n-1}] → size (1 + k_n)
        row_meta_sizes = nnz_per_row + 1                                  # [H]
        mid_offsets = torch.cat([torch.zeros(1, dtype = torch.long),
                                 row_meta_sizes.cumsum(0)])[:-1]          # [H]

        # Flat metadata list for InputLayer
        metadata_list = []
        for h in range(H):
            k_h = int(nnz_per_row[h].item())
            metadata_list.append(float(k_h))
            if k_h > 0:
                rs = int(row_ptr[h].item())
                re = int(row_ptr[h + 1].item())
                metadata_list.extend(col_ids_row[rs:re].tolist())

        # Cache everything
        self._num_nodes = H
        self._nnz = nnz
        self._row_ptr = row_ptr
        self._col_ids_row_major = col_ids_row.contiguous()
        self._csc_to_row_perm = perm.contiguous()
        self._metadata_list = metadata_list
        self._mid_offsets = mid_offsets.contiguous()
        self._pid_offsets = pid_offsets.contiguous()

        # Return zero-initialized values (row-major)
        return torch.zeros(max(nnz, 1), dtype = torch.float32)

    def csc_values_to_row_major(self, csc_values: torch.Tensor) -> torch.Tensor:
        """Permute a CSC-ordered value tensor to the internal row-major order."""
        assert self._csc_to_row_perm is not None, "Sparsity pattern not set."
        assert csc_values.numel() == self._nnz
        return csc_values.reshape(-1)[self._csc_to_row_perm]

    def row_major_to_csc_values(self, row_values: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`csc_values_to_row_major`."""
        assert self._csc_to_row_perm is not None, "Sparsity pattern not set."
        assert row_values.numel() == self._nnz
        out = torch.empty_like(row_values.reshape(-1))
        out[self._csc_to_row_perm] = row_values.reshape(-1)
        return out

    def get_metadata(self):
        assert self._metadata_list is not None, "Sparsity pattern not set."
        return list(self._metadata_list)

    def num_parameters(self):
        # Per-node parameter count is not well defined for variable sparsity; the
        # InputLayer routes through num_parameters_total / compute_pid_offsets instead.
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
        assert self._pid_offsets is not None, "Sparsity pattern not set."
        assert num_nodes == self._num_nodes
        return self._pid_offsets.clone()

    def compute_pfid_offsets(self, num_nodes: int) -> torch.Tensor:
        return self.compute_pid_offsets(num_nodes)

    def compute_mid_offsets(self, num_nodes: int) -> torch.Tensor:
        assert self._mid_offsets is not None, "Sparsity pattern not set."
        assert num_nodes == self._num_nodes
        return self._mid_offsets.clone()

    def normalize_parameters(self, params: torch.Tensor):
        """Row-wise normalize values so each latent's active probabilities sum to 1."""
        assert self._row_ptr is not None, "Sparsity pattern not set."
        params = params.reshape(-1).clone()
        row_ptr = self._row_ptr
        H = self._num_nodes
        for h in range(H):
            rs = int(row_ptr[h].item())
            re = int(row_ptr[h + 1].item())
            if re > rs:
                s = params[rs:re].sum()
                if s > 0:
                    params[rs:re] = params[rs:re] / s
                else:
                    params[rs:re] = 1.0 / (re - rs)
        return params

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0,
                        params: Optional[torch.Tensor] = None, **kwargs):
        assert self._nnz is not None, "Sparsity pattern not set."
        assert num_nodes == self._num_nodes
        if params is not None:
            assert isinstance(params, torch.Tensor)
            assert params.numel() == max(self._nnz, 1)
            return params.reshape(-1)
        vals = torch.exp(torch.rand(max(self._nnz, 1), dtype = torch.float32) * -perturbation)
        return self.normalize_parameters(vals)

    def get_data_dtype(self):
        return torch.long

    def _get_constructor(self):
        return SparseCategorical, {"num_cats": self.num_cats}

    def _need_2nd_kernel_dim(self):
        return True

    # -----------------------------------------------------------------
    # Triton kernels
    # -----------------------------------------------------------------

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask,
                  num_vars_per_node, BLOCK_SIZE):
        # Load row header: k_n = number of nonzeros in this row.
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        k_n = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_k = tl.max(k_n, axis = 0)

        LOG_EPS_VAL = -23.0258509299
        log_probs = tl.zeros([BLOCK_SIZE], dtype = tl.float32) + LOG_EPS_VAL

        # Linear scan over row-n's column ids, looking for a match with `data`.
        for i in range(max_k):
            slot_mask = mask & (i < k_n)
            col_id = tl.load(metadata_ptr + s_mids + 1 + i, mask = slot_mask, other = 0).to(tl.int64)
            hit_mask = slot_mask & (col_id == data)
            val = tl.load(params_ptr + s_pids + i, mask = hit_mask, other = 0)
            log_probs = tl.where(hit_mask, tl.log(val), log_probs)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr,
                   param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                   num_vars_per_node, BLOCK_SIZE):
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        k_n = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_k = tl.max(k_n, axis = 0)

        for i in range(max_k):
            slot_mask = mask & (i < k_n)
            col_id = tl.load(metadata_ptr + s_mids + 1 + i, mask = slot_mask, other = 0).to(tl.int64)
            hit_mask = slot_mask & (col_id == data)
            tl.atomic_add(param_flows_ptr + s_pfids + i, flows, mask = hit_mask)

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr,
                  metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        k_n = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_k = tl.max(k_n, axis = 0)

        rnd_val = tl.rand(seed, tl.arange(0, BLOCK_SIZE))
        sampled_col = tl.zeros([BLOCK_SIZE], dtype = tl.int64) - 1
        cum_param = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

        for i in range(max_k):
            slot_mask = mask & (i < k_n)
            val = tl.load(params_ptr + s_pids + i, mask = slot_mask, other = 0)
            cum_param += val
            col_id = tl.load(metadata_ptr + s_mids + 1 + i, mask = slot_mask, other = 0).to(tl.int64)
            pick = slot_mask & (cum_param >= rnd_val) & (sampled_col == -1)
            sampled_col = tl.where(pick, col_id, sampled_col)

        sampled_col = tl.where(sampled_col == -1, 0, sampled_col)

        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_col, mask = mask)

    @staticmethod
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr,
              s_mids_ptr, mask, step_size, pseudocount, BLOCK_SIZE):
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        k_n = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_k = tl.max(k_n, axis = 0)

        # Pass 1: cumulate flows within each row.
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for i in range(max_k):
            slot_mask = mask & (i < k_n)
            flow = tl.load(param_flows_ptr + s_pfids + i, mask = slot_mask, other = 0)
            if keep_zero_params:
                param = tl.load(params_ptr + s_pids + i, mask = slot_mask, other = 0)
                k_n_f = k_n.to(tl.float32)
                cum_flow += tl.where(param < 1e-12, 0.0, flow + pseudocount / k_n_f)
            else:
                cum_flow += flow

        cum_flow += pseudocount

        # Pass 2: update parameters (row-wise normalization).
        for i in range(max_k):
            slot_mask = mask & (i < k_n)
            param = tl.load(params_ptr + s_pids + i, mask = slot_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + i, mask = slot_mask, other = 0)
            k_n_f = k_n.to(tl.float32)
            if keep_zero_params:
                new_param = (1.0 - step_size) * param + step_size * (flow + pseudocount / k_n_f) / (cum_flow - pseudocount)
                new_param = tl.where(param < 1e-12, 0.0, new_param)
            else:
                new_param = (1.0 - step_size) * param + step_size * (flow + pseudocount / k_n_f) / cum_flow
            tl.store(params_ptr + s_pids + i, new_param, mask = slot_mask)

    @staticmethod
    def partition_fn(local_offsets, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask,
                     BLOCK_SIZE, TILE_SIZE_K):
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        k_n = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_k = tl.max(k_n, axis = 0)

        partial = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for i in range(max_k):
            slot_mask = mask & (i < k_n)
            val = tl.load(params_ptr + s_pids + i, mask = slot_mask, other = 0)
            partial += val

        return tl.log(partial)
