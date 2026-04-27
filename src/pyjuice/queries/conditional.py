from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Union, Callable, Optional, Sequence
from functools import partial

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit
from pyjuice.layer import DenseCategoricalInputLayer
from pyjuice.nodes.methods import get_subsumed_scopes
from pyjuice.utils import BitSet
from pyjuice.utils.kernel_launcher import FastJITFunction
from .base import query

# `juice.distributions.ExternProductCategorical`
from pyjuice.nodes.distributions.external_categorical import ExternProductCategorical
from pyjuice.nodes.distributions.external_categorical import _condition_apply_ll_kernel, _prep_args_apply_ll_kernel
from pyjuice.nodes.distributions.external_categorical import _condition_apply_ll_w_mask_kernel, _prep_args_apply_ll_w_mask_kernel


## Categorical layer ##

@triton.jit
def _soft_evi_categorical_fw_kernel(data_ptr, node_mars_ptr, params_ptr, vids_ptr, psids_ptr, node_nchs_ptr, local_ids,
                                    sid: tl.constexpr, num_nodes: tl.constexpr, num_cats: tl.constexpr, 
                                    batch_size: tl.constexpr, partial: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_nodes * batch_size

    # Get node ID and category ID
    ns_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    # Get number of children (categories)
    node_nch = tl.load(node_nchs_ptr + ns_offsets, mask = mask, other = 0)

    # Get variable ID
    vid = tl.load(vids_ptr + ns_offsets, mask = mask, other = 0)

    # Get param start ID
    psid = tl.load(psids_ptr + ns_offsets, mask = mask, other = 0)

    # Compute soft evidence per category
    node_vals = tl.zeros((BLOCK_SIZE,), tl.float32)
    for cat_id in range(num_cats):

        cmask = mask & (cat_id < node_nch)

        # Get data (soft evidence)
        data_offsets = vid * (num_cats * batch_size) + cat_id * batch_size + batch_offsets
        d_soft_evi = tl.load(data_ptr + data_offsets, mask = cmask, other = 0)

        # Get param
        param = tl.load(params_ptr + psid + cat_id, mask = cmask, other = 0)

        # Compute current likelihood and accumulate
        node_vals += d_soft_evi * param

    # Write back
    if not partial:
        tl.store(node_mars_ptr + offsets + (sid * batch_size), tl.log(node_vals), mask = mask)
    else:
        global_nid = tl.load(local_ids + ns_offsets, mask = mask, other = 0) + sid
        tl.store(node_mars_ptr + global_nid * batch_size + batch_offsets, tl.log(node_vals), mask = mask)


def _categorical_forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
                         missing_mask: Optional[torch.Tensor] = None, **kwargs):

    batch_size, num_vars = inputs.size(0), inputs.size(1)

    if inputs.dim() == 2:
        # Hard evidence
        assert inputs.dtype == torch.long

        inputs = inputs.permute(1, 0).contiguous()

        layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask)

    elif inputs.dim() == 3:
        # Soft evidence
        assert inputs.dtype == torch.float32 and inputs.min() >= 0.0 and inputs.max() <= 1.0

        if missing_mask is not None:
            if missing_mask.dim() == 1:
                inputs[:,missing_mask,:] = 1.0
            else:
                assert missing_mask.dim() == 2
                inputs = inputs.flatten(0, 1)
                inputs[missing_mask.flatten(),:] = 1.0
                inputs = inputs.reshape(batch_size, num_vars, -1)

        inputs = inputs.permute(1, 2, 0) # [num_vars, num_cats, B]
        num_cats = inputs.size(1)

        sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]
        num_nodes = eid - sid

        node_nchs = layer.metadata[layer.s_mids]

        grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)

        _soft_evi_categorical_fw_kernel[grid](
            inputs.reshape(-1).contiguous(), node_mars, layer.params, layer.vids.reshape(-1), layer.s_pids, node_nchs,
            None, sid, num_nodes, num_cats, batch_size, partial = False, BLOCK_SIZE = 512
        )

        node_mars[sid:eid,:] = node_mars[sid:eid,:].clip(max = 0.0)

    else:
        raise NotImplementedError("Unknown method to compute the forward pass for `Categorical`.")

    return None


def _external_categorical_forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
                                  missing_mask: Optional[torch.Tensor] = None, **kwargs):
    
    inputs = inputs.permute(1, 0).contiguous()

    layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask, **kwargs)


@triton.jit
def _soft_evi_discrete_logistic_fw_kernel(data_ptr, node_mars_ptr, params_ptr, vids_ptr, psids_ptr, s_mids_ptr, metadata_ptr, 
                                          local_ids, sid: tl.constexpr, num_nodes: tl.constexpr, num_cats: tl.constexpr, 
                                          batch_size: tl.constexpr, partial: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_nodes * batch_size

    # Get node ID and category ID
    ns_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    # Get variable ID
    vid = tl.load(vids_ptr + ns_offsets, mask = mask, other = 0)

    # Get params
    psid = tl.load(psids_ptr + ns_offsets, mask = mask, other = 0)
    mu = tl.load(params_ptr + psid, mask = mask, other = 0)
    s = tl.load(params_ptr + psid + 1, mask = mask, other = 0)

    # Get metadata
    s_mids = tl.load(s_mids_ptr + ns_offsets, mask = mask, other = 0)
    range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
    range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
    node_nch = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

    # Compute soft evidence per category
    node_vals = tl.zeros((BLOCK_SIZE,), tl.float32)
    for cat_id in range(num_cats):

        cmask = mask & (cat_id < node_nch)

        # Get data (soft evidence)
        data_offsets = vid * (num_cats * batch_size) + cat_id * batch_size + batch_offsets
        d_soft_evi = tl.load(data_ptr + data_offsets, mask = cmask, other = 0)

        # Get param
        interval = (range_high - range_low) / node_nch
        vlow = cat_id * interval + range_low
        vhigh = vlow + interval

        cdfhigh = tl.where(cat_id == node_nch - 1, 1.0, 1.0 / (1.0 + tl.exp((mu - vhigh) / s)))
        cdflow = tl.where(cat_id == 0, 0.0, 1.0 / (1.0 + tl.exp((mu - vlow) / s)))

        param = tl.maximum(cdfhigh - cdflow, 0.0)

        # Compute current likelihood and accumulate
        node_vals += d_soft_evi * param

    # Write back
    if not partial:
        tl.store(node_mars_ptr + offsets + (sid * batch_size), tl.log(node_vals), mask = mask) # debug
    else:
        global_nid = tl.load(local_ids + ns_offsets, mask = mask, other = 0) + sid
        tl.store(node_mars_ptr + global_nid * batch_size + batch_offsets, tl.log(node_vals), mask = mask)


def _discrete_logistic_forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
                               missing_mask: Optional[torch.Tensor] = None, **kwargs):
    
    batch_size, num_vars = inputs.size(0), inputs.size(1)

    if inputs.dim() == 2:
        # Hard evidence
        if layer.nodes[0].dist.input_type == "discrete":
            assert inputs.dtype == torch.long, "Input dtype should be `torch.float32`."
        else: 
            assert layer.nodes[0].dist.input_type == "continuous"
            assert inputs.dtype == torch.float32, "Input dtype should be `torch.float32`."

        inputs = inputs.permute(1, 0).contiguous()

        layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask)

    elif inputs.dim() == 3:
        # Soft evidence
        assert inputs.dtype == torch.float32 and inputs.min() >= 0.0 and inputs.max() <= 1.0

        if missing_mask is not None:
            if missing_mask.dim() == 1:
                inputs[:,missing_mask,:] = 1.0
            else:
                assert missing_mask.dim() == 2
                inputs = inputs.flatten(0, 1)
                inputs[missing_mask.flatten(),:] = 1.0
                inputs = inputs.reshape(batch_size, num_vars, -1)

        inputs = inputs.permute(1, 2, 0) # [num_vars, num_cats, B]
        num_cats = inputs.size(1)

        sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]
        num_nodes = eid - sid

        grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)

        _soft_evi_discrete_logistic_fw_kernel[grid](
            inputs.reshape(-1).contiguous(), node_mars, layer.params, layer.vids.reshape(-1), layer.s_pids, 
            layer.s_mids, layer.metadata, None, sid, num_nodes, num_cats, batch_size, 
            partial = False, BLOCK_SIZE = 512
        )

        node_mars[sid:eid,:] = node_mars[sid:eid,:].clip(max = 0.0)

    else:
        raise NotImplementedError("Unknown method to compute the forward pass for `DiscreteLogistic`.")

    return None


@triton.jit
def _categorical_backward_kernel(cat_probs_ptr, node_flows_ptr, local_ids_ptr, rev_vars_mapping_ptr, vids_ptr, psids_ptr, 
                                 node_nchs_ptr, params_ptr, sid, eid, num_target_nodes, batch_size: tl.constexpr, 
                                 num_cats: tl.constexpr, partial_eval: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < num_target_nodes * batch_size)

    # Get node offsets and batch offsets
    local_offsets = (offsets // batch_size)
    if partial_eval == 1: 
        local_node_offsets = tl.load(local_ids_ptr + local_offsets, mask = mask, other = 0)
    else:
        local_node_offsets = local_offsets
    batch_offsets = (offsets % batch_size)

    global_node_offsets = local_node_offsets + sid

    # Get variable ID
    origin_vid = tl.load(vids_ptr + local_node_offsets, mask = mask, other = 0)
    vid = tl.load(rev_vars_mapping_ptr + origin_vid, mask = mask, other = 0)

    # Get number of children per node
    node_nch = tl.load(node_nchs_ptr + local_node_offsets, mask = mask, other = 0)

    # Get param start ID
    psid = tl.load(psids_ptr + local_node_offsets, mask = mask, other = 0)

    # Get flow
    nflow_offsets = global_node_offsets * batch_size + batch_offsets
    nflow = tl.load(node_flows_ptr + nflow_offsets, mask = mask, other = 0)

    # Compute edge flows and accumulate
    for cat_id in range(num_cats):
        cmask = mask & (cat_id < node_nch)

        param = tl.load(params_ptr + psid + cat_id, mask = cmask, other = 0)
        eflow = nflow * param

        p_offsets = vid * num_cats * batch_size + cat_id * batch_size + batch_offsets
        tl.atomic_add(cat_probs_ptr + p_offsets, eflow, mask = cmask)


def _categorical_backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
                          params: Optional[torch.Tensor] = None, **kwargs):

    if params is None:
        params = layer.params

    # Dense fast-path: the layer class knows its layout is `[V, K, C]` and
    # replaces the atomic-add scatter kernel with a single bmm/matmul.
    if isinstance(layer, DenseCategoricalInputLayer):
        return layer.dense_conditional_backward(
            node_flows = node_flows, params = params,
            target_vars = kwargs.get("target_vars", None),
        )

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]

    num_nodes = eid - sid
    num_vars = layer.vids.max().item() + 1
    num_cats = int(layer.metadata[layer.s_mids].max().item())
    batch_size = node_flows.size(1)

    target_vars_arg = kwargs.get("target_vars", None)

    if target_vars_arg is not None:
        target_vars = target_vars_arg

        rev_vars_mapping = torch.zeros([num_vars], dtype = torch.long)
        for i, var in enumerate(target_vars):
            rev_vars_mapping[var] = i
        rev_vars_mapping = rev_vars_mapping.to(node_flows.device)
    else:
        target_vars = [var for var in range(num_vars)]

        rev_vars_mapping = torch.arange(0, num_vars, device = node_flows.device)

    num_target_vars = len(target_vars)

    cat_probs = torch.zeros([num_target_vars * num_cats * batch_size], dtype = torch.float32, device = node_flows.device)

    if len(target_vars) < num_vars:
        local_ids = layer.enable_partial_evaluation(bk_scopes = target_vars, return_ids = True).to(node_flows.device)
        num_target_nodes = local_ids.size(0)
        partial_eval = 1
    else:
        local_ids = None
        num_target_nodes = eid - sid
        partial_eval = 0

    node_nchs = layer.metadata[layer.s_mids]

    grid = lambda meta: (triton.cdiv(num_target_nodes * batch_size, meta['BLOCK_SIZE']),)

    _categorical_backward_kernel[grid](
        cat_probs, node_flows, local_ids, rev_vars_mapping, layer.vids, layer.s_pids, node_nchs, layer.params,
        sid, eid, num_target_nodes, batch_size, num_cats, partial_eval = partial_eval, BLOCK_SIZE = 512
    )

    cat_probs = cat_probs.reshape(num_target_vars, num_cats, batch_size)

    cat_probs /= (cat_probs.sum(dim = 1, keepdim = True) + 1e-12)
    cat_probs = cat_probs.permute(2, 0, 1)

    return cat_probs


def _external_categorical_backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
                                   params: Optional[torch.Tensor] = None, **kwargs):
    
    assert "target_vars" not in kwargs or kwargs["target_vars"] is None

    # num_vars = layer.vids.max().item() + 1
    num_cats = int(layer.metadata[layer.s_mids].max().item())
    batch_size = node_flows.size(1)

    num_vars = layer.var_idmapping.size(0)

    cat_probs = torch.zeros([batch_size, num_vars, num_cats], dtype = torch.float32, device = node_flows.device)
    kwargs["external_categorical_logps_grad"] = cat_probs

    kwargs["no_param_update"] = True
    kwargs["extern_product_categorical_mode"] = "normalizing_constant"

    layer.backward(inputs, node_flows, node_mars, **kwargs)

    cat_probs /= (cat_probs.sum(dim = 2, keepdim = True) + 1e-12)

    return cat_probs

@triton.jit
def _discrete_logistic_backward_kernel(cat_probs_ptr, node_flows_ptr, local_ids_ptr, rev_vars_mapping_ptr, vids_ptr, psids_ptr, 
                                       msids_ptr, metadata_ptr, params_ptr, sid, eid, num_target_nodes, 
                                       batch_size: tl.constexpr, num_cats: tl.constexpr, partial_eval: tl.constexpr, 
                                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < num_target_nodes * batch_size)

    # Get node offsets and batch offsets
    local_offsets = (offsets // batch_size)
    if partial_eval == 1: 
        local_node_offsets = tl.load(local_ids_ptr + local_offsets, mask = mask, other = 0)
    else:
        local_node_offsets = local_offsets
    batch_offsets = (offsets % batch_size)

    global_node_offsets = local_node_offsets + sid

    # Get variable ID
    origin_vid = tl.load(vids_ptr + local_node_offsets, mask = mask, other = 0)
    vid = tl.load(rev_vars_mapping_ptr + origin_vid, mask = mask, other = 0)

    # Get params
    psid = tl.load(psids_ptr + local_offsets, mask = mask, other = 0)
    mu = tl.load(params_ptr + psid, mask = mask, other = 0)
    s = tl.load(params_ptr + psid + 1, mask = mask, other = 0)

    # Get metadata
    s_mids = tl.load(msids_ptr + local_offsets, mask = mask, other = 0)
    range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
    range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
    node_nch = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

    # Get flow
    nflow_offsets = global_node_offsets * batch_size + batch_offsets
    nflow = tl.load(node_flows_ptr + nflow_offsets, mask = mask, other = 0)

    # Compute edge flows and accumulate
    for cat_id in range(num_cats):
        cmask = mask & (cat_id < node_nch)

        # Get param
        interval = (range_high - range_low) / node_nch
        vlow = cat_id * interval + range_low
        vhigh = vlow + interval

        cdfhigh = tl.where(cat_id == node_nch - 1, 1.0, 1.0 / (1.0 + tl.exp((mu - vhigh) / s)))
        cdflow = tl.where(cat_id == 0, 0.0, 1.0 / (1.0 + tl.exp((mu - vlow) / s)))

        param = tl.maximum(cdfhigh - cdflow, 0.0)
        eflow = nflow * param

        p_offsets = vid * num_cats * batch_size + cat_id * batch_size + batch_offsets
        tl.atomic_add(cat_probs_ptr + p_offsets, eflow, mask = cmask)


def _discrete_logistic_backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
                                params: Optional[torch.Tensor] = None, **kwargs):
    
    if params is None:
        params = layer.params

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]

    num_nodes = eid - sid
    num_vars = layer.vids.max().item() + 1
    num_cats = int(layer.metadata[layer.s_mids + 2].max().item())
    batch_size = node_flows.size(1)

    if "target_vars" in kwargs and kwargs["target_vars"] is not None:
        target_vars = kwargs["target_vars"]

        rev_vars_mapping = torch.zeros([num_vars], dtype = torch.long)
        for i, var in enumerate(target_vars):
            rev_vars_mapping[var] = i
        rev_vars_mapping = rev_vars_mapping.to(node_flows.device)
    else:
        target_vars = [var for var in range(num_vars)]

        rev_vars_mapping = torch.arange(0, num_vars, device = node_flows.device)

    num_target_vars = len(target_vars)

    cat_probs = torch.zeros([num_target_vars * num_cats * batch_size], dtype = torch.float32, device = node_flows.device)

    if len(target_vars) < num_vars:
        local_ids = layer.enable_partial_evaluation(bk_scopes = target_vars, return_ids = True).to(node_flows.device)
        num_target_nodes = local_ids.size(0)
        partial_eval = 1
    else:
        local_ids = None
        num_target_nodes = eid - sid
        partial_eval = 0

    grid = lambda meta: (triton.cdiv(num_target_nodes * batch_size, meta['BLOCK_SIZE']),)

    _discrete_logistic_backward_kernel[grid](
        cat_probs, node_flows, local_ids, rev_vars_mapping, layer.vids, layer.s_pids, 
        layer.s_mids, layer.metadata, layer.params,
        sid, eid, num_target_nodes, batch_size, num_cats, partial_eval = partial_eval, BLOCK_SIZE = 512
    )

    cat_probs = cat_probs.reshape(num_target_vars, num_cats, batch_size)

    cat_probs /= (cat_probs.sum(dim = 1, keepdim = True) + 1e-12)
    cat_probs = cat_probs.permute(2, 0, 1)

    return cat_probs


## SparseCategorical layer ##


def _sparse_categorical_forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
                                 missing_mask: Optional[torch.Tensor] = None, **kwargs):
    """Delegate to ``InputLayer.forward`` (which dispatches to
    :meth:`SparseCategorical.custom_forward`) for hard evidence. Soft evidence
    is not yet supported — the CSC kernel path would need a row-wise mixture
    scatter that doesn't exist yet; raise explicitly so callers can either
    pre-materialise soft evidence elsewhere or extend this function."""
    if inputs.dim() == 2:
        assert inputs.dtype == torch.long
        inputs = inputs.permute(1, 0).contiguous()
        layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask)
    else:
        raise NotImplementedError(
            "SparseCategorical conditional forward currently supports only hard evidence "
            "(`inputs.dim() == 2`). Soft evidence requires a row-wise CSC mixture kernel "
            "that hasn't been implemented."
        )

    return None


@triton.jit(
    do_not_specialize=["vid_out", "node_offset", "param_base",
                        "batch_size", "num_cats"],
    do_not_specialize_on_alignment=["cat_probs_ptr", "node_flows_ptr",
                                     "params_ptr", "csc_indptr_ptr",
                                     "csc_indices_ptr"],
)
def _sparse_categorical_cond_backward_kernel(
    cat_probs_ptr, node_flows_ptr, params_ptr,
    csc_indptr_ptr, csc_indices_ptr,
    vid_out, node_offset, param_base,
    batch_size, num_cats,
    BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """Per-column CSC gather: for CSC column ``pid_col``, tile its active rows
    into ``BLOCK_K`` chunks; for each ``(row, col)`` slot read
    ``node_flows[node_offset + row, b] * params[param_base + slot]`` and
    accumulate into ``cat_probs[vid_out, col, b]``. Different ``pid_k`` tiles
    write to the same output cell, so we use ``atomic_add``."""
    pid_col = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < batch_size

    col_start = tl.load(csc_indptr_ptr + pid_col)
    col_end = tl.load(csc_indptr_ptr + pid_col + 1)
    k_col = col_end - col_start
    slot_mask = offs_k < k_col

    slot_idx = col_start + offs_k
    row_id = tl.load(csc_indices_ptr + slot_idx, mask = slot_mask, other = 0)
    param = tl.load(params_ptr + param_base + slot_idx, mask = slot_mask, other = 0.0)

    flow_offs = (node_offset + row_id)[:, None] * batch_size + offs_b[None, :]
    full_mask = slot_mask[:, None] & mask_b[None, :]
    flow = tl.load(node_flows_ptr + flow_offs, mask = full_mask, other = 0.0)

    contrib = flow * param[:, None]
    contrib_sum = tl.sum(contrib, axis = 0)  # [BLOCK_B]

    cat_offs = (vid_out * num_cats + pid_col) * batch_size + offs_b
    tl.atomic_add(cat_probs_ptr + cat_offs, contrib_sum, mask = mask_b)


@triton.jit(
    do_not_specialize=["vid_out", "param_base", "num_cats"],
    do_not_specialize_on_alignment=["cat_probs_ptr", "sv_indices_ptr",
                                     "sv_values_ptr", "params_ptr",
                                     "csr_indptr_ptr", "csr_cols_ptr",
                                     "csr_to_csc_ptr"],
)
def _sparse_categorical_cond_backward_sparse_kernel(
    cat_probs_ptr, sv_indices_ptr, sv_values_ptr, params_ptr,
    csr_indptr_ptr, csr_cols_ptr, csr_to_csc_ptr,
    vid_out, param_base, num_cats,
    BLOCK_R: tl.constexpr,
):
    """Sparse-native, B=1 specialization of
    :func:`_sparse_categorical_cond_backward_kernel`.

    Iterates over the active rows that actually carry non-zero flow (the
    observed CSC column's active rows — the count is set by the launch
    grid's first dim, not a kernel arg, so the same compiled binary serves
    every input token regardless of ``total_nnz``), then for each row walks
    its CSR slots — for each ``(row, col)`` membership do
    ``atomic_add(cat_probs[col], flow * params[csc_slot])``. ``node_flows``
    is never touched; flow comes straight from the
    :class:`SparseNodeValues` packed values array.

    ``sv_indices_ptr`` is a sliced view of ``dist._csc_indices`` whose
    16-byte alignment depends on ``col_start`` parity, and the grid's first
    dim varies with the observed token. The two ``do_not_specialize`` /
    ``do_not_specialize_on_alignment`` lists above keep that data-dependent
    state out of Triton's specialization key, so this kernel is JIT-compiled
    exactly once across token sequences.

    Grid: ``(K_active, ceil(max_nnz_per_row / BLOCK_R))``. Per program covers
    one (active row, row-slot tile). The batch axis is collapsed because the
    sparse path is only entered with ``batch_size == 1``; ``cat_probs`` is
    indexed as ``[vid_out, col]`` (the trailing B=1 axis is a no-op stride).
    """
    pid_j = tl.program_id(0)
    pid_r = tl.program_id(1)

    # Active row + flow for this program. ``sv.values`` is [K_active], no
    # batch axis (B=1 layout).
    row = tl.load(sv_indices_ptr + pid_j)
    flow_scalar = tl.load(sv_values_ptr + pid_j)

    # Walk this row's CSR slot range. Cast to int32 — register pressure on
    # the [BLOCK_R] index tiles is the main cost driver here.
    row_start = tl.load(csr_indptr_ptr + row).to(tl.int32)
    row_end = tl.load(csr_indptr_ptr + row + 1).to(tl.int32)
    row_nnz = row_end - row_start

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    slot_mask = offs_r < row_nnz
    csr_slot = row_start + offs_r

    cols = tl.load(csr_cols_ptr + csr_slot, mask = slot_mask, other = 0).to(tl.int32)
    csc_slots = tl.load(csr_to_csc_ptr + csr_slot, mask = slot_mask, other = 0)
    params_v = tl.load(params_ptr + param_base + csc_slots,
                       mask = slot_mask, other = 0.0)

    contrib = flow_scalar * params_v   # [BLOCK_R]

    # cat_probs is laid out as [num_target_vars, num_cats] (the B=1 axis
    # collapses). Multiple programs may hit the same col, so atomic_add.
    addr = vid_out * num_cats + cols
    tl.atomic_add(cat_probs_ptr + addr, contrib, mask = slot_mask)


def _sparse_categorical_backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor,
                                  node_mars: torch.Tensor,
                                  params: Optional[torch.Tensor] = None, **kwargs):
    """CSC-native version of :func:`_categorical_backward`. For each ``InputNodes``
    in ``layer.nodes`` with variable ``vid`` that appears in ``target_vars``,
    iterate over the CSC sparsity pattern and accumulate
    ``cat_probs[vid, c, b] += node_flows[row, b] * params[(row, c) slot]``.

    Rows/cols outside the CSC pattern contribute ``~1e-10 * flow`` in the
    reference ``Categorical`` forward; we omit that constant here — it is
    negligible (and washed out by the final per-(vid, b) renormalisation) —
    which makes a dense-pattern sparse build compare exactly against the
    reference up to atomic-add rounding.
    """
    if params is None:
        params = layer.params

    num_vars = int(layer.vids.max().item()) + 1
    num_cats = int(layer.nodes[0].dist.num_cats)
    batch_size = node_flows.size(1)

    target_vars_arg = kwargs.get("target_vars", None)
    if target_vars_arg is not None:
        target_vars = list(target_vars_arg)
        rev_vars_mapping = torch.full((num_vars,), -1, dtype = torch.long)
        for i, var in enumerate(target_vars):
            rev_vars_mapping[var] = i
    else:
        target_vars = [var for var in range(num_vars)]
        rev_vars_mapping = torch.arange(0, num_vars, dtype = torch.long)

    num_target_vars = len(target_vars)

    cat_probs = torch.zeros(
        [num_target_vars * num_cats * batch_size],
        dtype = torch.float32, device = node_flows.device,
    )

    for ns in layer.nodes:
        var_id = ns.scope.to_list()[0]
        vid_out = int(rev_vars_mapping[var_id].item())
        if vid_out < 0:
            continue

        dist = ns.dist
        if dist._nnz == 0:
            continue
        assert dist.num_cats == num_cats, (
            "SparseCategorical conditional backward assumes uniform num_cats across ns."
        )

        node_offset = ns._output_ind_range[0]
        param_base = ns._param_range[0]

        # Sparse fast path: if this ns is consumed by a SparseProdLayer
        # AND that layer cached an ``sv_flow`` from the most recent backward,
        # iterate over the K_active rows directly via the CSR view —
        # ``node_flows`` is never touched. The dense path below is the
        # fallback for plain compiles or partial backwards.
        owner = getattr(ns, "_sparse_flow_owner", None)
        sv_flow = None
        if owner is not None:
            owner_layer, owner_ns_idx = owner
            sv_flow = owner_layer._sparse_flows[owner_ns_idx]

        if sv_flow is not None and sv_flow.total_nnz > 0:
            assert batch_size == 1, (
                "SparseCategorical sparse-flow conditional backward path "
                "is only valid for batch_size == 1 (sv.values is [K_active])."
            )
            assert dist._csr_indptr is not None, (
                "SparseCategorical: CSR side info not built (set_meta_parameters "
                "should have populated _csr_indptr). Re-run set_meta_parameters."
            )
            BLOCK_R = max(triton.next_power_of_2(dist._max_nnz_per_row), 4)
            grid = (
                sv_flow.total_nnz,
                triton.cdiv(dist._max_nnz_per_row, BLOCK_R),
            )
            _sparse_categorical_cond_backward_sparse_kernel[grid](
                cat_probs_ptr = cat_probs,
                sv_indices_ptr = sv_flow.indices,
                sv_values_ptr = sv_flow.values,
                params_ptr = params,
                csr_indptr_ptr = dist._csr_indptr,
                csr_cols_ptr = dist._csr_cols,
                csr_to_csc_ptr = dist._csr_to_csc,
                vid_out = vid_out,
                param_base = param_base,
                num_cats = num_cats,
                BLOCK_R = BLOCK_R,
            )
            continue

        BLOCK_K = max(triton.next_power_of_2(dist._max_nnz_per_col), 4)
        BLOCK_B = min(64, max(1, triton.next_power_of_2(batch_size)))

        grid = (
            dist.num_cats,
            triton.cdiv(dist._max_nnz_per_col, BLOCK_K),
            triton.cdiv(batch_size, BLOCK_B),
        )

        _sparse_categorical_cond_backward_kernel[grid](
            cat_probs_ptr = cat_probs,
            node_flows_ptr = node_flows,
            params_ptr = params,
            csc_indptr_ptr = dist._csc_indptr,
            csc_indices_ptr = dist._csc_indices,
            vid_out = vid_out,
            node_offset = node_offset,
            param_base = param_base,
            batch_size = batch_size,
            num_cats = num_cats,
            BLOCK_K = BLOCK_K,
            BLOCK_B = BLOCK_B,
        )

    cat_probs = cat_probs.reshape(num_target_vars, num_cats, batch_size)
    cat_probs /= (cat_probs.sum(dim = 1, keepdim = True) + 1e-12)
    cat_probs = cat_probs.permute(2, 0, 1)
    return cat_probs


def _conditional_fw_input_fn(layer, inputs, node_mars, **kwargs):
    if layer.dist_signature in ("Categorical", "DenseCategorical"):
        _categorical_forward(layer, inputs, node_mars, **kwargs)

    elif layer.dist_signature == "DiscreteLogistic":
        _discrete_logistic_forward(layer, inputs, node_mars, **kwargs)

    elif layer.dist_signature == "ExternProductCategorical":
        _external_categorical_forward(layer, inputs, node_mars, **kwargs)

    elif layer.dist_signature == "SparseCategorical":
        _sparse_categorical_forward(layer, inputs, node_mars, **kwargs)

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)} for the forward pass. Please implement and provide your own `fw_input_fn`.")


def _conditional_bk_input_fn(layer, inputs, node_flows, node_mars, outputs = None, **kwargs):
    if layer.dist_signature in ("Categorical", "DenseCategorical"):
        outputs.append(
            _categorical_backward(layer, inputs, node_flows, node_mars, layer.params, **kwargs)
        )

    elif layer.dist_signature == "DiscreteLogistic":
        outputs.append(
            _discrete_logistic_backward(layer, inputs, node_flows, node_mars, layer.params, **kwargs)
        )

    elif layer.dist_signature == "ExternProductCategorical":
        outputs.append(
            _external_categorical_backward(layer, inputs, node_flows, node_mars, layer.params, **kwargs)
        )

    elif layer.dist_signature == "SparseCategorical":
        outputs.append(
            _sparse_categorical_backward(layer, inputs, node_flows, node_mars, layer.params, **kwargs)
        )

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)} for the backward pass. Please implement and provide your own `bk_input_fn`.")


## Main API ##


def conditional(pc: TensorCircuit, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                target_vars: Optional[Sequence[int]] = None,
                fw_input_fn: Optional[Union[str,Callable]] = None, 
                bk_input_fn: Optional[Union[str,Callable]] = None, **kwargs):
    """
    Compute the conditional probability given hard or soft evidence, i.e., P(o|e).

    :param pc: the input PC
    :type pc: TensorCircuit

    :param data: data of size [B, num_vars] (hard evidence) or a custom shape paired with `fw_input_fn`
    :type data: torch.Tensor

    :param missing_mask: a boolean mask indicating marginalized variables; the size can be [num_vars] or [B, num_vars]
    :type missing_mask: torch.Tensor

    :param fw_input_fn: an optional custom function for the forward pass of input layers
    :type fw_input_fn: Optional[Union[str,Callable]]

    :param bk_input_fn: an optional custom function for the backward pass of input layers
    :type bk_input_fn: Optional[Union[str,Callable]]
    """

    outputs = []

    _wrapped_bk_input_fn = partial(_conditional_bk_input_fn, outputs = outputs)

    kwargs["target_vars"] = target_vars

    query(pc, inputs = data, run_backward = True, 
          fw_input_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn, 
          bk_input_fn = _wrapped_bk_input_fn if bk_input_fn is None else bk_input_fn, 
          missing_mask = missing_mask, **kwargs)

    return outputs[0]
