"""Triton bitonic top-K for the TopK inference fast path.

Drop-in replacement for ``torch.topk(sl, k=K, dim=0, largest=True, ...)``
on ``[H, B]`` float32 inputs. Two stages:

1. ``_bitonic_topk_chunk_kernel`` — each program sorts a ``CHUNK_SIZE``
   slice of one ``(chunk_idx, batch)`` and emits the top ``c =
   min(CHUNK_SIZE, K)`` entries.
2. ``_bitonic_merge_kernel`` — multi-level grouped merge over the
   per-chunk partials. The last level writes directly into the
   user-supplied ``out_vals`` / ``out_idx`` buffers (no extra copy, no
   int64 -> int32 narrowing).

Both kernels are batched: a second grid dim over ``B`` lets one launch
service the whole ``[H, B]`` input. The merge kernel uses a single
``tl.sort`` over the merged tile per program — the register-level
hypercube ``merge+rebuild`` fast path from the prune-pc port is omitted
because it tripped a codegen edge in Triton 3.2.0 (the
``dev-cuda124`` env). ``tl.sort`` over a ``GROUP_SIZE * K_IN`` tile is
still a register-only sort and well below the cost of ``torch.topk``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# ============================================================================
# Bit-packing helpers
#
# Pack a (float32 value, int32 index) pair into a single int64 so that
# tl.sort sorts both together; descending order on the int64 reproduces
# descending order on the float (with stable index association).
# ============================================================================

INT64_MIN = tl.constexpr(-9223372036854775808)


@triton.jit
def _float_to_sortable(f):
    """float32 -> int32 that preserves order under signed comparison.
    Positive floats keep their bit pattern; negative floats are XORed with
    0x7FFFFFFF to invert the negative range."""
    b = f.to(tl.int32, bitcast=True)
    mask = tl.where(b < 0, 0x7FFFFFFF, 0)
    return b ^ mask


@triton.jit
def _sortable_to_float(s):
    mask = tl.where(s < 0, 0x7FFFFFFF, 0)
    return (s ^ mask).to(tl.float32, bitcast=True)


@triton.jit
def _pack(sortable_val, idx):
    return (sortable_val.to(tl.int64) << 32) | (idx.to(tl.int64) & 0xFFFFFFFF)


@triton.jit
def _unpack_vals(packed):
    return (packed >> 32).to(tl.int32)


@triton.jit
def _unpack_idx(packed):
    return (packed & 0xFFFFFFFF).to(tl.int32)


# ============================================================================
# Kernels
# ============================================================================


@triton.jit
def _bitonic_topk_chunk_kernel(
    values_ptr,            # [H, B] float32 (any strides)
    out_vals_ptr,          # output (intermediate B-major OR final [K,B] H-major)
    out_idx_ptr,
    H,
    in_stride_h,
    in_stride_b,
    out_stride_chunk,      # element stride between chunks
    out_stride_pos,        # element stride within a chunk
    out_stride_b,          # element stride between batches
    c: tl.constexpr,             # candidates per chunk = min(CHUNK_SIZE, K)
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,    # next_power_of_2(CHUNK_SIZE), for tl.sort
):
    """Stage 0: sort one (chunk, batch) slice, emit top-c entries.

    Grid: ``(num_chunks, B)``. Each program owns ``CHUNK_SIZE`` h-values
    of a single batch column.
    """
    pid_chunk = tl.program_id(0)
    pid_b = tl.program_id(1)

    h_start = pid_chunk * CHUNK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    h_global = h_start + offs
    valid = h_global < H

    vals = tl.load(
        values_ptr + pid_b * in_stride_b + h_global * in_stride_h,
        mask=valid, other=float('-inf'),
    )

    sortable = _float_to_sortable(vals)
    packed = _pack(sortable, h_global.to(tl.int32))
    packed = tl.where(
        valid, packed,
        tl.full([BLOCK_SIZE], INT64_MIN, dtype=tl.int64),
    )

    packed = tl.sort(packed, descending=True)

    out_mask = offs < c
    out_sortable = _unpack_vals(packed)
    out_idx = _unpack_idx(packed)
    out_vals = _sortable_to_float(out_sortable)

    out_offset = pid_b * out_stride_b + pid_chunk * out_stride_chunk
    tl.store(out_vals_ptr + out_offset + offs * out_stride_pos,
             out_vals, mask=out_mask)
    tl.store(out_idx_ptr + out_offset + offs * out_stride_pos,
             out_idx, mask=out_mask)


@triton.jit
def _bitonic_merge_kernel(
    partial_vals_ptr,      # [B, num_blocks * K_IN] float32, B-major
    partial_idx_ptr,       # [B, num_blocks * K_IN] int32
    out_vals_ptr,          # output (intermediate B-major OR final [K,B] H-major)
    out_idx_ptr,
    num_blocks,
    in_stride_b,           # row stride in B-major scratch
    out_stride_k,          # element stride along K (output)
    out_stride_b,          # element stride along B (output)
    K_IN: tl.constexpr,
    K_OUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MERGE_SIZE: tl.constexpr,  # next_power_of_2(GROUP_SIZE * K_IN)
):
    """Stage >= 1: merge ``GROUP_SIZE`` adjacent partials per program.

    Loads up to ``GROUP_SIZE * K_IN`` values, ``tl.sort``s them
    descending, and emits the top ``K_OUT``. Last-group masking handles
    ``num_blocks % GROUP_SIZE != 0``.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    in_base = pid_b * in_stride_b + pid * GROUP_SIZE * K_IN
    out_base = pid_b * out_stride_b + pid * K_OUT * out_stride_k

    actual_blocks = tl.minimum(GROUP_SIZE, num_blocks - pid * GROUP_SIZE)
    actual_count = actual_blocks * K_IN

    offs = tl.arange(0, MERGE_SIZE)
    valid = offs < actual_count

    vals = tl.load(partial_vals_ptr + in_base + offs,
                   mask=valid, other=float('-inf'))
    idx = tl.load(partial_idx_ptr + in_base + offs,
                  mask=valid, other=-1)

    sortable = _float_to_sortable(vals)
    packed = _pack(sortable, idx)
    packed = tl.where(valid, packed,
                      tl.full([MERGE_SIZE], INT64_MIN, dtype=tl.int64))
    packed = tl.sort(packed, descending=True)

    out_mask = offs < K_OUT
    out_sortable = _unpack_vals(packed)
    out_idx_v = _unpack_idx(packed)
    out_vals_v = _sortable_to_float(out_sortable)

    tl.store(out_vals_ptr + out_base + offs * out_stride_k,
             out_vals_v, mask=out_mask)
    tl.store(out_idx_ptr + out_base + offs * out_stride_k,
             out_idx_v, mask=out_mask)


# ============================================================================
# Wrapper
# ============================================================================


def _max_stage_capacity(num_chunks: int, c: int, K: int, group_size: int) -> int:
    """Largest (n_partials * k_in) value across all merge stages — sets the
    scratch row length needed."""
    cap = num_chunks * c
    n = num_chunks
    k = c
    while n > 1:
        n = (n + group_size - 1) // group_size
        k = min(K, group_size * k)
        cap = max(cap, n * k)
    return cap


class BitonicScratch:
    """Ping-pong scratch for the multi-level merge.

    Holds two ``[B, capacity]`` buffer pairs (values + indices). Reused
    across calls; resized lazily when the largest required capacity grows.
    """

    __slots__ = ("a_v", "a_i", "b_v", "b_i", "_cap", "_B", "_device")

    def __init__(self) -> None:
        self.a_v: Optional[torch.Tensor] = None
        self.a_i: Optional[torch.Tensor] = None
        self.b_v: Optional[torch.Tensor] = None
        self.b_i: Optional[torch.Tensor] = None
        self._cap: int = 0
        self._B: int = 0
        self._device: Optional[torch.device] = None

    def ensure(self, B: int, capacity: int, device: torch.device) -> None:
        if (self.a_v is not None
                and B <= self._B and capacity <= self._cap
                and self._device == device):
            return
        new_B = max(self._B, B)
        new_cap = max(self._cap, capacity)
        self.a_v = torch.empty((new_B, new_cap), dtype=torch.float32, device=device)
        self.a_i = torch.empty((new_B, new_cap), dtype=torch.int32, device=device)
        self.b_v = torch.empty((new_B, new_cap), dtype=torch.float32, device=device)
        self.b_i = torch.empty((new_B, new_cap), dtype=torch.int32, device=device)
        self._cap = new_cap
        self._B = new_B
        self._device = device


def bitonic_topk(
    values: torch.Tensor,
    K: int,
    *,
    out_vals: torch.Tensor,
    out_idx: torch.Tensor,
    scratch: Optional[BitonicScratch] = None,
    chunk_size: int = 128,
    group_size: int = 2,
) -> None:
    """Batched top-K along ``dim=0`` of a ``[H, B]`` tensor.

    Writes the top-K (values, indices) of each batch column into the
    pre-allocated ``out_vals`` (float32) and ``out_idx`` (int32). Output
    rows are sorted descending: ``out_vals[0, b]`` is the per-column max.

    A ``BitonicScratch`` is held across calls to keep this allocation-free
    in steady state. Pass one in to amortize; otherwise a temporary one
    is created (and discarded) per call.
    """
    assert values.dim() == 2, f"values must be [H, B], got shape {tuple(values.shape)}"
    assert out_vals.shape == (K, values.size(1)), \
        f"out_vals must be [K={K}, B={values.size(1)}], got {tuple(out_vals.shape)}"
    assert out_idx.shape == (K, values.size(1)), \
        f"out_idx must be [K={K}, B={values.size(1)}], got {tuple(out_idx.shape)}"
    assert values.dtype == torch.float32, f"values must be float32, got {values.dtype}"
    assert out_vals.dtype == torch.float32, f"out_vals must be float32, got {out_vals.dtype}"
    assert out_idx.dtype == torch.int32, f"out_idx must be int32, got {out_idx.dtype}"
    assert K >= 1
    assert K <= values.size(0), f"K={K} must be <= H={values.size(0)}"

    H, B = values.shape
    device = values.device

    num_chunks = triton.cdiv(H, chunk_size)
    c = min(chunk_size, K)
    BLOCK_SIZE = triton.next_power_of_2(chunk_size)

    in_stride_h, in_stride_b = values.stride()

    if num_chunks == 1:
        # Single chunk: the chunk kernel's sorted output is the answer.
        # Skip merge entirely and write straight into the user buffer.
        assert c == K, f"single-chunk path requires c == K; got c={c}, K={K}"
        _bitonic_topk_chunk_kernel[(1, B)](
            values, out_vals, out_idx,
            H, in_stride_h, in_stride_b,
            out_vals.stride(0),     # out_stride_chunk (only one chunk; unused)
            out_vals.stride(0),     # out_stride_pos = K-axis stride
            out_vals.stride(1),     # out_stride_b   = B-axis stride
            c=c, CHUNK_SIZE=chunk_size, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=min(max(BLOCK_SIZE // 64, 1), 16),
        )
        return

    capacity = _max_stage_capacity(num_chunks, c, K, group_size)
    if scratch is None:
        scratch = BitonicScratch()
    scratch.ensure(B=B, capacity=capacity, device=device)
    scratch_stride_b = scratch._cap

    # Stage 0: chunk sort -> scratch.a (B-major: [B, num_chunks * c])
    _bitonic_topk_chunk_kernel[(num_chunks, B)](
        values, scratch.a_v, scratch.a_i,
        H, in_stride_h, in_stride_b,
        c,                       # out_stride_chunk = c (B-major)
        1,                       # out_stride_pos   = 1
        scratch_stride_b,        # out_stride_b     = scratch row stride
        c=c, CHUNK_SIZE=chunk_size, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(max(BLOCK_SIZE // 64, 1), 16),
    )

    # Multi-level merge. The last stage writes into out_vals / out_idx.
    n_partials = num_chunks
    k_in = c
    ping_v, ping_i = scratch.a_v, scratch.a_i
    pong_v, pong_i = scratch.b_v, scratch.b_i

    while True:
        n_groups = (n_partials + group_size - 1) // group_size
        k_out = min(K, group_size * k_in)
        is_last = (n_groups == 1)

        if is_last:
            assert k_out == K, f"final stage k_out={k_out} != K={K}"
            out_v = out_vals
            out_i = out_idx
            out_sk = out_vals.stride(0)
            out_sb = out_vals.stride(1)
        else:
            out_v = pong_v
            out_i = pong_i
            out_sk = 1
            out_sb = scratch_stride_b

        MERGE_SIZE = triton.next_power_of_2(group_size * k_in)

        _bitonic_merge_kernel[(n_groups, B)](
            ping_v, ping_i,
            out_v, out_i,
            n_partials,
            scratch_stride_b,
            out_sk, out_sb,
            K_IN=k_in,
            K_OUT=k_out,
            GROUP_SIZE=group_size,
            MERGE_SIZE=MERGE_SIZE,
            num_warps=min(max(MERGE_SIZE // 64, 1), 16),
        )

        if is_last:
            break

        ping_v, pong_v = pong_v, ping_v
        ping_i, pong_i = pong_i, ping_i
        n_partials = n_groups
        k_in = k_out
