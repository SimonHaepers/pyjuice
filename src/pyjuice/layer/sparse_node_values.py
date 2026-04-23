from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


LOG_EPS = -23.0258509299  # log(1e-10), matches SparseCategorical.


@dataclass
class SparseNodeValues:
    """
    Jagged CSR container for per-batch (row, value) pairs produced by
    ``SparseProdLayer``. The sparsity pattern (``ptr``, ``indices``,
    ``csc_slots``, ``batch_ids``) is a function of the observed tokens and
    is reused between forward and backward within a single mini-batch.
    ``values`` is recomputed each pass.

    Fields:
      ptr       [B+1] long   — prefix sums of k_v[b]; total_nnz = ptr[-1].
      indices   [total_nnz]  — row id within the owning ns's output block (0..H).
      values    [total_nnz]  — log_emit + Σ log_trans (forward) or gathered flow (backward).
      csc_slots [total_nnz]  — global offset into the SparseCategorical's CSC values
                                array (param_base + csc_slot indexes into input_layer.params).
      batch_ids [total_nnz]  — which batch item each jagged entry belongs to.
      num_rows   H (int)
      batch_size B (int)
    """

    ptr: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    csc_slots: torch.Tensor
    batch_ids: torch.Tensor
    num_rows: int
    batch_size: int

    @property
    def total_nnz(self) -> int:
        return int(self.indices.numel())

    @property
    def device(self) -> torch.device:
        return self.indices.device

    def scatter_to_dense(self, out: torch.Tensor, out_base: int,
                         fill_value: float = LOG_EPS) -> None:
        """Fill ``out[out_base:out_base+H, :] = fill_value`` then overwrite
        the active slots with ``values``."""
        device = out.device
        B = self.batch_size
        H = self.num_rows

        # Fill
        BLOCK_R, BLOCK_B = 32, 64
        grid_fill = (triton.cdiv(H, BLOCK_R), triton.cdiv(B, BLOCK_B))
        _svals_fill_kernel[grid_fill](
            out_ptr=out, out_base=out_base, num_rows=H, batch_size=B,
            fill_value=fill_value, BLOCK_R=BLOCK_R, BLOCK_B=BLOCK_B,
        )

        # Overwrite active slots
        tot = self.total_nnz
        if tot == 0:
            return
        BLOCK = 256
        grid_write = (triton.cdiv(tot, BLOCK),)
        _svals_scatter_write_kernel[grid_write](
            out_ptr=out, out_base=out_base, batch_size=B,
            indices_ptr=self.indices, batch_ids_ptr=self.batch_ids,
            values_ptr=self.values, total_nnz=tot, BLOCK=BLOCK,
        )

    def gather_from_dense(self, src: torch.Tensor, src_base: int) -> "SparseNodeValues":
        """Return a new ``SparseNodeValues`` sharing this object's pattern
        (``ptr``, ``indices``, ``csc_slots``, ``batch_ids``) whose
        ``values[j] = src[src_base + indices[j], batch_ids[j]]``."""
        tot = self.total_nnz
        B = self.batch_size
        if tot == 0:
            return SparseNodeValues(
                ptr=self.ptr, indices=self.indices,
                values=torch.empty(0, dtype=torch.float32, device=self.device),
                csc_slots=self.csc_slots, batch_ids=self.batch_ids,
                num_rows=self.num_rows, batch_size=B,
            )

        new_values = torch.empty(tot, dtype=torch.float32, device=self.device)
        BLOCK = 256
        grid = (triton.cdiv(tot, BLOCK),)
        _svals_gather_kernel[grid](
            src_ptr=src, src_base=src_base, batch_size=B,
            indices_ptr=self.indices, batch_ids_ptr=self.batch_ids,
            values_out_ptr=new_values, total_nnz=tot, BLOCK=BLOCK,
        )
        return SparseNodeValues(
            ptr=self.ptr, indices=self.indices, values=new_values,
            csc_slots=self.csc_slots, batch_ids=self.batch_ids,
            num_rows=self.num_rows, batch_size=B,
        )


# =====================================================================
# Triton kernels (module-level so @triton.jit can find them)
# =====================================================================


@triton.jit
def _svals_fill_kernel(out_ptr, out_base, num_rows, batch_size, fill_value,
                       BLOCK_R: tl.constexpr, BLOCK_B: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = (offs_r < num_rows)[:, None] & (offs_b < batch_size)[None, :]
    addr = (out_base + offs_r)[:, None] * batch_size + offs_b[None, :]
    tl.store(out_ptr + addr, fill_value, mask=mask)


@triton.jit
def _svals_scatter_write_kernel(out_ptr, out_base, batch_size,
                                indices_ptr, batch_ids_ptr, values_ptr, total_nnz,
                                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_j = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs_j < total_nnz
    row = tl.load(indices_ptr + offs_j, mask=mask, other=0)
    b = tl.load(batch_ids_ptr + offs_j, mask=mask, other=0)
    val = tl.load(values_ptr + offs_j, mask=mask, other=0.0)
    addr = (out_base + row) * batch_size + b
    tl.store(out_ptr + addr, val, mask=mask)


@triton.jit
def _svals_gather_kernel(src_ptr, src_base, batch_size,
                         indices_ptr, batch_ids_ptr, values_out_ptr, total_nnz,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs_j = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs_j < total_nnz
    row = tl.load(indices_ptr + offs_j, mask=mask, other=0)
    b = tl.load(batch_ids_ptr + offs_j, mask=mask, other=0)
    addr = (src_base + row) * batch_size + b
    val = tl.load(src_ptr + addr, mask=mask, other=0.0)
    tl.store(values_out_ptr + offs_j, val, mask=mask)
