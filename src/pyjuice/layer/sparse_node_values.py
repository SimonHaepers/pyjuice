from __future__ import annotations

from dataclasses import dataclass

import torch


LOG_EPS = -23.0258509299  # log(1e-10), matches SparseCategorical.


@dataclass
class SparseNodeValues:
    """
    Single-batch sparse (row, value) container for the active CSC column
    observed by a :class:`SparseCategorical` input at one timestep. Built by
    :meth:`SparseCategorical.build_sparse_pattern` from the observed token,
    then filled in-place with per-active-row log-probabilities by
    :class:`SparseProdLayer` (forward) or with gathered flows (backward).

    The container holds a **view** into the dist's ``_csc_indices`` array —
    there is no per-call allocation for ``indices``. Only ``values`` is
    fresh per pass. For the same reason ``csc_slots`` is represented as a
    scalar ``col_start`` + implicit ``arange(total_nnz)`` rather than as a
    tensor.

    Only ``batch_size == 1`` is supported by design (sparse path is
    inference-only); higher-B circuits must use the plain
    :class:`ProdLayer` / :class:`SumLayer` path.

    Fields:
      col_start  int            — offset into ``dist._csc_indices`` /
                                   ``input_layer.params[csc_values_base:]``.
      total_nnz  int            — length of the active column.
      indices    [total_nnz]    — view of ``dist._csc_indices[col_start:col_start+total_nnz]``
                                   (row id within the owning ns's output block, 0..H).
      values     [total_nnz]    — log_emit + Σ log_trans (forward) or gathered flow (backward).
      num_rows   int            — H.
    """

    col_start: int
    total_nnz: int
    indices: torch.Tensor
    values: torch.Tensor
    num_rows: int

    @property
    def device(self) -> torch.device:
        return self.values.device

    def scatter_to_dense(self, out: torch.Tensor, out_base: int,
                         fill_value: float = LOG_EPS) -> None:
        """Fill ``out[out_base:out_base+H, 0] = fill_value`` then overwrite
        active rows with ``values``. Single-batch: writes at column 0 only."""
        H = self.num_rows
        # Dense fill of the H-row slice at batch col 0.
        torch.cuda.nvtx.range_push(f"svals_fill(H={H})")
        out[out_base:out_base + H, 0] = fill_value
        torch.cuda.nvtx.range_pop()

        if self.total_nnz == 0:
            return
        torch.cuda.nvtx.range_push(f"svals_scatter_write(nnz={self.total_nnz})")
        # ``indices`` are already absolute row ids within this ns's H-slice.
        out[out_base + self.indices, 0] = self.values
        torch.cuda.nvtx.range_pop()

    def gather_from_dense(self, src: torch.Tensor, src_base: int) -> "SparseNodeValues":
        """Return a new :class:`SparseNodeValues` sharing this object's
        pattern (``col_start``, ``indices``) whose
        ``values[j] = src[src_base + indices[j], 0]``."""
        if self.total_nnz == 0:
            return SparseNodeValues(
                col_start=self.col_start, total_nnz=0,
                indices=self.indices,
                values=torch.empty(0, dtype=torch.float32, device=self.device),
                num_rows=self.num_rows,
            )

        torch.cuda.nvtx.range_push(f"svals_gather(nnz={self.total_nnz})")
        new_values = src[src_base + self.indices, 0].contiguous()
        torch.cuda.nvtx.range_pop()
        return SparseNodeValues(
            col_start=self.col_start, total_nnz=self.total_nnz,
            indices=self.indices, values=new_values,
            num_rows=self.num_rows,
        )
