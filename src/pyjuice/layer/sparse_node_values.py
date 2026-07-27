from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


LOG_EPS = -23.0258509299  # log(1e-10), matches SparseCategorical.


@dataclass
class SparseNodeValues:
    """
    Sparse (row, value) container for the active CSC column(s) observed by a
    :class:`SparseCategorical` input at one timestep. Built by
    :meth:`SparseCategorical.build_sparse_pattern` from the observed token(s),
    then filled in-place with per-active-row log-probabilities by
    :class:`SparseProdLayer` (forward) or with gathered flows (backward).

    Two layouts share this class, selected by ``batch_size``:

    **B=1** (batched fields ``None``): ``indices`` is a **view** into the
    dist's ``_csc_indices[col_start:col_start+total_nnz]`` — no per-call
    allocation — and ``values`` is a 1-D ``[total_nnz]`` buffer. ``csc_slots``
    is represented as the scalar ``col_start`` + implicit
    ``arange(total_nnz)`` rather than as a tensor.

    **B>1**: each sample observes its own token, hence its own column.
    ``indices`` is the FULL ``dist._csc_indices`` tensor (sample ``b``'s
    ``j``-th active row is ``indices[col_starts[b] + j]``), ``values`` is a
    2-D ``[B, K_stride]`` workspace view whose valid region per sample is
    ``values[b, :nnz_list[b]]`` (``K_stride = dist._max_nnz_per_col``), and
    ``total_nnz`` holds ``max(nnz_list)`` — the kernel grid bound; 0 ⇔ every
    sample's column is empty. ``col_start`` is 0 and unused.

    Fields:
      col_start  int            — B=1: offset into ``dist._csc_indices`` /
                                   ``input_layer.params[csc_values_base:]``.
                                   B>1: 0 (per-sample offsets in ``col_starts``).
      total_nnz  int            — B=1: length of the active column.
                                   B>1: max per-sample column length.
      indices    Tensor         — B=1: ``[total_nnz]`` view. B>1: the full
                                   ``_csc_indices`` (base for ``col_starts``).
      values     Tensor         — B=1: ``[total_nnz]``. B>1: ``[B, K_stride]``;
                                   log_emit + Σ log_trans (forward) or gathered
                                   flow (backward). Row stride is passed to
                                   kernels as a runtime arg.
      num_rows   int            — H.
      max_val    Tensor | None  — optional pre-computed per-sample max of
                                   ``values`` (f32, ``[1]`` at B=1 / ``[B]``
                                   at B>1), produced inline by the fused
                                   prod kernels so downstream sum layers can
                                   skip a per-block ``values.max()`` torch
                                   dispatch. ``None`` for backward containers
                                   and for any forward path that didn't fuse.
      batch_size int            — number of samples described (default 1).
      col_starts Tensor | None  — B>1 only: int32 ``[B]`` device tensor of
                                   per-sample column offsets.
      nnz        Tensor | None  — B>1 only: int32 ``[B]`` device tensor of
                                   per-sample column lengths.
      nnz_list   list | None    — B>1 only: host copy of ``nnz`` for empty
                                   checks, grid sizing and tests.
    """

    col_start: int
    total_nnz: int
    indices: torch.Tensor
    values: torch.Tensor
    num_rows: int
    max_val: Optional[torch.Tensor] = None
    batch_size: int = 1
    col_starts: Optional[torch.Tensor] = None
    nnz: Optional[torch.Tensor] = None
    nnz_list: Optional[List[int]] = None

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def is_batched(self) -> bool:
        return self.batch_size > 1

    def like_pattern(self, values: torch.Tensor,
                     max_val: Optional[torch.Tensor] = None) -> "SparseNodeValues":
        """New container sharing this one's pattern fields (``col_start(s)``,
        ``indices``, ``nnz``, ...) with fresh ``values`` — the flow-mirror
        constructor used by the sum layers' backward, so batched fields can
        never be forgotten on flow containers."""
        return SparseNodeValues(
            col_start=self.col_start, total_nnz=self.total_nnz,
            indices=self.indices, values=values, num_rows=self.num_rows,
            max_val=max_val,
            batch_size=self.batch_size, col_starts=self.col_starts,
            nnz=self.nnz, nnz_list=self.nnz_list,
        )

    def scatter_to_dense(self, out: torch.Tensor, out_base: int,
                         fill_value: float = LOG_EPS) -> None:
        """Fill ``out[out_base:out_base+H, 0] = fill_value`` then overwrite
        active rows with ``values``. Single-batch dense bridge (mixed-consumer
        topologies): writes at column 0 only."""
        assert not self.is_batched, (
            "scatter_to_dense is a B=1 dense bridge; the batched chain is "
            "packed-only (_skip_scatter)."
        )
        H = self.num_rows
        out[out_base:out_base + H, 0] = fill_value

        if self.total_nnz == 0:
            return
        # ``indices`` are already absolute row ids within this ns's H-slice.
        out[out_base + self.indices, 0] = self.values

    def gather_from_dense(self, src: torch.Tensor, src_base: int) -> "SparseNodeValues":
        """Return a new :class:`SparseNodeValues` sharing this object's
        pattern (``col_start``, ``indices``) whose
        ``values[j] = src[src_base + indices[j], 0]``. B=1 dense bridge."""
        assert not self.is_batched, (
            "gather_from_dense is a B=1 dense bridge; the batched chain is "
            "packed-only (_skip_scatter)."
        )
        if self.total_nnz == 0:
            return SparseNodeValues(
                col_start=self.col_start, total_nnz=0,
                indices=self.indices,
                values=torch.empty(0, dtype=torch.float32, device=self.device),
                num_rows=self.num_rows,
            )

        new_values = src[src_base + self.indices, 0].contiguous()
        return SparseNodeValues(
            col_start=self.col_start, total_nnz=self.total_nnz,
            indices=self.indices, values=new_values,
            num_rows=self.num_rows,
        )
