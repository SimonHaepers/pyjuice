from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


LOG_EPS = -23.0258509299  # log(1e-10), matches BlockedCategorical.


@dataclass
class BlockedNodeValues:
    """
    Packed values of the ``k`` latents activated by a
    :class:`BlockedCategorical` input at one variable — the blocked twin of
    :class:`SparseNodeValues`.

    Unlike the CSC container there are no index arrays: sample ``b``'s active
    rows are ``block_id(b) * k + arange(k)`` with
    ``block_id(b) = token_block[data[var_id, b]]``. The pattern is therefore
    fully described by the (``data``, ``var_id``, ``token_block``) triple,
    which the consuming kernels read directly — no host lookup, no per-call
    pattern build, one layout for every batch size.

    Fields:
      values      Tensor  — ``[B, k]`` contiguous (row stride ``k``); log
                            values (forward) or flows (backward).
      k           int     — latents per emission block.
      num_rows    int     — H (rows of the producing ns).
      data        Tensor  — ``[num_vars, B]`` long, on device (the circuit's
                            permuted input tensor).
      var_id      int     — variable whose observed token selects the block.
      token_block Tensor  — ``[V]`` long, on device.
      batch_size  int
      max_val     Tensor | None — ``[B]`` f32 per-sample max of ``values``,
                            fused into the producing kernel (forward only).
    """

    values: torch.Tensor
    k: int
    num_rows: int
    data: torch.Tensor
    var_id: int
    token_block: torch.Tensor
    batch_size: int
    max_val: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.values.device

    def like_pattern(self, values: torch.Tensor,
                     max_val: Optional[torch.Tensor] = None) -> "BlockedNodeValues":
        """New container sharing this one's pattern with fresh ``values``."""
        assert values.shape == (self.batch_size, self.k)
        return BlockedNodeValues(
            values=values, k=self.k, num_rows=self.num_rows,
            data=self.data, var_id=self.var_id, token_block=self.token_block,
            batch_size=self.batch_size, max_val=max_val,
        )

    def block_ids(self) -> torch.Tensor:
        """``[B]`` long: the emission block of each sample's observed token."""
        return self.token_block[self.data[self.var_id]]

    def row_ids(self) -> torch.Tensor:
        """``[B, k]`` long: absolute active row ids within the ns's H-slice."""
        return self.block_ids()[:, None] * self.k + torch.arange(
            self.k, device=self.device, dtype=torch.long)[None, :]

    def scatter_to_dense(self, out: torch.Tensor, out_base: int,
                         fill_value: float = LOG_EPS) -> None:
        """``out[out_base:out_base+H, :] = fill_value`` then overwrite the
        active rows of every sample with ``values`` (dense bridge for
        mixed-consumer topologies)."""
        H = self.num_rows
        B = self.batch_size
        out[out_base:out_base + H, :B] = fill_value
        rows = self.row_ids()                                          # [B, k]
        cols = torch.arange(B, device=self.device)[:, None].expand(B, self.k)
        out[out_base + rows.reshape(-1), cols.reshape(-1)] = self.values.reshape(-1)

    def gather_from_dense(self, src: torch.Tensor, src_base: int) -> "BlockedNodeValues":
        """Container with ``values[b, j] = src[src_base + row_ids[b, j], b]``."""
        B = self.batch_size
        rows = self.row_ids()
        cols = torch.arange(B, device=self.device)[:, None].expand(B, self.k)
        vals = src[src_base + rows.reshape(-1), cols.reshape(-1)].reshape(B, self.k).contiguous()
        return self.like_pattern(vals)
