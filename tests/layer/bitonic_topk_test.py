"""Correctness tests for ``pyjuice.layer.bitonic_topk``.

The Triton implementation is compared against ``torch.topk`` on the same
input. Output values must bit-equal the reference values (after
descending sort), and the index set per batch column must match.
"""

import pytest
import torch

from pyjuice.layer.bitonic_topk import BitonicScratch, bitonic_topk


# ---- helpers ----------------------------------------------------------- #


def _reference_topk(values: torch.Tensor, K: int):
    """torch.topk reference, returned sorted descending for direct comparison."""
    ref_vals, ref_idx = torch.topk(values, k=K, dim=0, largest=True, sorted=True)
    return ref_vals, ref_idx


def _check_topk(values: torch.Tensor, K: int, *, scratch=None,
                chunk_size: int = 128, group_size: int = 2):
    H, B = values.shape
    out_vals = torch.empty((K, B), dtype=torch.float32, device=values.device)
    out_idx  = torch.empty((K, B), dtype=torch.int32,   device=values.device)

    bitonic_topk(values, K, out_vals=out_vals, out_idx=out_idx,
                 scratch=scratch, chunk_size=chunk_size, group_size=group_size)

    ref_vals, ref_idx = _reference_topk(values, K)

    # Values must be bit-equal after sorting descending (which the bitonic
    # path produces directly).
    assert torch.equal(out_vals, ref_vals), (
        f"value mismatch: max abs diff = "
        f"{(out_vals - ref_vals).abs().max().item()}"
    )

    # Indices: the index *set* per column must match (order may differ
    # when there are tied values, but bitonic also sorts so usually it
    # agrees exactly with sorted=True).
    out_idx64 = out_idx.to(torch.int64)
    ref_set = ref_idx.sort(dim=0).values
    out_set = out_idx64.sort(dim=0).values
    assert torch.equal(out_set, ref_set), (
        f"index set mismatch on at least one column"
    )

    # Sanity: gathered values from input via bitonic indices match out_vals.
    gathered = values.gather(0, out_idx64)
    assert torch.equal(gathered, out_vals)


# ---- basic shapes ------------------------------------------------------ #


@pytest.mark.parametrize("H", [64, 128, 129, 256, 1024])
@pytest.mark.parametrize("B", [1, 4, 16])
@pytest.mark.parametrize("K", [1, 8, 16, 64])
def test_bitonic_topk_basic(H, B, K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if K > H:
        pytest.skip("K > H is not supported (caller must fall back)")
    torch.manual_seed(0xb1701c)
    values = torch.randn(H, B, device="cuda", dtype=torch.float32)
    _check_topk(values, K)


# ---- non-power-of-two K and H ----------------------------------------- #


@pytest.mark.parametrize("K", [3, 7, 17, 33, 65, 100])
def test_bitonic_topk_non_po2_k(K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    values = torch.randn(2048, 8, device="cuda", dtype=torch.float32)
    _check_topk(values, K)


@pytest.mark.parametrize("H", [120, 200, 1000, 4097])
def test_bitonic_topk_non_aligned_h(H):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    values = torch.randn(H, 4, device="cuda", dtype=torch.float32)
    _check_topk(values, K=16)


# ---- K relative to chunk_size ----------------------------------------- #


@pytest.mark.parametrize("K", [127, 128, 129, 192, 256, 512])
def test_bitonic_topk_k_around_chunk_size(K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    values = torch.randn(4096, 4, device="cuda", dtype=torch.float32)
    _check_topk(values, K)


# ---- single-chunk edge case (H <= chunk_size) ------------------------- #


@pytest.mark.parametrize("H", [16, 32, 64, 128])
@pytest.mark.parametrize("K", [1, 4, 8])
def test_bitonic_topk_single_chunk(H, K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if K >= H:
        pytest.skip("K must be strictly less than H")
    torch.manual_seed(0xb1701c)
    values = torch.randn(H, 4, device="cuda", dtype=torch.float32)
    _check_topk(values, K)


# ---- in-place output: reuse pre-allocated buffers --------------------- #


def test_bitonic_topk_in_place_no_realloc():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    H, B, K = 1024, 8, 32
    values = torch.randn(H, B, device="cuda", dtype=torch.float32)
    out_vals = torch.empty((K, B), dtype=torch.float32, device="cuda")
    out_idx  = torch.empty((K, B), dtype=torch.int32, device="cuda")
    v_ptr, i_ptr = out_vals.data_ptr(), out_idx.data_ptr()

    scratch = BitonicScratch()
    bitonic_topk(values, K, out_vals=out_vals, out_idx=out_idx, scratch=scratch)
    assert out_vals.data_ptr() == v_ptr
    assert out_idx.data_ptr() == i_ptr

    ref_vals, _ = _reference_topk(values, K)
    assert torch.equal(out_vals, ref_vals)


# ---- scratch reuse across varying shapes ------------------------------ #


def test_bitonic_topk_scratch_reuse():
    """A single scratch object must give correct results across calls
    with growing B / H / K."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    scratch = BitonicScratch()

    for H, B, K in [(256, 4, 8), (1024, 8, 16), (1024, 8, 64), (4096, 16, 32)]:
        values = torch.randn(H, B, device="cuda", dtype=torch.float32)
        _check_topk(values, K, scratch=scratch)


# ---- large H (slow) --------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("H", [8192, 32768])
@pytest.mark.parametrize("K", [16, 64, 256])
def test_bitonic_topk_large(H, K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    values = torch.randn(H, 32, device="cuda", dtype=torch.float32)
    _check_topk(values, K)


# ---- determinism ------------------------------------------------------ #


def test_bitonic_topk_deterministic():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    values = torch.randn(2048, 16, device="cuda", dtype=torch.float32)
    K = 32

    outs = []
    for _ in range(2):
        out_vals = torch.empty((K, 16), dtype=torch.float32, device="cuda")
        out_idx  = torch.empty((K, 16), dtype=torch.int32, device="cuda")
        bitonic_topk(values, K, out_vals=out_vals, out_idx=out_idx)
        outs.append((out_vals.clone(), out_idx.clone()))

    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


# ---- non-contiguous-along-H input (matches element_mars layout) ------ #


def test_bitonic_topk_slice_input():
    """element_mars[a:b, :] is the typical input shape — verify the
    wrapper handles a row slice of a larger 2-D tensor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0xb1701c)
    full = torch.randn(4096, 8, device="cuda", dtype=torch.float32)
    sl = full[100:1100, :]
    _check_topk(sl, K=24)
