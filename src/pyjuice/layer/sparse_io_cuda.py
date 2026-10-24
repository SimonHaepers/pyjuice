"""CUDA C++ kernels of :class:`SparseIOSumLayer`'s weight-stationary path
("sparse" dispatch), compiled at run time by :mod:`pyjuice.utils.nvrtc`.

The batched step ``y[b, h] = log sum_c W[h, c] exp(x[b, c] - max_b) + max_b``
at the active parents ``h`` of sample ``b`` over its active children ``c``
does 0.2-3 % of the dense ``B * H_out * H_in`` multiply-adds on the trained
chains, but the tensor cores need dense tiles: the tiles pass multiplies
~45 % of the dense product, and the samples of a step share too few tokens to
fill MMA rows with useful work (2026-10-23 analysis). These kernels form only
the products of the active pairs on the CUDA cores (up to the parent rows of a
small tile), exact fp32, with W in shared memory:

* the parent axis is cut into tiles of ``TMS`` rows (4 or 8) and the child
  axis into ranges of ``KS`` children whose ``[KS, TMS]`` W slab fits in
  shared memory (64 KB, three CTAs per SM); a CTA loads one slab with TMA and
  keeps it for its whole life, so a step reads W from HBM once;
* the samples with an active parent in the tile (lists built on the device
  by :data:`ws_slab_lists`, one atomic per sample and tile) are pulled by the
  warps one at a time; the lanes take every 32nd child of the sample in the
  range from its packed input column (:data:`ws_slab_pack`: per child the
  byte offset of its slab row and ``exp(v - max)``), read the 16-byte rows of
  the 4-row quarters of the tile that hold a parent of the sample, and
  accumulate;
* the warp sums its lanes with a transpose reduction and writes the packed
  output values of the sample's parents in the tile directly (with several
  ranges the linear partial sums add up in the output, and
  :data:`ws_slab_finalize` takes the log; with three or more ranges the
  order of those atomic additions, and so the last bits of a result, varies
  between runs).

On 8k / 16k chains of density 0.01 this is 1.2-2x faster than the tiles pass
from B=256 (2026-10-24 measurements); at density 0.1 the active pairs are ten
times more and the tensor-core passes win. A first design that streamed the
W of 32-row tiles through a ring of shared-memory chunks lost to it: the work
per (sample, chunk) was a few dozen products against ~60 instructions of
overhead.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from pyjuice.utils.nvrtc import CudaKernel, TensorMap, tensor_map_2d

SMEM_LIMIT = 227 * 1024
"""Dynamic shared memory a CTA may use (H100: 227 KB)."""

_SLAB_SRC = r"""
#define NT (NWARP * 32)
#define NSPLIT ((H_IN + KS - 1) / KS)
#define FULL 0xffffffffu

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
struct __align__(64) TmaDesc { unsigned long long d[16]; };

// Per step: xp[b, j] = (byte offset of child c_j's slab row, exp(v_j - max_b))
// for the input column of sample b; cofs[b, s] = the first j with
// c_j >= s * KS (s = 0..NSPLIT; the column ascends). Grid (cdiv(K + 1, 256), B).
extern "C" __global__ void __launch_bounds__(256)
ws_slab_pack(int2* __restrict__ xp, long long xp_stride, int* __restrict__ cofs,
             const long long* __restrict__ in_idx, const float* __restrict__ in_vals,
             long long in_stride, const int* __restrict__ in_cs, const int* __restrict__ in_nnz,
             const float* __restrict__ max_val)
{
    const int b = blockIdx.y;
    const int j = blockIdx.x * 256 + threadIdx.x;
    const int n = in_nnz[b];
    if (j > n) return;                                 // thread j == n closes the column
    const long long* ip = in_idx + in_cs[b];
    int* cb = cofs + (long long)b * (NSPLIT + 1);
    const int kp = j == 0 ? -1 : (int)(ip[j - 1] / KS);
    const int c = j < n ? (int)ip[j] : 0;
    const int kc = j < n ? c / KS : NSPLIT;
    for (int k = kp + 1; k <= kc; ++k) cb[k] = j;
    if (j < n) {
        const float x = expf(in_vals[(long long)b * in_stride + j] - fmaxf(max_val[b], -1e30f));
        xp[(long long)b * xp_stride + j] = make_int2((c % KS) * TMS * 4, __float_as_int(x));
    }
}

// Per step, the samples of every TMS-row parent tile: for the first active
// row of sample b in tile t, ent[t, pos] = (b, ps << TMS | rows) with pos from
// an atomic counter (cnt zeroed before; the order within a list is
// irrelevant), ps the index of that row in b's output column and rows the
// mask of b's active rows in the tile (the column ascends). Grid
// (cdiv(K_out, 256), B).
extern "C" __global__ void __launch_bounds__(256)
ws_slab_lists(int2* __restrict__ ent, int* __restrict__ cnt, const long long* __restrict__ out_idx,
              const int* __restrict__ out_cs, const int* __restrict__ out_nnz, int B,
              float* __restrict__ out_vals, long long out_stride)
{
    const int b = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int n = out_nnz[b];
    if (i >= n) return;
#if NSPLIT > 1
    out_vals[(long long)b * out_stride + i] = 0.f;    // the ranges' partial sums add up here
#endif
    const long long* op = out_idx + out_cs[b];
    const int h = (int)op[i], t = h / TMS;
    if (i > 0 && (int)op[i - 1] / TMS == t) return;
    unsigned rows = 1u << (h - t * TMS);
    #pragma unroll
    for (int k = 1; k < TMS; ++k) {
        if (i + k >= n) break;
        const int hk = (int)op[i + k];
        if (hk / TMS != t) break;
        rows |= 1u << (hk - t * TMS);
    }
    const int pos = atomicAdd(cnt + t, 1);
    ent[(long long)t * B + pos] = make_int2(b, (i << TMS) | (int)rows);
}

#if TMS >= 4
// 4-row quarters of a TMS-row tile with an active row
__device__ __forceinline__ unsigned quarters(unsigned rows) {
    unsigned q = 0;
    #pragma unroll
    for (int k = 0; k < TMS / 4; ++k) q |= ((rows >> (4 * k)) & 15u) ? 1u << k : 0u;
    return q;
}
// a[4q..4q+3] += x_j * W[c_j, 4q..4q+3] over the lane's children j (every
// 32nd from j up to j1) for the quarters q in QM, four children in flight;
// past j1 a lane adds 0 x row 0
template <unsigned QM>
__device__ __forceinline__ void slab_walk(float* a, const int2* __restrict__ pp, const char* sb,
                                          int j, int j1) {
    for (; j < j1; j += 128, pp += 128) {
        int2 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = j + 32 * u < j1 ? __ldg(pp + 32 * u) : make_int2(0, 0);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float x = __int_as_float(v[u].y);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (!((QM >> q) & 1)) continue;
                const float4 wv = reinterpret_cast<const float4*>(sb + v[u].x)[q];
                a[4 * q] = fmaf(x, wv.x, a[4 * q]);
                a[4 * q + 1] = fmaf(x, wv.y, a[4 * q + 1]);
                a[4 * q + 2] = fmaf(x, wv.z, a[4 * q + 2]);
                a[4 * q + 3] = fmaf(x, wv.w, a[4 * q + 3]);
            }
        }
    }
}
// the same for any quarter mask (uniform branches)
__device__ __forceinline__ void slab_walk_any(float* a, const int2* __restrict__ pp, const char* sb,
                                              int j, int j1, unsigned qm) {
    for (; j < j1; j += 128, pp += 128) {
        int2 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = j + 32 * u < j1 ? __ldg(pp + 32 * u) : make_int2(0, 0);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float x = __int_as_float(v[u].y);
            #pragma unroll
            for (int q = 0; q < TMS / 4; ++q) {
                if (!((qm >> q) & 1)) continue;
                const float4 wv = reinterpret_cast<const float4*>(sb + v[u].x)[q];
                a[4 * q] = fmaf(x, wv.x, a[4 * q]);
                a[4 * q + 1] = fmaf(x, wv.y, a[4 * q + 1]);
                a[4 * q + 2] = fmaf(x, wv.z, a[4 * q + 2]);
                a[4 * q + 3] = fmaf(x, wv.w, a[4 * q + 3]);
            }
        }
    }
}
#endif

__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
    unsigned ok;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(smem_addr(bar)), "r"(parity) : "memory");
    } while (!ok);
}

// copy the W slab [KS children][TMS parents] of tile h0, children c0.., into
// dst; `bar` completes when it has landed (TMA: thread 0; cp.async: every
// thread)
__device__ __forceinline__ void load_slab(float* dst, unsigned long long* bar, int h0, int c0,
                                          const float* __restrict__ w, long long w_start,
                                          const TmaDesc* tmap, int tid) {
#if LOADER == 8
    if (tid == 0) {
        constexpr int BOXR = KS < 256 ? KS : 256;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     :: "r"(smem_addr(bar)), "r"(KS * TMS * 4) : "memory");
        const int row0 = (h0 / BS) * H_IN + c0, col0 = h0 % BS;
        for (int part = 0; part < KS / BOXR; ++part)
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];\n"
                         :: "r"(smem_addr(dst + part * BOXR * TMS)), "l"(tmap), "r"(col0),
                            "r"(row0 + part * BOXR), "r"(smem_addr(bar)) : "memory");
    }
#else
    for (int e = tid; e < KS * TMS; e += NT) {
        const int r = e / TMS, col = e - r * TMS;
        const int c = c0 + r, h = h0 + col;
        if (c < H_IN && h < H_OUT)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                         :: "r"(smem_addr(dst + e)),
                            "l"(w + w_start + ((long long)(h / BS) * H_IN + c) * BS + (h % BS)) : "memory");
    }
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(smem_addr(bar)) : "memory");
#endif
}

// CTA = (child range s of KS children, parent tile m of TMS rows): the tile's
// W slab [KS children][TMS parents] sits in shared memory for the whole CTA;
// the warps pull the samples of the tile's list one at a time, each lane
// walks every 32nd child of the sample in the range (one LDS of the weights
// of the 4-row quarters with a parent of the sample, their FMAs), the warp
// sums the lanes and writes log(acc) + max (NSPLIT == 1) or adds the linear
// partial to out_vals (finished by ws_slab_finalize). Exact fp32.
extern "C" __global__ void __launch_bounds__(NT, MINB)
ws_slab_fwd(float* __restrict__ out_vals, long long out_stride,
            const float* __restrict__ w, long long w_start,
            const int2* __restrict__ xp, int xp_stride, const int* __restrict__ cofs,
            const float* __restrict__ max_val, const int* __restrict__ in_nnz,
            const int2* __restrict__ ent, const int* __restrict__ cnt, int B,
            const __grid_constant__ TmaDesc tmap)
{
    extern __shared__ __align__(128) float smem[];
    float* slab = smem;                                              // [KS][TMS]
    unsigned long long* bar = reinterpret_cast<unsigned long long*>(slab + KS * TMS);
    int* s_item = reinterpret_cast<int*>(bar + 1);

    const int m = blockIdx.y, sp = blockIdx.x;
    const int n = cnt[m];
    if (n == 0) return;
    const int tid = threadIdx.x, lane = tid & 31;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                     :: "r"(smem_addr(bar)), "r"(LOADER == 8 ? 1 : NT) : "memory");
        *s_item = 0;
    }
    __syncthreads();
    load_slab(slab, bar, m * TMS, sp * KS, w, w_start, &tmap, tid);
    mbar_wait(bar, 0);
    const char* sb = reinterpret_cast<const char*>(slab);

    const int2* lst = ent + (long long)m * B;
    // items: the samples of the tile's list, pulled one at a time; the next
    // item's entry and child range are loaded while the current one runs
    int i = 0;
    if (lane == 0) i = atomicAdd(s_item, 1);
    i = __shfl_sync(FULL, i, 0);
    int2 e = make_int2(0, 0), cr = make_int2(0, 0);
    if (i < n) {
        e = lst[i];
        cr = make_int2(cofs[(long long)e.x * (NSPLIT + 1) + sp], cofs[(long long)e.x * (NSPLIT + 1) + sp + 1]);
    }
    while (i < n) {
        const int b = e.x, ps = e.y >> TMS;
        const unsigned rows = e.y & ((1 << TMS) - 1);
        const int j0 = cr.x, j1 = cr.y;
        int inext = 0;
        if (lane == 0) inext = atomicAdd(s_item, 1);
        inext = __shfl_sync(FULL, inext, 0);
        if (inext < n) {
            e = lst[inext];
            cr = make_int2(cofs[(long long)e.x * (NSPLIT + 1) + sp], cofs[(long long)e.x * (NSPLIT + 1) + sp + 1]);
        }

        float a[TMS];
        #pragma unroll
        for (int r = 0; r < TMS; ++r) a[r] = 0.f;
        const int2* __restrict__ pp = xp + ((unsigned)(b * xp_stride) + (unsigned)(j0 + lane));
#if TMS >= 8
        // only the 4-row quarters of the tile that hold one of the sample's
        // parents are read and multiplied (uniform per item)
        const unsigned qmask = quarters(rows);
        if (qmask == 1u) slab_walk<1>(a, pp, sb, j0 + lane, j1);
        else if (qmask == 2u) slab_walk<2>(a, pp, sb, j0 + lane, j1);
        else if (qmask == 3u) slab_walk<3>(a, pp, sb, j0 + lane, j1);
#if TMS == 16
        else slab_walk_any(a, pp, sb, j0 + lane, j1, qmask);
#endif
#elif TMS == 4
        slab_walk<1>(a, pp, sb, j0 + lane, j1);
#else
        // four children per lane in flight; past j1 a lane adds 0 x row 0
        for (int j = j0 + lane; j < j1; j += 128, pp += 128) {
            int2 v[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) v[u] = j + 32 * u < j1 ? __ldg(pp + 32 * u) : make_int2(0, 0);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float x = __int_as_float(v[u].y);
                #pragma unroll
                for (int r = 0; r < TMS; ++r)
                    a[r] = fmaf(x, reinterpret_cast<const float*>(sb + v[u].x)[r], a[r]);
            }
        }
#endif
        // the warp's sum per parent row (transpose reduction: at each of the
        // first LOG_TMS levels a lane keeps half of its rows and sends the
        // other half): row r = lane >> (5 - LOG_TMS) ends in lane r * (32 / TMS)
        #pragma unroll
        for (int lv = 0; lv < LOG_TMS; ++lv) {
            const int off = 16 >> lv, h = TMS >> (lv + 1);
            const bool hi = lane & off;
            #pragma unroll
            for (int k = 0; k < h; ++k) {
                const float keep = hi ? a[k + h] : a[k], send = hi ? a[k] : a[k + h];
                a[k] = keep + __shfl_xor_sync(FULL, send, off);
            }
        }
        float acc = a[0];
        #pragma unroll
        for (int off = 16 >> LOG_TMS; off; off >>= 1) acc += __shfl_xor_sync(FULL, acc, off);
        // active rows of the tile ascend: row r's output is ps + rank(r)
        const int r = lane >> (5 - LOG_TMS);
        if ((lane & ((32 >> LOG_TMS) - 1)) == 0 && ((rows >> r) & 1)) {
            float* dst = out_vals + (long long)b * out_stride + ps + __popc(rows & ((1u << r) - 1));
#if NSPLIT == 1
            *dst = in_nnz[b] == 0 ? __int_as_float(0xff800000) : logf(acc) + fmaxf(max_val[b], -1e30f);
#else
            atomicAdd(dst, acc);
#endif
        }
        i = inext;
    }
}

// NSPLIT > 1: out[b, k] = log(sum of the partials) + max_b (-inf for an
// empty input column), k < nnz_out[b]. Grid (cdiv(K_out, 256), B).
extern "C" __global__ void __launch_bounds__(256)
ws_slab_finalize(float* __restrict__ out_vals, long long out_stride, const int* __restrict__ out_nnz,
                 const int* __restrict__ in_nnz, const float* __restrict__ max_val)
{
    const int b = blockIdx.y;
    const int k = blockIdx.x * 256 + threadIdx.x;
    if (k >= out_nnz[b]) return;
    float* dst = out_vals + (long long)b * out_stride + k;
    *dst = in_nnz[b] == 0 ? __int_as_float(0xff800000) : logf(*dst) + fmaxf(max_val[b], -1e30f);
}
"""

_SLAB_FWD_ARGS = ["ptr", "i64", "ptr", "i64", "ptr", "i32", "ptr", "ptr", "ptr",
                  "ptr", "ptr", "i32", "tmap"]
_SLAB_LIST_ARGS = ["ptr", "ptr", "ptr", "ptr", "ptr", "i32", "ptr", "i64"]
_SLAB_FIN_ARGS = ["ptr", "i64", "ptr", "ptr", "ptr"]
_SLAB_KERNELS: Dict[tuple, tuple] = {}
_PACK_ARGS = ["ptr", "i64", "ptr", "ptr", "ptr", "i64", "ptr", "ptr", "ptr"]

FORCE_LOADER: Optional[int] = (int(os.environ["PYJUICE_SPARSE_IO_WS_SPARSE_LOADER"])
                               if os.environ.get("PYJUICE_SPARSE_IO_WS_SPARSE_LOADER") else None)
"""Tests / A-B runs (env ``PYJUICE_SPARSE_IO_WS_SPARSE_LOADER``): force the
copy of the W slabs — 8 TMA (needs 4 or 8-row tiles inside one PC block and
16-byte aligned rows; the default then), 1 4-byte ``cp.async`` (any layout)."""

_TMAPS: Dict[tuple, TensorMap] = {}
_NO_TMAP = TensorMap()


def slab_shape(H_in: int, TMS: int, slab_bytes: int) -> Tuple[int, int]:
    """``(KS, NSPLIT)``: the fewest ranges of ``KS`` children (a multiple of
    32, or of 256 from 256 on: the TMA box) whose ``KS x TMS`` fp32 slab
    stays within ``slab_bytes``, balanced over ``H_in``."""
    max_rows = max(32, slab_bytes // (TMS * 4))
    nsplit = -(-H_in // max_rows)
    while True:
        rows = -(-H_in // nsplit)
        unit = 256 if rows >= 256 else 32
        KS = -(-rows // unit) * unit
        if KS <= max_rows or unit == 32:
            return KS, -(-H_in // KS)
        nsplit += 1


def slab_smem(KS: int, TMS: int) -> int:
    """Dynamic shared memory of the forward: the slab, its barrier and the
    item counter."""
    return KS * TMS * 4 + 16


def slab_kernels(H_in: int, H_out: int, BS: int, TMS: int, KS: int, NWARP: int,
                 loader: int, min_blocks: int) -> tuple:
    """``(lists, pack, forward, finalize kernels, forward smem bytes)`` of a
    block shape; compiled on first use (cubins cached on disk)."""
    key = (H_in, H_out, BS, TMS, KS, NWARP, loader, min_blocks)
    hit = _SLAB_KERNELS.get(key)
    if hit is None:
        smem = slab_smem(KS, TMS)
        defines = dict(H_IN=H_in, H_OUT=H_out, BS=BS, TMS=TMS, LOG_TMS=TMS.bit_length() - 1, KS=KS,
                       NWARP=NWARP, LOADER=loader, MINB=min_blocks)
        src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + _SLAB_SRC
        hit = _SLAB_KERNELS[key] = (CudaKernel(src, "ws_slab_lists", _SLAB_LIST_ARGS),
                                    CudaKernel(src, "ws_slab_pack", _PACK_ARGS),
                                    CudaKernel(src, "ws_slab_fwd", _SLAB_FWD_ARGS, max_shared=smem),
                                    CudaKernel(src, "ws_slab_finalize", _SLAB_FIN_ARGS), smem)
    return hit


def ws_slab_forward(out_vals: torch.Tensor, w: torch.Tensor, w_start: int, sv_in, sv_out,
                    max_val: torch.Tensor, ent: torch.Tensor, cnt: torch.Tensor,
                    xp: torch.Tensor, cofs: torch.Tensor, B: int, H_in: int, H_out: int, BS: int,
                    TMS: int = 4, slab_bytes: int = 64 * 1024, NWARP: int = 16,
                    min_blocks: int = 3) -> None:
    """Build the per-tile sample lists, pack the input columns and run the
    slab forward for one block, writing ``out_vals[b, i]`` (row stride
    ``out_vals.stride(0)``) for every active parent ``i`` of every sample;
    ``w[w_start:]`` holds W in the block layout of :class:`SparseIOSumLayer`.
    Scratch: ``ent`` (int32 ``[cdiv(H_out, TMS), 2 B]``), ``cnt`` (int32
    ``[cdiv(H_out, TMS)]``), ``xp`` (int32 ``[B, >= 2 K_in]``), ``cofs``
    (int32 ``[B, NSPLIT + 1]``, :func:`slab_shape`). Graph-capturable, no
    host sync."""
    assert TMS in (1, 2, 4, 8, 16)
    KS, nsplit = slab_shape(H_in, TMS, slab_bytes)
    assert sv_out.values.shape[-1] < 1 << (31 - TMS), "output columns too long for the list entries"
    n_m = -(-H_out // TMS)
    aligned = (TMS >= 4 and BS % TMS == 0 and (w.data_ptr() // 4 + w_start) % 4 == 0
               and w.data_ptr() % 4 == 0)
    loader = FORCE_LOADER if FORCE_LOADER in (1, 8) else (8 if aligned else 1)
    assert loader == 1 or aligned
    lists, pack, fwd, fin, smem = slab_kernels(H_in, H_out, BS, TMS, KS, NWARP, loader, min_blocks)
    K = sv_in.total_nnz
    assert xp.stride(0) >= 2 * K and xp.stride(0) % 2 == 0 and cofs.stride(0) == nsplit + 1
    assert ent.shape[0] >= n_m and ent.stride(0) == 2 * B and cnt.numel() >= n_m
    device = out_vals.device
    if loader == 8:
        key = (w.data_ptr(), w_start, H_in, H_out, BS, TMS, KS)
        tmap = _TMAPS.get(key)
        if tmap is None:
            # W as [NB * H_in rows (block, child), BS parents], boxes of TMS parents x <= 256 children
            tmap = _TMAPS[key] = tensor_map_2d(w, w_start, H_out // BS * H_in, BS, BS,
                                               min(KS, 256), TMS, swizzle=0)
    else:
        tmap = _NO_TMAP
    cnt.zero_()
    lists((-(-sv_out.total_nnz // 256), B), (256,),
          [ent, cnt, sv_out.indices, sv_out.col_starts, sv_out.nnz, B, out_vals, out_vals.stride(0)],
          device=device)
    pack((-(-(K + 1) // 256), B), (256,),
         [xp, xp.stride(0) // 2, cofs, sv_in.indices, sv_in.values, sv_in.values.stride(0),
          sv_in.col_starts, sv_in.nnz, max_val], device=device)
    fwd((nsplit, n_m), (NWARP * 32,),
        [out_vals, out_vals.stride(0), w, w_start, xp, xp.stride(0) // 2, cofs, max_val, sv_in.nnz,
         ent, cnt, B, tmap], shared_mem=smem, device=device)
    if nsplit > 1:
        fin((-(-sv_out.total_nnz // 256), B), (256,),
            [out_vals, out_vals.stride(0), sv_out.nnz, sv_in.nnz, max_val], device=device)
