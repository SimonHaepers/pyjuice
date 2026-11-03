"""CUDA C++ kernels of :class:`SparseIOSumLayer`'s batched sparse pass,
compiled at run time by :mod:`pyjuice.utils.nvrtc`.

The batched step ``y[b, h] = log sum_c W[h, c] exp(x[b, c] - max_b) + max_b``
at the active parents ``h`` of sample ``b`` over its active children ``c``
does 0.2-3 % of the dense ``B * H_out * H_in`` multiply-adds on the trained
chains, and the samples of a step share too few tokens to fill tensor-core
tiles with useful work (2026-10-23 analysis). These kernels form only the
products of the active pairs on the CUDA cores, with W in shared memory:

* the parent axis is cut into tiles of ``TMS`` rows (4 by default) and the
  child axis into ranges of ``KS`` children whose ``[KS, TMS]`` W slab fits in
  shared memory (64 KB, three CTAs per SM); a CTA loads one slab with bulk
  copies and keeps it for its whole life, so a step reads W from HBM once.
  The slabs come from a per-pass copy of W in slab order (:func:`slab_wcopy`),
  in fp32 or bf16 (``w16``: half the bytes of every slab load and of every
  W read from shared memory; the forward and the element-flow backward then
  round W to 8 bits of mantissa);
* the samples with an active parent in the tile (lists built on the device
  by :data:`ws_slab_lists`, one atomic per sample and tile) are handed to the
  warps round-robin; the 32 lanes take every 32nd child of the sample in the
  range from its packed input column (:data:`ws_slab_pack`: per child the
  byte offset of its slab row and ``exp(v - max)``), read the 4-row quarters
  of the tile that hold a parent of the sample (one 16-byte or 8-byte read
  each), and accumulate;
* the lanes' sums meet in a transpose reduction and write the packed output
  values of the sample's parents in the tile directly (with several ranges
  the linear partial sums add up in the output, and :data:`ws_slab_finalize`
  takes the log; with three or more ranges the order of those atomic
  additions, and so the last bits of a result, varies between runs).

The element-flow backward, ``flow_in[b, c] = x[b, c] * sum_h g[b, h] W[h, c]``
over the sample's active parents, is the same kernel compiled with ``BWD``:
tiles of children, ranges of parents (from the backward slab-order copy,
``[KS parents][TMS children]``), lanes over the sample's parents packed as
(slab offset, ``g``) by :data:`ws_slab_pack_g`.

The parameter flows of tied transitions are accumulated separately
(:func:`pflow_tile_stash` / :func:`pflow_tile_flush`): every step's packed
inputs are stashed, and every few steps a CTA per tile of 4 parent rows
adds the stored steps' ``g x`` products of the active pairs to the tile in
32-bit fixed point in shared memory, the samples with the same child list
(next token) summed in registers first, and adds the tile to the caller's
dense accumulator ``S``; the caller multiplies ``S`` by W at the pass end.

Measurements: prune-pc ``results/sparse_io_paths`` (2026-10-24 to
2026-11-02) and ``REPORT_2026-10-26_tiling.md``. Earlier variants — TMA
boxes from W itself, swizzled slabs, several items per warp, a shared item
counter, list entries loaded two items ahead, lists reordered by bank group,
samples with the same column walked together, software-pipelined walks,
row-by-row slabs, parameter flows fused into the backward kernel, fp32
compare-and-swap tiles — measured no better and were removed on 2026-11-02.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from pyjuice.utils.nvrtc import CudaKernel

SMEM_LIMIT = 227 * 1024
"""Dynamic shared memory a CTA may use on the H100 (227 KB); :func:`smem_limit` asks the device (A100: 164 KB)."""

_SMEM_LIMIT: dict = {}


def smem_limit(device) -> int:
    """Dynamic shared memory a CTA may use on ``device`` (opt-in maximum)."""
    idx = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    hit = _SMEM_LIMIT.get(idx)
    if hit is None:
        props = torch.cuda.get_device_properties(idx)
        hit = _SMEM_LIMIT[idx] = int(getattr(props, "shared_memory_per_block_optin", SMEM_LIMIT))
    return hit

_SLAB_SRC = r"""
#define NT (NWARP * 32)
#define NSPLIT NSPLIT_OVERRIDE
#if BWD
#define SLAB_MAIN ws_slab_bwd
#else
#define SLAB_MAIN ws_slab_fwd
#endif
#define FULL 0xffffffffu
// W16: the slab holds W in bf16 (WB = 2 bytes per element) instead of fp32: the slab-order copies are written in
// bf16 by the copy kernels (round to nearest even), a quarter (4 rows) of a child is one 8-byte read unpacked by
// shifts.
#ifndef W16
#define W16 0
#endif
#define WB (W16 ? 2 : 4)
#define SLAB_BYTES (KS * TMS * WB)
__device__ __forceinline__ unsigned short f2bf_(float f) {
    const unsigned u = __float_as_uint(f);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float4 unpack16_(uint2 u) {
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                       __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
#if W16
#define WSTORE(dst, i, v) reinterpret_cast<unsigned short*>(dst)[i] = f2bf_(v)
#define QUARTER(wv, base, q) const float4 wv = unpack16_(reinterpret_cast<const uint2*>(base)[q]);
#else
#define WSTORE(dst, i, v) (dst)[i] = (v)
#define QUARTER(wv, base, q) const float4 wv = reinterpret_cast<const float4*>(base)[q];
#endif

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

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
        xp[(long long)b * xp_stride + j] = make_int2((c % KS) * (TMS * WB), __float_as_int(x));
    }
}

#if !BWD
// W in forward slab order: the slab of (tile m, range s) is the KS x TMS elements at
// ((m * NSPLIT + s) * KS) * TMS, row c the tile's TMS parents of child s * KS + c; padding is
// zero. Through shared memory: a block reads 32 children x 32 parents (128-byte rows of W,
// parents contiguous) and writes the 32 / TMS tiles' slab rows contiguously.
// Grid (cdiv(H_out, 32), NSPLIT * KS / 32), block 256.
extern "C" __global__ void __launch_bounds__(256)
ws_slab_wcopy(float* __restrict__ dst, const float* __restrict__ w, long long w_start)
{
    __shared__ float t[32][33];                               // [child][parent]
    const int h0 = blockIdx.x * 32, j0 = blockIdx.y * 32;     // j: child position in [0, NSPLIT * KS)
    for (int e = threadIdx.x; e < 32 * 32; e += 256) {
        const int cl = e / 32, hl = e - cl * 32;
        const int h = h0 + hl, c = j0 + cl;
        t[cl][hl] = (h < H_OUT && c < H_IN) ? w[w_start + ((long long)(h / BS) * H_IN + c) * BS + (h % BS)] : 0.f;
    }
    __syncthreads();
    const int sp = j0 / KS;
    for (int e = threadIdx.x; e < 32 * 32; e += 256) {
        // e = (tile in the block, child, row): each tile's 32 rows of TMS elements are contiguous
        const int tb = e / (32 * TMS), rem = e - tb * (32 * TMS), cl = rem / TMS, r = rem - cl * TMS;
        const int m = (h0 + tb * TMS) / TMS, cin = j0 - sp * KS + cl;
        if (m * TMS < ((H_OUT + TMS - 1) / TMS) * TMS)
            WSTORE(dst, ((long long)(m * NSPLIT + sp) * KS + cin) * TMS + r, t[cl][tb * TMS + r]);
    }
}
#endif

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
__device__ __forceinline__ void slab_walk(float* a, const int2* __restrict__ pp, const char* sb, int j, int j1) {
    for (; j < j1; j += 4 * 32, pp += 4 * 32) {
        int2 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = j + 32 * u < j1 ? __ldg(pp + 32 * u) : make_int2(0, 0);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float x = __int_as_float(v[u].y);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (!((QM >> q) & 1)) continue;
                QUARTER(wv, sb + v[u].x, q)
                a[4 * q] = fmaf(x, wv.x, a[4 * q]);
                a[4 * q + 1] = fmaf(x, wv.y, a[4 * q + 1]);
                a[4 * q + 2] = fmaf(x, wv.z, a[4 * q + 2]);
                a[4 * q + 3] = fmaf(x, wv.w, a[4 * q + 3]);
            }
        }
    }
}
// the same for any quarter mask (16-row tiles with three or four active quarters)
__device__ __forceinline__ void slab_walk_any(float* a, const int2* __restrict__ pp, const char* sb,
                                              int j, int j1, unsigned qm) {
    for (; j < j1; j += 4 * 32, pp += 4 * 32) {
        int2 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = j + 32 * u < j1 ? __ldg(pp + 32 * u) : make_int2(0, 0);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float x = __int_as_float(v[u].y);
            #pragma unroll
            for (int q = 0; q < TMS / 4; ++q) {
                if (!((qm >> q) & 1)) continue;
                QUARTER(wv, sb + v[u].x, q)
                a[4 * q] = fmaf(x, wv.x, a[4 * q]);
                a[4 * q + 1] = fmaf(x, wv.y, a[4 * q + 1]);
                a[4 * q + 2] = fmaf(x, wv.z, a[4 * q + 2]);
                a[4 * q + 3] = fmaf(x, wv.w, a[4 * q + 3]);
            }
        }
    }
}

// The W slab of tile t0.. (TMS rows) and range k0.. (KS rows) is contiguous in the slab-order copy
// (either direction). sm_90: thread 0 issues bulk copies of up to 16 KB that complete on the
// mbarrier `bar` (slab_init / slab_ready wrap it); earlier GPUs (the A100, sm_80): every thread
// copies 16-byte pieces and slab_ready is a __syncthreads().
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void slab_init(unsigned long long* bar, int tid) {
    if (tid == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(bar)), "r"(1) : "memory");
    __syncthreads();
}

__device__ __forceinline__ void load_slab(float* dst, unsigned long long* bar, int t0, int k0,
                                          const float* __restrict__ w, long long w_start, int tid) {
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     :: "r"(smem_addr(bar)), "r"(SLAB_BYTES) : "memory");
        const char* src = reinterpret_cast<const char*>(w + w_start)
                          + ((long long)(t0 / TMS) * NSPLIT + k0 / KS) * SLAB_BYTES;
        for (int off = 0; off < SLAB_BYTES; off += 16384) {
            const int sz = SLAB_BYTES - off < 16384 ? SLAB_BYTES - off : 16384;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         :: "r"(smem_addr(dst) + off), "l"(src + off), "r"(sz), "r"(smem_addr(bar)) : "memory");
        }
    }
}

__device__ __forceinline__ void slab_ready(unsigned long long* bar) {
    unsigned ok;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(smem_addr(bar)), "r"(0) : "memory");
    } while (!ok);
}
#else
__device__ __forceinline__ void slab_init(unsigned long long*, int) {}

__device__ __forceinline__ void load_slab(float* dst, unsigned long long*, int t0, int k0,
                                          const float* __restrict__ w, long long w_start, int tid) {
    const int4* src = reinterpret_cast<const int4*>(
        reinterpret_cast<const char*>(w + w_start) + ((long long)(t0 / TMS) * NSPLIT + k0 / KS) * SLAB_BYTES);
    int4* d = reinterpret_cast<int4*>(dst);
    for (int i = tid; i < SLAB_BYTES / 16; i += NT) d[i] = __ldg(src + i);
}

__device__ __forceinline__ void slab_ready(unsigned long long*) { __syncthreads(); }
#endif

// CTA = (child range s of KS children, parent tile m of TMS rows): the tile's
// W slab [KS children][TMS parents] sits in shared memory for the whole CTA;
// warp w takes the samples w, w + NWARP, ... of the tile's list, each lane
// walks every 32nd child of the sample in the range (one read of the weights
// of the 4-row quarters with a parent of the sample, their FMAs), the warp
// sums the lanes and writes log(acc) + max (NSPLIT == 1) or adds the linear
// partial to out_vals (finished by ws_slab_finalize). fp32 multiply-adds.
extern "C" __global__ void __launch_bounds__(NT, MINB)
SLAB_MAIN(float* __restrict__ out_vals, long long out_stride,
          const float* __restrict__ w, long long w_start,
          const int2* __restrict__ xp, int xp_stride, const int* __restrict__ cofs,
          const float* __restrict__ max_val, const int* __restrict__ in_nnz,
          const int2* __restrict__ ent, const int* __restrict__ cnt, int B,
          const float* __restrict__ in_vals, long long in_stride)
{
    extern __shared__ __align__(1024) float smem[];
    float* slab = smem;                                              // [KS][TMS], SLAB_BYTES
    unsigned long long* bar = reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(smem) + SLAB_BYTES);

    const int m = blockIdx.y, sp = blockIdx.x;
    const int n = cnt[m];
    if (n == 0) return;
    const int tid = threadIdx.x, lane = tid & 31;
    slab_init(bar, tid);
    load_slab(slab, bar, m * TMS, sp * KS, w, w_start, tid);
    const char* sb = reinterpret_cast<const char*>(slab);
    slab_ready(bar);

    // the warp's items: the next item's entry and child range are loaded while the current one runs
    const int2* lst = ent + (long long)m * B;
    int i = tid >> 5;
    int2 e = make_int2(0, 0), cr = make_int2(0, 0);
    if (i < n) {
        e = lst[i];
        cr = make_int2(cofs[(long long)e.x * (NSPLIT + 1) + sp], cofs[(long long)e.x * (NSPLIT + 1) + sp + 1]);
    }
    while (i < n) {
        const int b = e.x, ps = e.y >> TMS;
        const unsigned rows = e.y & ((1 << TMS) - 1);
        const int j0 = cr.x, j1 = cr.y;
        const int inext = i + NWARP;
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
#else
        slab_walk<1>(a, pp, sb, j0 + lane, j1);
#endif
        // the warp's sum per parent row (transpose reduction: at each of the
        // first LOG_TMS levels a lane keeps half of its rows and sends the
        // other half), then plain sums finish row r = lane >> (5 - LOG_TMS)
        // in lane r * (32 / TMS)
        #pragma unroll
        for (int lv = 0; lv < LOG_TMS; ++lv) {
            const int off = 32 >> (lv + 1), h = TMS >> (lv + 1);
            const bool hi = lane & off;
            #pragma unroll
            for (int k = 0; k < h; ++k) {
                const float keep = hi ? a[k + h] : a[k], send = hi ? a[k] : a[k + h];
                a[k] = keep + __shfl_xor_sync(FULL, send, off);
            }
        }
        #pragma unroll
        for (int off = 32 >> (LOG_TMS + 1); off; off >>= 1) a[0] += __shfl_xor_sync(FULL, a[0], off);
        const int r = lane >> (5 - LOG_TMS);
        const bool writer = (lane & ((32 >> LOG_TMS) - 1)) == 0;
        // active rows of the tile ascend: row r's output is ps + rank(r)
        if (writer && ((rows >> r) & 1)) {
            const int o = ps + __popc(rows & ((1u << r) - 1));
            float* dst = out_vals + (long long)b * out_stride + o;
            const float acc = a[0];
#if NSPLIT > 1
            atomicAdd(dst, acc);
#elif BWD
            *dst = expf(in_vals[(long long)b * in_stride + o] - fmaxf(max_val[b], -1e30f)) * acc;
#else
            *dst = in_nnz[b] == 0 ? __int_as_float(0xff800000) : logf(acc) + fmaxf(max_val[b], -1e30f);
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

#if BWD
// W in backward slab order: the slab of (child tile m, parent range s) is the KS x TMS elements at
// ((m * NSPLIT + s) * KS) * TMS, row p the tile's TMS children of parent s * KS + p; padding zero.
// Through shared memory, so that both the reads (parents contiguous in W) and the writes (children
// contiguous) coalesce. Grid (cdiv(H_in, TMS), NSPLIT * KS / 32), block 256.
extern "C" __global__ void __launch_bounds__(256)
ws_slab_wcopy_bwd(float* __restrict__ dst, const float* __restrict__ w, long long w_start)
{
    __shared__ float t[32][TMS + 1];
    const int m = blockIdx.x, j0 = blockIdx.y * 32;           // j: parent position in [0, NSPLIT * KS)
    for (int e = threadIdx.x; e < 32 * TMS; e += 256) {
        const int r = e / 32, pl = e - r * 32;
        const int p = j0 + pl, c = m * TMS + r;               // p < NSPLIT * KS; real parent p (ranges are contiguous)
        t[pl][r] = (c < H_IN && p < H_OUT) ? w[w_start + ((long long)(p / BS) * H_IN + c) * BS + (p % BS)] : 0.f;
    }
    __syncthreads();
    const int sp = j0 / KS;
    for (int e = threadIdx.x; e < 32 * TMS; e += 256) {
        const int pl = e / TMS, r = e - pl * TMS;
        const int pin = j0 - sp * KS + pl;                    // parent row within the slab
        WSTORE(dst, ((long long)(m * NSPLIT + sp) * KS + pin) * TMS + r, t[pl][r]);
    }
}

// Backward, NSPLIT > 1: flow_in[b, k] = exp(in[b, k] - max_b) * (sum of the
// partials), k < nnz_in[b]. Grid (cdiv(K_in, 256), B).
extern "C" __global__ void __launch_bounds__(256)
ws_slab_finalize_flow(float* __restrict__ flow, long long flow_stride, const int* __restrict__ in_nnz,
                      const float* __restrict__ in_vals, long long in_stride,
                      const float* __restrict__ max_val)
{
    const int b = blockIdx.y;
    const int k = blockIdx.x * 256 + threadIdx.x;
    if (k >= in_nnz[b]) return;
    float* dst = flow + (long long)b * flow_stride + k;
    *dst *= expf(in_vals[(long long)b * in_stride + k] - fmaxf(max_val[b], -1e30f));
}

// Backward, per step: xp[b, i] = (byte offset of parent h_i's row in its
// range's slab, g_i) with g = nflow * exp(max_b - nmars) the parent's flow
// scale (0 for a sample without children or a dead parent), for the output
// column of sample b; cofs[b, s] = the first i with h_i >= s * KS. Grid
// (cdiv(K_out + 1, 256), B).
extern "C" __global__ void __launch_bounds__(256)
ws_slab_pack_g(int2* __restrict__ xp, long long xp_stride, int* __restrict__ cofs,
               const long long* __restrict__ out_idx, const int* __restrict__ out_cs,
               const int* __restrict__ out_nnz, const float* __restrict__ nflow, long long flow_stride,
               const float* __restrict__ nmars, long long mars_stride,
               const float* __restrict__ max_val, const int* __restrict__ in_nnz)
{
    const int b = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int n = out_nnz[b];
    if (i > n) return;
    const long long* op = out_idx + out_cs[b];
    int* cb = cofs + (long long)b * (NSPLIT + 1);
    const int kp = i == 0 ? -1 : (int)(op[i - 1] / KS);
    const int h = i < n ? (int)op[i] : 0;
    const int kc = i < n ? h / KS : NSPLIT;
    for (int k = kp + 1; k <= kc; ++k) cb[k] = i;
    if (i < n) {
        const float nm = nmars[(long long)b * mars_stride + i];
        float g = nflow[(long long)b * flow_stride + i] * expf(fmaxf(max_val[b], -1e30f) - nm);
        if (in_nnz[b] == 0 || nm == __int_as_float(0xff800000)) g = 0.f;
        xp[(long long)b * xp_stride + i] = make_int2((h % KS) * (TMS * WB), __float_as_int(g));
    }
}
#endif
"""

_PFLOW_SRC = r"""
// Sparse parameter flows: S[h, c] += g[b, h] x[b, c] over the active (parent h, child c) pairs of every sample b of
// N stored steps; the caller multiplies S by W at the pass end. A CTA owns a tile of 4 parent rows x KS children in
// shared memory and adds the tile's list entries of all stored steps. The entries are sorted, in chunks of the CTA's
// threads, by the address of their sample's child list: the samples with the same next token share that list, so a
// run of equal addresses is a group whose products are summed in registers and added to the tile once per (child,
// active row) of the group. The warps share the tile, which is kept in 32-bit fixed point, each row scaled by
// 2^31 / G_r, G_r = the row's sum of g over the tile's entries (a pre-pass): every element's sum is at most G_r since
// x = exp(v - max) <= 1, so native integer shared adds (no reads, no compare-and-swaps; the result does not depend on
// the order of the adds; a term's rounding error is at most 2^-32 G_r). Pairs in the corner [0, pc) x [0, cc) are
// left out (the caller covers them). pf_bench2.py (2026-10-29, 32 steps per accumulation, ms per EM step): 8k B=1024
// 28.2 -> 15.8, 16k B=512 86.7 -> 51.9 against the version of 2026-10-28 (one-row copies of the item, branch-free
// adds, a table of the items' runs, 16-byte write-back, fixed point, 4 blocks per item).
#define FULL 0xffffffffu
#define CH (NW * 32)
#define LUTN 8192                                     // items per chunk in pf_tile's table of their runs

// the samples of every 4-row parent tile: ent[t, pos] = (b, ps << 4 | rows) (as ws_slab_lists)
extern "C" __global__ void __launch_bounds__(256)
pf_lists(int2* __restrict__ ent, int* __restrict__ cnt, const long long* __restrict__ out_idx,
         const int* __restrict__ out_cs, const int* __restrict__ out_nnz, int B)
{
    const int b = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x, n = out_nnz[b];
    if (i >= n) return;
    const long long* op = out_idx + out_cs[b];
    const int h = (int)op[i], t = h >> 2;
    if (i > 0 && (int)op[i - 1] >> 2 == t) return;
    unsigned rows = 1u << (h & 3);
    #pragma unroll
    for (int k = 1; k < 4; ++k) {
        if (i + k >= n) break;
        const int hk = (int)op[i + k];
        if (hk >> 2 != t) break;
        rows |= 1u << (hk & 3);
    }
    const int pos = atomicAdd(cnt + t, 1);
    ent[(long long)t * B + pos] = make_int2(b, (i << 4) | (int)rows);
}

// children: xo[b, j] = c % KS, xv[b, j] = exp(v - max_b); cofs[b, k] = the first j with c_j >= k KS (k = 0..NSPLIT);
// gkey[b] = the address of the sample's child list (equal for samples with the same list)
extern "C" __global__ void __launch_bounds__(256)
pf_pack_x(int* __restrict__ xo, float* __restrict__ xv, int* __restrict__ cofs, unsigned long long* __restrict__ gkey, int K,
          const long long* __restrict__ in_idx, const float* __restrict__ in_vals, long long in_stride,
          const int* __restrict__ in_cs, const int* __restrict__ in_nnz, const float* __restrict__ max_val)
{
    const int b = blockIdx.y, j = blockIdx.x * 256 + threadIdx.x, n = in_nnz[b];
    if (j > n) return;
    const long long* ip = in_idx + in_cs[b];
    if (j == 0) gkey[b] = (unsigned long long)ip;
    int* cb = cofs + (long long)b * (NSPLIT + 1);
    const int kp = j == 0 ? -1 : (int)(ip[j - 1] / KS);
    const int c = j < n ? (int)ip[j] : 0;
    const int kc = j < n ? c / KS : NSPLIT;
    for (int k = kp + 1; k <= kc; ++k) cb[k] = j;
    if (j < n) {
        xo[(long long)b * K + j] = c % KS;
        xv[(long long)b * K + j] = expf(in_vals[(long long)b * in_stride + j] - fmaxf(max_val[b], -1e30f));
    }
}

// parents: gp[b, i] = flow * exp(max_b - value) (0 for a sample without children or a dead parent)
extern "C" __global__ void __launch_bounds__(256)
pf_pack_g(float* __restrict__ gp, int K, const float* __restrict__ nflow, long long flow_stride,
          const float* __restrict__ nmars, long long mars_stride, const float* __restrict__ max_val,
          const int* __restrict__ in_nnz, const int* __restrict__ out_nnz)
{
    const int b = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i >= out_nnz[b]) return;
    const float nm = nmars[(long long)b * mars_stride + i];
    float g = nflow[(long long)b * flow_stride + i] * expf(fmaxf(max_val[b], -1e30f) - nm);
    if (in_nnz[b] == 0 || nm == __int_as_float(0xff800000)) g = 0.f;
    gp[(long long)b * K + i] = g;
}

// exclusive prefix sum of one value per thread over the CTA (all threads call it); *total: the sum
__device__ __forceinline__ int block_exscan(int v, int* s_wsum, int lane, int warp, int* total) {
    int x = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(FULL, x, o); if (lane >= o) x += y; }
    if (lane == 31) s_wsum[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < NW ? s_wsum[lane] : 0;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(FULL, w, o); if (lane >= o) w += y; }
        if (lane < NW) s_wsum[lane] = w;
    }
    __syncthreads();
    const int before = (warp ? s_wsum[warp - 1] : 0) + x - v;
    *total = s_wsum[NW - 1];
    __syncthreads();                                  // s_wsum is reused by the next scan
    return before;
}

// one item: the children jb + 32 k (k < IU) of the group of sorted entries t0..t1, multiplied by the members' g and
// added to the tile's rows of the group: RM, one row, at compile time (most groups have one active row in a tile of 4),
// else all 4 rows multiplied and the rows of rmv added. A lane without a child, or a row outside the group's, adds 0
// to the lane's own scratch word, so that no branch diverges.
template <unsigned RM>
__device__ __forceinline__ void pf_item(float* __restrict__ T, const unsigned long long* __restrict__ skey,
                                        const float4* __restrict__ sg, const int* __restrict__ sxo,
                                        const int* __restrict__ xo, const float* __restrict__ xv,
                                        int t0, int t1, int jb, int len, int xol, unsigned rmv, float* scr,
                                        float4 sc)
{
    const unsigned rm = RM ? RM : rmv;
    int off[IU];
    #pragma unroll
    for (int k = 0; k < IU; ++k) off[k] = jb + 32 * k < len ? xo[xol + jb + 32 * k] : -1;
    float acc[IU][4];
    #pragma unroll
    for (int k = 0; k < IU; ++k) { acc[k][0] = 0.f; acc[k][1] = 0.f; acc[k][2] = 0.f; acc[k][3] = 0.f; }
    for (int t = t0; t < t1; ++t) {
        const int mi = (int)(skey[t] & 2047);
        const float4 g = sg[mi];
        const int xom = sxo[mi];
        float xm[IU];
        #pragma unroll
        for (int k = 0; k < IU; ++k) xm[k] = off[k] >= 0 ? xv[xom + jb + 32 * k] : 0.f;
        #pragma unroll
        for (int k = 0; k < IU; ++k) {
            if (!RM || (RM & 1u)) acc[k][0] = fmaf(g.x, xm[k], acc[k][0]);
            if (!RM || (RM & 2u)) acc[k][1] = fmaf(g.y, xm[k], acc[k][1]);
            if (!RM || (RM & 4u)) acc[k][2] = fmaf(g.z, xm[k], acc[k][2]);
            if (!RM || (RM & 8u)) acc[k][3] = fmaf(g.w, xm[k], acc[k][3]);
        }
    }
    const float scv[4] = {sc.x, sc.y, sc.z, sc.w};
    #pragma unroll
    for (int k = 0; k < IU; ++k)
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            if (!RM || ((RM >> r) & 1u)) {
                const bool on = off[k] >= 0 && ((rm >> r) & 1u);
                atomicAdd(on ? reinterpret_cast<unsigned*>(T + r * KS + off[k]) : reinterpret_cast<unsigned*>(scr),
                          on ? __float2uint_rn(acc[k][r] * scv[r]) : 0u);
            }
}

// one row: a copy per row; more rows: at run time
__device__ __forceinline__ void pf_item_rows(unsigned rm, float* __restrict__ T,
                                             const unsigned long long* __restrict__ skey, const float4* __restrict__ sg,
                                             const int* __restrict__ sxo, const int* __restrict__ xo,
                                             const float* __restrict__ xv, int t0, int t1, int jb, int len, int xol,
                                             float* scr, float4 sc)
{
    switch (rm) {
        case 1: pf_item<1>(T, skey, sg, sxo, xo, xv, t0, t1, jb, len, xol, 0u, scr, sc); break;
        case 2: pf_item<2>(T, skey, sg, sxo, xo, xv, t0, t1, jb, len, xol, 0u, scr, sc); break;
        case 4: pf_item<4>(T, skey, sg, sxo, xo, xv, t0, t1, jb, len, xol, 0u, scr, sc); break;
        case 8: pf_item<8>(T, skey, sg, sxo, xo, xv, t0, t1, jb, len, xol, 0u, scr, sc); break;
        default: pf_item<0>(T, skey, sg, sxo, xo, xv, t0, t1, jb, len, xol, rm, scr, sc); break;
    }
}

// one CTA per (child range sp, parent tile m). Per chunk of the tile's entries (all stored steps in order): load,
// bitonic sort by (list address, position) of the chunk's entries, runs of equal address and length = groups (their
// rows: the union of the members'), items = (group, block of 32 IU children) handed out GI at a time through a shared
// counter, an item's run from a table (up to LUTN items per chunk; more: binary search from the warp's last run)
extern "C" __global__ void __launch_bounds__(NW * 32, MINB)
pf_tile(float* __restrict__ S, long long s_stride, const int2* __restrict__ ent, const int* __restrict__ cnt,
        const int* __restrict__ xo, const float* __restrict__ xv, const int* __restrict__ cofs,
        const float* __restrict__ gp, const unsigned long long* __restrict__ gkey, int nslot, int B, int Kx, int Kg,
        int n_m, float sign, int pc, int cc)
{
    if (blockIdx.y * 4 < pc && (blockIdx.x + 1) * KS <= cc) return;     // inside the caller's corner
    extern __shared__ __align__(16) float smem[];
    float* T = smem;                                                         // [4][KS]
    unsigned long long* skey = reinterpret_cast<unsigned long long*>(T + 4 * KS);   // [CH] (address >> 3) << 11 | pos
    float4* sg = reinterpret_cast<float4*>(skey + CH);                       // [CH] g of the entry's 4 rows
    int* sxo = reinterpret_cast<int*>(sg + CH);                              // [CH] xo / xv index of the range's first child
    int* slen = sxo + CH;                                                    // [CH] children in the range
    int* srows = slen + CH;                                                  // [CH] active rows
    int* sst = srows + CH;                                                   // [CH + 1] sorted start of run k
    int* soff = sst + CH + 1;                                                // [CH + 1] items before run k
    unsigned* srm = reinterpret_cast<unsigned*>(soff + CH + 1);              // [CH] rows of run k
    float* sscr = reinterpret_cast<float*>(srm + CH);                        // [CH] a scratch word per thread
    unsigned short* slut = reinterpret_cast<unsigned short*>(sscr + CH);     // [LUTN] run of item it
    __shared__ int s_ne[MAXSLOT + 1];
    __shared__ int s_wsum[32];
    __shared__ int s_next;
    __shared__ float s_red[4 * NW], s_sc[8];                                 // row sums of g, scales
    const int sp = blockIdx.x, m = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* scr = sscr + tid;
    if (tid == 0) {
        int q = 0;
        for (int s = 0; s < nslot; ++s) { s_ne[s] = q; q += cnt[(long long)s * n_m + m]; }
        s_ne[nslot] = q;
    }
    for (int e = tid; e < 4 * KS; e += CH) T[e] = 0.f;
    sscr[tid] = 0.f;
    __syncthreads();
    const int NE = s_ne[nslot];
    if (NE == 0) return;                                                     // CTA-uniform
    float4 sc = make_float4(0.f, 0.f, 0.f, 0.f);
    {   // G_r = the row's sum of g over the tile's entries: scale 2^31 / G_r (headroom 2: no element reaches 2^32)
        float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
        for (int q = tid; q < NE; q += CH) {
            int s = 0;
            while (q >= s_ne[s + 1]) ++s;
            const int2 e = ent[((long long)s * n_m + m) * B + (q - s_ne[s])];
            const int b = e.x;
            const unsigned rows = e.y & 15u;
            const int* cb = cofs + ((long long)s * B + b) * (NSPLIT + 1) + sp;
            if (cb[1] > cb[0]) {
                const float* gb = gp + ((long long)s * B + b) * Kg + (e.y >> 4);
                if (rows & 1u) g0 += gb[0];
                if (rows & 2u) g1 += gb[__popc(rows & 1u)];
                if (rows & 4u) g2 += gb[__popc(rows & 3u)];
                if (rows & 8u) g3 += gb[__popc(rows & 7u)];
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            g0 += __shfl_xor_sync(FULL, g0, o); g1 += __shfl_xor_sync(FULL, g1, o);
            g2 += __shfl_xor_sync(FULL, g2, o); g3 += __shfl_xor_sync(FULL, g3, o);
        }
        if (lane == 0) { s_red[4 * warp] = g0; s_red[4 * warp + 1] = g1; s_red[4 * warp + 2] = g2; s_red[4 * warp + 3] = g3; }
        __syncthreads();
        if (tid < 4) {
            float G = 0.f;
            for (int w = 0; w < NW; ++w) G += s_red[4 * w + tid];
            G *= 1.0001f;                                 // the rounding of the sum
            s_sc[tid] = G > 0.f ? 2147483648.f / G : 0.f;
            s_sc[4 + tid] = G / 2147483648.f;
        }
        __syncthreads();
        sc = make_float4(s_sc[0], s_sc[1], s_sc[2], s_sc[3]);
    }
    for (int c0 = 0; c0 < NE; c0 += CH) {
        unsigned long long key = ~0ull;
        const int q = c0 + tid;
        if (q < NE) {
            int s = 0;
            while (q >= s_ne[s + 1]) ++s;
            const int2 e = ent[((long long)s * n_m + m) * B + (q - s_ne[s])];
            const int b = e.x;
            const unsigned rows = e.y & 15u;
            const int* cb = cofs + ((long long)s * B + b) * (NSPLIT + 1) + sp;
            const int j0 = cb[0], j1 = cb[1];
            if (j1 > j0) {
                const float* gb = gp + ((long long)s * B + b) * Kg + (e.y >> 4);
                float4 g = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rows & 1u) g.x = gb[0];
                if (rows & 2u) g.y = gb[__popc(rows & 1u)];
                if (rows & 4u) g.z = gb[__popc(rows & 3u)];
                if (rows & 8u) g.w = gb[__popc(rows & 7u)];
                sg[tid] = g;
                sxo[tid] = (s * B + b) * Kx + j0;
                slen[tid] = j1 - j0;
                srows[tid] = (int)rows;
                key = ((gkey[(long long)s * B + b] >> 3) << 11) | (unsigned long long)tid;
            }
        }
        skey[tid] = key;
        __syncthreads();
        const int nc = NE - c0 < CH ? NE - c0 : CH;      // sort the chunk's entries only (next power of two)
        int n2 = 2;
        while (n2 < nc) n2 <<= 1;
        for (int k = 2; k <= n2; k <<= 1)
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int p = tid ^ j;
                if (p > tid && p < n2) {
                    const unsigned long long a = skey[tid], c = skey[p];
                    if ((a > c) == ((tid & k) == 0)) { skey[tid] = c; skey[p] = a; }
                }
                __syncthreads();
            }
        // runs: equal list address and length
        const unsigned long long a = skey[tid];
        const bool valid = a != ~0ull;
        bool start = false;
        if (valid) {
            start = tid == 0;
            if (!start) {
                const unsigned long long pa = skey[tid - 1];
                start = (pa >> 11) != (a >> 11) || slen[pa & 2047] != slen[a & 2047];
            }
        }
        int nrun = 0, nvalid = 0, nitem = 0;
        const int rk = block_exscan(start ? 1 : 0, s_wsum, lane, warp, &nrun);
        block_exscan(valid ? 1 : 0, s_wsum, lane, warp, &nvalid);
        if (start) sst[rk] = tid;
        if (tid == 0) { sst[nrun] = nvalid; s_next = 0; }
        __syncthreads();
        int ni = 0;
        if (tid < nrun) {
            const int t0 = sst[tid], t1 = sst[tid + 1];
            ni = (slen[skey[t0] & 2047] + 32 * IU - 1) / (32 * IU);
            unsigned rmu = 0;
            for (int t = t0; t < t1; ++t) rmu |= (unsigned)srows[skey[t] & 2047];
            srm[tid] = rmu;
        }
        const int io = block_exscan(ni, s_wsum, lane, warp, &nitem);
        if (tid < nrun) soff[tid] = io;
        if (tid == 0) soff[nrun] = nitem;
        if (nitem <= LUTN && tid < nrun)                  // the run of every item (CTA-uniform condition)
            for (int it = io; it < io + ni; ++it) slut[it] = (unsigned short)tid;
        __syncthreads();
        int lo = 0;                                       // the warp's current run
        while (true) {
            int it0 = 0;
            if (lane == 0) it0 = atomicAdd(&s_next, GI);
            it0 = __shfl_sync(FULL, it0, 0);
            if (it0 >= nitem) break;
            const int itn = it0 + GI < nitem ? it0 + GI : nitem;
            if (nitem <= LUTN) {
                for (int it = it0; it < itn; ++it) {
                    const int k = slut[it];
                    const int t0 = sst[k], t1 = sst[k + 1];
                    const int lead = (int)(skey[t0] & 2047);
                    pf_item_rows(srm[k], T, skey, sg, sxo, xo, xv, t0, t1, (it - soff[k]) * 32 * IU + lane, slen[lead],
                                 sxo[lead], scr, sc);
                }
                continue;
            }
            {                                             // the run of it0: the other warps took the items in between
                int lb = lo, ub = nrun;
                while (ub - lb > 1) { const int mid = (lb + ub) >> 1; if (soff[mid] <= it0) lb = mid; else ub = mid; }
                lo = lb;
            }
            for (int it = it0; it < itn; ++it) {
                while (soff[lo + 1] <= it) ++lo;
                const int t0 = sst[lo], t1 = sst[lo + 1];
                const int lead = (int)(skey[t0] & 2047);
                pf_item_rows(srm[lo], T, skey, sg, sxo, xo, xv, t0, t1, (it - soff[lo]) * 32 * IU + lane, slen[lead],
                             sxo[lead], scr, sc);
            }
        }
        __syncthreads();                                  // before the next chunk reuses the arrays
    }
    // S += sign T (T u32, times G_r / 2^31): 16-byte reads and writes of 4 consecutive children where S's rows
    // allow it (4 per thread in flight, groups of 4 zeros skipped), else 4 single values per thread in flight
#if H_IN % 4 == 0
    const bool vec = (((unsigned long long)S | (unsigned long long)(s_stride * 4)) & 15ull) == 0;
#else
    const bool vec = false;
#endif
    if (vec) {
        for (int q0 = tid; q0 < KS; q0 += 4 * CH) {
            float4 tv[4], sv[4];
            float4* ad[4];
            bool ok[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int q = q0 + u * CH, r = q / (KS / 4), c4 = q - r * (KS / 4), h = m * 4 + r, col = sp * KS + 4 * c4;
                ok[u] = q < KS && h < H_OUT && col < H_IN;
                const uint4 tu = ok[u] ? reinterpret_cast<const uint4*>(T)[q] : make_uint4(0u, 0u, 0u, 0u);
                const float f = s_sc[4 + (r & 3)];
                tv[u] = make_float4(__uint2float_rn(tu.x) * f, __uint2float_rn(tu.y) * f, __uint2float_rn(tu.z) * f,
                                    __uint2float_rn(tu.w) * f);
                ok[u] = ok[u] && (tv[u].x != 0.f || tv[u].y != 0.f || tv[u].z != 0.f || tv[u].w != 0.f);
                ad[u] = reinterpret_cast<float4*>(S + (long long)h * s_stride + col);
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) if (ok[u]) sv[u] = *ad[u];
            #pragma unroll
            for (int u = 0; u < 4; ++u)
                if (ok[u]) *ad[u] = make_float4(sv[u].x + sign * tv[u].x, sv[u].y + sign * tv[u].y,
                                                sv[u].z + sign * tv[u].z, sv[u].w + sign * tv[u].w);
        }
    } else {
        for (int e0 = tid; e0 < 4 * KS; e0 += 4 * CH) {
            float tv[4], sv[4];
            long long ad[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int e = e0 + u * CH, r = e / KS, c = e - r * KS, h = m * 4 + r, col = sp * KS + c;
                tv[u] = (e < 4 * KS && h < H_OUT && col < H_IN)
                    ? __uint2float_rn(reinterpret_cast<const unsigned*>(T)[e]) * s_sc[4 + (r & 3)] : 0.f;
                ad[u] = (long long)h * s_stride + col;
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) sv[u] = tv[u] != 0.f ? S[ad[u]] : 0.f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) if (tv[u] != 0.f) S[ad[u]] = sv[u] + sign * tv[u];
        }
    }
}
"""
_PFLOW_KERNELS: Dict[tuple, tuple] = {}
_PFLOW_MAXSLOT = 32
"""Stored steps one accumulation can take (pf_tile's shared slot offsets)."""


def pflow_tile_shape(H_in: int, H_out: int, slab_bytes: int = 128 * 1024) -> Tuple[int, int, int]:
    """``(KS, NSPLIT, n_m)`` of the parameter-flow tiles: 4 parent rows x ``KS`` children (fp32 within
    ``slab_bytes``), ``NSPLIT`` ranges of children, ``n_m`` parent tiles."""
    KS, nsplit = slab_shape(H_in, 4, slab_bytes)
    assert KS % 32 == 0, KS
    return KS, nsplit, -(-H_out // 4)


def pflow_tile_iu(H_in: int, H_out: int) -> int:
    """Blocks of 32 children per item of ``pf_tile`` (pf_bench2 2026-10-29: 4 blocks with 2 items per counter step
    beat 2 x 4, 3 x 4 and 4 x 4 on 8k B=512..2048 and 16k B=256..1024)."""
    return 4


def pflow_tile_smem(KS: int, NW: int) -> int:
    """Shared memory of the tile kernel: the fp32 tile (4 rows x ``KS``), the sort chunk of ``32 NW`` entries, the
    scan arrays and the item table."""
    ch = NW * 32
    return 4 * KS * 4 + ch * (8 + 16 + 3 * 4) + 2 * (ch + 1) * 4 + 2 * ch * 4 + 8192 * 2


_PF_TILE_BYTES: dict = {}


def pflow_tile_bytes(H_in: int, H_out: int, device, want: int, NW: int = 32) -> int:
    """The tile size for ``device``: ``want`` (``_WS_SPARSE_PF_TILE_BYTES``, 128 KB by default) halved until the
    tile kernel's shared memory (:func:`pflow_tile_smem`) fits the device (the H100 takes 128 KB tiles, the A100
    64 KB)."""
    limit = smem_limit(device)
    key = (H_in, H_out, want, NW, limit)
    hit = _PF_TILE_BYTES.get(key)
    if hit is None:
        hit = want
        while hit > 4096 and pflow_tile_smem(pflow_tile_shape(H_in, H_out, hit)[0], NW) > limit:
            hit //= 2
        _PF_TILE_BYTES[key] = hit
    return hit


def pflow_tile_kernels(H_in: int, H_out: int, NW: int = 32, min_blocks: int = 1,
                       slab_bytes: int = 128 * 1024) -> tuple:
    """``(lists, pack_x, pack_g, tile kernel, tile smem bytes)`` for ``W [H_out, H_in]``; ``NW`` warps per tile
    (the sort chunk: ``32 NW`` entries)."""
    assert 1 <= NW <= 32 and NW & (NW - 1) == 0, NW
    KS, nsplit, _ = pflow_tile_shape(H_in, H_out, slab_bytes)
    iu = pflow_tile_iu(H_in, H_out)
    key = (H_in, H_out, NW, min_blocks, KS, iu)
    hit = _PFLOW_KERNELS.get(key)
    if hit is None:
        smem = pflow_tile_smem(KS, NW)
        assert smem <= SMEM_LIMIT, (smem, KS, NW)
        # iu blocks of 32 children per item, 2 items per counter step (pf_bench2 2026-10-29; 64 KB tiles with 16
        # warps, 2 per SM, are slower)
        defines = dict(H_IN=H_in, H_OUT=H_out, KS=KS, NSPLIT=nsplit, NW=NW, MINB=min_blocks, IU=iu, GI=2,
                       MAXSLOT=_PFLOW_MAXSLOT)
        src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + _PFLOW_SRC
        hit = (CudaKernel(src, "pf_lists", ["ptr", "ptr", "ptr", "ptr", "ptr", "i32"]),
               CudaKernel(src, "pf_pack_x", ["ptr", "ptr", "ptr", "ptr", "i32", "ptr", "ptr", "i64", "ptr", "ptr",
                                             "ptr"]),
               CudaKernel(src, "pf_pack_g", ["ptr", "i32", "ptr", "i64", "ptr", "i64", "ptr", "ptr", "ptr"]),
               CudaKernel(src, "pf_tile", ["ptr", "i64", "ptr", "ptr", "ptr", "ptr", "ptr", "ptr", "ptr", "i32",
                                           "i32", "i32", "i32", "i32", "f32", "i32", "i32"], max_shared=smem), smem)
        _PFLOW_KERNELS[key] = hit
    return hit


def pflow_tile_stash(bufs: dict, slot: int, sv_in, sv_out, sv_flow_out, max_val: torch.Tensor, B: int,
                     H_in: int, H_out: int, slab_bytes: int = 128 * 1024) -> None:
    """Pack one step's parameter-flow inputs into stash ``slot`` of ``bufs`` (``ent [S, n_m, B, 2]``,
    ``cnt [S, n_m]``, ``xo [S, B, Kx]``, ``xv [S, B, Kx]`` (float32), ``cofs [S, B, NSPLIT + 1]``, ``gkey [S, B]``
    (int64), ``gp [S, B, Kg]``): the samples of every parent tile, the children (offset in the range, x), their range
    starts and the address of the child list, the parents' g. Graph-capturable, no host sync."""
    lists, pack_x, pack_g, _, _ = pflow_tile_kernels(H_in, H_out, slab_bytes=slab_bytes)
    ent, cnt, xo, xv, cofs, gkey, gp = (bufs[k][slot] for k in ("ent", "cnt", "xo", "xv", "cofs", "gkey", "gp"))
    Kx, Kg = xo.shape[1], gp.shape[1]
    assert sv_in.total_nnz <= Kx and sv_out.total_nnz <= Kg
    device = gp.device
    cnt.zero_()
    lists((-(-sv_out.total_nnz // 256), B), (256,),
          [ent, cnt, sv_out.indices, sv_out.col_starts, sv_out.nnz, B], device=device)
    pack_x((-(-(sv_in.total_nnz + 1) // 256), B), (256,),
           [xo, xv, cofs, gkey, Kx, sv_in.indices, sv_in.values, sv_in.values.stride(0), sv_in.col_starts,
            sv_in.nnz, max_val], device=device)
    pack_g((-(-sv_out.total_nnz // 256), B), (256,),
           [gp, Kg, sv_flow_out.values, sv_flow_out.values.stride(0), sv_out.values, sv_out.values.stride(0),
            max_val, sv_in.nnz, sv_out.nnz], device=device)


def pflow_tile_flush(S: torch.Tensor, bufs: dict, nslot: int, B: int, H_in: int, H_out: int,
                     sign: float = 1.0, NW: int = 32, min_blocks: int = 1, slab_bytes: int = 128 * 1024,
                     corner: Tuple[int, int] = (0, 0)) -> None:
    """``S [H_out, H_in] += sign * sum over the first nslot stored steps of g x^T`` (active pairs), except the
    pairs with parent < ``corner[0]`` and child < ``corner[1]`` (a multiple of the tile width). The tiles are
    kept in 32-bit fixed point, which needs ``x = exp(v - max_val) <= 1``, i.e. ``max_val`` at least the sample's
    largest input value, as the upstream product layers give it."""
    if nslot == 0:
        return
    assert nslot <= _PFLOW_MAXSLOT
    _, _, _, tile, smem = pflow_tile_kernels(H_in, H_out, NW, min_blocks, slab_bytes)
    KS, nsplit, n_m = pflow_tile_shape(H_in, H_out, slab_bytes)
    assert corner[1] % KS == 0 or corner[1] >= H_in, (corner, KS)
    ent, cnt, xo, xv, cofs, gkey, gp = (bufs[k] for k in ("ent", "cnt", "xo", "xv", "cofs", "gkey", "gp"))
    assert xo.shape[0] * B * xo.shape[2] < 1 << 31, "stash too large for 32-bit child offsets"
    tile((nsplit, n_m), (NW * 32,),
         [S, S.stride(0), ent, cnt, xo, xv, cofs, gp, gkey, nslot, B, xo.shape[2], gp.shape[2], n_m, float(sign),
          int(corner[0]), int(corner[1])], shared_mem=smem, device=S.device)


_SLAB_MAIN_ARGS = ["ptr", "i64", "ptr", "i64", "ptr", "i32", "ptr", "ptr", "ptr", "ptr", "ptr", "i32", "ptr", "i64"]
_SLAB_FIN_FLOW_ARGS = ["ptr", "i64", "ptr", "ptr", "i64", "ptr"]
_SLAB_PACK_G_ARGS = ["ptr", "i64", "ptr", "ptr", "ptr", "ptr", "ptr", "i64", "ptr", "i64", "ptr", "ptr"]
_SLAB_LIST_ARGS = ["ptr", "ptr", "ptr", "ptr", "ptr", "i32", "ptr", "i64"]
_SLAB_FIN_ARGS = ["ptr", "i64", "ptr", "ptr", "ptr"]
_SLAB_KERNELS: Dict[tuple, tuple] = {}
_PACK_ARGS = ["ptr", "i64", "ptr", "ptr", "ptr", "i64", "ptr", "ptr", "ptr"]

TILE_HEIGHTS = (4, 8, 16)
"""Parent rows of a slab tile the kernels are built for."""


def slab_shape(H_in: int, TMS: int, slab_bytes: int) -> Tuple[int, int]:
    """``(KS, NSPLIT)``: the fewest ranges of ``KS`` children (a multiple of
    32, or of 256 from 256 on) whose ``KS x TMS`` fp32 slab stays within
    ``slab_bytes``, balanced over ``H_in``. (A bf16 slab of the same shape
    takes half of it: the ranges are the same in both precisions, so the
    forward and backward copies of a pass keep one layout.)"""
    max_rows = max(32, slab_bytes // (TMS * 4))
    nsplit = -(-H_in // max_rows)
    while True:
        rows = -(-H_in // nsplit)
        unit = 256 if rows >= 256 else 32
        KS = -(-rows // unit) * unit
        if KS <= max_rows or unit == 32:
            return KS, -(-H_in // KS)
        nsplit += 1


def slab_smem(KS: int, TMS: int, w16: bool = False) -> int:
    """Dynamic shared memory of a slab CTA: the slab and its barrier."""
    return KS * TMS * (2 if w16 else 4) + 16


def slab_kernels(H_in: int, H_out: int, BS: int, TMS: int, KS: int, NWARP: int,
                 min_blocks: int, bwd: bool = False, w16: bool = False) -> tuple:
    """``(lists, pack, main, finalize, main smem bytes)`` of a block shape: the
    forward (tiles of parents, ranges of children) or with ``bwd`` the
    element-flow backward (tiles of children, ranges of parents); compiled on
    first use (cubins cached on disk). ``w16``: the slabs in bf16."""
    assert TMS in TILE_HEIGHTS, TMS
    w16 = bool(w16)
    key = (H_in, H_out, BS, TMS, KS, NWARP, min_blocks, bwd, w16)
    hit = _SLAB_KERNELS.get(key)
    if hit is None:
        smem = slab_smem(KS, TMS, w16)
        n_split = -(-(H_out if bwd else H_in) // KS)
        defines = dict(H_IN=H_in, H_OUT=H_out, BS=BS, TMS=TMS, LOG_TMS=TMS.bit_length() - 1, KS=KS,
                       NWARP=NWARP, MINB=min_blocks, BWD=int(bwd), W16=int(w16))
        src = ("".join(f"#define {k} {v}\n" for k, v in defines.items())
               + f"#define NSPLIT_OVERRIDE {n_split}\n" + _SLAB_SRC)
        main = CudaKernel(src, "ws_slab_bwd" if bwd else "ws_slab_fwd", _SLAB_MAIN_ARGS, max_shared=smem)
        if bwd:
            hit = (CudaKernel(src, "ws_slab_lists", _SLAB_LIST_ARGS),
                   CudaKernel(src, "ws_slab_pack_g", _SLAB_PACK_G_ARGS), main,
                   CudaKernel(src, "ws_slab_finalize_flow", _SLAB_FIN_FLOW_ARGS), smem)
        else:
            hit = (CudaKernel(src, "ws_slab_lists", _SLAB_LIST_ARGS),
                   CudaKernel(src, "ws_slab_pack", _PACK_ARGS), main,
                   CudaKernel(src, "ws_slab_finalize", _SLAB_FIN_ARGS), smem)
        _SLAB_KERNELS[key] = hit
    return hit


_WCOPY_KERNELS: Dict[tuple, CudaKernel] = {}


def slab_copy_numel(H_in: int, H_out: int, TMS: int, slab_bytes: int, bwd: bool = False) -> int:
    """Elements of the slab-order copy of W (:func:`slab_wcopy`): forward tiles of parents and
    ranges of children, or with ``bwd`` tiles of children and ranges of parents."""
    H_t, H_r = (H_in, H_out) if bwd else (H_out, H_in)
    KS, nsplit = slab_shape(H_r, TMS, slab_bytes)
    return -(-H_t // TMS) * TMS * nsplit * KS


def slab_wcopy(dst: torch.Tensor, w: torch.Tensor, w_start: int, H_in: int, H_out: int, BS: int,
               TMS: int, slab_bytes: int, bwd: bool = False, w16: bool = False) -> None:
    """Write W (block layout of :class:`SparseIOSumLayer` at ``w[w_start:]``) into ``dst`` in
    slab order: one contiguous ``[KS, TMS]`` slab per (parent tile, child range) for the
    forward, per (child tile, parent range) with ``bwd``, as the bulk loader of
    :func:`ws_slab_forward` / :func:`ws_slab_backward` reads it. One pass over W. With ``w16``
    the copy is bf16 (``dst`` of dtype ``torch.bfloat16``, the same number of elements), read
    by the kernels built with ``w16``."""
    assert TMS in TILE_HEIGHTS, TMS
    H_r = H_out if bwd else H_in
    KS, nsplit = slab_shape(H_r, TMS, slab_bytes)
    w16 = bool(w16)
    key = (H_in, H_out, BS, TMS, KS, bwd, w16)
    k = _WCOPY_KERNELS.get(key)
    if k is None:
        defines = dict(H_IN=H_in, H_OUT=H_out, BS=BS, TMS=TMS, LOG_TMS=TMS.bit_length() - 1, KS=KS,
                       NWARP=4, MINB=1, BWD=int(bwd), W16=int(w16))
        src = ("".join(f"#define {k_} {v}\n" for k_, v in defines.items())
               + f"#define NSPLIT_OVERRIDE {nsplit}\n" + _SLAB_SRC)
        k = _WCOPY_KERNELS[key] = CudaKernel(src, "ws_slab_wcopy_bwd" if bwd else "ws_slab_wcopy",
                                             ["ptr", "ptr", "i64"])
    n = slab_copy_numel(H_in, H_out, TMS, slab_bytes, bwd)
    assert dst.numel() >= n and dst.dtype == (torch.bfloat16 if w16 else torch.float32)
    assert (nsplit * KS) % 32 == 0 and KS % 32 == 0
    if bwd:
        k((-(-H_in // TMS), nsplit * KS // 32), (256,), [dst, w, w_start], device=dst.device)
    else:
        k((-(-H_out // 32), nsplit * KS // 32), (256,), [dst, w, w_start], device=dst.device)


def ws_slab_forward(out_vals: torch.Tensor, wslab: torch.Tensor, sv_in, sv_out,
                    max_val: torch.Tensor, ent: torch.Tensor, cnt: torch.Tensor,
                    xp: torch.Tensor, cofs: torch.Tensor, B: int, H_in: int, H_out: int, BS: int,
                    TMS: int = 4, slab_bytes: int = 64 * 1024, NWARP: int = 16,
                    min_blocks: int = 3, w16: bool = False) -> None:
    """Build the per-tile sample lists, pack the input columns and run the
    slab forward for one block, writing ``out_vals[b, i]`` (row stride
    ``out_vals.stride(0)``) for every active parent ``i`` of every sample;
    ``wslab`` holds W in forward slab order (:func:`slab_wcopy` with the same
    ``TMS`` / ``slab_bytes`` / ``w16``). Scratch: ``ent`` (int32
    ``[cdiv(H_out, TMS), 2 B]``), ``cnt`` (int32 ``[cdiv(H_out, TMS)]``),
    ``xp`` (int32 ``[B, >= 2 K_in]``), ``cofs`` (int32 ``[B, NSPLIT + 1]``,
    :func:`slab_shape`). Graph-capturable, no host sync."""
    assert TMS in TILE_HEIGHTS, TMS
    KS, nsplit = slab_shape(H_in, TMS, slab_bytes)
    assert sv_out.values.shape[-1] < 1 << (31 - TMS), "output columns too long for the list entries"
    n_m = -(-H_out // TMS)
    w16 = bool(w16)
    assert wslab.dtype == (torch.bfloat16 if w16 else torch.float32)
    lists, pack, fwd, fin, smem = slab_kernels(H_in, H_out, BS, TMS, KS, NWARP, min_blocks, w16=w16)
    K = sv_in.total_nnz
    assert xp.stride(0) >= 2 * K and xp.stride(0) % 2 == 0 and cofs.stride(0) == nsplit + 1
    assert ent.shape[0] >= n_m and ent.stride(0) == 2 * B and cnt.numel() >= n_m
    device = out_vals.device
    cnt.zero_()
    lists((-(-sv_out.total_nnz // 256), B), (256,),
          [ent, cnt, sv_out.indices, sv_out.col_starts, sv_out.nnz, B, out_vals, out_vals.stride(0)],
          device=device)
    pack((-(-(K + 1) // 256), B), (256,),
         [xp, xp.stride(0) // 2, cofs, sv_in.indices, sv_in.values, sv_in.values.stride(0),
          sv_in.col_starts, sv_in.nnz, max_val], device=device)
    fwd((nsplit, n_m), (NWARP * 32,),
        [out_vals, out_vals.stride(0), wslab, 0, xp, xp.stride(0) // 2, cofs, max_val, sv_in.nnz,
         ent, cnt, B, max_val, 0], shared_mem=smem, device=device)
    if nsplit > 1:
        fin((-(-sv_out.total_nnz // 256), B), (256,),
            [out_vals, out_vals.stride(0), sv_out.nnz, sv_in.nnz, max_val], device=device)


def ws_slab_backward(flow_vals: torch.Tensor, wslab: torch.Tensor, sv_in, sv_out,
                     sv_flow_out, max_val: torch.Tensor, ent: torch.Tensor, cnt: torch.Tensor,
                     xp: torch.Tensor, cofs: torch.Tensor, B: int, H_in: int, H_out: int, BS: int,
                     TMS: int = 4, slab_bytes: int = 64 * 1024, NWARP: int = 16,
                     min_blocks: int = 3, w16: bool = False) -> None:
    """Element flows of one block: ``flow_vals[b, k] = exp(in[b, k] - max_b)
    * sum_h g[b, h] W[h, c_k]`` over the sample's active parents ``h`` (``g =
    nflow * exp(max_b - nmars)``), for every active child ``c_k`` — the slab
    forward's mirror: tiles of ``TMS`` children, the parents in ranges whose
    ``[KS, TMS]`` slab fits ``slab_bytes``; ``wslab`` holds W in backward slab
    order (:func:`slab_wcopy` with ``bwd``, the same ``TMS`` / ``slab_bytes`` /
    ``w16``). Scratch: ``ent`` (int32 ``[cdiv(H_in, TMS), 2 B]``), ``cnt``
    (int32 ``[cdiv(H_in, TMS)]``), ``xp`` (int32 ``[B, >= 2 K_out]``), ``cofs``
    (int32 ``[B, NSPLIT + 1]`` over the parents). Graph-capturable, no host
    sync."""
    assert TMS in TILE_HEIGHTS, TMS
    KS, nsplit = slab_shape(H_out, TMS, slab_bytes)
    assert sv_in.values.shape[-1] < 1 << (31 - TMS), "input columns too long for the list entries"
    n_t = -(-H_in // TMS)
    w16 = bool(w16)
    assert wslab.dtype == (torch.bfloat16 if w16 else torch.float32)
    lists, pack, main, fin, smem = slab_kernels(H_in, H_out, BS, TMS, KS, NWARP, min_blocks, bwd=True, w16=w16)
    K = sv_out.total_nnz
    assert xp.stride(0) >= 2 * K and xp.stride(0) % 2 == 0 and cofs.stride(0) == nsplit + 1
    assert ent.shape[0] >= n_t and ent.stride(0) == 2 * B and cnt.numel() >= n_t
    device = flow_vals.device
    cnt.zero_()
    lists((-(-sv_in.total_nnz // 256), B), (256,),
          [ent, cnt, sv_in.indices, sv_in.col_starts, sv_in.nnz, B, flow_vals, flow_vals.stride(0)],
          device=device)
    pack((-(-(K + 1) // 256), B), (256,),
         [xp, xp.stride(0) // 2, cofs, sv_out.indices, sv_out.col_starts, sv_out.nnz,
          sv_flow_out.values, sv_flow_out.values.stride(0), sv_out.values, sv_out.values.stride(0),
          max_val, sv_in.nnz], device=device)
    main((nsplit, n_t), (NWARP * 32,),
         [flow_vals, flow_vals.stride(0), wslab, 0, xp, xp.stride(0) // 2, cofs, max_val, sv_in.nnz,
          ent, cnt, B, sv_in.values, sv_in.values.stride(0)], shared_mem=smem, device=device)
    if nsplit > 1:
        fin((-(-sv_in.total_nnz // 256), B), (256,),
            [flow_vals, flow_vals.stride(0), sv_in.nnz, sv_in.values, sv_in.values.stride(0), max_val],
            device=device)
