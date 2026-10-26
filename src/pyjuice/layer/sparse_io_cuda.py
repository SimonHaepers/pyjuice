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
  shared memory (64 KB, three CTAs per SM); a CTA loads one slab and keeps it
  for its whole life, so a step reads W from HBM once. The slabs come from a
  per-pass copy of W in slab order (:func:`slab_wcopy`, contiguous 64 KB bulk
  copies, 8 / 16-row slabs stored with the 32B / 64B swizzle so that a warp's
  16-byte reads of random rows spread over all bank groups), or with TMA boxes
  from W itself;
* the samples with an active parent in the tile (lists built on the device
  by :data:`ws_slab_lists`, one atomic per sample and tile) are handed to the
  warps round-robin (or through a shared counter); ``GL`` lanes take every
  ``GL``-th child of the sample in the range from its packed input column
  (:data:`ws_slab_pack`: per child the byte offset of its slab row and
  ``exp(v - max)``), read the 16-byte rows of the 4-row quarters of the tile
  that hold a parent of the sample, and accumulate;
* the lanes' sums meet in a transpose reduction and write the packed output
  values of the sample's parents in the tile directly (with several ranges
  the linear partial sums add up in the output, and :data:`ws_slab_finalize`
  takes the log; with three or more ranges the order of those atomic
  additions, and so the last bits of a result, varies between runs).

The element-flow backward, ``flow_in[b, c] = x[b, c] * sum_h g[b, h] W[h, c]``
over the sample's active parents, is the same kernel compiled with ``BWD``:
tiles of children, ranges of parents (from the backward slab-order copy,
``[KS parents][TMS children]``, or TMA boxes of 256 parents, which are
contiguous in W, by the tile's children), lanes over the sample's parents
packed as (slab offset, ``g``) by :data:`ws_slab_pack_g`. With ``PF`` it also
accumulates the slab's parameter flows ``g W x`` over the active pairs in
shared memory (fp32, or fixed point with native integer atomics) and adds
them to the parameter flows; one shared atomic per active pair costs about
2.5x the element-flow walk, so the deferred dense GEMM stays the default
(2026-10-26).

Measurements: 2026-10-24 (first slab kernels) and 2026-10-26 (slab-order
copies, swizzle, static items, fused parameter flows) in prune-pc's
``results/sparse_io_paths``. A first design that streamed the W of 32-row
tiles through a ring of shared-memory chunks lost to the slab: the work per
(sample, chunk) was a few dozen products against ~60 instructions of
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
#define NSPLIT NSPLIT_OVERRIDE
// GL lanes walk one item (32 / GL items per warp)
#ifndef GL
#define GL 32
#endif
// PF (backward): also accumulate the parameter flows of the CTA's slab,
// S[h, c] += g[b, h] W[h, c] x[b, c] over the active pairs, in shared memory
// next to the W slab, and add S to the param flows when the CTA ends.
// PF 1: fp32 atomicAdd (a CAS loop on shared memory). PF 2: fixed point, each
// pair flow (in [0, 1]) as round(v 2^32) split into an 18-bit low part and the
// rest, added with native 32-bit shared atomics to two u32 arrays (no CAS, no
// return value; deterministic; resolution 2^-32; < 2^14 adds per element and CTA)
#ifndef PF
#define PF 0
#endif
// SWZ (TMS 8 / 16, slab-order copies): the slab rows are stored with the 32B / 64B swizzle
// (CuTe Swizzle<1,4,3> / <2,4,3>: byte-address bits [4, 4 + b) ^= bits [7, 7 + b)), so the
// 16-byte quarter q of row c sits at chunk q ^ ((c * TMS * 4 >> 7) & SWM) of its row
#ifndef SWZ
#define SWZ 0
#endif
// STATIC: warp w takes the items w * IPW.., then (w + NWARP) * IPW.. (no shared counter)
#ifndef STATIC
#define STATIC 0
#endif
#define SWM (TMS == 16 ? 3 : 1)
// BNAT: backward slabs from the slab-order copy (LOADER 9), [KS parents][TMS children]
// (children contiguous, swizzled with SWZ) instead of TMA boxes [TMS children][BOXW parents]
#define BNAT (BWD && LOADER == 9)
#define LOG_GL (GL >= 32 ? 5 : GL >= 16 ? 4 : GL >= 8 ? 3 : GL >= 4 ? 2 : GL >= 2 ? 1 : 0)
#define IPW (32 / GL)
#define LT (LOG_TMS < LOG_GL ? LOG_TMS : LOG_GL)
#if BWD
#define SLAB_MAIN ws_slab_bwd
#else
#define SLAB_MAIN ws_slab_fwd
#endif
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

#if !BWD
// W in forward slab order (LOADER 9): the slab of (tile m, range s) is the KS x TMS floats at
// ((m * NSPLIT + s) * KS) * TMS, row c the tile's TMS parents of child s * KS + c (with SWZ
// in the swizzled chunk order the reads expect); padding is zero. Through shared memory:
// a block reads 32 children x 32 parents (128-byte rows of W, parents contiguous) and writes
// the 32 / TMS tiles' slab rows contiguously. Grid (cdiv(H_out, 32), NSPLIT * KS / 32), block 256.
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
        // e = (tile in the block, child, row): each tile's 32 rows of TMS floats are contiguous
        const int tb = e / (32 * TMS), rem = e - tb * (32 * TMS), cl = rem / TMS, r = rem - cl * TMS;
        const int m = (h0 + tb * TMS) / TMS, cin = j0 - sp * KS + cl;
        const int rr = SWZ ? r ^ ((((cin * TMS * 4) >> 7) & SWM) << 2) : r;
        if (m * TMS < ((H_OUT + TMS - 1) / TMS) * TMS)
            dst[((long long)(m * NSPLIT + sp) * KS + cin) * TMS + rr] = t[cl][tb * TMS + r];
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

#if BWD && !BNAT
// backward slab: the TMS rows (children) of a parent are BOXW floats apart
#define QUARTER(wv, base, q) float4 wv; { const char* qp_ = (base) + 4 * (q) * BOXW * 4; \
    wv.x = *reinterpret_cast<const float*>(qp_); wv.y = *reinterpret_cast<const float*>(qp_ + BOXW * 4); \
    wv.z = *reinterpret_cast<const float*>(qp_ + 2 * BOXW * 4); wv.w = *reinterpret_cast<const float*>(qp_ + 3 * BOXW * 4); }
#define WROW(base, r) (*reinterpret_cast<const float*>((base) + (r) * BOXW * 4))
#define ROWOFF(base, r) ((r) * BOXW * 4)
#elif SWZ
#define QUARTER(wv, base, q) const float4 wv = *reinterpret_cast<const float4*>((base) + \
    (((q) ^ (int)(((unsigned)reinterpret_cast<size_t>(base) >> 7) & SWM)) << 4));
#define WROW(base, r) (reinterpret_cast<const float*>(base)[r])
#define ROWOFF(base, r) (((((r) >> 2) ^ (int)(((unsigned)reinterpret_cast<size_t>(base) >> 7) & SWM)) << 4) + ((r) & 3) * 4)
#else
// forward slab: the TMS rows (parents) of a child are one contiguous row
#define QUARTER(wv, base, q) const float4 wv = reinterpret_cast<const float4*>(base)[q];
#define WROW(base, r) (reinterpret_cast<const float*>(base)[r])
#define ROWOFF(base, r) ((r) * 4)
#endif

#if PF
// add pair flow v to the accumulator element at byte offset p of the S slab
__device__ __forceinline__ void pf_add(char* p, float v) {
#if PF == 2
    const unsigned long long f = __float2ull_rn(v * 4294967296.0f);
    const unsigned lo = (unsigned)f & 0x3FFFFu, hi = (unsigned)(f >> 18);
    if (lo) atomicAdd(reinterpret_cast<unsigned*>(p), lo);
    if (hi) atomicAdd(reinterpret_cast<unsigned*>(p + KS * TMS * 4), hi);
#else
    atomicAdd(reinterpret_cast<float*>(p), v);
#endif
}
// S[child row r of the lane's parent] += g * W * x_r for the active rows of quarter q
#endif

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
#if PF
// (S sits a multiple of 1 KB after the W slab: the same swizzle bits)
#define PF_QUARTER(q, wv, g) if (g != 0.f) { \
    if ((rows >> (4 * (q))) & 1u) pf_add(Sb + ROWOFF(Sb, 4 * (q)), g * wv.x * xr[4 * (q)]); \
    if ((rows >> (4 * (q) + 1)) & 1u) pf_add(Sb + ROWOFF(Sb, 4 * (q) + 1), g * wv.y * xr[4 * (q) + 1]); \
    if ((rows >> (4 * (q) + 2)) & 1u) pf_add(Sb + ROWOFF(Sb, 4 * (q) + 2), g * wv.z * xr[4 * (q) + 2]); \
    if ((rows >> (4 * (q) + 3)) & 1u) pf_add(Sb + ROWOFF(Sb, 4 * (q) + 3), g * wv.w * xr[4 * (q) + 3]); }
#define PF_ARGS , unsigned rows, const float* xr, char* sS
#define PF_PASS , rows, xr, sS
#else
#define PF_QUARTER(q, wv, g)
#define PF_ARGS
#define PF_PASS
#endif
template <unsigned QM>
__device__ __forceinline__ void slab_walk(float* a, const int2* __restrict__ pp, const char* sb,
                                          int j, int j1 PF_ARGS) {
    for (; j < j1; j += 4 * GL, pp += 4 * GL) {
        int2 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = j + GL * u < j1 ? __ldg(pp + GL * u) : make_int2(0, 0);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float x = __int_as_float(v[u].y);
#if PF
            char* Sb = sS + v[u].x;
#endif
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (!((QM >> q) & 1)) continue;
                QUARTER(wv, sb + v[u].x, q)
                a[4 * q] = fmaf(x, wv.x, a[4 * q]);
                a[4 * q + 1] = fmaf(x, wv.y, a[4 * q + 1]);
                a[4 * q + 2] = fmaf(x, wv.z, a[4 * q + 2]);
                a[4 * q + 3] = fmaf(x, wv.w, a[4 * q + 3]);
                PF_QUARTER(q, wv, x)
            }
        }
    }
}
// the same for any quarter mask (uniform branches with GL = 32, per-lane
// predicates with several items per warp)
__device__ __forceinline__ void slab_walk_any(float* a, const int2* __restrict__ pp, const char* sb,
                                              int j, int j1, unsigned qm PF_ARGS) {
    for (; j < j1; j += 4 * GL, pp += 4 * GL) {
        int2 v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = j + GL * u < j1 ? __ldg(pp + GL * u) : make_int2(0, 0);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float x = __int_as_float(v[u].y);
#if PF
            char* Sb = sS + v[u].x;
#endif
            #pragma unroll
            for (int q = 0; q < TMS / 4; ++q) {
                if (!((qm >> q) & 1)) continue;
                QUARTER(wv, sb + v[u].x, q)
                a[4 * q] = fmaf(x, wv.x, a[4 * q]);
                a[4 * q + 1] = fmaf(x, wv.y, a[4 * q + 1]);
                a[4 * q + 2] = fmaf(x, wv.z, a[4 * q + 2]);
                a[4 * q + 3] = fmaf(x, wv.w, a[4 * q + 3]);
                PF_QUARTER(q, wv, x)
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

// copy the W slab of tile t0.. (TMS rows) and range k0.. (KS rows) into dst;
// `bar` completes when it has landed (TMA: thread 0; cp.async: every thread).
// Forward: tile = parents, range = children, layout [KS][TMS]. Backward:
// tile = children, range = parents, layout [KS / BOXW][TMS][BOXW] (TMA boxes
// of BOXW parents, which are contiguous in W, by the TMS children).
__device__ __forceinline__ void load_slab(float* dst, unsigned long long* bar, int t0, int k0,
                                          const float* __restrict__ w, long long w_start,
                                          const TmaDesc* tmap, int tid) {
#if LOADER == 9
    // the slab is contiguous in the slab-order copy (either direction): bulk copies of up to 16 KB
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     :: "r"(smem_addr(bar)), "r"(KS * TMS * 4) : "memory");
        const char* src = reinterpret_cast<const char*>(
            w + w_start + ((long long)(t0 / TMS) * NSPLIT + k0 / KS) * KS * TMS);
        for (int off = 0; off < KS * TMS * 4; off += 16384) {
            const int sz = KS * TMS * 4 - off < 16384 ? KS * TMS * 4 - off : 16384;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         :: "r"(smem_addr(dst) + off), "l"(src + off), "r"(sz), "r"(smem_addr(bar)) : "memory");
        }
    }
#elif BWD
#if LOADER == 8
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                     :: "r"(smem_addr(bar)), "r"(KS * TMS * 4) : "memory");
        for (int part = 0; part < KS / BOXW; ++part) {
            const int p = k0 + part * BOXW;
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1, {%2, %3}], [%4];\n"
                         :: "r"(smem_addr(dst + part * TMS * BOXW)), "l"(tmap), "r"(p % BS),
                            "r"((p / BS) * H_IN + t0), "r"(smem_addr(bar)) : "memory");
        }
    }
#else
    for (int e = tid; e < KS * TMS; e += NT) {
        const int part = e / (TMS * BOXW), rem = e - part * (TMS * BOXW);
        const int r = rem / BOXW, p = k0 + part * BOXW + (rem - r * BOXW), c = t0 + r;
        if (c < H_IN && p < H_OUT)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                         :: "r"(smem_addr(dst + e)),
                            "l"(w + w_start + ((long long)(p / BS) * H_IN + c) * BS + (p % BS)) : "memory");
    }
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(smem_addr(bar)) : "memory");
#endif
#else
    const int h0 = t0, c0 = k0;
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
        const int d = e;
        if (c < H_IN && h < H_OUT)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                         :: "r"(smem_addr(dst + d)),
                            "l"(w + w_start + ((long long)(h / BS) * H_IN + c) * BS + (h % BS)) : "memory");
    }
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(smem_addr(bar)) : "memory");
#endif
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
SLAB_MAIN(float* __restrict__ out_vals, long long out_stride,
            const float* __restrict__ w, long long w_start,
            const int2* __restrict__ xp, int xp_stride, const int* __restrict__ cofs,
            const float* __restrict__ max_val, const int* __restrict__ in_nnz,
            const int2* __restrict__ ent, const int* __restrict__ cnt, int B,
            const float* __restrict__ in_vals, long long in_stride,
            const __grid_constant__ TmaDesc tmap,
            float* __restrict__ pflow, long long pf_start, int pf_neg)
{
    extern __shared__ __align__(1024) float smem[];
    float* slab = smem;                                              // [KS][TMS]
#if PF
    float* sacc = slab + KS * TMS;                                   // S, the slab's layout (PF 2: lo, hi)
    unsigned long long* bar = reinterpret_cast<unsigned long long*>(sacc + KS * TMS * PF);
#else
    unsigned long long* bar = reinterpret_cast<unsigned long long*>(slab + KS * TMS);
#endif
    int* s_item = reinterpret_cast<int*>(bar + 1);

    const int m = blockIdx.y, sp = blockIdx.x;
    const int n = cnt[m];
    if (n == 0) return;
    const int tid = threadIdx.x, lane = tid & 31;
#if PF
    for (int e = tid; e < KS * TMS * PF / 4; e += NT) reinterpret_cast<float4*>(sacc)[e] = make_float4(0.f, 0.f, 0.f, 0.f);
    char* sS = reinterpret_cast<char*>(sacc);
#endif
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                     :: "r"(smem_addr(bar)), "r"(LOADER >= 8 ? 1 : NT) : "memory");   // TMA / bulk: thread 0 arrives
        *s_item = 0;
    }
    __syncthreads();
    load_slab(slab, bar, m * TMS, sp * KS, w, w_start, &tmap, tid);
    mbar_wait(bar, 0);
    const char* sb = reinterpret_cast<const char*>(slab);

    const int2* lst = ent + (long long)m * B;
    // items: the samples of the tile's list; each group of GL lanes takes one,
    // a warp pulls IPW at a time; the next items' entries and child ranges are
    // loaded while the current ones run
    const int grp = lane / GL, gl = lane & (GL - 1);
#if STATIC
    int i0 = (tid >> 5) * IPW;
#else
    int i0 = 0;
    if (lane == 0) i0 = atomicAdd(s_item, IPW);
    i0 = __shfl_sync(FULL, i0, 0);
#endif
    int i = i0 + grp;
    int2 e = make_int2(0, 0), cr = make_int2(0, 0);
    if (i < n) {
        e = lst[i];
        cr = make_int2(cofs[(long long)e.x * (NSPLIT + 1) + sp], cofs[(long long)e.x * (NSPLIT + 1) + sp + 1]);
    }
    while (i0 < n) {
        const bool has = i < n;                       // a group past the list's end idles
        const int b = e.x, ps = e.y >> TMS;
        const unsigned rows = has ? (e.y & ((1 << TMS) - 1)) : 0u;
        const int j0 = cr.x, j1 = has ? cr.y : cr.x;
#if STATIC
        const int n0 = i0 + NWARP * IPW;
#else
        int n0 = 0;
        if (lane == 0) n0 = atomicAdd(s_item, IPW);
        n0 = __shfl_sync(FULL, n0, 0);
#endif
        const int inext = n0 + grp;
        if (inext < n) {
            e = lst[inext];
            cr = make_int2(cofs[(long long)e.x * (NSPLIT + 1) + sp], cofs[(long long)e.x * (NSPLIT + 1) + sp + 1]);
        }

        float a[TMS];
        #pragma unroll
        for (int r = 0; r < TMS; ++r) a[r] = 0.f;
        const int2* __restrict__ pp = xp + ((unsigned)(b * xp_stride) + (unsigned)(j0 + gl));
#if PF
        // x of the item's active children (rows): lane r of the group computes
        // row r's, the group shares them
        float xr[TMS];
        const float mx = fmaxf(max_val[b], -1e30f);
        #pragma unroll
        for (int r0 = 0; r0 < TMS; r0 += GL) {
            const int r = r0 + gl;
            float xv = 0.f;
            if (r < TMS && ((rows >> r) & 1u))
                xv = expf(in_vals[(long long)b * in_stride + ps + __popc(rows & ((1u << r) - 1))] - mx);
            #pragma unroll
            for (int k = 0; k < (TMS - r0 < GL ? TMS - r0 : GL); ++k) xr[r0 + k] = __shfl_sync(FULL, xv, grp * GL + k);
        }
#endif
#if TMS >= 8 && GL == 32
        // only the 4-row quarters of the tile that hold one of the sample's
        // parents are read and multiplied (uniform per item)
        const unsigned qmask = quarters(rows);
        if (qmask == 1u) slab_walk<1>(a, pp, sb, j0 + gl, j1 PF_PASS);
        else if (qmask == 2u) slab_walk<2>(a, pp, sb, j0 + gl, j1 PF_PASS);
        else if (qmask == 3u) slab_walk<3>(a, pp, sb, j0 + gl, j1 PF_PASS);
#if TMS == 16
        else slab_walk_any(a, pp, sb, j0 + gl, j1, qmask PF_PASS);
#endif
#elif TMS >= 8
        slab_walk_any(a, pp, sb, j0 + gl, j1, quarters(rows) PF_PASS);
#elif TMS == 4
        slab_walk<1>(a, pp, sb, j0 + gl, j1 PF_PASS);
#elif TMS == 2 && !BWD
        // one 8-byte read per child; four children per lane in flight
        for (int j = j0 + gl; j < j1; j += 4 * GL, pp += 4 * GL) {
            int2 v[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) v[u] = j + GL * u < j1 ? __ldg(pp + GL * u) : make_int2(0, 0);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float x = __int_as_float(v[u].y);
                const float2 w2 = *reinterpret_cast<const float2*>(sb + v[u].x);
                a[0] = fmaf(x, w2.x, a[0]);
                a[1] = fmaf(x, w2.y, a[1]);
            }
        }
#else
        // four children per lane in flight; past j1 a lane adds 0 x row 0
        for (int j = j0 + gl; j < j1; j += 4 * GL, pp += 4 * GL) {
            int2 v[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) v[u] = j + GL * u < j1 ? __ldg(pp + GL * u) : make_int2(0, 0);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float x = __int_as_float(v[u].y);
                #pragma unroll
                for (int r = 0; r < TMS; ++r) {
                    const float wr = WROW(sb + v[u].x, r);
                    a[r] = fmaf(x, wr, a[r]);
#if PF
                    if (x != 0.f && ((rows >> r) & 1u))
                        pf_add(sS + v[u].x + ROWOFF(sS + v[u].x, r), x * wr * xr[r]);
#endif
                }
            }
        }
#endif
        // the group's sum per parent row (transpose reduction: at each of the
        // first LT levels a lane keeps half of its rows and sends the other
        // half; xor offsets below GL stay inside the group). Then either each
        // lane holds TMS / GL whole rows (TMS >= GL: rows gl * (TMS / GL) + k)
        // or plain sums finish row r = gl >> (LOG_GL - LOG_TMS) in lane
        // r * (GL / TMS).
        #pragma unroll
        for (int lv = 0; lv < LT; ++lv) {
            const int off = GL >> (lv + 1), h = TMS >> (lv + 1);
            const bool hi = gl & off;
            #pragma unroll
            for (int k = 0; k < h; ++k) {
                const float keep = hi ? a[k + h] : a[k], send = hi ? a[k] : a[k + h];
                a[k] = keep + __shfl_xor_sync(FULL, send, off);
            }
        }
#if LOG_TMS < LOG_GL
        #pragma unroll
        for (int off = GL >> (LT + 1); off; off >>= 1) a[0] += __shfl_xor_sync(FULL, a[0], off);
        constexpr int RPL = 1;
        const int r0 = gl >> (LOG_GL - LOG_TMS);
        const bool writer = (gl & ((GL >> LOG_TMS) - 1)) == 0;
#else
        constexpr int RPL = TMS >> LOG_GL;
        const int r0 = gl * RPL;
        const bool writer = true;
#endif
        // active rows of the tile ascend: row r's output is ps + rank(r)
        #pragma unroll
        for (int k = 0; k < RPL; ++k) {
            const int r = r0 + k;
            if (writer && ((rows >> r) & 1)) {
                const int o = ps + __popc(rows & ((1u << r) - 1));
                float* dst = out_vals + (long long)b * out_stride + o;
                const float acc = a[k];
#if NSPLIT > 1
                atomicAdd(dst, acc);
#elif BWD
                *dst = expf(in_vals[(long long)b * in_stride + o] - fmaxf(max_val[b], -1e30f)) * acc;
#else
                *dst = in_nnz[b] == 0 ? __int_as_float(0xff800000) : logf(acc) + fmaxf(max_val[b], -1e30f);
#endif
            }
        }
        i0 = n0;
        i = inext;
    }
#if PF
    // the slab's param flows: CTA (child tile m, parent range sp) owns them;
    // layout [KS / BOXW][TMS children][BOXW parents], parents contiguous in W
    __syncthreads();
    for (int e = tid; e < KS * TMS; e += NT) {
#if PF == 2
        const unsigned lo = reinterpret_cast<const unsigned*>(sacc)[e];
        const unsigned hi = reinterpret_cast<const unsigned*>(sacc)[e + KS * TMS];
        if ((lo | hi) == 0u) continue;
        const float v = (float)(((double)hi * 262144.0 + (double)lo) * 2.3283064365386963e-10);
#else
        const float v = sacc[e];
        if (v == 0.f) continue;
#endif
#if BNAT
        // [KS parents][TMS children], row chunks swizzled with SWZ
        const int pl = e / TMS, pr = e - pl * TMS;
        const int r = SWZ ? pr ^ ((((pl * TMS * 4) >> 7) & SWM) << 2) : pr;
        const int p = sp * KS + pl, c = m * TMS + r;
#else
        const int part = e / (TMS * BOXW), rem = e - part * (TMS * BOXW);
        const int r = rem / BOXW, p = sp * KS + part * BOXW + (rem - r * BOXW), c = m * TMS + r;
#endif
        if (c < H_IN && p < H_OUT) {
            float* dst = pflow + pf_start + ((long long)(p / BS) * H_IN + c) * BS + (p % BS);
            *dst += pf_neg ? -v : v;
        }
    }
#endif
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
// W in backward slab order (LOADER 9): the slab of (child tile m, parent range s) is the
// KS x TMS floats at ((m * NSPLIT + s) * KS) * TMS, row p the tile's TMS children of parent
// s * KS + p (swizzled chunks with SWZ); padding zero. Through shared memory, so that both the
// reads (parents contiguous in W) and the writes (children contiguous) coalesce. Grid
// (cdiv(H_in, TMS), NSPLIT * KS / 32), block 256.
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
        const int rr = SWZ ? r ^ ((((pin * TMS * 4) >> 7) & SWM) << 2) : r;
        dst[((long long)(m * NSPLIT + sp) * KS + pin) * TMS + rr] = t[pl][r];
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
        const int hl = h % KS;
        xp[(long long)b * xp_stride + i] = make_int2(BNAT ? hl * TMS * 4 : ((hl / BOXW) * TMS * BOXW + hl % BOXW) * 4,
                                                     __float_as_int(g));
    }
}
#endif
"""

_SLAB_FWD_ARGS = ["ptr", "i64", "ptr", "i64", "ptr", "i32", "ptr", "ptr", "ptr",
                  "ptr", "ptr", "i32", "ptr", "i64", "tmap", "ptr", "i64", "i32"]
_SLAB_FIN_FLOW_ARGS = ["ptr", "i64", "ptr", "ptr", "i64", "ptr"]
_SLAB_PACK_G_ARGS = ["ptr", "i64", "ptr", "ptr", "ptr", "ptr", "ptr", "i64", "ptr", "i64", "ptr", "ptr"]
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


def slab_smem(KS: int, TMS: int, pf: int = 0) -> int:
    """Dynamic shared memory of a slab CTA: the slab (and the param-flow
    accumulator: ``pf`` 1 fp32, 2 fixed point in two u32 arrays), its barrier
    and the item counter."""
    return KS * TMS * 4 * (1 + int(pf)) + 16


def slab_kernels(H_in: int, H_out: int, BS: int, TMS: int, KS: int, NWARP: int,
                 loader: int, min_blocks: int, bwd: bool = False, GL: int = 32,
                 pf: int = 0, swz: bool = False, static: bool = False) -> tuple:
    """``(lists, pack, main, finalize kernels, main smem bytes)`` of a block
    shape: the forward (tiles of parents, ranges of children) or with
    ``bwd`` the element-flow backward (tiles of children, ranges of
    parents); compiled on first use (cubins cached on disk). ``GL`` lanes
    walk one item (``32 / GL`` items per warp); ``pf`` (backward) accumulates the
    slab's parameter flows too (1 fp32, 2 fixed point); ``swz`` (slab-order copies,
    ``loader`` 9, 8 / 16-row tiles) reads swizzled rows; ``static`` hands the
    items to the warps round-robin."""
    assert GL in (4, 8, 16, 32) and not (pf and not bwd)
    assert not (swz and (loader != 9 or TMS not in (8, 16)))
    key = (H_in, H_out, BS, TMS, KS, NWARP, loader, min_blocks, bwd, GL, pf, swz, static)
    hit = _SLAB_KERNELS.get(key)
    if hit is None:
        smem = slab_smem(KS, TMS, pf)
        n_split = -(-(H_out if bwd else H_in) // KS)
        defines = dict(H_IN=H_in, H_OUT=H_out, BS=BS, TMS=TMS, LOG_TMS=TMS.bit_length() - 1, KS=KS,
                       NWARP=NWARP, LOADER=loader, MINB=min_blocks, BWD=int(bwd), BOXW=min(KS, 256),
                       GL=GL, PF=int(pf), SWZ=int(swz), STATIC=int(static))
        src = ("".join(f"#define {k} {v}\n" for k, v in defines.items())
               + f"#define NSPLIT_OVERRIDE {n_split}\n" + _SLAB_SRC)
        main = CudaKernel(src, "ws_slab_bwd" if bwd else "ws_slab_fwd", _SLAB_FWD_ARGS, max_shared=smem)
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
    """Floats of the slab-order copy of W (:func:`slab_wcopy`): forward tiles of parents and
    ranges of children, or with ``bwd`` tiles of children and ranges of parents."""
    H_t, H_r = (H_in, H_out) if bwd else (H_out, H_in)
    KS, nsplit = slab_shape(H_r, TMS, slab_bytes)
    return -(-H_t // TMS) * TMS * nsplit * KS


def slab_wcopy(dst: torch.Tensor, w: torch.Tensor, w_start: int, H_in: int, H_out: int, BS: int,
               TMS: int, slab_bytes: int, swz: bool = False, bwd: bool = False) -> None:
    """Write W (block layout of :class:`SparseIOSumLayer` at ``w[w_start:]``) into ``dst`` in
    slab order: one contiguous ``[KS, TMS]`` slab per (parent tile, child range) for the
    forward, per (child tile, parent range) with ``bwd``, as the bulk loader of
    :func:`ws_slab_forward` / :func:`ws_slab_backward` (``wslab``) reads it. One pass over W."""
    H_r = H_out if bwd else H_in
    KS, nsplit = slab_shape(H_r, TMS, slab_bytes)
    swz = swz and TMS in (8, 16)
    key = (H_in, H_out, BS, TMS, KS, swz, bwd)
    k = _WCOPY_KERNELS.get(key)
    if k is None:
        defines = dict(H_IN=H_in, H_OUT=H_out, BS=BS, TMS=TMS, LOG_TMS=TMS.bit_length() - 1, KS=KS,
                       NWARP=4, LOADER=9, MINB=1, BWD=int(bwd), BOXW=min(KS, 256), SWZ=int(swz))
        src = ("".join(f"#define {k_} {v}\n" for k_, v in defines.items())
               + f"#define NSPLIT_OVERRIDE {nsplit}\n" + _SLAB_SRC)
        k = _WCOPY_KERNELS[key] = CudaKernel(src, "ws_slab_wcopy_bwd" if bwd else "ws_slab_wcopy",
                                             ["ptr", "ptr", "i64"])
    n = slab_copy_numel(H_in, H_out, TMS, slab_bytes, bwd)
    assert dst.numel() >= n and dst.dtype == torch.float32
    assert (nsplit * KS) % 32 == 0 and KS % 32 == 0
    if bwd:
        k((-(-H_in // TMS), nsplit * KS // 32), (256,), [dst, w, w_start], device=dst.device)
    else:
        assert TMS <= 32 and 32 % TMS == 0
        k((-(-H_out // 32), nsplit * KS // 32), (256,), [dst, w, w_start], device=dst.device)


def ws_slab_forward(out_vals: torch.Tensor, w: torch.Tensor, w_start: int, sv_in, sv_out,
                    max_val: torch.Tensor, ent: torch.Tensor, cnt: torch.Tensor,
                    xp: torch.Tensor, cofs: torch.Tensor, B: int, H_in: int, H_out: int, BS: int,
                    TMS: int = 4, slab_bytes: int = 64 * 1024, NWARP: int = 16,
                    min_blocks: int = 3, GL: int = 32, swz: bool = False,
                    static: bool = False, wslab: Optional[torch.Tensor] = None) -> None:
    """Build the per-tile sample lists, pack the input columns and run the
    slab forward for one block, writing ``out_vals[b, i]`` (row stride
    ``out_vals.stride(0)``) for every active parent ``i`` of every sample;
    ``w[w_start:]`` holds W in the block layout of :class:`SparseIOSumLayer`.
    Scratch: ``ent`` (int32 ``[cdiv(H_out, TMS), 2 B]``), ``cnt`` (int32
    ``[cdiv(H_out, TMS)]``), ``xp`` (int32 ``[B, >= 2 K_in]``), ``cofs``
    (int32 ``[B, NSPLIT + 1]``, :func:`slab_shape`). With ``wslab`` (W in slab
    order, :func:`slab_wcopy` with the same ``TMS`` / ``slab_bytes`` / ``swz``)
    the slabs are read from it with bulk copies, any tile height. Graph-capturable,
    no host sync."""
    assert TMS in (1, 2, 4, 8, 16)
    KS, nsplit = slab_shape(H_in, TMS, slab_bytes)
    assert sv_out.values.shape[-1] < 1 << (31 - TMS), "output columns too long for the list entries"
    n_m = -(-H_out // TMS)
    aligned = (TMS >= 4 and BS % TMS == 0 and (w.data_ptr() // 4 + w_start) % 4 == 0
               and w.data_ptr() % 4 == 0)
    loader = FORCE_LOADER if FORCE_LOADER in (1, 8) else (8 if aligned else 1)
    assert loader == 1 or aligned
    swz = swz and wslab is not None and TMS in (8, 16)
    if wslab is not None:                             # slab-order copy of W: bulk loads
        loader, w, w_start = 9, wslab, 0
    lists, pack, fwd, fin, smem = slab_kernels(H_in, H_out, BS, TMS, KS, NWARP, loader, min_blocks,
                                               GL=GL, swz=swz, static=static)
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
         ent, cnt, B, max_val, 0, tmap, max_val, 0, 0], shared_mem=smem, device=device)
    if nsplit > 1:
        fin((-(-sv_out.total_nnz // 256), B), (256,),
            [out_vals, out_vals.stride(0), sv_out.nnz, sv_in.nnz, max_val], device=device)


def ws_slab_backward(flow_vals: torch.Tensor, w: torch.Tensor, w_start: int, sv_in, sv_out,
                     sv_flow_out, max_val: torch.Tensor, ent: torch.Tensor, cnt: torch.Tensor,
                     xp: torch.Tensor, cofs: torch.Tensor, B: int, H_in: int, H_out: int, BS: int,
                     TMS: int = 4, slab_bytes: int = 64 * 1024, NWARP: int = 16,
                     min_blocks: int = 3, GL: int = 32, static: bool = False, pflow: Optional[torch.Tensor] = None,
                     pf_start: int = 0, pf_neg: bool = False, pf_fixed: bool = False,
                     wslab: Optional[torch.Tensor] = None, swz: bool = False) -> None:
    """Element flows of one block: ``flow_vals[b, k] = exp(in[b, k] - max_b)
    * sum_h g[b, h] W[h, c_k]`` over the sample's active parents ``h`` (``g =
    nflow * exp(max_b - nmars)``), for every active child ``c_k`` — the slab
    forward's mirror: tiles of ``TMS`` children, the parents in ranges whose
    ``[KS, TMS]`` slab fits ``slab_bytes``. Scratch: ``ent`` (int32
    ``[cdiv(H_in, TMS), 2 B]``), ``cnt`` (int32 ``[cdiv(H_in, TMS)]``), ``xp``
    (int32 ``[B, >= 2 K_out]``), ``cofs`` (int32 ``[B, NSPLIT + 1]`` over the
    parents). With ``pflow`` the kernel also adds the parameter flows
    ``W[h, c] sum_b g[b, h] x[b, c]`` of the active pairs to
    ``pflow[pf_start:]`` (same block layout as W; subtracted with ``pf_neg``),
    accumulated per slab in shared memory (twice the slab's shared memory;
    with ``pf_fixed`` in 32-bit fixed point, three times, deterministic). With
    ``wslab`` (:func:`slab_wcopy` with ``bwd``, same ``TMS`` / ``slab_bytes`` /
    ``swz``) the slabs come from that copy with bulk loads, ``[KS parents][TMS
    children]``: one 16-byte read per quarter instead of four.
    Graph-capturable, no host sync."""
    assert TMS in (1, 2, 4, 8, 16)
    KS, nsplit = slab_shape(H_out, TMS, slab_bytes)
    assert sv_in.values.shape[-1] < 1 << (31 - TMS), "input columns too long for the list entries"
    boxw = min(KS, 256)
    n_t = -(-H_in // TMS)
    aligned = (BS % boxw == 0 and (w.data_ptr() // 4 + w_start) % 4 == 0 and w.data_ptr() % 4 == 0)
    loader = FORCE_LOADER if FORCE_LOADER in (1, 8) else (8 if aligned else 1)
    assert loader == 1 or aligned
    swz = swz and wslab is not None and TMS in (8, 16)
    if wslab is not None:                             # slab-order copy of W: bulk loads
        loader, w, w_start = 9, wslab, 0
    pf = 0 if pflow is None else (2 if pf_fixed else 1)
    assert pf != 2 or B < 1 << 14, "fixed-point param flows: fewer than 2^14 samples"
    lists, pack, main, fin, smem = slab_kernels(H_in, H_out, BS, TMS, KS, NWARP, loader, min_blocks,
                                                bwd=True, GL=GL, pf=pf, static=static, swz=swz)
    K = sv_out.total_nnz
    assert xp.stride(0) >= 2 * K and xp.stride(0) % 2 == 0 and cofs.stride(0) == nsplit + 1
    assert ent.shape[0] >= n_t and ent.stride(0) == 2 * B and cnt.numel() >= n_t
    device = flow_vals.device
    if loader == 8:
        key = ("bwd", w.data_ptr(), w_start, H_in, H_out, BS, TMS, KS)
        tmap = _TMAPS.get(key)
        if tmap is None:
            # W as [NB * H_in rows (block, child), BS parents], boxes of BOXW parents x TMS children
            tmap = _TMAPS[key] = tensor_map_2d(w, w_start, H_out // BS * H_in, BS, BS,
                                               TMS, boxw, swizzle=0)
    else:
        tmap = _NO_TMAP
    cnt.zero_()
    lists((-(-sv_in.total_nnz // 256), B), (256,),
          [ent, cnt, sv_in.indices, sv_in.col_starts, sv_in.nnz, B, flow_vals, flow_vals.stride(0)],
          device=device)
    pack((-(-(K + 1) // 256), B), (256,),
         [xp, xp.stride(0) // 2, cofs, sv_out.indices, sv_out.col_starts, sv_out.nnz,
          sv_flow_out.values, sv_flow_out.values.stride(0), sv_out.values, sv_out.values.stride(0),
          max_val, sv_in.nnz], device=device)
    main((nsplit, n_t), (NWARP * 32,),
         [flow_vals, flow_vals.stride(0), w, w_start, xp, xp.stride(0) // 2, cofs, max_val, sv_in.nnz,
          ent, cnt, B, sv_in.values, sv_in.values.stride(0), tmap,
          pflow if pf else flow_vals, pf_start, int(pf_neg)], shared_mem=smem, device=device)
    if nsplit > 1:
        fin((-(-sv_in.total_nnz // 256), B), (256,),
            [flow_vals, flow_vals.stride(0), sv_in.nnz, sv_in.values, sv_in.values.stride(0), max_val],
            device=device)
