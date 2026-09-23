// ============================================================
// Flash attention (bf16) via Metal 4 TensorOps, compiled at
// MTLLanguageVersion4_1 — the macOS 27 port of mtlflashattn's v2_bf16 kernel.
//
// Why this exists: mtlflashattn JIT-compiles through torch.mps.compile_shader,
// and on macOS 27 that path no longer exposes mpp::tensor_ops (the MPP header
// contents are guarded out at compile_shader's language version), so its fast
// tiers silently degrade to a chunked-torch fallback that is 3.5-4.7x SLOWER
// than ComfyUI's sub-quad attention at Wan/Z-Image shapes. This extension
// compiles the SAME kernel through newLibraryWithSource with an explicit
// languageVersion 4.1 — the only Metal-4 compile path that works here (same
// pipeline as fp8_ext / fp8s_ext / int8_ext).
//
// Kernel body: _V2_DTYPE_MSL from mtlflashattn (_kernel.py), bfloat
// specialization, head dim baked per compile. MIT License,
// Copyright (c) Paweł Mazurkiewicz (github.com/pawel-mazurkiewicz/mtlflashattn).
// Host dispatch mirrors _flash_v2_dtype: threadgroups (ceil(Lq/32), B*Hq, 1),
// 128 threads (4 simdgroups), encoded on torch's MPS stream, no COMMIT_AND_WAIT.
//
// Scope (v1 port): bf16 q/k/v/out, no softcap / sliding window / alibi (the
// ComfyUI router only routes plain non-causal SDPA); causal supported since
// it is one flag.
//
// Tuning history (M5 Max, flux-edit B1 H32 S8704 D128): BC=32 port 69ms ->
// BC=128 templated tiles 44 -> skip-rescale + setBytes 41.5 -> static
// reduction dims + bf16 score tiles with P merged into the score buffer
// (halves threadgroup memory -> doubles TG residency) 33.7ms = 37 TF/s.
// NSG=8 and BC>=192 both collapse; BR must stay 32 (ctO spills).
// ============================================================

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <map>
#include <array>
#include <string>
#include "../metal_template.h"

using namespace at::mps;

static const char* kSrcTemplate = R"MTL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BR = @BR@;
constant constexpr int BC = @BC@;   // KV tile width; wide tiles feed the matmul
constant constexpr int DD = @D@;    // units properly (BC=32 starves them at ~18 TF/s)
constant constexpr int NSG = @NSG@;
constant constexpr int RPG = BR / NSG;   // rows per simdgroup
constant constexpr int CCH = BC / 32;    // 32-lane column chunks per row

kernel void flash_attn_fwd_bf16(
    device @DT@* Q  [[buffer(0)]],
    device @DT@* K  [[buffer(1)]],
    device @DT@* V  [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal]
    constant float* PR [[buffer(5)]],  // [scale]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], causal=SH[5];
    const float scale = PR[0];

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x) * BR;

    device @DT@* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device @DT@* Kp = K + ulong(b*Hkv+hkv)*ulong(Lk)*DD;
    device @DT@* Vp = V + ulong(b*Hkv+hkv)*ulong(Lk)*DD;

    // Reduction dims are compile-time constants — declare them static
    // (@QKS@ = DD or dynamic, @PVS@ = BC or dynamic; templated for A/B).
    constexpr auto qk_desc = matmul2d_descriptor(
        BR, BC, @QKS@, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;

    constexpr auto pv_desc = matmul2d_descriptor(
        BR, DD, @PVS@, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    // MERGEP: P overwrites the score tile in place (bf16 score tiles only —
    // same threadgroup buffer), halving threadgroup memory -> more resident
    // threadgroups per core.
    constexpr bool MERGEP = @MP@;
    threadgroup @ST@ tgS[BR*BC];
    threadgroup @DT@ tgP[MERGEP ? 1 : BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];
    threadgroup int tgAnyCorr;
    threadgroup @DT@* Pbuf = MERGEP ? (threadgroup @DT@*)&tgS[0] : &tgP[0];

    auto mQ = tensor(Qp + q0*DD, dextents<int,2>{DD, min(BR, Lq - q0)}, array<int,2>{1, DD});
    auto tS = tensor(&tgS[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});
    auto tP = tensor(Pbuf, dextents<int,2>{BC, BR}, array<int,2>{1, BC});

    using PvA = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
    using PvB = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctO = pv_op.get_destination_cooperative_tensor<PvA, PvB, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i)
        if (ctO.is_valid_element(i)) ctO[i] = 0.0f;

    const int r0 = int(sgid) * RPG;
    float m[RPG], l[RPG];
    #pragma clang loop unroll(full)
    for (int r = 0; r < RPG; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    const int shift = Lk - Lq;
    for (int j0 = 0; j0 < Lk; j0 += BC) {
        if (causal && j0 > q0 + BR - 1 + shift) break;

        // No tgS zero-fill: out-of-bounds columns are masked to -inf below,
        // so stale values are never read.
        if (tid == 0) tgAnyCorr = 0;
        auto mK = tensor(Kp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        qk_op.run(mQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax over the BC-wide tile, 32 lanes sweeping CCH chunks
        // per row; scores held in registers between the max and exp passes.
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r) {
            const int row  = r0 + r;
            const int qrow = q0 + row;
            float sreg[CCH];
            float mb = -INFINITY;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const int j = j0 + cc*32 + int(lane);
                float s = float(tgS[row*BC + cc*32 + lane]) * scale;
                const bool ok = (j < Lk) && (qrow < Lq) && (!causal || j <= qrow + shift);
                s = ok ? s : -INFINITY;
                sreg[cc] = s;
                mb = max(mb, simd_max(s));
            }
            const float mn = max(m[r], mb);
            const float c = (mn == -INFINITY) ? 1.0f : exp(m[r] - mn);
            float lsum = 0.0f;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const float p = (mn == -INFINITY) ? 0.0f : exp(sreg[cc] - mn);
                Pbuf[row*BC + cc*32 + lane] = @DT@(p);
                lsum += simd_sum(p);
            }
            l[r] = l[r]*c + lsum;
            m[r] = mn;
            tgCorr[row] = c;
            if (c != 1.0f) tgAnyCorr = 1;   // benign race: all writers store 1
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale prior accumulators only when some row's max actually moved
        // (rare after the first few chunks). MUST be a threadgroup-wide flag:
        // each simdgroup's cooperative-tensor elements span the FULL BR tile,
        // not just its own softmax rows.
        if (tgAnyCorr != 0) {
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
                if (!ctO.is_valid_element(i)) continue;
                auto idx = ctO.get_multidimensional_index(i);
                ctO[i] *= tgCorr[int(idx[1])];
            }
        }
        auto mV = tensor(Vp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        pv_op.run(tP, mV, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r)
            tgInvL[r0 + r] = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device @DT@* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= Lq || c >= DD) continue;
        Ob[r*DD + c] = @DT@(ctO[i] * tgInvL[row]);
    }
}
// ── v2r: register-resident P (mtlflashattn's v2r recipe, bf16) ──
// Empirical API constraints (mirrored from mtlflashattn): input cooperative
// tensors require SINGLE-simdgroup ops; the source cooperative tensor's
// element type must equal the left-input element type (=> S rounds to bfloat,
// same class as the bmm path's bf16 score matrix); the PV descriptor needs a
// STATIC inner dim => Lk must be a multiple of BCV (host gates on this instead
// of paying a pad copy — Flux2-edit's 8704 = 68*128 qualifies).
constant constexpr int SR  = @SR@;    // query rows per simdgroup (16 = native
constant constexpr int BCV = @BCV@;   // matmul granularity); BCV = v2r KV tile

kernel void flash_attn_fwd_v2r_bf16(
    device bfloat* Q  [[buffer(0)]],
    device bfloat* K  [[buffer(1)]],
    device bfloat* V  [[buffer(2)]],
    device bfloat* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal]
    constant float* PR [[buffer(5)]],  // [scale]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], causal=SH[5];
    const float scale = PR[0];

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x)*(SR*NSG) + int(sgid)*SR;

    threadgroup float tgM[NSG][SR], tgL[NSG][SR], tgCorr[NSG][SR], tgBlk[NSG][SR];

    if (q0 >= Lq) return;             // uniform per simdgroup

    device bfloat* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device bfloat* Kp = K + ulong(b*Hkv+hkv)*ulong(Lk)*DD;
    device bfloat* Vp = V + ulong(b*Hkv+hkv)*ulong(Lk)*DD;

    constexpr auto qk_desc_r = matmul2d_descriptor(
        SR, BCV, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc_r, execution_simdgroup> qk_op;
    constexpr auto pv_desc_r = matmul2d_descriptor(
        SR, DD, BCV, false, false, false,   // static k: left-input reuse requires it
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc_r, execution_simdgroup> pv_op;

    if (lane < SR) { tgM[sgid][lane] = -INFINITY; tgL[sgid][lane] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    auto mQ = tensor(Qp + q0*DD, dextents<int,2>{DD, min(SR, Lq - q0)}, array<int,2>{1, DD});
    using TT = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctS = qk_op.get_destination_cooperative_tensor<TT, TT, bfloat>();
    using CtP = decltype(pv_op.get_left_input_cooperative_tensor<bfloat, bfloat, float>());
    auto ctO = pv_op.get_destination_cooperative_tensor<CtP, TT, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) ctO[i] = 0.0f;

    const int shift = Lk - Lq;        // bottom-right causal alignment
    for (int j0 = 0; j0 < Lk; j0 += BCV) {
        if (causal && j0 > q0 + SR - 1 + shift) break;

        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctS.get_capacity(); ++i) ctS[i] = bfloat(0.0f);
        auto mK = tensor(Kp + j0*DD, dextents<int,2>{DD, BCV}, array<int,2>{1, DD});
        qk_op.run(mQ, mK, ctS);

        // scale + bounds/causal mask (bfloat storage, fp32 math)
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {
            if (!ctS.is_valid_element(i)) continue;
            auto idx = ctS.get_multidimensional_index(i);
            const int jj = j0 + int(idx[0]);
            const int ii = q0 + int(idx[1]);
            const bool ok = (jj < Lk) && (ii < Lq) && (!causal || jj <= ii + shift);
            ctS[i] = ok ? bfloat(float(ctS[i]) * scale) : bfloat(-INFINITY);
        }

        // block row-max via reduce_rows (legal at single-simdgroup scope)
        auto ctMb = qk_op.get_row_reduction_destination_cooperative_tensor<TT, TT, bfloat>();
        reduce_rows(ctS, ctMb, reduction_operation::max, bfloat(-INFINITY));
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctMb.get_capacity(); ++i) {
            if (!ctMb.is_valid_element(i)) continue;
            auto idx = ctMb.get_multidimensional_index(i);
            tgBlk[sgid][int(idx[0])] = float(ctMb[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < SR) {
            const float mo = tgM[sgid][lane];
            const float mn = max(mo, tgBlk[sgid][lane]);
            tgCorr[sgid][lane] = (mn == -INFINITY) ? 1.0f : exp(mo - mn);
            tgM[sgid][lane] = mn;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // P = exp(S - m) in place (bfloat)
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {
            if (!ctS.is_valid_element(i)) continue;
            auto idx = ctS.get_multidimensional_index(i);
            const float mn = tgM[sgid][int(idx[1])];
            ctS[i] = (mn == -INFINITY) ? bfloat(0.0f) : bfloat(exp(float(ctS[i]) - mn));
        }

        // block row-sum via reduce_rows, merge running l
        auto ctLb = qk_op.get_row_reduction_destination_cooperative_tensor<TT, TT, bfloat>();
        reduce_rows(ctS, ctLb, reduction_operation::sum, bfloat(0.0f));
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctLb.get_capacity(); ++i) {
            if (!ctLb.is_valid_element(i)) continue;
            auto idx = ctLb.get_multidimensional_index(i);
            tgBlk[sgid][int(idx[0])] = float(ctLb[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < SR)
            tgL[sgid][lane] = tgL[sgid][lane]*tgCorr[sgid][lane] + tgBlk[sgid][lane];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // rescale running O, accumulate PV with register-resident P
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
            if (!ctO.is_valid_element(i)) continue;
            auto idx = ctO.get_multidimensional_index(i);
            ctO[i] *= tgCorr[sgid][int(idx[1])];
        }
        auto ctP = pv_op.get_left_input_cooperative_tensor<bfloat, bfloat, float>(ctS);
        auto mV = tensor(Vp + j0*DD, dextents<int,2>{DD, BCV}, array<int,2>{1, DD});
        pv_op.run(ctP, mV, ctO);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    device bfloat* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= Lq || c >= DD) continue;
        const float l = tgL[sgid][row];
        Ob[r*DD + c] = bfloat((l > 0.0f) ? (ctO[i] / l) : 0.0f);
    }
}

// ── frag: staged-KV per-simdgroup kernel (attention rewrite, phase 1) ──
// Phase-0 finding: at video seqs the cooperative kernel is COMPUTE/LATENCY-
// bound, not bandwidth-bound (TF/s flat across 55MB..440MB working sets;
// halving K/V traffic via BR=128 made it SLOWER). So the fix is structural:
//   * K and V tiles staged ONCE per threadgroup into threadgroup memory
//     (vectorized all-thread copy), then every simdgroup computes against
//     the shared tiles — v2r's fatal 4x redundant K/V streams are gone;
//   * each simdgroup owns SR=16 query rows; S/P live in its registers
//     (destination cooperative tensor -> official left-input conversion);
//   * softmax bookkeeping (m, l, corr) lives in LANE REGISTERS: row r's
//     state is held by lane r and broadcast with simd_shuffle — the kernel
//     uses zero threadgroup memory beyond the K/V tiles, so BC=64 fits the
//     full 32KB budget; 2 barriers per KV tile (vs 3 in the cooperative
//     kernel, on half the tile width).
// Numerics: raw QK scores round once to @DT@ (the matmul's own store);
// scale folds into the fp32 max/exp math (one FEWER rounding than the
// cooperative kernel's dt score tile); P rounds to @DT@ like production.
kernel void flash_attn_fwd_frag(
    device @DT@* Q  [[buffer(0)]],
    device @DT@* K  [[buffer(1)]],
    device @DT@* V  [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal]
    constant float* PR [[buffer(5)]],  // [scale]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], causal=SH[5];
    const float scale = PR[0];
    const int bh   = int(tgid.y);
    const int b    = bh / Hq;
    const int hq   = bh % Hq;
    const int hkv  = hq / (Hq / Hkv);
    const int q0tg = int(tgid.x) * (SR * NSG);
    const int q0   = q0tg + int(sgid) * SR;

    device @DT@* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device @DT@* Kp = K + ulong(b*Hkv+hkv)*ulong(Lk)*DD;
    device @DT@* Vp = V + ulong(b*Hkv+hkv)*ulong(Lk)*DD;

    // @DB@=1: double-buffered tiles (1 barrier/tile, needs 4*BC*DD*2 <= 32KB);
    // @DB@=0: single buffer, 2 barriers/tile. Both prefetch the NEXT tile's
    // K/V into registers BEFORE computing the current one, so the device-load
    // latency hides behind the matmuls (the phase-0 latency-bound finding).
    constexpr bool DB = @DB@;
    constexpr int NBUF = DB ? 2 : 1;
    threadgroup @DT@ tgK[NBUF*BC*DD];
    threadgroup @DT@ tgV[NBUF*BC*DD];
    constexpr int NVEC = BC * (DD * 2 / 16);              // uint4 vectors per tile
    constexpr int PFV  = (NVEC + NSG*32 - 1) / (NSG*32);  // vectors per thread

    constexpr auto qk_desc = matmul2d_descriptor(
        SR, BC, DD, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        SR, DD, BC, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    const bool active = q0 < Lq;
    const int qrows = active ? min(SR, Lq - q0) : 1;
    auto mQ = tensor(Qp + ulong(active ? q0 : 0)*DD,
                     dextents<int,2>{DD, qrows}, array<int,2>{1, DD});
    auto tK0 = tensor(&tgK[0], dextents<int,2>{DD, BC}, array<int,2>{1, DD});
    using TT = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;
    using KT = __tensor_ops_detail::__remove_addrspace_t<decltype(tK0)>;

    auto ctS = qk_op.get_destination_cooperative_tensor<TT, KT, @DT@>();
    using CtP = decltype(pv_op.get_left_input_cooperative_tensor<@DT@, @DT@, float>());
    auto ctO = pv_op.get_destination_cooperative_tensor<CtP, KT, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i)
        if (ctO.is_valid_element(i)) ctO[i] = 0.0f;

    // Row state in lane registers: lane r (r < SR) owns row r of this SG.
    float m_row = -INFINITY, l_row = 0.0f;

    const int shift = Lk - Lq;
    const int jmax = causal ? min(Lk, q0tg + SR*NSG + shift) : Lk;

    // Preload tile 0 into buffer 0 (K/V beyond Lk load as exact zeros: K
    // zeros are masked, V zeros multiply an exact-0 P).
    {
        const int nv0 = min(BC, Lk) * (DD * 2 / 16);
        device const uint4* sK = (device const uint4*)Kp;
        device const uint4* sV = (device const uint4*)Vp;
        threadgroup uint4* dK = (threadgroup uint4*)&tgK[0];
        threadgroup uint4* dV = (threadgroup uint4*)&tgV[0];
        #pragma clang loop unroll(full)
        for (int t = 0; t < PFV; ++t) {
            const int i = int(tid) + t*NSG*32;
            if (i < NVEC) {
                const bool ok = i < nv0;
                dK[i] = ok ? sK[i] : uint4(0);
                dV[i] = ok ? sV[i] : uint4(0);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int p = 0;
    for (int j0 = 0; j0 < jmax; j0 += BC) {
        // Prefetch the NEXT tile's K/V into registers before computing this
        // one — the device-load latency hides behind the matmuls below.
        uint4 kreg[PFV], vreg[PFV];
        const int j1 = j0 + BC;
        const bool hasnext = j1 < jmax;
        if (hasnext) {
            const int nv1 = min(BC, Lk - j1) * (DD * 2 / 16);
            device const uint4* sK = (device const uint4*)(Kp + ulong(j1)*DD);
            device const uint4* sV = (device const uint4*)(Vp + ulong(j1)*DD);
            #pragma clang loop unroll(full)
            for (int t = 0; t < PFV; ++t) {
                const int i = int(tid) + t*NSG*32;
                const bool ok = i < nv1;
                kreg[t] = ok ? sK[i] : uint4(0);
                vreg[t] = ok ? sV[i] : uint4(0);
            }
        }

        auto tK = tensor(&tgK[p*BC*DD], dextents<int,2>{DD, BC}, array<int,2>{1, DD});
        auto tV = tensor(&tgV[p*BC*DD], dextents<int,2>{DD, BC}, array<int,2>{1, DD});

        const bool sg_do = active && (!causal || j0 <= q0 + SR - 1 + shift);
        if (sg_do) {
            qk_op.run(mQ, tK, ctS);

            // Mask (bounds + causal) in place; raw scores stay in @DT@.
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {
                if (!ctS.is_valid_element(i)) continue;
                auto idx = ctS.get_multidimensional_index(i);
                const int jj = j0 + int(idx[0]);
                const int ii = q0 + int(idx[1]);
                if (!((jj < Lk) && (ii < Lq) && (!causal || jj <= ii + shift)))
                    ctS[i] = @DT@(-INFINITY);
            }

            // Block row-max (raw): official row reduction, then gather the
            // per-row values into owner lanes (lane r holds row r).
            // Block row-max (raw scores): official row reduction, then a
            // UNIFORM owner-lane gather (simd ops must not sit behind
            // per-element validity branches).
            auto ctMb = qk_op.get_row_reduction_destination_cooperative_tensor<TT, KT, @DT@>();
            reduce_rows(ctS, ctMb, reduction_operation::max, @DT@(-INFINITY));
            float mb_row = -INFINITY;
            for (ushort r = 0; r < SR; ++r) {
                float loc = -INFINITY;
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < ctMb.get_capacity(); ++i) {
                    if (ctMb.is_valid_element(i)
                        && int(ctMb.get_multidimensional_index(i)[0]) == int(r))
                        loc = max(loc, float(ctMb[i]));
                }
                const float cand = simd_max(loc);
                if (int(lane) == int(r)) mb_row = max(mb_row, cand);
            }

            // Owner-lane running-max update; corr factor c stays in the lane.
            const float mn_row = max(m_row, mb_row * scale);
            const float c_row = (mn_row == -INFINITY) ? 1.0f : exp(m_row - mn_row);
            m_row = mn_row;

            // P = exp(s*scale - m), rounded to @DT@ in place. Shuffles are
            // hoisted out of the validity guard for uniform execution.
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {
                const bool val = ctS.is_valid_element(i);
                ushort src = 0;
                if (val) src = ushort(ctS.get_multidimensional_index(i)[1]);
                const float mn = simd_shuffle(m_row, src);
                if (val) {
                    const float s = float(ctS[i]);
                    ctS[i] = (mn == -INFINITY) ? @DT@(0.0f)
                                               : @DT@(exp(s * scale - mn));
                }
            }

            // Block row-sum of P; merge into running l in the owner lane.
            // NOTE: reduction-dest elements hold each row's COMPLETE result,
            // replicated across the lanes that held that row's elements (the
            // v2r TG-overwrite pattern relies on this) — so the gather must
            // pick a representative (simd_max of identical copies), never sum.
            auto ctLb = qk_op.get_row_reduction_destination_cooperative_tensor<TT, KT, @DT@>();
            reduce_rows(ctS, ctLb, reduction_operation::sum, @DT@(0.0f));
            float lb_row = 0.0f;
            for (ushort r = 0; r < SR; ++r) {
                float loc = -INFINITY;
                #pragma clang loop unroll(full)
                for (uint16_t i = 0; i < ctLb.get_capacity(); ++i) {
                    if (ctLb.is_valid_element(i)
                        && int(ctLb.get_multidimensional_index(i)[0]) == int(r))
                        loc = max(loc, float(ctLb[i]));
                }
                const float cand = simd_max(loc);   // row sums are >= 0
                if (int(lane) == int(r) && cand > -INFINITY) lb_row = cand;
            }
            l_row = l_row * c_row + lb_row;

            // Rescale running O by the row's corr factor, then accumulate PV.
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
                const bool val = ctO.is_valid_element(i);
                ushort src = 0;
                if (val) src = ushort(ctO.get_multidimensional_index(i)[1]);
                const float c = simd_shuffle(c_row, src);
                if (val) ctO[i] *= c;
            }
            auto ctP = pv_op.get_left_input_cooperative_tensor<@DT@, @DT@, float>(ctS);
            pv_op.run(ctP, tV, ctO);
        }

        if (!DB)   // single buffer: wait for all reads before overwriting
            threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hasnext) {
            const int pn = DB ? (p ^ 1) : 0;
            threadgroup uint4* dK = (threadgroup uint4*)&tgK[pn*BC*DD];
            threadgroup uint4* dV = (threadgroup uint4*)&tgV[pn*BC*DD];
            #pragma clang loop unroll(full)
            for (int t = 0; t < PFV; ++t) {
                const int i = int(tid) + t*NSG*32;
                if (i < NVEC) { dK[i] = kreg[t]; dV[i] = vreg[t]; }
            }
            p = pn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!active) return;   // whole simdgroup: uniform (no barriers follow)
    const float inv_row = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    device @DT@* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        const bool val = ctO.is_valid_element(i);
        ushort row = 0;
        int cc = 0;
        if (val) {
            auto idx = ctO.get_multidimensional_index(i);
            row = ushort(idx[1]);
            cc = int(idx[0]);
        }
        const float inv = simd_shuffle(inv_row, row);
        if (val) {
            const int r = q0 + int(row);
            if (r < Lq && cc < DD)
                Ob[ulong(r)*DD + cc] = @DT@(ctO[i] * inv);
        }
    }
}

// ── na: per-simdgroup independent flash, DrawThings NAX recipe ──
// Design source: liuliu/ccv lib/nnc/mfa/kernels/NAAttentionKernel.cpp
// (BSD-3-Clause, Copyright (c) 2010 Liu Liu) — the shipped M5 config:
//   * NSG independent simdgroups per TG, each runs its OWN flash over
//     NASR=16 query rows with execution_simdgroups<1> ops — one SG's
//     softmax latency hides behind 15 siblings' matmuls; the hot loop
//     has NO threadgroup barriers (only simdgroup_barrier for the
//     per-SG-private P slice);
//   * K/V loaded straight from device memory (no TG staging — the MPP
//     guide's M5 rule);
//   * TWO NABC=64 column blocks share ONE online-softmax update;
//   * running max/sum/correction live in row-reduction cooperative
//     tensors (row results replicated across lanes — elementwise math,
//     no shuffles); map_iterator broadcasts row stats to score elements;
//   * fast::exp2 with log2(e)*scale folded in (exp2-domain throughout);
//   * relaxed_precision templated (@NARP@; DrawThings ships true).
// Host gates: non-causal, Lk % 128 == 0 (static-extent K/V tiles; no
// tail path in v1 — everything else falls back to the cooperative kernel).
constant constexpr int NASR  = 16;   // query rows per simdgroup (native tile M)
constant constexpr int NABC  = @NABCV@;  // KV block width
constant constexpr int NABLK = @NABLK@;  // column blocks per softmax pass (1|2)
constant constexpr int NANSG = @NANSG@;
constant constexpr int NAKB  = @NAKB@;   // head-dim blocks (1: D<=128; 2: D=256)
constant constexpr int NAHB  = DD / NAKB;

kernel void flash_attn_fwd_na(
    device @DT@* Q  [[buffer(0)]],
    device @DT@* K  [[buffer(1)]],
    device @DT@* V  [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal,LqStride,LkStride]
    constant float* PR [[buffer(5)]],  // [scale]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=@NAHQ@, Hkv=@NAHKV@, Lq=@NALQ@, Lk=@NALK@;
    // Padded per-head row strides: the host zero-pads ragged L dims to tile
    // boundaries, so buffer strides differ from the TRUE lengths used for
    // masking/loops. Mixing them up shifts every head after the first.
    const int LqS=@NALQS@, LkS=@NALKS@;
    const float dot_scale = PR[0] * 1.4426950408889634f;   // exp2 domain

#if @NAMORTON@
    // Interleave query-tile and head bits for locality across neighboring
    // threadgroups. The CCV/DrawThings NA backend uses this traversal too.
    // Each SIMD group still computes the same rows in the same order.
    const uint qgroups = uint((Lq + NANSG * NASR - 1) / (NANSG * NASR));
    const uint qbits = qgroups > 1u ? 32u - clz(qgroups - 1u) : 0u;
    const uint hbits = Hq > 1 ? 32u - clz(uint(Hq - 1)) : 0u;
    uint code = tgid.x, tile = 0u, head = 0u;
    for (uint bit = 0; bit < max(qbits, hbits); ++bit) {
        if (bit < qbits) { tile |= (code & 1u) << bit; code >>= 1; }
        if (bit < hbits) { head |= (code & 1u) << bit; code >>= 1; }
    }
    if (tile >= qgroups || head >= uint(Hq)) return;
    const int b = int(tgid.y), hq = int(head), bh = b * Hq + hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0 = (int(tile) * NANSG + int(sgid)) * NASR;
#else
    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = (int(tgid.x) * NANSG + int(sgid)) * NASR;
#endif

#if !@NABYP@
    // Per-SG-private P slice; 16 SGs x 16x64 x 2B = 32KB (the full budget).
    threadgroup @DT@ tgPall[NANSG * NASR * NABC];
    threadgroup @DT@* P_buf = tgPall + NASR * NABC * int(sgid);
#endif

#if !@NASYNC@
    if (q0 >= Lq) return;   // uniform per SG; no threadgroup barriers follow
#endif

    // Head strides use the PADDED lengths (LqS/LkS): the host zero-pads
    // ragged L dims, so buffer geometry differs from the true Lq/Lk used
    // for masking — mixing them shifts every head after the first.
    device @DT@* Qp = Q + ulong(bh)*ulong(LqS)*DD;
    device @DT@* Kp = K + ulong(b*Hkv+hkv)*ulong(LkS)*DD;
    device @DT@* Vp = V + ulong(b*Hkv+hkv)*ulong(LkS)*DD;

    constexpr auto na_qk_desc = matmul2d_descriptor(
        NASR, NABC, DD, false, true, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<na_qk_desc, execution_simdgroups<1>> qk_op;
    constexpr auto na_pv_desc = matmul2d_descriptor(
        NASR, NAHB, NABC, false, false, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<na_pv_desc, execution_simdgroups<1>> pv_op;

    // Parent tensors carry the true bounds; STATIC-extent slices feed the
    // matmuls (no per-access bounds math — the Apple MPP guide's M5 rule;
    // OOB rows of the q-tail slice are clamped by the parent's extents).
    auto Qt = tensor<device @DT@, dextents<int,2>, tensor_inline>(Qp, dextents<int,2>(DD, LqS));
    auto Kt = tensor<device @DT@, dextents<int,2>, tensor_inline>(Kp, dextents<int,2>(DD, LkS));
    auto Vt = tensor<device @DT@, dextents<int,2>, tensor_inline>(Vp, dextents<int,2>(DD, LkS));
    auto mQ = Qt.slice<DD, NASR>(0, q0);
    auto mK = Kt.slice<DD, NABC>(0, 0);
    auto mV = Vt.slice<NAHB, NABC>(0, 0);
    using TQ = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;
    using TK = __tensor_ops_detail::__remove_addrspace_t<decltype(mK)>;
    using TV = __tensor_ops_detail::__remove_addrspace_t<decltype(mV)>;
#if !@NABYP@
    auto tP = tensor(P_buf, dextents<int,2>{NABC, NASR}, array<int,2>{1, NABC});
    using TP = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
#endif

    auto cS0 = qk_op.get_destination_cooperative_tensor<TQ, TK, float>();
#if @NABLK@ == 2
    auto cS1 = qk_op.get_destination_cooperative_tensor<TQ, TK, float>();
#endif
    auto cM   = qk_op.get_row_reduction_destination_cooperative_tensor<TQ, TK, float>();
    auto cL   = qk_op.get_row_reduction_destination_cooperative_tensor<TQ, TK, float>();
    auto cCor = qk_op.get_row_reduction_destination_cooperative_tensor<TQ, TK, float>();
#if @NABYP@
    // Register chain: P never touches threadgroup memory — converted in
    // registers into the PV left-input cooperative tensor (same element
    // enumeration as the QK destination; parity-checked by the probe).
    auto cP  = pv_op.get_left_input_cooperative_tensor<@DT@, @DT@, float>();
    // remove_addrspace: decltype on a local is `thread`-qualified, and the
    // macOS 27 beta 26A5388g headers' enable_if rejects qualified operand
    // types (same break as int8_gemm's ct_a/ct_b).
    auto ctO0 = pv_op.get_destination_cooperative_tensor<
        __tensor_ops_detail::__remove_addrspace_t<decltype(cP)>, TV, float>();
#if @NAKB@ == 2
    auto ctO1 = pv_op.get_destination_cooperative_tensor<
        __tensor_ops_detail::__remove_addrspace_t<decltype(cP)>, TV, float>();
#endif
#else
    auto ctO0 = pv_op.get_destination_cooperative_tensor<TP, TV, float>();
#if @NAKB@ == 2
    auto ctO1 = pv_op.get_destination_cooperative_tensor<TP, TV, float>();
#endif
#endif

    #pragma clang loop unroll(full)
    for (uint16_t k = 0; k < cM.get_capacity(); ++k) {
        if (cM.is_valid_element(k)) {
            cM[k] = -INFINITY;
            cL[k] = numeric_limits<float>::denorm_min();
        }
    }

    for (int c = 0; c < Lk; c += NABLK*NABC) {
#if @NASYNC@
        // All query groups participate, including zero-padded tail groups.
        // Keep the long H3 traversal in phase without changing its reductions.
        if (c % 1024 == 0) threadgroup_barrier(mem_flags::mem_none);
#endif
        // Init S (accumulate-mode matmul adds Q.K^T onto the init value):
        // 0 for valid columns, -inf for tail columns beyond Lk — the -inf
        // survives the accumulate and masks the padding keys (DrawThings'
        // C_remainder recipe).
        const bool tail = (c + NABLK * NABC > Lk);
        if (tail) {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    auto idx = cS0.get_multidimensional_index(k);
                    cS0[k] = (c + int(idx[0]) < Lk) ? 0.0f : -INFINITY;
#if @NABLK@ == 2
                    cS1[k] = (c + NABC + int(idx[0]) < Lk) ? 0.0f : -INFINITY;
#endif
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    cS0[k] = 0.0f;
#if @NABLK@ == 2
                    cS1[k] = 0.0f;
#endif
                }
            }
        }
        auto mK0 = Kt.slice<DD, NABC>(0, c);
        qk_op.run(mQ, mK0, cS0);
#if @NABLK@ == 2
        auto mK1 = Kt.slice<DD, NABC>(0, c + NABC);
        qk_op.run(mQ, mK1, cS1);
#endif

        // Diagnostic modes (@NADIAG@, wrong results): 1 = no softmax,
        // 2 = QK only, 3 = QK + PV but no P write/barrier (stale P).
#if @NADIAG@ != 0 && !@NABYP@ && @NABLK@ == 2 && @NAKB@ == 1
        if (@NADIAG@ != 0) {
            if (c == 0) {
                #pragma clang loop unroll(full)
                for (uint16_t k = 0; k < ctO0.get_capacity(); ++k)
                    if (ctO0.is_valid_element(k)) ctO0[k] = 0.0f;
            }
            if (@NADIAG@ == 2) continue;
            if (@NADIAG@ == 1) {
                #pragma clang loop unroll(full)
                for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                    if (cS0.is_valid_element(k)) {
                        auto idx = cS0.get_multidimensional_index(k);
                        P_buf[idx[0] + idx[1]*NABC] = @DT@(cS0[k]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            auto mVd0 = Vt.slice<NAHB, NABC>(0, c);
            pv_op.run(tP, mVd0, ctO0);
            if (@NADIAG@ == 1) {
                #pragma clang loop unroll(full)
                for (uint16_t k = 0; k < cS1.get_capacity(); ++k) {
                    if (cS1.is_valid_element(k)) {
                        auto idx = cS1.get_multidimensional_index(k);
                        P_buf[idx[0] + idx[1]*NABC] = @DT@(cS1[k]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            auto mVd1 = Vt.slice<NAHB, NABC>(0, c + NABC);
            pv_op.run(tP, mVd1, ctO0);
            continue;
        }
#endif

        // One online-softmax update shared by both column blocks.
        auto cM0 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ, TK, float>();
        reduce_rows(cS0, cM0, reduction_operation::max, -INFINITY);
#if @NABLK@ == 2
        auto cM1 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ, TK, float>();
        reduce_rows(cS1, cM1, reduction_operation::max, -INFINITY);
#endif
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cM.get_capacity(); ++k) {
            if (cM.is_valid_element(k)) {
                cCor[k] = 1.0f;
#if @NABLK@ == 2
                const float Mn = max(cM0[k], cM1[k]) * dot_scale;
#else
                const float Mn = cM0[k] * dot_scale;
#endif
                if (Mn > cM[k]) {
                    cCor[k] = fast::exp2(cM[k] - Mn);
                    cM[k] = Mn;
                }
            }
        }
        // S -> P = exp2(s*dot_scale - m) in place (fp32 registers).
        if (tail) {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    auto it = cS0.get_iterator(k);
                    auto mit = cM.map_iterator(it);
                    auto idx = cS0.get_multidimensional_index(k);
                    cS0[k] = (c + int(idx[0]) < Lk)
                        ? fast::exp2(cS0[k] * dot_scale - *mit) : 0.0f;
#if @NABLK@ == 2
                    cS1[k] = (c + NABC + int(idx[0]) < Lk)
                        ? fast::exp2(cS1[k] * dot_scale - *mit) : 0.0f;
#endif
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    auto it = cS0.get_iterator(k);
                    auto mit = cM.map_iterator(it);
                    cS0[k] = fast::exp2(cS0[k] * dot_scale - *mit);
#if @NABLK@ == 2
                    cS1[k] = fast::exp2(cS1[k] * dot_scale - *mit);
#endif
                }
            }
        }
        auto cL0 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ, TK, float>();
        reduce_rows(cS0, cL0, reduction_operation::sum, 0.0f);
#if @NABLK@ == 2
        auto cL1 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ, TK, float>();
        reduce_rows(cS1, cL1, reduction_operation::sum, 0.0f);
#endif
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cL.get_capacity(); ++k) {
            if (cL.is_valid_element(k)) {
#if @NABLK@ == 2
                cL[k] = cL[k] * cCor[k] + cL0[k] + cL1[k];
#else
                cL[k] = cL[k] * cCor[k] + cL0[k];
#endif
            }
        }

        // Rescale running O (first pass: init instead — cM was -inf, and
        // fast::exp2(-inf - Mn) is not trusted under fast math). NOTE: a
        // "skip when correction==1" simd-vote branch here measured 2x SLOWER
        // (divergent flow around cooperative tensors spills registers) —
        // keep the sweep unconditional.
        if (c == 0) {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < ctO0.get_capacity(); ++k) {
                if (ctO0.is_valid_element(k)) {
                    ctO0[k] = 0.0f;
#if @NAKB@ == 2
                    ctO1[k] = 0.0f;
#endif
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < ctO0.get_capacity(); ++k) {
                if (ctO0.is_valid_element(k)) {
                    auto it = ctO0.get_iterator(k);
                    auto cit = cCor.map_iterator(it);
                    ctO0[k] *= *cit;
#if @NAKB@ == 2
                    ctO1[k] *= *cit;
#endif
                }
            }
        }

#if @NABYP@
        // P blocks stay in registers: convert -> PV accumulate, twice.
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS0.get_capacity(); ++k)
            if (cS0.is_valid_element(k)) cP[k] = @DT@(cS0[k]);
        simdgroup_barrier(mem_flags::mem_none);
        auto mV0a = Vt.slice<NAHB, NABC>(0, c);
        pv_op.run(cP, mV0a, ctO0);
#if @NAKB@ == 2
        auto mV0b = Vt.slice<NAHB, NABC>(NAHB, c);
        pv_op.run(cP, mV0b, ctO1);
#endif
#if @NABLK@ == 2
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS1.get_capacity(); ++k)
            if (cS1.is_valid_element(k)) cP[k] = @DT@(cS1[k]);
        auto mV1a = Vt.slice<NAHB, NABC>(0, c + NABC);
        pv_op.run(cP, mV1a, ctO0);
#if @NAKB@ == 2
        auto mV1b = Vt.slice<NAHB, NABC>(NAHB, c + NABC);
        pv_op.run(cP, mV1b, ctO1);
#endif
#endif
    }
#else
        // P block 0 -> private TG slice -> PV accumulate; then block 1.
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
            if (cS0.is_valid_element(k)) {
                auto idx = cS0.get_multidimensional_index(k);
                P_buf[idx[0] + idx[1]*NABC] = @DT@(cS0[k]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto mV0a = Vt.slice<NAHB, NABC>(0, c);
        pv_op.run(tP, mV0a, ctO0);
#if @NAKB@ == 2
        auto mV0b = Vt.slice<NAHB, NABC>(NAHB, c);
        pv_op.run(tP, mV0b, ctO1);
#endif

#if @NABLK@ == 2
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS1.get_capacity(); ++k) {
            if (cS1.is_valid_element(k)) {
                auto idx = cS1.get_multidimensional_index(k);
                P_buf[idx[0] + idx[1]*NABC] = @DT@(cS1[k]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto mV1a = Vt.slice<NAHB, NABC>(0, c + NABC);
        pv_op.run(tP, mV1a, ctO0);
#if @NAKB@ == 2
        auto mV1b = Vt.slice<NAHB, NABC>(NAHB, c + NABC);
        pv_op.run(tP, mV1b, ctO1);
#endif
#endif
    }
#endif

    device @DT@* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t k = 0; k < ctO0.get_capacity(); ++k) {
        if (!ctO0.is_valid_element(k)) continue;
        auto idx = ctO0.get_multidimensional_index(k);
        const int row = int(idx[1]);
        const int col = int(idx[0]);
        if (q0 + row >= Lq || col >= NAHB) continue;
        auto it = ctO0.get_iterator(k);
        auto lit = cL.map_iterator(it);
        const float invl = fast::divide(1.0f, *lit);
        Ob[ulong(q0 + row)*DD + col] = @DT@(ctO0[k] * invl);
#if @NAKB@ == 2
        Ob[ulong(q0 + row)*DD + NAHB + col] = @DT@(ctO1[k] * invl);
#endif
    }
}

// ── na_fp8qk: block-scaled FP8 QK, bf16/fp16 PV (experimental) ──
// Q and K are quantized once per native attention tile: [16,D] for Q and
// [NABC,D] for K.  The QK TensorOp consumes e4m3 directly and accumulates f32;
// scale products are folded into the exp2-domain scores.  P and V remain the
// input dtype, so this experiment isolates FP8 QK from the much more sensitive
// value path.  It is intentionally not routed by attn_router_mps.py.

inline uchar na_fp8_e4m3fn(float value) {
    const bool negative = signbit(value);
    float x = min(fabs(value), 448.0f);
    if (x == 0.0f) return negative ? uchar(0x80) : uchar(0);

    int exponent = 0;
    const float fraction = frexp(x, exponent);  // x=fraction*2^exponent
    int biased = exponent + 6;                  // (exponent-1)+bias(7)
    int mantissa = 0;
    if (biased <= 0) {
        // e4m3fn subnormal quantum is 2^-9. rint is ties-to-even.
        mantissa = int(rint(x * 512.0f));
        biased = 0;
        if (mantissa >= 8) { biased = 1; mantissa = 0; }
    } else {
        const float normal = ldexp(fraction, 1); // [1,2)
        mantissa = int(rint((normal - 1.0f) * 8.0f));
        if (mantissa >= 8) { ++biased; mantissa = 0; }
        // exp=15,mantissa=7 is NaN; 0x7e is the finite max (448).
        if (biased >= 15) { biased = 15; mantissa = min(mantissa, 6); }
    }
    const uchar bits = uchar((biased << 3) | mantissa);
    return negative ? uchar(bits | 0x80) : bits;
}

kernel void na_fp8_quant(
    device @DT@* X [[buffer(0)]],
    device uchar* X8 [[buffer(1)]],
    device float* SCL [[buffer(2)]],       // [BH, ntiles]
    constant int* QP [[buffer(3)]],        // [LS,Ltrue,tile,ntiles]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[8];
    const int LS=QP[0], Ltrue=QP[1], TS=QP[2], ntiles=QP[3];
    const int tile=int(tgid.x), bh=int(tgid.y), r0=tile*TS;
    const int extent=max(0, min(TS, Ltrue-r0));
    const ulong base=ulong(bh)*ulong(LS)*DD + ulong(r0)*DD;
    const int valid=extent*DD, capacity=TS*DD;

    float local_max=0.0f;
    for (int i=int(tid); i<valid; i+=256)
        local_max=max(local_max, fabs(float(X[base+i])));
    local_max=max(local_max, simd_shuffle_xor(local_max, 16));
    local_max=max(local_max, simd_shuffle_xor(local_max, 8));
    local_max=max(local_max, simd_shuffle_xor(local_max, 4));
    local_max=max(local_max, simd_shuffle_xor(local_max, 2));
    local_max=max(local_max, simd_shuffle_xor(local_max, 1));
    if (lane == 0) scratch[sgid]=local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float absmax=scratch[0];
    for (int s=1; s<8; ++s) absmax=max(absmax, scratch[s]);
    const float scale=absmax > 0.0f ? absmax/448.0f : 1.0f/448.0f;
    const float inv=absmax > 0.0f ? 448.0f/absmax : 448.0f;
    if (tid == 0) SCL[ulong(bh)*ntiles+tile]=scale;
    for (int i=int(tid); i<capacity; i+=256)
        X8[base+i]=(i < valid) ? na_fp8_e4m3fn(float(X[base+i])*inv) : uchar(0);
}

kernel void flash_attn_fwd_na_fp8qk(
    device uchar* Q8 [[buffer(0)]],
    device uchar* K8 [[buffer(1)]],
    device @DT@* V [[buffer(2)]],
    device @DT@* O [[buffer(3)]],
    constant int* SH [[buffer(4)]],
    constant float* PR [[buffer(5)]],
    device float* QSCL [[buffer(6)]],
    device float* KSCL [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4];
    const int LqS=SH[6], LkS=SH[7];
    const float dot_scale=PR[0]*1.4426950408889634f;
    const int bh=int(tgid.y), b=bh/Hq, hq=bh%Hq;
    const int hkv=hq/(Hq/Hkv), bhkv=b*Hkv+hkv;
    const int q0=(int(tgid.x)*NANSG+int(sgid))*NASR;
#if !@NABYP@
    threadgroup @DT@ tgPall[NANSG*NASR*NABC];
    threadgroup @DT@* P_buf=tgPall+NASR*NABC*int(sgid);
#endif
    if (q0 >= Lq) return;

    device uchar* Qp=Q8+ulong(bh)*ulong(LqS)*DD;
    device uchar* Kp=K8+ulong(bhkv)*ulong(LkS)*DD;
    device @DT@* Vp=V+ulong(bhkv)*ulong(LkS)*DD;
    using fp8_t=metal::metal_fp8_e4m3_format;
    constexpr auto qk_desc=matmul2d_descriptor(
        NASR, NABC, DD, false, true, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<1>> qk_op;
    constexpr auto pv_desc=matmul2d_descriptor(
        NASR, DD, NABC, false, false, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<1>> pv_op;

    auto Qt=tensor<device fp8_t,dextents<int,2>,tensor_inline>(
        Qp, dextents<int,2>(DD,LqS));
    auto Kt=tensor<device fp8_t,dextents<int,2>,tensor_inline>(
        Kp, dextents<int,2>(DD,LkS));
    auto Vt=tensor<device @DT@,dextents<int,2>,tensor_inline>(
        Vp, dextents<int,2>(DD,LkS));
    auto mQ=Qt.slice<DD,NASR>(0,q0);
    auto mK=Kt.slice<DD,NABC>(0,0);
    auto mV=Vt.slice<DD,NABC>(0,0);
    using TQ=__tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;
    using TK=__tensor_ops_detail::__remove_addrspace_t<decltype(mK)>;
    using TV=__tensor_ops_detail::__remove_addrspace_t<decltype(mV)>;
#if !@NABYP@
    auto tP=tensor(P_buf,dextents<int,2>{NABC,NASR},array<int,2>{1,NABC});
    using TP=__tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
#endif
    auto cS0=qk_op.get_destination_cooperative_tensor<TQ,TK,float>();
#if @NABLK@ == 2
    auto cS1=qk_op.get_destination_cooperative_tensor<TQ,TK,float>();
#endif
    auto cM=qk_op.get_row_reduction_destination_cooperative_tensor<TQ,TK,float>();
    auto cL=qk_op.get_row_reduction_destination_cooperative_tensor<TQ,TK,float>();
    auto cCor=qk_op.get_row_reduction_destination_cooperative_tensor<TQ,TK,float>();
#if @NABYP@
    auto cP=pv_op.get_left_input_cooperative_tensor<@DT@,@DT@,float>();
    auto ctO=pv_op.get_destination_cooperative_tensor<
        __tensor_ops_detail::__remove_addrspace_t<decltype(cP)>,TV,float>();
#else
    auto ctO=pv_op.get_destination_cooperative_tensor<TP,TV,float>();
#endif
    #pragma clang loop unroll(full)
    for (uint16_t i=0; i<cM.get_capacity(); ++i) if (cM.is_valid_element(i)) {
        cM[i]=-INFINITY; cL[i]=numeric_limits<float>::denorm_min();
    }

    const int qtiles=LqS/NASR, kvtiles=LkS/NABC;
    const float qscale=QSCL[ulong(bh)*qtiles+q0/NASR];
    for (int c=0; c<Lk; c+=NABLK*NABC) {
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cS0.get_capacity(); ++i) if (cS0.is_valid_element(i)) {
            cS0[i]=0.0f;
#if @NABLK@ == 2
            cS1[i]=0.0f;
#endif
        }
        auto mK0=Kt.slice<DD,NABC>(0,c);
        qk_op.run(mQ,mK0,cS0);
#if @NABLK@ == 2
        auto mK1=Kt.slice<DD,NABC>(0,c+NABC);
        qk_op.run(mQ,mK1,cS1);
#endif
        const bool tail=c+NABLK*NABC>Lk;
        const float f0=qscale*KSCL[ulong(bhkv)*kvtiles+c/NABC]*dot_scale;
#if @NABLK@ == 2
        const int t1=min(c/NABC+1,kvtiles-1);
        const float f1=qscale*KSCL[ulong(bhkv)*kvtiles+t1]*dot_scale;
#endif
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cS0.get_capacity(); ++i) if (cS0.is_valid_element(i)) {
            auto idx=cS0.get_multidimensional_index(i);
            cS0[i]=(!tail || c+int(idx[0])<Lk) ? cS0[i]*f0 : -INFINITY;
#if @NABLK@ == 2
            cS1[i]=(!tail || c+NABC+int(idx[0])<Lk) ? cS1[i]*f1 : -INFINITY;
#endif
        }

        auto cM0=qk_op.get_row_reduction_destination_cooperative_tensor<TQ,TK,float>();
        reduce_rows(cS0,cM0,reduction_operation::max,-INFINITY);
#if @NABLK@ == 2
        auto cM1=qk_op.get_row_reduction_destination_cooperative_tensor<TQ,TK,float>();
        reduce_rows(cS1,cM1,reduction_operation::max,-INFINITY);
#endif
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cM.get_capacity(); ++i) if (cM.is_valid_element(i)) {
            cCor[i]=1.0f;
#if @NABLK@ == 2
            const float mn=max(cM0[i],cM1[i]);
#else
            const float mn=cM0[i];
#endif
            if (mn>cM[i]) { cCor[i]=fast::exp2(cM[i]-mn); cM[i]=mn; }
        }
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cS0.get_capacity(); ++i) if (cS0.is_valid_element(i)) {
            auto it=cS0.get_iterator(i); auto mit=cM.map_iterator(it);
            auto idx=cS0.get_multidimensional_index(i);
            cS0[i]=(!tail || c+int(idx[0])<Lk)
                ? fast::exp2(cS0[i]-*mit) : 0.0f;
#if @NABLK@ == 2
            cS1[i]=(!tail || c+NABC+int(idx[0])<Lk)
                ? fast::exp2(cS1[i]-*mit) : 0.0f;
#endif
        }
        auto cL0=qk_op.get_row_reduction_destination_cooperative_tensor<TQ,TK,float>();
        reduce_rows(cS0,cL0,reduction_operation::sum,0.0f);
#if @NABLK@ == 2
        auto cL1=qk_op.get_row_reduction_destination_cooperative_tensor<TQ,TK,float>();
        reduce_rows(cS1,cL1,reduction_operation::sum,0.0f);
#endif
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cL.get_capacity(); ++i) if (cL.is_valid_element(i)) {
#if @NABLK@ == 2
            cL[i]=cL[i]*cCor[i]+cL0[i]+cL1[i];
#else
            cL[i]=cL[i]*cCor[i]+cL0[i];
#endif
        }
        if (c==0) {
            #pragma clang loop unroll(full)
            for (uint16_t i=0; i<ctO.get_capacity(); ++i)
                if (ctO.is_valid_element(i)) ctO[i]=0.0f;
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t i=0; i<ctO.get_capacity(); ++i) if (ctO.is_valid_element(i)) {
                auto it=ctO.get_iterator(i); auto cit=cCor.map_iterator(it);
                ctO[i]*=*cit;
            }
        }
#if @NABYP@
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cS0.get_capacity(); ++i)
            if (cS0.is_valid_element(i)) cP[i]=@DT@(cS0[i]);
        simdgroup_barrier(mem_flags::mem_none);
        auto mV0=Vt.slice<DD,NABC>(0,c);
        pv_op.run(cP,mV0,ctO);
#if @NABLK@ == 2
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cS1.get_capacity(); ++i)
            if (cS1.is_valid_element(i)) cP[i]=@DT@(cS1[i]);
        auto mV1=Vt.slice<DD,NABC>(0,c+NABC);
        pv_op.run(cP,mV1,ctO);
#endif
#else
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cS0.get_capacity(); ++i) if (cS0.is_valid_element(i)) {
            auto idx=cS0.get_multidimensional_index(i);
            P_buf[idx[0]+idx[1]*NABC]=@DT@(cS0[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto mV0=Vt.slice<DD,NABC>(0,c);
        pv_op.run(tP,mV0,ctO);
#if @NABLK@ == 2
        #pragma clang loop unroll(full)
        for (uint16_t i=0; i<cS1.get_capacity(); ++i) if (cS1.is_valid_element(i)) {
            auto idx=cS1.get_multidimensional_index(i);
            P_buf[idx[0]+idx[1]*NABC]=@DT@(cS1[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto mV1=Vt.slice<DD,NABC>(0,c+NABC);
        pv_op.run(tP,mV1,ctO);
#endif
#endif
    }
    device @DT@* Ob=O+ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i=0; i<ctO.get_capacity(); ++i) if (ctO.is_valid_element(i)) {
        auto idx=ctO.get_multidimensional_index(i);
        const int row=int(idx[1]), col=int(idx[0]);
        if (q0+row>=Lq || col>=DD) continue;
        auto it=ctO.get_iterator(i); auto lit=cL.map_iterator(it);
        Ob[ulong(q0+row)*DD+col]=@DT@(ctO[i]*fast::divide(1.0f,*lit));
    }
}

// ── na_i8: int8-quantized attention (opt-in, rounding-class, NOT bit-exact) ──
// Recipe: liuliu/ccv NAInt8AttentionKernel.cpp (BSD-3) — on-the-fly symmetric
// int8 quant of Q/K (per-row-tile scales aligned to processing tiles) and of
// mean-centered V; QK runs int8xint8->int32 (the 2x half); P stays @DT@ with
// the block's V scale folded in; PV is mixed @DT@ x int8 -> fp32 accumulate;
// V's mean adds back EXACTLY as +mean after normalization (sum P = L).
// M (row max) is reduced over @DT@-rounded scores — any M error cancels
// identically in P and L (flash invariant); exp2 args recompute from the
// exact int32 score. Tail masking via @DT@ -inf scores / P=0; int8 pad-row
// garbage is annihilated by exact integer x zero (no NaN class in int8).

// Pre-pass: per-(head, dim) mean of V over the TRUE sequence length.
// grid (DD/4, BHkv); 256 threads; float4 partial sums, simd + tg reduce.
kernel void na_i8_v_mean(
    device @DT@* V [[buffer(0)]],          // [BHkv, LkS, DD]
    device float* MEAN [[buffer(1)]],       // [BHkv, DD]
    constant int* QP [[buffer(2)]],         // [LkS, Lk, 0, 0]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    threadgroup float4 scr[8];
    const int LkS = QP[0], Lk = QP[1];
    const int vec4 = int(tgid.x);
    const ulong base = ulong(tgid.y) * ulong(LkS) * DD;
    float4 acc = 0.0f;
    for (int r = int(tid); r < Lk; r += 256) {
        device @DT@* p = V + base + ulong(r) * DD + vec4 * 4;
        acc += float4(p[0], p[1], p[2], p[3]);
    }
    acc += simd_shuffle_xor(acc, 16);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) scr[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float4 t = scr[0];
        for (int s = 1; s < 8; ++s) t += scr[s];
        t *= (1.0f / float(max(Lk, 1)));
        device float* m = MEAN + ulong(tgid.y) * DD + vec4 * 4;
        m[0] = t.x; m[1] = t.y; m[2] = t.z; m[3] = t.w;
    }
}

// Pre-pass: symmetric int8 quant per row tile: scale = absmax/127 (1/127 for
// all-zero tiles), clamp +-127. use_mean subtracts MEAN (the V path). Rows
// beyond the true length are left untouched (pad garbage is P=0-masked).
kernel void na_i8_quant(
    device @DT@* X [[buffer(0)]],           // [BH, LS, DD]
    device int8_t* X8 [[buffer(1)]],
    device float* SCL [[buffer(2)]],        // [BH, ntiles]
    device float* MEAN [[buffer(3)]],       // [BH, DD] (V only)
    constant int* QP [[buffer(4)]],         // [LS, Ltrue, tile, use_mean, ntiles]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    threadgroup float scr[8];
    const int LS = QP[0], Ltrue = QP[1], TS = QP[2], use_mean = QP[3], ntiles = QP[4];
    const int tile = int(tgid.x);
    const int bh = int(tgid.y);
    const int r0 = tile * TS;
    const int extent = max(0, min(TS, Ltrue - r0));
    const ulong base = ulong(bh) * ulong(LS) * DD + ulong(r0) * DD;
    const int total = extent * DD;
    float lmax = 0.0f;
    for (int i = int(tid); i < total; i += 256) {
        float v = float(X[base + i]);
        if (use_mean) v -= MEAN[ulong(bh) * DD + (i % DD)];
        lmax = fmax(lmax, fabs(v));
    }
    lmax = fmax(lmax, simd_shuffle_xor(lmax, 16));
    lmax = fmax(lmax, simd_shuffle_xor(lmax, 8));
    lmax = fmax(lmax, simd_shuffle_xor(lmax, 4));
    lmax = fmax(lmax, simd_shuffle_xor(lmax, 2));
    lmax = fmax(lmax, simd_shuffle_xor(lmax, 1));
    if (lane == 0) scr[sgid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = scr[0];
    for (int s = 1; s < 8; ++s) m = fmax(m, scr[s]);
    const float scale = m > 0.0f ? m / 127.0f : (1.0f / 127.0f);
    const float inv_scale = m > 0.0f ? 127.0f / m : 127.0f;
    if (tid == 0) SCL[ulong(bh) * ntiles + tile] = scale;
    for (int i = int(tid); i < total; i += 256) {
        float v = float(X[base + i]);
        if (use_mean) v -= MEAN[ulong(bh) * DD + (i % DD)];
        X8[base + i] = int8_t(clamp(rint(v * inv_scale), -127.0f, 127.0f));
    }
}

// Main kernel: clone of flash_attn_fwd_na (NAKB=1 / D<=128 only), int8 I/O.
// Structure notes (this macOS beta's MPP headers):
//   * int8 operands require an int32 matmul destination (float dest tensors
//     are CREATABLE but not runnable) — so cS0i/cS1i take the qk_op.run and
//     float destination-layout twins cS0/cS1 from the SAME op carry the
//     dequantized scores for the softmax machinery (reduce_rows/map_iterator
//     accept destination layouts only, and demand matching element types);
//   * per-block dequant factor f = q_scale * k_scale * dot_scale folds in at
//     the int->float copy; row max/L/corrections then run EXACTLY like the
//     fp16 na kernel;
//   * P converts into the @DT@ PV-left tensors with the block's V scale
//     folded in (after the L reduce), PV runs @DT@ x int8 -> fp32;
//   * epilogue adds V's mean back: O = ctO/L + vmean (exact: sum P = L).
kernel void flash_attn_fwd_na_i8(
    device int8_t* Q8 [[buffer(0)]],
    device int8_t* K8 [[buffer(1)]],
    device int8_t* V8 [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal,LqStride,LkStride]
    constant float* PR [[buffer(5)]],  // [scale]
    device float* QSCL [[buffer(6)]],  // [BHq, LqS/16]
    device float* KSCL [[buffer(7)]],  // [BHkv, LkS/NABC]
    device float* VSCL [[buffer(8)]],  // [BHkv, LkS/NABC]
    device float* VMEAN [[buffer(9)]], // [BHkv, DD]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4];
    const int LqS=SH[6], LkS=SH[7];
    const float dot_scale = PR[0] * 1.4426950408889634f;   // exp2 domain

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int bhkv = b * Hkv + hkv;
    const int q0  = (int(tgid.x) * NANSG + int(sgid)) * NASR;

    if (q0 >= Lq) return;   // uniform per SG; no threadgroup barriers follow

    device int8_t* Qp = Q8 + ulong(bh)*ulong(LqS)*DD;
    device int8_t* Kp = K8 + ulong(bhkv)*ulong(LkS)*DD;
    device int8_t* Vp = V8 + ulong(bhkv)*ulong(LkS)*DD;
    const int kv_tiles = LkS / NABC;
    const float q_scale = QSCL[ulong(bh) * (LqS / NASR) + (q0 / NASR)];

    constexpr auto nai_qk_desc = matmul2d_descriptor(
        NASR, NABC, @NAIK@, false, true, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<nai_qk_desc, execution_simdgroups<1>> qk_op;
    constexpr auto nai_pv_desc = matmul2d_descriptor(
        NASR, DD, NABC, false, false, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<nai_pv_desc, execution_simdgroups<1>> pv_op;

    auto Qt = tensor<device int8_t, dextents<int,2>, tensor_inline>(Qp, dextents<int,2>(DD, LqS));
    auto Kt = tensor<device int8_t, dextents<int,2>, tensor_inline>(Kp, dextents<int,2>(DD, LkS));
    auto Vt = tensor<device int8_t, dextents<int,2>, tensor_inline>(Vp, dextents<int,2>(DD, LkS));
    auto mQ = Qt.slice<@NAIK@, NASR>(0, q0);
    auto mK = Kt.slice<@NAIK@, NABC>(0, 0);
    auto mV = Vt.slice<DD, NABC>(0, 0);
    using TQ8 = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;
    using TK8 = __tensor_ops_detail::__remove_addrspace_t<decltype(mK)>;
    using TV8 = __tensor_ops_detail::__remove_addrspace_t<decltype(mV)>;

    auto cS0i = qk_op.get_destination_cooperative_tensor<TQ8, TK8, int>();
    auto cS0  = qk_op.get_destination_cooperative_tensor<TQ8, TK8, float>();
#if @NABLK@ == 2
    auto cS1i = qk_op.get_destination_cooperative_tensor<TQ8, TK8, int>();
    auto cS1  = qk_op.get_destination_cooperative_tensor<TQ8, TK8, float>();
#endif
    auto cM   = qk_op.get_row_reduction_destination_cooperative_tensor<TQ8, TK8, float>();
    auto cL   = qk_op.get_row_reduction_destination_cooperative_tensor<TQ8, TK8, float>();
    auto cCor = qk_op.get_row_reduction_destination_cooperative_tensor<TQ8, TK8, float>();
    // Register chain: P converts in registers into the PV left tensors (the
    // float QK-destination twin and the @DT@ PV-left share element
    // enumeration — same identity the fp16 byp path relies on).
    auto cP  = pv_op.get_left_input_cooperative_tensor<@DT@, int8_t, float>();
    auto ctO0 = pv_op.get_destination_cooperative_tensor<
        __tensor_ops_detail::__remove_addrspace_t<decltype(cP)>, TV8, float>();

    #pragma clang loop unroll(full)
    for (uint16_t k = 0; k < cM.get_capacity(); ++k) {
        if (cM.is_valid_element(k)) {
            cM[k] = -INFINITY;
            cL[k] = numeric_limits<float>::denorm_min();
        }
    }

    for (int c = 0; c < Lk; c += NABLK*NABC) {
        const bool tail = (c + NABLK * NABC > Lk);
        const float f0 = q_scale * KSCL[ulong(bhkv) * kv_tiles + (c / NABC)] * dot_scale;
        const float vs0 = VSCL[ulong(bhkv) * kv_tiles + (c / NABC)];
#if @NABLK@ == 2
        const int t1 = min(c / NABC + 1, kv_tiles - 1);
        const float f1 = q_scale * KSCL[ulong(bhkv) * kv_tiles + t1] * dot_scale;
        const float vs1 = VSCL[ulong(bhkv) * kv_tiles + t1];
#endif
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS0i.get_capacity(); ++k) {
            if (cS0i.is_valid_element(k)) {
                cS0i[k] = 0;
#if @NABLK@ == 2
                cS1i[k] = 0;
#endif
            }
        }
        for (int kk = 0; kk < DD; kk += @NAIK@) {
            auto mQs = Qt.slice<@NAIK@, NASR>(kk, q0);
            auto mK0 = Kt.slice<@NAIK@, NABC>(kk, c);
            qk_op.run(mQs, mK0, cS0i);
#if @NABLK@ == 2
            auto mK1 = Kt.slice<@NAIK@, NABC>(kk, c + NABC);
            qk_op.run(mQs, mK1, cS1i);
#endif
        }

        // Dequantize into the float twins (same-op destination layouts share
        // enumeration); tail columns -> -inf so max/exp2 mask them.
        if (tail) {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    auto idx = cS0.get_multidimensional_index(k);
                    cS0[k] = (c + int(idx[0]) < Lk)
                        ? float(cS0i[k]) * f0 : -INFINITY;
#if @NABLK@ == 2
                    cS1[k] = (c + NABC + int(idx[0]) < Lk)
                        ? float(cS1i[k]) * f1 : -INFINITY;
#endif
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    cS0[k] = float(cS0i[k]) * f0;
#if @NABLK@ == 2
                    cS1[k] = float(cS1i[k]) * f1;
#endif
                }
            }
        }

        // One online-softmax update shared by both column blocks — identical
        // to the fp16 na kernel from here on (dot_scale already folded).
        auto cM0 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ8, TK8, float>();
        reduce_rows(cS0, cM0, reduction_operation::max, -INFINITY);
#if @NABLK@ == 2
        auto cM1 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ8, TK8, float>();
        reduce_rows(cS1, cM1, reduction_operation::max, -INFINITY);
#endif
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cM.get_capacity(); ++k) {
            if (cM.is_valid_element(k)) {
                cCor[k] = 1.0f;
#if @NABLK@ == 2
                const float Mn = max(cM0[k], cM1[k]);
#else
                const float Mn = cM0[k];
#endif
                if (Mn > cM[k]) {
                    cCor[k] = fast::exp2(cM[k] - Mn);
                    cM[k] = Mn;
                }
            }
        }
        // S -> P = exp2(s - m) in place (fp32 registers; s already scaled).
        if (tail) {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    auto it = cS0.get_iterator(k);
                    auto mit = cM.map_iterator(it);
                    auto idx = cS0.get_multidimensional_index(k);
                    cS0[k] = (c + int(idx[0]) < Lk)
                        ? fast::exp2(cS0[k] - *mit) : 0.0f;
#if @NABLK@ == 2
                    cS1[k] = (c + NABC + int(idx[0]) < Lk)
                        ? fast::exp2(cS1[k] - *mit) : 0.0f;
#endif
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
                if (cS0.is_valid_element(k)) {
                    auto it = cS0.get_iterator(k);
                    auto mit = cM.map_iterator(it);
                    cS0[k] = fast::exp2(cS0[k] - *mit);
#if @NABLK@ == 2
                    cS1[k] = fast::exp2(cS1[k] - *mit);
#endif
                }
            }
        }
        auto cL0 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ8, TK8, float>();
        reduce_rows(cS0, cL0, reduction_operation::sum, 0.0f);
#if @NABLK@ == 2
        auto cL1 = qk_op.get_row_reduction_destination_cooperative_tensor<TQ8, TK8, float>();
        reduce_rows(cS1, cL1, reduction_operation::sum, 0.0f);
#endif
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cL.get_capacity(); ++k) {
            if (cL.is_valid_element(k)) {
#if @NABLK@ == 2
                cL[k] = cL[k] * cCor[k] + cL0[k] + cL1[k];
#else
                cL[k] = cL[k] * cCor[k] + cL0[k];
#endif
            }
        }

        if (c == 0) {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < ctO0.get_capacity(); ++k)
                if (ctO0.is_valid_element(k)) ctO0[k] = 0.0f;
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < ctO0.get_capacity(); ++k) {
                if (ctO0.is_valid_element(k)) {
                    auto it = ctO0.get_iterator(k);
                    auto cit = cCor.map_iterator(it);
                    ctO0[k] *= *cit;
                }
            }
        }

        // P (x the block's V scale) into the PV-left tensor, PV accumulate.
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS0.get_capacity(); ++k)
            if (cS0.is_valid_element(k)) cP[k] = @DT@(cS0[k] * vs0);
        simdgroup_barrier(mem_flags::mem_none);
        auto mV0 = Vt.slice<DD, NABC>(0, c);
        pv_op.run(cP, mV0, ctO0);
#if @NABLK@ == 2
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS1.get_capacity(); ++k)
            if (cS1.is_valid_element(k)) cP[k] = @DT@(cS1[k] * vs1);
        auto mV1 = Vt.slice<DD, NABC>(0, c + NABC);
        pv_op.run(cP, mV1, ctO0);
#endif
    }

    device @DT@* Ob = O + ulong(bh)*ulong(Lq)*DD;
    device float* Mnp = VMEAN + ulong(bhkv) * DD;
    #pragma clang loop unroll(full)
    for (uint16_t k = 0; k < ctO0.get_capacity(); ++k) {
        if (!ctO0.is_valid_element(k)) continue;
        auto idx = ctO0.get_multidimensional_index(k);
        const int row = int(idx[1]);
        const int col = int(idx[0]);
        if (q0 + row >= Lq || col >= DD) continue;
        auto it = ctO0.get_iterator(k);
        auto lit = cL.map_iterator(it);
        const float invl = fast::divide(1.0f, *lit);
        Ob[ulong(q0 + row)*DD + col] = @DT@(ctO0[k] * invl + Mnp[col]);
    }
}

// probe: can the (single-simdgroup, bfloat) QK destination feed PV as left input?
kernel void v2r_compat_bf16(
    device bfloat* DUMMY [[buffer(0)]],
    device int*    OUT   [[buffer(1)]])
{
    constexpr auto qk_desc_r = matmul2d_descriptor(
        SR, BCV, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc_r, execution_simdgroup> qk_op;
    constexpr auto pv_desc_r = matmul2d_descriptor(
        SR, DD, BCV, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc_r, execution_simdgroup> pv_op;

    auto t = tensor(DUMMY, dextents<int,2>{DD, SR}, array<int,2>{1, DD});
    using TT = __tensor_ops_detail::__remove_addrspace_t<decltype(t)>;
    auto ctS = qk_op.get_destination_cooperative_tensor<TT, TT, bfloat>();
    OUT[0] = pv_op.is_compatible_as_left_input<bfloat, bfloat, float>(ctS) ? 1 : 0;
}
)MTL";

// ── Host: per-D library/pipeline cache ───────────────────────────
static std::map<std::string, id<MTLComputePipelineState>> g_pso;
static std::string g_compile_error;

// BR query rows per threadgroup; NSG = BR/8 simdgroups (8 rows per SG) capped [4,16].
static int nsg_for(int BR) { return BR <= 32 ? 4 : (BR <= 64 ? 8 : 16); }

static id<MTLComputePipelineState> pso_for(int D, int BR, int BC, bool bf16s,
                                           int SR = 16, int BCV = 64,
                                           const char* fn_name = "flash_attn_fwd_bf16",
                                           bool pv_static = false, int nsg = 0,
                                           bool half_io = false, bool db = false,
                                           bool na_rp = true, bool na_byp = false,
                                           int na_bc = 64, int na_blk = 2,
                                           std::array<int32_t, 6> na_shape = {},
                                           bool na_morton = false) {
  const std::string function(fn_name);
  std::string key =
      std::to_string(D) + ":" + std::to_string(BR) + ":" +
      std::to_string(BC) + ":" + std::to_string(bf16s) + ":" +
      std::to_string(SR) + ":" + std::to_string(BCV) + ":" + function + ":" +
      std::to_string(pv_static) + ":" + std::to_string(nsg) + ":" +
      std::to_string(half_io) + ":" + std::to_string(db) + ":" +
      std::to_string(na_rp) + ":" + std::to_string(na_byp) + ":" +
      std::to_string(na_bc) + ":" + std::to_string(na_blk);
  for (int32_t value : na_shape) key += ":" + std::to_string(value);
  key += ":" + std::to_string(na_morton);
  auto it = g_pso.find(key);
  if (it != g_pso.end()) return it->second;

  asfp8::MetalTemplate source(kSrcTemplate);
  source.set("D", std::to_string(D));
  source.set("BR", std::to_string(BR));
  source.set("BC", std::to_string(BC));
  const char* dt = half_io ? "half" : "bfloat";
  source.set("DT", dt);
  source.set("ST", bf16s ? dt : "float");
  source.set("NSG", std::to_string(nsg > 0 ? nsg : nsg_for(BR)));
  // P merges into the score tile only when the tile is bfloat (same type).
  source.set("MP", bf16s ? "true" : "false");
  source.set("SR", std::to_string(SR));
  source.set("BCV", std::to_string(BCV));
  source.set("DB", db ? "true" : "false");
  // na kernel: simdgroups per TG and the relaxed_precision descriptor flag.
  source.set("NANSG", std::to_string(nsg > 0 ? nsg : 16));
  source.set("NAKB", std::to_string(D > 128 ? 2 : 1));
  source.set("NARP", na_rp ? "true" : "false");
  source.set("NABYP", na_byp ? "1" : "0");
  source.set("NABCV", std::to_string(na_bc));
  source.set("NABLK", std::to_string(na_blk));
  // Long H3: specialize true lengths AND padded strides. This removes
  // dynamic index/bounds work without changing attention arithmetic.
  const bool na_specialized = na_shape[0] > 0;
  source.set("NAMORTON", na_morton ? "1" : "0");
  source.set("NASYNC", na_morton ? "1" : "0");
  const char* shape_keys[] = {"NAHQ", "NAHKV", "NALQ", "NALK", "NALQS", "NALKS"};
  const char* shape_defaults[] = {"SH[1]", "SH[2]", "SH[3]", "SH[4]", "SH[6]", "SH[7]"};
  for (int i = 0; i < 6; ++i)
    source.set(shape_keys[i], na_specialized ? std::to_string(na_shape[i]) : shape_defaults[i]);
  const int naik = getenv("ASFP8_FA41_NAIK") ? atoi(getenv("ASFP8_FA41_NAIK")) : 0;
  source.set("NAIK", naik > 0 ? std::to_string(naik) : std::to_string(D));
  // diagnostic: ASFP8_FA41_NA_DIAG = 1 no-softmax / 2 QK-only / 3 no-P-roundtrip
  const int nadiag = getenv("ASFP8_FA41_NA_DIAG") ? atoi(getenv("ASFP8_FA41_NA_DIAG")) : 0;
  source.set("NADIAG", std::to_string(nadiag));
  // Static reduction dims where safe: qk's inner dim is always exactly D;
  // pv's inner dim (BC) is static only when the host proved Lk % BC == 0
  // (a static-k op reads the full tile — a short tail would read OOB).
  // ASFP8_FA41_DYNK=1 reverts both to dynamic for A/B.
  const bool dynk = getenv("ASFP8_FA41_DYNK") && atoi(getenv("ASFP8_FA41_DYNK"));
  source.set("QKS", dynk ? "static_cast<int>(dynamic_extent)" : "DD");
  source.set("PVS", (pv_static && !dynk) ? "BC" : "static_cast<int>(dynamic_extent)");
  std::string src = std::move(source).finish();

  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                         options:opts error:&err];
  if (!lib) {
    g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
    TORCH_CHECK(false, "flash41 Metal library compile failed (D=", D, ", BR=", BR, ", BC=", BC, "): ", g_compile_error);
  }
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
  TORCH_CHECK(fn, fn_name, " not found in compiled library");
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "flash41 pipeline creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_pso[key] = pso;
  return pso;
}

// Cached on-device probe: can the single-simdgroup bfloat QK destination feed
// PV as a left input at this (D, SR, BCV)? (the v2r register-resident-P recipe)
static std::map<long, bool> g_v2r_ok;

static bool v2r_probe(int D, int SR, int BCV) {
  const long key = ((long)D * 100 + SR) * 1000 + BCV;
  auto it = g_v2r_ok.find(key);
  if (it != g_v2r_ok.end()) return it->second;
  bool ok = false;
  try {
    id<MTLComputePipelineState> pso = pso_for(D, 32, 128, false, SR, BCV, "v2r_compat_bf16");
    auto dummy = torch::zeros({32 * (long)D},
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kMPS));
    auto flag = torch::zeros({1},
        torch::TensorOptions().dtype(torch::kInt).device(torch::kMPS));
    id<MTLBuffer> dB = __builtin_bit_cast(id<MTLBuffer>, dummy.storage().data());
    id<MTLBuffer> fB = __builtin_bit_cast(id<MTLBuffer>, flag.storage().data());
    MPSStream* stream = getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^(){
      @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
        [enc setComputePipelineState:pso];
        [enc setBuffer:dB offset:0 atIndex:0];
        [enc setBuffer:fB offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }
    });
    ok = flag.item<int>() != 0;   // .item() syncs the stream (probe runs once)
  } catch (...) {
    ok = false;
  }
  g_v2r_ok[key] = ok;
  return ok;
}

// O[B,Hq,Lq,D] bf16 = flash_attention(Q, K, V). GQA via Hq % Hkv == 0.
// v2r: 0 off (default), 1 force on (errors if ineligible), -1 auto (on when
// Lk is a BCV multiple and the device probe passes). Default OFF because
// register-resident P LOSES at D=128 (measured M5 Max, flux-edit shapes:
// best v2r config SR16/BCV32 = 20.1 TF/s vs 29.8 for the cooperative TG
// kernel, with 5x the error from bfloat S) — the single-simdgroup constraint
// forfeits the 4-SG cooperative matmul that makes BC=128 tiles fast.
torch::Tensor flash_fwd_bf16(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             double scale, bool causal, int64_t br, int64_t bc, bool bf16s,
                             int64_t v2r, int64_t sr, int64_t bcv, int64_t nsg_ovr) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type(),
              "inputs must be bf16 or fp16 (matching)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "inputs must be [B,H,L,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1), Lq = (int)q.size(2), D = (int)q.size(3);
  const int Hkv = (int)k.size(1), Lk = (int)k.size(2);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head_dim mismatch");
  TORCH_CHECK(v.size(2) == Lk && k.size(0) == B && v.size(0) == B, "kv shape mismatch");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 8 == 0 && D <= 128, "head_dim must be a multiple of 8, <= 128");

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto out = torch::empty({B, Hq, Lq, D}, q.options());

  // Params go via setBytes — creating tiny CPU tensors and .to(mps)-ing them
  // per call stalls the pipeline (measured +6s on a Krea render).
  std::array<int32_t, 6> shArr = {B, Hq, Hkv, Lq, Lk, causal ? 1 : 0};
  std::array<float, 1> prArr = {(float)scale};

  int BR = (int)br;
  // BR=32 is compute-optimal at image seqs; at video seqs (>=12k) the kernel
  // turns K/V-bandwidth-bound and BR=64 halves K/V re-reads (measured M5 Max
  // fp16 mergeP: 21.5k tok 442 -> 386 ms; crossover between 8.7k and 12.8k).
  if (BR <= 0) BR = (Lk >= 12288) ? 64 : 32;
  int BC = (int)bc;
  if (BC <= 0) BC = 128;                      // wide KV tiles feed the matmul units
  TORCH_CHECK(BC % 32 == 0 && BC <= 256, "bc must be a multiple of 32, <= 256");
  int SR = (int)sr;
  if (SR <= 0) SR = 16;                       // native matmul granularity (8/32 collapse)
  int BCV = (int)bcv;
  if (BCV <= 0) BCV = 64;
  TORCH_CHECK(BCV % 32 == 0 && BCV <= 256, "bcv must be a multiple of 32, <= 256");

  bool use_v2r = false;
  if (half_io) {
    v2r = 0;   // the v2r experiment is bfloat-internal only
  }
  if (v2r > 0) {
    TORCH_CHECK(Lk % BCV == 0, "v2r requires Lk % BCV == 0 (static-k PV descriptor)");
    TORCH_CHECK(v2r_probe(D, SR, BCV), "v2r not supported on this device for D=", D);
    use_v2r = true;
  } else if (v2r < 0) {
    use_v2r = (Lk % BCV == 0) && Lk >= 256 && v2r_probe(D, SR, BCV);
  }

  const int nsg = (int)nsg_ovr > 0 ? (int)nsg_ovr : nsg_for(BR);
  TORCH_CHECK(BR % nsg == 0, "BR must be divisible by NSG");
  id<MTLComputePipelineState> pso =
      use_v2r ? pso_for(D, BR, BC, bf16s, SR, BCV, "flash_attn_fwd_v2r_bf16")
              : pso_for(D, BR, BC, bf16s, SR, BCV, "flash_attn_fwd_bf16",
                        /*pv_static=*/Lk % BC == 0, nsg, half_io);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const int rows_per_tg = use_v2r ? SR * nsg : BR;
  const NSUInteger gx = (NSUInteger)((Lq + rows_per_tg - 1) / rows_per_tg);
  const NSUInteger gy = (NSUInteger)(B * Hq);

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:qB offset:qO atIndex:0];
      [enc setBuffer:kB offset:kO atIndex:1];
      [enc setBuffer:vB offset:vO atIndex:2];
      [enc setBuffer:oB offset:oO atIndex:3];
      [enc setBytes:shArr.data() length:sizeof(shArr) atIndex:4];
      [enc setBytes:prArr.data() length:sizeof(prArr) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(nsg * 32), 1, 1)];
    }
  });
  // No COMMIT_AND_WAIT — encoded on torch's MPS stream (see int8_gemm.mm).
  return out;
}

// Staged-KV per-simdgroup kernel (attention rewrite phase 1).
// SR rows per simdgroup, NSG simdgroups (BR = SR*NSG rows/TG), BC-wide
// K/V tiles staged in threadgroup memory (2*BC*D*2 bytes <= 32KB).
torch::Tensor flash_fwd_frag(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             double scale, bool causal,
                             int64_t sr, int64_t nsg, int64_t bc, int64_t db) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type(),
              "inputs must be bf16 or fp16 (matching)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "inputs must be [B,H,L,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1), Lq = (int)q.size(2), D = (int)q.size(3);
  const int Hkv = (int)k.size(1), Lk = (int)k.size(2);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head_dim mismatch");
  TORCH_CHECK(v.size(2) == Lk && k.size(0) == B && v.size(0) == B, "kv shape mismatch");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 8 == 0 && D <= 128, "head_dim must be a multiple of 8, <= 128");

  int SR = (int)sr > 0 ? (int)sr : 16;
  int NSG = (int)nsg > 0 ? (int)nsg : 4;
  int BC = (int)bc > 0 ? (int)bc : 64;
  TORCH_CHECK(SR <= 32, "SR must be <= 32 (owner-lane bookkeeping)");
  TORCH_CHECK(BC % 32 == 0, "bc must be a multiple of 32");
  const bool dbuf = db != 0;
  TORCH_CHECK((dbuf ? 4 : 2) * BC * D * (int)q.element_size() <= 32768,
              "K+V tiles exceed 32KB threadgroup memory");

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto out = torch::empty({B, Hq, Lq, D}, q.options());
  std::array<int32_t, 6> shArr = {B, Hq, Hkv, Lq, Lk, causal ? 1 : 0};
  std::array<float, 1> prArr = {(float)scale};

  id<MTLComputePipelineState> pso =
      pso_for(D, SR * NSG, BC, /*bf16s=*/true, SR, 64, "flash_attn_fwd_frag",
              /*pv_static=*/false, NSG, half_io, dbuf);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const int rows_per_tg = SR * NSG;
  const NSUInteger gx = (NSUInteger)((Lq + rows_per_tg - 1) / rows_per_tg);
  const NSUInteger gy = (NSUInteger)(B * Hq);

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:qB offset:qO atIndex:0];
      [enc setBuffer:kB offset:kO atIndex:1];
      [enc setBuffer:vB offset:vO atIndex:2];
      [enc setBuffer:oB offset:oO atIndex:3];
      [enc setBytes:shArr.data() length:sizeof(shArr) atIndex:4];
      [enc setBytes:prArr.data() length:sizeof(prArr) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(NSG * 32), 1, 1)];
    }
  });
  return out;
}

// DrawThings-recipe per-simdgroup kernel: NSG independent SGs x NASR=16 rows,
// two 64-col blocks per softmax pass, K/V straight from device memory.
// Gates: non-causal, Lk % 128 == 0 (no tail path in v1).
torch::Tensor flash_fwd_na(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           double scale, int64_t nsg, int64_t rp, int64_t byp,
                           int64_t bc, int64_t blk, bool long_schedule) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type(),
              "inputs must be bf16 or fp16 (matching)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "inputs must be [B,H,L,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1), Lq = (int)q.size(2), D = (int)q.size(3);
  const int Hkv = (int)k.size(1), Lk = (int)k.size(2);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head_dim mismatch");
  TORCH_CHECK(v.size(2) == Lk && k.size(0) == B && v.size(0) == B, "kv shape mismatch");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK((D % 32 == 0 && D <= 128) || D == 256,
              "na kernel needs head_dim % 32 == 0 and <= 128, or exactly 256 "
              "(two 128-wide head blocks)");
  int BC = (int)bc > 0 ? (int)bc : 64;
  int BLK = (int)blk > 0 ? (int)blk : 2;
  TORCH_CHECK(BC % 32 == 0 && BC <= 256, "bc must be a multiple of 32, <= 256");
  TORCH_CHECK(BLK == 1 || BLK == 2, "blk must be 1 or 2");

  int NSG = (int)nsg > 0 ? (int)nsg : 16;
  TORCH_CHECK(NSG >= 1 && NSG <= 16, "nsg must be in [1,16]");
  TORCH_CHECK(byp != 0 || NSG * 16 * BC * (int)q.element_size() <= 32768,
              "P slices exceed 32KB threadgroup memory");

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  // ZERO-PAD ragged L dims to tile boundaries. The kernel's static-extent
  // slices do NOT bounds-check (that is their perf point), so a ragged Lk
  // tail read past the buffer end — heap garbage decoding as Inf/NaN made
  // 0 * garbage = NaN in the PV accumulate and intermittently corrupted
  // whole images (krea, Lk=4300). Zero pad rows are exact: pad keys are
  // -inf-masked in the score init, zero V rows meet P=0, and zero Q pad
  // rows are never stored (row-guarded epilogue). SH keeps the TRUE Lq/Lk.
  const int rows_tg = 16 * NSG;
  // Each independent simdgroup exits before loading Q when q0 >= Lq.
  // Ultra only needs its active 16-row slices padded, avoiding a full Q
  // allocation/copy at Z-Image's 4192-row shape. K/V padding is unchanged.
  static const bool ultra =
      [MPSDevice::getInstance()->device().name containsString:@"M5 Ultra"];
  const bool morton = long_schedule && ultra && D == 128 && Hq == 56 && Hkv == 56
      && Lq == Lk && Lq >= 8192 && NSG == 16 && BC == 64 && BLK == 2
      && rp != 0 && byp != 0;
  // Synchronized long H3 must keep every SIMD group alive at the barriers.
  // Pad its whole query tile; other Ultra paths retain per-group padding.
  const int q_alignment = morton ? rows_tg : (ultra ? 16 : rows_tg);
  const int64_t LqP = ((Lq + q_alignment - 1) / q_alignment) * (int64_t)q_alignment;
  const int64_t LkP = ((Lk + BC * BLK - 1) / (BC * BLK)) * (int64_t)(BC * BLK);
  if (LqP != Lq)
    qc = at::constant_pad_nd(qc, {0, 0, 0, LqP - Lq}, 0);
  if (LkP != Lk) {
    kc = at::constant_pad_nd(kc, {0, 0, 0, LkP - Lk}, 0);
    vc = at::constant_pad_nd(vc, {0, 0, 0, LkP - Lk}, 0);
  }
  auto out = torch::empty({B, Hq, Lq, D}, q.options());
  std::array<int32_t, 8> shArr = {B, Hq, Hkv, Lq, Lk, 0,
                                  (int32_t)LqP, (int32_t)LkP};
  std::array<float, 1> prArr = {(float)scale};

  // Keep shorter sequences and other attention geometries on their existing
  // path. Validated on M5 Ultra H3; no dispatch change for other chips.
  // For aligned lengths, specializing away the tail changes Metal's floating
  // point code generation (observed at L=8192). Keep the original dynamic
  // dimensions there; Morton traversal itself does not change reduction order.
  const std::array<int32_t, 6> na_shape = morton && Lq % (BC * BLK) != 0
      ? std::array<int32_t, 6>{Hq, Hkv, Lq, Lk, (int32_t)LqP, (int32_t)LkP}
      : std::array<int32_t, 6>{};
  id<MTLComputePipelineState> pso =
      pso_for(D, 16, 64, /*bf16s=*/true, 16, 64, "flash_attn_fwd_na",
              /*pv_static=*/false, NSG, half_io, /*db=*/false, rp != 0, byp != 0,
              BC, BLK, na_shape, morton);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const int rows_per_tg = 16 * NSG;
  NSUInteger gx = (NSUInteger)((Lq + rows_per_tg - 1) / rows_per_tg);
  NSUInteger gy = (NSUInteger)(B * Hq);
  if (morton) {
    NSUInteger q_extent = 1, h_extent = 1;
    while (q_extent < gx) q_extent <<= 1;
    while (h_extent < (NSUInteger)Hq) h_extent <<= 1;
    gx = q_extent * h_extent;
    gy = B;
  }

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:qB offset:qO atIndex:0];
      [enc setBuffer:kB offset:kO atIndex:1];
      [enc setBuffer:vB offset:vO atIndex:2];
      [enc setBuffer:oB offset:oO atIndex:3];
      [enc setBytes:shArr.data() length:sizeof(shArr) atIndex:4];
      [enc setBytes:prArr.data() length:sizeof(prArr) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(NSG * 32), 1, 1)];
    }
  });
  return out;
}

// Experimental block-scaled FP8 QK with the normal bf16/fp16 P x V path.
// Kept as an explicit extension entry point: no production router calls it.
torch::Tensor flash_fwd_na_fp8qk(torch::Tensor q, torch::Tensor k,
                                  torch::Tensor v, double scale,
                                  int64_t nsg, int64_t rp,
                                  int64_t byp, int64_t bc, int64_t blk) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type(),
              "inputs must be bf16 or fp16 (matching)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "inputs must be [B,H,L,D]");
  const int B=(int)q.size(0), Hq=(int)q.size(1), Lq=(int)q.size(2);
  const int D=(int)q.size(3), Hkv=(int)k.size(1), Lk=(int)k.size(2);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head_dim mismatch");
  TORCH_CHECK(v.size(2) == Lk && k.size(0) == B && v.size(0) == B,
              "kv shape mismatch");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0,
              "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 32 == 0 && D <= 128,
              "na_fp8qk needs head_dim % 32 == 0 and <= 128");
  int BC=(int)bc > 0 ? (int)bc : 64;
  int BLK=(int)blk > 0 ? (int)blk : 2;
  int NSG=(int)nsg > 0 ? (int)nsg : 16;
  TORCH_CHECK(BC % 32 == 0 && BC <= 256,
              "bc must be a multiple of 32, <= 256");
  TORCH_CHECK(BLK == 1 || BLK == 2, "blk must be 1 or 2");
  TORCH_CHECK(NSG >= 1 && NSG <= 16, "nsg must be in [1,16]");
  TORCH_CHECK(byp != 0 || NSG * 16 * BC * (int)q.element_size() <= 32768,
              "P slices exceed 32KB threadgroup memory");

  auto qc=q.contiguous(), kc=k.contiguous(), vc=v.contiguous();
  const int rows_tg=16*NSG;
  const int64_t LqP=((Lq+rows_tg-1)/rows_tg)*(int64_t)rows_tg;
  const int64_t LkP=((Lk+BC*BLK-1)/(BC*BLK))*(int64_t)(BC*BLK);
  if (LqP != Lq) qc=at::constant_pad_nd(qc,{0,0,0,LqP-Lq},0);
  if (LkP != Lk) {
    kc=at::constant_pad_nd(kc,{0,0,0,LkP-Lk},0);
    vc=at::constant_pad_nd(vc,{0,0,0,LkP-Lk},0);
  }
  auto byte_opts=q.options().dtype(torch::kByte);
  auto float_opts=q.options().dtype(torch::kFloat);
  auto q8=torch::empty({B,Hq,LqP,D},byte_opts);
  auto k8=torch::empty({B,Hkv,LkP,D},byte_opts);
  const int qtiles=(int)(LqP/16), kvtiles=(int)(LkP/BC);
  auto qscl=torch::empty({(long)B*Hq,qtiles},float_opts);
  auto kscl=torch::empty({(long)B*Hkv,kvtiles},float_opts);
  auto out=torch::empty({B,Hq,Lq,D},q.options());
  std::array<int32_t,8> shArr={B,Hq,Hkv,Lq,Lk,0,(int32_t)LqP,(int32_t)LkP};
  std::array<float,1> prArr={(float)scale};
  std::array<int32_t,4> qqP={(int32_t)LqP,Lq,16,qtiles};
  std::array<int32_t,4> kqP={(int32_t)LkP,Lk,BC,kvtiles};

  id<MTLComputePipelineState> psoQuant=pso_for(
      D,16,64,true,16,64,"na_fp8_quant",false,NSG,half_io,false,
      rp != 0,byp != 0,BC,BLK);
  id<MTLComputePipelineState> psoMain=pso_for(
      D,16,64,true,16,64,"flash_attn_fwd_na_fp8qk",false,NSG,half_io,
      false,rp != 0,byp != 0,BC,BLK);
  MPSStream* stream=getCurrentMPSStream();
  id<MTLBuffer> qB=__builtin_bit_cast(id<MTLBuffer>,qc.storage().data());
  id<MTLBuffer> kB=__builtin_bit_cast(id<MTLBuffer>,kc.storage().data());
  id<MTLBuffer> vB=__builtin_bit_cast(id<MTLBuffer>,vc.storage().data());
  id<MTLBuffer> q8B=__builtin_bit_cast(id<MTLBuffer>,q8.storage().data());
  id<MTLBuffer> k8B=__builtin_bit_cast(id<MTLBuffer>,k8.storage().data());
  id<MTLBuffer> qsB=__builtin_bit_cast(id<MTLBuffer>,qscl.storage().data());
  id<MTLBuffer> ksB=__builtin_bit_cast(id<MTLBuffer>,kscl.storage().data());
  id<MTLBuffer> oB=__builtin_bit_cast(id<MTLBuffer>,out.storage().data());
  const NSUInteger qO=qc.storage_offset()*qc.element_size();
  const NSUInteger kO=kc.storage_offset()*kc.element_size();
  const NSUInteger vO=vc.storage_offset()*vc.element_size();
  const NSUInteger oO=out.storage_offset()*out.element_size();
  const NSUInteger gx=(NSUInteger)((Lq+rows_tg-1)/rows_tg);
  const NSUInteger gyq=(NSUInteger)(B*Hq), gykv=(NSUInteger)(B*Hkv);
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc=stream->commandEncoder();
      [enc setComputePipelineState:psoQuant];
      [enc setBuffer:qB offset:qO atIndex:0];
      [enc setBuffer:q8B offset:0 atIndex:1];
      [enc setBuffer:qsB offset:0 atIndex:2];
      [enc setBytes:qqP.data() length:sizeof(qqP) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)qtiles,gyq,1)
          threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [enc setBuffer:kB offset:kO atIndex:0];
      [enc setBuffer:k8B offset:0 atIndex:1];
      [enc setBuffer:ksB offset:0 atIndex:2];
      [enc setBytes:kqP.data() length:sizeof(kqP) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kvtiles,gykv,1)
          threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [enc setComputePipelineState:psoMain];
      [enc setBuffer:q8B offset:0 atIndex:0];
      [enc setBuffer:k8B offset:0 atIndex:1];
      [enc setBuffer:vB offset:vO atIndex:2];
      [enc setBuffer:oB offset:oO atIndex:3];
      [enc setBytes:shArr.data() length:sizeof(shArr) atIndex:4];
      [enc setBytes:prArr.data() length:sizeof(prArr) atIndex:5];
      [enc setBuffer:qsB offset:0 atIndex:6];
      [enc setBuffer:ksB offset:0 atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake(gx,gyq,1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(NSG*32),1,1)];
    }
  });
  return out;
}

// int8-quantized na attention (rounding-class, NOT bit-exact): on-the-fly
// symmetric int8 quant of Q/K and mean-centered V (per-row-tile scales),
// int8 QK -> int32, @DT@ P x int8 V -> fp32, +vmean epilogue. Opt-in route.
torch::Tensor flash_fwd_na_i8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              double scale, int64_t nsg, int64_t rp,
                              int64_t bc, int64_t blk) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type(),
              "inputs must be bf16 or fp16 (matching)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "inputs must be [B,H,L,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1), Lq = (int)q.size(2), D = (int)q.size(3);
  const int Hkv = (int)k.size(1), Lk = (int)k.size(2);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head_dim mismatch");
  TORCH_CHECK(v.size(2) == Lk && k.size(0) == B && v.size(0) == B, "kv shape mismatch");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 32 == 0 && D <= 128, "na_i8 needs head_dim % 32 == 0 and <= 128");
  int BC = (int)bc > 0 ? (int)bc : 64;
  int BLK = (int)blk > 0 ? (int)blk : 2;
  TORCH_CHECK(BC % 32 == 0 && BC <= 256, "bc must be a multiple of 32, <= 256");
  TORCH_CHECK(BLK == 1 || BLK == 2, "blk must be 1 or 2");
  int NSG = (int)nsg > 0 ? (int)nsg : 16;
  TORCH_CHECK(NSG >= 1 && NSG <= 16, "nsg must be in [1,16]");

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  // Same zero-padding rules as flash_fwd_na (SH keeps the TRUE Lq/Lk). Pad
  // rows of the int8 buffers stay unquantized garbage — exact-integer P=0
  // annihilates them (no NaN class in int8); every SCALE slot is written
  // (grids cover the padded tile count) so no fp garbage enters the math.
  const int rows_tg = 16 * NSG;
  const int64_t LqP = ((Lq + rows_tg - 1) / rows_tg) * (int64_t)rows_tg;
  const int64_t LkP = ((Lk + BC * BLK - 1) / (BC * BLK)) * (int64_t)(BC * BLK);
  if (LqP != Lq)
    qc = at::constant_pad_nd(qc, {0, 0, 0, LqP - Lq}, 0);
  if (LkP != Lk) {
    kc = at::constant_pad_nd(kc, {0, 0, 0, LkP - Lk}, 0);
    vc = at::constant_pad_nd(vc, {0, 0, 0, LkP - Lk}, 0);
  }
  auto opts_i8 = q.options().dtype(torch::kChar);
  auto opts_f = q.options().dtype(torch::kFloat);
  auto q8 = torch::empty({B, Hq, LqP, D}, opts_i8);
  auto k8 = torch::empty({B, Hkv, LkP, D}, opts_i8);
  auto v8 = torch::empty({B, Hkv, LkP, D}, opts_i8);
  const int q_tiles = (int)(LqP / 16);
  const int kv_tiles = (int)(LkP / BC);
  auto qscl = torch::empty({(long)B * Hq, q_tiles}, opts_f);
  auto kscl = torch::empty({(long)B * Hkv, kv_tiles}, opts_f);
  auto vscl = torch::empty({(long)B * Hkv, kv_tiles}, opts_f);
  auto vmean = torch::empty({(long)B * Hkv, D}, opts_f);
  auto out = torch::empty({B, Hq, Lq, D}, q.options());

  std::array<int32_t, 8> shArr = {B, Hq, Hkv, Lq, Lk, 0,
                                  (int32_t)LqP, (int32_t)LkP};
  std::array<float, 1> prArr = {(float)scale};
  std::array<int32_t, 4> vmP = {(int32_t)LkP, Lk, 0, 0};
  std::array<int32_t, 5> qqP = {(int32_t)LqP, Lq, 16, 0, q_tiles};
  std::array<int32_t, 5> kqP = {(int32_t)LkP, Lk, BC, 0, kv_tiles};
  std::array<int32_t, 5> vqP = {(int32_t)LkP, Lk, BC, 1, kv_tiles};

  id<MTLComputePipelineState> psoMean =
      pso_for(D, 16, 64, true, 16, 64, "na_i8_v_mean", false, NSG, half_io,
              false, rp != 0, true, BC, BLK);
  id<MTLComputePipelineState> psoQuant =
      pso_for(D, 16, 64, true, 16, 64, "na_i8_quant", false, NSG, half_io,
              false, rp != 0, true, BC, BLK);
  id<MTLComputePipelineState> psoMain =
      pso_for(D, 16, 64, true, 16, 64, "flash_attn_fwd_na_i8", false, NSG,
              half_io, false, rp != 0, true, BC, BLK);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> q8B = __builtin_bit_cast(id<MTLBuffer>, q8.storage().data());
  id<MTLBuffer> k8B = __builtin_bit_cast(id<MTLBuffer>, k8.storage().data());
  id<MTLBuffer> v8B = __builtin_bit_cast(id<MTLBuffer>, v8.storage().data());
  id<MTLBuffer> qsB = __builtin_bit_cast(id<MTLBuffer>, qscl.storage().data());
  id<MTLBuffer> ksB = __builtin_bit_cast(id<MTLBuffer>, kscl.storage().data());
  id<MTLBuffer> vsB = __builtin_bit_cast(id<MTLBuffer>, vscl.storage().data());
  id<MTLBuffer> vmB = __builtin_bit_cast(id<MTLBuffer>, vmean.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const NSUInteger gx = (NSUInteger)((Lq + rows_tg - 1) / rows_tg);
  const NSUInteger gyq = (NSUInteger)(B * Hq);
  const NSUInteger gykv = (NSUInteger)(B * Hkv);

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      // v mean (true Lk rows only)
      [enc setComputePipelineState:psoMean];
      [enc setBuffer:vB offset:vO atIndex:0];
      [enc setBuffer:vmB offset:0 atIndex:1];
      [enc setBytes:vmP.data() length:sizeof(vmP) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(D / 4), gykv, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // quantize Q (16-row tiles)
      [enc setComputePipelineState:psoQuant];
      [enc setBuffer:qB offset:qO atIndex:0];
      [enc setBuffer:q8B offset:0 atIndex:1];
      [enc setBuffer:qsB offset:0 atIndex:2];
      [enc setBuffer:vmB offset:0 atIndex:3];   // unused (use_mean=0)
      [enc setBytes:qqP.data() length:sizeof(qqP) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_tiles, gyq, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      // quantize K (BC-row tiles)
      [enc setBuffer:kB offset:kO atIndex:0];
      [enc setBuffer:k8B offset:0 atIndex:1];
      [enc setBuffer:ksB offset:0 atIndex:2];
      [enc setBytes:kqP.data() length:sizeof(kqP) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_tiles, gykv, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      // quantize V (mean-centered, BC-row tiles)
      [enc setBuffer:vB offset:vO atIndex:0];
      [enc setBuffer:v8B offset:0 atIndex:1];
      [enc setBuffer:vsB offset:0 atIndex:2];
      [enc setBytes:vqP.data() length:sizeof(vqP) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_tiles, gykv, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // main attention
      [enc setComputePipelineState:psoMain];
      [enc setBuffer:q8B offset:0 atIndex:0];
      [enc setBuffer:k8B offset:0 atIndex:1];
      [enc setBuffer:v8B offset:0 atIndex:2];
      [enc setBuffer:oB offset:oO atIndex:3];
      [enc setBytes:shArr.data() length:sizeof(shArr) atIndex:4];
      [enc setBytes:prArr.data() length:sizeof(prArr) atIndex:5];
      [enc setBuffer:qsB offset:0 atIndex:6];
      [enc setBuffer:ksB offset:0 atIndex:7];
      [enc setBuffer:vsB offset:0 atIndex:8];
      [enc setBuffer:vmB offset:0 atIndex:9];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gyq, 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(NSG * 32), 1, 1)];
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_fwd_na_fp8qk", &flash_fwd_na_fp8qk,
        "block-scaled fp8 QK + bf16/fp16 PV na flash fwd (experimental)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("nsg") = 16, py::arg("rp") = 1,
        py::arg("byp") = 1, py::arg("bc") = 64, py::arg("blk") = 2);
  m.def("flash_fwd_na_i8", &flash_fwd_na_i8,
        "int8-quantized na flash fwd (DrawThings NAInt8 recipe; rounding-class)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("nsg") = 16, py::arg("rp") = 1,
        py::arg("bc") = 64, py::arg("blk") = 2);
  m.def("flash_fwd_bf16", &flash_fwd_bf16,
        "bf16 flash attention fwd (Metal 4.1 TensorOps) on MPS",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("causal") = false, py::arg("br") = 0, py::arg("bc") = 0, py::arg("bf16s") = false,
        py::arg("v2r") = 0, py::arg("sr") = 0, py::arg("bcv") = 0, py::arg("nsg") = 0);
  m.def("flash_fwd_frag", &flash_fwd_frag,
        "staged-KV per-simdgroup flash attention fwd (Metal 4.1 TensorOps) on MPS",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("causal") = false, py::arg("sr") = 0, py::arg("nsg") = 0,
        py::arg("bc") = 0, py::arg("db") = 0);
  m.def("flash_fwd_na", &flash_fwd_na,
        "per-simdgroup independent flash fwd (DrawThings NAX recipe) on MPS",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("nsg") = 16, py::arg("rp") = 1, py::arg("byp") = 1,
        py::arg("bc") = 64, py::arg("blk") = 2,
        py::arg("long_schedule") = true);
  m.def("warmup", [](int64_t D, int64_t BR, int64_t BC) { pso_for((int)D, (int)BR, (int)BC, false); return true; },
        "compile + build pipeline for head dim D / BR rows / BC cols",
        py::arg("D") = 128, py::arg("BR") = 32, py::arg("BC") = 128);
  m.def("v2r_supported", [](int64_t D, int64_t SR, int64_t BCV) {
          return v2r_probe((int)D, (int)SR, (int)BCV);
        }, "device probe: register-resident-P support at (D, SR, BCV)",
        py::arg("D") = 128, py::arg("SR") = 16, py::arg("BCV") = 64);
}
