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

kernel void flash_attn_fwd_na(
    device @DT@* Q  [[buffer(0)]],
    device @DT@* K  [[buffer(1)]],
    device @DT@* V  [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal]
    constant float* PR [[buffer(5)]],  // [scale]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4];
    const float dot_scale = PR[0] * 1.4426950408889634f;   // exp2 domain

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = (int(tgid.x) * NANSG + int(sgid)) * NASR;

#if !@NABYP@
    // Per-SG-private P slice; 16 SGs x 16x64 x 2B = 32KB (the full budget).
    threadgroup @DT@ tgPall[NANSG * NASR * NABC];
    threadgroup @DT@* P_buf = tgPall + NASR * NABC * int(sgid);
#endif

    if (q0 >= Lq) return;   // uniform per SG; no threadgroup barriers follow

    device @DT@* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device @DT@* Kp = K + ulong(b*Hkv+hkv)*ulong(Lk)*DD;
    device @DT@* Vp = V + ulong(b*Hkv+hkv)*ulong(Lk)*DD;

    constexpr auto na_qk_desc = matmul2d_descriptor(
        NASR, NABC, DD, false, true, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<na_qk_desc, execution_simdgroups<1>> qk_op;
    constexpr auto na_pv_desc = matmul2d_descriptor(
        NASR, DD, NABC, false, false, @NARP@,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<na_pv_desc, execution_simdgroups<1>> pv_op;

    // Parent tensors carry the true bounds; STATIC-extent slices feed the
    // matmuls (no per-access bounds math — the Apple MPP guide's M5 rule;
    // OOB rows of the q-tail slice are clamped by the parent's extents).
    auto Qt = tensor<device @DT@, dextents<int,2>, tensor_inline>(Qp, dextents<int,2>(DD, Lq));
    auto Kt = tensor<device @DT@, dextents<int,2>, tensor_inline>(Kp, dextents<int,2>(DD, Lk));
    auto Vt = tensor<device @DT@, dextents<int,2>, tensor_inline>(Vp, dextents<int,2>(DD, Lk));
    auto mQ = Qt.slice<DD, NASR>(0, q0);
    auto mK = Kt.slice<DD, NABC>(0, 0);
    auto mV = Vt.slice<DD, NABC>(0, 0);
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
    auto ctO = pv_op.get_destination_cooperative_tensor<decltype(cP), TV, float>();
#else
    auto ctO = pv_op.get_destination_cooperative_tensor<TP, TV, float>();
#endif

    #pragma clang loop unroll(full)
    for (uint16_t k = 0; k < cM.get_capacity(); ++k) {
        if (cM.is_valid_element(k)) {
            cM[k] = -INFINITY;
            cL[k] = numeric_limits<float>::denorm_min();
        }
    }

    for (int c = 0; c < Lk; c += NABLK*NABC) {
        // Zero-init S (accumulate-mode matmul writes S += Q.K^T).
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS0.get_capacity(); ++k) {
            if (cS0.is_valid_element(k)) {
                cS0[k] = 0.0f;
#if @NABLK@ == 2
                cS1[k] = 0.0f;
#endif
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
#if @NADIAG@ != 0 && !@NABYP@ && @NABLK@ == 2
        if (@NADIAG@ != 0) {
            if (c == 0) {
                #pragma clang loop unroll(full)
                for (uint16_t k = 0; k < ctO.get_capacity(); ++k)
                    if (ctO.is_valid_element(k)) ctO[k] = 0.0f;
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
            auto mVd0 = Vt.slice<DD, NABC>(0, c);
            pv_op.run(tP, mVd0, ctO);
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
            auto mVd1 = Vt.slice<DD, NABC>(0, c + NABC);
            pv_op.run(tP, mVd1, ctO);
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
            for (uint16_t k = 0; k < ctO.get_capacity(); ++k)
                if (ctO.is_valid_element(k)) ctO[k] = 0.0f;
        } else {
            #pragma clang loop unroll(full)
            for (uint16_t k = 0; k < ctO.get_capacity(); ++k) {
                if (ctO.is_valid_element(k)) {
                    auto it = ctO.get_iterator(k);
                    auto cit = cCor.map_iterator(it);
                    ctO[k] *= *cit;
                }
            }
        }

#if @NABYP@
        // P blocks stay in registers: convert -> PV accumulate, twice.
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS0.get_capacity(); ++k)
            if (cS0.is_valid_element(k)) cP[k] = @DT@(cS0[k]);
        simdgroup_barrier(mem_flags::mem_none);
        auto mV0 = Vt.slice<DD, NABC>(0, c);
        pv_op.run(cP, mV0, ctO);
#if @NABLK@ == 2
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS1.get_capacity(); ++k)
            if (cS1.is_valid_element(k)) cP[k] = @DT@(cS1[k]);
        auto mV1 = Vt.slice<DD, NABC>(0, c + NABC);
        pv_op.run(cP, mV1, ctO);
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
        auto mV0 = Vt.slice<DD, NABC>(0, c);
        pv_op.run(tP, mV0, ctO);

#if @NABLK@ == 2
        #pragma clang loop unroll(full)
        for (uint16_t k = 0; k < cS1.get_capacity(); ++k) {
            if (cS1.is_valid_element(k)) {
                auto idx = cS1.get_multidimensional_index(k);
                P_buf[idx[0] + idx[1]*NABC] = @DT@(cS1[k]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto mV1 = Vt.slice<DD, NABC>(0, c + NABC);
        pv_op.run(tP, mV1, ctO);
#endif
    }
#endif

    device @DT@* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t k = 0; k < ctO.get_capacity(); ++k) {
        if (!ctO.is_valid_element(k)) continue;
        auto idx = ctO.get_multidimensional_index(k);
        const int row = int(idx[1]);
        const int col = int(idx[0]);
        if (q0 + row >= Lq || col >= DD) continue;
        auto it = ctO.get_iterator(k);
        auto lit = cL.map_iterator(it);
        Ob[ulong(q0 + row)*DD + col] = @DT@(ctO[k] * fast::divide(1.0f, *lit));
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
static std::map<long, id<MTLComputePipelineState>> g_pso;
static std::string g_compile_error;

static void sub_all(std::string& src, const std::string& tok, const std::string& val) {
  size_t pos;
  while ((pos = src.find(tok)) != std::string::npos) src.replace(pos, tok.size(), val);
}

// BR query rows per threadgroup; NSG = BR/8 simdgroups (8 rows per SG) capped [4,16].
static int nsg_for(int BR) { return BR <= 32 ? 4 : (BR <= 64 ? 8 : 16); }

static id<MTLComputePipelineState> pso_for(int D, int BR, int BC, bool bf16s,
                                           int SR = 16, int BCV = 64,
                                           const char* fn_name = "flash_attn_fwd_bf16",
                                           bool pv_static = false, int nsg = 0,
                                           bool half_io = false, bool db = false,
                                           bool na_rp = true, bool na_byp = false,
                                           int na_bc = 64, int na_blk = 2) {
  const long key = ((((((((((long)D * 1000 + BR) * 1000 + BC) * 2 + (bf16s ? 1 : 0))
                    * 100 + SR) * 1000 + BCV) * 8
                   + (std::string(fn_name) == "flash_attn_fwd_bf16" ? 0
                      : (std::string(fn_name) == "flash_attn_fwd_v2r_bf16" ? 1
                         : (std::string(fn_name) == "flash_attn_fwd_frag" ? 2
                            : (std::string(fn_name) == "flash_attn_fwd_na" ? 3 : 4)))))
                   * 2 + (pv_static ? 1 : 0)) * 32 + nsg) * 4
                   + (half_io ? 1 : 0) + (db ? 2 : 0)) * 4 + (na_rp ? 1 : 0)
                   + (na_byp ? 2 : 0);
  const long key2 = key * 1024 + na_bc * 4 + na_blk;
  auto it = g_pso.find(key2);
  if (it != g_pso.end()) return it->second;

  std::string src(kSrcTemplate);
  sub_all(src, "@D@", std::to_string(D));
  sub_all(src, "@BR@", std::to_string(BR));
  sub_all(src, "@BC@", std::to_string(BC));
  const char* dt = half_io ? "half" : "bfloat";
  sub_all(src, "@DT@", dt);
  sub_all(src, "@ST@", bf16s ? dt : "float");
  sub_all(src, "@NSG@", std::to_string(nsg > 0 ? nsg : nsg_for(BR)));
  // P merges into the score tile only when the tile is bfloat (same type).
  sub_all(src, "@MP@", bf16s ? "true" : "false");
  sub_all(src, "@SR@", std::to_string(SR));
  sub_all(src, "@BCV@", std::to_string(BCV));
  sub_all(src, "@DB@", db ? "true" : "false");
  // na kernel: simdgroups per TG and the relaxed_precision descriptor flag.
  sub_all(src, "@NANSG@", std::to_string(nsg > 0 ? nsg : 16));
  sub_all(src, "@NARP@", na_rp ? "true" : "false");
  sub_all(src, "@NABYP@", na_byp ? "1" : "0");
  sub_all(src, "@NABCV@", std::to_string(na_bc));
  sub_all(src, "@NABLK@", std::to_string(na_blk));
  // diagnostic: ASFP8_FA41_NA_DIAG = 1 no-softmax / 2 QK-only / 3 no-P-roundtrip
  const int nadiag = getenv("ASFP8_FA41_NA_DIAG") ? atoi(getenv("ASFP8_FA41_NA_DIAG")) : 0;
  sub_all(src, "@NADIAG@", std::to_string(nadiag));
  // Static reduction dims where safe: qk's inner dim is always exactly D;
  // pv's inner dim (BC) is static only when the host proved Lk % BC == 0
  // (a static-k op reads the full tile — a short tail would read OOB).
  // ASFP8_FA41_DYNK=1 reverts both to dynamic for A/B.
  const bool dynk = getenv("ASFP8_FA41_DYNK") && atoi(getenv("ASFP8_FA41_DYNK"));
  sub_all(src, "@QKS@", dynk ? "static_cast<int>(dynamic_extent)" : "DD");
  sub_all(src, "@PVS@", (pv_static && !dynk) ? "BC" : "static_cast<int>(dynamic_extent)");

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
  g_pso[key2] = pso;
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
  TORCH_CHECK(D % 32 == 0 && D <= 128,
              "na kernel needs head_dim % 32 == 0 (static-K descriptor; "
              "OS 26.1 requires K a multiple of 32), <= 128");
  int BC = (int)bc > 0 ? (int)bc : 64;
  int BLK = (int)blk > 0 ? (int)blk : 2;
  TORCH_CHECK(BC % 32 == 0 && BC <= 256, "bc must be a multiple of 32, <= 256");
  TORCH_CHECK(BLK == 1 || BLK == 2, "blk must be 1 or 2");
  TORCH_CHECK(Lk % (BC * BLK) == 0, "na kernel requires Lk % (bc*blk) == 0");

  int NSG = (int)nsg > 0 ? (int)nsg : 16;
  TORCH_CHECK(NSG >= 1 && NSG <= 16, "nsg must be in [1,16]");
  TORCH_CHECK(byp != 0 || NSG * 16 * BC * (int)q.element_size() <= 32768,
              "P slices exceed 32KB threadgroup memory");

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto out = torch::empty({B, Hq, Lq, D}, q.options());
  std::array<int32_t, 6> shArr = {B, Hq, Hkv, Lq, Lk, 0};
  std::array<float, 1> prArr = {(float)scale};

  id<MTLComputePipelineState> pso =
      pso_for(D, 16, 64, /*bf16s=*/true, 16, 64, "flash_attn_fwd_na",
              /*pv_static=*/false, NSG, half_io, /*db=*/false, rp != 0, byp != 0,
              BC, BLK);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
        py::arg("bc") = 64, py::arg("blk") = 2);
  m.def("warmup", [](int64_t D, int64_t BR, int64_t BC) { pso_for((int)D, (int)BR, (int)BC, false); return true; },
        "compile + build pipeline for head dim D / BR rows / BC cols",
        py::arg("D") = 128, py::arg("BR") = 32, py::arg("BC") = 128);
  m.def("v2r_supported", [](int64_t D, int64_t SR, int64_t BCV) {
          return v2r_probe((int)D, (int)SR, (int)BCV);
        }, "device probe: register-resident-P support at (D, SR, BCV)",
        py::arg("D") = 128, py::arg("SR") = 16, py::arg("BCV") = 64);
}
