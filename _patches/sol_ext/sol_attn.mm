// ============================================================
// Block-sparse flash attention with log-sum-exp output (Metal 4.1 TensorOps).
//
// Sol-style attention keeps only the KV blocks a per-(head, query-block) router
// selects, and recovers the dropped blocks' mass from their centroids. This
// kernel computes the EXACT half: it iterates only the selected KV blocks and
// returns both the partial attention output and the running log-sum-exp, so the
// centroid correction can be merged afterwards in PyTorch with no augmented K/V
// tensors and no score-modification plumbing inside the kernel.
//
// Based on this package's mfa4_ext/flash41.mm, a Metal-4.1 port of
// mtlflashattn's v2 kernel (MIT, Copyright (c) Paweł Mazurkiewicz).
// The earlier FlashVSR project name referred to the maintainer's local
// prototype, not a third-party project. See THIRD_PARTY_NOTICES.md.
//
// Sparse-kernel extensions required by MiniMax H3:
//   * routing block size is a template parameter (64 or 128), not fixed at 128;
//   * ragged sequences: Lq/Lk need not be multiples of the block size. Callers
//     pad the tensors; the kernel masks columns past Lk_valid and never writes
//     rows past Lq_valid, so padding contents are irrelevant (never read into a
//     score that survives, never written back);
//   * per-row log-sum-exp output for the centroid merge.
//
// Scope: D <= 128, B == 1, non-causal (the router's mask encodes locality).
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
constant constexpr int BC = @BLK@;        // KV tile == router block
constant constexpr int QBLK = @BLK@;      // query router block
constant constexpr int DD = @D@;
constant constexpr int NSG = @NSG@;
constant constexpr int RPG = BR / NSG;    // rows per simdgroup
constant constexpr int CCH = BC / 32;     // 32-lane column chunks per row

kernel void sol_sparse_fwd(
    device @DT@* Q  [[buffer(0)]],
    device @DT@* K  [[buffer(1)]],
    device @DT@* V  [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],   // see host encode() for the field order
    constant float* PR [[buffer(5)]],  // [scale]
    device const int* BIDX [[buffer(6)]],  // [Hq, nQ, MMAX] selected kv block ids
    device const int* BCNT [[buffer(7)]],  // [Hq, nQ] valid entries per row
    device float* LSE [[buffer(8)]],       // [Hq, LqPad] row log-sum-exp
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], LqPad=SH[3], LqValid=SH[5], LkValid=SH[6];
    const int nQ=SH[7], MMAX=SH[8];
    const int qHead=SH[9], qRow=SH[10], kHead=SH[11], kRow=SH[12];
    const int oHead=SH[13], oRow=SH[14];
    const float scale = PR[0];

    const int bh  = int(tgid.y);
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x) * BR;
    const int qb  = q0 / QBLK;                 // router q-block of this row tile

    device const int* blist = BIDX + (ulong(hq) * ulong(nQ) + ulong(qb)) * ulong(MMAX);
    const int cnt = BCNT[hq * nQ + qb];

    // Strided access: supports contiguous [B,H,L,D] and packed [B,L,H*D].
    device @DT@* Qp = Q + ulong(hq)*ulong(qHead);
    device @DT@* Kp = K + ulong(hkv)*ulong(kHead);
    device @DT@* Vp = V + ulong(hkv)*ulong(kHead);

    // Reduction dims stay static (DD for QK, BC for PV): the ragged tail is
    // handled by masking scores, not by shrinking the tile.
    constexpr auto qk_desc = matmul2d_descriptor(
        BR, BC, DD, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;

    constexpr auto pv_desc = matmul2d_descriptor(
        BR, DD, BC, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    constexpr bool MERGEP = @MP@;
    threadgroup @ST@ tgS[BR*BC];
    threadgroup @DT@ tgP[MERGEP ? 1 : BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];
    threadgroup int tgAnyCorr;
    threadgroup @DT@* Pbuf = MERGEP ? (threadgroup @DT@*)&tgS[0] : &tgP[0];

    auto mQ = tensor(Qp + q0*qRow, dextents<int,2>{DD, BR}, array<int,2>{1, qRow});
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

    for (int jj = 0; jj < cnt; ++jj) {
        const int j0 = blist[jj] * BC;

        if (tid == 0) tgAnyCorr = 0;
        auto mK = tensor(Kp + j0*kRow, dextents<int,2>{DD, BC}, array<int,2>{1, kRow});
        qk_op.run(mQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax over the selected tile; scores held in registers
        // between the max and exp passes. Columns past the real sequence end
        // are forced to -inf so padded K/V never contribute.
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r) {
            const int row = r0 + r;
            float sreg[CCH];
            float mb = -INFINITY;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const int col = j0 + cc*32 + int(lane);
                const float s = (col < LkValid)
                    ? float(tgS[row*BC + cc*32 + lane]) * scale
                    : -INFINITY;
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

        if (tgAnyCorr != 0) {
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
                if (!ctO.is_valid_element(i)) continue;
                auto idx = ctO.get_multidimensional_index(i);
                ctO[i] *= tgCorr[int(idx[1])];
            }
        }
        auto mV = tensor(Vp + j0*kRow, dextents<int,2>{DD, BC}, array<int,2>{1, kRow});
        pv_op.run(tP, mV, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r) {
            const int row = r0 + r;
            tgInvL[row] = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
            const int gr = q0 + row;
            if (gr < LqValid)
                LSE[ulong(hq)*ulong(LqPad) + ulong(gr)] =
                    (l[r] > 0.0f) ? (m[r] + log(l[r])) : -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device @DT@* Ob = O + ulong(hq)*ulong(oHead);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= LqValid || c >= DD) continue;
        Ob[ulong(r)*ulong(oRow) + c] = @DT@(ctO[i] * tgInvL[row]);
    }
}
)MTL";
// ============================================================
// Phase E: routing + sparse compute + centroid correction in ONE pass.
//
// The two-stage design above leaves routing and the centroid merge in PyTorch,
// where they cost MORE than the kernel itself (measured 25.5 ms + 93.6 ms
// against an 85.8 ms kernel and 220 ms dense, at S=17000 H=56 D=128). No
// density makes that profitable. This kernel folds both in.
//
// Three streaming passes per query tile, no proxy map ever materialized:
//
//   A. Stream the pooled keys; average each block's proxy column over the
//      tile's rows into a small [nB] vector, then derive one threshold
//      tau = mu + beta*sigma from its own statistics.
//   B. Stream the pooled keys again: every REJECTED block's centroid enters
//      the online softmax right here, as one column scored with THIS ROW's
//      own proxy plus log(block_length). The correction stops being a second
//      pass over the sequence — it is just more columns.
//   C. Walk the real KV blocks, computing only the selected ones.
//
// Why the keep/drop decision is per TILE and not per row: the matmul operates
// on a BR-row tile, so a block must be computed if any row wants it. Measured,
// per-row thresholds make that union 91-100% dense — neighbouring rows simply
// do not agree enough — which erases the sparsity entirely. Deciding from the
// tile's mean query reproduces the two-stage path's 26-47% density, while the
// correction stays per row (each row scores rejected blocks with its own
// proxy), so quality does not regress to block granularity.
//
// Online softmax is order-independent, so running centroids (B) before exact
// blocks (C) is safe.
// ============================================================
static const char* kFusedTemplate = R"MTL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BR = @BR@;
constant constexpr int BC = @BLK@;        // KV tile == router block
constant constexpr int DD = @D@;
constant constexpr int NSG = @NSG@;
constant constexpr int RPG = BR / NSG;
constant constexpr int CCH = BC / 32;
constant constexpr int TGT = NSG * 32;    // threads per threadgroup
constant constexpr int MW  = @MW@;        // selection words (32 -> nB <= 1024)
constant constexpr int MAXB = MW * 32;

kernel void sol_fused_fwd(
    device @DT@* Q  [[buffer(0)]],
    device @DT@* K  [[buffer(1)]],
    device @DT@* V  [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],
    constant float* PR [[buffer(5)]],   // [scale, beta]
    // non-const: matmul2d requires operand types to match the cooperative
    // tensor exactly, and a const pointer yields tensor<const device T>
    device @DT@* KB [[buffer(6)]],      // [Hkv, nB, D] block-mean keys
    device @DT@* VB [[buffer(7)]],      // [Hkv, nB, D] block-mean values
    device float* LOGL [[buffer(8)]],   // [nB] log(valid block length)
    device int* STAT [[buffer(9)]],     // [B*Hq, nQtiles] blocks computed
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], LqValid=SH[5], LkValid=SH[6], nB=SH[7];
    const int qHead=SH[8], qRow=SH[9], kHead=SH[10], kRow=SH[11];
    const int oHead=SH[12], oRow=SH[13], kbHead=SH[14], kbRow=SH[15];
    const int prefRows=SH[16], aR0=SH[17], aR1=SH[18];
    const int prefBlk=SH[19], aB0=SH[20], aB1=SH[21], radius=SH[22];
    const int blkSize=SH[23], nQtiles=SH[24];
    const int videoStart=SH[25], rowsPerFrame=SH[26], taps=SH[27];
    const float scale = PR[0], beta = PR[1];

    const int bh  = int(tgid.y);
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x) * BR;

    device @DT@* Qp = Q + ulong(hq)*ulong(qHead);
    device @DT@* Kp = K + ulong(hkv)*ulong(kHead);
    device @DT@* Vp = V + ulong(hkv)*ulong(kHead);
    device @DT@* KBp = KB + ulong(hkv)*ulong(kbHead);
    device @DT@* VBp = VB + ulong(hkv)*ulong(kbHead);

    constexpr auto qk_desc = matmul2d_descriptor(
        BR, BC, DD, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;

    constexpr auto pv_desc = matmul2d_descriptor(
        BR, DD, BC, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    constexpr bool MERGEP = @MP@;
    threadgroup @ST@ tgS[BR*BC];
    threadgroup @DT@ tgP[MERGEP ? 1 : BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];
    threadgroup float tgProxy[MAXB];
    threadgroup uint  tgSel[MW];
    threadgroup float tgRed[2*NSG];
    threadgroup float tgTau;
    threadgroup int   tgAnyCorr;
    threadgroup @DT@* Pbuf = MERGEP ? (threadgroup @DT@*)&tgS[0] : &tgP[0];

    auto mQ = tensor(Qp + q0*qRow, dextents<int,2>{DD, BR}, array<int,2>{1, qRow});
    auto tS = tensor(&tgS[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});
    auto tP = tensor(Pbuf, dextents<int,2>{BC, BR}, array<int,2>{1, BC});

    using PvA = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
    using PvB = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctO = pv_op.get_destination_cooperative_tensor<PvA, PvB, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i)
        if (ctO.is_valid_element(i)) ctO[i] = 0.0f;

    const int nWords = (nB + 31) / 32;
    const int r0 = int(sgid) * RPG;
    float m[RPG], l[RPG];
    #pragma clang loop unroll(full)
    for (int r = 0; r < RPG; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    // ---------- Pass A: tile proxy vector -> threshold ----------
    for (int t0 = 0; t0 < nB; t0 += BC) {
        auto mKB = tensor(KBp + t0*kbRow, dextents<int,2>{DD, BC}, array<int,2>{1, kbRow});
        qk_op.run(mQ, mKB, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = tid; c < uint(BC); c += uint(TGT)) {
            const int j = t0 + int(c);
            if (j < nB) {
                float s = 0.0f;
                for (int r = 0; r < BR; ++r) s += float(tgS[r*BC + int(c)]);
                tgProxy[j] = s * (1.0f / float(BR)) * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    {
        float part = 0.0f, partsq = 0.0f;
        for (int j = int(tid); j < nB; j += TGT) {
            const float p = tgProxy[j];
            part += p; partsq += p * p;
        }
        part = simd_sum(part); partsq = simd_sum(partsq);
        if (lane == 0) { tgRed[sgid] = part; tgRed[NSG + sgid] = partsq; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s = 0.0f, sq = 0.0f;
            for (int i = 0; i < NSG; ++i) { s += tgRed[i]; sq += tgRed[NSG + i]; }
            const float inv = 1.0f / float(max(nB, 1));
            const float mu = s * inv;
            const float var = max(0.0f, sq * inv - mu * mu);
            tgTau = mu + beta * sqrt(var);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------- selection: one decision per (tile, block) ----------
    // A tile holding any prefix/audio query row stays fully dense.
    const bool tileDense = (q0 < prefRows) || (q0 + BR > aR0 && q0 < aR1);
    const int qb0 = q0 / blkSize, qb1 = (q0 + BR - 1) / blkSize;

    // Temporal band: the co-located block in ADJACENT LATENT FRAMES. One frame
    // is rowsPerFrame rows (1620 at 864x480, ~25 blocks), so it sits far
    // outside the spatial band above -- which is local within a single frame
    // and never reaches another one. Without this, a static region has no
    // guaranteed reference in any other frame and its context changes from tile
    // to tile, head to head and step to step; that inconsistency is what reads
    // as background flicker/morphing on content that should be still.
    int tband[4];
    int ntb = 0;
    if (rowsPerFrame > 0 && taps > 0) {
        const int qc = q0 + BR/2;
        if (qc >= videoStart) {
            for (int m = 1; m <= taps && ntb < 4; ++m) {
                const int rp = qc + m*rowsPerFrame;
                if (rp < LqValid && ntb < 4) tband[ntb++] = rp / blkSize;
                const int rm = qc - m*rowsPerFrame;
                if (rm >= videoStart && ntb < 4) tband[ntb++] = rm / blkSize;
            }
        }
    }

    for (uint w = tid; w < uint(nWords); w += uint(TGT)) {
        uint bits = 0u;
        for (int b = 0; b < 32; ++b) {
            const int j = int(w) * 32 + b;
            if (j >= nB) break;
            bool forced = tileDense || (j < prefBlk) || (j >= aB0 && j < aB1)
                          || (j >= qb0 - radius && j <= qb1 + radius);
            for (int t = 0; t < ntb; ++t)
                if (j >= tband[t] - radius && j <= tband[t] + radius) forced = true;
            if (forced || tgProxy[j] > tgTau) bits |= (1u << b);
        }
        tgSel[w] = bits;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- Pass B: rejected-block centroids, scored PER ROW ----------
    for (int t0 = 0; t0 < nB; t0 += BC) {
        if (tid == 0) tgAnyCorr = 0;
        auto mKB = tensor(KBp + t0*kbRow, dextents<int,2>{DD, BC}, array<int,2>{1, kbRow});
        qk_op.run(mQ, mKB, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r) {
            const int row = r0 + r;
            float sreg[CCH];
            float mb = -INFINITY;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const int j = t0 + cc*32 + int(lane);
                const bool valid = (j < nB);
                const bool sel = valid && (((tgSel[j >> 5] >> (j & 31)) & 1u) != 0u);
                // a selected block is computed exactly in pass C; masking it
                // here is what stops the two contributions double counting
                float cs = -INFINITY;
                if (valid && !sel)
                    cs = float(tgS[row*BC + cc*32 + lane]) * scale + LOGL[j];
                sreg[cc] = cs;
                mb = max(mb, simd_max(cs));
            }
            const float mn = max(m[r], mb);
            const float c = (mn == -INFINITY) ? 1.0f : exp(m[r] - mn);
            float lsum = 0.0f;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const float pe = (mn == -INFINITY) ? 0.0f : exp(sreg[cc] - mn);
                Pbuf[row*BC + cc*32 + lane] = @DT@(pe);
                lsum += simd_sum(pe);
            }
            l[r] = l[r]*c + lsum;
            m[r] = mn;
            tgCorr[row] = c;
            if (c != 1.0f) tgAnyCorr = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tgAnyCorr != 0) {
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
                if (!ctO.is_valid_element(i)) continue;
                auto idx = ctO.get_multidimensional_index(i);
                ctO[i] *= tgCorr[int(idx[1])];
            }
        }
        auto mVB = tensor(VBp + t0*kbRow, dextents<int,2>{DD, BC}, array<int,2>{1, kbRow});
        pv_op.run(tP, mVB, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------- Pass C: exact selected blocks ----------
    int computed = 0;
    for (int j = 0; j < nB; ++j) {
        if (((tgSel[j >> 5] >> (j & 31)) & 1u) == 0u) continue;
        ++computed;
        const int j0 = j * BC;
        if (tid == 0) tgAnyCorr = 0;
        auto mK = tensor(Kp + j0*kRow, dextents<int,2>{DD, BC}, array<int,2>{1, kRow});
        qk_op.run(mQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r) {
            const int row = r0 + r;
            float sreg[CCH];
            float mb = -INFINITY;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const int col = j0 + cc*32 + int(lane);
                const float s = (col < LkValid)
                    ? float(tgS[row*BC + cc*32 + lane]) * scale : -INFINITY;
                sreg[cc] = s;
                mb = max(mb, simd_max(s));
            }
            const float mn = max(m[r], mb);
            const float c = (mn == -INFINITY) ? 1.0f : exp(m[r] - mn);
            float lsum = 0.0f;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const float pe = (mn == -INFINITY) ? 0.0f : exp(sreg[cc] - mn);
                Pbuf[row*BC + cc*32 + lane] = @DT@(pe);
                lsum += simd_sum(pe);
            }
            l[r] = l[r]*c + lsum;
            m[r] = mn;
            tgCorr[row] = c;
            if (c != 1.0f) tgAnyCorr = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tgAnyCorr != 0) {
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
                if (!ctO.is_valid_element(i)) continue;
                auto idx = ctO.get_multidimensional_index(i);
                ctO[i] *= tgCorr[int(idx[1])];
            }
        }
        auto mV = tensor(Vp + j0*kRow, dextents<int,2>{DD, BC}, array<int,2>{1, kRow});
        pv_op.run(tP, mV, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r)
            tgInvL[r0 + r] = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
    }
    if (tid == 0) STAT[bh * nQtiles + int(tgid.x)] = computed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device @DT@* Ob = O + ulong(hq)*ulong(oHead);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= LqValid || c >= DD) continue;
        Ob[ulong(r)*ulong(oRow) + c] = @DT@(ctO[i] * tgInvL[row]);
    }
}
)MTL";

// ============================================================
// diag64 stage 1: the route.
//
// NVIDIA's released H3 router, one threadgroup per (head, 64-row query block):
//
//   proxy(j)  = <q_centroid, k_centroid[j]> * scale * log2(e)
//   mu        = <q_centroid, key_mean>      * scale * log2(e)
//   sigma2    = <q_centroid^2, key_var>     * (scale * log2(e))^2
//   threshold = mu + tau * sqrt(sigma2 + 1e-6)
//   keep(j)   = proxy(j) > threshold
//
// The threshold is ANALYTIC — predicted from the key centroids' own diagonal
// covariance, which does not depend on the query at all. That is the whole
// difference from the empirical router this replaces, which measured the mean
// and variance OF THE PROXY ROW ITSELF and so tightened its threshold exactly
// where the row was flat and the approximation was least safe.
//
// Everything here is fp32, including the pooled query centroid, which is
// averaged from the bf16 query rows in fp32. The `>` is a strict comparison
// against a threshold built from D squared products; bf16 cannot hold that to
// the precision a strict comparison needs, and a wrong side of the comparison
// is a whole 64x64 block of attention gained or lost.
//
// Output is a compact bitmask [Hq, nQ, MW] plus per-query-block selected
// counts, so the attention kernel needs no routing state of its own and the
// host never has to synchronize to learn the density.
// ============================================================
static const char* kRouteTemplate = R"MTL(
#include <metal_stdlib>
using namespace metal;

constant constexpr int DD  = @D@;
constant constexpr int MW  = @MW@;
constant constexpr int NSG = 4;
constant constexpr int TGT = NSG * 32;
constant constexpr float LOG2E = 1.44269504088896340736f;

kernel void sol_diag_route(
    device @DT@* Q          [[buffer(0)]],   // [B,Hq,LqPad,D] padded queries
    device const float* KC  [[buffer(1)]],   // [Hkv, nB, D] fp32 key centroids
    device const float* KM  [[buffer(2)]],   // [Hkv, D] centroid mean
    device const float* KVR [[buffer(3)]],   // [Hkv, D] centroid diagonal variance
    device uint* MASK       [[buffer(4)]],   // [Hq, nQ, MW] selection bits
    device int* CNT         [[buffer(5)]],   // [Hq, nQ] selected block count
    constant int* SH        [[buffer(6)]],
    constant float* PR      [[buffer(7)]],   // [scale, tau]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[0], Hkv=SH[1], LqValid=SH[2], nB=SH[3], nQ=SH[4];
    const int qHead=SH[5], qRow=SH[6], blkSize=SH[7];
    const int sinkBlk=SH[8], radius=SH[9];
    const float s = PR[0] * LOG2E, tau = PR[1];

    const int qb  = int(tgid.x);
    const int hq  = int(tgid.y);
    const int hkv = hq / (Hq / Hkv);

    threadgroup float tgQC[DD];
    threadgroup float tgRed[2 * NSG];
    threadgroup float tgThr;
    threadgroup uchar tgSel[MW * 32];

    device @DT@* Qp = Q + ulong(hq) * ulong(qHead);

    // Pool this query block's REAL rows only. A ragged final block divides by
    // its real length; dividing by blkSize would pull the centroid towards
    // zero and shift the whole row's threshold.
    const int r0 = qb * blkSize;
    int rows = LqValid - r0;
    rows = (rows > blkSize) ? blkSize : rows;
    if (rows < 1) rows = 1;
    for (uint d = tid; d < uint(DD); d += uint(TGT)) {
        float acc = 0.0f;
        for (int r = 0; r < rows; ++r)
            acc += float(Qp[ulong(r0 + r) * ulong(qRow) + d]);
        tgQC[d] = acc / float(rows);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mu and sigma2 in one reduction over D.
    {
        float pm = 0.0f, pv = 0.0f;
        for (uint d = tid; d < uint(DD); d += uint(TGT)) {
            const float a = tgQC[d];
            pm += a * KM[hkv * DD + int(d)];
            pv += a * a * KVR[hkv * DD + int(d)];
        }
        pm = simd_sum(pm);
        pv = simd_sum(pv);
        if (lane == 0) { tgRed[sgid] = pm; tgRed[NSG + sgid] = pv; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float mu = 0.0f, var = 0.0f;
            for (int i = 0; i < NSG; ++i) { mu += tgRed[i]; var += tgRed[NSG + i]; }
            mu *= s;
            var = max(0.0f, var) * s * s;
            tgThr = mu + tau * sqrt(var + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float thr = tgThr;

    // A query block overlapping the exact sink is fully dense. The sink is
    // rounded outward to a block boundary by the host, so `qb < sinkBlk` is
    // exactly "this block holds at least one prefix row".
    const bool qDense = (qb < sinkBlk);
    device const float* kc = KC + ulong(hkv) * ulong(nB) * ulong(DD);
    for (uint j = tid; j < uint(nB); j += uint(TGT)) {
        device const float* row = kc + ulong(j) * ulong(DD);
        float acc = 0.0f;
        for (int d = 0; d < DD; ++d) acc += tgQC[d] * row[d];
        const int jj = int(j);
        const bool forced = qDense || (jj < sinkBlk)
                            || (jj >= qb - radius && jj <= qb + radius);
        tgSel[j] = (forced || (acc * s > thr)) ? uchar(1) : uchar(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nWords = (nB + 31) / 32;
    device uint* mrow = MASK + (ulong(hq) * ulong(nQ) + ulong(qb)) * ulong(MW);
    int cnt = 0;
    for (uint w = tid; w < uint(MW); w += uint(TGT)) {
        uint bits = 0u;
        if (int(w) < nWords) {
            for (int b = 0; b < 32; ++b) {
                const int j = int(w) * 32 + b;
                if (j >= nB) break;
                if (tgSel[j] != uchar(0)) { bits |= (1u << b); ++cnt; }
            }
        }
        mrow[w] = bits;      // words past nWords are zeroed, never read as data
    }
    cnt = int(simd_sum(cnt));
    if (lane == 0) tgRed[sgid] = float(cnt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NSG; ++i) total += tgRed[i];
        CNT[hq * nQ + qb] = int(total);
    }
}
)MTL";

// ============================================================
// diag64 stage 2: corrected attention under a PRECOMPUTED route.
//
// Same two streaming passes as the fused kernel — rejected-block centroids as
// extra softmax columns, then the exact selected blocks — but the keep/drop
// decision arrives as a bitmask instead of being derived here. That matters for
// more than tidiness: the fused kernel decided per 32-row compute tile, so the
// route changed with the tile size. Here the mask is per 64-row query block and
// BOTH 32-row halves read the same word, so the compute tiling is free to stay
// at whatever the TensorOps hardware likes without touching the policy.
//
// Online softmax is order-independent, so running centroids before exact blocks
// is safe. A fully-dense query block (the sink) selects everything, which makes
// the centroid pass contribute nothing and the exact pass compute all of it —
// i.e. the sink rows fall back to exact attention with no special case.
// ============================================================
static const char* kMaskedTemplate = R"MTL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BR = @BR@;
constant constexpr int BC = @BLK@;
constant constexpr int DD = @D@;
constant constexpr int NSG = @NSG@;
constant constexpr int RPG = BR / NSG;
constant constexpr int CCH = BC / 32;
constant constexpr int TGT = NSG * 32;
constant constexpr int MW  = @MW@;

kernel void sol_masked_corrected_fwd(
    device @DT@* Q  [[buffer(0)]],
    device @DT@* K  [[buffer(1)]],
    device @DT@* V  [[buffer(2)]],
    device @DT@* O  [[buffer(3)]],
    constant int*  SH [[buffer(4)]],
    constant float* PR [[buffer(5)]],
    // non-const: matmul2d requires operand types to match the cooperative
    // tensor exactly, and a const pointer yields tensor<const device T>
    device @DT@* KB [[buffer(6)]],       // [Hkv, nBpad, D] block-mean keys
    device @DT@* VB [[buffer(7)]],       // [Hkv, nBpad, D] block-mean values
    device float* LOGL [[buffer(8)]],    // [nBpad] log(valid block length)
    device const uint* MASK [[buffer(9)]],  // [Hq, nQ, MW] route from stage 1
    device int* STAT [[buffer(10)]],     // [B*Hq, nQtiles] blocks computed
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], LqValid=SH[5], LkValid=SH[6], nB=SH[7];
    const int qHead=SH[8], qRow=SH[9], kHead=SH[10], kRow=SH[11];
    const int oHead=SH[12], oRow=SH[13], kbHead=SH[14], kbRow=SH[15];
    const int blkSize=SH[16], nQtiles=SH[17], nQ=SH[18], correct=SH[19];
    const float scale = PR[0];

    const int bh  = int(tgid.y);
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x) * BR;

    device @DT@* Qp = Q + ulong(hq)*ulong(qHead);
    device @DT@* Kp = K + ulong(hkv)*ulong(kHead);
    device @DT@* Vp = V + ulong(hkv)*ulong(kHead);
    device @DT@* KBp = KB + ulong(hkv)*ulong(kbHead);
    device @DT@* VBp = VB + ulong(hkv)*ulong(kbHead);

    constexpr auto qk_desc = matmul2d_descriptor(
        BR, BC, DD, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;

    constexpr auto pv_desc = matmul2d_descriptor(
        BR, DD, BC, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    constexpr bool MERGEP = @MP@;
    threadgroup @ST@ tgS[BR*BC];
    threadgroup @DT@ tgP[MERGEP ? 1 : BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];
    threadgroup uint  tgSel[MW];
    threadgroup int   tgAnyCorr;
    threadgroup @DT@* Pbuf = MERGEP ? (threadgroup @DT@*)&tgS[0] : &tgP[0];

    auto mQ = tensor(Qp + q0*qRow, dextents<int,2>{DD, BR}, array<int,2>{1, qRow});
    auto tS = tensor(&tgS[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});
    auto tP = tensor(Pbuf, dextents<int,2>{BC, BR}, array<int,2>{1, BC});

    using PvA = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
    using PvB = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctO = pv_op.get_destination_cooperative_tensor<PvA, PvB, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i)
        if (ctO.is_valid_element(i)) ctO[i] = 0.0f;

    const int nWords = (nB + 31) / 32;
    const int r0 = int(sgid) * RPG;
    float m[RPG], l[RPG];
    #pragma clang loop unroll(full)
    for (int r = 0; r < RPG; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    // ---------- the route, verbatim from stage 1 ----------
    // q0/blkSize, not tgid.x: with BR=32 and blkSize=64 the two compute tiles
    // of a query block MUST read the same word, or the halves of one 64-row
    // routing decision would disagree.
    {
        device const uint* mrow = MASK
            + (ulong(hq) * ulong(nQ) + ulong(q0 / blkSize)) * ulong(MW);
        for (uint w = tid; w < uint(MW); w += uint(TGT))
            tgSel[w] = (int(w) < nWords) ? mrow[w] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---------- Pass B: rejected-block centroids, scored PER ROW ----------
    if (correct != 0) {
    for (int t0 = 0; t0 < nB; t0 += BC) {
        if (tid == 0) tgAnyCorr = 0;
        auto mKB = tensor(KBp + t0*kbRow, dextents<int,2>{DD, BC}, array<int,2>{1, kbRow});
        qk_op.run(mQ, mKB, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r) {
            const int row = r0 + r;
            float sreg[CCH];
            float mb = -INFINITY;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const int j = t0 + cc*32 + int(lane);
                const bool valid = (j < nB);
                const bool sel = valid && (((tgSel[j >> 5] >> (j & 31)) & 1u) != 0u);
                // a selected block is computed exactly in pass C; masking it
                // here is what stops the two contributions double counting
                float cs = -INFINITY;
                if (valid && !sel)
                    cs = float(tgS[row*BC + cc*32 + lane]) * scale + LOGL[j];
                sreg[cc] = cs;
                mb = max(mb, simd_max(cs));
            }
            const float mn = max(m[r], mb);
            const float c = (mn == -INFINITY) ? 1.0f : exp(m[r] - mn);
            float lsum = 0.0f;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const float pe = (mn == -INFINITY) ? 0.0f : exp(sreg[cc] - mn);
                Pbuf[row*BC + cc*32 + lane] = @DT@(pe);
                lsum += simd_sum(pe);
            }
            l[r] = l[r]*c + lsum;
            m[r] = mn;
            tgCorr[row] = c;
            if (c != 1.0f) tgAnyCorr = 1;   // benign race: all writers store 1
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tgAnyCorr != 0) {
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
                if (!ctO.is_valid_element(i)) continue;
                auto idx = ctO.get_multidimensional_index(i);
                ctO[i] *= tgCorr[int(idx[1])];
            }
        }
        auto mVB = tensor(VBp + t0*kbRow, dextents<int,2>{DD, BC}, array<int,2>{1, kbRow});
        pv_op.run(tP, mVB, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }

    // ---------- Pass C: exact selected blocks ----------
    int computed = 0;
    for (int j = 0; j < nB; ++j) {
        if (((tgSel[j >> 5] >> (j & 31)) & 1u) == 0u) continue;
        ++computed;
        const int j0 = j * BC;
        if (tid == 0) tgAnyCorr = 0;
        auto mK = tensor(Kp + j0*kRow, dextents<int,2>{DD, BC}, array<int,2>{1, kRow});
        qk_op.run(mQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r) {
            const int row = r0 + r;
            float sreg[CCH];
            float mb = -INFINITY;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const int col = j0 + cc*32 + int(lane);
                const float s = (col < LkValid)
                    ? float(tgS[row*BC + cc*32 + lane]) * scale : -INFINITY;
                sreg[cc] = s;
                mb = max(mb, simd_max(s));
            }
            const float mn = max(m[r], mb);
            const float c = (mn == -INFINITY) ? 1.0f : exp(m[r] - mn);
            float lsum = 0.0f;
            #pragma clang loop unroll(full)
            for (int cc = 0; cc < CCH; ++cc) {
                const float pe = (mn == -INFINITY) ? 0.0f : exp(sreg[cc] - mn);
                Pbuf[row*BC + cc*32 + lane] = @DT@(pe);
                lsum += simd_sum(pe);
            }
            l[r] = l[r]*c + lsum;
            m[r] = mn;
            tgCorr[row] = c;
            if (c != 1.0f) tgAnyCorr = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tgAnyCorr != 0) {
            #pragma clang loop unroll(full)
            for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
                if (!ctO.is_valid_element(i)) continue;
                auto idx = ctO.get_multidimensional_index(i);
                ctO[i] *= tgCorr[int(idx[1])];
            }
        }
        auto mV = tensor(Vp + j0*kRow, dextents<int,2>{DD, BC}, array<int,2>{1, kRow});
        pv_op.run(tP, mV, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPG; ++r)
            tgInvL[r0 + r] = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
    }
    if (tid == 0) STAT[bh * nQtiles + int(tgid.x)] = computed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device @DT@* Ob = O + ulong(hq)*ulong(oHead);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= LqValid || c >= DD) continue;
        Ob[ulong(r)*ulong(oRow) + c] = @DT@(ctO[i] * tgInvL[row]);
    }
}
)MTL";

static std::map<long, id<MTLComputePipelineState>> g_pso;
static std::map<long, id<MTLComputePipelineState>> g_pso_fused;
static std::map<long, id<MTLComputePipelineState>> g_pso_route;
static std::map<long, id<MTLComputePipelineState>> g_pso_masked;

static void sub_all(std::string& s, const std::string& from, const std::string& to) {
  for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size())
    s.replace(pos, from.size(), to);
}

static id<MTLComputePipelineState> pso_for(int D, int BR, int BLK, bool half_s, bool half_io) {
  const long key = (((long)D * 256 + BR) * 256 + BLK) * 4 + (half_s ? 1 : 0) + (half_io ? 2 : 0);
  auto it = g_pso.find(key);
  if (it != g_pso.end()) return it->second;

  const int nsg = 4;
  std::string src(kSrcTemplate);
  sub_all(src, "@D@", std::to_string(D));
  sub_all(src, "@BR@", std::to_string(BR));
  sub_all(src, "@BLK@", std::to_string(BLK));
  sub_all(src, "@NSG@", std::to_string(nsg));
  const char* dt = half_io ? "half" : "bfloat";
  sub_all(src, "@DT@", dt);
  sub_all(src, "@ST@", half_s ? dt : "float");
  sub_all(src, "@MP@", half_s ? "true" : "false");

  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                         options:opts error:&err];
  TORCH_CHECK(lib, "sol_attn Metal compile failed (D=", D, ", BR=", BR, ", BLK=", BLK, "): ",
              err ? err.localizedDescription.UTF8String : "unknown");
  id<MTLFunction> fn = [lib newFunctionWithName:@"sol_sparse_fwd"];
  TORCH_CHECK(fn, "sol_sparse_fwd not found in compiled library");
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "sol_attn pipeline creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_pso[key] = pso;
  return pso;
}

// O[1,H,LqPad,D], LSE[H,LqPad] = sparse_attention(Q, K, V, block lists).
// Q/K/V are [B,H,L,D] with L already padded up to a multiple of BLK; the valid
// lengths bound what is read into surviving scores and what is written out.
std::vector<torch::Tensor> sol_sparse_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor block_idx, torch::Tensor block_count,
    double scale, int64_t lq_valid, int64_t lk_valid,
    int64_t blk, int64_t br, bool half_s) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "inputs must be [B,H,L,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1);
  const int LqPad = (int)q.size(2), D = (int)q.size(3);
  const int Hkv = (int)k.size(1), LkPad = (int)k.size(2);
  TORCH_CHECK(B == 1, "B must be 1");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == LkPad, "kv shape mismatch");
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type(),
              "inputs must be bf16 or fp16 (matching)");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 8 == 0 && D <= 128, "head_dim must be a multiple of 8, <= 128");
  const int BLK = (int)blk;
  TORCH_CHECK(BLK == 64 || BLK == 128, "block size must be 64 or 128");
  TORCH_CHECK(LqPad % BLK == 0 && LkPad % BLK == 0, "padded lengths must be multiples of the block size");
  const int BR = br > 0 ? (int)br : 32;
  TORCH_CHECK(BR % 4 == 0 && BLK % BR == 0, "br must divide the block size (rows share a q-block)");
  TORCH_CHECK(lq_valid > 0 && lq_valid <= LqPad, "lq_valid out of range");
  TORCH_CHECK(lk_valid > 0 && lk_valid <= LkPad, "lk_valid out of range");
  const int nQ = LqPad / BLK;
  TORCH_CHECK(block_idx.scalar_type() == torch::kInt && block_count.scalar_type() == torch::kInt,
              "block lists must be int32");
  TORCH_CHECK(block_idx.is_mps() && block_count.is_mps(), "block lists must be on mps");
  TORCH_CHECK(block_idx.dim() == 3 && block_idx.size(0) == Hq && block_idx.size(1) == nQ,
              "block_idx must be [H, LqPad/BLK, MMAX]");
  TORCH_CHECK(block_count.dim() == 2 && block_count.size(0) == Hq && block_count.size(1) == nQ,
              "block_count must be [H, LqPad/BLK]");

  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto ic = block_idx.contiguous();
  auto cc = block_count.contiguous();
  auto out = torch::zeros({B, Hq, LqPad, D}, q.options());
  auto lse = torch::full({Hq, LqPad}, -std::numeric_limits<float>::infinity(),
                         q.options().dtype(torch::kFloat));

  const long qHead = (long)LqPad * D, qRow = D;
  const long kHead = (long)LkPad * D, kRow = D;
  const long oHead = (long)LqPad * D, oRow = D;
  std::array<int32_t, 15> shArr = {
      B, Hq, Hkv, LqPad, LkPad, (int32_t)lq_valid, (int32_t)lk_valid, nQ,
      (int32_t)block_idx.size(2), (int32_t)qHead, (int32_t)qRow,
      (int32_t)kHead, (int32_t)kRow, (int32_t)oHead, (int32_t)oRow};
  std::array<float, 1> prArr = {(float)scale};

  id<MTLComputePipelineState> pso = pso_for(D, BR, BLK, half_s, half_io);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  id<MTLBuffer> iB = __builtin_bit_cast(id<MTLBuffer>, ic.storage().data());
  id<MTLBuffer> nB = __builtin_bit_cast(id<MTLBuffer>, cc.storage().data());
  id<MTLBuffer> lB = __builtin_bit_cast(id<MTLBuffer>, lse.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const NSUInteger iO = ic.storage_offset() * ic.element_size();
  const NSUInteger nO = cc.storage_offset() * cc.element_size();
  const NSUInteger lO = lse.storage_offset() * lse.element_size();
  const NSUInteger gx = (NSUInteger)(LqPad / BR);
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
      [enc setBuffer:iB offset:iO atIndex:6];
      [enc setBuffer:nB offset:nO atIndex:7];
      [enc setBuffer:lB offset:lO atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
  });
  // No COMMIT_AND_WAIT — encoded on torch's MPS stream.
  return {out, lse};
}

// nB <= 1024 blocks. Enough for 32k tokens even at 32-token blocks, and the
// tgProxy[MAXB] array it sizes competes with the score tiles for the 32 KB of
// threadgroup memory — 2048 overflowed it at BLK=128.
static const int kMaskWords = 32;

static id<MTLComputePipelineState> pso_for_fused(int D, int BR, int BLK,
                                                 bool half_s, bool half_io) {
  const long key = (((long)D * 256 + BR) * 256 + BLK) * 4 + (half_s ? 1 : 0) + (half_io ? 2 : 0);
  auto it = g_pso_fused.find(key);
  if (it != g_pso_fused.end()) return it->second;

  const int nsg = 4;
  std::string src(kFusedTemplate);
  sub_all(src, "@D@", std::to_string(D));
  sub_all(src, "@BR@", std::to_string(BR));
  sub_all(src, "@BLK@", std::to_string(BLK));
  sub_all(src, "@NSG@", std::to_string(nsg));
  sub_all(src, "@MW@", std::to_string(kMaskWords));
  const char* dt = half_io ? "half" : "bfloat";
  sub_all(src, "@DT@", dt);
  sub_all(src, "@ST@", half_s ? dt : "float");
  sub_all(src, "@MP@", half_s ? "true" : "false");

  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                         options:opts error:&err];
  TORCH_CHECK(lib, "sol_fused Metal compile failed (D=", D, ", BR=", BR, ", BLK=", BLK, "): ",
              err ? err.localizedDescription.UTF8String : "unknown");
  id<MTLFunction> fn = [lib newFunctionWithName:@"sol_fused_fwd"];
  TORCH_CHECK(fn, "sol_fused_fwd not found in compiled library");
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "sol_fused pipeline creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_pso_fused[key] = pso;
  return pso;
}

// Fused route + sparse compute + centroid correction. Returns (out, stat),
// where stat[bh, qtile] is the number of KV blocks actually computed — the
// union over the tile's rows, i.e. the density that costs real time.
std::vector<torch::Tensor> sol_fused_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor kbar, torch::Tensor vbar, torch::Tensor loglen,
    double scale, double beta, int64_t lq_valid, int64_t lk_valid,
    int64_t blk, int64_t br, bool half_s,
    int64_t prefix_rows, int64_t audio_row_start, int64_t audio_row_stop,
    int64_t prefix_blocks, int64_t audio_blk_start, int64_t audio_blk_stop,
    int64_t local_radius, int64_t video_row_start, int64_t rows_per_frame,
    int64_t temporal_taps) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "inputs must be [B,H,L,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1);
  const int LqPad = (int)q.size(2), D = (int)q.size(3);
  const int Hkv = (int)k.size(1), LkPad = (int)k.size(2);
  TORCH_CHECK(B == 1, "B must be 1");
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type()
              && kbar.scalar_type() == q.scalar_type()
              && vbar.scalar_type() == q.scalar_type(),
              "q/k/v/kbar/vbar must share dtype (bf16 or fp16)");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 8 == 0 && D <= 128, "head_dim must be a multiple of 8, <= 128");
  const int BLK = (int)blk;
  TORCH_CHECK(BLK == 32 || BLK == 64 || BLK == 128, "block size must be 32, 64 or 128");
  TORCH_CHECK(LqPad % BLK == 0 && LkPad % BLK == 0, "padded lengths must be block multiples");
  // BR defaults to the block size so a 32-token block still tiles cleanly
  const int BR = br > 0 ? (int)br : (BLK < 32 ? BLK : 32);
  TORCH_CHECK(BR % 4 == 0 && BLK % BR == 0, "br must divide the block size");
  const int nB = (int)kbar.size(1);
  TORCH_CHECK(kbar.dim() == 3 && vbar.sizes() == kbar.sizes(), "kbar/vbar must be [Hkv,nB,D]");
  TORCH_CHECK((int)kbar.size(0) == Hkv && (int)kbar.size(2) == D, "kbar must be [Hkv,nB,D]");
  TORCH_CHECK(nB <= kMaskWords * 32, "too many blocks for the selection bitmask");
  TORCH_CHECK(loglen.dim() == 1 && (int)loglen.size(0) == nB
              && loglen.scalar_type() == torch::kFloat, "loglen must be fp32 [nB]");
  TORCH_CHECK(lq_valid > 0 && lq_valid <= LqPad && lk_valid > 0 && lk_valid <= LkPad,
              "valid lengths out of range");

  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  auto kb = kbar.contiguous(), vb = vbar.contiguous(), ll = loglen.contiguous();

  // The pooled-key tiles are BC wide, so a trailing partial tile makes the
  // matmul read past the end of kbar/vbar. Those columns are masked to -inf in
  // the score path and so LOOK harmless -- but the PV matmul then evaluates
  // 0 * garbage, and 0 * NaN is NaN, which silently poisons the accumulator
  // depending on whatever the allocator last left on that page. Zero-pad so
  // out-of-range reads are exactly zero.
  const int nBpad = ((nB + BLK - 1) / BLK) * BLK;
  if (nBpad != nB) {
    auto kbp = torch::zeros({Hkv, nBpad, D}, kb.options());
    kbp.slice(1, 0, nB).copy_(kb);
    auto vbp = torch::zeros({Hkv, nBpad, D}, vb.options());
    vbp.slice(1, 0, nB).copy_(vb);
    auto llp = torch::zeros({nBpad}, ll.options());
    llp.slice(0, 0, nB).copy_(ll);
    kb = kbp; vb = vbp; ll = llp;
  }
  auto out = torch::zeros({B, Hq, LqPad, D}, q.options());
  const int nQtiles = LqPad / BR;
  auto stat = torch::zeros({B * Hq, nQtiles}, q.options().dtype(torch::kInt));

  const long qHead = (long)LqPad * D, qRow = D;
  const long kHead = (long)LkPad * D, kRow = D;
  const long oHead = (long)LqPad * D, oRow = D;
  const long kbHead = (long)nBpad * D, kbRow = D;
  std::array<int32_t, 28> shArr = {
      B, Hq, Hkv, LqPad, LkPad, (int32_t)lq_valid, (int32_t)lk_valid, nB,
      (int32_t)qHead, (int32_t)qRow, (int32_t)kHead, (int32_t)kRow,
      (int32_t)oHead, (int32_t)oRow, (int32_t)kbHead, (int32_t)kbRow,
      (int32_t)prefix_rows, (int32_t)audio_row_start, (int32_t)audio_row_stop,
      (int32_t)prefix_blocks, (int32_t)audio_blk_start, (int32_t)audio_blk_stop,
      (int32_t)local_radius, BLK, nQtiles,
      (int32_t)video_row_start, (int32_t)rows_per_frame, (int32_t)temporal_taps};
  std::array<float, 2> prArr = {(float)scale, (float)beta};

  id<MTLComputePipelineState> pso = pso_for_fused(D, BR, BLK, half_s, half_io);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  id<MTLBuffer> kbB = __builtin_bit_cast(id<MTLBuffer>, kb.storage().data());
  id<MTLBuffer> vbB = __builtin_bit_cast(id<MTLBuffer>, vb.storage().data());
  id<MTLBuffer> llB = __builtin_bit_cast(id<MTLBuffer>, ll.storage().data());
  id<MTLBuffer> stB = __builtin_bit_cast(id<MTLBuffer>, stat.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const NSUInteger kbO = kb.storage_offset() * kb.element_size();
  const NSUInteger vbO = vb.storage_offset() * vb.element_size();
  const NSUInteger llO = ll.storage_offset() * ll.element_size();
  const NSUInteger stO = stat.storage_offset() * stat.element_size();
  const NSUInteger gx = (NSUInteger)nQtiles;
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
      [enc setBuffer:kbB offset:kbO atIndex:6];
      [enc setBuffer:vbB offset:vbO atIndex:7];
      [enc setBuffer:llB offset:llO atIndex:8];
      [enc setBuffer:stB offset:stO atIndex:9];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
  });
  return {out, stat};
}

// ---- diag64 stage 1 host ---------------------------------------------------

static id<MTLComputePipelineState> pso_for_route(int D, bool half_io) {
  const long key = (long)D * 2 + (half_io ? 1 : 0);
  auto it = g_pso_route.find(key);
  if (it != g_pso_route.end()) return it->second;

  std::string src(kRouteTemplate);
  sub_all(src, "@D@", std::to_string(D));
  sub_all(src, "@MW@", std::to_string(kMaskWords));
  sub_all(src, "@DT@", half_io ? "half" : "bfloat");

  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                         options:opts error:&err];
  TORCH_CHECK(lib, "sol_diag_route Metal compile failed (D=", D, "): ",
              err ? err.localizedDescription.UTF8String : "unknown");
  id<MTLFunction> fn = [lib newFunctionWithName:@"sol_diag_route"];
  TORCH_CHECK(fn, "sol_diag_route not found in compiled library");
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "sol_diag_route pipeline creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_pso_route[key] = pso;
  return pso;
}

// (mask [Hq, nQ, MW] int32, counts [Hq, nQ] int32) = diag64 route(Q, centroids).
// Q is [1, Hq, LqPad, D] with LqPad a multiple of blk; lq_valid bounds which
// rows are pooled into a query centroid.
std::vector<torch::Tensor> sol_diag_route(
    torch::Tensor q, torch::Tensor kcent, torch::Tensor kmean, torch::Tensor kvar,
    double scale, double tau, int64_t lq_valid, int64_t blk,
    int64_t sink_blocks, int64_t local_radius) {
  TORCH_CHECK(q.dim() == 4, "q must be [B,Hq,LqPad,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1);
  const int LqPad = (int)q.size(2), D = (int)q.size(3);
  TORCH_CHECK(B == 1, "B must be 1");
  TORCH_CHECK(q.is_mps() && kcent.is_mps() && kmean.is_mps() && kvar.is_mps(),
              "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 || half_io, "q must be bf16 or fp16");
  TORCH_CHECK(kcent.dim() == 3 && kcent.scalar_type() == torch::kFloat,
              "kcent must be fp32 [Hkv, nB, D]");
  const int Hkv = (int)kcent.size(0), nB = (int)kcent.size(1);
  TORCH_CHECK((int)kcent.size(2) == D, "kcent head_dim must match q");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(kmean.dim() == 2 && kvar.dim() == 2
              && (int)kmean.size(0) == Hkv && (int)kvar.size(0) == Hkv
              && (int)kmean.size(1) == D && (int)kvar.size(1) == D
              && kmean.scalar_type() == torch::kFloat
              && kvar.scalar_type() == torch::kFloat,
              "kmean/kvar must be fp32 [Hkv, D]");
  const int BLK = (int)blk;
  TORCH_CHECK(BLK > 0 && LqPad % BLK == 0, "LqPad must be a multiple of blk");
  TORCH_CHECK(D % 8 == 0 && D <= 128, "head_dim must be a multiple of 8, <= 128");
  TORCH_CHECK(nB <= kMaskWords * 32, "too many blocks for the selection bitmask");
  TORCH_CHECK(lq_valid > 0 && lq_valid <= LqPad, "lq_valid out of range");
  const int nQ = LqPad / BLK;
  // Every query block must hold at least one real row, or its centroid would be
  // pooled from padding alone and route on garbage.
  TORCH_CHECK((int)((lq_valid + BLK - 1) / BLK) == nQ,
              "LqPad must not exceed lq_valid rounded up to a block boundary");

  auto qc = q.contiguous();
  auto kc = kcent.contiguous(), km = kmean.contiguous(), kv = kvar.contiguous();
  auto mask = torch::zeros({Hq, nQ, kMaskWords}, q.options().dtype(torch::kInt));
  auto counts = torch::zeros({Hq, nQ}, q.options().dtype(torch::kInt));

  const long qHead = (long)LqPad * D, qRow = D;
  std::array<int32_t, 10> shArr = {
      Hq, Hkv, (int32_t)lq_valid, nB, nQ, (int32_t)qHead, (int32_t)qRow,
      BLK, (int32_t)sink_blocks, (int32_t)local_radius};
  std::array<float, 2> prArr = {(float)scale, (float)tau};

  id<MTLComputePipelineState> pso = pso_for_route(D, half_io);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> cB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> mB = __builtin_bit_cast(id<MTLBuffer>, km.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, kv.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, mask.storage().data());
  id<MTLBuffer> nB_ = __builtin_bit_cast(id<MTLBuffer>, counts.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger cO = kc.storage_offset() * kc.element_size();
  const NSUInteger mO = km.storage_offset() * km.element_size();
  const NSUInteger vO = kv.storage_offset() * kv.element_size();
  const NSUInteger kO = mask.storage_offset() * mask.element_size();
  const NSUInteger nO = counts.storage_offset() * counts.element_size();

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:qB offset:qO atIndex:0];
      [enc setBuffer:cB offset:cO atIndex:1];
      [enc setBuffer:mB offset:mO atIndex:2];
      [enc setBuffer:vB offset:vO atIndex:3];
      [enc setBuffer:kB offset:kO atIndex:4];
      [enc setBuffer:nB_ offset:nO atIndex:5];
      [enc setBytes:shArr.data() length:sizeof(shArr) atIndex:6];
      [enc setBytes:prArr.data() length:sizeof(prArr) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nQ, (NSUInteger)Hq, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
  });
  return {mask, counts};
}

// ---- diag64 stage 2 host ---------------------------------------------------

static id<MTLComputePipelineState> pso_for_masked(int D, int BR, int BLK,
                                                  bool half_s, bool half_io) {
  const long key = (((long)D * 256 + BR) * 256 + BLK) * 4 + (half_s ? 1 : 0) + (half_io ? 2 : 0);
  auto it = g_pso_masked.find(key);
  if (it != g_pso_masked.end()) return it->second;

  const int nsg = 4;
  std::string src(kMaskedTemplate);
  sub_all(src, "@D@", std::to_string(D));
  sub_all(src, "@BR@", std::to_string(BR));
  sub_all(src, "@BLK@", std::to_string(BLK));
  sub_all(src, "@NSG@", std::to_string(nsg));
  sub_all(src, "@MW@", std::to_string(kMaskWords));
  const char* dt = half_io ? "half" : "bfloat";
  sub_all(src, "@DT@", dt);
  sub_all(src, "@ST@", half_s ? dt : "float");
  sub_all(src, "@MP@", half_s ? "true" : "false");

  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                         options:opts error:&err];
  TORCH_CHECK(lib, "sol_masked_corrected Metal compile failed (D=", D, ", BR=", BR,
              ", BLK=", BLK, "): ", err ? err.localizedDescription.UTF8String : "unknown");
  id<MTLFunction> fn = [lib newFunctionWithName:@"sol_masked_corrected_fwd"];
  TORCH_CHECK(fn, "sol_masked_corrected_fwd not found in compiled library");
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "sol_masked_corrected pipeline creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_pso_masked[key] = pso;
  return pso;
}

// Corrected sparse attention under a route computed by sol_diag_route.
// Returns (out, stat), where stat[bh, qtile] is the number of KV blocks the
// tile actually computed — it must equal the stage-1 count for that tile's
// query block, which is what the route-parity test asserts.
std::vector<torch::Tensor> sol_masked_corrected_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor kbar, torch::Tensor vbar, torch::Tensor loglen,
    torch::Tensor mask, double scale, int64_t lq_valid, int64_t lk_valid,
    int64_t blk, int64_t br, bool half_s, bool correct) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "inputs must be [B,H,L,D]");
  const int B = (int)q.size(0), Hq = (int)q.size(1);
  const int LqPad = (int)q.size(2), D = (int)q.size(3);
  const int Hkv = (int)k.size(1), LkPad = (int)k.size(2);
  TORCH_CHECK(B == 1, "B must be 1");
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps() && mask.is_mps(),
              "inputs must be on mps");
  const bool half_io = q.scalar_type() == torch::kHalf;
  TORCH_CHECK((q.scalar_type() == torch::kBFloat16 || half_io)
              && k.scalar_type() == q.scalar_type()
              && v.scalar_type() == q.scalar_type()
              && kbar.scalar_type() == q.scalar_type()
              && vbar.scalar_type() == q.scalar_type(),
              "q/k/v/kbar/vbar must share dtype (bf16 or fp16)");
  TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  TORCH_CHECK(D % 8 == 0 && D <= 128, "head_dim must be a multiple of 8, <= 128");
  const int BLK = (int)blk;
  TORCH_CHECK(BLK == 32 || BLK == 64 || BLK == 128, "block size must be 32, 64 or 128");
  TORCH_CHECK(LqPad % BLK == 0 && LkPad % BLK == 0, "padded lengths must be block multiples");
  const int BR = br > 0 ? (int)br : (BLK < 32 ? BLK : 32);
  TORCH_CHECK(BR % 4 == 0 && BLK % BR == 0, "br must divide the block size");
  const int nB = (int)kbar.size(1);
  TORCH_CHECK(kbar.dim() == 3 && vbar.sizes() == kbar.sizes(), "kbar/vbar must be [Hkv,nB,D]");
  TORCH_CHECK((int)kbar.size(0) == Hkv && (int)kbar.size(2) == D, "kbar must be [Hkv,nB,D]");
  TORCH_CHECK(nB <= kMaskWords * 32, "too many blocks for the selection bitmask");
  TORCH_CHECK(loglen.dim() == 1 && (int)loglen.size(0) == nB
              && loglen.scalar_type() == torch::kFloat, "loglen must be fp32 [nB]");
  TORCH_CHECK(lq_valid > 0 && lq_valid <= LqPad && lk_valid > 0 && lk_valid <= LkPad,
              "valid lengths out of range");
  const int nQ = LqPad / BLK;
  TORCH_CHECK(mask.dim() == 3 && (int)mask.size(0) == Hq && (int)mask.size(1) == nQ
              && (int)mask.size(2) == kMaskWords
              && mask.scalar_type() == torch::kInt,
              "mask must be int32 [Hq, LqPad/blk, ", kMaskWords, "]");

  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  auto kb = kbar.contiguous(), vb = vbar.contiguous(), ll = loglen.contiguous();
  auto mk = mask.contiguous();

  // The pooled-key tiles are BC wide, so a trailing partial tile makes the
  // matmul read past the end of kbar/vbar. Those columns are masked to -inf in
  // the score path and so LOOK harmless -- but the PV matmul then evaluates
  // 0 * garbage, and 0 * NaN is NaN, which silently poisons the accumulator
  // depending on whatever the allocator last left on that page. Zero-pad so
  // out-of-range reads are exactly zero.
  const int nBpad = ((nB + BLK - 1) / BLK) * BLK;
  if (nBpad != nB) {
    auto kbp = torch::zeros({Hkv, nBpad, D}, kb.options());
    kbp.slice(1, 0, nB).copy_(kb);
    auto vbp = torch::zeros({Hkv, nBpad, D}, vb.options());
    vbp.slice(1, 0, nB).copy_(vb);
    auto llp = torch::zeros({nBpad}, ll.options());
    llp.slice(0, 0, nB).copy_(ll);
    kb = kbp; vb = vbp; ll = llp;
  }
  auto out = torch::zeros({B, Hq, LqPad, D}, q.options());
  const int nQtiles = LqPad / BR;
  auto stat = torch::zeros({B * Hq, nQtiles}, q.options().dtype(torch::kInt));

  const long qHead = (long)LqPad * D, qRow = D;
  const long kHead = (long)LkPad * D, kRow = D;
  const long oHead = (long)LqPad * D, oRow = D;
  const long kbHead = (long)nBpad * D, kbRow = D;
  std::array<int32_t, 20> shArr = {
      B, Hq, Hkv, LqPad, LkPad, (int32_t)lq_valid, (int32_t)lk_valid, nB,
      (int32_t)qHead, (int32_t)qRow, (int32_t)kHead, (int32_t)kRow,
      (int32_t)oHead, (int32_t)oRow, (int32_t)kbHead, (int32_t)kbRow,
      BLK, nQtiles, nQ, correct ? 1 : 0};
  std::array<float, 1> prArr = {(float)scale};

  id<MTLComputePipelineState> pso = pso_for_masked(D, BR, BLK, half_s, half_io);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  id<MTLBuffer> kbB = __builtin_bit_cast(id<MTLBuffer>, kb.storage().data());
  id<MTLBuffer> vbB = __builtin_bit_cast(id<MTLBuffer>, vb.storage().data());
  id<MTLBuffer> llB = __builtin_bit_cast(id<MTLBuffer>, ll.storage().data());
  id<MTLBuffer> mkB = __builtin_bit_cast(id<MTLBuffer>, mk.storage().data());
  id<MTLBuffer> stB = __builtin_bit_cast(id<MTLBuffer>, stat.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const NSUInteger kbO = kb.storage_offset() * kb.element_size();
  const NSUInteger vbO = vb.storage_offset() * vb.element_size();
  const NSUInteger llO = ll.storage_offset() * ll.element_size();
  const NSUInteger mkO = mk.storage_offset() * mk.element_size();
  const NSUInteger stO = stat.storage_offset() * stat.element_size();

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
      [enc setBuffer:kbB offset:kbO atIndex:6];
      [enc setBuffer:vbB offset:vbO atIndex:7];
      [enc setBuffer:llB offset:llO atIndex:8];
      [enc setBuffer:mkB offset:mkO atIndex:9];
      [enc setBuffer:stB offset:stO atIndex:10];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nQtiles, (NSUInteger)(B * Hq), 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
  });
  return {out, stat};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sol_diag_route", &sol_diag_route,
        "diag64 block router: (mask, counts) from pooled query/key centroids",
        py::arg("q"), py::arg("kcent"), py::arg("kmean"), py::arg("kvar"),
        py::arg("scale"), py::arg("tau"), py::arg("lq_valid"),
        py::arg("blk") = 64, py::arg("sink_blocks") = 0,
        py::arg("local_radius") = 1);
  m.def("sol_masked_corrected_fwd", &sol_masked_corrected_fwd,
        "corrected sparse attention under a precomputed diag64 route",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("kbar"), py::arg("vbar"), py::arg("loglen"), py::arg("mask"),
        py::arg("scale"), py::arg("lq_valid"), py::arg("lk_valid"),
        py::arg("blk") = 64, py::arg("br") = 0, py::arg("half_s") = false,
        py::arg("correct") = true);
  m.def("sol_fused_fwd", &sol_fused_fwd,
        "fused route + sparse attention + centroid correction (Phase E) on MPS",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("kbar"), py::arg("vbar"), py::arg("loglen"),
        py::arg("scale"), py::arg("beta"), py::arg("lq_valid"), py::arg("lk_valid"),
        py::arg("blk") = 64, py::arg("br") = 0, py::arg("half_s") = false,
        py::arg("prefix_rows") = 0, py::arg("audio_row_start") = 0,
        py::arg("audio_row_stop") = 0, py::arg("prefix_blocks") = 0,
        py::arg("audio_blk_start") = 0, py::arg("audio_blk_stop") = 0,
        py::arg("local_radius") = 1, py::arg("video_row_start") = 0,
        py::arg("rows_per_frame") = 0, py::arg("temporal_taps") = 0);
  m.def("sol_sparse_fwd", &sol_sparse_fwd,
        "block-sparse flash attention returning (out, lse) on MPS, [B,H,L,D]",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("block_idx"), py::arg("block_count"), py::arg("scale"),
        py::arg("lq_valid"), py::arg("lk_valid"),
        py::arg("blk") = 64, py::arg("br") = 0, py::arg("half_s") = false);
  m.def("warmup", [](int64_t D, int64_t BR, int64_t BLK, bool half_s, bool half_io) {
          pso_for((int)D, (int)BR, (int)BLK, half_s, half_io); return true;
        }, "compile the pipeline ahead of first use",
        py::arg("D") = 128, py::arg("BR") = 32, py::arg("blk") = 64,
        py::arg("half_s") = false, py::arg("half_io") = false);
}
