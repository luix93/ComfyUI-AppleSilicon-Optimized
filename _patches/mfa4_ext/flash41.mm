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
    device bfloat* Q  [[buffer(0)]],
    device bfloat* K  [[buffer(1)]],
    device bfloat* V  [[buffer(2)]],
    device bfloat* O  [[buffer(3)]],
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

    device bfloat* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device bfloat* Kp = K + ulong(b*Hkv+hkv)*ulong(Lk)*DD;
    device bfloat* Vp = V + ulong(b*Hkv+hkv)*ulong(Lk)*DD;

    constexpr auto qk_desc = matmul2d_descriptor(
        BR, BC, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;

    constexpr auto pv_desc = matmul2d_descriptor(
        BR, DD, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    threadgroup @ST@ tgS[BR*BC];
    threadgroup bfloat tgP[BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];
    threadgroup int tgAnyCorr;

    auto mQ = tensor(Qp + q0*DD, dextents<int,2>{DD, min(BR, Lq - q0)}, array<int,2>{1, DD});
    auto tS = tensor(&tgS[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});
    auto tP = tensor(&tgP[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});

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
                tgP[row*BC + cc*32 + lane] = bfloat(p);
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

    device bfloat* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= Lq || c >= DD) continue;
        Ob[r*DD + c] = bfloat(ctO[i] * tgInvL[row]);
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
                                           const char* fn_name = "flash_attn_fwd_bf16") {
  const long key = ((((((long)D * 1000 + BR) * 1000 + BC) * 2 + (bf16s ? 1 : 0))
                    * 100 + SR) * 1000 + BCV) * 4
                   + (std::string(fn_name) == "flash_attn_fwd_bf16" ? 0
                      : (std::string(fn_name) == "flash_attn_fwd_v2r_bf16" ? 1 : 2));
  auto it = g_pso.find(key);
  if (it != g_pso.end()) return it->second;

  std::string src(kSrcTemplate);
  sub_all(src, "@D@", std::to_string(D));
  sub_all(src, "@BR@", std::to_string(BR));
  sub_all(src, "@BC@", std::to_string(BC));
  sub_all(src, "@ST@", bf16s ? "bfloat" : "float");
  sub_all(src, "@NSG@", std::to_string(nsg_for(BR)));
  sub_all(src, "@SR@", std::to_string(SR));
  sub_all(src, "@BCV@", std::to_string(BCV));

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
                             int64_t v2r, int64_t sr, int64_t bcv) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "inputs must be on mps");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16
              && v.scalar_type() == torch::kBFloat16, "inputs must be bf16");
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
  if (BR <= 0) BR = 32;                       // bigger BR spills ctO registers
  int BC = (int)bc;
  if (BC <= 0) BC = 128;                      // wide KV tiles feed the matmul units
  TORCH_CHECK(BC % 32 == 0 && BC <= 256, "bc must be a multiple of 32, <= 256");
  int SR = (int)sr;
  if (SR <= 0) SR = 16;                       // native matmul granularity (8/32 collapse)
  int BCV = (int)bcv;
  if (BCV <= 0) BCV = 64;
  TORCH_CHECK(BCV % 32 == 0 && BCV <= 256, "bcv must be a multiple of 32, <= 256");

  bool use_v2r = false;
  if (v2r > 0) {
    TORCH_CHECK(Lk % BCV == 0, "v2r requires Lk % BCV == 0 (static-k PV descriptor)");
    TORCH_CHECK(v2r_probe(D, SR, BCV), "v2r not supported on this device for D=", D);
    use_v2r = true;
  } else if (v2r < 0) {
    use_v2r = (Lk % BCV == 0) && Lk >= 256 && v2r_probe(D, SR, BCV);
  }

  const int nsg = nsg_for(BR);
  id<MTLComputePipelineState> pso =
      use_v2r ? pso_for(D, BR, BC, bf16s, SR, BCV, "flash_attn_fwd_v2r_bf16")
              : pso_for(D, BR, BC, bf16s, SR, BCV);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_fwd_bf16", &flash_fwd_bf16,
        "bf16 flash attention fwd (Metal 4.1 TensorOps) on MPS",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("causal") = false, py::arg("br") = 0, py::arg("bc") = 0, py::arg("bf16s") = false,
        py::arg("v2r") = 0, py::arg("sr") = 0, py::arg("bcv") = 0);
  m.def("warmup", [](int64_t D, int64_t BR, int64_t BC) { pso_for((int)D, (int)BR, (int)BC, false); return true; },
        "compile + build pipeline for head dim D / BR rows / BC cols",
        py::arg("D") = 128, py::arg("BR") = 32, py::arg("BC") = 128);
  m.def("v2r_supported", [](int64_t D, int64_t SR, int64_t BCV) {
          return v2r_probe((int)D, (int)SR, (int)BCV);
        }, "device probe: register-resident-P support at (D, SR, BCV)",
        py::arg("D") = 128, py::arg("SR") = 16, py::arg("BCV") = 64);
}
