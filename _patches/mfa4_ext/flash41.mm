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
#include <string>

using namespace at::mps;

static const char* kSrcTemplate = R"MTL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BR = @BR@;
constant constexpr int BC = 32;
constant constexpr int DD = @D@;
constant constexpr int NSG = @NSG@;

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

    threadgroup float tgS[BR*BC];
    threadgroup bfloat tgP[BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];

    auto mQ = tensor(Qp + q0*DD, dextents<int,2>{DD, min(BR, Lq - q0)}, array<int,2>{1, DD});
    auto tS = tensor(&tgS[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});
    auto tP = tensor(&tgP[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});

    using PvA = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
    using PvB = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctO = pv_op.get_destination_cooperative_tensor<PvA, PvB, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i)
        if (ctO.is_valid_element(i)) ctO[i] = 0.0f;

    const int r0 = int(sgid) * (BR / NSG);
    float m[BR / NSG], l[BR / NSG];
    #pragma clang loop unroll(full)
    for (int r = 0; r < BR/NSG; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    const int shift = Lk - Lq;
    for (int j0 = 0; j0 < Lk; j0 += BC) {
        if (causal && j0 > q0 + BR - 1 + shift) break;

        for (int i = int(tid); i < BR*BC; i += NSG*32) tgS[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mK = tensor(Kp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        qk_op.run(mQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int j = j0 + int(lane);
        #pragma clang loop unroll(full)
        for (int r = 0; r < BR/NSG; ++r) {
            const int row  = r0 + r;
            const int qrow = q0 + row;
            float s = tgS[row*BC + lane] * scale;
            const bool ok = (j < Lk) && (qrow < Lq) && (!causal || j <= qrow + shift);
            s = ok ? s : -INFINITY;
            const float mb = simd_max(s);
            const float mn = max(m[r], mb);
            float p, c;
            if (mn == -INFINITY) { p = 0.0f; c = 1.0f; }
            else { p = exp(s - mn); c = exp(m[r] - mn); }
            l[r] = l[r]*c + simd_sum(p);
            m[r] = mn;
            tgCorr[row] = c;
            tgP[row*BC + lane] = bfloat(p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
            if (!ctO.is_valid_element(i)) continue;
            auto idx = ctO.get_multidimensional_index(i);
            ctO[i] *= tgCorr[int(idx[1])];
        }
        auto mV = tensor(Vp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        pv_op.run(tP, mV, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < BR/NSG; ++r)
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

static id<MTLComputePipelineState> pso_for(int D, int BR) {
  const long key = (long)D * 1000 + BR;
  auto it = g_pso.find(key);
  if (it != g_pso.end()) return it->second;

  std::string src(kSrcTemplate);
  sub_all(src, "@D@", std::to_string(D));
  sub_all(src, "@BR@", std::to_string(BR));
  sub_all(src, "@NSG@", std::to_string(nsg_for(BR)));

  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                         options:opts error:&err];
  if (!lib) {
    g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
    TORCH_CHECK(false, "flash41 Metal library compile failed (D=", D, ", BR=", BR, "): ", g_compile_error);
  }
  id<MTLFunction> fn = [lib newFunctionWithName:@"flash_attn_fwd_bf16"];
  TORCH_CHECK(fn, "flash_attn_fwd_bf16 not found in compiled library");
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "flash41 pipeline creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_pso[key] = pso;
  return pso;
}

// O[B,Hq,Lq,D] bf16 = flash_attention(Q, K, V). GQA via Hq % Hkv == 0.
torch::Tensor flash_fwd_bf16(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             double scale, bool causal, int64_t br) {
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

  // Small param tensors on-device (mirrors mtlflashattn's SH/PR convention).
  auto sh = torch::tensor({B, Hq, Hkv, Lq, Lk, causal ? 1 : 0},
                          torch::dtype(torch::kInt32)).to(q.device());
  auto pr = torch::tensor({(float)scale}, torch::dtype(torch::kFloat32)).to(q.device());

  int BR = (int)br;
  if (BR <= 0) BR = (Lq >= 8192) ? 128 : (Lq >= 2048 ? 64 : 32);   // K/V restream amortization
  id<MTLComputePipelineState> pso = pso_for(D, BR);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> qB = __builtin_bit_cast(id<MTLBuffer>, qc.storage().data());
  id<MTLBuffer> kB = __builtin_bit_cast(id<MTLBuffer>, kc.storage().data());
  id<MTLBuffer> vB = __builtin_bit_cast(id<MTLBuffer>, vc.storage().data());
  id<MTLBuffer> oB = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  id<MTLBuffer> sB = __builtin_bit_cast(id<MTLBuffer>, sh.storage().data());
  id<MTLBuffer> pB = __builtin_bit_cast(id<MTLBuffer>, pr.storage().data());
  const NSUInteger qO = qc.storage_offset() * qc.element_size();
  const NSUInteger kO = kc.storage_offset() * kc.element_size();
  const NSUInteger vO = vc.storage_offset() * vc.element_size();
  const NSUInteger oO = out.storage_offset() * out.element_size();
  const NSUInteger sO = sh.storage_offset() * sh.element_size();
  const NSUInteger pO = pr.storage_offset() * pr.element_size();
  const NSUInteger gx = (NSUInteger)((Lq + BR - 1) / BR);
  const NSUInteger gy = (NSUInteger)(B * Hq);

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:qB offset:qO atIndex:0];
      [enc setBuffer:kB offset:kO atIndex:1];
      [enc setBuffer:vB offset:vO atIndex:2];
      [enc setBuffer:oB offset:oO atIndex:3];
      [enc setBuffer:sB offset:sO atIndex:4];
      [enc setBuffer:pB offset:pO atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(nsg_for(BR) * 32), 1, 1)];
    }
  });
  // No COMMIT_AND_WAIT — encoded on torch's MPS stream (see int8_gemm.mm).
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_fwd_bf16", &flash_fwd_bf16,
        "bf16 flash attention fwd (Metal 4.1 TensorOps) on MPS",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("causal") = false, py::arg("br") = 0);
  m.def("warmup", [](int64_t D, int64_t BR) { pso_for((int)D, (int)BR); return true; },
        "compile + build pipeline for head dim D / BR rows",
        py::arg("D") = 128, py::arg("BR") = 32);
}
