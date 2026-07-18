// ============================================================
// Staged W8A16 GEMM: half × fp8(e4m3) → f32 TensorOps kernel (M5+, Metal 4.1)
//
// C[M,N] (f32) = A[M,K] (half) @ B[N,K]^T (fp8 e4m3)   -- NT layout
//
// Design note (why this is NOT a byte-for-byte Cider port): the SDK's
// cooperative-operand run path (register-staged fragments, what
// int8_gemm.mm uses) supports no fp8 element types at all — fp8 operands
// are only accepted through memory-tensor runs, where the implementation
// does its own internal (double-buffered) staging. So this kernel keeps
// the single full-K op.run over device tensors from fp8_ext, and instead
// upgrades everything AROUND it that the simple kernel left on the table:
//   • larger threadgroup tiles (128x128 / 8 SG default, plus 64x128 and
//     128x64 variants for bench-driven selection) vs 64x64 / 4 SG,
//   • the int8 kernel's L2-friendly grid swizzle (tile rasterization),
//   • a small-M pipeline (32x128 / 4 SG) for M <= 64,
//   • fused epilogue: D = bf16(acc_f32 + bias_f32[n]) straight from the
//     destination cooperative tensor — the f32 [M,N] product never
//     round-trips through global memory.
//
// Host dispatch mirrors int8_gemm.mm: encoded on torch's MPS stream, no
// COMMIT_AND_WAIT (a per-call wait serializes the whole pipeline).
// ============================================================

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using namespace at::mps;

static const char* kSrc = R"MTL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#if !defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__)
#error "Metal 4.1 fp8 format type unavailable"
#endif
using fp8_t = metal::metal_fp8_e4m3_format;

// ── GEMM body: one full-K op.run per threadgroup tile ───────────
// The swizzle remaps the flat grid so consecutive threadgroups walk small
// column bands of the output (int8_gemm.mm policy) — B tiles stay hot in L2.
template <int BM, int BN, int NSG, typename StoreFn>
void w8a16_body(device half *A, device uchar *B,
                uint M, uint N, uint K, uint swizzle_log, uint tiles_m,
                uint tiles_n, uint2 tgid, StoreFn store_fn) {
  uint tid_y = (tgid.y << swizzle_log) + (tgid.x & ((1u << swizzle_log) - 1u));
  uint tid_x = tgid.x >> swizzle_log;
  if (tid_x >= tiles_n || tid_y >= tiles_m) return;

  const int m0 = int(tid_y) * BM;
  const int n0 = int(tid_x) * BN;

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  // Full-K operands, K-first extents (NT: op reads K from extent(0)); the
  // implementation picks its internal tileK and double-buffers the staging.
  auto mA = tensor(A + ulong(m0) * K,
                   dextents<int, 2>{int(K), min(BM, int(M) - m0)},
                   array<int, 2>{1, int(K)});
  auto mB = tensor<device fp8_t, dextents<int, 2>, tensor_inline>(
      B + ulong(n0) * K,
      dextents<int, 2>{int(K), min(BN, int(N) - n0)},
      array<int, 2>{1, int(K)});
  using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
  using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
  auto cC = op.template get_destination_cooperative_tensor<AT, BT, float>();
  for (uint16_t i = 0; i < cC.get_capacity(); ++i)
    if (cC.is_valid_element(i)) cC[i] = 0.0f;

  op.run(mA, mB, cC);

  store_fn(cC, uint(m0), uint(n0));
}

// ── Store epilogues (destination cooperative tensor index API) ──
struct StoreF32 {
  device float *C;
  uint M, N;
  template <typename CT>
  void operator()(thread CT &ct, uint m0, uint n0) thread {
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
      if (!ct.is_valid_element(i)) continue;
      auto ix = ct.get_multidimensional_index(i);
      const uint m = m0 + uint(ix[1]);
      const uint n = n0 + uint(ix[0]);
      if (m < M && n < N) C[ulong(m) * N + n] = ct[i];
    }
  }
};

// D = bf16(acc_f32 + bias_f32[n]) — identical rounding to patch #18's Python
// epilogue (f32 bias add, single rounding to bf16).
struct StoreBiasBF16 {
  device bfloat *D;
  const device float *bias;
  uint M, N, has_bias;
  template <typename CT>
  void operator()(thread CT &ct, uint m0, uint n0) thread {
    for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
      if (!ct.is_valid_element(i)) continue;
      auto ix = ct.get_multidimensional_index(i);
      const uint m = m0 + uint(ix[1]);
      const uint n = n0 + uint(ix[0]);
      if (m < M && n < N) {
        float acc = ct[i];
        if (has_bias != 0u) acc += bias[n];
        D[ulong(m) * N + n] = bfloat(acc);
      }
    }
  }
};

// ── Entry-point stamping ────────────────────────────────────────
#define W8A16_F32_KERNEL(NAME, BM, BN, NSG)                                    \
kernel void NAME(                                                              \
    device half *A [[buffer(0)]], device uchar *B [[buffer(1)]],              \
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],            \
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],           \
    constant uint &swizzle_log [[buffer(6)]],                                 \
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],\
    uint2 tgid [[threadgroup_position_in_grid]]) {                            \
  StoreF32 st{C, M, N};                                                       \
  w8a16_body<BM, BN, NSG>(A, B, M, N, K, swizzle_log, tiles_m, tiles_n,       \
                          tgid, st);                                          \
}

#define W8A16_BIAS_KERNEL(NAME, BM, BN, NSG)                                   \
kernel void NAME(                                                              \
    device half *A [[buffer(0)]], device uchar *B [[buffer(1)]],              \
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],           \
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],           \
    constant uint &swizzle_log [[buffer(6)]],                                 \
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],\
    const device float *bias [[buffer(9)]],                                   \
    constant uint &has_bias [[buffer(10)]],                                   \
    uint2 tgid [[threadgroup_position_in_grid]]) {                            \
  StoreBiasBF16 st{D, bias, M, N, has_bias};                                  \
  w8a16_body<BM, BN, NSG>(A, B, M, N, K, swizzle_log, tiles_m, tiles_n,       \
                          tgid, st);                                          \
}

W8A16_F32_KERNEL(w8a16_f32_g128x128x8, 128, 128, 8)
W8A16_F32_KERNEL(w8a16_f32_g64x128x4,   64, 128, 4)
W8A16_F32_KERNEL(w8a16_f32_g128x64x8,  128,  64, 8)
W8A16_F32_KERNEL(w8a16_f32_small,       32, 128, 4)

W8A16_BIAS_KERNEL(w8a16_bias_g128x128x8, 128, 128, 8)
W8A16_BIAS_KERNEL(w8a16_bias_g64x128x4,   64, 128, 4)
W8A16_BIAS_KERNEL(w8a16_bias_g128x64x8,  128,  64, 8)
W8A16_BIAS_KERNEL(w8a16_bias_small,       32, 128, 4)
)MTL";

// ── Host ─────────────────────────────────────────────────────────
struct Variant { const char* f32_name; const char* bias_name; int BM, BN, NSG; };
// Order matters: index is the `variant` selector exposed to Python.
static const Variant kVariants[] = {
  {"w8a16_f32_g128x128x8", "w8a16_bias_g128x128x8", 128, 128, 8},
  {"w8a16_f32_g64x128x4",  "w8a16_bias_g64x128x4",   64, 128, 4},
  {"w8a16_f32_g128x64x8",  "w8a16_bias_g128x64x8",  128,  64, 8},
  {"w8a16_f32_small",      "w8a16_bias_small",       32, 128, 4},
};
static constexpr int kNumVariants = 4;
static constexpr int kSmallVariant = 3;

static id<MTLComputePipelineState> g_pso_f32[kNumVariants]  = {nil, nil, nil, nil};
static id<MTLComputePipelineState> g_pso_bias[kNumVariants] = {nil, nil, nil, nil};
static id<MTLLibrary> g_lib = nil;
static std::string g_compile_error;

static id<MTLLibrary> build_library() {
  if (g_lib) return g_lib;
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                         options:opts error:&err];
  if (!lib) {
    g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
    TORCH_CHECK(false, "w8a16 Metal library compile failed: ", g_compile_error);
  }
  g_lib = lib;
  return lib;
}

static id<MTLComputePipelineState> pso_for(int variant, bool fused) {
  id<MTLComputePipelineState>* slot = fused ? &g_pso_bias[variant] : &g_pso_f32[variant];
  if (*slot) return *slot;
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  id<MTLLibrary> lib = build_library();
  const char* name = fused ? kVariants[variant].bias_name : kVariants[variant].f32_name;
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  TORCH_CHECK(fn, "kernel function not found: ", name);
  NSError* err = nil;
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "pipeline state creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  *slot = pso;
  return pso;
}

struct Geom {
  int variant;
  uint THREADS, swizzle_log, grid_x, grid_y, tiles_m, tiles_n;
};

// variant < 0 → auto (small pipeline for M<=64, else the 128x128 default).
static Geom geom_for(int M, int N, int variant) {
  Geom g;
  g.variant = (variant >= 0 && variant < kNumVariants)
                  ? variant
                  : (M <= 64 ? kSmallVariant : 0);
  const Variant& v = kVariants[g.variant];
  g.THREADS = uint(v.NSG) * 32u;
  g.tiles_m = uint((M + v.BM - 1) / v.BM);
  g.tiles_n = uint((N + v.BN - 1) / v.BN);
  if (g.tiles_m <= 3) g.swizzle_log = 0;
  else if (g.tiles_m <= 6) g.swizzle_log = 1;
  else g.swizzle_log = 2;
  uint tile = 1u << g.swizzle_log;
  g.grid_x = g.tiles_n * tile;
  g.grid_y = (g.tiles_m + tile - 1) / tile;
  return g;
}

static void check_inputs(const torch::Tensor& a, const torch::Tensor& w_u8) {
  TORCH_CHECK(a.is_mps() && w_u8.is_mps(), "inputs must be on mps");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "activations must be half");
  TORCH_CHECK(w_u8.scalar_type() == torch::kByte, "weight must be uint8 (fp8 e4m3 bytes)");
  TORCH_CHECK(a.dim() == 2 && w_u8.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(a.size(1) == w_u8.size(1), "K mismatch: A[M,K], W[N,K]");
  TORCH_CHECK(a.is_contiguous() && w_u8.is_contiguous(), "inputs must be contiguous");
}

static void dispatch_kernel(id<MTLComputePipelineState> pso, const Geom& g,
                            const torch::Tensor& a, const torch::Tensor& w_u8,
                            const torch::Tensor& out,
                            const torch::Tensor* bias, uint has_bias, bool fused) {
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a.storage().data());
  id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, w_u8.storage().data());
  id<MTLBuffer> oBuf = __builtin_bit_cast(id<MTLBuffer>, out.storage().data());
  id<MTLBuffer> biasBuf = bias ? __builtin_bit_cast(id<MTLBuffer>, bias->storage().data()) : aBuf;
  const NSUInteger aOff = a.storage_offset() * a.element_size();
  const NSUInteger bOff = w_u8.storage_offset() * w_u8.element_size();
  const NSUInteger oOff = out.storage_offset() * out.element_size();
  const NSUInteger biasOff = bias ? bias->storage_offset() * bias->element_size() : 0;
  uint Mu = (uint)a.size(0), Nu = (uint)w_u8.size(0), Ku = (uint)a.size(1);
  uint swizzle_log = g.swizzle_log, tiles_m = g.tiles_m, tiles_n = g.tiles_n;
  uint THREADS = g.THREADS, grid_x = g.grid_x, grid_y = g.grid_y;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:aBuf offset:aOff atIndex:0];
      [enc setBuffer:bBuf offset:bOff atIndex:1];
      [enc setBuffer:oBuf offset:oOff atIndex:2];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:5];
      [enc setBytes:&swizzle_log length:sizeof(uint) atIndex:6];
      [enc setBytes:&tiles_m length:sizeof(uint) atIndex:7];
      [enc setBytes:&tiles_n length:sizeof(uint) atIndex:8];
      if (fused) {
        [enc setBuffer:biasBuf offset:biasOff atIndex:9];
        [enc setBytes:&has_bias length:sizeof(uint) atIndex:10];
      }
      [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    }
  });
  // No COMMIT_AND_WAIT — encoded on torch's MPS stream; torch commits at its
  // normal sync points (same as int8_gemm.mm).
}

// C[M,N] f32 = A[M,K] half @ W[N,K]^T fp8(e4m3 bytes). variant=-1 → auto.
torch::Tensor w8a16_f32(torch::Tensor a, torch::Tensor w_u8, int64_t variant) {
  check_inputs(a, w_u8);
  auto C = torch::empty({a.size(0), w_u8.size(0)}, a.options().dtype(torch::kFloat32));
  Geom g = geom_for((int)a.size(0), (int)w_u8.size(0), (int)variant);
  dispatch_kernel(pso_for(g.variant, false), g, a, w_u8, C, nullptr, 0u, false);
  return C;
}

// D[M,N] bf16 = bf16(A@W^T + bias_f32[n]) — fused epilogue. variant=-1 → auto.
torch::Tensor w8a16_bias_bf16(torch::Tensor a, torch::Tensor w_u8,
                              c10::optional<torch::Tensor> bias_opt, int64_t variant) {
  check_inputs(a, w_u8);
  uint has_bias = 0u;
  const torch::Tensor* bias = nullptr;
  torch::Tensor bias_t;
  if (bias_opt.has_value()) {
    bias_t = bias_opt.value();
    TORCH_CHECK(bias_t.is_mps() && bias_t.scalar_type() == torch::kFloat32,
                "bias must be f32 on mps");
    TORCH_CHECK(bias_t.numel() == w_u8.size(0), "bias must be [N]");
    TORCH_CHECK(bias_t.is_contiguous(), "bias must be contiguous");
    bias = &bias_t;
    has_bias = 1u;
  }
  auto D = torch::empty({a.size(0), w_u8.size(0)}, a.options().dtype(torch::kBFloat16));
  Geom g = geom_for((int)a.size(0), (int)w_u8.size(0), (int)variant);
  dispatch_kernel(pso_for(g.variant, true), g, a, w_u8, D, bias, has_bias, true);
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_f32", &w8a16_f32,
        "staged half x fp8(e4m3) -> f32 GEMM (NT) on MPS",
        py::arg("a"), py::arg("w_u8"), py::arg("variant") = -1);
  m.def("w8a16_bias_bf16", &w8a16_bias_bf16,
        "staged half x fp8(e4m3) + bias -> bf16 GEMM (NT, fused epilogue) on MPS",
        py::arg("a"), py::arg("w_u8"), py::arg("bias") = c10::nullopt,
        py::arg("variant") = -1);
  m.def("num_variants", []() { return (int64_t)kNumVariants; });
  m.def("warmup", []() {
    for (int v = 0; v < kNumVariants; ++v) { pso_for(v, false); pso_for(v, true); }
    return true;
  }, "compile library + build all pipelines");
}
