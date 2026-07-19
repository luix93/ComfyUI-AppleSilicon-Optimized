// ============================================================
// Wan 2.x hot-path kernels for torch / MPS — dtype-templated (@DT@:
// half | bfloat) because Wan computes in FP16 (unlike the bf16 families
// served by dit_ext). Compiled per dtype on demand at MTLMathModeSafe.
//
// ln_mod:       LayerNorm(no-affine) + (1+scale)+shift modulate with
//               PER-FRAME modulation rows (kills the repeat_e copies:
//               eager repeat_interleaves [B,F,C] -> [B,L,C] 220 MB per use,
//               6 uses per block) — 3+ passes + copies -> 1 pass.
// rms_rope_wide: RMSNorm(weighted, over the FULL C=H*D row) + optional
//               flux-format RoPE + write straight into [B,H,S,D] attention
//               layout (mode 0: plain relayout for v; 1: norm only, for
//               cross-attn q/k; 2: norm+rope, for self-attn q/k).
// addcmul_rows: out = x + y * gate with per-frame gate rows (the e[2]/e[5]
//               residual gates) — no repeat copy, one pass.
//
// Rounding mirrors eager op-by-op: fp32 math, one @DT@ rounding at each
// point eager rounds (LN out, addcmul out, rms out, rope out). float->half
// and the bf16_rte bit-twiddle both round-to-nearest-even.
// ============================================================

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <array>
#include <map>
#include <string>

using namespace at::mps;

static const char* kSrcTemplate = R"MTL(
#include <metal_stdlib>
using namespace metal;

// Deterministic RTE float -> storage dtype. For half the compiler cast is
// already RTE; for bfloat use the bit-twiddle (compiler picks truncation in
// some store paths).
inline half dt_rte_half(float f) { return half(f); }
inline bfloat dt_rte_bfloat(float f) {
  uint u = as_type<uint>(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return as_type<bfloat>(ushort(u >> 16));
}
#define RTE dt_rte_@DTN@

// LayerNorm (no affine) + modulate: out = shift + ln(x) * (1 + scale),
// shift/scale in per-frame rows [B, F, C] with ppf tokens per frame.
kernel void ln_mod(
    device const @DT@* x     [[buffer(0)]],  // [B*L, C]
    device const @DT@* shift [[buffer(1)]],  // [B, F, C]
    device const @DT@* scale [[buffer(2)]],  // [B, F, C]
    device @DT@*       out   [[buffer(3)]],
    constant uint&  L    [[buffer(4)]],
    constant uint&  F    [[buffer(5)]],
    constant uint&  C    [[buffer(6)]],
    constant float& eps  [[buffer(7)]],
    constant uint&  PPF  [[buffer(8)]],
    uint  tgid      [[threadgroup_position_in_grid]],
    ushort lid      [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint b = tgid / L;
  const uint l = tgid % L;
  const uint f = (F == 1u) ? 0u : min(l / PPF, F - 1u);
  const device @DT@* row = x + (ulong)tgid * C;

  float s1 = 0.0f, s2 = 0.0f;
  for (uint d = lid; d < C; d += 256) {
    float v = float(row[d]);
    s1 += v; s2 += v * v;
  }
  s1 = simd_sum(s1); s2 = simd_sum(s2);
  threadgroup float tg1[8], tg2[8];
  if (simd_lane == 0) { tg1[simd_id] = s1; tg2[simd_id] = s2; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t1 = 0.0f, t2 = 0.0f;
  for (ushort i = 0; i < 8; i++) { t1 += tg1[i]; t2 += tg2[i]; }
  const float mean = precise::divide(t1, float(C));
  const float var = precise::divide(t2, float(C)) - mean * mean;
  const float inv = precise::rsqrt(max(var, 0.0f) + eps);

  const device @DT@* sh = shift + ((ulong)b * F + f) * C;
  const device @DT@* sc = scale + ((ulong)b * F + f) * C;
  device @DT@* o = out + (ulong)tgid * C;
  for (uint d = lid; d < C; d += 256) {
    @DT@ h = RTE((float(row[d]) - mean) * inv);          // LN output rounding
    @DT@ s1v = RTE(1.0f + float(sc[d]));                 // (1 + scale)
    o[d] = RTE(float(sh[d]) + float(h) * float(s1v));    // addcmul rounding
  }
}

// Wide weighted RMSNorm over C = H*D + optional flux RoPE, written straight
// to [B, H, S, D]. mode: 0 = relayout only (v), 1 = norm (cross q/k),
// 2 = norm + rope (self q/k). freqs: [S, D/2, 2, 2] fp32, head-broadcast.
kernel void rms_rope_wide(
    device const @DT@*  xin   [[buffer(0)]],  // [B*S, C]
    device const @DT@*  w     [[buffer(1)]],  // [C] rms weight (mode>=1)
    device const float* freqs [[buffer(2)]],  // [S, D/2, 2, 2] (mode==2)
    device @DT@*        out   [[buffer(3)]],  // [B, H, S, D]
    constant uint&  S    [[buffer(4)]],
    constant uint&  H    [[buffer(5)]],
    constant uint&  D    [[buffer(6)]],
    constant float& eps  [[buffer(7)]],
    constant uint&  mode [[buffer(8)]],
    uint  tgid      [[threadgroup_position_in_grid]],
    ushort lid      [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint C = H * D;
  const uint b = tgid / S;
  const uint s = tgid % S;
  const device @DT@* row = xin + (ulong)tgid * C;

  float inv = 1.0f;
  if (mode >= 1u) {
    float ss = 0.0f;
    for (uint d = lid; d < C; d += 256) {
      float v = float(row[d]);
      ss += v * v;
    }
    ss = simd_sum(ss);
    threadgroup float tg[8];
    if (simd_lane == 0) tg[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (ushort i = 0; i < 8; i++) tot += tg[i];
    inv = precise::rsqrt(precise::divide(tot, float(C)) + eps);
  }

  for (uint d = lid; d < C; d += 256) {
    const uint h = d / D;
    const uint c = d % D;
    float y;
    if (mode == 0u) {
      y = float(row[d]);
    } else {
      // F.rms_norm: fp32 normalize -> dtype round -> dtype weight multiply
      @DT@ n0 = RTE(float(row[d]) * inv);
      y = float(RTE(float(n0) * float(w[d])));
    }
    if (mode == 2u) {
      // flux apply_rope1 pairs (2i, 2i+1) within the head
      const uint i = c >> 1;
      const uint j = c & 1u;
      const uint cp = (j == 0u) ? d + 1u : d - 1u;   // partner column
      float py;
      {
        @DT@ pn0 = RTE(float(row[cp]) * inv);
        py = float(RTE(float(pn0) * float(w[cp])));
      }
      const device float* fptr = freqs + ((ulong)s * (D >> 1) + i) * 4u + j * 2u;
      const float x0 = (j == 0u) ? y : py;
      const float x1 = (j == 0u) ? py : y;
      y = fptr[0] * x0 + fptr[1] * x1;
    }
    out[(((ulong)b * H + h) * S + s) * D + c] = RTE(y);
  }
}

// out = x + y * gate, gate in per-frame rows [B, F, C].
kernel void addcmul_rows(
    device const @DT@* x    [[buffer(0)]],  // [B*L, C]
    device const @DT@* y    [[buffer(1)]],  // [B*L, C]
    device const @DT@* gate [[buffer(2)]],  // [B, F, C]
    device @DT@*       out  [[buffer(3)]],
    constant uint& L   [[buffer(4)]],
    constant uint& F   [[buffer(5)]],
    constant uint& C   [[buffer(6)]],
    constant uint& PPF [[buffer(7)]],
    constant uint& N   [[buffer(8)]],       // B*L*C
    uint gid [[thread_position_in_grid]])
{
  if (gid >= N) return;
  const uint d = gid % C;
  const uint t = gid / C;          // b*L + l
  const uint b = t / L;
  const uint l = t % L;
  const uint f = (F == 1u) ? 0u : min(l / PPF, F - 1u);
  const float g = float(gate[((ulong)b * F + f) * C + d]);
  out[gid] = RTE(float(x[gid]) + float(y[gid]) * g);
}
)MTL";

// ── Host ─────────────────────────────────────────────────────────
static std::map<std::string, id<MTLComputePipelineState>> g_psos;
static std::map<std::string, id<MTLLibrary>> g_libs;

static void sub_all(std::string& src, const std::string& tok, const std::string& val) {
  size_t pos;
  while ((pos = src.find(tok)) != std::string::npos) src.replace(pos, tok.size(), val);
}

static id<MTLComputePipelineState> pso_for(const char* name, bool half_io) {
  const std::string key = std::string(name) + (half_io ? "|h" : "|b");
  auto it = g_psos.find(key);
  if (it != g_psos.end()) return it->second;

  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  const std::string lkey = half_io ? "h" : "b";
  id<MTLLibrary> lib = g_libs.count(lkey) ? g_libs[lkey] : nil;
  if (!lib) {
    std::string src(kSrcTemplate);
    sub_all(src, "@DT@", half_io ? "half" : "bfloat");
    sub_all(src, "@DTN@", half_io ? "half" : "bfloat");
    MTLCompileOptions* opts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      opts.mathMode = MTLMathModeSafe;
    } else {
      opts.fastMathEnabled = NO;
    }
    NSError* err = nil;
    lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                            options:opts error:&err];
    TORCH_CHECK(lib, "wan_ext Metal compile failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    g_libs[lkey] = lib;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  TORCH_CHECK(fn, "kernel not found: ", name);
  NSError* err = nil;
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "wan_ext pipeline failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_psos[key] = pso;
  return pso;
}

static bool half_io_of(const torch::Tensor& t) {
  const auto st = t.scalar_type();
  TORCH_CHECK(st == torch::kHalf || st == torch::kBFloat16,
              "wan_ext kernels take fp16 or bf16");
  return st == torch::kHalf;
}

struct Enc {
  std::array<id<MTLBuffer>, 4> bufs;
  std::array<NSUInteger, 4> offs;
  int n;
};

static Enc enc_of(std::initializer_list<torch::Tensor> ts) {
  Enc a{};
  int i = 0;
  for (const auto& t : ts) {
    a.bufs[i] = __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
    a.offs[i] = t.storage_offset() * t.element_size();
    i++;
  }
  a.n = i;
  return a;
}

// x [B,L,C]; shift/scale [B,F,C] with ppf tokens per frame.
torch::Tensor ln_mod(torch::Tensor x, torch::Tensor shift, torch::Tensor scale,
                     double eps, int64_t ppf) {
  TORCH_CHECK(x.is_mps() && x.dim() == 3 && x.is_contiguous(), "x must be contiguous [B,L,C]");
  const bool h = half_io_of(x);
  const int64_t B = x.size(0), L = x.size(1), C = x.size(2);
  auto sh = shift.contiguous();
  auto sc = scale.contiguous();
  TORCH_CHECK(sh.dim() == 3 && sh.sizes() == sc.sizes() && sh.size(0) == B
              && sh.size(2) == C, "shift/scale must be [B,F,C]");
  if (ppf <= 0) ppf = 1;
  auto out = torch::empty_like(x);
  auto a = enc_of({x, sh, sc, out});
  id<MTLComputePipelineState> p = pso_for("ln_mod", h);
  MPSStream* stream = getCurrentMPSStream();
  uint Lu = (uint)L, Fu = (uint)sh.size(1), Cu = (uint)C, Pu = (uint)ppf;
  float ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.n; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Lu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Fu length:sizeof(uint) atIndex:5];
      [enc setBytes:&Cu length:sizeof(uint) atIndex:6];
      [enc setBytes:&ef length:sizeof(float) atIndex:7];
      [enc setBytes:&Pu length:sizeof(uint) atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * L), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// x [B,S,H*D] -> [B,H,S,D]; mode 0=relayout, 1=+wide rms, 2=+rope.
torch::Tensor rms_rope_wide(torch::Tensor x, torch::Tensor w, torch::Tensor freqs,
                            int64_t H, int64_t mode, double eps) {
  TORCH_CHECK(x.is_mps() && x.dim() == 3 && x.is_contiguous(), "x must be contiguous [B,S,C]");
  const bool h = half_io_of(x);
  const int64_t B = x.size(0), S = x.size(1), C = x.size(2);
  TORCH_CHECK(C % H == 0, "C % H != 0");
  const int64_t D = C / H;
  if (mode >= 1) {
    TORCH_CHECK(w.numel() == C, "weight numel mismatch");
    w = w.contiguous();
  }
  if (mode == 2) {
    TORCH_CHECK(freqs.scalar_type() == torch::kFloat && freqs.is_contiguous()
                && freqs.numel() == S * (D / 2) * 4, "freqs must be [S,D/2,2,2] fp32");
    TORCH_CHECK(D % 2 == 0, "D must be even for rope");
  }
  auto out = torch::empty({B, H, S, D}, x.options());
  auto wa = (mode >= 1) ? w : x;          // dummy binding for mode 0
  auto fa = (mode == 2) ? freqs : torch::empty({1}, x.options().dtype(torch::kFloat));
  auto a = enc_of({x, wa, fa, out});
  id<MTLComputePipelineState> p = pso_for("rms_rope_wide", h);
  MPSStream* stream = getCurrentMPSStream();
  uint Su = (uint)S, Hu = (uint)H, Du = (uint)D, Mu = (uint)mode;
  float ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.n; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Su length:sizeof(uint) atIndex:4];
      [enc setBytes:&Hu length:sizeof(uint) atIndex:5];
      [enc setBytes:&Du length:sizeof(uint) atIndex:6];
      [enc setBytes:&ef length:sizeof(float) atIndex:7];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// out = x + y * gate_rows: x/y [B,L,C], gate [B,F,C].
torch::Tensor addcmul_rows(torch::Tensor x, torch::Tensor y, torch::Tensor gate,
                           int64_t ppf) {
  TORCH_CHECK(x.is_mps() && x.dim() == 3 && x.is_contiguous() && y.is_contiguous()
              && x.sizes() == y.sizes(), "x/y must be matching contiguous [B,L,C]");
  const bool h = half_io_of(x);
  const int64_t B = x.size(0), L = x.size(1), C = x.size(2);
  auto g = gate.contiguous();
  TORCH_CHECK(g.dim() == 3 && g.size(0) == B && g.size(2) == C, "gate must be [B,F,C]");
  if (ppf <= 0) ppf = 1;
  auto out = torch::empty_like(x);
  auto a = enc_of({x, y, g, out});
  id<MTLComputePipelineState> p = pso_for("addcmul_rows", h);
  MPSStream* stream = getCurrentMPSStream();
  const int64_t N = B * L * C;
  uint Lu = (uint)L, Fu = (uint)g.size(1), Cu = (uint)C, Pu = (uint)ppf, Nu = (uint)N;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.n; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Lu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Fu length:sizeof(uint) atIndex:5];
      [enc setBytes:&Cu length:sizeof(uint) atIndex:6];
      [enc setBytes:&Pu length:sizeof(uint) atIndex:7];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 255) / 256), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_mod", &ln_mod,
        "LayerNorm(no-affine) + (1+scale)+shift with per-frame rows (fp16/bf16, MPS)");
  m.def("rms_rope_wide", &rms_rope_wide,
        "Wide weighted RMSNorm + optional flux RoPE -> [B,H,S,D] (fp16/bf16, MPS)");
  m.def("addcmul_rows", &addcmul_rows,
        "x + y*gate with per-frame gate rows (fp16/bf16, MPS)");
}
