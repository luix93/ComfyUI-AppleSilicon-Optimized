// ============================================================
// DiT hot-path kernels for torch / MPS (Apple Silicon)
//
// qkv_norm_rope: one fused pass replacing the per-layer eager chain in
// Lumina/Z-Image JointAttention:
//     split(qkv) -> view -> q_norm -> k_norm -> apply_rope -> movedim -> copy
// i.e. 2x F.rms_norm over tiny head_dim rows (pathological on MPS: ~1.6 ms
// each at 4432x30x128), a ~3.6 ms eager RoPE, plus 3 transpose copies —
// ~7.5 ms/layer eager, ~0.5 ms fused.
//
// Numerics match the eager chain step for step:
//   norm:  y = bf16(x_f32 * rsqrt(mean(x_f32^2) + eps)) *bf16 w
//          (F.rms_norm on MPS normalizes in fp32, rounds to bf16, then does a
//          bf16 weight multiply — verified bit-identical in tests)
//   rope:  out[j] = bf16(f[j][0]*y0_f32 + f[j][1]*y1_f32)   (flux apply_rope,
//          fp32 math on the bf16-rounded normed values, freqs fp32)
// The row-sum reduction order (simd butterfly + sequential combine) can differ
// from MPS's mean() tree, so the result is exact-or-1ulp; tests bound it.
//
// Grid: one threadgroup per (token, head-slot); D threads per group (D<=128,
// D%32==0). Head slots [0,Hq) = q (norm+rope), [Hq,Hq+Hkv) = k (norm+rope),
// [Hq+Hkv, Hq+2Hkv) = v (pure transposed copy). Outputs are written directly
// in attention layout [B, H, S, D] — the movedim disappears.
// ============================================================

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using namespace at::mps;

static const char* kSrc = R"MTL(
#include <metal_stdlib>
using namespace metal;

// Deterministic round-to-nearest-even float -> bfloat. The compiler's own
// float->bfloat conversion picks different roundings per code path (observed:
// RTE in arithmetic contexts, truncation in some store paths), which breaks
// parity with torch's RTE casts. Finite values only (activations).
inline bfloat bf16_rte(float f) {
  uint u = as_type<uint>(f);
  uint lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return as_type<bfloat>(ushort(u >> 16));
}

kernel void qkv_norm_rope_bf16(
    device const bfloat* qkv   [[buffer(0)]],  // [B*S, (Hq+2*Hkv)*D]
    device const float*  freqs [[buffer(1)]],  // [B*S, D/2, 2, 2] fp32
    device const bfloat* qw    [[buffer(2)]],  // [D] q_norm weight
    device const bfloat* kw    [[buffer(3)]],  // [D] k_norm weight
    device bfloat* qo [[buffer(4)]],           // [B, Hq,  S, D]
    device bfloat* ko [[buffer(5)]],           // [B, Hkv, S, D]
    device bfloat* vo [[buffer(6)]],           // [B, Hkv, S, D]
    constant uint&  S     [[buffer(7)]],
    constant uint&  Hq    [[buffer(8)]],
    constant uint&  Hkv   [[buffer(9)]],
    constant uint&  D     [[buffer(10)]],
    constant float& eps_q [[buffer(11)]],
    constant float& eps_k [[buffer(12)]],
    uint2  tgid      [[threadgroup_position_in_grid]],
    ushort lid       [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint token = tgid.x;           // b*S + s
  const uint b = token / S, s = token % S;
  const uint ht = tgid.y;
  const uint QS = (Hq + 2u * Hkv) * D;

  const bool is_q = ht < Hq;
  const bool is_v = ht >= Hq + Hkv;
  uint h, off;
  if (is_q)      { h = ht;            off = h * D; }
  else if (!is_v){ h = ht - Hq;       off = (Hq + h) * D; }
  else           { h = ht - Hq - Hkv; off = (Hq + Hkv + h) * D; }
  const device bfloat* row = qkv + (ulong)token * QS + off;

  if (is_v) {
    vo[(((ulong)b * Hkv + h) * S + s) * D + lid] = row[lid];
    return;
  }

  // fp32 sum of squares across the D threads (simd butterfly + combine).
  float x = float(row[lid]);
  float ss = simd_sum(x * x);
  threadgroup float tg[4];
  if (simd_lane == 0) tg[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  const ushort nsg = ushort(D >> 5);
  for (ushort i = 0; i < nsg; i++) tot += tg[i];

  const float eps = is_q ? eps_q : eps_k;
  const float inv = precise::rsqrt(precise::divide(tot, float(D)) + eps);
  const device bfloat* w = is_q ? qw : kw;
  // F.rms_norm rounding: fp32 normalize -> bf16 round -> bf16 weight multiply.
  bfloat y0 = bf16_rte(x * inv);
  bfloat y = bf16_rte(float(y0) * float(w[lid]));

  // RoPE (flux apply_rope): consecutive pairs (2i, 2i+1); fp32 math on the
  // bf16-rounded normed value; partner via lane shuffle (pair is lid^1).
  float yf = float(y);
  float partner = simd_shuffle_xor(yf, ushort(1));
  const uint i = lid >> 1;
  const uint j = lid & 1u;
  const device float* f = freqs + ((ulong)token * (D >> 1) + i) * 4u + j * 2u;
  const float x0 = (j == 0u) ? yf : partner;
  const float x1 = (j == 0u) ? partner : yf;
  const float o = f[0] * x0 + f[1] * x1;

  device bfloat* outp = is_q ? qo : ko;
  const uint Hh = is_q ? Hq : Hkv;
  outp[(((ulong)b * Hh + h) * S + s) * D + lid] = bf16_rte(o);
}
// RoPE-only: q/k already in attention layout [B, H, S, D]. One threadgroup per
// (token, head-slot); slots [0,H) rotate q, [H,2H) rotate k. Same math and
// rounding as flux apply_rope (fp32 pair rotation on bf16 values, RTE store).
kernel void rope_qk_bf16(
    device const bfloat* qin  [[buffer(0)]],  // [B, H, S, D]
    device const bfloat* kin  [[buffer(1)]],  // [B, H, S, D]
    device const float*  freqs[[buffer(2)]],  // [B*S, D/2, 2, 2] fp32
    device bfloat* qo [[buffer(3)]],
    device bfloat* ko [[buffer(4)]],
    constant uint& S [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint2  tgid [[threadgroup_position_in_grid]],
    ushort lid  [[thread_index_in_threadgroup]])
{
  const uint token = tgid.x;            // b*S + s
  const uint b = token / S, s = token % S;
  const uint ht = tgid.y;
  const bool is_q = ht < H;
  const uint h = is_q ? ht : ht - H;
  const ulong idx = (((ulong)b * H + h) * S + s) * D + lid;
  const device bfloat* src = is_q ? qin : kin;

  float x = float(src[idx]);
  float partner = simd_shuffle_xor(x, ushort(1));
  const uint i = lid >> 1;
  const uint j = lid & 1u;
  const device float* f = freqs + ((ulong)token * (D >> 1) + i) * 4u + j * 2u;
  const float x0 = (j == 0u) ? x : partner;
  const float x1 = (j == 0u) ? partner : x;
  device bfloat* dst = is_q ? qo : ko;
  dst[idx] = bf16_rte(f[0] * x0 + f[1] * x1);
}

// rmsnorm(x)*w then modulate by (1+scale): out = (bf16(x_f*inv)*w) * bf16(1+scale).
// One threadgroup (256 threads) per token row; row width Dm arbitrary.
kernel void norm_modulate_bf16(
    device const bfloat* x     [[buffer(0)]],  // [B*S, Dm]
    device const bfloat* w     [[buffer(1)]],  // [Dm]
    device const bfloat* scale [[buffer(2)]],  // [B, Dm]
    device bfloat*       out   [[buffer(3)]],  // [B*S, Dm]
    constant uint&  S    [[buffer(4)]],
    constant uint&  Dm   [[buffer(5)]],
    constant float& eps  [[buffer(6)]],
    uint  tgid      [[threadgroup_position_in_grid]],
    ushort lid      [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint b = tgid / S;
  const device bfloat* row = x + (ulong)tgid * Dm;
  float ss = 0.0f;
  for (uint d = lid; d < Dm; d += 256) {
    float v = float(row[d]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  threadgroup float tg[8];
  if (simd_lane == 0) tg[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (ushort i = 0; i < 8; i++) tot += tg[i];
  const float inv = precise::rsqrt(precise::divide(tot, float(Dm)) + eps);

  const device bfloat* sc = scale + (ulong)b * Dm;
  device bfloat* o = out + (ulong)tgid * Dm;
  for (uint d = lid; d < Dm; d += 256) {
    bfloat y1 = bf16_rte(float(row[d]) * inv);
    bfloat y = bf16_rte(float(y1) * float(w[d]));
    bfloat m = bf16_rte(1.0f + float(sc[d]));
    o[d] = bf16_rte(float(y) * float(m));
  }
}

// out = xres + tanh_gate * (rmsnorm(y)*w) — the post-attention/post-FFN tail.
// tanh_gate is precomputed in torch (tiny [B, Dm]) so its rounding matches eager.
kernel void norm_gate_residual_bf16(
    device const bfloat* xres [[buffer(0)]],  // [B*S, Dm]
    device const bfloat* y    [[buffer(1)]],  // [B*S, Dm]
    device const bfloat* w    [[buffer(2)]],  // [Dm]
    device const bfloat* tgate[[buffer(3)]],  // [B, Dm] (already tanh'ed)
    device bfloat*       out  [[buffer(4)]],  // [B*S, Dm]
    constant uint&  S    [[buffer(5)]],
    constant uint&  Dm   [[buffer(6)]],
    constant float& eps  [[buffer(7)]],
    uint  tgid      [[threadgroup_position_in_grid]],
    ushort lid      [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint b = tgid / S;
  const device bfloat* row = y + (ulong)tgid * Dm;
  float ss = 0.0f;
  for (uint d = lid; d < Dm; d += 256) {
    float v = float(row[d]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  threadgroup float tg[8];
  if (simd_lane == 0) tg[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (ushort i = 0; i < 8; i++) tot += tg[i];
  const float inv = precise::rsqrt(precise::divide(tot, float(Dm)) + eps);

  const device bfloat* g = tgate + (ulong)b * Dm;
  const device bfloat* xr = xres + (ulong)tgid * Dm;
  device bfloat* o = out + (ulong)tgid * Dm;
  for (uint d = lid; d < Dm; d += 256) {
    // Manual RTE at every rounding point, mirroring eager op-by-op:
    // n = bf16(norm)*w ; p = g*n ; out = x + p.
    bfloat n1 = bf16_rte(float(row[d]) * inv);
    bfloat n = bf16_rte(float(n1) * float(w[d]));
    bfloat p = bf16_rte(float(g[d]) * float(n));
    o[d] = bf16_rte(float(xr[d]) + float(p));
  }
}

// out = silu(a) * b, elementwise (SwiGLU gating). silu in fp32 + RTE like MPS.
kernel void silu_mul_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out     [[buffer(2)]],
    constant uint& N       [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= N) return;
  float av = float(a[gid]);
  float s = av / (1.0f + precise::exp(-av));
  out[gid] = bfloat(s) * b[gid];
}
)MTL";

static id<MTLLibrary> g_lib = nil;
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_psos = nil;

static id<MTLComputePipelineState> pso_named(NSString* name) {
  if (!g_psos) g_psos = [NSMutableDictionary new];
  id<MTLComputePipelineState> cached = g_psos[name];
  if (cached) return cached;
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  if (!g_lib) {
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // Safe math: fast-math FMA-contracts chains like f0*x0 + f1*x1, skipping
    // the per-op RTE rounding torch's eager ops perform. These kernels are
    // memory-bound elementwise/reduction passes — safe math costs nothing.
    if (@available(macOS 15.0, *)) {
      opts.mathMode = MTLMathModeSafe;
    } else {
      opts.fastMathEnabled = NO;
    }
    NSError* err = nil;
    g_lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                              options:opts error:&err];
    TORCH_CHECK(g_lib, "dit_ext Metal compile failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
  }
  id<MTLFunction> fn = [g_lib newFunctionWithName:name];
  TORCH_CHECK(fn, "kernel not found: ", name.UTF8String);
  NSError* err = nil;
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "dit_ext pipeline failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  g_psos[name] = pso;
  return pso;
}

static id<MTLComputePipelineState> pso() { return pso_named(@"qkv_norm_rope_bf16"); }

// qkv [B, S, (Hq+2*Hkv)*D] bf16, freqs [B*S, D/2, 2, 2] fp32 (contiguous),
// qw/kw [D] bf16 -> (q [B,Hq,S,D], k [B,Hkv,S,D], v [B,Hkv,S,D]) bf16.
std::vector<torch::Tensor> qkv_norm_rope(torch::Tensor qkv, torch::Tensor freqs,
                                         torch::Tensor qw, torch::Tensor kw,
                                         int64_t Hq, int64_t Hkv, int64_t D,
                                         double eps_q, double eps_k) {
  TORCH_CHECK(qkv.is_mps() && freqs.is_mps() && qw.is_mps() && kw.is_mps(),
              "inputs must be on mps");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv must be bf16");
  TORCH_CHECK(freqs.scalar_type() == torch::kFloat, "freqs must be fp32");
  TORCH_CHECK(qkv.dim() == 3 && qkv.is_contiguous(), "qkv must be [B,S,QS] contiguous");
  TORCH_CHECK(freqs.is_contiguous(), "freqs must be contiguous");
  TORCH_CHECK(D <= 128 && (D % 32) == 0, "D must be <=128 and a multiple of 32");
  TORCH_CHECK(qkv.size(2) == (Hq + 2 * Hkv) * D, "qkv width mismatch");
  TORCH_CHECK(freqs.numel() == qkv.size(0) * qkv.size(1) * (D / 2) * 4,
              "freqs numel mismatch");

  const int64_t B = qkv.size(0), S = qkv.size(1);
  auto opts = qkv.options();
  auto q = torch::empty({B, Hq, S, D}, opts);
  auto k = torch::empty({B, Hkv, S, D}, opts);
  auto v = torch::empty({B, Hkv, S, D}, opts);

  id<MTLComputePipelineState> p = pso();
  MPSStream* stream = getCurrentMPSStream();
  std::array<id<MTLBuffer>, 7> bufs;
  std::array<NSUInteger, 7> offs;
  torch::Tensor ts[7] = {qkv, freqs, qw, kw, q, k, v};
  for (int i = 0; i < 7; i++) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, ts[i].storage().data());
    offs[i] = ts[i].storage_offset() * ts[i].element_size();
  }
  uint Su = (uint)S, Hqu = (uint)Hq, Hkvu = (uint)Hkv, Du = (uint)D;
  float eq = (float)eps_q, ek = (float)eps_k;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < 7; i++) [enc setBuffer:bufs[i] offset:offs[i] atIndex:i];
      [enc setBytes:&Su   length:sizeof(uint)  atIndex:7];
      [enc setBytes:&Hqu  length:sizeof(uint)  atIndex:8];
      [enc setBytes:&Hkvu length:sizeof(uint)  atIndex:9];
      [enc setBytes:&Du   length:sizeof(uint)  atIndex:10];
      [enc setBytes:&eq   length:sizeof(float) atIndex:11];
      [enc setBytes:&ek   length:sizeof(float) atIndex:12];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), (NSUInteger)(Hq + 2 * Hkv), 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)D, 1, 1)];
    }
  });
  return {q, k, v};
}

// Shared helper: encode an N-buffer + consts kernel on torch's MPS stream.
struct EncArgs {
  std::array<id<MTLBuffer>, 5> bufs;
  std::array<NSUInteger, 5> offs;
  int nbuf;
};

static EncArgs enc_args(std::initializer_list<torch::Tensor> ts) {
  EncArgs a{};
  int i = 0;
  for (const auto& t : ts) {
    a.bufs[i] = __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
    a.offs[i] = t.storage_offset() * t.element_size();
    i++;
  }
  a.nbuf = i;
  return a;
}

// out = (rmsnorm(x)*w) * (1+scale): x [B,S,Dm] bf16, w [Dm], scale [B,Dm].
torch::Tensor norm_modulate(torch::Tensor x, torch::Tensor w, torch::Tensor scale,
                            double eps) {
  TORCH_CHECK(x.is_mps() && x.scalar_type() == torch::kBFloat16 && x.dim() == 3
              && x.is_contiguous(), "x must be contiguous bf16 [B,S,Dm] on mps");
  const int64_t B = x.size(0), S = x.size(1), Dm = x.size(2);
  TORCH_CHECK(w.numel() == Dm && scale.numel() == B * Dm, "w/scale shape mismatch");
  auto out = torch::empty_like(x);
  auto a = enc_args({x, w.contiguous(), scale.contiguous(), out});
  id<MTLComputePipelineState> p = pso_named(@"norm_modulate_bf16");
  MPSStream* stream = getCurrentMPSStream();
  uint Su = (uint)S, Dmu = (uint)Dm; float ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Su  length:sizeof(uint)  atIndex:4];
      [enc setBytes:&Dmu length:sizeof(uint)  atIndex:5];
      [enc setBytes:&ef  length:sizeof(float) atIndex:6];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// out = xres + tgate * (rmsnorm(y)*w): xres/y [B,S,Dm] bf16, tgate [B,Dm] (pre-tanh'ed).
torch::Tensor norm_gate_residual(torch::Tensor xres, torch::Tensor y, torch::Tensor w,
                                 torch::Tensor tgate, double eps) {
  TORCH_CHECK(xres.is_mps() && xres.scalar_type() == torch::kBFloat16 && xres.dim() == 3
              && xres.is_contiguous() && y.is_contiguous(), "inputs must be contiguous bf16");
  const int64_t B = xres.size(0), S = xres.size(1), Dm = xres.size(2);
  TORCH_CHECK(y.sizes() == xres.sizes() && w.numel() == Dm && tgate.numel() == B * Dm,
              "shape mismatch");
  auto out = torch::empty_like(xres);
  auto a = enc_args({xres, y, w.contiguous(), tgate.contiguous(), out});
  id<MTLComputePipelineState> p = pso_named(@"norm_gate_residual_bf16");
  MPSStream* stream = getCurrentMPSStream();
  uint Su = (uint)S, Dmu = (uint)Dm; float ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Su  length:sizeof(uint)  atIndex:5];
      [enc setBytes:&Dmu length:sizeof(uint)  atIndex:6];
      [enc setBytes:&ef  length:sizeof(float) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// out = silu(a) * b elementwise (SwiGLU gating), bf16.
torch::Tensor silu_mul(torch::Tensor a_t, torch::Tensor b_t) {
  TORCH_CHECK(a_t.is_mps() && a_t.scalar_type() == torch::kBFloat16
              && a_t.is_contiguous() && b_t.is_contiguous()
              && a_t.sizes() == b_t.sizes(), "inputs must be matching contiguous bf16");
  auto out = torch::empty_like(a_t);
  const int64_t N = a_t.numel();
  auto a = enc_args({a_t, b_t, out});
  id<MTLComputePipelineState> p = pso_named(@"silu_mul_bf16");
  MPSStream* stream = getCurrentMPSStream();
  uint Nu = (uint)N;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 255) / 256), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// RoPE-only on attention-layout q/k [B,H,S,D] bf16; freqs [B*S, D/2, 2, 2] fp32.
std::vector<torch::Tensor> rope_qk(torch::Tensor q, torch::Tensor k, torch::Tensor freqs) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && freqs.is_mps(), "inputs must be on mps");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16,
              "q/k must be bf16");
  TORCH_CHECK(freqs.scalar_type() == torch::kFloat && freqs.is_contiguous(),
              "freqs must be contiguous fp32");
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes(), "q/k must be matching [B,H,S,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q/k must be contiguous");
  const int64_t B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  TORCH_CHECK(D <= 256 && (D % 32) == 0, "D must be <=256 and a multiple of 32");
  TORCH_CHECK(freqs.numel() == B * S * (D / 2) * 4, "freqs numel mismatch");

  auto qo = torch::empty_like(q);
  auto ko = torch::empty_like(k);
  id<MTLComputePipelineState> p = pso_named(@"rope_qk_bf16");
  MPSStream* stream = getCurrentMPSStream();
  auto a = enc_args({q, k, freqs, qo, ko});
  uint Su = (uint)S, Hu = (uint)H, Du = (uint)D;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Su length:sizeof(uint) atIndex:5];
      [enc setBytes:&Hu length:sizeof(uint) atIndex:6];
      [enc setBytes:&Du length:sizeof(uint) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), (NSUInteger)(2 * H), 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)D, 1, 1)];
    }
  });
  return {qo, ko};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_qk", &rope_qk,
        "RoPE on attention-layout q/k [B,H,S,D] (bf16, MPS; flux apply_rope semantics)");
  m.def("qkv_norm_rope", &qkv_norm_rope,
        "Fused qkv split + QK RMSNorm + RoPE + head transpose (bf16, MPS)");
  m.def("norm_modulate", &norm_modulate,
        "Fused rmsnorm + (1+scale) modulate (bf16, MPS)");
  m.def("norm_gate_residual", &norm_gate_residual,
        "Fused rmsnorm + gate mul + residual add (bf16, MPS)");
  m.def("silu_mul", &silu_mul, "Fused silu(a)*b SwiGLU gating (bf16, MPS)");
  m.def("warmup", []() {
        pso();
        pso_named(@"norm_modulate_bf16");
        pso_named(@"norm_gate_residual_bf16");
        pso_named(@"silu_mul_bf16");
        return true;
      }, "compile pipelines");
}
