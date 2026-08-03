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
#include <array>

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
    device const bfloat* qkv   [[buffer(0)]],  // [B*S, rowStride] (qkv slice at col 0)
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
    constant uint&  rowStride [[buffer(13)]],  // elements per token row (>= (Hq+2Hkv)*D)
    uint2  tgid      [[threadgroup_position_in_grid]],
    ushort lid       [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint token = tgid.x;           // b*S + s
  const uint b = token / S, s = token % S;
  const uint ht = tgid.y;

  const bool is_q = ht < Hq;
  const bool is_v = ht >= Hq + Hkv;
  uint h, off;
  if (is_q)      { h = ht;            off = h * D; }
  else if (!is_v){ h = ht - Hq;       off = (Hq + h) * D; }
  else           { h = ht - Hq - Hkv; off = (Hq + Hkv + h) * D; }
  const device bfloat* row = qkv + (ulong)token * rowStride + off;

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
// Two-source variant for flux DoubleStreamBlock: txt_qkv (S_a tokens) and
// img_qkv (S_b tokens) each get their own QK RMSNorm weights, RoPE over the
// CONCATENATED position table, and are written straight into concatenated
// attention layout [B, H, S_a+S_b, D] (txt first — matching
// q = cat(txt_q, img_q)). Eliminates 4 eager RMSNorms over non-contiguous
// [B,H,S,128] views, 2 view/permutes, the eager RoPE, and 3 big seq-cats.
kernel void qkv2_norm_rope_bf16(
    device const bfloat* qkv_a [[buffer(0)]],  // [B*S_a, 3*H*D] txt
    device const bfloat* qkv_b [[buffer(1)]],  // [B*S_b, 3*H*D] img
    device const float*  freqs [[buffer(2)]],  // [B*(S_a+S_b), D/2, 2, 2] fp32
    device const bfloat* qw_a  [[buffer(3)]],  // [D] txt q_norm weight
    device const bfloat* kw_a  [[buffer(4)]],  // [D] txt k_norm weight
    device const bfloat* qw_b  [[buffer(5)]],  // [D] img q_norm weight
    device const bfloat* kw_b  [[buffer(6)]],  // [D] img k_norm weight
    device bfloat* qo [[buffer(7)]],           // [B, H, S_a+S_b, D]
    device bfloat* ko [[buffer(8)]],
    device bfloat* vo [[buffer(9)]],
    constant uint&  Sa    [[buffer(10)]],
    constant uint&  Sb    [[buffer(11)]],
    constant uint&  H     [[buffer(12)]],
    constant uint&  D     [[buffer(13)]],
    constant float& eps_q [[buffer(14)]],
    constant float& eps_k [[buffer(15)]],
    uint2  tgid      [[threadgroup_position_in_grid]],
    ushort lid       [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint S = Sa + Sb;
  const uint token = tgid.x;           // b*S + s (concatenated index)
  const uint b = token / S, s = token % S;
  const uint ht = tgid.y;
  const uint QS = 3u * H * D;

  const bool is_q = ht < H;
  const bool is_v = ht >= 2u * H;
  uint h, off;
  if (is_q)      { h = ht;          off = h * D; }
  else if (!is_v){ h = ht - H;      off = (H + h) * D; }
  else           { h = ht - 2u * H; off = (2u * H + h) * D; }

  const bool from_a = s < Sa;
  const device bfloat* src = from_a ? qkv_a : qkv_b;
  const uint s_loc = from_a ? s : s - Sa;
  const uint S_loc = from_a ? Sa : Sb;
  const device bfloat* row = src + ((ulong)b * S_loc + s_loc) * QS + off;

  if (is_v) {
    vo[(((ulong)b * H + h) * S + s) * D + lid] = row[lid];
    return;
  }

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
  const device bfloat* w = is_q ? (from_a ? qw_a : qw_b) : (from_a ? kw_a : kw_b);
  bfloat y0 = bf16_rte(x * inv);
  bfloat y = bf16_rte(float(y0) * float(w[lid]));

  // RoPE indexed by the CONCATENATED token position (pe covers txt+img).
  float yf = float(y);
  float partner = simd_shuffle_xor(yf, ushort(1));
  const uint i = lid >> 1;
  const uint j = lid & 1u;
  const device float* f = freqs + ((ulong)token * (D >> 1) + i) * 4u + j * 2u;
  const float x0 = (j == 0u) ? yf : partner;
  const float x1 = (j == 0u) ? partner : yf;
  const float o = f[0] * x0 + f[1] * x1;

  device bfloat* outp = is_q ? qo : ko;
  outp[(((ulong)b * H + h) * S + s) * D + lid] = bf16_rte(o);
}

// Three-source GQA variant for Krea2 Attention: separate wq/wk/wv outputs
// (q: Hq heads, k/v: Hkv heads), Krea2 RMSNorm semantics (fp32 weight already
// including the +1, SINGLE final bf16 rounding — unlike flux's round-then-
// bf16-weight-multiply), RoPE on q/k, v transposed copy. Outputs stay at
// their native head counts so flash41's GQA path runs without the eager
// repeat_interleave (which materializes 4x KV and quadruples KV reads).
// ── qkv_norm_rope_sh: fused qkv-split + per-head affine RMSNorm + SPLIT-HALF
// RoPE (llama/Qwen apply_rope: out[:h] = x*cos + x[h:]*nsin; out[h:] = x*cos +
// x[:h]*sin), for Ideogram 4 (D=256, MRoPE folded into cos/sin by the host).
// One threadgroup per (token, slot) with D/2 threads; each thread owns the
// element pair (d, d+D/2) so the rotation needs no cross-lane traffic.
// Rounding matches eager: bf16 after the norm, fp32 rope, final bf16 round.
kernel void qkv_norm_rope_sh_bf16(
    device const bfloat* qkv  [[buffer(0)]],  // [B*S, 3*H*D] ([3,H,D] inner)
    device const float*  cosb [[buffer(1)]],  // [S, D]
    device const float*  sinb [[buffer(2)]],  // [S, D/2] (first half)
    device const float*  nsinb[[buffer(3)]],  // [S, D/2] (-sin second half)
    device const float*  qw   [[buffer(4)]],  // [D] fp32
    device const float*  kw   [[buffer(5)]],  // [D] fp32
    device bfloat* qo [[buffer(6)]],          // [B, H, S, D]
    device bfloat* ko [[buffer(7)]],
    device bfloat* vo [[buffer(8)]],
    constant uint&  S   [[buffer(9)]],
    constant uint&  H   [[buffer(10)]],
    constant uint&  D   [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    uint2  tgid      [[threadgroup_position_in_grid]],
    ushort lid       [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint token = tgid.x;           // b*S + s
  const uint b = token / S, s = token % S;
  const uint slot = tgid.y;            // [0, 3H): q rows, then k rows, then v
  const uint kind = slot / H;          // 0=q 1=k 2=v
  const uint h = slot % H;
  const uint Dh = D >> 1;
  const uint d0 = lid, d1 = lid + Dh;

  const device bfloat* row = qkv + (ulong)token * (3u * H * D)
                                 + ((ulong)kind * H + h) * D;
  device bfloat* outp = (kind == 0u) ? qo : (kind == 1u ? ko : vo);
  device bfloat* orow = outp + (((ulong)b * H + h) * S + s) * D;

  if (kind == 2u) {                    // v: pure relayout
    orow[d0] = row[d0];
    orow[d1] = row[d1];
    return;
  }

  float x0 = float(row[d0]);
  float x1 = float(row[d1]);
  float ss = simd_sum(x0 * x0 + x1 * x1);
  threadgroup float tg[8];
  if (simd_lane == 0) tg[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  const ushort nsg = ushort(Dh >> 5);
  for (ushort i = 0; i < nsg; i++) tot += tg[i];

  const float inv = precise::rsqrt(precise::divide(tot, float(D)) + eps);
  const device float* w = (kind == 0u) ? qw : kw;
  const float y0 = float(bf16_rte(x0 * inv * w[d0]));
  const float y1 = float(bf16_rte(x1 * inv * w[d1]));

  const device float* c  = cosb  + (ulong)s * D;
  const device float* sn = sinb  + (ulong)s * Dh;
  const device float* ns = nsinb + (ulong)s * Dh;
  orow[d0] = bf16_rte(y0 * c[d0] + y1 * ns[d0]);
  orow[d1] = bf16_rte(y1 * c[d1] + y0 * sn[d0]);
}

kernel void qkv3_norm_rope_bf16(
    device const bfloat* qin   [[buffer(0)]],  // [B*S, Hq*D]
    device const bfloat* kin   [[buffer(1)]],  // [B*S, Hkv*D]
    device const bfloat* vin   [[buffer(2)]],  // [B*S, Hkv*D]
    device const float*  freqs [[buffer(3)]],  // [B*S, D/2, 2, 2] fp32
    device const float*  qw    [[buffer(4)]],  // [D] fp32, = qnorm.scale + 1
    device const float*  kw    [[buffer(5)]],  // [D] fp32, = knorm.scale + 1
    device bfloat* qo [[buffer(6)]],           // [B, Hq,  S, D]
    device bfloat* ko [[buffer(7)]],           // [B, Hkv, S, D]
    device bfloat* vo [[buffer(8)]],           // [B, Hkv, S, D]
    constant uint&  S     [[buffer(9)]],
    constant uint&  Hq    [[buffer(10)]],
    constant uint&  Hkv   [[buffer(11)]],
    constant uint&  D     [[buffer(12)]],
    constant float& eps   [[buffer(13)]],
    uint2  tgid      [[threadgroup_position_in_grid]],
    ushort lid       [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint token = tgid.x;           // b*S + s
  const uint b = token / S, s = token % S;
  const uint ht = tgid.y;

  const bool is_q = ht < Hq;
  const bool is_v = ht >= Hq + Hkv;
  uint h;
  const device bfloat* src;
  uint W;
  if (is_q)      { h = ht;            src = qin; W = Hq * D; }
  else if (!is_v){ h = ht - Hq;       src = kin; W = Hkv * D; }
  else           { h = ht - Hq - Hkv; src = vin; W = Hkv * D; }
  const device bfloat* row = src + (ulong)token * W + h * D;

  if (is_v) {
    vo[(((ulong)b * Hkv + h) * S + s) * D + lid] = row[lid];
    return;
  }

  float x = float(row[lid]);
  float ss = simd_sum(x * x);
  threadgroup float tg[4];
  if (simd_lane == 0) tg[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  const ushort nsg = ushort(D >> 5);
  for (ushort i = 0; i < nsg; i++) tot += tg[i];

  // Krea2 RMSNorm: fp32 normalize * fp32 weight, one rounding at the end.
  const float inv = precise::rsqrt(precise::divide(tot, float(D)) + eps);
  const device float* w = is_q ? qw : kw;
  bfloat y = bf16_rte(x * inv * w[lid]);

  // RoPE (flux apply_rope): fp32 pair rotation on the bf16-rounded values.
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

// LTX split-RoPE: x [B,T,H*D] with per-head cos/sin [B,H,T,D/2] accessed via
// explicit strides (the eager path materializes TWO transposed copies of x per
// call plus ~7 elementwise passes; this is one pass, zero copies).
// Half-pairs (i, i+D/2): out1 = bf16(bf16(x1*c) - s*x2); out2 = bf16(bf16(x2*c) + s*x1)
// — mirroring eager's mul-store then addcmul-store rounding chain.
kernel void rope_split_bf16(
    device const bfloat* x    [[buffer(0)]],  // [B,T,H*D] contiguous
    device const bfloat* cosf [[buffer(1)]],  // [B,H,T,r] via strides
    device const bfloat* sinf [[buffer(2)]],
    device bfloat*       out  [[buffer(3)]],  // [B,T,H*D]
    constant uint&  T   [[buffer(4)]],
    constant uint&  H   [[buffer(5)]],
    constant uint&  R   [[buffer(6)]],        // D/2
    constant uint4& cst [[buffer(7)]],        // cos strides (b,h,t,i) in elements
    constant uint&  N   [[buffer(8)]],        // B*T*H*R pairs
    uint gid [[thread_position_in_grid]])
{
  if (gid >= N) return;
  const uint i = gid % R;
  const uint h = (gid / R) % H;
  const uint t = (gid / (R * H)) % T;
  const uint b = gid / (R * H * T);
  const uint D = 2u * R;

  const ulong xrow = ((ulong)b * T + t) * (ulong)(H * D) + h * D;
  const float x1 = float(x[xrow + i]);
  const float x2 = float(x[xrow + R + i]);
  const ulong f = (ulong)b * cst.x + (ulong)h * cst.y + (ulong)t * cst.z + (ulong)i * cst.w;
  const float c = float(cosf[f]);
  const float s = float(sinf[f]);

  const bfloat o1m = bf16_rte(x1 * c);
  const bfloat o2m = bf16_rte(x2 * c);
  out[xrow + i]     = bf16_rte(float(o1m) - s * x2);
  out[xrow + R + i] = bf16_rte(float(o2m) + s * x1);
}

// LTX fused no-weight RMSNorm + AdaLN modulate:
// out = bf16(bf16(x_f32*inv) * bf16(1+scale) + shift), scale/shift [B,Sm,C]
// with Sm in {1, S} (per-batch or per-token timesteps). One 256-thread
// threadgroup per token row.
kernel void rms_mod_bf16(
    device const bfloat* x     [[buffer(0)]],  // [B*S, C]
    device const bfloat* scale [[buffer(1)]],  // [B, Sm, C]
    device const bfloat* shift [[buffer(2)]],  // [B, Sm, C]
    device bfloat*       out   [[buffer(3)]],  // [B*S, C]
    constant uint&  S    [[buffer(4)]],
    constant uint&  Sm   [[buffer(5)]],
    constant uint&  C    [[buffer(6)]],
    constant float& eps  [[buffer(7)]],
    constant uint&  PPF  [[buffer(8)]],   // tokens per modulation row (frame)
    uint  tgid      [[threadgroup_position_in_grid]],
    ushort lid      [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  const uint b = tgid / S;
  const uint m = (Sm == 1u) ? 0u : ((tgid % S) / PPF);
  const device bfloat* row = x + (ulong)tgid * C;
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
  const float inv = precise::rsqrt(precise::divide(tot, float(C)) + eps);

  const device bfloat* sc = scale + ((ulong)b * Sm + m) * C;
  const device bfloat* sh = shift + ((ulong)b * Sm + m) * C;
  device bfloat* o = out + (ulong)tgid * C;
  for (uint d = lid; d < C; d += 256) {
    bfloat y = bf16_rte(float(row[d]) * inv);            // F.rms_norm output
    bfloat mscale = bf16_rte(1.0f + float(sc[d]));       // (1 + scale)
    bfloat p = bf16_rte(float(y) * float(mscale));
    o[d] = bf16_rte(float(p) + float(sh[d]));
  }
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

// ── rms_mod_seg: affine RMSNorm + per-SEGMENT modulate in one pass.
// MiniMax H3 packs [text | video | audio] into one sequence and modulates each
// segment with its own adaln row, so eager runs RMSNorm (2 passes) then, per
// segment, mul_(1+scale) and add_(shift) (4 more) over a 57 MB tensor — twice
// per block, 50 blocks. Here: one pass, with the token's mod row looked up
// from a precomputed [S] int32 map. Rounding mirrors eager step for step:
// bf16(x*inv) -> *w -> *(1+scale) -> +shift, each rounded.
kernel void rms_mod_seg_bf16(
    device const bfloat* x     [[buffer(0)]],  // [S, Dm]
    device const bfloat* w     [[buffer(1)]],  // [Dm]
    device const bfloat* scale [[buffer(2)]],  // [M, Dm]
    device const bfloat* shift [[buffer(3)]],  // [M, Dm]
    device const int*    rows  [[buffer(4)]],  // [S] token -> mod row
    device bfloat*       out   [[buffer(5)]],  // [S, Dm]
    constant uint&  Dm   [[buffer(6)]],
    constant float& eps  [[buffer(7)]],
    uint  tgid       [[threadgroup_position_in_grid]],
    ushort lid       [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  threadgroup float tg[8];
  const device bfloat* row = x + (ulong)tgid * Dm;
  float ss = 0.0f;
  for (uint d = lid; d < Dm; d += 256) {
    const float v = float(row[d]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (simd_lane == 0) tg[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (ushort i = 0; i < 8; i++) tot += tg[i];
  const float inv = precise::rsqrt(precise::divide(tot, float(Dm)) + eps);

  const ulong mrow = (ulong)rows[tgid] * Dm;
  const device bfloat* sc = scale + mrow;
  const device bfloat* sh = shift + mrow;
  device bfloat* o = out + (ulong)tgid * Dm;
  for (uint d = lid; d < Dm; d += 256) {
    const bfloat yn = bf16_rte(float(row[d]) * inv);
    const bfloat y  = bf16_rte(float(yn) * float(w[d]));
    const bfloat m  = bf16_rte(1.0f + float(sc[d]));
    o[d] = bf16_rte(bf16_rte(float(y) * float(m)) + float(sh[d]));
  }
}

// ── silu_mul_split: SwiGLU gating straight off the fused [M, 2F] projection
// output, so the two halves never need a .contiguous() copy (2 x 154 MB per
// call at H3's fc2). silu in fp32 + RTE, matching the eager path.
kernel void silu_mul_split_bf16(
    device const bfloat* x   [[buffer(0)]],   // [M, 2F]
    device bfloat*       out [[buffer(1)]],   // [M, F]
    constant uint& F         [[buffer(2)]],
    constant uint& N         [[buffer(3)]],   // M*F
    uint gid [[thread_position_in_grid]])
{
  if (gid >= N) return;
  const uint m = gid / F, f = gid % F;
  const ulong base = (ulong)m * (2u * F);
  const float av = float(x[base + f]);
  const float s = av / (1.0f + precise::exp(-av));
  out[gid] = bfloat(s) * x[base + F + f];
}

// ── rms_rope_sh_partial: per-head affine RMSNorm over the FULL head dim +
// PARTIAL split-half RoPE over the leading rot dims, applied IN PLACE on
// strided q/k views of a fused qkv buffer (MiniMax H3: D=128, rot=96, rows
// strided by 3*H*D). Replaces comfy_kitchen's eager rms_rope_split_half_,
// which walks ~10 elementwise passes over two 77 MB tensors per block.
//
// Rounding mirrors the eager path element for element: fp32 norm accumulation
// with ONE bf16 round, then the 2x2 rotation evaluated in bf16 STEPS — each
// product rounded, then their sum rounded — because the rotation table is
// stored at model dtype (bf16), so torch rounds after every op.
// Pairs are (i, i+R), NOT adjacent lanes: eager reshapes [.., 2, R] and moves
// the pair axis last, so lane i partners with lane i+R.
kernel void rms_rope_sh_partial_bf16(
    device bfloat* q        [[buffer(0)]],   // [S,H,D] view, row stride rstride
    device bfloat* k        [[buffer(1)]],
    device const bfloat* fr [[buffer(2)]],   // [S,R,2,2] rotation matrices
    device const float* qw  [[buffer(3)]],   // [D] fp32
    device const float* kw  [[buffer(4)]],
    constant uint&  H       [[buffer(5)]],
    constant uint&  D       [[buffer(6)]],
    constant uint&  R       [[buffer(7)]],   // rot_dim / 2
    constant uint&  rstride [[buffer(8)]],   // elements between token rows
    constant float& eps     [[buffer(9)]],
    uint2  tgid      [[threadgroup_position_in_grid]],
    ushort lid       [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]])
{
  threadgroup float tg[8];
  threadgroup float ybuf[256];

  const uint s = tgid.x;
  const uint slot = tgid.y;              // [0, 2H): q rows, then k rows
  const uint kind = slot / H;            // 0 = q, 1 = k
  const uint h = slot % H;
  const uint d = lid;

  device bfloat* base = (kind == 0u ? q : k) + (ulong)s * rstride + (ulong)h * D;
  const device float* w = (kind == 0u) ? qw : kw;

  const float x = float(base[d]);
  const float ss = simd_sum(x * x);
  if (simd_lane == 0) tg[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  const ushort nsg = ushort(D >> 5);
  for (ushort i = 0; i < nsg; i++) tot += tg[i];

  const float inv = precise::rsqrt(precise::divide(tot, float(D)) + eps);
  // F.rms_norm(x, weight=w) is NOT one fused fp32 expression: it rounds the
  // normalized value to bf16 and THEN multiplies by the bf16 weight (verified
  // — eager equals rms_norm(x) * w exactly), so round twice, in that order.
  const bfloat yn = bf16_rte(x * inv);
  const bfloat y  = bf16_rte(float(yn) * w[d]);
  ybuf[d] = float(y);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (d >= (R * 2u)) {                   // tail: normed, unrotated pass-through
    base[d] = y;
    return;
  }
  const uint i = (d < R) ? d : (d - R);
  const uint r = (d < R) ? 0u : 1u;      // row of the 2x2 rotation matrix
  const device bfloat* m = fr + (((ulong)s * R + i) * 2u + r) * 2u;
  const bfloat p0 = bf16_rte(float(m[0]) * ybuf[i]);
  const bfloat p1 = bf16_rte(float(m[1]) * ybuf[i + R]);
  base[d] = bf16_rte(float(p0) + float(p1));
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

// qkv [B, S, W] bf16 with the q|k|v slice starting at column 0 and W >=
// (Hq+2*Hkv)*D (W > that means extra columns — e.g. flux SingleStreamBlock's
// fused qkv+mlp linear1 output feeds in directly, no split copy), freqs
// [B*S, D/2, 2, 2] fp32 (contiguous), qw/kw [D] bf16
// -> (q [B,Hq,S,D], k [B,Hkv,S,D], v [B,Hkv,S,D]) bf16.
std::vector<torch::Tensor> qkv_norm_rope(torch::Tensor qkv, torch::Tensor freqs,
                                         torch::Tensor qw, torch::Tensor kw,
                                         int64_t Hq, int64_t Hkv, int64_t D,
                                         double eps_q, double eps_k) {
  TORCH_CHECK(qkv.is_mps() && freqs.is_mps() && qw.is_mps() && kw.is_mps(),
              "inputs must be on mps");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv must be bf16");
  TORCH_CHECK(freqs.scalar_type() == torch::kFloat, "freqs must be fp32");
  TORCH_CHECK(qkv.dim() == 3 && qkv.is_contiguous(), "qkv must be [B,S,W] contiguous");
  TORCH_CHECK(freqs.is_contiguous(), "freqs must be contiguous");
  TORCH_CHECK(D <= 128 && (D % 32) == 0, "D must be <=128 and a multiple of 32");
  TORCH_CHECK(qkv.size(2) >= (Hq + 2 * Hkv) * D, "qkv width too small");
  TORCH_CHECK(freqs.numel() == qkv.size(0) * qkv.size(1) * (D / 2) * 4,
              "freqs numel mismatch");

  const int64_t B = qkv.size(0), S = qkv.size(1);
  const uint rowStride = (uint)qkv.size(2);
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
      [enc setBytes:&rowStride length:sizeof(uint) atIndex:13];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), (NSUInteger)(Hq + 2 * Hkv), 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)D, 1, 1)];
    }
  });
  return {q, k, v};
}

// Two-source flux DoubleStreamBlock variant: txt_qkv [B,S_a,3HD] + img_qkv
// [B,S_b,3HD], per-source QK norm weights, RoPE over the concatenated position
// table -> (q, k, v) each [B, H, S_a+S_b, D] with txt tokens first.
std::vector<torch::Tensor> qkv2_norm_rope(torch::Tensor qkv_a, torch::Tensor qkv_b,
                                          torch::Tensor freqs,
                                          torch::Tensor qw_a, torch::Tensor kw_a,
                                          torch::Tensor qw_b, torch::Tensor kw_b,
                                          int64_t H, int64_t D,
                                          double eps_q, double eps_k) {
  TORCH_CHECK(qkv_a.is_mps() && qkv_b.is_mps() && freqs.is_mps(), "inputs must be on mps");
  TORCH_CHECK(qkv_a.scalar_type() == torch::kBFloat16
              && qkv_b.scalar_type() == torch::kBFloat16, "qkv must be bf16");
  TORCH_CHECK(freqs.scalar_type() == torch::kFloat && freqs.is_contiguous(),
              "freqs must be contiguous fp32");
  TORCH_CHECK(qkv_a.dim() == 3 && qkv_a.is_contiguous()
              && qkv_b.dim() == 3 && qkv_b.is_contiguous(),
              "qkv tensors must be [B,S,3HD] contiguous");
  TORCH_CHECK(D <= 128 && (D % 32) == 0, "D must be <=128 and a multiple of 32");
  TORCH_CHECK(qkv_a.size(2) == 3 * H * D && qkv_b.size(2) == 3 * H * D,
              "qkv width mismatch");
  TORCH_CHECK(qkv_a.size(0) == qkv_b.size(0), "batch mismatch");
  const int64_t B = qkv_a.size(0), Sa = qkv_a.size(1), Sb = qkv_b.size(1);
  const int64_t S = Sa + Sb;
  TORCH_CHECK(freqs.numel() == B * S * (D / 2) * 4, "freqs numel mismatch");

  auto opts = qkv_a.options();
  auto q = torch::empty({B, H, S, D}, opts);
  auto k = torch::empty({B, H, S, D}, opts);
  auto v = torch::empty({B, H, S, D}, opts);

  id<MTLComputePipelineState> p = pso_named(@"qkv2_norm_rope_bf16");
  MPSStream* stream = getCurrentMPSStream();
  std::array<id<MTLBuffer>, 10> bufs;
  std::array<NSUInteger, 10> offs;
  torch::Tensor ts[10] = {qkv_a, qkv_b, freqs, qw_a, kw_a, qw_b, kw_b, q, k, v};
  for (int i = 0; i < 10; i++) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, ts[i].storage().data());
    offs[i] = ts[i].storage_offset() * ts[i].element_size();
  }
  uint Sau = (uint)Sa, Sbu = (uint)Sb, Hu = (uint)H, Du = (uint)D;
  float eq = (float)eps_q, ek = (float)eps_k;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < 10; i++) [enc setBuffer:bufs[i] offset:offs[i] atIndex:i];
      [enc setBytes:&Sau length:sizeof(uint)  atIndex:10];
      [enc setBytes:&Sbu length:sizeof(uint)  atIndex:11];
      [enc setBytes:&Hu  length:sizeof(uint)  atIndex:12];
      [enc setBytes:&Du  length:sizeof(uint)  atIndex:13];
      [enc setBytes:&eq  length:sizeof(float) atIndex:14];
      [enc setBytes:&ek  length:sizeof(float) atIndex:15];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), (NSUInteger)(3 * H), 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)D, 1, 1)];
    }
  });
  return {q, k, v};
}

// Krea2 GQA variant: separate q [B,S,Hq*D] / k,v [B,S,Hkv*D] linear outputs,
// fp32 (scale+1) norm weights -> (q [B,Hq,S,D], k [B,Hkv,S,D], v [B,Hkv,S,D]).
std::vector<torch::Tensor> qkv3_norm_rope(torch::Tensor qin, torch::Tensor kin,
                                          torch::Tensor vin, torch::Tensor freqs,
                                          torch::Tensor qw, torch::Tensor kw,
                                          int64_t Hq, int64_t Hkv, int64_t D,
                                          double eps) {
  TORCH_CHECK(qin.is_mps() && kin.is_mps() && vin.is_mps() && freqs.is_mps(),
              "inputs must be on mps");
  TORCH_CHECK(qin.scalar_type() == torch::kBFloat16
              && kin.scalar_type() == torch::kBFloat16
              && vin.scalar_type() == torch::kBFloat16, "q/k/v must be bf16");
  TORCH_CHECK(freqs.scalar_type() == torch::kFloat && freqs.is_contiguous(),
              "freqs must be contiguous fp32");
  TORCH_CHECK(qw.scalar_type() == torch::kFloat && kw.scalar_type() == torch::kFloat
              && qw.numel() == D && kw.numel() == D, "weights must be fp32 [D]");
  TORCH_CHECK(qin.dim() == 3 && qin.is_contiguous() && kin.is_contiguous()
              && vin.is_contiguous(), "q/k/v must be [B,S,H*D] contiguous");
  TORCH_CHECK(D <= 128 && (D % 32) == 0, "D must be <=128 and a multiple of 32");
  TORCH_CHECK(qin.size(2) == Hq * D && kin.size(2) == Hkv * D
              && vin.size(2) == Hkv * D, "width mismatch");
  const int64_t B = qin.size(0), S = qin.size(1);
  TORCH_CHECK(kin.size(0) == B && kin.size(1) == S && vin.sizes() == kin.sizes(),
              "batch/seq mismatch");
  TORCH_CHECK(freqs.numel() == B * S * (D / 2) * 4, "freqs numel mismatch");

  auto opts = qin.options();
  auto q = torch::empty({B, Hq, S, D}, opts);
  auto k = torch::empty({B, Hkv, S, D}, opts);
  auto v = torch::empty({B, Hkv, S, D}, opts);

  id<MTLComputePipelineState> p = pso_named(@"qkv3_norm_rope_bf16");
  MPSStream* stream = getCurrentMPSStream();
  std::array<id<MTLBuffer>, 9> bufs;
  std::array<NSUInteger, 9> offs;
  torch::Tensor ts[9] = {qin, kin, vin, freqs, qw, kw, q, k, v};
  for (int i = 0; i < 9; i++) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, ts[i].storage().data());
    offs[i] = ts[i].storage_offset() * ts[i].element_size();
  }
  uint Su = (uint)S, Hqu = (uint)Hq, Hkvu = (uint)Hkv, Du = (uint)D;
  float ef = (float)eps;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < 9; i++) [enc setBuffer:bufs[i] offset:offs[i] atIndex:i];
      [enc setBytes:&Su   length:sizeof(uint)  atIndex:9];
      [enc setBytes:&Hqu  length:sizeof(uint)  atIndex:10];
      [enc setBytes:&Hkvu length:sizeof(uint)  atIndex:11];
      [enc setBytes:&Du   length:sizeof(uint)  atIndex:12];
      [enc setBytes:&ef   length:sizeof(float) atIndex:13];
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

// LTX split-RoPE on [B,T,H*D] with strided per-head cos/sin [B,H,T,D/2].
torch::Tensor rope_split(torch::Tensor x, torch::Tensor cosf, torch::Tensor sinf,
                         int64_t H) {
  TORCH_CHECK(x.is_mps() && cosf.is_mps() && sinf.is_mps(), "inputs must be on mps");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16
              && cosf.scalar_type() == torch::kBFloat16
              && sinf.scalar_type() == torch::kBFloat16, "inputs must be bf16");
  TORCH_CHECK(x.dim() == 3 && x.is_contiguous(), "x must be [B,T,H*D] contiguous");
  TORCH_CHECK(cosf.dim() == 4 && sinf.sizes() == cosf.sizes()
              && sinf.strides() == cosf.strides(), "cos/sin must match [B,H,T,r]");
  const int64_t B = x.size(0), T = x.size(1), W = x.size(2);
  const int64_t R = cosf.size(3);
  TORCH_CHECK(W == H * 2 * R, "width mismatch: HD != H*2*r");
  TORCH_CHECK(cosf.size(1) == H && cosf.size(2) == T
              && (cosf.size(0) == B || cosf.size(0) == 1), "cos shape mismatch");

  auto out = torch::empty_like(x);
  id<MTLComputePipelineState> p = pso_named(@"rope_split_bf16");
  MPSStream* stream = getCurrentMPSStream();
  auto a = enc_args({x, cosf, sinf, out});
  uint Tu = (uint)T, Hu = (uint)H, Ru = (uint)R;
  const int64_t bs = (cosf.size(0) == 1) ? 0 : cosf.stride(0);  // broadcast batch
  // std::array, not a C array: ObjC blocks cannot capture C arrays.
  std::array<uint, 4> cst = {(uint)bs, (uint)cosf.stride(1), (uint)cosf.stride(2), (uint)cosf.stride(3)};
  const int64_t N = B * T * H * R;
  uint Nu = (uint)N;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Tu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Hu length:sizeof(uint) atIndex:5];
      [enc setBytes:&Ru length:sizeof(uint) atIndex:6];
      [enc setBytes:cst.data() length:sizeof(cst) atIndex:7];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((N + 255) / 256), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// LTX fused no-weight rmsnorm + (1+scale)+shift: x [B,S,C], scale/shift
// [B,Sm,C] with Sm*ppf == S (per-frame CompressedTimestep rows) or Sm == 1.
torch::Tensor rms_mod(torch::Tensor x, torch::Tensor scale, torch::Tensor shift,
                      double eps, int64_t ppf) {
  TORCH_CHECK(x.is_mps() && x.scalar_type() == torch::kBFloat16 && x.dim() == 3
              && x.is_contiguous(), "x must be contiguous bf16 [B,S,C] on mps");
  const int64_t B = x.size(0), S = x.size(1), C = x.size(2);
  if (ppf <= 0) ppf = 1;
  TORCH_CHECK(scale.dim() == 3 && scale.sizes() == shift.sizes()
              && scale.size(0) == B && scale.size(2) == C
              && (scale.size(1) == 1 || scale.size(1) * ppf == S),
              "scale/shift must be [B,1,C] or [B,S/ppf,C]");
  TORCH_CHECK(scale.scalar_type() == torch::kBFloat16
              && shift.scalar_type() == torch::kBFloat16, "scale/shift must be bf16");
  auto sc = scale.contiguous();
  auto sh = shift.contiguous();
  auto out = torch::empty_like(x);
  auto a = enc_args({x, sc, sh, out});
  id<MTLComputePipelineState> p = pso_named(@"rms_mod_bf16");
  MPSStream* stream = getCurrentMPSStream();
  uint Su = (uint)S, Smu = (uint)sc.size(1), Cu = (uint)C, Pu = (uint)ppf;
  float ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Su  length:sizeof(uint)  atIndex:4];
      [enc setBytes:&Smu length:sizeof(uint)  atIndex:5];
      [enc setBytes:&Cu  length:sizeof(uint)  atIndex:6];
      [enc setBytes:&ef  length:sizeof(float) atIndex:7];
      [enc setBytes:&Pu  length:sizeof(uint)  atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
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

// Ideogram 4: fused qkv [B,S,3HD] bf16 + cos [S,D] / sin,nsin [S,D/2] fp32
// (llama precompute_freqs_cis triple, leading batch dim squeezed) + fp32
// affine norm weights -> (q, k, v) [B,H,S,D] bf16, split-half RoPE applied.
std::vector<torch::Tensor> qkv_norm_rope_sh(torch::Tensor qkv, torch::Tensor cosb,
                                            torch::Tensor sinb, torch::Tensor nsinb,
                                            torch::Tensor qw, torch::Tensor kw,
                                            int64_t H, int64_t D, double eps) {
  TORCH_CHECK(qkv.is_mps() && qkv.scalar_type() == torch::kBFloat16
              && qkv.dim() == 3 && qkv.is_contiguous(),
              "qkv must be contiguous bf16 [B,S,3HD] on mps");
  TORCH_CHECK(cosb.scalar_type() == torch::kFloat && cosb.is_contiguous()
              && sinb.is_contiguous() && nsinb.is_contiguous(),
              "cos/sin/nsin must be contiguous fp32");
  TORCH_CHECK(D % 64 == 0 && D <= 512, "D must be a multiple of 64, <= 512");
  const int64_t B = qkv.size(0), S = qkv.size(1);
  TORCH_CHECK(qkv.size(2) == 3 * H * D, "qkv width mismatch");
  TORCH_CHECK(cosb.numel() == S * D && sinb.numel() == S * (D / 2)
              && nsinb.numel() == S * (D / 2), "cos/sin shape mismatch");
  TORCH_CHECK(qw.numel() == D && kw.numel() == D
              && qw.scalar_type() == torch::kFloat
              && kw.scalar_type() == torch::kFloat, "qw/kw must be fp32 [D]");

  auto opts = qkv.options();
  auto q = torch::empty({B, H, S, D}, opts);
  auto k = torch::empty({B, H, S, D}, opts);
  auto v = torch::empty({B, H, S, D}, opts);

  id<MTLComputePipelineState> p = pso_named(@"qkv_norm_rope_sh_bf16");
  MPSStream* stream = getCurrentMPSStream();
  std::array<id<MTLBuffer>, 9> bufs;
  std::array<NSUInteger, 9> offs;
  torch::Tensor ts[9] = {qkv, cosb, sinb, nsinb, qw, kw, q, k, v};
  for (int i = 0; i < 9; i++) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, ts[i].storage().data());
    offs[i] = ts[i].storage_offset() * ts[i].element_size();
  }
  uint Su = (uint)S, Hu = (uint)H, Du = (uint)D;
  float ef = (float)eps;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < 9; i++) [enc setBuffer:bufs[i] offset:offs[i] atIndex:i];
      [enc setBytes:&Su length:sizeof(uint)  atIndex:9];
      [enc setBytes:&Hu length:sizeof(uint)  atIndex:10];
      [enc setBytes:&Du length:sizeof(uint)  atIndex:11];
      [enc setBytes:&ef length:sizeof(float) atIndex:12];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(B * S), (NSUInteger)(3 * H), 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)(D / 2), 1, 1)];
    }
  });
  return {q, k, v};
}

// MiniMax H3: affine RMSNorm + per-segment modulate, one pass. x [S,Dm] bf16
// contiguous; scale/shift [M,Dm] bf16; rows [S] int32 mapping token -> mod row.
torch::Tensor rms_mod_seg(torch::Tensor x, torch::Tensor w, torch::Tensor scale,
                          torch::Tensor shift, torch::Tensor rows, double eps) {
  TORCH_CHECK(x.is_mps() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2
              && x.is_contiguous(), "x must be contiguous bf16 [S,Dm] on mps");
  const int64_t S = x.size(0), Dm = x.size(1);
  TORCH_CHECK(w.numel() == Dm && w.scalar_type() == torch::kBFloat16 && w.is_contiguous(),
              "w must be contiguous bf16 [Dm]");
  TORCH_CHECK(scale.dim() == 2 && scale.size(1) == Dm && shift.sizes() == scale.sizes()
              && scale.scalar_type() == torch::kBFloat16
              && shift.scalar_type() == torch::kBFloat16
              && scale.is_contiguous() && shift.is_contiguous(),
              "scale/shift must be contiguous bf16 [M,Dm]");
  TORCH_CHECK(rows.numel() == S && rows.scalar_type() == torch::kInt
              && rows.is_contiguous(), "rows must be contiguous int32 [S]");
  TORCH_CHECK(Dm % 32 == 0, "Dm must be a multiple of 32");

  auto out = torch::empty_like(x);
  id<MTLComputePipelineState> p = pso_named(@"rms_mod_seg_bf16");
  MPSStream* stream = getCurrentMPSStream();
  std::array<id<MTLBuffer>, 6> bufs;
  std::array<NSUInteger, 6> offs;
  torch::Tensor ts[6] = {x, w, scale, shift, rows, out};
  for (int i = 0; i < 6; i++) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, ts[i].storage().data());
    offs[i] = ts[i].storage_offset() * ts[i].element_size();
  }
  uint Dmu = (uint)Dm;
  float ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < 6; i++) [enc setBuffer:bufs[i] offset:offs[i] atIndex:i];
      [enc setBytes:&Dmu length:sizeof(uint)  atIndex:6];
      [enc setBytes:&ef  length:sizeof(float) atIndex:7];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)S, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// SwiGLU gating read straight from a fused [M, 2F] projection (no half copies).
torch::Tensor silu_mul_split(torch::Tensor x) {
  TORCH_CHECK(x.is_mps() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2
              && x.is_contiguous(), "x must be contiguous bf16 [M,2F] on mps");
  const int64_t M = x.size(0), W = x.size(1);
  TORCH_CHECK(W % 2 == 0, "last dim must be even");
  const int64_t F = W / 2;
  auto out = torch::empty({M, F}, x.options());
  id<MTLComputePipelineState> p = pso_named(@"silu_mul_split_bf16");
  MPSStream* stream = getCurrentMPSStream();
  auto a = enc_args({x, out});
  uint Fu = (uint)F, N = (uint)(M * F);
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Fu length:sizeof(uint) atIndex:2];
      [enc setBytes:&N  length:sizeof(uint) atIndex:3];
      [enc dispatchThreads:MTLSizeMake((NSUInteger)N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return out;
}

// MiniMax H3: in-place per-head RMSNorm + partial split-half RoPE on strided
// q/k views of the fused qkv buffer. q/k: [1,S,H,D] bf16 whose inner [H,D] is
// contiguous but whose token rows are strided (the qkv split); fr: the
// [1,S,1,R,2,2] bf16 rotation table; qw/kw: fp32 [D] norm weights.
void rms_rope_sh_partial(torch::Tensor q, torch::Tensor k, torch::Tensor fr,
                         torch::Tensor qw, torch::Tensor kw, double eps) {
  TORCH_CHECK(q.is_mps() && k.is_mps() && fr.is_mps(), "inputs must be on mps");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16
              && fr.scalar_type() == torch::kBFloat16, "q/k/freqs must be bf16");
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.size(0) == 1,
              "q/k must be matching [1,S,H,D]");
  const int64_t S = q.size(1), H = q.size(2), D = q.size(3);
  TORCH_CHECK(D <= 256 && (D % 32) == 0, "D must be <= 256 and a multiple of 32");
  // Inner [H,D] contiguous; token rows may be strided (views into qkv).
  TORCH_CHECK(q.stride(3) == 1 && q.stride(2) == D && k.stride(3) == 1
              && k.stride(2) == D && q.stride(1) == k.stride(1),
              "q/k must have contiguous [H,D] rows and a common row stride");
  TORCH_CHECK(fr.is_contiguous(), "freqs table must be contiguous");
  const int64_t R = fr.size(-3);
  TORCH_CHECK(fr.numel() == S * R * 4, "freqs numel mismatch");
  TORCH_CHECK(2 * R <= D, "rot_dim must be <= D");
  TORCH_CHECK(qw.numel() == D && kw.numel() == D && qw.is_contiguous()
              && kw.is_contiguous() && qw.scalar_type() == torch::kFloat
              && kw.scalar_type() == torch::kFloat, "qw/kw must be contiguous fp32 [D]");

  id<MTLComputePipelineState> p = pso_named(@"rms_rope_sh_partial_bf16");
  MPSStream* stream = getCurrentMPSStream();
  auto a = enc_args({q, k, fr, qw, kw});
  uint Hu = (uint)H, Du = (uint)D, Ru = (uint)R, rst = (uint)q.stride(1);
  float ef = (float)eps;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:p];
      for (int i = 0; i < a.nbuf; i++) [enc setBuffer:a.bufs[i] offset:a.offs[i] atIndex:i];
      [enc setBytes:&Hu  length:sizeof(uint)  atIndex:5];
      [enc setBytes:&Du  length:sizeof(uint)  atIndex:6];
      [enc setBytes:&Ru  length:sizeof(uint)  atIndex:7];
      [enc setBytes:&rst length:sizeof(uint)  atIndex:8];
      [enc setBytes:&ef  length:sizeof(float) atIndex:9];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)S, (NSUInteger)(2 * H), 1)
          threadsPerThreadgroup:MTLSizeMake((NSUInteger)D, 1, 1)];
    }
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_mod_seg", &rms_mod_seg,
        "Affine RMSNorm + per-segment modulate in one pass (bf16, MPS)");
  m.def("silu_mul_split", &silu_mul_split,
        "SwiGLU gating from a fused [M,2F] buffer, no half copies (bf16, MPS)");
  m.def("rms_rope_sh_partial", &rms_rope_sh_partial,
        "In-place per-head RMSNorm + partial split-half RoPE on strided q/k (bf16, MPS)");
  m.def("rope_qk", &rope_qk,
        "RoPE on attention-layout q/k [B,H,S,D] (bf16, MPS; flux apply_rope semantics)");
  m.def("qkv_norm_rope", &qkv_norm_rope,
        "Fused qkv split + QK RMSNorm + RoPE + head transpose (bf16, MPS)");
  m.def("qkv2_norm_rope", &qkv2_norm_rope,
        "Fused two-source (txt+img) qkv split + QK RMSNorm + RoPE + seq concat (bf16, MPS)");
  m.def("qkv_norm_rope_sh", &qkv_norm_rope_sh,
        "Fused qkv split + affine RMSNorm + split-half RoPE (Ideogram 4, bf16, MPS)");
  m.def("qkv3_norm_rope", &qkv3_norm_rope,
        "Fused three-source GQA q/k/v + Krea2 QK RMSNorm + RoPE, native head counts (bf16, MPS)");
  m.def("rope_split", &rope_split,
        "LTX split-RoPE, one pass, strided per-head cos/sin (bf16, MPS)");
  m.def("rms_mod", &rms_mod,
        "LTX fused no-weight RMSNorm + (1+scale)+shift modulate (bf16, MPS)");
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
