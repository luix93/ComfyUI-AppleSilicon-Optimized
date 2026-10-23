// Fibo 2.0 exact memory-pass fusions. The RMS reduction follows PyTorch
// RMSNorm.metal (adapted from MLX, Copyright 2024 Apple Inc.): N_READS=4,
// ceil(D/128)*32 threads per row, simd_sum then threadgroup simd_sum,
// precise rsqrt and w*(x*inv) in FP32. Every BF16 rounding boundary of the
// upstream modulate / apply_gate / in-place residual chain is retained.
// See THIRD_PARTY_NOTICES.md and _patches/krea2_ext/{PYTORCH,MLX}_LICENSE.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline bfloat rte(float f) {
  uint u = as_type<uint>(f); u += 0x7fffu + ((u >> 16) & 1u);
  return as_type<bfloat>(ushort(u >> 16));
}

inline float row_inv(device const bfloat* x, uint D, float eps, uint tid, uint nt,
                     uint lane, uint sg, threadgroup float* sums, threadgroup float& inv) {
  float acc = 0;
  for (uint r = 0; r < D; r += nt * 4)
    for (uint j = 0; j < 4; j++)
      if (r + tid * 4 + j < D) { float v = float(x[r + tid * 4 + j]); acc += v * v; }
  acc = simd_sum(acc);
  if (sg == 0) sums[lane] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    acc = simd_sum(sums[lane]);
    if (lane == 0) inv = precise::rsqrt(acc / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return inv;
}

// out = rms_norm(x, w) * factor[sel], sel = 1 for tokens in [s0, s1), else 0
// (upstream modulate: x[:, s] *= (1 + scale)[noisy], the rest *= (1 + scale)[zero]).
kernel void norm_mod(device const bfloat* x [[buffer(0)]], device const bfloat* w [[buffer(1)]],
                     device const bfloat* factor [[buffer(2)]], device bfloat* out [[buffer(3)]],
                     constant uint& D [[buffer(4)]], constant uint& S [[buffer(5)]],
                     constant uint& s0 [[buffer(6)]], constant uint& s1 [[buffer(7)]],
                     constant float& eps [[buffer(8)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[32], inv;
  device const bfloat* xr = x + ulong(row) * D;
  float r_inv = row_inv(xr, D, eps, tid, nt, lane, sg, sums, inv);
  uint s = row % S;
  device const bfloat* f = factor + ((s >= s0 && s < s1) ? D : 0);
  device bfloat* o = out + ulong(row) * D;
  for (uint r = 0; r < D; r += nt * 4)
    for (uint j = 0; j < 4; j++)
      if (r + tid * 4 + j < D) {
        uint d = r + tid * 4 + j;
        bfloat n = rte(float(w[d]) * (float(xr[d]) * r_inv));
        o[d] = rte(float(n) * float(f[d]));
      }
}

// x += rms_norm(y, w) * gate[sel], in place (upstream apply_gate then residual add_).
kernel void norm_gate_residual(device bfloat* x [[buffer(0)]], device const bfloat* y [[buffer(1)]],
                               device const bfloat* w [[buffer(2)]], device const bfloat* gate [[buffer(3)]],
                               constant uint& D [[buffer(4)]], constant uint& S [[buffer(5)]],
                               constant uint& s0 [[buffer(6)]], constant uint& s1 [[buffer(7)]],
                               constant float& eps [[buffer(8)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint nt [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[32], inv;
  device const bfloat* yr = y + ulong(row) * D;
  float r_inv = row_inv(yr, D, eps, tid, nt, lane, sg, sums, inv);
  uint s = row % S;
  device const bfloat* g = gate + ((s >= s0 && s < s1) ? D : 0);
  device bfloat* xr = x + ulong(row) * D;
  for (uint r = 0; r < D; r += nt * 4)
    for (uint j = 0; j < 4; j++)
      if (r + tid * 4 + j < D) {
        uint d = r + tid * 4 + j;
        bfloat n = rte(float(w[d]) * (float(yr[d]) * r_inv));
        bfloat v = rte(float(n) * float(g[d]));
        xr[d] = rte(float(xr[d]) + float(v));
      }
}
