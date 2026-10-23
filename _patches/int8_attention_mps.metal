// Opt-in dense INT8 attention for M5 (approximate; every key participates).
// Matrix fragments and the online-softmax structure are adapted from CCV
// NAInt8SolAttentionKernel.cpp, revision 6a611be1aab6470ae279115ae4eaf7e01bc87135
// (Liu Liu and contributors), BSD-3-Clause; full notice:
// ../third_party/ccv-BSD-3-Clause.txt.
// Pack changes: arbitrary [B,H,N,128] input/output strides, unequal query/key
// lengths, midrange K centering, mean-centered per-channel V scales applied
// once at the end, 255-level probabilities stored as P-128 (the Kitchen
// offset trick, restored here from V prefix sums), and INT32 PV accumulation
// across key tiles. Accumulators are rescaled in place only when a row maximum
// rises, so no per-tile float epilogue runs. Padded keys score exactly zero
// and carry zero V; their normalizer mass is removed once at the end.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Params (int64): B, H, LQ, LK, q strides (b,h,n), k strides, v strides, o strides.
#define I8A_B ((uint)P[0])
#define I8A_H ((uint)P[1])
#define I8A_LQ ((uint)P[2])
#define I8A_LK ((uint)P[3])
#define I8A_STATS_ROWS 256u

inline ulong i8a_index(device const long* P, uint which, uint bh, uint n) {
  const uint b = bh / I8A_H, h = bh % I8A_H;
  return ulong(b) * ulong(P[4 + which * 3]) + ulong(h) * ulong(P[5 + which * 3]) + ulong(n) * ulong(P[6 + which * 3]);
}

// Per-channel K/V minimum/maximum over 256-token chunks.
kernel void i8a_stats(device const IN_T* K [[buffer(0)]], device const IN_T* V [[buffer(1)]],
  device float4* partial [[buffer(2)]], device float* partial_sum [[buffer(4)]], device const long* P [[buffer(3)]],
  uint d [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint chunk = group.x, bh = group.y, chunks = (I8A_LK + I8A_STATS_ROWS - 1) / I8A_STATS_ROWS;
  float kmin = INFINITY, kmax = -INFINITY, vmin = INFINITY, vmax = -INFINITY, vsum = 0;
  const uint end = min(I8A_LK, (chunk + 1) * I8A_STATS_ROWS);
  for (uint n = chunk * I8A_STATS_ROWS; n < end; ++n) {
    const float k = float(K[i8a_index(P, 1, bh, n) + d]);
    const float v = float(V[i8a_index(P, 2, bh, n) + d]);
    kmin = min(kmin, k); kmax = max(kmax, k); vmin = min(vmin, v); vmax = max(vmax, v); vsum += v;
  }
  partial[(ulong(bh) * chunks + chunk) * 128 + d] = float4(kmin, kmax, vmin, vmax);
  partial_sum[(ulong(bh) * chunks + chunk) * 128 + d] = vsum;
}

kernel void i8a_finalize(device const float4* partial [[buffer(0)]], device float* KC [[buffer(1)]],
  device float* VC [[buffer(2)]], device float* VSC [[buffer(3)]], device const long* P [[buffer(4)]],
  device const float* partial_sum [[buffer(5)]],
  uint d [[thread_index_in_threadgroup]], uint bh [[threadgroup_position_in_grid]]) {
  const uint chunks = (I8A_LK + I8A_STATS_ROWS - 1) / I8A_STATS_ROWS;
  float4 r = float4(INFINITY, -INFINITY, INFINITY, -INFINITY);
  float sum = 0;
  for (uint c = 0; c < chunks; ++c) {
    const float4 x = partial[(ulong(bh) * chunks + c) * 128 + d];
    r = float4(min(r.x, x.x), max(r.y, x.y), min(r.z, x.z), max(r.w, x.w));
    sum += partial_sum[(ulong(bh) * chunks + c) * 128 + d];
  }
  // Mean-centered V keeps probability mass lost to rounding near the
  // mean instead of an arbitrary center; this matters for diffuse attention.
  const float mean = sum / float(I8A_LK);
  KC[bh * 128 + d] = 0.5f * (r.x + r.y);
  VC[bh * 128 + d] = mean;
  const float extent = max(r.w - mean, mean - r.z);
  VSC[bh * 128 + d] = extent > 0 ? extent / 127.0f : 1.0f / 127.0f;
}

// One threadgroup quantizes 16 query rows; 128 threads, one per channel.
kernel void i8a_quant_q(device const IN_T* Q [[buffer(0)]], device int8_t* QI [[buffer(1)]],
  device float* QS [[buffer(2)]], device const long* P [[buffer(3)]],
  uint d [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint block = group.x, bh = group.y, LQP = (I8A_LQ + 63) / 64 * 64;
  float values[16]; float m = 0;
  for (uint r = 0; r < 16; ++r) {
    const uint n = block * 16 + r;
    values[r] = n < I8A_LQ ? float(Q[i8a_index(P, 0, bh, n) + d]) : 0.0f;
    m = max(m, abs(values[r]));
  }
  threadgroup float partials[4];
  m = simd_max(m);
  if (lane == 0) partials[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
  const float inverse = m > 0 ? 127.0f / m : 127.0f;
  if (d == 0) QS[bh * (LQP / 16) + block] = m > 0 ? m / 127.0f : 1.0f / 127.0f;
  for (uint r = 0; r < 16; ++r)
    QI[(ulong(bh) * LQP + block * 16 + r) * 128 + d] = int8_t(clamp(rint(values[r] * inverse), -127.0f, 127.0f));
}

// One threadgroup quantizes a 64-key block: centered K with one block scale,
// centered V with the per-channel scale from i8a_finalize.
kernel void i8a_quant_kv(device const IN_T* K [[buffer(0)]], device const IN_T* V [[buffer(1)]],
  device int8_t* KI [[buffer(2)]], device int8_t* VI [[buffer(3)]], device float* KS [[buffer(4)]],
  device const float* KC [[buffer(5)]], device const float* VC [[buffer(6)]], device const float* VSC [[buffer(7)]],
  device const long* P [[buffer(8)]], device int* VPRE [[buffer(9)]],
  uint d [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint block = group.x, bh = group.y, LKP = (I8A_LK + 63) / 64 * 64;
  const float kc = KC[bh * 128 + d], vc = VC[bh * 128 + d], vinv = 1.0f / VSC[bh * 128 + d];
  float m = 0;
  for (uint r = 0; r < 64; ++r) {
    const uint n = block * 64 + r;
    if (n < I8A_LK) m = max(m, abs(float(K[i8a_index(P, 1, bh, n) + d]) - kc));
  }
  threadgroup float partials[4];
  m = simd_max(m);
  if (lane == 0) partials[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
  const float inverse = m > 0 ? 127.0f / m : 127.0f;
  if (d == 0) KS[bh * (LKP / 64) + block] = m > 0 ? m / 127.0f : 1.0f / 127.0f;
  int vsum = 0;
  for (uint r = 0; r < 64; ++r) {
    const uint n = block * 64 + r;
    const ulong dst = (ulong(bh) * LKP + n) * 128 + d;
    if (n < I8A_LK) {
      KI[dst] = int8_t(clamp(rint((float(K[i8a_index(P, 1, bh, n) + d]) - kc) * inverse), -127.0f, 127.0f));
      const int8_t vq = int8_t(clamp(rint((float(V[i8a_index(P, 2, bh, n) + d]) - vc) * vinv), -127.0f, 127.0f));
      VI[dst] = vq; vsum += vq;
    } else {
      KI[dst] = 0; VI[dst] = 0;
    }
  }
  VPRE[(ulong(bh) * (LKP / 64 + 1) + block + 1) * 128 + d] = vsum;
}

// Inclusive scan of per-block V sums: VPRE[t] = sum of quantized V before tile t.
kernel void i8a_prefix(device int* VPRE [[buffer(0)]], device const long* P [[buffer(1)]],
  uint d [[thread_index_in_threadgroup]], uint bh [[threadgroup_position_in_grid]]) {
  const uint tiles = (I8A_LK + 63) / 64;
  device int* row = VPRE + ulong(bh) * (tiles + 1) * 128 + d;
  int total = 0;
  row[0] = 0;
  for (uint t = 1; t <= tiles; ++t) { total += row[t * 128]; row[t * 128] = total; }
}

kernel void i8a_attention(device int8_t* QI [[buffer(0)]], device int8_t* KI [[buffer(1)]],
  device int8_t* VI [[buffer(2)]], device const float* QS [[buffer(3)]], device const float* KS [[buffer(4)]],
  device const float* VSC [[buffer(5)]], device const float* VC [[buffer(6)]], device OUT_T* O [[buffer(7)]],
  device const long* P [[buffer(8)]], constant float& g [[buffer(9)]], device const int* VPRE [[buffer(10)]],
  constant float& headroom [[buffer(11)]],
  uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint LQ = I8A_LQ, LK = I8A_LK;
  const uint QG = (LQ + 63) / 64, LQP = QG * 64, LKP = (LK + 63) / 64 * 64;
  const uint qg = group.x % QG, bh = group.x / QG;
  if (bh >= I8A_B * I8A_H) return;
  const uint row = qg * 64 + sg * 16;

  auto QT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(QI + ulong(bh) * LQP * 128, dextents<int32_t, 2>(128, LQP));
  auto KT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(KI + ulong(bh) * LKP * 128, dextents<int32_t, 2>(128, LKP));
  auto VT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(VI + ulong(bh) * LKP * 128, dextents<int32_t, 2>(128, LKP));
  constexpr auto qk_desc = matmul2d_descriptor(16, 64, 64, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto pv_desc = matmul2d_descriptor(16, 32, 64, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> qk;
  matmul2d<pv_desc, execution_simdgroups<1>> pv;
  auto mq8 = QT.slice<64, 16>(0, row);
  auto mk8 = KT.slice<64, 64>(0, 0);
  auto mv8 = VT.slice<32, 64>(0, 0);
  auto S8 = qk.get_destination_cooperative_tensor<decltype(mq8), decltype(mk8), int>();
  using score_t = decltype(qk.get_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>());
  thread score_t& S = reinterpret_cast<thread score_t&>(S8);
  auto M = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto L = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto correction = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto P8 = pv.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto Z0 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), int>();
  auto Z1 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), int>();
  auto Z2 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), int>();
  auto Z3 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), int>();
  // Same layout viewed as float, used only to map elements onto row tensors.
  using zf_t = decltype(pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), float>());
  thread zf_t& Z0f = reinterpret_cast<thread zf_t&>(Z0);
  // Query fragments stay in registers for the whole key traversal.
  auto CQ0 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto CQ1 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  CQ0.load(QT.slice<64, 16>(0, row)); CQ1.load(QT.slice<64, 16>(64, row));
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < M.get_capacity(); ++i) if (M.is_valid_element(i)) { M[i] = -INFINITY; L[i] = 0; }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < Z0.get_capacity(); ++i) if (Z0.is_valid_element(i)) { Z0[i] = 0; Z1[i] = 0; Z2[i] = 0; Z3[i] = 0; }
  const float qscale = QS[bh * (LQP / 16) + row / 16] * g;
  device const float* kscales = KS + bh * (LKP / 64);
  const bool sync_long = LK >= 32768;
  device const int* vprefix = VPRE + ulong(bh) * (LKP / 64 + 1) * 128;
  // Per row: the tile through which the P-128 V-offset term has been restored.
  auto F = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < F.get_capacity(); ++i) if (F.is_valid_element(i)) F[i] = 0;

  for (uint c = 0, tile = 0; c < LKP; c += 64, ++tile) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S8.get_capacity(); ++i) if (S8.is_valid_element(i)) S8[i] = 0;
    auto k0 = KT.slice<64, 64>(0, c); auto k1 = KT.slice<64, 64>(64, c);
    qk.run(CQ0, k0, S8); qk.run(CQ1, k1, S8);
    const float scale = qscale * kscales[tile];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) S[i] = float(S8[i]) * scale;
    auto nextM = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
    reduce_rows(S, nextM, reduction_operation::max, -INFINITY);
    bool raised = false;
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < M.get_capacity(); ++i) if (M.is_valid_element(i)) {
      // A raised maximum gets optional headroom so small later rises do not
      // trigger another rescale; probabilities then stay below exp2(-headroom).
      const bool up = nextM[i] > M[i];
      const float next = up ? nextM[i] + headroom : M[i];
      raised = raised || up;
      correction[i] = up ? fast::exp2(M[i] - next) : 1.0f;
      M[i] = next;
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) S[i] = fast::exp2(S[i] - *M.map_iterator(S.get_iterator(i)));
    auto sum = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
    reduce_rows(S, sum, reduction_operation::sum, 0.0f);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < L.get_capacity(); ++i) if (L.is_valid_element(i)) L[i] = L[i] * correction[i] + sum[i];
    // Integer online-softmax rescale: only when some row's maximum rose.
    // Restore 128 * sum(V) for the P-128 encoding of the tiles accumulated
    // since the last rescale, then rescale rows whose maximum rose.
    if (simd_any(raised)) {
      device const int* b = vprefix + tile * 128;
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < Z0.get_capacity(); ++i) if (Z0.is_valid_element(i)) {
        const float f = *correction.map_iterator(Z0f.get_iterator(i));
        if (f != 1.0f) {
          device const int* a = vprefix + uint(*F.map_iterator(Z0f.get_iterator(i))) * 128;
          const uint d = Z0.get_multidimensional_index(i)[0];
          Z0[i] = int(rint(float(Z0[i] + 128 * (b[d] - a[d])) * f));
          Z1[i] = int(rint(float(Z1[i] + 128 * (b[d + 32] - a[d + 32])) * f));
          Z2[i] = int(rint(float(Z2[i] + 128 * (b[d + 64] - a[d + 64])) * f));
          Z3[i] = int(rint(float(Z3[i] + 128 * (b[d + 96] - a[d + 96])) * f));
        }
      }
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < F.get_capacity(); ++i) if (F.is_valid_element(i) && correction[i] != 1.0f) F[i] = float(tile);
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) P8[i] = int8_t(int(S[i] * 255.0f + 0.5f) - 128);
    auto v0 = VT.slice<32, 64>(0, c); auto v1 = VT.slice<32, 64>(32, c);
    auto v2 = VT.slice<32, 64>(64, c); auto v3 = VT.slice<32, 64>(96, c);
    pv.run(P8, v0, Z0); pv.run(P8, v1, Z1); pv.run(P8, v2, Z2); pv.run(P8, v3, Z3);
    // Keep SIMD groups near the same K/V block on long traversals.
    if (sync_long && (tile & 1) && c + 64 < LKP) threadgroup_barrier(mem_flags::mem_none);
  }

  {
    device const int* b = vprefix + (LKP / 64) * 128;
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < Z0.get_capacity(); ++i) if (Z0.is_valid_element(i)) {
      device const int* a = vprefix + uint(*F.map_iterator(Z0f.get_iterator(i))) * 128;
      const uint d = Z0.get_multidimensional_index(i)[0];
      Z0[i] += 128 * (b[d] - a[d]); Z1[i] += 128 * (b[d + 32] - a[d + 32]);
      Z2[i] += 128 * (b[d + 64] - a[d + 64]); Z3[i] += 128 * (b[d + 96] - a[d + 96]);
    }
  }
  // Padded keys scored exactly zero: remove their exp2(0 - M) mass once.
  const float padded = float(LKP - LK);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < L.get_capacity(); ++i) if (L.is_valid_element(i)) L[i] -= padded * fast::exp2(-M[i]);
  device const float* vscale = VSC + bh * 128;
  device const float* vcenter = VC + bh * 128;
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < Z0.get_capacity(); ++i) if (Z0.is_valid_element(i)) {
    const auto idx = Z0.get_multidimensional_index(i);
    const uint r = row + idx[1];
    if (r < LQ) {
      const float inverse = 1.0f / (255.0f * *L.map_iterator(Z0f.get_iterator(i)));
      device OUT_T* out = O + i8a_index(P, 3, bh, r);
      const uint d = idx[0];
      out[d] = OUT_T(float(Z0[i]) * vscale[d] * inverse + vcenter[d]);
      out[d + 32] = OUT_T(float(Z1[i]) * vscale[d + 32] * inverse + vcenter[d + 32]);
      out[d + 64] = OUT_T(float(Z2[i]) * vscale[d + 64] * inverse + vcenter[d + 64]);
      out[d + 96] = OUT_T(float(Z3[i]) * vscale[d + 96] * inverse + vcenter[d + 96]);
    }
  }
}

// ---------------------------------------------------------------------------
// Kitchen-numerics variant ("tiled"): randomized Hadamard Q/K rotation and
// per-key-tile probability scaling merged into FP32 accumulators. The rotation
// sign pattern and the deferred tile rescale follow the pack's Kitchen-policy
// Metal port (Comfy Kitchen v0.2.38, Apache-2.0; see kitchen_attention_mps.metal).

// Randomized orthonormal 128-point Hadamard: lane holds channels lane*4..lane*4+3.
inline float4 i8a_rotate128(float4 x, uint lane) {
  const uint signs[4] = {0x1035997bu, 0x8087f5eeu, 0xee2e4e1au, 0x71132418u};
  for (uint c = 0; c < 4; ++c)
    x[c] = as_type<float>(as_type<uint>(x[c]) ^ ((((signs[lane / 8] >> ((lane % 8) * 4 + c)) & 1u) ^ 1u) << 31));
  float a = x.x + x.y, b = x.x - x.y, c = x.z + x.w, d = x.z - x.w;
  x = float4(a + c, b + d, a - c, b - d) * 0.5f;
  for (uint bit = 1; bit < 32; bit *= 2) for (uint c = 0; c < 4; ++c) {
    const float other = simd_shuffle_xor(x[c], bit);
    x[c] = (lane & bit) ? other - x[c] : x[c] + other;
  }
  return x * 0.1767766952966369f;
}

// 16 query rows per threadgroup; each SIMD group rotates 4 rows.
kernel void i8a_quant_q_rot(device const IN_T* Q [[buffer(0)]], device int8_t* QI [[buffer(1)]],
  device float* QS [[buffer(2)]], device const long* P [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
  uint2 group [[threadgroup_position_in_grid]]) {
  const uint block = group.x, bh = group.y, LQP = (I8A_LQ + 63) / 64 * 64;
  float4 x[4]; float m = 0;
  for (uint j = 0; j < 4; ++j) {
    const uint n = block * 16 + sg * 4 + j;
    float4 v = 0;
    if (n < I8A_LQ) { device const IN_T* src = Q + i8a_index(P, 0, bh, n) + lane * 4; v = float4(src[0], src[1], src[2], src[3]); }
    x[j] = i8a_rotate128(v, lane);
    m = max(m, max(max(abs(x[j].x), abs(x[j].y)), max(abs(x[j].z), abs(x[j].w))));
  }
  threadgroup float partials[4];
  m = simd_max(m);
  if (lane == 0) partials[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
  const float inverse = m > 0 ? 127.0f / m : 127.0f;
  if (sg == 0 && lane == 0) QS[bh * (LQP / 16) + block] = m > 0 ? m / 127.0f : 1.0f / 127.0f;
  for (uint j = 0; j < 4; ++j) {
    device int8_t* dst = QI + (ulong(bh) * LQP + block * 16 + sg * 4 + j) * 128 + lane * 4;
    for (uint c = 0; c < 4; ++c) dst[c] = int8_t(clamp(rint(x[j][c] * inverse), -127.0f, 127.0f));
  }
}

// 64-key block: centered, rotated K with one block scale; V as in i8a_quant_kv.
// Writes per-block quantized V sums to VBLK[block] (no prefix scan needed).
kernel void i8a_quant_kv_rot(device const IN_T* K [[buffer(0)]], device const IN_T* V [[buffer(1)]],
  device int8_t* KI [[buffer(2)]], device int8_t* VI [[buffer(3)]], device float* KS [[buffer(4)]],
  device const float* KC [[buffer(5)]], device const float* VC [[buffer(6)]], device const float* VSC [[buffer(7)]],
  device const long* P [[buffer(8)]], device int* VBLK [[buffer(9)]],
  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
  uint2 group [[threadgroup_position_in_grid]]) {
  const uint block = group.x, bh = group.y, LKP = (I8A_LK + 63) / 64 * 64;
  const float4 kc = float4(KC[bh * 128 + lane * 4], KC[bh * 128 + lane * 4 + 1], KC[bh * 128 + lane * 4 + 2], KC[bh * 128 + lane * 4 + 3]);
  const float4 vc = float4(VC[bh * 128 + lane * 4], VC[bh * 128 + lane * 4 + 1], VC[bh * 128 + lane * 4 + 2], VC[bh * 128 + lane * 4 + 3]);
  const float4 vinv = 1.0f / float4(VSC[bh * 128 + lane * 4], VSC[bh * 128 + lane * 4 + 1], VSC[bh * 128 + lane * 4 + 2], VSC[bh * 128 + lane * 4 + 3]);
  float4 x[16]; float m = 0; int4 vsum = 0;
  for (uint j = 0; j < 16; ++j) {
    const uint n = block * 64 + sg * 16 + j;
    float4 k = 0;
    const ulong dst = (ulong(bh) * LKP + n) * 128 + lane * 4;
    if (n < I8A_LK) {
      device const IN_T* ks = K + i8a_index(P, 1, bh, n) + lane * 4;
      device const IN_T* vs = V + i8a_index(P, 2, bh, n) + lane * 4;
      k = float4(ks[0], ks[1], ks[2], ks[3]) - kc;
      const float4 v = (float4(vs[0], vs[1], vs[2], vs[3]) - vc) * vinv;
      for (uint c = 0; c < 4; ++c) { const int8_t q = int8_t(clamp(rint(v[c]), -127.0f, 127.0f)); VI[dst + c] = q; vsum[c] += q; }
    } else {
      for (uint c = 0; c < 4; ++c) VI[dst + c] = 0;
    }
    // Padded rows rotate zero to zero, preserving the exact-zero padded score.
    x[j] = i8a_rotate128(k, lane);
    m = max(m, max(max(abs(x[j].x), abs(x[j].y)), max(abs(x[j].z), abs(x[j].w))));
  }
  threadgroup float partials[4];
  threadgroup int4 sums[4][32];
  m = simd_max(m);
  if (lane == 0) partials[sg] = m;
  sums[sg][lane] = vsum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
  const float inverse = m > 0 ? 127.0f / m : 127.0f;
  if (sg == 0) {
    if (lane == 0) KS[bh * (LKP / 64) + block] = m > 0 ? m / 127.0f : 1.0f / 127.0f;
    const int4 total = sums[0][lane] + sums[1][lane] + sums[2][lane] + sums[3][lane];
    device int* out = VBLK + (ulong(bh) * (LKP / 64) + block) * 128 + lane * 4;
    out[0] = total.x; out[1] = total.y; out[2] = total.z; out[3] = total.w;
  }
  for (uint j = 0; j < 16; ++j) {
    device int8_t* dst = KI + (ulong(bh) * LKP + block * 64 + sg * 16 + j) * 128 + lane * 4;
    for (uint c = 0; c < 4; ++c) dst[c] = int8_t(clamp(rint(x[j][c] * inverse), -127.0f, 127.0f));
  }
}

// Per-tile probabilities: P = exp2(s - tile max), quantized to TILE_LEVELS
// (255 uses the P-128 encoding with exact V-sum correction), merged into FP32
// accumulators with deferred rescaling. ROUNDED_L uses the rounded sum.
kernel void i8a_attention_tiled(device int8_t* QI [[buffer(0)]], device int8_t* KI [[buffer(1)]],
  device int8_t* VI [[buffer(2)]], device const float* QS [[buffer(3)]], device const float* KS [[buffer(4)]],
  device const float* VSC [[buffer(5)]], device const float* VC [[buffer(6)]], device OUT_T* O [[buffer(7)]],
  device const long* P [[buffer(8)]], constant float& g [[buffer(9)]], device const int* VBLK [[buffer(10)]],
  uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint LQ = I8A_LQ, LK = I8A_LK;
  const uint QG = (LQ + 63) / 64, LQP = QG * 64, LKP = (LK + 63) / 64 * 64;
  const uint qg = group.x % QG, bh = group.x / QG;
  if (bh >= I8A_B * I8A_H) return;
  const uint row = qg * 64 + sg * 16;
  auto QT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(QI + ulong(bh) * LQP * 128, dextents<int32_t, 2>(128, LQP));
  auto KT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(KI + ulong(bh) * LKP * 128, dextents<int32_t, 2>(128, LKP));
  auto VT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(VI + ulong(bh) * LKP * 128, dextents<int32_t, 2>(128, LKP));
  constexpr auto qk_desc = matmul2d_descriptor(16, 64, 64, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto pv_desc = matmul2d_descriptor(16, 32, 64, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<qk_desc, execution_simdgroups<1>> qk;
  matmul2d<pv_desc, execution_simdgroups<1>> pv;
  auto mq8 = QT.slice<64, 16>(0, row);
  auto mk8 = KT.slice<64, 64>(0, 0);
  auto mv8 = VT.slice<32, 64>(0, 0);
  auto S8 = qk.get_destination_cooperative_tensor<decltype(mq8), decltype(mk8), int>();
  using score_t = decltype(qk.get_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>());
  thread score_t& S = reinterpret_cast<thread score_t&>(S8);
  auto M = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto L = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto F = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto TF = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto TL = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
  auto P8 = pv.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto O8 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), int>();
  auto O0 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), float>();
  auto O1 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), float>();
  auto O2 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), float>();
  auto O3 = pv.get_destination_cooperative_tensor<decltype(P8), decltype(mv8), float>();
  auto CQ0 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  auto CQ1 = qk.get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  CQ0.load(QT.slice<64, 16>(0, row)); CQ1.load(QT.slice<64, 16>(64, row));
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < M.get_capacity(); ++i) if (M.is_valid_element(i)) { M[i] = -INFINITY; L[i] = 0; }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < O0.get_capacity(); ++i) if (O0.is_valid_element(i)) { O0[i] = 0; O1[i] = 0; O2[i] = 0; O3[i] = 0; }
  const float qscale = QS[bh * (LQP / 16) + row / 16] * g;
  device const float* kscales = KS + bh * (LKP / 64);
  device const int* vblock = VBLK + ulong(bh) * (LKP / 64) * 128;
  const bool sync_long = LK >= 32768;
  for (uint c = 0, tile = 0; c < LKP; c += 64, ++tile) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S8.get_capacity(); ++i) if (S8.is_valid_element(i)) S8[i] = 0;
    auto k0 = KT.slice<64, 64>(0, c); auto k1 = KT.slice<64, 64>(64, c);
    qk.run(CQ0, k0, S8); qk.run(CQ1, k1, S8);
    const float scale = qscale * kscales[tile];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) S[i] = float(S8[i]) * scale;
    auto TM = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
    reduce_rows(S, TM, reduction_operation::max, -INFINITY);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < M.get_capacity(); ++i) if (M.is_valid_element(i)) {
      const float next = max(M[i], TM[i]);
      F[i] = M[i] == -INFINITY ? 0.0f : fast::exp2(M[i] - next);
      TF[i] = fast::exp2(TM[i] - next);
      TL[i] = TM[i];
      M[i] = next;
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i))
      S[i] = rint(fast::exp2(S[i] - *TM.map_iterator(S.get_iterator(i))) * TILE_LEVELS);
    auto sum = qk.get_row_reduction_destination_cooperative_tensor<decltype(mq8), decltype(mk8), float>();
    reduce_rows(S, sum, reduction_operation::sum, 0.0f);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < L.get_capacity(); ++i) if (L.is_valid_element(i)) L[i] = L[i] * F[i] + sum[i] * TF[i];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < S.get_capacity(); ++i) if (S.is_valid_element(i)) P8[i] = int8_t(int(S[i]) - P_OFFSET);
    auto v0 = VT.slice<32, 64>(0, c); auto v1 = VT.slice<32, 64>(32, c);
    auto v2 = VT.slice<32, 64>(64, c); auto v3 = VT.slice<32, 64>(96, c);
    device const int* vb = vblock + tile * 128;
    pv.run(P8, v0, O8);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < O8.get_capacity(); ++i) if (O8.is_valid_element(i)) {
      const uint d = O8.get_multidimensional_index(i)[0];
      O0[i] = fma(float(O8[i] + P_OFFSET * vb[d]), *TF.map_iterator(O0.get_iterator(i)), O0[i] * *F.map_iterator(O0.get_iterator(i)));
    }
    pv.run(P8, v1, O8);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < O8.get_capacity(); ++i) if (O8.is_valid_element(i)) {
      const uint d = O8.get_multidimensional_index(i)[0] + 32;
      O1[i] = fma(float(O8[i] + P_OFFSET * vb[d]), *TF.map_iterator(O1.get_iterator(i)), O1[i] * *F.map_iterator(O1.get_iterator(i)));
    }
    pv.run(P8, v2, O8);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < O8.get_capacity(); ++i) if (O8.is_valid_element(i)) {
      const uint d = O8.get_multidimensional_index(i)[0] + 64;
      O2[i] = fma(float(O8[i] + P_OFFSET * vb[d]), *TF.map_iterator(O2.get_iterator(i)), O2[i] * *F.map_iterator(O2.get_iterator(i)));
    }
    pv.run(P8, v3, O8);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < O8.get_capacity(); ++i) if (O8.is_valid_element(i)) {
      const uint d = O8.get_multidimensional_index(i)[0] + 96;
      O3[i] = fma(float(O8[i] + P_OFFSET * vb[d]), *TF.map_iterator(O3.get_iterator(i)), O3[i] * *F.map_iterator(O3.get_iterator(i)));
    }
    if (sync_long && (tile & 1) && c + 64 < LKP) threadgroup_barrier(mem_flags::mem_none);
  }
  // Padded keys scored exactly zero and occur only in the last tile, which
  // added rint(exp2(-tile max) * levels) * TF per padded key to L; remove it.
  const float padded = float(LKP - LK);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < L.get_capacity(); ++i) if (L.is_valid_element(i))
    L[i] -= padded * rint(fast::exp2(-TL[i]) * TILE_LEVELS) * TF[i];
  device const float* vscale = VSC + bh * 128;
  device const float* vcenter = VC + bh * 128;
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < O0.get_capacity(); ++i) if (O0.is_valid_element(i)) {
    const auto idx = O0.get_multidimensional_index(i);
    const uint r = row + idx[1];
    if (r < LQ) {
      const float inverse = 1.0f / *L.map_iterator(O0.get_iterator(i));
      device OUT_T* out = O + i8a_index(P, 3, bh, r);
      const uint d = idx[0];
      out[d] = OUT_T(O0[i] * vscale[d] * inverse + vcenter[d]);
      out[d + 32] = OUT_T(O1[i] * vscale[d + 32] * inverse + vcenter[d + 32]);
      out[d + 64] = OUT_T(O2[i] * vscale[d + 64] * inverse + vcenter[d + 64]);
      out[d + 96] = OUT_T(O3[i] * vscale[d + 96] * inverse + vcenter[d + 96]);
    }
  }
}
