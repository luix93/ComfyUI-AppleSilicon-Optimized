// Generated/adapted from CCV NAInt8AttentionKernel (Liu Liu and contributors).
// Copyright (c) 2010, Liu Liu. All rights reserved. BSD-3-Clause.
// Full notice: ../CCV_LICENSE and the repository's third_party/ccv-BSD-3-Clause.txt.
// Changes: model geometry and PyTorch MPS host bindings; research prototype.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;



constant uint QUANTIZE_Q_THREADS = 128;
constant uint QUANTIZE_KV_THREADS = 256;
constant uint QUANTIZE_V_MEAN_THREADS = 256;
constant uint QUANTIZE_SIMD_LANES = 32;
constant uint QUANTIZE_Q_SIMDGROUPS = QUANTIZE_Q_THREADS / QUANTIZE_SIMD_LANES;
constant uint QUANTIZE_KV_SIMDGROUPS = QUANTIZE_KV_THREADS / QUANTIZE_SIMD_LANES;
constant uint QUANTIZE_V_MEAN_SIMDGROUPS = QUANTIZE_V_MEAN_THREADS / QUANTIZE_SIMD_LANES;

constant uint QUANTIZE_Q_HEADS [[function_constant(902)]];

constant uint QUANTIZE_KV_HEADS [[function_constant(903)]];

constant uint QUANTIZE_Q_TILE_SIZE [[function_constant(904)]];

constant uint QUANTIZE_KV_TILE_SIZE [[function_constant(905)]];



inline float quantize_reduce_max(float value,
                                 threadgroup float *scratch,
                                 ushort sgid,
                                 ushort lane_id,
                                 uint simdgroup_count) {
  value = max(value, simd_shuffle_xor(value, 16));
  value = max(value, simd_shuffle_xor(value, 8));
  value = max(value, simd_shuffle_xor(value, 4));
  value = max(value, simd_shuffle_xor(value, 2));
  value = max(value, simd_shuffle_xor(value, 1));
  if (lane_id == 0)
    scratch[sgid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    value = lane_id < simdgroup_count ? scratch[lane_id] : 0.0f;
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane_id == 0)
      scratch[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}



using io_vec4 = vec<bfloat, 4>;
using v_mean_vec4 = vec<float, 4>;

template <ushort tile_rows, ushort simdgroups>
inline void quantize_qk_tile(
    device const bfloat *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    threadgroup float *scratch,
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sequence,
    bool has_remainder,
    uint heads,
    uint tile_size,
    uint scale_tiles,
    uint batch_stride,
    uint scale_batch_stride,
    uint thread_count,
    uint simdgroup_count) {
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint start = tile * tile_size;
  const uint extent = has_remainder ? min(tile_size, sequence - start) : tile_size;
  const uint vectors_per_row = 128 / 4;
  const uint total_vectors = extent * vectors_per_row;
  device const io_vec4 *src4 = reinterpret_cast<device const io_vec4 *>(src);
  device char4 *dst4 = reinterpret_cast<device char4 *>(dst);
  float local_max = 0;
  for (uint i = tid; i < total_vectors; i += thread_count) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * batch_stride +
        ((row * heads + head) * 128 + vec_dim * 4);
    const float4 value = float4(src4[index / 4]);
    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])),
        max(fabs(value[2]), fabs(value[3]))));
  }
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id, simdgroup_count);
  const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[batch * scale_batch_stride + head * scale_tiles + tile] = scale;
  for (uint i = tid; i < total_vectors; i += thread_count) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * batch_stride +
        ((row * heads + head) * 128 + vec_dim * 4);
    const float4 value = float4(src4[index / 4]) * inv_scale;
    const int4 rounded = int4(rint(value));
    dst4[index / 4] = char4(clamp(rounded, int4(-127), int4(127)));
  }
}



kernel void quantize_q(
    constant uint* dimensions [[buffer(21)]],
    device const bfloat *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  const uniform<uint> QUANTIZE_Q_SEQUENCE = make_uniform(dimensions[0]);
  const uniform<uint> QUANTIZE_KV_SEQUENCE = make_uniform(dimensions[1]);
  const uniform<uint> QUANTIZE_Q_SCALE_TILES = make_uniform((uint(QUANTIZE_Q_SEQUENCE) + 15u) / 16u);
  const uniform<uint> QUANTIZE_KV_SCALE_TILES = make_uniform((uint(QUANTIZE_KV_SEQUENCE) + 63u) / 64u);
  const uniform<uint> QUANTIZE_Q_BATCH_STRIDE = make_uniform(dimensions[2]);
  const uniform<uint> QUANTIZE_K_BATCH_STRIDE = make_uniform(dimensions[3]);
  const uniform<uint> QUANTIZE_V_BATCH_STRIDE = make_uniform(dimensions[4]);
  const uniform<uint> QUANTIZE_Q_SCALE_BATCH_STRIDE = make_uniform(dimensions[10]);
  const uniform<uint> QUANTIZE_KV_SCALE_BATCH_STRIDE = make_uniform(dimensions[11]);
  threadgroup float scratch[QUANTIZE_Q_SIMDGROUPS];
  quantize_qk_tile<16, 4>(src, dst, scales, scratch, tid, sgid, lane_id, tgid,
      QUANTIZE_Q_SEQUENCE,
      true, QUANTIZE_Q_HEADS,
      QUANTIZE_Q_TILE_SIZE,
      QUANTIZE_Q_SCALE_TILES,
      QUANTIZE_Q_BATCH_STRIDE,
      QUANTIZE_Q_SCALE_BATCH_STRIDE,
      QUANTIZE_Q_THREADS,
      QUANTIZE_Q_SIMDGROUPS);
}

kernel void quantize_k(
    constant uint* dimensions [[buffer(21)]],
    device const bfloat *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  const uniform<uint> QUANTIZE_Q_SEQUENCE = make_uniform(dimensions[0]);
  const uniform<uint> QUANTIZE_KV_SEQUENCE = make_uniform(dimensions[1]);
  const uniform<uint> QUANTIZE_Q_SCALE_TILES = make_uniform((uint(QUANTIZE_Q_SEQUENCE) + 15u) / 16u);
  const uniform<uint> QUANTIZE_KV_SCALE_TILES = make_uniform((uint(QUANTIZE_KV_SEQUENCE) + 63u) / 64u);
  const uniform<uint> QUANTIZE_Q_BATCH_STRIDE = make_uniform(dimensions[2]);
  const uniform<uint> QUANTIZE_K_BATCH_STRIDE = make_uniform(dimensions[3]);
  const uniform<uint> QUANTIZE_V_BATCH_STRIDE = make_uniform(dimensions[4]);
  const uniform<uint> QUANTIZE_Q_SCALE_BATCH_STRIDE = make_uniform(dimensions[10]);
  const uniform<uint> QUANTIZE_KV_SCALE_BATCH_STRIDE = make_uniform(dimensions[11]);
  threadgroup float scratch[QUANTIZE_KV_SIMDGROUPS];
  quantize_qk_tile<64, 8>(src, dst, scales, scratch, tid, sgid, lane_id, tgid,
      QUANTIZE_KV_SEQUENCE,
      true, QUANTIZE_KV_HEADS,
      QUANTIZE_KV_TILE_SIZE,
      QUANTIZE_KV_SCALE_TILES,
      QUANTIZE_K_BATCH_STRIDE,
      QUANTIZE_KV_SCALE_BATCH_STRIDE,
      QUANTIZE_KV_THREADS,
      QUANTIZE_KV_SIMDGROUPS);
}

inline uint compact_morton_even_bits(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

inline uint2 morton_decode_2d(uint code) {
  return uint2(compact_morton_even_bits(code),
               compact_morton_even_bits(code >> 1));
}

inline uint lower_bits_mask(uint bit_count) {
  if (bit_count == 0)
    return 0;
  return (1u << bit_count) - 1;
}

inline uint2 morton_decode_rectangular_2d(uint code,
                                          uint x_bits,
                                          uint y_bits) {
  const uint paired_bits = min(x_bits, y_bits);
  const uint paired_code = code & lower_bits_mask(paired_bits * 2);
  uint2 tile = morton_decode_2d(paired_code);
  uint tail = code >> (paired_bits * 2);
  if (x_bits > paired_bits) {
    const uint x_extra_bits = x_bits - paired_bits;
    tile.x |= (tail & lower_bits_mask(x_extra_bits)) << paired_bits;
    tail >>= x_extra_bits;
  }
  if (y_bits > paired_bits) {
    tile.y |= tail << paired_bits;
  }
  return tile;
}

inline uint ceil_log2_u32(uint x) {
  if (x <= 1)
    return 0;
  x -= 1;
  uint bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}



// Each threadgroup sums one chunk of rows for one batch, KV head and operand.
// SIMD groups read complete rows, so no cache line is split across
// threadgroups. Rows, SIMD groups and chunks combine in a fixed order, so the
// mean is deterministic for a given shape.
kernel void compute_v_mean(
    constant uint* dimensions [[buffer(21)]],
    device const bfloat *v_src [[buffer(0)]],
    device float *v_mean [[buffer(1)]],
    device const bfloat *k_src [[buffer(2)]],
    device float *k_mean [[buffer(3)]],
    device float4 *partials [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  const uniform<uint> QUANTIZE_Q_SEQUENCE = make_uniform(dimensions[0]);
  const uniform<uint> QUANTIZE_KV_SEQUENCE = make_uniform(dimensions[1]);
  const uniform<uint> QUANTIZE_Q_SCALE_TILES = make_uniform((uint(QUANTIZE_Q_SEQUENCE) + 15u) / 16u);
  const uniform<uint> QUANTIZE_KV_SCALE_TILES = make_uniform((uint(QUANTIZE_KV_SEQUENCE) + 63u) / 64u);
  const uniform<uint> QUANTIZE_Q_BATCH_STRIDE = make_uniform(dimensions[2]);
  const uniform<uint> QUANTIZE_K_BATCH_STRIDE = make_uniform(dimensions[3]);
  const uniform<uint> QUANTIZE_V_BATCH_STRIDE = make_uniform(dimensions[4]);
  const uniform<uint> QUANTIZE_Q_SCALE_BATCH_STRIDE = make_uniform(dimensions[10]);
  const uniform<uint> QUANTIZE_KV_SCALE_BATCH_STRIDE = make_uniform(dimensions[11]);
  const uint chunk = tgid.x, operand_head = tgid.y, batch = tgid.z;
  const uint head = operand_head % QUANTIZE_KV_HEADS;
  const uint operand = 1 > 1 ? operand_head / QUANTIZE_KV_HEADS : 0;
  device const bfloat *src = operand ? k_src : v_src;
  const uint start = min(chunk * 512u, uint(QUANTIZE_KV_SEQUENCE));
  const uint end = min(start + 512u, uint(QUANTIZE_KV_SEQUENCE));
  const uint stream = sgid * (32 / 32) + lane_id / 32;
  const uint lane = lane_id % 32;
  float4 sums[1];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 1; ++i)
    sums[i] = 0.0f;
  // Issue four rows' loads before accumulating them. Short grids have too few
  // SIMD groups to hide one dependent load per row; row order is unchanged.
  uint first = start + stream;
  for (; first + 3 * 8 < end; first += 4 * 8) {
    io_vec4 loaded[4][1];
    #pragma clang loop unroll(full)
    for (uint u = 0; u < 4; ++u) {
      const uint row = first + u * 8;
      device const io_vec4 *values = reinterpret_cast<device const io_vec4 *>(src + batch * (operand ? QUANTIZE_K_BATCH_STRIDE : QUANTIZE_V_BATCH_STRIDE) + (row * QUANTIZE_KV_HEADS + head) * 128);
      #pragma clang loop unroll(full)
      for (uint i = 0; i < 1; ++i) {
        const uint vec = i * 32 + lane;
        loaded[u][i] = vec < 32 ? values[vec] : io_vec4(0);
      }
    }
    #pragma clang loop unroll(full)
    for (uint u = 0; u < 4; ++u)
      #pragma clang loop unroll(full)
      for (uint i = 0; i < 1; ++i)
        sums[i] += float4(loaded[u][i]);
  }
  for (uint row = first; row < end; row += 8) {
    device const io_vec4 *values = reinterpret_cast<device const io_vec4 *>(src + batch * (operand ? QUANTIZE_K_BATCH_STRIDE : QUANTIZE_V_BATCH_STRIDE) + (row * QUANTIZE_KV_HEADS + head) * 128);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 1; ++i) {
      const uint vec = i * 32 + lane;
      if (vec < 32)
        sums[i] += float4(values[vec]);
    }
  }
  // Fold rows packed into one SIMD group, then add SIMD groups in index order.
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 1; ++i)
    for (ushort offset = 32; offset < 32; offset *= 2)
      sums[i] += simd_shuffle_xor(sums[i], offset);
  threadgroup float4 simd_sums[QUANTIZE_V_MEAN_SIMDGROUPS * 32];
  if (lane_id < 32) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 1; ++i) {
      const uint vec = i * 32 + lane;
      if (vec < 32)
        simd_sums[sgid * 32 + vec] = sums[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // One chunk covers the sequence: write the mean and skip finalize_v_mean.
  const bool single_chunk = uint(QUANTIZE_KV_SEQUENCE) <= 512u;
  const uint chunks = (uint(QUANTIZE_KV_SEQUENCE) + 512u - 1) / 512u;
  for (uint vec = tid; vec < 32; vec += QUANTIZE_V_MEAN_THREADS) {
    float4 total = simd_sums[vec];
    for (uint s = 1; s < QUANTIZE_V_MEAN_SIMDGROUPS; ++s)
      total += simd_sums[s * 32 + vec];
    if (single_chunk)
      reinterpret_cast<device float4 *>(operand ? k_mean : v_mean)[(batch * QUANTIZE_KV_HEADS + head) * 32 + vec] =
          total * (1.0f / float(QUANTIZE_KV_SEQUENCE));
    else
      partials[((batch * 1 * QUANTIZE_KV_HEADS + operand_head) * chunks + chunk) * 32 + vec] = total;
  }
}

// Sums the per-chunk partials in chunk order.
kernel void finalize_v_mean(
    constant uint* dimensions [[buffer(21)]],
    device float *v_mean [[buffer(1)]],
    device float *k_mean [[buffer(3)]],
    device const float4 *partials [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
  ) {
  const uniform<uint> QUANTIZE_Q_SEQUENCE = make_uniform(dimensions[0]);
  const uniform<uint> QUANTIZE_KV_SEQUENCE = make_uniform(dimensions[1]);
  const uniform<uint> QUANTIZE_Q_SCALE_TILES = make_uniform((uint(QUANTIZE_Q_SEQUENCE) + 15u) / 16u);
  const uniform<uint> QUANTIZE_KV_SCALE_TILES = make_uniform((uint(QUANTIZE_KV_SEQUENCE) + 63u) / 64u);
  const uniform<uint> QUANTIZE_Q_BATCH_STRIDE = make_uniform(dimensions[2]);
  const uniform<uint> QUANTIZE_K_BATCH_STRIDE = make_uniform(dimensions[3]);
  const uniform<uint> QUANTIZE_V_BATCH_STRIDE = make_uniform(dimensions[4]);
  const uniform<uint> QUANTIZE_Q_SCALE_BATCH_STRIDE = make_uniform(dimensions[10]);
  const uniform<uint> QUANTIZE_KV_SCALE_BATCH_STRIDE = make_uniform(dimensions[11]);
  const uint vec = gid.x, operand_head = gid.y, batch = gid.z;
  const uint head = operand_head % QUANTIZE_KV_HEADS;
  const uint operand = 1 > 1 ? operand_head / QUANTIZE_KV_HEADS : 0;
  if (vec >= 32)
    return;
  const uint chunks = (uint(QUANTIZE_KV_SEQUENCE) + 512u - 1) / 512u;
  device const float4 *chunk_sums = partials + (batch * 1 * QUANTIZE_KV_HEADS + operand_head) * chunks * 32 + vec;
  float4 total = chunk_sums[0];
  for (uint chunk = 1; chunk < chunks; ++chunk)
    total += chunk_sums[chunk * 32];
  reinterpret_cast<device float4 *>(operand ? k_mean : v_mean)[(batch * QUANTIZE_KV_HEADS + head) * 32 + vec] =
      total * (1.0f / float(QUANTIZE_KV_SEQUENCE));
}


kernel void quantize_v(
    constant uint* dimensions [[buffer(21)]],
    device const bfloat *src [[buffer(0)]],
    device int8_t *dst [[buffer(1)]],
    device float *scales [[buffer(2)]],
    device const float *mean [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  const uniform<uint> QUANTIZE_Q_SEQUENCE = make_uniform(dimensions[0]);
  const uniform<uint> QUANTIZE_KV_SEQUENCE = make_uniform(dimensions[1]);
  const uniform<uint> QUANTIZE_Q_SCALE_TILES = make_uniform((uint(QUANTIZE_Q_SEQUENCE) + 15u) / 16u);
  const uniform<uint> QUANTIZE_KV_SCALE_TILES = make_uniform((uint(QUANTIZE_KV_SEQUENCE) + 63u) / 64u);
  const uniform<uint> QUANTIZE_Q_BATCH_STRIDE = make_uniform(dimensions[2]);
  const uniform<uint> QUANTIZE_K_BATCH_STRIDE = make_uniform(dimensions[3]);
  const uniform<uint> QUANTIZE_V_BATCH_STRIDE = make_uniform(dimensions[4]);
  const uniform<uint> QUANTIZE_Q_SCALE_BATCH_STRIDE = make_uniform(dimensions[10]);
  const uniform<uint> QUANTIZE_KV_SCALE_BATCH_STRIDE = make_uniform(dimensions[11]);
  threadgroup float scratch[QUANTIZE_KV_SIMDGROUPS];
  const uint tile = tgid.x;
  const uint head = tgid.y;
  const uint batch = tgid.z;
  const uint start = tile * QUANTIZE_KV_TILE_SIZE;
  const uint extent = min(QUANTIZE_KV_TILE_SIZE, QUANTIZE_KV_SEQUENCE - start);
  const uint vectors_per_row = 128 / 4;
  const uint total_vectors = extent * vectors_per_row;
  device const io_vec4 *src4 = reinterpret_cast<device const io_vec4 *>(src);
  device const v_mean_vec4 *mean4 = reinterpret_cast<device const v_mean_vec4 *>(mean);
  device char4 *dst4 = reinterpret_cast<device char4 *>(dst);
  float local_max = 0;
  for (uint i = tid; i < total_vectors; i += QUANTIZE_KV_THREADS) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE +
        ((row * QUANTIZE_KV_HEADS + head) * 128 + vec_dim * 4);
    const float4 value = float4(src4[index / 4]) -
        float4(mean4[((batch * QUANTIZE_KV_HEADS + head) * 128 + vec_dim * 4) / 4]);
    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])),
        max(fabs(value[2]), fabs(value[3]))));
  }
  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id, QUANTIZE_KV_SIMDGROUPS);
  const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
  const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[batch * QUANTIZE_KV_SCALE_BATCH_STRIDE + head * QUANTIZE_KV_SCALE_TILES + tile] = scale;
  for (uint i = tid; i < total_vectors; i += QUANTIZE_KV_THREADS) {
    const uint row = start + i / vectors_per_row;
    const uint vec_dim = i % vectors_per_row;
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE +
        ((row * QUANTIZE_KV_HEADS + head) * 128 + vec_dim * 4);
    const float4 value = float4(src4[index / 4]) -
        float4(mean4[((batch * QUANTIZE_KV_HEADS + head) * 128 + vec_dim * 4) / 4]);
    const int4 rounded = int4(rint(value * inv_scale));
    dst4[index / 4] = char4(clamp(rounded, int4(-127), int4(127)));
  }
}


constant uint dO_batch_stride [[function_constant(6)]];

constant uint dV_batch_stride [[function_constant(7)]];

constant uint dK_batch_stride [[function_constant(8)]];

constant uint dQ_batch_stride [[function_constant(9)]];

constant uint dO_scale_batch_stride [[function_constant(13)]];

constant uint V_mean_batch_stride [[function_constant(14)]];



constant uint Hq = 48;
constant uint Hk = 12;
constant uint K_Hq = 128 * Hq;
constant uint K_Hk = 128 * Hk;
constant uint Q_scale_tile_size = 16;
constant uint KV_scale_tile_size = 64;


constant uint K_edge = 128 + 1 - 32;


kernel void int8_attention(
    constant uint* dimensions [[buffer(21)]],


    device int8_t *Q_buf [[buffer(0)]],
    device int8_t *K_buf [[buffer(1)]],
    device int8_t *V_buf [[buffer(2)]],
    device bfloat *O_buf [[buffer(3)]],
    device float *L_buf [[buffer(4)]],
    device float *Q_scale_buf [[buffer(10)]],
    device float *K_scale_buf [[buffer(11)]],
    device float *V_scale_buf [[buffer(12)]],
    device const float *V_mean_buf [[buffer(14)]],




	    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
	    ushort tid [[thread_index_in_threadgroup]],
	    ushort sgid [[simdgroup_index_in_threadgroup]],
	    uint3 tgid [[threadgroup_position_in_grid]]
	  ) {
  // Online softmax uses -infinity. Preserve INF/NaN only in the forward kernel.
  #pragma METAL fp math_mode(relaxed)
  const uniform<uint> R = make_uniform(dimensions[0]);
  const uniform<uint> Q_scale_tiles = make_uniform((uint(R) + 15u) / 16u);
  const uniform<uint> R_edge = make_uniform(max(uint(R), 15u) + 1u - 16u);
  const uniform<uint> R_remainder = make_uniform(uint(R) % 16u);
  const uniform<uint> C = make_uniform(dimensions[1]);
  const uniform<uint> K_scale_tiles = make_uniform((uint(C) + 63u) / 64u);
  const uniform<uint> C_edge = make_uniform(max(uint(C), 63u) + 1u - 64u);
  const uniform<uint> C_remainder = make_uniform(uint(C) % 64u);
  const uniform<uint> Q_batch_stride = make_uniform(dimensions[2]);
  const uniform<uint> K_batch_stride = make_uniform(dimensions[3]);
  const uniform<uint> V_batch_stride = make_uniform(dimensions[4]);
  const uniform<uint> O_batch_stride = make_uniform(dimensions[5]);
  const uniform<uint> Q_scale_batch_stride = make_uniform(dimensions[10]);
  const uniform<uint> K_scale_batch_stride = make_uniform(dimensions[11]);
  const uniform<uint> V_scale_batch_stride = make_uniform(dimensions[12]);
  const uint row_group_count = (R + 16 * 4 - 1) / (16 * 4);

  tgid = uint3(tgid.x % row_group_count, tgid.x / row_group_count, tgid.z);
  if (tgid.y >= 48 || tgid.x >= row_group_count) {
    return;
  }
  tgid.x = tgid.x * 4 + sgid;
  if (tgid.x * 16 >= R) {
    return;
  }


  Q_buf += tgid.z * Q_batch_stride;
  K_buf += tgid.z * K_batch_stride;
  V_buf += tgid.z * V_batch_stride;
  O_buf += tgid.z * O_batch_stride;
  L_buf += (tgid.z * Hq + tgid.y) * R;
  Q_scale_buf += tgid.z * Q_scale_batch_stride + tgid.y * Q_scale_tiles + (tgid.x * 16) / Q_scale_tile_size;
  K_scale_buf += tgid.z * K_scale_batch_stride + (tgid.y  / 4) * K_scale_tiles;
  V_scale_buf += tgid.z * V_scale_batch_stride + (tgid.y  / 4) * K_scale_tiles;
  V_mean_buf += tgid.z * V_mean_batch_stride + (tgid.y  / 4) * 128;



  auto Q = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  constexpr auto qk_desc = matmul2d_descriptor(16, 64, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;


  auto mQ = Q.slice<32, 16>(tgid.y * 128, tgid.x * 16);
  auto mK = K.slice<32, 64>(tgid.y / 4* 128, 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), int32_t>();
  using qk_float_tensor_t = decltype(matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>());
  thread qk_float_tensor_t& cP_0 = reinterpret_cast<thread qk_float_tensor_t&>(cS_0);
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();


  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = -numeric_limits<float>::infinity();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }


  auto mV = V.slice<32, 64>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor(16, 32, 64, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
  using pv_float_left_tensor_t = decltype(matmul_pv_op.get_left_input_cooperative_tensor<half, int8_t, float>());
  constexpr auto pv_int8_desc = matmul2d_descriptor(16, 32, 64, false, false, true, matmul2d_descriptor::mode::multiply);
  matmul2d<pv_int8_desc, execution_simdgroups<1>> matmul_pv_int8_op;
  using pv_int8_left_tensor_t = decltype(matmul_pv_int8_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>());
  auto cOq = matmul_pv_int8_op.get_destination_cooperative_tensor<pv_int8_left_tensor_t, decltype(mV), int32_t>();

  auto cO_0 = matmul_pv_op.get_destination_cooperative_tensor<pv_float_left_tensor_t, decltype(mV), float>();

  auto cO_1 = matmul_pv_op.get_destination_cooperative_tensor<pv_float_left_tensor_t, decltype(mV), float>();

  auto cO_2 = matmul_pv_op.get_destination_cooperative_tensor<pv_float_left_tensor_t, decltype(mV), float>();

  auto cO_3 = matmul_pv_op.get_destination_cooperative_tensor<pv_float_left_tensor_t, decltype(mV), float>();


  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {

      cO_0[k] = 0;

      cO_1[k] = 0;

      cO_2[k] = 0;

      cO_3[k] = 0;


    }
  }




  for (uint c = 0; c < C; c += 64) {


    const float block_scale = Q_scale_buf[0] * K_scale_buf[c / KV_scale_tile_size] * 0.127517432;
    const float v_scale_recip_127 = V_scale_buf[c / KV_scale_tile_size] / 127.0f;


    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += 32) {
      auto mQ = Q.slice<32, 16>(tgid.y * 128 + k, tgid.x * 16);
      auto mK_0 = K.slice<32, 64>(tgid.y / 4* 128 + k, c);
      matmul_qk_op.run(mQ, mK_0, cS_0);
    }


    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        const int score_0 = cS_0[k];

        // Bounded K/V loads zero-fill the last tile; exclude its invalid scores.
        const int column = int(c) + cP_0.get_multidimensional_index(k)[0];
        cP_0[k] = column < int(C) ? (float)score_0 * block_scale : -numeric_limits<float>::infinity();

      }
    }


    auto cM_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP_0, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_new[k];
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
        cL[k] *= correction[k];
      }
    }


    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        auto it = cP_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cP_0[k] = fast::exp2(cP_0[k] - *dst_it);
      }
    }


    auto cL_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cP_0, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if (cL.is_valid_element(k)) {
        cL[k] += cL_new[k];
      }
    }
    thread pv_int8_left_tensor_t& cP_q_0 = reinterpret_cast<thread pv_int8_left_tensor_t&>(cS_0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cP_0.get_capacity(); ++k) {
      if (cP_0.is_valid_element(k)) {
        const int quantized = (int)rint(cP_0[k] * 127.0f);
        cP_q_0[k] = (int8_t)clamp(quantized, 0, 127);
      }
    }


    auto mV_0_0 = V.slice<32, 64>(tgid.y / 4* 128 + 0, c);
    matmul_pv_int8_op.run(cP_q_0, mV_0_0, cOq);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cOq.get_capacity(); ++k) {
      if (cOq.is_valid_element(k)) {


        auto it = cO_0.get_iterator(k);
        auto factor = correction.map_iterator(it);
        cO_0[k] = fma(cO_0[k], *factor, (float)cOq[k] * v_scale_recip_127);


      }
    }


    auto mV_0_1 = V.slice<32, 64>(tgid.y / 4* 128 + 32, c);
    matmul_pv_int8_op.run(cP_q_0, mV_0_1, cOq);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cOq.get_capacity(); ++k) {
      if (cOq.is_valid_element(k)) {


        auto it = cO_0.get_iterator(k);
        auto factor = correction.map_iterator(it);
        cO_1[k] = fma(cO_1[k], *factor, (float)cOq[k] * v_scale_recip_127);


      }
    }


    auto mV_0_2 = V.slice<32, 64>(tgid.y / 4* 128 + 64, c);
    matmul_pv_int8_op.run(cP_q_0, mV_0_2, cOq);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cOq.get_capacity(); ++k) {
      if (cOq.is_valid_element(k)) {


        auto it = cO_0.get_iterator(k);
        auto factor = correction.map_iterator(it);
        cO_2[k] = fma(cO_2[k], *factor, (float)cOq[k] * v_scale_recip_127);


      }
    }


    auto mV_0_3 = V.slice<32, 64>(tgid.y / 4* 128 + 96, c);
    matmul_pv_int8_op.run(cP_q_0, mV_0_3, cOq);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cOq.get_capacity(); ++k) {
      if (cOq.is_valid_element(k)) {


        auto it = cO_0.get_iterator(k);
        auto factor = correction.map_iterator(it);
        cO_3[k] = fma(cO_3[k], *factor, (float)cOq[k] * v_scale_recip_127);


      }
    }


    if (2 > 0 &&
        c + 64 < C &&
        ((((c / 64) + 1) % 2) == 0)) {
      threadgroup_barrier(mem_flags::mem_none);
    }
  }


  auto O = O_buf + tgid.x * (16 * K_Hq) + tgid.y * 128;
  auto L = L_buf + tgid.x * 16;
  if (R_remainder > 0 && tgid.x * 16 >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = cL.map_iterator(it);
          auto L_reciprocal = fast::divide(1, *dst_it);


          O[idx[0] + 0 + idx[1] * K_Hq] = (bfloat)(cO_0[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 0]
);


          O[idx[0] + 32 + idx[1] * K_Hq] = (bfloat)(cO_1[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 32]
);


          O[idx[0] + 64 + idx[1] * K_Hq] = (bfloat)(cO_2[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 64]
);


          O[idx[0] + 96 + idx[1] * K_Hq] = (bfloat)(cO_3[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 96]
);


        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        if (idx[0] < (int)R_remainder) {
          L[idx[0]] = (float)(cM[k] + fast::log2(cL[k]));
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);


        O[idx[0] + 0 + idx[1] * K_Hq] = (bfloat)(cO_0[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 0]
);


        O[idx[0] + 32 + idx[1] * K_Hq] = (bfloat)(cO_1[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 32]
);


        O[idx[0] + 64 + idx[1] * K_Hq] = (bfloat)(cO_2[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 64]
);


        O[idx[0] + 96 + idx[1] * K_Hq] = (bfloat)(cO_3[k] * L_reciprocal
 + (float)V_mean_buf[idx[0] + 96]
);


      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        auto idx = cM.get_multidimensional_index(k);
        L[idx[0]] = (float)(cM[k] + fast::log2(cL[k]));
      }
    }
  }

}
