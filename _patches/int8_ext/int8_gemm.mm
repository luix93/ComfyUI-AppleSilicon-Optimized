// ============================================================
// Bit-exact INT8×INT8→INT32 TensorOps GEMM for torch / MPS (M5+, Metal 4)
//
// C[M,N] (int32) = A[M,K] (int8) @ B[N,K]^T (int8)   -- NT layout
//
// This is the matmul our S14 probe could not make fast: the naive
// "single op.run over the whole block" cooperative-tensor kernel
// plateaued at ~75 TF/s (fp16-class). The structure here instead does
// CUTLASS-style manual register tiling — small 16x32x16 cooperative
// fragments, int32 accumulators held in registers, explicit BK=512/SK=32
// K-blocking across 16 simdgroups (WM=4 x WN=4) — which saturates the
// int8 units (~1.8x over bf16 at GEMM shapes).
//
// Two epilogues share one GEMM body (w8a8_gemm_compute):
//   • int8_matmul_int32      — stores raw INT32 C (bit-exact vs _int_mm).
//   • int8_matmul_fused      — fuses the W8A8 rescale + optional bias into
//     the store: D[m,n] = bf16(float(C[m,n]) * row_scale[m]) + bf16(bias[n]),
//     so the int32 product never round-trips through global memory (the
//     "fused dequant" variant; cuts the per-call epilogue traffic).
//
// The Metal kernel body (helpers + GEMM compute + entry points) is ported
// from Cider (https://github.com/Mininglamp-AI/cider,
// cider/kernels/w8a8_matmul.metal — including its w8a8_matmul_fused_dequant
// epilogue), MIT License, Copyright (c) 2026 Mininglamp contributors. Only
// the host dispatch is ours (torch MPS ObjC++ shim mirroring
// _patches/fp8_ext/fp8_matmul2d.mm).
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
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// Deterministic round-to-nearest-even float -> bfloat. The compiler's own
// float->bfloat conversion truncates in store paths (observed at midpoints),
// which diverges from torch's RTE casts by 1 ulp. Finite values only.
inline bfloat bf16_rte(float f) {
  uint u = as_type<uint>(f);
  uint lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return as_type<bfloat>(ushort(u >> 16));
}

// ── NAXFrag layout constants ────────────────────────────────────
constant constexpr short kElemsPerFrag = 8;
constant constexpr short kElemCols = 4;
constant constexpr short kElemRowsJump = 8;

// ── NAXFrag coordinate mapping ──────────────────────────────────
inline short2 nax_get_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}

// ── Fragment load: device → register ────────────────────────────
template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src, int ld, short2 sc,
                          short off_m = 0, short off_n = 0) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
    }
  }
}

// ── Fragment store: raw INT32 (no dequant) ───────────────────
inline void nax_frag_store_int32(const thread int32_t *src, device int32_t *dst,
                                 int ld, short2 sc, short off_m, short off_n,
                                 uint M, uint N, uint m_base, uint n_base) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] =
            src[i * kElemCols + j];
      }
    }
  }
}

// ── Fragment store: fused dequant + optional bias ───────────────
// D[m,n] = OutT(float(C[m,n]) * row_scale[m]) + OutT(bias[n]).
// Matches the eager epilogue exactly: int32→fp32 convert, fp32 rescale by
// the per-row factor (weight_scale * x_row_scale), round to OutT, then a
// SEPARATE OutT-precision bias add (bf16 + bf16) — so it is bit-identical
// to chunk.float()*scale -> .to(bf16) -> + bias.to(bf16).
template <typename OutT>
inline void nax_frag_store_dequant(const thread int32_t *src, device OutT *dst,
                                   int ld, short2 sc, short off_m, short off_n,
                                   uint M, uint N, uint m_base, uint n_base,
                                   const device float *row_scale,
                                   const device OutT *bias, bool has_bias) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        float acc = float(src[i * kElemCols + j]) * row_scale[mi];
        OutT r = bf16_rte(acc);
        if (has_bias) {
          r = bf16_rte(float(r) + float(bias[ni]));
        }
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] = r;
      }
    }
  }
}

// ── Raw INT32 GEMM compute (B is [N, K], transpose_b=true) ──────
// Fills c_frags (int32 register accumulators) for this simdgroup's tile
// of C[M,N] = A[M,K] × B[N,K]^T, and reports the tile origin (m_base,
// n_base) + per-thread NAXFrag coord (sc) for the caller's store epilogue.
// Returns false when this threadgroup maps outside the problem (skip store).
template <bool Enabled> struct compiler_barrier_slot {
  inline void touch() {}
};
template <> struct compiler_barrier_slot<true> {
  volatile int value;
  inline void touch() { (void)value; }
};

template <int BM, int BN, int BK, int SK, int WM, int WN,
          bool USE_COMPILER_BARRIER>
bool w8a8_gemm_compute(
    const device int8_t *A, const device int8_t *B,
    thread int32_t (&c_frags)[((BM / WM) / 16) * ((BN / WN) / 16)][kElemsPerFrag],
    thread short2 &sc_out, thread uint &m_base_out, thread uint &n_base_out,
    uint M, uint N, uint K, uint swizzle_log, uint tiles_m, uint tiles_n,
    uint2 tgid, uint sgid, uint lid) {
  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  uint tid_y = (tgid.y << swizzle_log) + (tgid.x & ((1u << swizzle_log) - 1u));
  uint tid_x = tgid.x >> swizzle_log;
  if (tid_x >= tiles_n || tid_y >= tiles_m) {
    return false;
  }

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  const device int8_t *sg_A = A + m_base * K;
  const device int8_t *sg_B = B + n_base * K;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      gemm_op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  // decltype on a local carries the `thread` addrspace qualifier; macOS 27
  // beta 26A5388g tightened the destination overload's enable_if so the
  // qualified type no longer satisfies __is_cooperative_tensor_type_v —
  // strip it (older headers accepted either form).
  auto ct_c =
      gemm_op.get_destination_cooperative_tensor<
          mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(ct_a)>,
          mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(ct_b)>,
          int32_t>();

  for (int f = 0; f < TM * TN; f++) {
    for (int i = 0; i < kElemsPerFrag; i++) {
      c_frags[f][i] = 0;
    }
  }

  int gemm_k_iters = int(K) / BK;
  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TN][TK][kElemsPerFrag];
      compiler_barrier_slot<USE_COMPILER_BARRIER> compiler_barrier;

      for (short mm = 0; mm < TM; mm++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc, short(mm * 16),
                        short(kk * 16));
        }
      }

      for (short nn = 0; nn < TN; nn++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(b_frags[nn][kk], sg_B + kk1, int(K), sc, short(nn * 16),
                        short(kk * 16));
        }
      }

      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_a[i] = a_frags[mm][kk][i];
            }
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[nn][kk][i];
              ct_b[kElemsPerFrag + i] = b_frags[nn + 1][kk][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kElemsPerFrag + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }
      compiler_barrier.touch();
    }
    sg_A += BK;
    sg_B += BK;
  }

  // Remainder K
  int rem_k = int(K) - gemm_k_iters * BK;
  for (int kk1 = 0; kk1 < rem_k; kk1 += 16) {
    int8_t a_frag[TM][kElemsPerFrag];
    int8_t b_frag[TN][kElemsPerFrag];
    short psk = short(max(0, rem_k - kk1));

    for (short mm = 0; mm < TM; mm++) {
      const device int8_t *ptr = sg_A + kk1 + (sc.y + mm * 16) * K + sc.x;
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          a_frag[mm][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
    }
    for (short nn = 0; nn < TN; nn++) {
      const device int8_t *ptr = sg_B + kk1 + (sc.y + nn * 16) * K + sc.x;
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          b_frag[nn][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
      }
    }
    for (short mm = 0; mm < TM; mm++) {
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_a[i] = a_frag[mm][i];
      }
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_b[i] = b_frag[0][i];
        ct_b[kElemsPerFrag + i] = b_frag[1][i];
      }
      short c0 = mm * TN, c1 = c0 + 1;
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_c[i] = c_frags[c0][i];
        ct_c[kElemsPerFrag + i] = c_frags[c1][i];
      }
      gemm_op.run(ct_a, ct_b, ct_c);
      for (short i = 0; i < kElemsPerFrag; i++) {
        c_frags[c0][i] = ct_c[i];
        c_frags[c1][i] = ct_c[kElemsPerFrag + i];
      }
    }
  }

  sc_out = sc;
  m_base_out = m_base;
  n_base_out = n_base;
  return true;
}

// ── INT32 entry points (raw store) ──────────────────────────────
kernel void int8_matmul_int32(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4, true>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc, short(mm * 16),
                           short(nn * 16), M, N, m_base, n_base);
}

kernel void int8_matmul_int32_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((32 / 1) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<32, 128, 512, 32, 1, 4, true>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc, short(mm * 16),
                           short(nn * 16), M, N, m_base, n_base);
}

// ── Fused (dequant + bias) entry points, bf16 output ────────────
kernel void int8_matmul_fused(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    const device float *row_scale [[buffer(9)]],
    const device bfloat *bias [[buffer(10)]],
    constant uint &has_bias [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4, true>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                     short(mm * 16), short(nn * 16), M, N,
                                     m_base, n_base, row_scale, bias,
                                     has_bias != 0u);
}

kernel void int8_matmul_fused_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    const device float *row_scale [[buffer(9)]],
    const device bfloat *bias [[buffer(10)]],
    constant uint &has_bias [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((32 / 1) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<32, 128, 512, 32, 1, 4, true>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                     short(mm * 16), short(nn * 16), M, N,
                                     m_base, n_base, row_scale, bias,
                                     has_bias != 0u);
}

// ── Fragment store: per-column (channelwise) dequant + optional bias ────────
// D[m,n] = OutT(float(C[m,n]) * (col_scale[n] * row_scale[m])) + OutT(bias[n]).
// Matches the eager chunked epilogue for channelwise weight scales exactly:
// torch computes (weight_scale[n] * x_scale[m]) FIRST, then multiplies C.
template <typename OutT>
inline void nax_frag_store_dequant_cs(const thread int32_t *src, device OutT *dst,
                                      int ld, short2 sc, short off_m, short off_n,
                                      uint M, uint N, uint m_base, uint n_base,
                                      const device float *row_scale,
                                      const device float *col_scale,
                                      const device OutT *bias, bool has_bias) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        float s = col_scale[ni] * row_scale[mi];
        float acc = float(src[i * kElemCols + j]) * s;
        OutT r = bf16_rte(acc);
        if (has_bias) {
          r = bf16_rte(float(r) + float(bias[ni]));
        }
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] = r;
      }
    }
  }
}

// H3 QKV epilogue: same channelwise dequant/bias arithmetic as above, but
// scatter columns from [M,3*H*D] directly into three contiguous [1,H,M,D]
// attention-layout tensors. This removes the full fused-qkv bf16 allocation;
// Q/K normalization and RoPE run in place in dit_ext immediately afterwards.
template <typename OutT>
inline void nax_frag_store_dequant_cs_qkv(
    const thread int32_t *src, device OutT *qo, device OutT *ko,
    device OutT *vo, short2 sc, short off_m, short off_n,
    uint M, uint N, uint m_base, uint n_base,
    const device float *row_scale, const device float *col_scale,
    const device OutT *bias, bool has_bias, uint H, uint D) {
  const uint W = H * D;
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      const uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      const uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        const float s = col_scale[ni] * row_scale[mi];
        OutT r = bf16_rte(float(src[i * kElemCols + j]) * s);
        if (has_bias) r = bf16_rte(float(r) + float(bias[ni]));
        const uint kind = ni / W;
        const uint inner = ni - kind * W;
        const uint h = inner / D;
        const uint d = inner - h * D;
        device OutT *dst = kind == 0u ? qo : (kind == 1u ? ko : vo);
        dst[((ulong)h * M + mi) * D + d] = r;
      }
    }
  }
}

kernel void int8_matmul_fused_cs(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    const device float *row_scale [[buffer(9)]],
    const device bfloat *bias [[buffer(10)]],
    constant uint &has_bias [[buffer(11)]],
    const device float *col_scale [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4, true>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant_cs<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                        short(mm * 16), short(nn * 16), M, N,
                                        m_base, n_base, row_scale, col_scale,
                                        bias, has_bias != 0u);
}

// Long-M channelwise path. Removing the volatile load/compute scheduling
// fence lets Metal overlap fragment traffic more effectively once M is large
// enough to provide abundant independent threadgroups. INT32 accumulation and
// the bf16 epilogue are identical to int8_matmul_fused_cs.
kernel void int8_matmul_fused_cs_long(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    const device float *row_scale [[buffer(9)]],
    const device bfloat *bias [[buffer(10)]],
    constant uint &has_bias [[buffer(11)]],
    const device float *col_scale [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4, false>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant_cs<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                        short(mm * 16), short(nn * 16), M, N,
                                        m_base, n_base, row_scale, col_scale,
                                        bias, has_bias != 0u);
}

kernel void int8_matmul_fused_cs_qkv(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *Q [[buffer(2)]], device bfloat *Kout [[buffer(3)]],
    device bfloat *V [[buffer(4)]], constant uint &M [[buffer(5)]],
    constant uint &N [[buffer(6)]], constant uint &Kin [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    const device float *row_scale [[buffer(11)]],
    const device bfloat *bias [[buffer(12)]],
    constant uint &has_bias [[buffer(13)]],
    const device float *col_scale [[buffer(14)]],
    constant uint &H [[buffer(15)]], constant uint &D [[buffer(16)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4, true>(
          A, B, c_frags, sc, m_base, n_base, M, N, Kin, swizzle_log,
          tiles_m, tiles_n, tgid, sgid, lid))
    return;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant_cs_qkv<bfloat>(
          c_frags[mm * TN + nn], Q, Kout, V, sc, short(mm * 16),
          short(nn * 16), M, N, m_base, n_base, row_scale, col_scale,
          bias, has_bias != 0u, H, D);
}

kernel void int8_matmul_fused_cs_qkv_long(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *Q [[buffer(2)]], device bfloat *Kout [[buffer(3)]],
    device bfloat *V [[buffer(4)]], constant uint &M [[buffer(5)]],
    constant uint &N [[buffer(6)]], constant uint &Kin [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    const device float *row_scale [[buffer(11)]],
    const device bfloat *bias [[buffer(12)]],
    constant uint &has_bias [[buffer(13)]],
    const device float *col_scale [[buffer(14)]],
    constant uint &H [[buffer(15)]], constant uint &D [[buffer(16)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4, false>(
          A, B, c_frags, sc, m_base, n_base, M, N, Kin, swizzle_log,
          tiles_m, tiles_n, tgid, sgid, lid))
    return;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant_cs_qkv<bfloat>(
          c_frags[mm * TN + nn], Q, Kout, V, sc, short(mm * 16),
          short(nn * 16), M, N, m_base, n_base, row_scale, col_scale,
          bias, has_bias != 0u, H, D);
}

// H3's D=128 heads line up exactly with the GEMM's 128-column output tile.
// Keep one complete Q/K tile in threadgroup memory so RMSNorm and partial
// split-half RoPE can consume the projection epilogue without first writing
// raw Q/K to device memory.  The union deliberately reuses the same 32 KiB:
// once a four-row batch has loaded its values into registers, the first few
// bytes temporarily hold the fp32 simdgroup reduction results.
union QKVEpilogueScratch {
  bfloat value[128 * 128];
  float sum[(128 * 128) / 2];
};

template <bool USE_COMPILER_BARRIER>
inline void int8_matmul_fused_cs_qkv_rms_rope_impl(
    const device int8_t *A, const device int8_t *B,
    device bfloat *Q, device bfloat *Kout, device bfloat *V,
    constant uint &M, constant uint &N, constant uint &Kin,
    constant uint &swizzle_log, constant uint &tiles_m,
    constant uint &tiles_n, const device float *row_scale,
    const device bfloat *bias, constant uint &has_bias,
    const device float *col_scale, constant uint &H,
    const device bfloat *fr, const device float *qw,
    const device float *kw, constant uint &R, constant float &eps,
    uint2 tgid, uint sgid, uint lid, uint tid,
    threadgroup QKVEpilogueScratch *work) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4,
                         USE_COMPILER_BARRIER>(
          A, B, c_frags, sc, m_base, n_base, M, N, Kin, swizzle_log,
          tiles_m, tiles_n, tgid, sgid, lid))
    return;

  const uint sg_row = sgid >> 2;
  const uint sg_col = sgid & 3u;
  const uint tile_m_base = m_base - sg_row * 32u;
  const uint tile_n_base = n_base - sg_col * 32u;
  const uint slot = tile_n_base >> 7;
  const uint kind = slot / H;
  const uint head = slot - kind * H;

  // V has no normalization or rotation, so retain the direct scatter store.
  if (kind == 2u) {
    for (short mm = 0; mm < TM; ++mm)
      for (short nn = 0; nn < TN; ++nn)
        nax_frag_store_dequant_cs_qkv<bfloat>(
            c_frags[mm * TN + nn], Q, Kout, V, sc, short(mm * 16),
            short(nn * 16), M, N, m_base, n_base, row_scale, col_scale,
            bias, has_bias != 0u, H, 128u);
    return;
  }

  // Materialize the exact channelwise dequant/bias result on chip.  Invalid
  // tail rows are zero-filled because every thread must participate in the
  // following barriers.
  for (short mm = 0; mm < TM; ++mm) {
    for (short nn = 0; nn < TN; ++nn) {
      const thread int32_t *src = c_frags[mm * TN + nn];
      for (short i = 0; i < 2; ++i) {
        for (short j = 0; j < kElemCols; ++j) {
          const uint row_in_tile =
              sg_row * 32u + uint(sc.y + mm * 16 + i * kElemRowsJump);
          const uint col_in_tile =
              sg_col * 32u + uint(sc.x + nn * 16 + j);
          const uint mi = tile_m_base + row_in_tile;
          const uint ni = tile_n_base + col_in_tile;
          bfloat value = bfloat(0.0f);
          if (mi < M && ni < N) {
            const float scale = col_scale[ni] * row_scale[mi];
            value = bf16_rte(float(src[i * kElemCols + j]) * scale);
            if (has_bias != 0u)
              value = bf16_rte(float(value) + float(bias[ni]));
          }
          work->value[row_in_tile * 128u + col_in_tile] = value;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_lane = tid >> 7;
  const uint d = tid & 127u;
  const device float *norm_weight = kind == 0u ? qw : kw;
  device bfloat *dst_base = kind == 0u ? Q : Kout;

  // 512 threads normalize four rows at once.  Each row uses four consecutive
  // simdgroups, matching rms_rope_attention_partial_bf16's reduction order.
  for (uint row_base = 0; row_base < 128u; row_base += 4u) {
    const uint row_in_tile = row_base + row_lane;
    const uint value_index = row_in_tile * 128u + d;
    const float x = float(work->value[value_index]);
    const float partial = simd_sum(x * x);
    // The aliased sum slots occupy row zero. Every simdgroup must finish its
    // value load before another simdgroup is allowed to overwrite those bytes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) work->sum[sgid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    const uint sum_base = row_lane * 4u;
    for (uint i = 0; i < 4u; ++i) total += work->sum[sum_base + i];
    // The fp32 sums alias row zero.  All readers must finish before normalized
    // values overwrite that storage.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = precise::rsqrt(
        precise::divide(total, 128.0f) + eps);
    const bfloat normalized = bf16_rte(x * inv);
    const bfloat y = bf16_rte(float(normalized) * norm_weight[d]);
    work->value[value_index] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mi = tile_m_base + row_in_tile;
    if (mi < M) {
      device bfloat *dst =
          dst_base + ((ulong)head * M + mi) * 128u;
      if (d >= 2u * R) {
        dst[d] = y;
      } else {
        const uint i = d < R ? d : d - R;
        const uint r = d < R ? 0u : 1u;
        const device bfloat *matrix =
            fr + (((ulong)mi * R + i) * 2u + r) * 2u;
        const bfloat p0 = bf16_rte(
            float(matrix[0]) * float(work->value[row_in_tile * 128u + i]));
        const bfloat p1 = bf16_rte(
            float(matrix[1]) *
            float(work->value[row_in_tile * 128u + i + R]));
        dst[d] = bf16_rte(float(p0) + float(p1));
      }
    }
    // Protect the current batch's split-half reads from the next batch's
    // aliased fp32 reduction writes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void int8_matmul_fused_cs_qkv_rms_rope(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *Q [[buffer(2)]], device bfloat *Kout [[buffer(3)]],
    device bfloat *V [[buffer(4)]], constant uint &M [[buffer(5)]],
    constant uint &N [[buffer(6)]], constant uint &Kin [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    const device float *row_scale [[buffer(11)]],
    const device bfloat *bias [[buffer(12)]],
    constant uint &has_bias [[buffer(13)]],
    const device float *col_scale [[buffer(14)]],
    constant uint &H [[buffer(15)]],
    const device bfloat *fr [[buffer(16)]],
    const device float *qw [[buffer(17)]], const device float *kw [[buffer(18)]],
    constant uint &R [[buffer(19)]], constant float &eps [[buffer(20)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup QKVEpilogueScratch work;
  int8_matmul_fused_cs_qkv_rms_rope_impl<true>(
      A, B, Q, Kout, V, M, N, Kin, swizzle_log, tiles_m, tiles_n,
      row_scale, bias, has_bias, col_scale, H, fr, qw, kw, R, eps,
      tgid, sgid, lid, tid, &work);
}

kernel void int8_matmul_fused_cs_qkv_rms_rope_long(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *Q [[buffer(2)]], device bfloat *Kout [[buffer(3)]],
    device bfloat *V [[buffer(4)]], constant uint &M [[buffer(5)]],
    constant uint &N [[buffer(6)]], constant uint &Kin [[buffer(7)]],
    constant uint &swizzle_log [[buffer(8)]],
    constant uint &tiles_m [[buffer(9)]], constant uint &tiles_n [[buffer(10)]],
    const device float *row_scale [[buffer(11)]],
    const device bfloat *bias [[buffer(12)]],
    constant uint &has_bias [[buffer(13)]],
    const device float *col_scale [[buffer(14)]],
    constant uint &H [[buffer(15)]],
    const device bfloat *fr [[buffer(16)]],
    const device float *qw [[buffer(17)]], const device float *kw [[buffer(18)]],
    constant uint &R [[buffer(19)]], constant float &eps [[buffer(20)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup QKVEpilogueScratch work;
  int8_matmul_fused_cs_qkv_rms_rope_impl<false>(
      A, B, Q, Kout, V, M, N, Kin, swizzle_log, tiles_m, tiles_n,
      row_scale, bias, has_bias, col_scale, H, fr, qw, kw, R, eps,
      tgid, sgid, lid, tid, &work);
}

// Runtime-LoRA H3 Q/K partition. PyTorch computes the standard low-rank
// projection in bf16, one Q/K/V partition at a time; this epilogue adds that
// delta to the exact INT8 base projection before the nonlinear RMS reduction.
// Processing one partition per dispatch keeps peak delta storage at [M,H*128]
// rather than materializing a full [M,3*H*128] adapter output.
template <bool USE_COMPILER_BARRIER>
inline void int8_matmul_fused_cs_qk_lora_rms_rope_impl(
    const device int8_t *A, const device int8_t *B, device bfloat *Out,
    constant uint &M, constant uint &N, constant uint &Kin,
    constant uint &swizzle_log, constant uint &tiles_m,
    constant uint &tiles_n, const device float *row_scale,
    const device bfloat *bias, constant uint &has_bias,
    const device float *col_scale, const device bfloat *delta,
    constant float &delta_scale, const device bfloat *fr,
    const device float *norm_weight, constant uint &R, constant float &eps,
    uint2 tgid, uint sgid, uint lid, uint tid,
    threadgroup QKVEpilogueScratch *work) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4,
                         USE_COMPILER_BARRIER>(
          A, B, c_frags, sc, m_base, n_base, M, N, Kin, swizzle_log,
          tiles_m, tiles_n, tgid, sgid, lid))
    return;

  const uint sg_row = sgid >> 2;
  const uint sg_col = sgid & 3u;
  const uint tile_m_base = m_base - sg_row * 32u;
  const uint tile_n_base = n_base - sg_col * 32u;
  const uint head = tile_n_base >> 7;
  const bfloat lora_scale = bf16_rte(delta_scale);

  for (short mm = 0; mm < TM; ++mm) {
    for (short nn = 0; nn < TN; ++nn) {
      const thread int32_t *src = c_frags[mm * TN + nn];
      for (short i = 0; i < 2; ++i) {
        for (short j = 0; j < kElemCols; ++j) {
          const uint row_in_tile =
              sg_row * 32u + uint(sc.y + mm * 16 + i * kElemRowsJump);
          const uint col_in_tile =
              sg_col * 32u + uint(sc.x + nn * 16 + j);
          const uint mi = tile_m_base + row_in_tile;
          const uint ni = tile_n_base + col_in_tile;
          bfloat value = bfloat(0.0f);
          if (mi < M && ni < N) {
            const float scale = col_scale[ni] * row_scale[mi];
            value = bf16_rte(float(src[i * kElemCols + j]) * scale);
            if (has_bias != 0u)
              value = bf16_rte(float(value) + float(bias[ni]));
            const bfloat scaled_delta = bf16_rte(
                float(delta[(ulong)mi * N + ni]) * float(lora_scale));
            value = bf16_rte(float(value) + float(scaled_delta));
          }
          work->value[row_in_tile * 128u + col_in_tile] = value;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_lane = tid >> 7;
  const uint d = tid & 127u;
  for (uint row_base = 0; row_base < 128u; row_base += 4u) {
    const uint row_in_tile = row_base + row_lane;
    const uint value_index = row_in_tile * 128u + d;
    const float x = float(work->value[value_index]);
    const float partial = simd_sum(x * x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) work->sum[sgid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    const uint sum_base = row_lane * 4u;
    for (uint i = 0; i < 4u; ++i) total += work->sum[sum_base + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = precise::rsqrt(
        precise::divide(total, 128.0f) + eps);
    const bfloat normalized = bf16_rte(x * inv);
    const bfloat y = bf16_rte(float(normalized) * norm_weight[d]);
    work->value[value_index] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mi = tile_m_base + row_in_tile;
    if (mi < M) {
      device bfloat *dst = Out + ((ulong)head * M + mi) * 128u;
      if (d >= 2u * R) {
        dst[d] = y;
      } else {
        const uint i = d < R ? d : d - R;
        const uint r = d < R ? 0u : 1u;
        const device bfloat *matrix =
            fr + (((ulong)mi * R + i) * 2u + r) * 2u;
        const bfloat p0 = bf16_rte(
            float(matrix[0]) * float(work->value[row_in_tile * 128u + i]));
        const bfloat p1 = bf16_rte(
            float(matrix[1]) *
            float(work->value[row_in_tile * 128u + i + R]));
        dst[d] = bf16_rte(float(p0) + float(p1));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <bool USE_COMPILER_BARRIER>
inline void int8_matmul_fused_cs_v_lora_impl(
    const device int8_t *A, const device int8_t *B, device bfloat *Out,
    constant uint &M, constant uint &N, constant uint &Kin,
    constant uint &swizzle_log, constant uint &tiles_m,
    constant uint &tiles_n, const device float *row_scale,
    const device bfloat *bias, constant uint &has_bias,
    const device float *col_scale, const device bfloat *delta,
    constant float &delta_scale, uint2 tgid, uint sgid, uint lid) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4,
                         USE_COMPILER_BARRIER>(
          A, B, c_frags, sc, m_base, n_base, M, N, Kin, swizzle_log,
          tiles_m, tiles_n, tgid, sgid, lid))
    return;
  const bfloat lora_scale = bf16_rte(delta_scale);

  for (short mm = 0; mm < TM; ++mm) {
    for (short nn = 0; nn < TN; ++nn) {
      const thread int32_t *src = c_frags[mm * TN + nn];
      for (short i = 0; i < 2; ++i) {
        for (short j = 0; j < kElemCols; ++j) {
          const uint mi = m_base + uint(sc.y + mm * 16 + i * kElemRowsJump);
          const uint ni = n_base + uint(sc.x + nn * 16 + j);
          if (mi < M && ni < N) {
            const float scale = col_scale[ni] * row_scale[mi];
            bfloat value = bf16_rte(float(src[i * kElemCols + j]) * scale);
            if (has_bias != 0u)
              value = bf16_rte(float(value) + float(bias[ni]));
            const bfloat scaled_delta = bf16_rte(
                float(delta[(ulong)mi * N + ni]) * float(lora_scale));
            value = bf16_rte(float(value) + float(scaled_delta));
            const uint head = ni >> 7;
            const uint d = ni & 127u;
            Out[((ulong)head * M + mi) * 128u + d] = value;
          }
        }
      }
    }
  }
}

#define QK_LORA_KERNEL(NAME, USE_BARRIER)                                      \
kernel void NAME(                                                              \
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],\
    device bfloat *Out [[buffer(2)]], constant uint &M [[buffer(3)]],           \
    constant uint &N [[buffer(4)]], constant uint &Kin [[buffer(5)]],           \
    constant uint &swizzle_log [[buffer(6)]],                                   \
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]], \
    const device float *row_scale [[buffer(9)]],                                \
    const device bfloat *bias [[buffer(10)]],                                   \
    constant uint &has_bias [[buffer(11)]],                                     \
    const device float *col_scale [[buffer(12)]],                               \
    const device bfloat *delta [[buffer(13)]],                                  \
    constant float &delta_scale [[buffer(14)]],                                 \
    const device bfloat *fr [[buffer(15)]],                                     \
    const device float *norm_weight [[buffer(16)]],                             \
    constant uint &R [[buffer(17)]], constant float &eps [[buffer(18)]],        \
    uint2 tgid [[threadgroup_position_in_grid]],                                \
    uint sgid [[simdgroup_index_in_threadgroup]],                               \
    uint lid [[thread_index_in_simdgroup]],                                     \
    uint tid [[thread_index_in_threadgroup]]) {                                 \
  threadgroup QKVEpilogueScratch work;                                          \
  int8_matmul_fused_cs_qk_lora_rms_rope_impl<USE_BARRIER>(                     \
      A, B, Out, M, N, Kin, swizzle_log, tiles_m, tiles_n, row_scale, bias,    \
      has_bias, col_scale, delta, delta_scale, fr, norm_weight, R, eps,         \
      tgid, sgid, lid, tid, &work);                                             \
}

QK_LORA_KERNEL(int8_matmul_fused_cs_qk_lora_rms_rope, true)
QK_LORA_KERNEL(int8_matmul_fused_cs_qk_lora_rms_rope_long, false)

#define V_LORA_KERNEL(NAME, USE_BARRIER)                                       \
kernel void NAME(                                                              \
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],\
    device bfloat *Out [[buffer(2)]], constant uint &M [[buffer(3)]],           \
    constant uint &N [[buffer(4)]], constant uint &Kin [[buffer(5)]],           \
    constant uint &swizzle_log [[buffer(6)]],                                   \
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]], \
    const device float *row_scale [[buffer(9)]],                                \
    const device bfloat *bias [[buffer(10)]],                                   \
    constant uint &has_bias [[buffer(11)]],                                     \
    const device float *col_scale [[buffer(12)]],                               \
    const device bfloat *delta [[buffer(13)]],                                  \
    constant float &delta_scale [[buffer(14)]],                                 \
    uint2 tgid [[threadgroup_position_in_grid]],                                \
    uint sgid [[simdgroup_index_in_threadgroup]],                               \
    uint lid [[thread_index_in_simdgroup]]) {                                   \
  int8_matmul_fused_cs_v_lora_impl<USE_BARRIER>(                               \
      A, B, Out, M, N, Kin, swizzle_log, tiles_m, tiles_n, row_scale, bias,    \
      has_bias, col_scale, delta, delta_scale, tgid, sgid, lid);                \
}

V_LORA_KERNEL(int8_matmul_fused_cs_v_lora, true)
V_LORA_KERNEL(int8_matmul_fused_cs_v_lora_long, false)

kernel void int8_matmul_fused_cs_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    const device float *row_scale [[buffer(9)]],
    const device bfloat *bias [[buffer(10)]],
    constant uint &has_bias [[buffer(11)]],
    const device float *col_scale [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((32 / 1) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<32, 128, 512, 32, 1, 4, true>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_dequant_cs<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                        short(mm * 16), short(nn * 16), M, N,
                                        m_base, n_base, row_scale, col_scale,
                                        bias, has_bias != 0u);
}

// MiniMax H3 FC1-to-FC2 bridge. Each threadgroup consumes one 256-wide pair
// from FC1, writes the exact SwiGLU + convrot result over the now-dead left
// half, and records its maximum. Keeping groups independent preserves the
// high occupancy of the materialized rotation without allocating [M,F].
kernel void i8_swiglu_rot_max_inplace_bf16(
    device bfloat *x [[buffer(0)]], device float *group_max [[buffer(1)]],
    constant uint &M [[buffer(2)]], constant uint &F [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lid [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float work[256];
  threadgroup float tg[8];
  const uint row = tgid.x, g = tgid.y;
  if (row >= M) return;
  device bfloat *xr = x + ulong(row) * (2ul * F);
  const uint i = g * 256u + lid;
  const float av = float(xr[i]);
  const float sil = av / (1.0f + precise::exp(-av));
  const bfloat act = bfloat(sil) * xr[F + i];
  work[lid] = float(act);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1; stride <= 64; stride *= 4) {
    const uint digit = (lid / stride) & 3u;
    const uint base = lid - digit * stride;
    const float a0 = work[base];
    const float a1 = work[base + stride];
    const float a2 = work[base + 2u * stride];
    const float a3 = work[base + 3u * stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value;
    if (digit == 0u)      value =  a0 + a1 + a2 - a3;
    else if (digit == 1u) value =  a0 + a1 - a2 + a3;
    else if (digit == 2u) value =  a0 - a1 + a2 + a3;
    else                  value = -a0 + a1 + a2 + a3;
    work[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const bfloat r = bf16_rte(work[lid] * 0.0625f);
  xr[i] = r;
  const float smax = simd_max(fabs(float(r)));
  if (simd_lane == 0) tg[simd_id] = smax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float m = tg[0];
    for (ushort s = 1; s < 8; ++s) m = fmax(m, tg[s]);
    group_max[ulong(row) * (F >> 8) + g] = m;
  }
}

// Finalize the exact row scale from the per-group maxima, then quantize the
// in-place rotated left half. Only one global BF16 read is required here;
// ordinary row quantization reads it once for max and again for conversion.
kernel void i8_quant_rowwise_premax_strided_bf16(
    const device bfloat *x [[buffer(0)]],
    const device float *group_max [[buffer(1)]], device char *q [[buffer(2)]],
    device float *scales [[buffer(3)]], constant uint &M [[buffer(4)]],
    constant uint &F [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lid [[thread_position_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg[8];
  if (row >= M) return;
  const uint groups = F >> 8;
  float lmax = 0.0f;
  for (uint g = lid; g < groups; g += 256)
    lmax = fmax(lmax, group_max[ulong(row) * groups + g]);
  const float smax = simd_max(lmax);
  if (simd_lane == 0) tg[simd_id] = smax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float m = tg[0];
    for (ushort s = 1; s < 8; ++s) m = fmax(m, tg[s]);
    const float scale = fmax(precise::divide(m, 127.0f), 1e-30f);
    scales[row] = scale;
    tg[0] = scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bfloat scale_cast = bfloat(tg[0]);
  if (float(scale_cast) == 0.0f)
    scale_cast = numeric_limits<bfloat>::min();
  const float scale_math = float(scale_cast);
  const device bfloat *xr = x + ulong(row) * (2ul * F);
  device char *qr = q + ulong(row) * F;
  for (uint i = lid; i < F; i += 256) {
    float value = float(bfloat(precise::divide(float(xr[i]), scale_math)));
    value = clamp(rint(value), -128.0f, 127.0f);
    qr[i] = char(value);
  }
}

// ── Rowwise int8 activation quant (bit-exact vs comfy_kitchen eager) ────────
// Replicates quantize_int8_rowwise(stochastic_rounding=0) step for step:
//   abs_max = x.abs().amax(-1)                    (exact — comparisons only)
//   scale   = clamp(float(abs_max)/127, min=1e-30)  -> returned fp32 scale
//   s_math  = scale.to(x.dtype); s_math==0 -> finfo(x.dtype).tiny
//   q       = int8(clamp(rint(x / s_math), -128, 127))
// The x.dtype division is emulated as float-divide + RTE cast to InT, which is
// how MPS itself lowers half/bfloat arithmetic — verified bit-identical in the
// extension tests. One threadgroup (256 threads, 8 simdgroups) per row.
template <typename InT>
void i8_quant_rowwise_body(const device InT *x, device char *q,
                           device float *scales, uint M, uint K, uint row,
                           ushort lid, ushort simd_lane, ushort simd_id,
                           threadgroup float *tg) {
  if (row >= M) return;
  const device InT *xr = x + ulong(row) * K;
  float local_max = 0.0f;
  for (uint i = lid; i < K; i += 256) {
    local_max = fmax(local_max, fabs(float(xr[i])));
  }
  float smax = simd_max(local_max);
  if (simd_lane == 0) tg[simd_id] = smax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float m = tg[0];
    for (ushort s = 1; s < 8; s++) m = fmax(m, tg[s]);
    // precise::divide: default fast-math lowers `/` to reciprocal-multiply,
    // which is 1-ulp off torch's correctly-rounded fp32 divide.
    float sc = fmax(precise::divide(m, 127.0f), 1e-30f);
    scales[row] = sc;
    tg[0] = sc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  InT s_cast = InT(tg[0]);
  if (float(s_cast) == 0.0f) {
    s_cast = numeric_limits<InT>::min();  // finfo tiny (smallest normal)
  }
  float s_math = float(s_cast);
  device char *qr = q + ulong(row) * K;
  for (uint i = lid; i < K; i += 256) {
    float d = float(InT(precise::divide(float(xr[i]), s_math)));  // x.dtype divide
    d = clamp(rint(d), -128.0f, 127.0f);
    qr[i] = char(d);
  }
}

kernel void i8_quant_rowwise_bf16(
    const device bfloat *x [[buffer(0)]], device char *q [[buffer(1)]],
    device float *scales [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lid [[thread_position_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg[8];
  i8_quant_rowwise_body<bfloat>(x, q, scales, M, K, row, lid, simd_lane,
                                simd_id, tg);
}

kernel void i8_quant_rowwise_f16(
    const device half *x [[buffer(0)]], device char *q [[buffer(1)]],
    device float *scales [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lid [[thread_position_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float tg[8];
  i8_quant_rowwise_body<half>(x, q, scales, M, K, row, lid, simd_lane,
                              simd_id, tg);
}

// Fused o-projection pre-pass for attention outputs whose merged width tiles
// the 256-element convrot groupsize (Ideogram: D=256; MiniMax H3: D=128 and
// each rotation group spans two adjacent heads): reads the attention result in
// its native [B,H,S,D] layout and emits the merged token row [B*S, H*D]
// already convrot-rotated and int8-quantized. Replaces three full passes over
// the activation (head-merge transpose copy, bf16 Hadamard matmul, rowwise
// quant read) with one.
//
// The rotation is the regular Hadamard H = H4^{kron 4} (comfy_kitchen
// _build_hadamard, entries ±2^-4 — exactly representable), applied as 4
// radix-4 butterfly stages in fp32 with a single RTE round to bf16 at the
// end. The eager path (bf16 torch.matmul) also multiplies by exact ±2^-4 and
// accumulates in fp32, so the two differ only in fp32 summation order —
// occasional last-ulp bf16 flips, NOT bit-identical by construction.
// Quant math below is byte-for-byte i8_quant_rowwise_body with K = H*D.
//
// rotate=0 skips the FWHT: the caller already rotated (e.g. via the eager
// bf16 matmul, which is bit-identical in [B,H,S,256] and merged layouts) and
// this kernel is a pure gather + quant — bit-exact by the same argument as
// i8_quant_rowwise (absmax is comparison-only; divide math is replicated).
kernel void i8_merge_rot_quant_bf16(
    const device bfloat *x [[buffer(0)]],   // [B,H,S,D] contiguous
    device char *q [[buffer(1)]],           // [B*S, H*D]
    device float *scales [[buffer(2)]],     // [B*S]
    constant uint &B [[buffer(3)]],
    constant uint &H [[buffer(4)]],
    constant uint &S [[buffer(5)]],
    constant uint &D [[buffer(6)]],
    constant uint &rotate [[buffer(7)]],
    threadgroup bfloat *rot [[threadgroup(0)]],  // H*D rotated values
    uint row [[threadgroup_position_in_grid]],
    ushort lid [[thread_position_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float work[256];
  threadgroup float tg[8];
  if (row >= B * S) return;
  const uint b = row / S, s = row % S;
  const uint K = H * D;
  float lmax = 0.0f;
  const uint groups = K >> 8;
  for (uint g = 0; g < groups; ++g) {
    const uint i = g * 256u + lid;
    const uint h = i / D, d = i - h * D;
    const device bfloat *src = x + ((ulong(b) * H + h) * S + s) * D;
    if (rotate == 0u) {
      const bfloat r = src[d];
      rot[i] = r;
      lmax = fmax(lmax, fabs(float(r)));
      continue;
    }
    work[lid] = float(src[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride <= 64; stride *= 4) {
      const uint digit = (lid / stride) & 3u;
      const uint base = lid - digit * stride;
      const float a0 = work[base];
      const float a1 = work[base + stride];
      const float a2 = work[base + 2u * stride];
      const float a3 = work[base + 3u * stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Row `digit` of H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
      // (symmetric, so row form applies on the output side too).
      float y;
      if (digit == 0u)      y =  a0 + a1 + a2 - a3;
      else if (digit == 1u) y =  a0 + a1 - a2 + a3;
      else if (digit == 2u) y =  a0 - a1 + a2 + a3;
      else                  y = -a0 + a1 + a2 + a3;
      work[lid] = y;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const bfloat r = bf16_rte(work[lid] * 0.0625f);  // 1/sqrt(256), exact 2^-4
    rot[i] = r;
    lmax = fmax(lmax, fabs(float(r)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float smax = simd_max(lmax);
  if (simd_lane == 0) tg[simd_id] = smax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float m = tg[0];
    for (ushort t = 1; t < 8; t++) m = fmax(m, tg[t]);
    float sc = fmax(precise::divide(m, 127.0f), 1e-30f);
    scales[row] = sc;
    tg[0] = sc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bfloat s_cast = bfloat(tg[0]);
  if (float(s_cast) == 0.0f) {
    s_cast = numeric_limits<bfloat>::min();  // finfo tiny (smallest normal)
  }
  float s_math = float(s_cast);
  device char *qr = q + ulong(row) * K;
  for (uint i = lid; i < K; i += 256) {
    float d = float(bfloat(precise::divide(float(rot[i]), s_math)));
    d = clamp(rint(d), -128.0f, 127.0f);
    qr[i] = char(d);
  }
}

// MiniMax H3 fc2 pre-pass: consume the fused [M,2F] fc1 projection and emit
// the SwiGLU activation already convrot-rotated.  One threadgroup handles one
// 256-element rotation group, exposing all 56 groups of an H3 row in parallel;
// a following i8_quant_rowwise pass performs the required cross-group absmax.
// This removes the intermediate unrotated [M,F] activation and the separate
// MPSGraph Hadamard matmul.
//
// SwiGLU matches silu_mul_split_bf16 exactly.  The FWHT is the same numerical
// mode as i8_merge_rot_quant(..., rotate=true): mathematically identical to
// comfy's normalized Hadamard matmul, with possible last-ulp bf16 differences
// from the reassociated summation tree.  Quantization itself is byte-for-byte
// the rowwise reference sequence after this pass.
kernel void i8_swiglu_rot_bf16(
    const device bfloat *x [[buffer(0)]],  // [M,2F]
    device bfloat *rot [[buffer(1)]],       // [M,F]
    constant uint &F [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]], // row, 256-wide group
    ushort lid [[thread_index_in_threadgroup]]) {
  threadgroup float work[256];
  const uint row = tgid.x, g = tgid.y;
  const device bfloat *xr = x + ulong(row) * (2ul * F);
  const uint i = g * 256u + lid;
  const float av = float(xr[i]);
  const float sil = av / (1.0f + precise::exp(-av));
  // Same bf16 operation sequence as silu_mul_split_bf16.
  const bfloat act = bfloat(sil) * xr[F + i];
  work[lid] = float(act);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1; stride <= 64; stride *= 4) {
    const uint digit = (lid / stride) & 3u;
    const uint base = lid - digit * stride;
    const float a0 = work[base];
    const float a1 = work[base + stride];
    const float a2 = work[base + 2u * stride];
    const float a3 = work[base + 3u * stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y;
    if (digit == 0u)      y =  a0 + a1 + a2 - a3;
    else if (digit == 1u) y =  a0 + a1 - a2 + a3;
    else if (digit == 2u) y =  a0 - a1 + a2 + a3;
    else                  y = -a0 + a1 + a2 + a3;
    work[lid] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  rot[ulong(row) * F + i] = bf16_rte(work[lid] * 0.0625f);
}
)MTL";

// ── Host: pipeline cache ─────────────────────────────────────────
static id<MTLComputePipelineState> g_pso_large = nil;
static id<MTLComputePipelineState> g_pso_small = nil;
static id<MTLComputePipelineState> g_pso_fused = nil;
static id<MTLComputePipelineState> g_pso_fused_small = nil;
static id<MTLLibrary> g_lib = nil;
static std::string g_compile_error;

static id<MTLLibrary> build_library() {
  if (g_lib) return g_lib;
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  // Safe math: default fast-math reassociates float chains (observed:
  // C*(cs*rs) -> (C*cs)*rs in the channelwise epilogue), breaking 1-ulp parity
  // with torch's op order. The GEMM core is integer TensorOps — unaffected.
  if (@available(macOS 15.0, *)) {
    opts.mathMode = MTLMathModeSafe;
  } else {
    opts.fastMathEnabled = NO;
  }
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                         options:opts error:&err];
  if (!lib) {
    g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
    TORCH_CHECK(false, "int8 Metal library compile failed: ", g_compile_error);
  }
  g_lib = lib;
  return lib;
}

static id<MTLComputePipelineState> pso_for_name(NSString* name) {
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  id<MTLLibrary> lib = build_library();
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  TORCH_CHECK(fn, "kernel function not found: ", name.UTF8String);
  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "pipeline state creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  return pso;
}

static id<MTLComputePipelineState> pso_for(bool small) {
  if (small && g_pso_small) return g_pso_small;
  if (!small && g_pso_large) return g_pso_large;
  id<MTLComputePipelineState> pso =
      pso_for_name(small ? @"int8_matmul_int32_small" : @"int8_matmul_int32");
  if (small) g_pso_small = pso; else g_pso_large = pso;
  return pso;
}

static id<MTLComputePipelineState> pso_for_fused(bool small) {
  if (small && g_pso_fused_small) return g_pso_fused_small;
  if (!small && g_pso_fused) return g_pso_fused;
  id<MTLComputePipelineState> pso =
      pso_for_name(small ? @"int8_matmul_fused_small" : @"int8_matmul_fused");
  if (small) g_pso_fused_small = pso; else g_pso_fused = pso;
  return pso;
}

// Shared grid/swizzle geometry for both epilogues.
struct Geom {
  bool small;
  uint THREADS, swizzle_log, grid_x, grid_y, tiles_m, tiles_n;
};

static Geom geom_for(int M, int N) {
  Geom g;
  g.small = (M <= 64);
  const uint BM = g.small ? 32u : 128u;
  const uint BN = 128u;
  g.THREADS = g.small ? 128u : 512u;
  g.tiles_m = (M + BM - 1) / BM;
  g.tiles_n = (N + BN - 1) / BN;
  if (g.tiles_m <= 3) g.swizzle_log = 0;
  else if (g.tiles_m <= 6) g.swizzle_log = 1;
  else g.swizzle_log = 2;
  uint tile = 1u << g.swizzle_log;
  g.grid_x = g.tiles_n * tile;
  g.grid_y = (g.tiles_m + tile - 1) / tile;
  return g;
}

// C[M,N] int32 = A[M,K] int8 @ B[N,K]^T int8.  B is the weight in its
// natural [N,K] storage (NT) — no transpose needed.
torch::Tensor i8_matmul2d_nt(torch::Tensor a_i8, torch::Tensor b_i8) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps(), "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar && b_i8.scalar_type() == torch::kChar,
              "inputs must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch: A[M,K], B[N,K]");
  TORCH_CHECK(a_i8.is_contiguous() && b_i8.is_contiguous(), "inputs must be contiguous");

  const int M = (int)a_i8.size(0);
  const int K = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  auto C = torch::empty({(long)M, (long)N}, a_i8.options().dtype(torch::kInt32));

  Geom g = geom_for(M, N);
  id<MTLComputePipelineState> pso = pso_for(g.small);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
  id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, b_i8.storage().data());
  id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
  const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
  const NSUInteger bOff = b_i8.storage_offset() * b_i8.element_size();
  const NSUInteger cOff = C.storage_offset() * C.element_size();
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)K;
  uint swizzle_log = g.swizzle_log, tiles_m = g.tiles_m, tiles_n = g.tiles_n;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:aBuf offset:aOff atIndex:0];
      [enc setBuffer:bBuf offset:bOff atIndex:1];
      [enc setBuffer:cBuf offset:cOff atIndex:2];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:5];
      [enc setBytes:&swizzle_log length:sizeof(uint) atIndex:6];
      [enc setBytes:&tiles_m length:sizeof(uint) atIndex:7];
      [enc setBytes:&tiles_n length:sizeof(uint) atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  // Do NOT COMMIT_AND_WAIT here: the kernel is encoded on torch's MPS stream, so
  // ordering with subsequent torch ops (the rescale that reads C) is preserved on
  // the same command buffer, and torch commits at its normal sync points. Forcing
  // a full GPU sync per call would serialize the whole pipeline (kills async
  // overlap) — that was the cause of the in-model slowdown.
  return C;
}

// Fused: D[M,N] bf16 = (A[M,K] int8 @ B[N,K]^T int8) rescaled per-row by
// row_scale[M] (= weight_scale * activation_row_scale, fp32) + optional
// bias[N] (bf16). The int32 product is consumed in registers and never
// written to global memory — this is the per-call epilogue we skip.
torch::Tensor i8_matmul2d_nt_fused(torch::Tensor a_i8, torch::Tensor b_i8,
                                   torch::Tensor row_scale,
                                   c10::optional<torch::Tensor> bias_opt) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps() && row_scale.is_mps(),
              "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar && b_i8.scalar_type() == torch::kChar,
              "a/b must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2, "a/b must be 2D");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch: A[M,K], B[N,K]");
  TORCH_CHECK(a_i8.is_contiguous() && b_i8.is_contiguous(), "a/b must be contiguous");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat && row_scale.is_contiguous(),
              "row_scale must be contiguous float32");
  TORCH_CHECK(row_scale.numel() == a_i8.size(0), "row_scale must have M elements");

  const int M = (int)a_i8.size(0);
  const int K = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  auto D = torch::empty({(long)M, (long)N}, a_i8.options().dtype(torch::kBFloat16));

  const bool has_bias = bias_opt.has_value();
  torch::Tensor bias = has_bias
      ? bias_opt.value().to(torch::kBFloat16).contiguous()
      : torch::zeros({1}, a_i8.options().dtype(torch::kBFloat16));
  if (has_bias) {
    TORCH_CHECK(bias.is_mps() && bias.numel() == N, "bias must be N MPS elements");
  }

  Geom g = geom_for(M, N);
  id<MTLComputePipelineState> pso = pso_for_fused(g.small);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
  id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, b_i8.storage().data());
  id<MTLBuffer> dBuf = __builtin_bit_cast(id<MTLBuffer>, D.storage().data());
  id<MTLBuffer> sBuf = __builtin_bit_cast(id<MTLBuffer>, row_scale.storage().data());
  id<MTLBuffer> biasBuf = __builtin_bit_cast(id<MTLBuffer>, bias.storage().data());
  const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
  const NSUInteger bOff = b_i8.storage_offset() * b_i8.element_size();
  const NSUInteger dOff = D.storage_offset() * D.element_size();
  const NSUInteger sOff = row_scale.storage_offset() * row_scale.element_size();
  const NSUInteger biasOff = bias.storage_offset() * bias.element_size();
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)K;
  uint swizzle_log = g.swizzle_log, tiles_m = g.tiles_m, tiles_n = g.tiles_n;
  uint has_bias_u = has_bias ? 1u : 0u;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:aBuf offset:aOff atIndex:0];
      [enc setBuffer:bBuf offset:bOff atIndex:1];
      [enc setBuffer:dBuf offset:dOff atIndex:2];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:5];
      [enc setBytes:&swizzle_log length:sizeof(uint) atIndex:6];
      [enc setBytes:&tiles_m length:sizeof(uint) atIndex:7];
      [enc setBytes:&tiles_n length:sizeof(uint) atIndex:8];
      [enc setBuffer:sBuf offset:sOff atIndex:9];
      [enc setBuffer:biasBuf offset:biasOff atIndex:10];
      [enc setBytes:&has_bias_u length:sizeof(uint) atIndex:11];
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  return D;
}

static id<MTLComputePipelineState> g_pso_fused_cs = nil;
static id<MTLComputePipelineState> g_pso_fused_cs_small = nil;
static id<MTLComputePipelineState> g_pso_fused_cs_long = nil;

static id<MTLComputePipelineState> pso_for_fused_cs(bool small,
                                                    bool long_schedule = false) {
  if (long_schedule && !small) {
    if (!g_pso_fused_cs_long)
      g_pso_fused_cs_long = pso_for_name(@"int8_matmul_fused_cs_long");
    return g_pso_fused_cs_long;
  }
  if (small && g_pso_fused_cs_small) return g_pso_fused_cs_small;
  if (!small && g_pso_fused_cs) return g_pso_fused_cs;
  id<MTLComputePipelineState> pso =
      pso_for_name(small ? @"int8_matmul_fused_cs_small" : @"int8_matmul_fused_cs");
  if (small) g_pso_fused_cs_small = pso; else g_pso_fused_cs = pso;
  return pso;
}

static void encode_fused_cs_pass(
    id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
    id<MTLBuffer> aBuf, NSUInteger aOff, id<MTLBuffer> bBuf, NSUInteger bOff,
    id<MTLBuffer> dBuf, NSUInteger dOff, id<MTLBuffer> sBuf, NSUInteger sOff,
    id<MTLBuffer> biasBuf, NSUInteger biasOff, id<MTLBuffer> cBuf,
    NSUInteger cOff, uint M, uint N, uint K, uint has_bias, const Geom& g) {
  uint swizzle_log = g.swizzle_log, tiles_m = g.tiles_m, tiles_n = g.tiles_n;
  [enc setComputePipelineState:pso];
  [enc setBuffer:aBuf offset:aOff atIndex:0];
  [enc setBuffer:bBuf offset:bOff atIndex:1];
  [enc setBuffer:dBuf offset:dOff atIndex:2];
  [enc setBytes:&M length:sizeof(uint) atIndex:3];
  [enc setBytes:&N length:sizeof(uint) atIndex:4];
  [enc setBytes:&K length:sizeof(uint) atIndex:5];
  [enc setBytes:&swizzle_log length:sizeof(uint) atIndex:6];
  [enc setBytes:&tiles_m length:sizeof(uint) atIndex:7];
  [enc setBytes:&tiles_n length:sizeof(uint) atIndex:8];
  [enc setBuffer:sBuf offset:sOff atIndex:9];
  [enc setBuffer:biasBuf offset:biasOff atIndex:10];
  [enc setBytes:&has_bias length:sizeof(uint) atIndex:11];
  [enc setBuffer:cBuf offset:cOff atIndex:12];
  [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
      threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
}

// Channelwise-fused: D[M,N] bf16 = (A@B^T) * (col_scale[N] * row_scale[M]) +
// optional bias[N]. For per-output-channel weight scales ([N,1] checkpoints —
// Z-Image / Krea2 convrot), where the scalar-fused epilogue can't apply.
torch::Tensor i8_matmul2d_nt_fused_cs(torch::Tensor a_i8, torch::Tensor b_i8,
                                      torch::Tensor row_scale, torch::Tensor col_scale,
                                      c10::optional<torch::Tensor> bias_opt,
                                      bool split_tail, bool long_schedule) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps() && row_scale.is_mps() && col_scale.is_mps(),
              "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar && b_i8.scalar_type() == torch::kChar,
              "a/b must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2, "a/b must be 2D");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch: A[M,K], B[N,K]");
  TORCH_CHECK(a_i8.is_contiguous() && b_i8.is_contiguous(), "a/b must be contiguous");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat && row_scale.is_contiguous(),
              "row_scale must be contiguous float32");
  TORCH_CHECK(col_scale.scalar_type() == torch::kFloat && col_scale.is_contiguous(),
              "col_scale must be contiguous float32");
  TORCH_CHECK(row_scale.numel() == a_i8.size(0), "row_scale must have M elements");
  TORCH_CHECK(col_scale.numel() == b_i8.size(0), "col_scale must have N elements");

  const int M = (int)a_i8.size(0);
  const int K = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  auto D = torch::empty({(long)M, (long)N}, a_i8.options().dtype(torch::kBFloat16));

  const bool has_bias = bias_opt.has_value();
  torch::Tensor bias = has_bias
      ? bias_opt.value().to(torch::kBFloat16).contiguous()
      : torch::zeros({1}, a_i8.options().dtype(torch::kBFloat16));
  if (has_bias) {
    TORCH_CHECK(bias.is_mps() && bias.numel() == N, "bias must be N MPS elements");
  }

  // A 128-row main tile is intentionally wide enough to saturate the GPU,
  // but H3's M=5378 leaves only two valid rows in its final tile. Optionally
  // encode the aligned prefix with the main kernel and the <=64-row suffix
  // with the 32-row kernel. Both dispatches share one command encoder, so
  // arithmetic and stream ordering are unchanged and launch overhead is low.
  const int tail = M % 128;
  const bool do_split = split_tail && M > 128 && tail > 0 && tail <= 64;
  const int main_m = do_split ? M - tail : M;
  const bool use_long = long_schedule && main_m >= 8192;
  Geom main_g = geom_for(main_m, N);
  Geom tail_g = do_split ? geom_for(tail, N) : main_g;
  id<MTLComputePipelineState> main_pso =
      pso_for_fused_cs(main_g.small, use_long);
  id<MTLComputePipelineState> tail_pso = do_split
      ? pso_for_fused_cs(tail_g.small) : main_pso;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
  id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, b_i8.storage().data());
  id<MTLBuffer> dBuf = __builtin_bit_cast(id<MTLBuffer>, D.storage().data());
  id<MTLBuffer> sBuf = __builtin_bit_cast(id<MTLBuffer>, row_scale.storage().data());
  id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, col_scale.storage().data());
  id<MTLBuffer> biasBuf = __builtin_bit_cast(id<MTLBuffer>, bias.storage().data());
  const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
  const NSUInteger bOff = b_i8.storage_offset() * b_i8.element_size();
  const NSUInteger dOff = D.storage_offset() * D.element_size();
  const NSUInteger sOff = row_scale.storage_offset() * row_scale.element_size();
  const NSUInteger cOff = col_scale.storage_offset() * col_scale.element_size();
  const NSUInteger biasOff = bias.storage_offset() * bias.element_size();
  uint Nu = (uint)N, Ku = (uint)K;
  uint has_bias_u = has_bias ? 1u : 0u;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      encode_fused_cs_pass(
          enc, main_pso, aBuf, aOff, bBuf, bOff, dBuf, dOff, sBuf, sOff,
          biasBuf, biasOff, cBuf, cOff, (uint)main_m, Nu, Ku, has_bias_u,
          main_g);
      if (do_split) {
        const NSUInteger aTail = aOff + (NSUInteger)main_m * K;
        const NSUInteger dTail = dOff + (NSUInteger)main_m * N * 2;
        const NSUInteger sTail = sOff + (NSUInteger)main_m * sizeof(float);
        encode_fused_cs_pass(
            enc, tail_pso, aBuf, aTail, bBuf, bOff, dBuf, dTail, sBuf, sTail,
            biasBuf, biasOff, cBuf, cOff, (uint)tail, Nu, Ku, has_bias_u,
            tail_g);
      }
    }
  });
  return D;
}

// H3 QKV specialization: identical INT8 GEMM and channelwise bf16 epilogue,
// with [M,3*H*D] columns scattered directly to [1,H,M,D] Q/K/V outputs.
std::vector<torch::Tensor> i8_matmul2d_nt_fused_cs_qkv(
    torch::Tensor a_i8, torch::Tensor b_i8, torch::Tensor row_scale,
    torch::Tensor col_scale, c10::optional<torch::Tensor> bias_opt,
    int64_t H, int64_t D, bool long_schedule) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps() && row_scale.is_mps()
              && col_scale.is_mps(), "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar
              && b_i8.scalar_type() == torch::kChar, "a/b must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2
              && a_i8.is_contiguous() && b_i8.is_contiguous(),
              "a/b must be contiguous 2D tensors");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat
              && col_scale.scalar_type() == torch::kFloat
              && row_scale.is_contiguous() && col_scale.is_contiguous(),
              "scales must be contiguous float32");
  const int M = (int)a_i8.size(0), Kin = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  TORCH_CHECK(M > 64, "H3 QKV specialization requires M > 64");
  TORCH_CHECK(H > 0 && D > 0 && N == 3 * H * D,
              "weight width must equal 3*H*D");
  TORCH_CHECK(row_scale.numel() == M && col_scale.numel() == N,
              "scale shape mismatch");

  auto opts = a_i8.options().dtype(torch::kBFloat16);
  auto Q = torch::empty({1L, H, (long)M, D}, opts);
  auto Kout = torch::empty({1L, H, (long)M, D}, opts);
  auto V = torch::empty({1L, H, (long)M, D}, opts);
  const bool has_bias = bias_opt.has_value();
  torch::Tensor bias = has_bias
      ? bias_opt.value().to(torch::kBFloat16).contiguous()
      : torch::zeros({1}, opts);
  if (has_bias)
    TORCH_CHECK(bias.is_mps() && bias.numel() == N,
                "bias must have N MPS elements");

  Geom g = geom_for(M, N);
  TORCH_CHECK(!g.small, "unexpected small QKV geometry");
  const bool use_long = long_schedule && M >= 8192;
  id<MTLComputePipelineState> pso = pso_for_name(
      use_long ? @"int8_matmul_fused_cs_qkv_long"
               : @"int8_matmul_fused_cs_qkv");
  MPSStream* stream = getCurrentMPSStream();
  torch::Tensor tensors[8] = {
      a_i8, b_i8, Q, Kout, V, row_scale, bias, col_scale};
  std::array<id<MTLBuffer>, 8> bufs;
  std::array<NSUInteger, 8> offs;
  for (int i = 0; i < 8; ++i) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, tensors[i].storage().data());
    offs[i] = tensors[i].storage_offset() * tensors[i].element_size();
  }
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)Kin;
  uint swizzle = g.swizzle_log, tm = g.tiles_m, tn = g.tiles_n;
  uint hb = has_bias ? 1u : 0u, Hu = (uint)H, Du = (uint)D;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      for (int i = 0; i < 5; ++i)
        [enc setBuffer:bufs[i] offset:offs[i] atIndex:i];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:5];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:6];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:7];
      [enc setBytes:&swizzle length:sizeof(uint) atIndex:8];
      [enc setBytes:&tm length:sizeof(uint) atIndex:9];
      [enc setBytes:&tn length:sizeof(uint) atIndex:10];
      [enc setBuffer:bufs[5] offset:offs[5] atIndex:11];
      [enc setBuffer:bufs[6] offset:offs[6] atIndex:12];
      [enc setBytes:&hb length:sizeof(uint) atIndex:13];
      [enc setBuffer:bufs[7] offset:offs[7] atIndex:14];
      [enc setBytes:&Hu length:sizeof(uint) atIndex:15];
      [enc setBytes:&Du length:sizeof(uint) atIndex:16];
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  return {Q, Kout, V};
}

// H3 D=128 specialization that consumes the dequantized Q/K projection tile
// directly in the epilogue, applying affine RMSNorm and partial split-half
// RoPE before the values ever reach device memory. V retains the normal
// direct-scatter epilogue. The arithmetic mirrors dit_ext's existing kernel.
std::vector<torch::Tensor> i8_matmul2d_nt_fused_cs_qkv_rms_rope(
    torch::Tensor a_i8, torch::Tensor b_i8, torch::Tensor row_scale,
    torch::Tensor col_scale, c10::optional<torch::Tensor> bias_opt,
    torch::Tensor fr, torch::Tensor qw, torch::Tensor kw,
    int64_t H, int64_t D, double eps, bool long_schedule) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps() && row_scale.is_mps()
              && col_scale.is_mps() && fr.is_mps() && qw.is_mps()
              && kw.is_mps(), "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar
              && b_i8.scalar_type() == torch::kChar, "a/b must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2
              && a_i8.is_contiguous() && b_i8.is_contiguous(),
              "a/b must be contiguous 2D tensors");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat
              && col_scale.scalar_type() == torch::kFloat
              && row_scale.is_contiguous() && col_scale.is_contiguous(),
              "scales must be contiguous float32");
  TORCH_CHECK(fr.scalar_type() == torch::kBFloat16 && fr.is_contiguous()
              && fr.dim() >= 3,
              "freqs must be contiguous bf16");
  TORCH_CHECK(qw.scalar_type() == torch::kFloat
              && kw.scalar_type() == torch::kFloat
              && qw.is_contiguous() && kw.is_contiguous(),
              "qw/kw must be contiguous float32");

  const int M = (int)a_i8.size(0), Kin = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  const int64_t R = fr.size(-3);
  TORCH_CHECK(M > 64, "H3 QKV specialization requires M > 64");
  TORCH_CHECK(H > 0 && D == 128 && N == 3 * H * D,
              "fused QKV norm/RoPE requires D=128 and width=3*H*D");
  TORCH_CHECK(row_scale.numel() == M && col_scale.numel() == N,
              "scale shape mismatch");
  TORCH_CHECK(R > 0 && 2 * R <= D && fr.numel() == M * R * 4,
              "freqs shape mismatch");
  TORCH_CHECK(qw.numel() == D && kw.numel() == D,
              "qw/kw must contain D elements");

  auto opts = a_i8.options().dtype(torch::kBFloat16);
  auto Q = torch::empty({1L, H, (long)M, D}, opts);
  auto Kout = torch::empty({1L, H, (long)M, D}, opts);
  auto V = torch::empty({1L, H, (long)M, D}, opts);
  const bool has_bias = bias_opt.has_value();
  torch::Tensor bias = has_bias
      ? bias_opt.value().to(torch::kBFloat16).contiguous()
      : torch::zeros({1}, opts);
  if (has_bias)
    TORCH_CHECK(bias.is_mps() && bias.numel() == N,
                "bias must have N MPS elements");

  Geom g = geom_for(M, N);
  TORCH_CHECK(!g.small && g.THREADS == 512,
              "unexpected fused QKV geometry");
  const bool use_long = long_schedule && M >= 8192;
  id<MTLComputePipelineState> pso = pso_for_name(
      use_long ? @"int8_matmul_fused_cs_qkv_rms_rope_long"
               : @"int8_matmul_fused_cs_qkv_rms_rope");
  MPSStream* stream = getCurrentMPSStream();
  torch::Tensor tensors[11] = {
      a_i8, b_i8, Q, Kout, V, row_scale, bias, col_scale, fr, qw, kw};
  std::array<id<MTLBuffer>, 11> bufs;
  std::array<NSUInteger, 11> offs;
  for (int i = 0; i < 11; ++i) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, tensors[i].storage().data());
    offs[i] = tensors[i].storage_offset() * tensors[i].element_size();
  }
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)Kin;
  uint swizzle = g.swizzle_log, tm = g.tiles_m, tn = g.tiles_n;
  uint hb = has_bias ? 1u : 0u, Hu = (uint)H, Ru = (uint)R;
  float ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      for (int i = 0; i < 5; ++i)
        [enc setBuffer:bufs[i] offset:offs[i] atIndex:i];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:5];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:6];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:7];
      [enc setBytes:&swizzle length:sizeof(uint) atIndex:8];
      [enc setBytes:&tm length:sizeof(uint) atIndex:9];
      [enc setBytes:&tn length:sizeof(uint) atIndex:10];
      [enc setBuffer:bufs[5] offset:offs[5] atIndex:11];
      [enc setBuffer:bufs[6] offset:offs[6] atIndex:12];
      [enc setBytes:&hb length:sizeof(uint) atIndex:13];
      [enc setBuffer:bufs[7] offset:offs[7] atIndex:14];
      [enc setBytes:&Hu length:sizeof(uint) atIndex:15];
      [enc setBuffer:bufs[8] offset:offs[8] atIndex:16];
      [enc setBuffer:bufs[9] offset:offs[9] atIndex:17];
      [enc setBuffer:bufs[10] offset:offs[10] atIndex:18];
      [enc setBytes:&Ru length:sizeof(uint) atIndex:19];
      [enc setBytes:&ef length:sizeof(float) atIndex:20];
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  return {Q, Kout, V};
}

torch::Tensor i8_matmul2d_nt_fused_cs_qkv_lora_part(
    torch::Tensor a_i8, torch::Tensor b_i8, torch::Tensor row_scale,
    torch::Tensor col_scale, c10::optional<torch::Tensor> bias_opt,
    torch::Tensor delta, torch::Tensor fr, torch::Tensor norm_weight,
    int64_t H, int64_t D, int64_t kind, double delta_scale,
    double eps, bool long_schedule) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps() && row_scale.is_mps()
              && col_scale.is_mps() && delta.is_mps() && fr.is_mps()
              && norm_weight.is_mps(), "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar
              && b_i8.scalar_type() == torch::kChar, "a/b must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2
              && a_i8.is_contiguous() && b_i8.is_contiguous(),
              "a/b must be contiguous 2D tensors");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat
              && col_scale.scalar_type() == torch::kFloat
              && row_scale.is_contiguous() && col_scale.is_contiguous(),
              "scales must be contiguous float32");
  TORCH_CHECK(delta.scalar_type() == torch::kBFloat16
              && delta.is_contiguous() && delta.dim() == 2,
              "LoRA delta must be contiguous bf16 [M,H*D]");
  TORCH_CHECK(fr.scalar_type() == torch::kBFloat16 && fr.is_contiguous()
              && fr.dim() >= 3,
              "freqs must be contiguous bf16");
  TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat
              && norm_weight.is_contiguous(),
              "norm weight must be contiguous float32");

  const int M = (int)a_i8.size(0), Kin = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  const int64_t R = fr.size(-3);
  TORCH_CHECK(M > 64, "H3 QKV specialization requires M > 64");
  TORCH_CHECK(H > 0 && D == 128 && N == H * D,
              "LoRA QKV partition requires D=128 and width=H*D");
  TORCH_CHECK(kind >= 0 && kind <= 2, "kind must be Q=0, K=1 or V=2");
  TORCH_CHECK(row_scale.numel() == M && col_scale.numel() == N,
              "scale shape mismatch");
  TORCH_CHECK(delta.size(0) == M && delta.size(1) == N,
              "LoRA delta shape mismatch");
  TORCH_CHECK(R > 0 && 2 * R <= D && fr.numel() == M * R * 4,
              "freqs shape mismatch");
  TORCH_CHECK(norm_weight.numel() == D,
              "norm weight must contain D elements");

  auto opts = a_i8.options().dtype(torch::kBFloat16);
  auto Out = torch::empty({1L, H, (long)M, D}, opts);
  const bool has_bias = bias_opt.has_value();
  torch::Tensor bias = has_bias
      ? bias_opt.value().to(torch::kBFloat16).contiguous()
      : torch::zeros({1}, opts);
  if (has_bias)
    TORCH_CHECK(bias.is_mps() && bias.numel() == N,
                "bias must have H*D MPS elements");

  Geom g = geom_for(M, N);
  TORCH_CHECK(!g.small && g.THREADS == 512,
              "unexpected LoRA QKV geometry");
  const bool use_long = long_schedule && M >= 8192;
  NSString *name;
  if (kind == 2) {
    name = use_long ? @"int8_matmul_fused_cs_v_lora_long"
                    : @"int8_matmul_fused_cs_v_lora";
  } else {
    name = use_long ? @"int8_matmul_fused_cs_qk_lora_rms_rope_long"
                    : @"int8_matmul_fused_cs_qk_lora_rms_rope";
  }
  id<MTLComputePipelineState> pso = pso_for_name(name);
  MPSStream* stream = getCurrentMPSStream();
  torch::Tensor tensors[9] = {
      a_i8, b_i8, Out, row_scale, bias, col_scale,
      delta, fr, norm_weight};
  std::array<id<MTLBuffer>, 9> bufs;
  std::array<NSUInteger, 9> offs;
  for (int i = 0; i < 9; ++i) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, tensors[i].storage().data());
    offs[i] = tensors[i].storage_offset() * tensors[i].element_size();
  }
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)Kin;
  uint swizzle = g.swizzle_log, tm = g.tiles_m, tn = g.tiles_n;
  uint hb = has_bias ? 1u : 0u, Ru = (uint)R;
  float ds = (float)delta_scale, ef = (float)eps;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:bufs[0] offset:offs[0] atIndex:0];
      [enc setBuffer:bufs[1] offset:offs[1] atIndex:1];
      [enc setBuffer:bufs[2] offset:offs[2] atIndex:2];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:5];
      [enc setBytes:&swizzle length:sizeof(uint) atIndex:6];
      [enc setBytes:&tm length:sizeof(uint) atIndex:7];
      [enc setBytes:&tn length:sizeof(uint) atIndex:8];
      [enc setBuffer:bufs[3] offset:offs[3] atIndex:9];
      [enc setBuffer:bufs[4] offset:offs[4] atIndex:10];
      [enc setBytes:&hb length:sizeof(uint) atIndex:11];
      [enc setBuffer:bufs[5] offset:offs[5] atIndex:12];
      [enc setBuffer:bufs[6] offset:offs[6] atIndex:13];
      [enc setBytes:&ds length:sizeof(float) atIndex:14];
      if (kind != 2) {
        [enc setBuffer:bufs[7] offset:offs[7] atIndex:15];
        [enc setBuffer:bufs[8] offset:offs[8] atIndex:16];
        [enc setBytes:&Ru length:sizeof(uint) atIndex:17];
        [enc setBytes:&ef length:sizeof(float) atIndex:18];
      }
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  return Out;
}

// Rowwise int8 activation quant: x[M,K] (bf16/f16) -> (q[M,K] int8,
// scales[M,1] fp32). Bit-exact vs comfy_kitchen quantize_int8_rowwise(sr=0).
static id<MTLComputePipelineState> g_pso_quant_bf16 = nil;
static id<MTLComputePipelineState> g_pso_quant_f16 = nil;
static id<MTLComputePipelineState> g_pso_mlp_bridge = nil;
static id<MTLComputePipelineState> g_pso_mlp_bridge_quant = nil;

std::vector<torch::Tensor> i8_quant_rowwise(torch::Tensor x) {
  TORCH_CHECK(x.is_mps(), "x must be on mps");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  const bool is_bf16 = x.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(is_bf16 || x.scalar_type() == torch::kHalf,
              "x must be bf16 or f16");

  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  auto q = torch::empty({(long)M, (long)K}, x.options().dtype(torch::kChar));
  auto scales = torch::empty({(long)M, 1L}, x.options().dtype(torch::kFloat));

  id<MTLComputePipelineState>& slot = is_bf16 ? g_pso_quant_bf16 : g_pso_quant_f16;
  if (!slot) {
    slot = pso_for_name(is_bf16 ? @"i8_quant_rowwise_bf16" : @"i8_quant_rowwise_f16");
  }
  id<MTLComputePipelineState> pso = slot;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> xBuf = __builtin_bit_cast(id<MTLBuffer>, x.storage().data());
  id<MTLBuffer> qBuf = __builtin_bit_cast(id<MTLBuffer>, q.storage().data());
  id<MTLBuffer> sBuf = __builtin_bit_cast(id<MTLBuffer>, scales.storage().data());
  const NSUInteger xOff = x.storage_offset() * x.element_size();
  const NSUInteger qOff = q.storage_offset() * q.element_size();
  const NSUInteger sOff = scales.storage_offset() * scales.element_size();
  uint Mu = (uint)M, Ku = (uint)K;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:xBuf offset:xOff atIndex:0];
      [enc setBuffer:qBuf offset:qOff atIndex:1];
      [enc setBuffer:sBuf offset:sOff atIndex:2];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return {q, scales};
}

// H3 FC1/SwiGLU/FC2 bridge front half. Returns FC2's rotated, row-quantized
// activation without materializing a global [M,F] rotated tensor.
std::vector<torch::Tensor> i8_mlp_bridge_prequant(
    torch::Tensor a_i8, torch::Tensor fc1_weight,
    torch::Tensor row_scale, torch::Tensor fc1_col_scale,
    c10::optional<torch::Tensor> fc1_bias_opt, bool long_schedule,
    int64_t tile_rows_arg) {
  TORCH_CHECK(a_i8.is_mps() && fc1_weight.is_mps()
              && row_scale.is_mps() && fc1_col_scale.is_mps(),
              "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar
              && fc1_weight.scalar_type() == torch::kChar,
              "activation and FC1 weight must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && fc1_weight.dim() == 2
              && a_i8.is_contiguous() && fc1_weight.is_contiguous(),
              "activation and FC1 weight must be contiguous 2D tensors");
  TORCH_CHECK(a_i8.size(1) == fc1_weight.size(1), "FC1 K mismatch");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat
              && fc1_col_scale.scalar_type() == torch::kFloat
              && row_scale.is_contiguous() && fc1_col_scale.is_contiguous(),
              "scales must be contiguous float32");

  const int M = (int)a_i8.size(0), K = (int)a_i8.size(1);
  const int W = (int)fc1_weight.size(0);
  TORCH_CHECK(M > 64 && W > 0 && (W % 512) == 0,
              "bridge requires M > 64 and F divisible by 256");
  const int F = W / 2;
  TORCH_CHECK(row_scale.numel() == M && fc1_col_scale.numel() == W,
              "scale shape mismatch");
  TORCH_CHECK(tile_rows_arg == 0 || tile_rows_arg >= 128,
              "tile_rows must be zero or at least 128");
  const int requested_rows = tile_rows_arg == 0
      ? M : (int)std::min<int64_t>(M, tile_rows_arg);
  const int tile_rows = requested_rows >= 128
      ? (requested_rows / 128) * 128 : requested_rows;

  auto bf16_opts = a_i8.options().dtype(torch::kBFloat16);
  auto wide = torch::empty({(long)tile_rows, (long)W}, bf16_opts);
  auto q = torch::empty({(long)M, (long)F}, a_i8.options());
  auto scales = torch::empty(
      {(long)M, 1L}, a_i8.options().dtype(torch::kFloat));
  auto group_max = torch::empty(
      {(long)tile_rows, (long)(F / 256)},
      a_i8.options().dtype(torch::kFloat));
  const bool has_bias = fc1_bias_opt.has_value();
  torch::Tensor bias = has_bias
      ? fc1_bias_opt.value().to(torch::kBFloat16).contiguous()
      : torch::zeros({1}, bf16_opts);
  if (has_bias)
    TORCH_CHECK(bias.is_mps() && bias.numel() == W,
                "FC1 bias must have 2F MPS elements");

  const bool use_long = long_schedule && M >= 8192;
  Geom tile_g = geom_for(tile_rows, W);
  const int remainder = M % tile_rows;
  const int remainder_tail = remainder % 128;
  const bool split_remainder = remainder > 128 && remainder_tail > 0
      && remainder_tail <= 64;
  const int remainder_main = split_remainder
      ? remainder - remainder_tail : remainder;
  Geom remainder_g = remainder
      ? geom_for(remainder_main, W) : tile_g;
  Geom remainder_tail_g = split_remainder
      ? geom_for(remainder_tail, W) : remainder_g;
  id<MTLComputePipelineState> tile_pso =
      pso_for_fused_cs(tile_g.small, use_long && !tile_g.small);
  id<MTLComputePipelineState> remainder_pso = remainder
      ? pso_for_fused_cs(
            remainder_g.small, use_long && !remainder_g.small)
      : tile_pso;
  id<MTLComputePipelineState> remainder_tail_pso = split_remainder
      ? pso_for_fused_cs(remainder_tail_g.small) : remainder_pso;
  if (!g_pso_mlp_bridge)
    g_pso_mlp_bridge = pso_for_name(@"i8_swiglu_rot_max_inplace_bf16");
  if (!g_pso_mlp_bridge_quant)
    g_pso_mlp_bridge_quant =
        pso_for_name(@"i8_quant_rowwise_premax_strided_bf16");

  MPSStream* stream = getCurrentMPSStream();
  torch::Tensor tensors[9] = {
      a_i8, fc1_weight, wide, row_scale,
      bias, fc1_col_scale, q, scales, group_max};
  std::array<id<MTLBuffer>, 9> bufs;
  std::array<NSUInteger, 9> offs;
  for (int i = 0; i < 9; ++i) {
    bufs[i] = __builtin_bit_cast(id<MTLBuffer>, tensors[i].storage().data());
    offs[i] = tensors[i].storage_offset() * tensors[i].element_size();
  }
  uint Wu = (uint)W, Fu = (uint)F, Ku = (uint)K;
  uint has_bias_u = has_bias ? 1u : 0u;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      for (int row_base = 0; row_base < M; row_base += tile_rows) {
        const int block_rows = std::min(tile_rows, M - row_base);
        const bool full_tile = block_rows == tile_rows;
        const Geom& block_g = full_tile ? tile_g : remainder_g;
        id<MTLComputePipelineState> block_pso =
            full_tile ? tile_pso : remainder_pso;
        const int gemm_rows = (!full_tile && split_remainder)
            ? remainder_main : block_rows;
        const NSUInteger a_block =
            offs[0] + (NSUInteger)row_base * K * sizeof(int8_t);
        const NSUInteger s_block =
            offs[3] + (NSUInteger)row_base * sizeof(float);
        encode_fused_cs_pass(
            enc, block_pso, bufs[0], a_block, bufs[1], offs[1],
            bufs[2], offs[2], bufs[3], s_block, bufs[4], offs[4],
            bufs[5], offs[5], (uint)gemm_rows, Wu, Ku, has_bias_u,
            block_g);
        if (!full_tile && split_remainder) {
          const NSUInteger a_tail =
              a_block + (NSUInteger)remainder_main * K * sizeof(int8_t);
          const NSUInteger d_tail =
              offs[2] + (NSUInteger)remainder_main * W * sizeof(uint16_t);
          const NSUInteger s_tail =
              s_block + (NSUInteger)remainder_main * sizeof(float);
          encode_fused_cs_pass(
              enc, remainder_tail_pso, bufs[0], a_tail, bufs[1], offs[1],
              bufs[2], d_tail, bufs[3], s_tail, bufs[4], offs[4],
              bufs[5], offs[5], (uint)remainder_tail, Wu, Ku, has_bias_u,
              remainder_tail_g);
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        uint block_mu = (uint)block_rows;
        [enc setComputePipelineState:g_pso_mlp_bridge];
        [enc setBuffer:bufs[2] offset:offs[2] atIndex:0];
        [enc setBuffer:bufs[8] offset:offs[8] atIndex:1];
        [enc setBytes:&block_mu length:sizeof(uint) atIndex:2];
        [enc setBytes:&Fu length:sizeof(uint) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)block_rows,
                                              (NSUInteger)(F / 256), 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        const NSUInteger q_block =
            offs[6] + (NSUInteger)row_base * F * sizeof(int8_t);
        const NSUInteger scale_block =
            offs[7] + (NSUInteger)row_base * sizeof(float);
        [enc setComputePipelineState:g_pso_mlp_bridge_quant];
        [enc setBuffer:bufs[2] offset:offs[2] atIndex:0];
        [enc setBuffer:bufs[8] offset:offs[8] atIndex:1];
        [enc setBuffer:bufs[6] offset:q_block atIndex:2];
        [enc setBuffer:bufs[7] offset:scale_block atIndex:3];
        [enc setBytes:&block_mu length:sizeof(uint) atIndex:4];
        [enc setBytes:&Fu length:sizeof(uint) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)block_rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (row_base + block_rows < M)
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    }
  });
  return {q, scales};
}

// Fused head-merge + convrot rotation + rowwise quant: x[B,H,S,D] bf16,
// where H*D tiles 256-wide rotation groups, ->
// (q[B*S, H*D] int8, scales[B*S,1] fp32). One threadgroup per token row.
static id<MTLComputePipelineState> g_pso_mrq = nil;
static id<MTLComputePipelineState> g_pso_swiglu_rot = nil;

std::vector<torch::Tensor> i8_merge_rot_quant(torch::Tensor x, bool rotate) {
  TORCH_CHECK(x.is_mps(), "x must be on mps");
  TORCH_CHECK(x.dim() == 4, "x must be [B,H,S,D]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  const int Bv = (int)x.size(0), Hv = (int)x.size(1);
  const int Sv = (int)x.size(2), Dv = (int)x.size(3);
  const int Kv = Hv * Dv;
  TORCH_CHECK(Dv > 0 && Dv <= 256 && (Kv % 256) == 0,
              "merged width must tile 256-element convrot groups");
  TORCH_CHECK(Hv >= 1 && Kv * 2 <= 30 * 1024,
              "H*D too large for threadgroup memory");
  const long M = (long)Bv * Sv;
  auto q = torch::empty({M, (long)Kv}, x.options().dtype(torch::kChar));
  auto scales = torch::empty({M, 1L}, x.options().dtype(torch::kFloat));

  if (!g_pso_mrq) g_pso_mrq = pso_for_name(@"i8_merge_rot_quant_bf16");
  id<MTLComputePipelineState> pso = g_pso_mrq;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> xBuf = __builtin_bit_cast(id<MTLBuffer>, x.storage().data());
  id<MTLBuffer> qBuf = __builtin_bit_cast(id<MTLBuffer>, q.storage().data());
  id<MTLBuffer> sBuf = __builtin_bit_cast(id<MTLBuffer>, scales.storage().data());
  const NSUInteger xOff = x.storage_offset() * x.element_size();
  const NSUInteger qOff = q.storage_offset() * q.element_size();
  const NSUInteger sOff = scales.storage_offset() * scales.element_size();
  uint Bu = (uint)Bv, Hu = (uint)Hv, Su = (uint)Sv, Du = (uint)Dv;
  uint rotu = rotate ? 1u : 0u;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:xBuf offset:xOff atIndex:0];
      [enc setBuffer:qBuf offset:qOff atIndex:1];
      [enc setBuffer:sBuf offset:sOff atIndex:2];
      [enc setBytes:&Bu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Hu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Su length:sizeof(uint) atIndex:5];
      [enc setBytes:&Du length:sizeof(uint) atIndex:6];
      [enc setBytes:&rotu length:sizeof(uint) atIndex:7];
      [enc setThreadgroupMemoryLength:(NSUInteger)(Kv * 2) atIndex:0];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return {q, scales};
}

torch::Tensor i8_swiglu_rot(torch::Tensor x) {
  TORCH_CHECK(x.is_mps() && x.scalar_type() == torch::kBFloat16,
              "x must be bf16 on mps");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(),
              "x must be contiguous [M,2F]");
  const int M = (int)x.size(0), W = (int)x.size(1);
  TORCH_CHECK((W % 512) == 0, "F must be a multiple of convrot groupsize 256");
  const int F = W / 2;
  auto rot = torch::empty({(long)M, (long)F}, x.options());
  if (!g_pso_swiglu_rot)
    g_pso_swiglu_rot = pso_for_name(@"i8_swiglu_rot_bf16");
  id<MTLComputePipelineState> pso = g_pso_swiglu_rot;
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> xBuf = __builtin_bit_cast(id<MTLBuffer>, x.storage().data());
  id<MTLBuffer> rBuf = __builtin_bit_cast(id<MTLBuffer>, rot.storage().data());
  const NSUInteger xOff = x.storage_offset() * x.element_size();
  const NSUInteger rOff = rot.storage_offset() * rot.element_size();
  uint Fu = (uint)F;
  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:xBuf offset:xOff atIndex:0];
      [enc setBuffer:rBuf offset:rOff atIndex:1];
      [enc setBytes:&Fu length:sizeof(uint) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M,
                                            (NSUInteger)(F / 256), 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return rot;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("i8_mlp_bridge_prequant", &i8_mlp_bridge_prequant,
        "H3 FC1 + fused SwiGLU/convrot/row-quant bridge: returns the "
        "row-quantized FC2 activation without a global rotated BF16 tensor",
        pybind11::arg("a_i8"), pybind11::arg("fc1_weight"),
        pybind11::arg("row_scale"), pybind11::arg("fc1_col_scale"),
        pybind11::arg("fc1_bias") = c10::optional<torch::Tensor>(),
        pybind11::arg("long_schedule") = true,
        pybind11::arg("tile_rows") = 1024);
  m.def("i8_swiglu_rot", &i8_swiglu_rot,
        "Fused H3 SwiGLU + 256-wide convrot FWHT: "
        "x[M,2F] bf16 -> rotated[M,F] bf16");
  m.def("i8_merge_rot_quant", &i8_merge_rot_quant,
        "Fused head-merge (+ optional convrot FWHT) + rowwise int8 quant: "
        "x[B,H,S,D] bf16 -> (q[B*S,H*D] int8, scales[B*S,1] fp32); "
        "rotate=False = pure gather+quant (bit-exact) on pre-rotated input",
        pybind11::arg("x"), pybind11::arg("rotate") = true);
  m.def("i8_matmul2d_nt_fused_cs", &i8_matmul2d_nt_fused_cs,
        "Channelwise-fused NT int8 matmul: D = (A@B^T)*(col_scale[N]*row_scale[M]) + bias",
        pybind11::arg("a_i8"), pybind11::arg("b_i8"), pybind11::arg("row_scale"),
        pybind11::arg("col_scale"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
        pybind11::arg("split_tail") = false,
        pybind11::arg("long_schedule") = true);
  m.def("i8_matmul2d_nt_fused_cs_qkv", &i8_matmul2d_nt_fused_cs_qkv,
        "H3 INT8 QKV matmul with direct [1,H,M,D] bf16 epilogue",
        pybind11::arg("a_i8"), pybind11::arg("b_i8"),
        pybind11::arg("row_scale"), pybind11::arg("col_scale"),
        pybind11::arg("bias"),
        pybind11::arg("heads"), pybind11::arg("head_dim"),
        pybind11::arg("long_schedule") = true);
  m.def("i8_matmul2d_nt_fused_cs_qkv_rms_rope",
        &i8_matmul2d_nt_fused_cs_qkv_rms_rope,
        "H3 INT8 QKV matmul with fused D=128 Q/K RMSNorm and partial "
        "split-half RoPE epilogue",
        pybind11::arg("a_i8"), pybind11::arg("b_i8"),
        pybind11::arg("row_scale"), pybind11::arg("col_scale"),
        pybind11::arg("bias"), pybind11::arg("freqs"),
        pybind11::arg("q_weight"), pybind11::arg("k_weight"),
        pybind11::arg("heads"), pybind11::arg("head_dim"),
        pybind11::arg("eps"), pybind11::arg("long_schedule") = true);
  m.def("i8_matmul2d_nt_fused_cs_qkv_lora_part",
        &i8_matmul2d_nt_fused_cs_qkv_lora_part,
        "H3 INT8 Q/K/V partition with a bf16 runtime-LoRA delta added before "
        "the D=128 Q/K RMSNorm and partial split-half RoPE epilogue",
        pybind11::arg("a_i8"), pybind11::arg("b_i8"),
        pybind11::arg("row_scale"), pybind11::arg("col_scale"),
        pybind11::arg("bias"), pybind11::arg("delta"),
        pybind11::arg("freqs"), pybind11::arg("norm_weight"),
        pybind11::arg("heads"), pybind11::arg("head_dim"),
        pybind11::arg("kind"), pybind11::arg("delta_scale"),
        pybind11::arg("eps"), pybind11::arg("long_schedule") = true);
  m.def("i8_quant_rowwise", &i8_quant_rowwise,
        "Rowwise int8 activation quant on MPS: x[M,K] -> (q int8, scales[M,1] fp32); "
        "bit-exact vs comfy_kitchen quantize_int8_rowwise(sr=0)");
  m.def("i8_matmul2d_nt", &i8_matmul2d_nt,
        "Bit-exact NT int8 matmul: C[M,N] int32 = A[M,K] @ B[N,K]^T on MPS");
  m.def("i8_matmul2d_nt_fused", &i8_matmul2d_nt_fused,
        "Fused NT int8 matmul: D[M,N] bf16 = (A@B^T)*row_scale[M] + bias[N]",
        pybind11::arg("a_i8"), pybind11::arg("b_i8"), pybind11::arg("row_scale"),
        pybind11::arg("bias") = c10::optional<torch::Tensor>());
  m.def("warmup", []() {
        pso_for(false); pso_for(true);
        pso_for_fused(false); pso_for_fused(true);
        if (!g_pso_swiglu_rot)
          g_pso_swiglu_rot = pso_for_name(@"i8_swiglu_rot_bf16");
        if (!g_pso_fused_cs_long)
          g_pso_fused_cs_long = pso_for_name(@"int8_matmul_fused_cs_long");
        if (!g_pso_mlp_bridge)
          g_pso_mlp_bridge =
              pso_for_name(@"i8_swiglu_rot_max_inplace_bf16");
        if (!g_pso_mlp_bridge_quant)
          g_pso_mlp_bridge_quant =
              pso_for_name(@"i8_quant_rowwise_premax_strided_bf16");
        pso_for_name(@"int8_matmul_fused_cs_qkv");
        pso_for_name(@"int8_matmul_fused_cs_qkv_long");
        pso_for_name(@"int8_matmul_fused_cs_qkv_rms_rope");
        pso_for_name(@"int8_matmul_fused_cs_qkv_rms_rope_long");
        pso_for_name(@"int8_matmul_fused_cs_qk_lora_rms_rope");
        pso_for_name(@"int8_matmul_fused_cs_qk_lora_rms_rope_long");
        pso_for_name(@"int8_matmul_fused_cs_v_lora");
        pso_for_name(@"int8_matmul_fused_cs_v_lora_long");
        return true;
      },
      "compile + build pipelines");
}
