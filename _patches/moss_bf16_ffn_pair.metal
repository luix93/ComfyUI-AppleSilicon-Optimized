// MOSS FC1/SiTU fusion: exact BF16 projection and activation rounding.
// INT8 GEMM adapted from Cider, MIT, Copyright (c) 2026 Mininglamp contributors.
// https://github.com/Mininglamp-AI/cider
// Local paired-column addressing and SiTU epilogue; no weight repacking.
// Full license: ../third_party/cider-MIT.txt.
#define ASFP8_INT8_ULTRA_SCHEDULE 1
#pragma clang fp contract(off)

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
inline void nax_frag_store_int32(const thread float *src, device float *dst,
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
inline void nax_frag_store_dequant(const thread float *src, device OutT *dst,
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

template <int BM, int BN, int DEFAULT_BK, int DEFAULT_SK, int WM, int WN,
          bool USE_COMPILER_BARRIER>
bool w8a8_pair_compute(
    const device bfloat *A, const device bfloat *B, const device bfloat *Bup,
    thread float (&c_frags)[((BM / WM) / 16) * ((BN / WN) / 16)][kElemsPerFrag],
    thread short2 &sc_out, thread uint &m_base_out, thread uint &n_base_out,
    uint M, uint N, uint K, uint swizzle_log, uint tiles_m, uint tiles_n,
    uint2 tgid, uint sgid, uint lid) {
#if ASFP8_INT8_ULTRA_SCHEDULE
  // M5 Ultra / macOS 27: one K fragment per load/compute group avoids the
  // SK=32 throughput cliff (81 -> 167 TOP/s at M=4300,K=N=6144).
  // BK=256 avoids the slower remainder loop for K=3840/5376. Two-way
  // unrolling improves Ultra scheduling without changing integer arithmetic.
  constexpr int SK = 16;
  constexpr int BK = 256;
#else
  constexpr int SK = DEFAULT_SK;
  constexpr int BK = DEFAULT_BK;
#endif
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

  const device bfloat *sg_A = A + m_base * K;
  const device bfloat *sg_B = B + (n_base/32)*16 * K;
  const device bfloat *sg_U = Bup + (n_base/32)*16 * K;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
  auto ct_b =
      gemm_op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  // decltype on a local carries the `thread` addrspace qualifier; macOS 27
  // beta 26A5388g tightened the destination overload's enable_if so the
  // qualified type no longer satisfies __is_cooperative_tensor_type_v —
  // strip it (older headers accepted either form).
  auto ct_c =
      gemm_op.get_destination_cooperative_tensor<
          mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(ct_a)>,
          mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(ct_b)>,
          float>();

  for (int f = 0; f < TM * TN; f++) {
    for (int i = 0; i < kElemsPerFrag; i++) {
      c_frags[f][i] = 0;
    }
  }

  int gemm_k_iters = int(K) / BK;
  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
#if ASFP8_INT8_ULTRA_SCHEDULE
    #pragma unroll 2
#endif
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      bfloat a_frags[TM][TK][kElemsPerFrag];
      bfloat b_frags[TN][TK][kElemsPerFrag];
      compiler_barrier_slot<USE_COMPILER_BARRIER> compiler_barrier;

      for (short mm = 0; mm < TM; mm++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc, short(mm * 16),
                        short(kk * 16));
        }
      }

      for (short nn = 0; nn < TN; nn++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(b_frags[nn][kk], (nn%2 ? sg_U : sg_B) + kk1, int(K), sc, short((nn/2)*16),
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
    sg_U += BK;
  }

  // Remainder K
  int rem_k = int(K) - gemm_k_iters * BK;
  for (int kk1 = 0; kk1 < rem_k; kk1 += 16) {
    bfloat a_frag[TM][kElemsPerFrag];
    bfloat b_frag[TN][kElemsPerFrag];
    short psk = short(max(0, rem_k - kk1));

    for (short mm = 0; mm < TM; mm++) {
      const device bfloat *ptr = sg_A + kk1 + (sc.y + mm * 16) * K + sc.x;
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          a_frag[mm][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : bfloat(0);
        }
    }
    for (short nn = 0; nn < TN; nn++) {
      const device bfloat *ptr = (nn%2 ? sg_U : sg_B) + kk1 + (sc.y + (nn/2)*16) * K + sc.x;
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          b_frag[nn][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : bfloat(0);
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




kernel void moss_64_128_2_4_0(
 const device bfloat *A, const device bfloat *B, const device bfloat *U,
 device bfloat *out, constant uint &M, constant uint &N, constant uint &K,
 constant uint &tiles_m, constant uint &tiles_n,
 const device bfloat *ht, const device bfloat *gt,
 uint2 tgid [[threadgroup_position_in_grid]],
 uint sgid [[simdgroup_index_in_threadgroup]], uint lid [[thread_index_in_simdgroup]]) {
 constexpr short TM=((64/2)/16), TN=((128/4)/16);
 float acc[TM*TN][kElemsPerFrag]; short2 sc; uint mb,nb;
 if (!w8a8_pair_compute<64,128,512,32,2,4,false>(
 A,B,U,acc,sc,mb,nb,M,N,K,0,tiles_m,tiles_n,tgid,sgid,lid)) return;
 for(short mm=0;mm<TM;mm++) for(short nn=0;nn<TN;nn+=2)
 for(short i=0;i<2;i++) for(short j=0;j<4;j++) {
  uint m=mb+sc.y+mm*16+i*8, n=nb+sc.x+nn*16+j;
  if(m<M && n+16<N) {
   uint col=(n/32)*16+n%16;
   bfloat a=bf16_rte(acc[mm*TN+nn][i*4+j]);
   bfloat b=bf16_rte(acc[mm*TN+nn+1][i*4+j]);
   bfloat value=bf16_rte(float(ht[as_type<ushort>(a)])*float(gt[as_type<ushort>(b)]));
   out[m*(N/2)+col]=value;
  }
 }
}
