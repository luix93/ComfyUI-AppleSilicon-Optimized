// Diagnostic derivative of the Cider-based INT8 GEMM in _patches/int8_ext/int8_gemm.mm.
// Copyright (c) 2026 Mininglamp contributors. MIT; see third_party/cider-MIT.txt.
template <int BM, int BN, int DEFAULT_BK, int DEFAULT_SK, int WM, int WN,
          bool USE_COMPILER_BARRIER>
bool fc1_dual_compute(
    const device int8_t *A, const device int8_t *B,
    thread int32_t (&c_frags)[((BM / WM) / 16) * ((BN / WN) / 16)][kElemsPerFrag],
    thread int32_t (&uout)[((BM / WM) / 16) * ((BN / WN) / 16)][kElemsPerFrag],
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
      c_frags[f][i] = 0; uout[f][i] = 0;
    }
  }

  int gemm_k_iters = int(K) / BK;
  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
#if ASFP8_INT8_ULTRA_SCHEDULE
    #pragma unroll 2
#endif
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TN][TK][kElemsPerFrag];
      int8_t u_frags[TN][TK][kElemsPerFrag];
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
      for (short nn = 0; nn < TN; nn++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(u_frags[nn][kk], sg_B + ulong(N)*K + kk1, int(K), sc, short(nn * 16),
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
              ct_b[i] = u_frags[nn][kk][i];
              ct_b[kElemsPerFrag + i] = u_frags[nn + 1][kk][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = uout[c0][i];
              ct_c[kElemsPerFrag + i] = uout[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              uout[c0][i] = ct_c[i];
              uout[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }
      compiler_barrier.touch();
    }
    sg_A += BK;
    sg_B += BK;
  }

  sc_out=sc; m_base_out=m_base; n_base_out=n_base;return true;
}
template<int BM,int WM> inline void fc1_dual_impl(const device int8_t *A,const device int8_t *B,const device float *rs,const device float *cs,const device bfloat *bias,device bfloat *rot,uint M,uint F,uint K,uint hb,uint2 g,uint sg,uint l) {
 constexpr int TM=2,TN=2;int32_t gate[TM*TN][kElemsPerFrag],up[TM*TN][kElemsPerFrag];short2 sc;uint mb,nb;
 if(!fc1_dual_compute<BM,128,256,16,WM,4,false>(A,B,gate,up,sc,mb,nb,M,F,K,2,(M+BM-1)/BM,F/128,g,sg,l))return;
 for(short mm=0;mm<TM;mm++) for(short nn=0;nn<TN;nn++) for(short i=0;i<2;i++) for(short j=0;j<4;j++){
 uint row=mb+sc.y+mm*16+i*8,col=nb+sc.x+nn*16+j;
 if(row<M) {
 bfloat a=bf16_rte(float(gate[mm*TN+nn][i*4+j])*(cs[col]*rs[row]));
 bfloat b=bf16_rte(float(up[mm*TN+nn][i*4+j])*(cs[col+F]*rs[row]));
 if(hb){a=bf16_rte(float(a)+float(bias[col]));b=bf16_rte(float(b)+float(bias[col+F]));}
 float av=float(a);float sil=av/(1.0f+precise::exp(-av));rot[ulong(row)*F+col]=bfloat(sil)*b;
 }
 }
}
#define DUAL_ENTRY(NAME,BM,WM) \
kernel void NAME(const device int8_t *A [[buffer(0)]],const device int8_t *B [[buffer(1)]], \
const device float *rs [[buffer(2)]],const device float *cs [[buffer(3)]],const device bfloat *bias [[buffer(4)]], \
device bfloat *rot [[buffer(5)]],device float *gm [[buffer(6)]],constant uint &M [[buffer(7)]], \
constant uint &F [[buffer(8)]],constant uint &K [[buffer(9)]],constant uint &hb [[buffer(10)]], \
uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]]) { \
fc1_dual_impl<BM,WM>(A,B,rs,cs,bias,rot,M,F,K,hb,g,sg,l); }
DUAL_ENTRY(fc1_dual128,128,4)
DUAL_ENTRY(fc1_dual64,64,2)
