// Diagnostic derivative of the Cider-based INT8 GEMM in _patches/int8_ext/int8_gemm.mm.
// Copyright (c) 2026 Mininglamp contributors. MIT; see third_party/cider-MIT.txt.
// Experimental exact FC1 + SwiGLU + ConvRot + group max, with no wide global FC1 output.

inline void fc1_pair_impl(const device int8_t *A, const device int8_t *B,
    const device float *rs, const device float *cs, const device bfloat *bias,
    device bfloat *rot, device float *gm, uint M, uint F, uint K, uint hb,
    uint2 tgid, uint sgid, uint lid, uint tid, threadgroup float *work) {
  constexpr int BM=128, TM=2, TN=2;
  int32_t c[TM*TN][kElemsPerFrag]; short2 sc; uint mb,nb;
  const uint tiles_m=(M+BM-1)/BM, tiles_n=F/128;
  if(!w8a8_gemm_compute<128,128,512,32,4,4,false>(A,B,c,sc,mb,nb,M,F,K,2,tiles_m,tiles_n,tgid,sgid,lid)) return;
  threadgroup bfloat *gate=reinterpret_cast<threadgroup bfloat *>(work);
  for(short mm=0;mm<TM;mm++) for(short nn=0;nn<TN;nn++)
    for(short i=0;i<2;i++) for(short j=0;j<4;j++) {
      uint row=(sgid/4)*32+sc.y+mm*16+i*8, col=(sgid%4)*32+sc.x+nn*16+j;
      uint mi=mb-(sgid/4)*32+row, ni=((tgid.x>>2)*128)+col;
      bfloat v=bfloat(0);
      if(mi<M) {v=bf16_rte(float(c[mm*TN+nn][i*4+j])*(cs[ni]*rs[mi])); if(hb) v=bf16_rte(float(v)+float(bias[ni]));}
      gate[row*128+col]=v;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  w8a8_gemm_compute<128,128,512,32,4,4,false>(A,B+ulong(F)*K,c,sc,mb,nb,M,F,K,2,tiles_m,tiles_n,tgid,sgid,lid);
  float acts[TM*TN][kElemsPerFrag];
  for(short mm=0;mm<TM;mm++) for(short nn=0;nn<TN;nn++)
    for(short i=0;i<2;i++) for(short j=0;j<4;j++) {
      uint row=(sgid/4)*32+sc.y+mm*16+i*8, col=(sgid%4)*32+sc.x+nn*16+j;
      uint mi=mb-(sgid/4)*32+row, ni=F+((tgid.x>>2)*128)+col;
      bfloat up=bfloat(0);
      if(mi<M) {up=bf16_rte(float(c[mm*TN+nn][i*4+j])*(cs[ni]*rs[mi])); if(hb) up=bf16_rte(float(up)+float(bias[ni]));}
      const float av=float(gate[row*128+col]);
      const float sil=av/(1.0f+precise::exp(-av));
      const bfloat act=bfloat(sil)*up;
      if(mi<M) rot[ulong(mi)*F+(ni-F)]=act;
    }
}
kernel void fc1_pair128(const device int8_t *A [[buffer(0)]],const device int8_t *B [[buffer(1)]],
const device float *rs [[buffer(2)]],const device float *cs [[buffer(3)]],const device bfloat *bias [[buffer(4)]],
device bfloat *rot [[buffer(5)]],device float *gm [[buffer(6)]],constant uint &M [[buffer(7)]],
constant uint &F [[buffer(8)]],constant uint &K [[buffer(9)]],constant uint &hb [[buffer(10)]],
uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]],uint t [[thread_index_in_threadgroup]]) {
threadgroup bfloat gate[128*128];fc1_pair_impl(A,B,rs,cs,bias,rot,gm,M,F,K,hb,g,sg,l,t,reinterpret_cast<threadgroup float *>(gate));
}
