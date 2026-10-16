// Diagnostic derivative of the Cider-based INT8 GEMM in _patches/int8_ext/int8_gemm.mm.
// Copyright (c) 2026 Mininglamp contributors. MIT; see third_party/cider-MIT.txt.
// Experimental exact FC1 + SwiGLU + ConvRot + group max, with no wide global FC1 output.
template<int BM>
inline void fc1_fused_impl(const device int8_t *A, const device int8_t *B,
    const device float *rs, const device float *cs, const device bfloat *bias,
    device bfloat *rot, device float *gm, uint M, uint F, uint K, uint hb,
    uint2 tgid, uint sgid, uint lid, uint tid, threadgroup float *work) {
  constexpr int TM=BM/16, TN=2;
  int32_t c[TM*TN][kElemsPerFrag]; short2 sc; uint mb,nb;
  const uint tiles_m=(M+BM-1)/BM, tiles_n=F/256;
  if(!w8a8_gemm_compute<BM,256,512,32,1,8,false>(A,B,c,sc,mb,nb,M,F,K,0,tiles_m,tiles_n,tgid,sgid,lid)) return;
  threadgroup bfloat *gate=reinterpret_cast<threadgroup bfloat *>(work);
  for(short mm=0;mm<TM;mm++) for(short nn=0;nn<TN;nn++)
    for(short i=0;i<2;i++) for(short j=0;j<4;j++) {
      uint row=sc.y+mm*16+i*8, col=sgid*32+sc.x+nn*16+j;
      uint mi=mb+row, ni=(tgid.x*256)+col;
      bfloat v=bfloat(0);
      if(mi<M) {v=bf16_rte(float(c[mm*TN+nn][i*4+j])*(cs[ni]*rs[mi])); if(hb) v=bf16_rte(float(v)+float(bias[ni]));}
      gate[row*256+col]=v;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  w8a8_gemm_compute<BM,256,512,32,1,8,false>(A,B+ulong(F)*K,c,sc,mb,nb,M,F,K,0,tiles_m,tiles_n,tgid,sgid,lid);
  float acts[TM*TN][kElemsPerFrag];
  for(short mm=0;mm<TM;mm++) for(short nn=0;nn<TN;nn++)
    for(short i=0;i<2;i++) for(short j=0;j<4;j++) {
      uint row=sc.y+mm*16+i*8, col=sgid*32+sc.x+nn*16+j;
      uint mi=mb+row, ni=F+(tgid.x*256)+col;
      bfloat up=bfloat(0);
      if(mi<M) {up=bf16_rte(float(c[mm*TN+nn][i*4+j])*(cs[ni]*rs[mi])); if(hb) up=bf16_rte(float(up)+float(bias[ni]));}
      const float av=float(gate[row*256+col]);
      const float sil=av/(1.0f+precise::exp(-av));
      const bfloat act=bfloat(sil)*up;
      acts[mm*TN+nn][i*4+j]=float(act);
    }
  // Read all aliased BF16 gate values before overwriting with FP32 rotation state.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(short mm=0;mm<TM;mm++) for(short nn=0;nn<TN;nn++)
    for(short i=0;i<2;i++) for(short j=0;j<4;j++) {
      uint row=sc.y+mm*16+i*8, col=sgid*32+sc.x+nn*16+j;
      work[row*256+col]=acts[mm*TN+nn][i*4+j];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float next[BM];
  for(uint stride=1;stride<=64;stride*=4) {
    for(uint row=0;row<BM;row++) {
      uint digit=(tid/stride)&3u, base=row*256+tid-digit*stride;
      float a0=work[base],a1=work[base+stride],a2=work[base+2*stride],a3=work[base+3*stride];
      if(digit==0) next[row]=a0+a1+a2-a3;
      else if(digit==1) next[row]=a0+a1-a2+a3;
      else if(digit==2) next[row]=a0-a1+a2+a3;
      else next[row]=-a0+a1+a2+a3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint row=0;row<BM;row++) work[row*256+tid]=next[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for(uint row=0;row<BM;row++) {
    bfloat r=bf16_rte(work[row*256+tid]*0.0625f);
    next[row]=simd_max(fabs(float(r)));
    if(mb+row<M) rot[ulong(mb+row)*F+tgid.x*256+tid]=r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lid==0) for(uint row=0;row<BM;row++) work[row*8+sgid]=next[row];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<BM && mb+tid<M) {
    float mx=work[tid*8];for(uint i=1;i<8;i++) mx=fmax(mx,work[tid*8+i]);
    gm[ulong(mb+tid)*(F/256)+tgid.x]=mx;
  }
}
#define FC1_ENTRY(NAME,BM) \
kernel void NAME(const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]], \
const device float *rs [[buffer(2)]],const device float *cs [[buffer(3)]],const device bfloat *bias [[buffer(4)]], \
device bfloat *rot [[buffer(5)]],device float *gm [[buffer(6)]],constant uint &M [[buffer(7)]], \
constant uint &F [[buffer(8)]],constant uint &K [[buffer(9)]],constant uint &hb [[buffer(10)]], \
uint2 g [[threadgroup_position_in_grid]],uint s [[simdgroup_index_in_threadgroup]],uint l [[thread_index_in_simdgroup]],uint t [[thread_index_in_threadgroup]]) { \
threadgroup float work[BM*256]; fc1_fused_impl<BM>(A,B,rs,cs,bias,rot,gm,M,F,K,hb,g,s,l,t,work); }
FC1_ENTRY(fc1_16,16)
FC1_ENTRY(fc1_32,32)
