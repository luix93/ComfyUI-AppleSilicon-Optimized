#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline bfloat rte(float f) {uint u=as_type<uint>(f);u+=0x7fffu+((u>>16)&1u);return as_type<bfloat>(ushort(u>>16));}
// One 64-thread group/head: same four contiguous reads and two-stage
// SIMD reduction as PyTorch 2.14 RMSNorm, with a materialized BF16 norm.
kernel void qkv_prep(device const bfloat* qkv,device const bfloat* qw,
 device const bfloat* kw,device const float* pe,device bfloat* oq,
 device bfloat* ok,device bfloat* ov,constant uint& S,constant uint& H,
 constant float& eps,uint3 gid [[threadgroup_position_in_grid]],
 uint lid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
 uint token=gid.x,head=gid.y,slot=gid.z;
 uint d=lid*4;ulong src=((ulong(token)*3+slot)*H+head)*256;
 ulong dst=(ulong(head)*S+token)*256;
 if(slot==2){for(uint j=0;j<4;j++)ov[dst+d+j]=qkv[src+d+j];return;}
 device const bfloat* w=(slot==0?qw:kw);
 threadgroup float sums[32];threadgroup float inv;threadgroup bfloat norm[256];
 float a[4];float acc=0;
 for(uint j=0;j<4;j++){a[j]=float(qkv[src+d+j]);acc+=a[j]*a[j];}
 acc=simd_sum(acc);
 if(sg==0)sums[lane]=0;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(lane==0)sums[sg]=acc;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg==0){float total=simd_sum(sums[lane]);if(lane==0)inv=precise::rsqrt(total/256.f+eps);}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=0;j<4;j++)norm[d+j]=rte(a[j]*inv*float(w[d+j]));
 threadgroup_barrier(mem_flags::mem_threadgroup);
 device bfloat* out=(slot==0?oq:ok);
 for(uint j=0;j<4;j++){
  uint c=d+j;uint side=c/128;uint f=c%128;
  ulong pos=ulong(token)*512+f*4;
  float diag=float(norm[c])*pe[pos+(side==0?0:3)];
  float off=float(norm[(c+128)%256])*pe[pos+(side==0?1:2)];
  out[dst+c]=rte(diag+off);
 }
}
