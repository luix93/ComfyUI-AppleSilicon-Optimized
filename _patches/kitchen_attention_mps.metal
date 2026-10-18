// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// Quantization and online-softmax policy adapted from Comfy Kitchen v0.2.38,
// commit 42081bf9bb5fad3992c1256aab4f01b27cbb5816.
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Changes: D128 BF16 Metal implementation, head-major buffers, signed-P
// correction for Metal MPP, bounded linear scratch. CUDA byte parity unverified.
// MPP tile construction also follows this pack's CCV-derived H3 kernel.
// Copyright (c) 2010, Liu Liu. All rights reserved.
// CCV revision 6a611be1aab6470ae279115ae4eaf7e01bc87135.
// Full BSD notice: ../third_party/ccv-BSD-3-Clause.txt.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal; using namespace mpp::tensor_ops;
constant constexpr uint NQ = @NQ@, NK = @NK@;
constant constexpr uint QP = (NQ+31)/32*32, KP = (NK+127)/128*128;
inline float4 rotate128(float4 x, uint lane) {
  const uint signs[4]={0x1035997bu,0x8087f5eeu,0xee2e4e1au,0x71132418u};
  for(uint c=0;c<4;++c) x[c]=as_type<float>(as_type<uint>(x[c])^(((((signs[lane/8]>>((lane%8)*4+c))&1u)^1u))<<31));
  float a=x.x+x.y,b=x.x-x.y,c=x.z+x.w,d=x.z-x.w;
  x=float4(a+c,b+d,a-c,b-d)*0.5f;
  for(uint bit=1;bit<32;bit*=2) for(uint c=0;c<4;++c) {
    float other=simd_shuffle_xor(x[c],bit);x[c]=(lane&bit)?other-x[c]:x[c]+other;
  }
  return x*0.1767766952966369f;
}
inline float down_sum(float x){for(uint bit=16;bit;bit/=2)x+=simd_shuffle_down(x,bit);return x;}
inline float down_max(float x){for(uint bit=16;bit;bit/=2)x=max(x,simd_shuffle_down(x,bit));return x;}
kernel void anchor(device const bfloat* K, device int* A, uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint lane=tid%32,w=tid/32;
  threadgroup float sums[13][4]; threadgroup uint chosen;
  float x[9],sum=0,energy=0,mx=0,dist[9];
  for(uint i=0;i<9;++i){x[i]=float(K[(ulong(h)*NK+i*(NK-1)/8)*128+tid]);sum+=x[i];energy=fma(x[i],x[i],energy);mx=max(mx,abs(x[i]));}
  for(uint i=0;i<9;++i){float d=9.f*x[i]-sum;dist[i]=down_sum(d*d);}
  energy=down_sum(energy);mx=down_max(mx);
  if(!lane){sums[0][w]=energy;sums[1][w]=mx;for(uint i=0;i<9;++i)sums[i+2][w]=dist[i];}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(!tid){float best=INFINITY;chosen=0;for(uint i=0;i<9;++i){float d=0;for(uint j=0;j<4;++j)d+=sums[i+2][j];if(d<best){best=d;chosen=i;}}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  energy=0;mx=0;for(uint i=0;i<9;++i){float d=x[i]-x[chosen];energy=fma(d,d,energy);mx=max(mx,abs(d));}
  energy=down_sum(energy);mx=down_max(mx);
  if(!lane){sums[11][w]=energy;sums[12][w]=mx;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(!tid){float e0=0,e1=0,m0=0,m1=0;for(uint j=0;j<4;++j){e0+=sums[0][j];e1+=sums[11][j];m0=max(m0,sums[1][j]);m1=max(m1,sums[12][j]);}A[h]=(e1<e0&&m1<=m0*1.125f)?int(chosen*(NK-1)/8):-1;}
}
kernel void quant_q(device const bfloat* Q,device int8_t* O,device float* S,uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint w [[simdgroup_index_in_threadgroup]]) {
 for(uint z=0;z<2;++z){uint subgroup=w*2+z;float4 x[4];float mx=0;
  for(uint j=0;j<4;++j){uint row=g.x*32+subgroup+j*8;float4 v=0;if(row<NQ)for(uint c=0;c<4;++c)v[c]=float(Q[(ulong(g.y)*NQ+row)*128+lane*4+c]);x[j]=rotate128(v,lane);mx=max(mx,max(max(abs(x[j].x),abs(x[j].y)),max(abs(x[j].z),abs(x[j].w))));}
  float scale=simd_max(mx)/127.f+1e-7f,inv=1.f/scale;
  if(!lane)S[g.y*(QP/4)+g.x*8+subgroup]=scale;
  for(uint j=0;j<4;++j){uint row=g.x*32+subgroup+j*8;for(uint c=0;c<4;++c)O[(ulong(g.y)*QP+row)*128+lane*4+c]=int8_t(clamp(rint(x[j][c]*inv),-127.f,127.f));}
 }
}
kernel void quant_k(device const bfloat* K,device int8_t* O,device float* S,device const int* A,uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint w [[simdgroup_index_in_threadgroup]]) {
 float4 x[32];float mx=0;float4 bias=0;
 if(A[g.y]>=0)for(uint c=0;c<4;++c)bias[c]=float(K[(ulong(g.y)*NK+A[g.y])*128+lane*4+c]);
 for(uint j=0;j<32;++j){uint row=g.x*128+(j/2)*8+w*2+j%2;float4 v=0;if(row<NK)for(uint c=0;c<4;++c)v[c]=float(K[(ulong(g.y)*NK+row)*128+lane*4+c])-bias[c];x[j]=rotate128(v,lane);mx=max(mx,max(max(abs(x[j].x),abs(x[j].y)),max(abs(x[j].z),abs(x[j].w))));}
 float scale=simd_max(mx)/127.f+1e-7f,inv=1.f/scale;
 if(!lane)S[g.y*(KP/32)+g.x*4+w]=scale;
 for(uint j=0;j<32;++j){uint row=g.x*128+(j/2)*8+w*2+j%2;for(uint c=0;c<4;++c)O[(ulong(g.y)*KP+row)*128+lane*4+c]=int8_t(clamp(rint(x[j][c]*inv),-127.f,127.f));}
}
kernel void quant_v(device const bfloat* V,device int8_t* O,device float* S,uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 uint channel=g.x*8+tid%8,part=tid/8;threadgroup float mxs[128];float mx=0;
 for(uint row=part;row<NK;row+=16)mx=max(mx,abs(float(V[(ulong(g.y)*NK+row)*128+channel])));
 mxs[tid]=mx;threadgroup_barrier(mem_flags::mem_threadgroup);
 mx=0;for(uint p=0;p<16;++p)mx=max(mx,mxs[p*8+tid%8]);
 float scale=max(mx*(1.f/127.f),1e-12f),inv=1.f/scale;if(!part)S[g.y*128+channel]=scale;
 for(uint row=part;row<KP;row+=16){float v=row<NK?float(V[(ulong(g.y)*NK+row)*128+channel]):0;O[(ulong(g.y)*KP+row)*128+channel]=int8_t(clamp(rint(v*inv),-127.f,127.f));}
}
// Convert unsigned P to signed P-128 in the matrix path. The exact INT32
// correction 128*sum(V) restores P@V, including padded tail entries.
kernel void v_sums(device const int8_t* V,device int* sums,uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 int s=0;for(uint r=0;r<128;++r)s+=int(V[(ulong(g.y)*KP+g.x*128+r)*128+tid]);sums[(g.y*(KP/128)+g.x)*128+tid]=128*s;
}
kernel void attend(device int8_t* Q,device int8_t* K,device int8_t* V,device const float* QS,device const float* KS,device const float* VS,device const int* VSum,device bfloat* Out,constant float& scale,uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=g.x*64+sg*16,h=g.y;
 auto qt=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(Q+ulong(h)*QP*128,dextents<int32_t,2>(128,QP));
 auto kt=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(K+ulong(h)*KP*128,dextents<int32_t,2>(128,KP));
 auto vt=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(V+ulong(h)*KP*128,dextents<int32_t,2>(128,KP));
 constexpr auto qd=matmul2d_descriptor(16,128,32,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
 constexpr auto pd=matmul2d_descriptor(16,32,128,false,false,true,matmul2d_descriptor::mode::multiply);
 matmul2d<qd,execution_simdgroups<1>> qk;matmul2d<pd,execution_simdgroups<1>> pv;
 using QT=decltype(qt.slice<32,16>(0,0));using KT=decltype(kt.slice<32,128>(0,0));
 auto I=qk.get_destination_cooperative_tensor<QT,KT,int>();
 auto S=qk.get_destination_cooperative_tensor<QT,KT,float>();
 auto M=qk.get_row_reduction_destination_cooperative_tensor<QT,KT,float>();auto L=M;auto TM=M;auto F=M;auto TF=M;auto SUM=M;
 auto P=pv.get_left_input_cooperative_tensor<int8_t,int8_t,int>();
 using VT=decltype(vt.slice<32,128>(0,0));
 auto O=pv.get_destination_cooperative_tensor<decltype(P),VT,int>();
 auto O0=pv.get_destination_cooperative_tensor<decltype(P),VT,float>();auto O1=O0;auto O2=O0;auto O3=O0;
 auto C0=qk.get_left_input_cooperative_tensor<int8_t,int8_t,int>();auto C1=C0;auto C2=C0;auto C3=C0;
 C0.load(qt.slice<32,16>(0,row));C1.load(qt.slice<32,16>(32,row));C2.load(qt.slice<32,16>(64,row));C3.load(qt.slice<32,16>(96,row));

 #pragma clang loop unroll(full)
 for(ushort i=0;i<M.get_capacity();++i)if(M.is_valid_element(i)){M[i]=-INFINITY;L[i]=0;}

 #pragma clang loop unroll(full)
 for(ushort i=0;i<O0.get_capacity();++i)if(O0.is_valid_element(i)){O0[i]=0;O1[i]=0;O2[i]=0;O3[i]=0;}
 for(uint col=0;col<KP;col+=128){
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<I.get_capacity();++i)if(I.is_valid_element(i))I[i]=0;
  auto k0=kt.slice<32,128>(0,col),k1=kt.slice<32,128>(32,col),k2=kt.slice<32,128>(64,col),k3=kt.slice<32,128>(96,col);
  qk.run(C0,k0,I);qk.run(C1,k1,I);qk.run(C2,k2,I);qk.run(C3,k3,I);
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<S.get_capacity();++i)if(S.is_valid_element(i)){
   auto idx=S.get_multidimensional_index(i);uint r=row+idx[1],c=col+idx[0];
   float sc=QS[h*(QP/4)+(min(r,QP-1)/32)*8+r%8]*KS[h*(KP/32)+(c/128)*4+(c%8)/2]*(scale*1.4426950408889634f);
   S[i]=(c<NK)?fma(float(I[i]),sc,-7.9943534f):-INFINITY;
  }
  reduce_rows(S,TM,reduction_operation::max,-INFINITY);
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<M.get_capacity();++i)if(M.is_valid_element(i)){float smaller=fast::exp2(-abs(M[i]-TM[i]));F[i]=M[i]<TM[i]?smaller:1;TF[i]=TM[i]<M[i]?smaller:1;M[i]=max(M[i],TM[i]);}
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<S.get_capacity();++i)if(S.is_valid_element(i)){
   auto idx=S.get_multidimensional_index(i);uint r=row+idx[1],c=col+idx[0];
   float sc=QS[h*(QP/4)+(min(r,QP-1)/32)*8+r%8]*KS[h*(KP/32)+(c/128)*4+(c%8)/2]*(scale*1.4426950408889634f);
   S[i]=c<NK?clamp(rint(fast::exp2(fma(float(I[i]),sc,-*TM.map_iterator(S.get_iterator(i))))),0.f,255.f):0.f;
   P[i]=int8_t(int(S[i])-128);
  }
  reduce_rows(S,SUM,reduction_operation::sum,0.f);
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<L.get_capacity();++i)if(L.is_valid_element(i))L[i]=fma(SUM[i],TF[i],L[i]*F[i]);
  simdgroup_barrier(mem_flags::mem_none);
  auto v0=vt.slice<32,128>(0,col);pv.run(P,v0,O);
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<O0.get_capacity();++i)if(O0.is_valid_element(i)){auto idx=O0.get_multidimensional_index(i);int corrected=O[i]+VSum[(h*(KP/128)+col/128)*128+idx[0]+0];O0[i]=fma(float(corrected),*TF.map_iterator(O0.get_iterator(i)),O0[i]**F.map_iterator(O0.get_iterator(i)));}
  auto v1=vt.slice<32,128>(32,col);pv.run(P,v1,O);
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<O1.get_capacity();++i)if(O1.is_valid_element(i)){auto idx=O1.get_multidimensional_index(i);int corrected=O[i]+VSum[(h*(KP/128)+col/128)*128+idx[0]+32];O1[i]=fma(float(corrected),*TF.map_iterator(O1.get_iterator(i)),O1[i]**F.map_iterator(O1.get_iterator(i)));}
  auto v2=vt.slice<32,128>(64,col);pv.run(P,v2,O);
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<O2.get_capacity();++i)if(O2.is_valid_element(i)){auto idx=O2.get_multidimensional_index(i);int corrected=O[i]+VSum[(h*(KP/128)+col/128)*128+idx[0]+64];O2[i]=fma(float(corrected),*TF.map_iterator(O2.get_iterator(i)),O2[i]**F.map_iterator(O2.get_iterator(i)));}
  auto v3=vt.slice<32,128>(96,col);pv.run(P,v3,O);
 
 #pragma clang loop unroll(full)
 for(ushort i=0;i<O3.get_capacity();++i)if(O3.is_valid_element(i)){auto idx=O3.get_multidimensional_index(i);int corrected=O[i]+VSum[(h*(KP/128)+col/128)*128+idx[0]+96];O3[i]=fma(float(corrected),*TF.map_iterator(O3.get_iterator(i)),O3[i]**F.map_iterator(O3.get_iterator(i)));}
 }

 #pragma clang loop unroll(full)
 for(ushort i=0;i<O0.get_capacity();++i)if(O0.is_valid_element(i)){auto idx=O0.get_multidimensional_index(i);if(row+idx[1]<NQ){float inv=1.f/ *L.map_iterator(O0.get_iterator(i));ulong at=(ulong(h)*NQ+row+idx[1])*128+idx[0];
 Out[at+0]=bfloat((O0[i]*inv)*VS[h*128+idx[0]+0]);
 Out[at+32]=bfloat((O1[i]*inv)*VS[h*128+idx[0]+32]);
 Out[at+64]=bfloat((O2[i]*inv)*VS[h*128+idx[0]+64]);
 Out[at+96]=bfloat((O3[i]*inv)*VS[h*128+idx[0]+96]);
 }}
}
