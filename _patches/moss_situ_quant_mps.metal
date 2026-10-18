// Experimental fused SiTU + regular Hadamard + rowwise INT8.
// Regular Hadamard/quantization semantics follow Comfy Kitchen (Apache-2.0).
// SiTU LUT preserves BF16 rounding; reassociated rotation is NOT bit-exact.
#include <metal_stdlib>
using namespace metal;
kernel void situ_quant(device const ushort* X,device const bfloat* A,device const bfloat* B,device int8_t* Q,device float* Scale,uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 float x[64];uint base=(tid/4)*256+(tid%4)*64;
 #pragma clang loop unroll(full)
 for(uint j=0;j<64;++j)x[j]=float(bfloat(float(A[X[ulong(row)*16384+base+j]])*float(B[X[ulong(row)*16384+8192+base+j]])));
 #pragma clang loop unroll(full)
 for(uint stride=1;stride<=16;stride*=4){
  #pragma clang loop unroll(full)
  for(uint g=0;g<64;g+=stride*4){
   #pragma clang loop unroll(full)
   for(uint i=0;i<stride;++i){uint k=g+i;float a=x[k],b=x[k+stride],c=x[k+stride*2],d=x[k+stride*3];
    x[k]=a+b+c-d;x[k+stride]=a+b-c+d;x[k+stride*2]=a-b+c+d;x[k+stride*3]=-a+b+c+d;
   }
  }
 }
 float mx=0;
 #pragma clang loop unroll(full)
 for(uint j=0;j<64;++j){uint b=lane&~3u;float a0=simd_shuffle(x[j],b),a1=simd_shuffle(x[j],b+1),a2=simd_shuffle(x[j],b+2),a3=simd_shuffle(x[j],b+3);float y;
  switch(lane%4){case 0:y=a0+a1+a2-a3;break;case 1:y=a0+a1-a2+a3;break;case 2:y=a0-a1+a2+a3;break;default:y=-a0+a1+a2+a3;}
  x[j]=float(bfloat(y*.0625f));mx=max(mx,abs(x[j]));
 }
 threadgroup float shared[4];mx=simd_max(mx);if(!lane)shared[tid/32]=mx;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(!tid){float m=max(max(shared[0],shared[1]),max(shared[2],shared[3]));shared[0]=max(precise::divide(m,127.f),1e-30f);Scale[row]=shared[0];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float scale=float(bfloat(shared[0]));if(scale==0)scale=1.1754943508222875e-38f;
 #pragma clang loop unroll(full)
 for(uint j=0;j<64;++j)Q[ulong(row)*8192+base+j]=int8_t(clamp(rint(float(bfloat(precise::divide(x[j],scale)))),-128.f,127.f));
}
