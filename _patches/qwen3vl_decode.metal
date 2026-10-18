
#include <metal_stdlib>
using namespace metal;
kernel void int8_gemv(device const char* x,device const char* w,
 device const float* xs,device const float* ws,device bfloat* out,
 constant uint& K,constant uint& N,constant uint& scalar_scale,
 uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 uint row=group*4+sg;if(row>=N)return;
 int total=0;
 for(uint k=lane;k<K;k+=32) total+=int(x[k])*int(w[ulong(row)*K+k]);
 total=simd_sum(total);
 if(lane==0){volatile float scale=xs[0]*ws[scalar_scale?0:row];
  volatile float scaled=float(total)*scale;out[row]=bfloat(scaled);}
}

#include <metal_stdlib>
using namespace metal;
kernel void qwen_rope(device const bfloat* Q, device const bfloat* K,
                      device const float* R, device bfloat* OQ,
                      device bfloat* OK, uint i [[thread_position_in_grid]]) {
    bool is_q = i < 2048;
    uint p = is_q ? i : i - 2048;
    uint head = p / 64, col = p % 64;
    device const bfloat* X = is_q ? Q : K;
    device bfloat* Y = is_q ? OQ : OK;
    float a = float(X[head * 128 + col]);
    float b = float(X[head * 128 + 64 + col]);
    // volatile retains the native intermediate rounding; contraction changes
    // several BF16 output bytes even for ordinary finite model activations.
    volatile float pa = a * R[col * 4];
    volatile float pb = b * R[col * 4 + 3];
    volatile float ca = b * R[col * 4 + 1];
    volatile float cb = a * R[col * 4 + 2];
    Y[head * 128 + col] = bfloat(pa + ca);
    Y[head * 128 + 64 + col] = bfloat(pb + cb);
}

#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline bfloat rte(float f){uint u=as_type<uint>(f);u+=0x7fffu+((u>>16)&1u);return as_type<bfloat>(ushort(u>>16));}
// Normalization ordering follows qwen21_fast.metal and the PyTorch/MLX
// convention credited in THIRD_PARTY_NOTICES.md; licenses in krea2_ext.
kernel void qk_norm_rope(device const bfloat* Q,device const bfloat* K,
 device const bfloat* QW,device const bfloat* KW,device const float* R,
 device bfloat* OQ,device bfloat* OK,constant float& eps,
 uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 bool isq=head<32;uint h=isq?head:head-32;
 device const bfloat* x=(isq?Q:K)+h*128;
 device const bfloat* w=isq?QW:KW;
 device bfloat* out=(isq?OQ:OK)+h*128;
 float a[4],sum=0.f;uint d=lane*4;
 for(uint j=0;j<4;j++){a[j]=float(x[d+j]);sum+=a[j]*a[j];}
 float inv=precise::rsqrt(simd_sum(sum)/128.f+eps);
 for(uint j=0;j<4;j++) {
  float n=float(rte(a[j]*inv*float(w[d+j])));
  float other=simd_shuffle_xor(n,16);
  uint c=(d+j)%64;
  float na=d<64?n:other,nb=d<64?other:n;
  volatile float pa=na*R[c*4+(d<64?0:2)];
  volatile float pb=nb*R[c*4+(d<64?1:3)];
  out[d+j]=rte(pa+pb);
 }
}
