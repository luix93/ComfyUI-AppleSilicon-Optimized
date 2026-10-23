// BF16 MOSS operations matching ComfyUI / Comfy Kitchen eager semantics.
// PyTorch evaluates the nonlinear tables; no approximate activation is used.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline bfloat rte(float f) {
    uint bits = as_type<uint>(f);
    bits += 0x7fff + ((bits >> 16) & 1);
    return as_type<bfloat>(ushort(bits >> 16));
}

kernel void situ(
    device const bfloat* x, device const bfloat* hidden_table,
    device const bfloat* gate_table, device bfloat* out,
    constant uint& width, constant uint& stride,
    uint i [[thread_position_in_grid]]) {
    uint row = i / width, col = i % width;
    float hidden = float(hidden_table[as_type<ushort>(x[row * stride + col])]);
    float gate = float(gate_table[as_type<ushort>(x[row * stride + width + col])]);
    out[i] = rte(hidden * gate);
}

kernel void rope(
    device const bfloat* q, device const bfloat* k, device const float* pe,
    device bfloat* qo, device bfloat* ko, constant uint& heads,
    constant uint& tokens, uint i [[thread_position_in_grid]]) {
    uint col = i % 128, half_index = col / 64, pair = col % 64;
    uint row = i / 128, token = (row / heads) % tokens;
    uint other = row * 128 + (1 - half_index) * 64 + pair;
    uint f = (token * 64 + pair) * 4 + half_index * 2;
    // Keep both FP32 products and addition separate: FMA changes output bytes.
    float qleft = float(q[i]) * pe[f + half_index];
    float qright = float(q[other]) * pe[f + 1 - half_index];
    float kleft = float(k[i]) * pe[f + half_index];
    float kright = float(k[other]) * pe[f + 1 - half_index];
    qo[i] = rte(qleft + qright);
    ko[i] = rte(kleft + kright);
}

kernel void mod(
    device const bfloat* x, device const bfloat* scale,
    device const bfloat* shift, device bfloat* out,
    constant uint& width, constant uint& tokens,
    uint i [[thread_position_in_grid]]) {
    uint j = (i / (tokens * width)) * width + i % width;
    bfloat s = rte(1.f + float(scale[j]));
    bfloat prod = rte(float(x[i]) * float(s));
    out[i] = rte(float(prod) + float(shift[j]));
}

kernel void square_bf16(device const bfloat* x,device float* squared,
                       uint i [[thread_position_in_grid]]) {
    float v=float(x[i]);squared[i]=v*v;
}
kernel void rms_mod_tail(device const bfloat* x,device const float* inv,
                        device const bfloat* folded,device const bfloat* shift,
                        device bfloat* out,constant uint& width,constant uint& tokens,
                        uint i [[thread_position_in_grid]]) {
    uint j=(i/(tokens*width))*width+i%width;
    bfloat normalized=rte(float(x[i])*inv[i/width]);
    bfloat scale=rte(1.0f+float(folded[j]));
    bfloat value=rte(float(normalized)*float(scale));
    out[i]=rte(float(value)+float(shift[j]));
}

// Adapted from this pack's Qwen RMS/RoPE kernel (PyTorch/MLX reduction);
// MOSS uses split-half rotation and strided packed Q/K views.
kernel void rms_rope_split(device const bfloat* q,device const bfloat* k,
 device const bfloat* qw,device const bfloat* kw,device const float* pe,
 device bfloat* oq,device bfloat* ok,constant uint& H,constant uint& qs,
 constant uint& ks,constant float& eps,
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint d=lane*4;uint stride=group.z==0?qs:ks;
 ulong base=ulong(group.x)*stride+group.y*128;
 device const bfloat* x=(group.z==0?q:k)+base;
 device const bfloat* w=group.z==0?qw:kw;
 float a[4];float acc=0;
 for(uint j=0;j<4;j++){a[j]=float(x[d+j]);acc+=a[j]*a[j];}
 float inv=precise::rsqrt(simd_sum(acc)/128.f+eps);
 bfloat n[4];for(uint j=0;j<4;j++)n[j]=rte(a[j]*inv*float(w[d+j]));
 device bfloat* out=(group.z==0?oq:ok)+(ulong(group.x)*H+group.y)*128;
 for(uint j=0;j<4;j++) {
  uint col=d+j,half_index=col/64,pair=col%64;
  ulong f=(ulong(group.x)*64+pair)*4+half_index*2;
  float own=float(n[j]),other=simd_shuffle_xor(own,16);
  float left=own*pe[f+half_index];float right=other*pe[f+1-half_index];
  out[col]=rte(left+right);
 }
}
