#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline bfloat rte(float f) {uint u=as_type<uint>(f);u+=0x7fffu+((u>>16)&1u);return as_type<bfloat>(ushort(u>>16));}
kernel void modulate(device const bfloat* x, device const bfloat* pre,
 device const bfloat* target,device const bfloat* zero,device bfloat* out,
 constant uint& D,constant uint& prefix,uint i [[thread_position_in_grid]]) {
 uint d=i%D;float s=float((i/D < prefix ? pre : target)[d]);
 bfloat factor=rte(1.f+s);
 out[i]=rte(float(rte(float(x[i])*float(factor)))+float(zero[d]));
}
kernel void residual(device bfloat* x,device const bfloat* y,
 device const bfloat* pre,device const bfloat* target,
 constant uint& D,constant uint& prefix,uint i [[thread_position_in_grid]]) {
 uint d=i%D;float g=float((i/D < prefix ? pre : target)[d]);
 x[i]=rte(float(x[i])+float(y[i])*g);
}
kernel void rms_rope(device const bfloat* q,device const bfloat* k,
 device const bfloat* qw,device const bfloat* kw,device const float* pe,
 device bfloat* oq,device bfloat* ok,constant uint& H,constant float& eps,
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 ulong base=(ulong(group.x)*H+group.y)*128;uint d=lane*4;
 device const bfloat* x=(group.z==0 ? q : k)+base;
 device const bfloat* w=(group.z==0 ? qw : kw);
 float a[4];float acc=0;
 for(uint j=0;j<4;j++){a[j]=float(x[d+j]);acc+=a[j]*a[j];}
 float inv=precise::rsqrt(simd_sum(acc)/128.f+eps);
 bfloat n[4];
 for(uint j=0;j<4;j++)n[j]=rte(a[j]*inv*float(w[d+j]));
 device bfloat* out=(group.z==0 ? oq : ok)+base;
 for(uint j=0;j<4;j++) {
  ulong f=ulong(group.x)*256+(d+j)*2;
  float diag=pe[f]*float(n[(j/2)*2]);
  out[d+j]=rte(diag+pe[f+1]*float(n[(j/2)*2+1]));
 }
}

// Qwen image VAE: preserve wan_ext arithmetic; distribute more spatial groups.
#define CR_ITERS 1u
#define CR_SBLK  (CR_ITERS * 256u)

kernel void vae_chan_rms(
    device const bfloat* x     [[buffer(0)]],  // [B, C, S]
    device const bfloat* gamma [[buffer(1)]],  // [C]
    device bfloat*       out   [[buffer(2)]],
    constant uint&  C     [[buffer(3)]],
    constant uint&  S     [[buffer(4)]],
    constant uint&  B     [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant float& eps   [[buffer(7)]],
    constant uint&  silu  [[buffer(8)]],
    uint  tgid [[threadgroup_position_in_grid]],
    ushort lid [[thread_index_in_threadgroup]])
{
  const uint nblk = (S + CR_SBLK - 1) / CR_SBLK;
  const uint b = tgid / nblk;
  if (b >= B) return;
  const uint s0 = (tgid % nblk) * CR_SBLK;
  const uint lim = min(CR_SBLK, S - s0);
  const ulong xbase = (ulong)b * C * S + s0;

  float acc[CR_ITERS];
  for (uint i = 0; i < CR_ITERS; i++) acc[i] = 0.0f;

  for (uint c = 0; c < C; c++) {
    const device bfloat* row = x + xbase + (ulong)c * S;
    if (lim == CR_SBLK) {
      for (uint i = 0; i < CR_ITERS; i++) {
        float v = float(row[i * 256u + lid]);
        acc[i] += v * v;
      }
    } else {
      for (uint i = 0; i < CR_ITERS; i++) {
        const uint off = i * 256u + lid;
        if (off < lim) { float v = float(row[off]); acc[i] += v * v; }
      }
    }
  }
  for (uint i = 0; i < CR_ITERS; i++) {
    acc[i] = precise::divide(1.0f, max(precise::sqrt(acc[i]), eps));
  }

  for (uint c = 0; c < C; c++) {
    const device bfloat* row = x + xbase + (ulong)c * S;
    device bfloat* orow = out + xbase + (ulong)c * S;
    const float g = float(gamma[c]);
    for (uint i = 0; i < CR_ITERS; i++) {
      const uint off = i * 256u + lid;
      if (off < lim) {
        // mirror eager rounding: normalize -> *scale -> *gamma, each in dtype
        bfloat n0 = rte(float(row[off]) * acc[i]);
        bfloat n1 = rte(float(n0) * scale);
        float y = float(rte(float(n1) * g));
        if (silu != 0u) {
          y = y / (1.0f + precise::exp(-y));
        }
        orow[off] = rte(y);
      }
    }
  }
}
