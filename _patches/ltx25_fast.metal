#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline bfloat rte(float f) {uint u=as_type<uint>(f);u+=0x7fffu+((u>>16)&1u);return as_type<bfloat>(ushort(u>>16));}

// Kitchen's split-half rotation: materialize the BF16 diagonal product
// before addcmul. Read the current [B,S,H,D/2,2,2] rotation format directly.
kernel void split_rope(device const bfloat* q, device const bfloat* k,
 device const bfloat* pe, device bfloat* oq, device bfloat* ok,
 constant uint& D, uint2 i [[thread_position_in_grid]]) {
 uint hd=D/2, d=i.x%D, r=d%hd;
 uint base=i.x-d, f=(base/2+r)*4;
 device const bfloat* x=i.y==0 ? q : k;
 device bfloat* out=i.y==0 ? oq : ok;
 float diag=float(pe[f+(d<hd ? 0 : 3)]);
 float off=float(pe[f+(d<hd ? 1 : 2)]);
 float v=float(x[base+(d<hd ? r+hd : r)]);
 float prod=float(rte(float(x[i.x])*diag));
 out[i.x]=rte(prod+v*off);
}

kernel void rms_rope64(device bfloat* q,device bfloat* k,device const bfloat* qw,device const bfloat* kw,
 device const float* pe, constant uint& H,constant float& eps,
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 ulong base=(ulong(group.x)*H+group.y)*64;uint d=lane*4;
 device bfloat* x=(group.z==0 ? q : k)+base;
 device const bfloat* w=(group.z==0 ? qw : kw);
 float a[4];float acc=0;
 for(uint j=0;j<4;j++){a[j]=d<64 ? float(x[d+j]) : 0.f;acc+=a[j]*a[j];}
 float inv=precise::rsqrt(simd_sum(acc)/64.f+eps);
 if(d>=64)return;
 bfloat n[4];for(uint j=0;j<4;j++)n[j]=rte(a[j]*inv*float(w[d+j]));
 for(uint j=0;j<4;j++) {
  ulong f=ulong(group.x)*128+(d+j)*2;
  float diag=pe[f]*float(n[(j/2)*2]);
  x[d+j]=rte(diag+pe[f+1]*float(n[(j/2)*2+1]));
 }
}
// Geometry-only mask construction; no floating arithmetic or GPU readback.
kernel void mask3d(device const int* p,device ushort* out,uint i [[thread_position_in_grid]]) {
 uint qt=p[0],qh=p[1],qw=p[2],kt=p[3],kh=p[4],kw=p[5];
 uint nk=kt*kh*kw,q=i/nk,k=i%nk;
 uint t=q/(qh*qw),h=(q/qw)%qh,w=q%qw;
 uint rt=k/(kh*kw),rh=(k/kw)%kh,rw=k%kw;
 bool yes=rt>=uint(p[6+t]) && rt<uint(p[6+qt+t])
  && rh>=uint(p[6+2*qt+h]) && rh<uint(p[6+2*qt+qh+h])
  && rw>=uint(p[6+2*qt+2*qh+w]) && rw<uint(p[6+2*qt+2*qh+qw+w]);
 out[i]=yes ? 0 : 0xff7f;
}

// The table holds PyTorch's BF16 SiLU for each input bit pattern. Keep its
// rounded result before multiplying by the independently projected up branch.
kernel void vae_swiglu(device const bfloat* gate,device const bfloat* up,
 device const bfloat* lut,device bfloat* out,uint i [[thread_position_in_grid]]) {
 out[i]=rte(float(lut[as_type<ushort>(gate[i])])*float(up[i]));
}

kernel void vae_modulate(device const bfloat* x,device const bfloat* scale,
 device const bfloat* shift,device bfloat* out,constant uint& D,
 uint i [[thread_position_in_grid]]) {
 bfloat factor=rte(1.f+float(scale[i%D]));
 bfloat product=rte(float(x[i])*float(factor));
 out[i]=rte(float(product)+float(shift[i%D]));
}

// Integer row maps preserve the eager neighborhood flattening order.
kernel void vae_gather(device const bfloat* q,device const bfloat* k,device const bfloat* v,
 device bfloat* oq,device bfloat* ok,device bfloat* ov,device const int* mapping,
 constant uint& nq,constant uint& nk,constant uint& channels,
 uint2 gid [[thread_position_in_grid]]) {
 uint i=gid.x,n=gid.y==0 ? nq : nk,rows=uint(mapping[0])*n;
 if(i>=rows*channels)return;
 uint row=i/channels,c=i%channels;
 uint off=uint(mapping[4+(gid.y==0 ? 0 : uint(mapping[0])*nq)+row])+c;
 if(gid.y==0)oq[i]=q[off];else if(gid.y==1)ok[i]=k[off];else ov[i]=v[off];
}

kernel void vae_scatter(device const bfloat* x,device bfloat* out,device const int* mapping,
 constant uint& channels,uint i [[thread_position_in_grid]]) {
 out[uint(mapping[4+i/channels])+i%channels]=x[i];
}
