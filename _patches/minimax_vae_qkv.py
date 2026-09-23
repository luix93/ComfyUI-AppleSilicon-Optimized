"""Optional FP16 H3 VAE QKV projection with FP32 accumulation.

Fuses the projection epilogue, Q/K RMSNorm, partial RoPE and final layout.
The accepted exact path remains available in minimax_vae_fast.
"""
import torch

SOURCE=r'''
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant constexpr int M=@M@, S=@S@, H=32, K=2048, BM=@BM@, BK=@BK@, NSG=@NSG@, R=@R@;
kernel void qkv(device half* A [[buffer(0)]], device half* W [[buffer(1)]],
 device half* B [[buffer(2)]], device half* ROPE [[buffer(3)]], device half* SCALE [[buffer(4)]],
 device half* Q [[buffer(5)]], device half* KEY [[buffer(6)]], device half* V [[buffer(7)]],
 uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]]) {
 const int m0=int(group.x)*BM, head=int(group.y)/3, slot=int(group.y)%3;
 const int n0=head*192+slot*64;
 constexpr auto desc=matmul2d_descriptor(BM,64,static_cast<int>(dynamic_extent),false,true,false,
   matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroups<NSG>> op;
 auto a0=tensor(A+ulong(m0)*K,dextents<int,2>{BK,min(BM,M-m0)},array<int,2>{1,K});
 auto w0=tensor(W+ulong(n0)*K,dextents<int,2>{BK,64},array<int,2>{1,K});
 using AT=__tensor_ops_detail::__remove_addrspace_t<decltype(a0)>;
 using WT=__tensor_ops_detail::__remove_addrspace_t<decltype(w0)>;
 auto c=op.get_destination_cooperative_tensor<AT,WT,float>();
 #pragma clang loop unroll(full)
 for(uint16_t i=0;i<c.get_capacity();++i) if(c.is_valid_element(i)) c[i]=0.0f;
 for(int k0=0;k0<K;k0+=BK) {
   auto a=tensor(A+ulong(m0)*K+k0,dextents<int,2>{BK,min(BM,M-m0)},array<int,2>{1,K});
   auto w=tensor(W+ulong(n0)*K+k0,dextents<int,2>{BK,64},array<int,2>{1,K});
   op.run(a,w,c);
 }
 threadgroup half buf[BM*64];
 #pragma clang loop unroll(full)
 for(uint16_t i=0;i<c.get_capacity();++i) if(c.is_valid_element(i)) {
   auto idx=c.get_multidimensional_index(i);
   const int r=int(idx[1]),d=int(idx[0]);
   if(m0+r<M) buf[r*64+d]=half(c[i]+float(B[n0+d]));
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(int r=int(sg);r<BM && m0+r<M;r+=NSG) {
   const int bs=m0+r;
   const ulong dst=(((ulong)(bs/S)*H+head)*S+bs%S)*64;
   if(slot==2) {
     for(int d=int(lane);d<64;d+=32) V[dst+d]=buf[r*64+d];
   } else {
     float acc=0.0f;
     if(lane<16) for(int i=0;i<4;++i) {float a=float(buf[r*64+lane*4+i]);acc+=a*a;}
     acc=simd_sum(acc);
     const float inv=precise::rsqrt(acc/64.0f+@EPS@f);
     half v0=SCALE[lane]*half(float(buf[r*64+lane])*inv);
     half v1=SCALE[lane+32]*half(float(buf[r*64+lane+32])*inv);
     simdgroup_barrier(mem_flags::mem_threadgroup);
     buf[r*64+lane]=v0;buf[r*64+lane+32]=v1;
     simdgroup_barrier(mem_flags::mem_threadgroup);
     device half* out=slot==0?Q:KEY;
     for(int d=int(lane);d<64;d+=32) {
       half y=buf[r*64+d];
       if(d<2*R) {
         const int pair=d%R,side=d/R;
         const ulong f=((ulong)bs*R+pair)*4;
         const half diagonal=y*ROPE[f+(side==0?0:3)];
         y=half(float(diagonal)+float(buf[r*64+(side==0?d+R:d-R)])*float(ROPE[f+(side==0?1:2)]));
       }
       out[dst+d]=y;
     }
   }
 }
}
'''

class Qkv:
    def __init__(self, tile=(64, 256, 4)):
        self.tile = tile
        self.cache = {}

    def __call__(self, x, weight, bias, rope, scale, eps):
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != 2048:
            raise ValueError("expected nonempty [B,S,2048] input")
        b, s, _ = x.shape
        if (weight.shape != (6144, 2048) or bias.shape != (6144,)
                or scale.shape != (64,) or rope.ndim != 6
                or rope.shape[:3] != (b, s, 1) or rope.shape[4:] != (2, 2)
                or not 0 < rope.shape[3] <= 32):
            raise ValueError("unsupported H3 FP16 QKV geometry")
        if not all(t.device == x.device and t.is_mps and t.dtype == torch.float16
                   and t.is_contiguous() for t in (x, weight, bias, rope, scale)):
            raise ValueError("expected contiguous FP16 tensors on one MPS device")
        key = (b, s, rope.shape[3], float(eps))
        if key not in self.cache:
            src = SOURCE
            values = dict(M=b * s, S=s, R=rope.shape[3], EPS=format(eps, '.17e'),
                          BM=self.tile[0], BK=self.tile[1], NSG=self.tile[2])
            for name, value in values.items():
                src = src.replace('@' + name + '@', str(value))
            self.cache[key] = torch.mps.compile_shader(src)
        out = [torch.empty((b, 32, s, 64), device=x.device, dtype=x.dtype) for _ in range(3)]
        bm, _, nsg = self.tile
        self.cache[key].qkv(x, weight, bias, rope, scale, *out,
                           threads=(((b * s + bm - 1) // bm) * nsg * 32, 96, 1),
                           group_size=(nsg * 32, 1, 1))
        return out
