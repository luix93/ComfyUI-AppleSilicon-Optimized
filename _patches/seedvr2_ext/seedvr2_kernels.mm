// SeedVR2 memory-pass fusions. All arithmetic compiles with safe math.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <array>
using namespace at::mps;

static const char* source = R"MTL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// Copy complete rows, optionally from a virtual concatenation. Both grid
// axes stay small; all buffer offsets are explicitly 64-bit.
kernel void gather_rows(device const ushort* a, device const ushort* b,
    device const long* indices, device ushort* out,
    constant ulong& A, constant ulong& SA, constant ulong& SB,
    constant ulong& C, uint2 xy [[thread_position_in_grid]]) {
  const ulong col=xy.x,row=xy.y,src=ulong(indices[row]);
  out[row*C+col]=src<A ? a[src*SA+col] : b[(src-A)*SB+col];
}
inline bfloat rte(float f) {
  uint u=as_type<uint>(f); u+=0x7fffu+((u>>16)&1u);
  return as_type<bfloat>(ushort(u>>16));
}
kernel void modulate(device const bfloat* x, device const bfloat* sc,
    device const bfloat* sh, device bfloat* out, constant uint& D,
    constant ulong& N, uint i [[thread_position_in_grid]]) {
  if(ulong(i)>=N)return;
  out[i]=rte(float(rte(float(x[i])*float(sc[i%D])))+float(sh[i%D]));
}
kernel void gate_residual(device const bfloat* x, device const bfloat* gate,
    device const bfloat* residual, device bfloat* out, constant uint& D,
    constant ulong& N, uint i [[thread_position_in_grid]]) {
  if(ulong(i)>=N)return;
  out[i]=rte(float(rte(float(x[i])*float(gate[i%D])))+float(residual[i]));
}
kernel void rms_square(device const bfloat* x, device float* out,
    constant ulong& N, uint i [[thread_position_in_grid]]) {
  if(ulong(i)<N) { float value=float(x[i]);out[i]=value*value; }
}
kernel void rms_apply(device const bfloat* x, device const float* inv,
    device const bfloat* sc, device const bfloat* sh, device bfloat* out,
    constant uint& D, constant ulong& N, uint i [[thread_position_in_grid]]) {
  if(ulong(i)>=N)return;
  const bfloat normalized=rte(float(x[i])*inv[i/D]);
  out[i]=rte(float(rte(float(normalized)*float(sc[i%D])))+float(sh[i%D]));
}
kernel void rope_pair(device const bfloat* q, device const bfloat* k,
    device const float* cs, device const float* sn,
    device bfloat* oq, device bfloat* ok,
    constant uint& H, constant uint& D, constant uint& R,
    constant ulong& N, uint i [[thread_position_in_grid]]) {
  if(ulong(i)>=N)return;
  uint d=i%D;
  if(d>=R) { oq[i]=q[i];ok[i]=k[i];return; }
  ulong token=ulong(i)/(H*D), f=token*R+d, partner=ulong(i)^1ul;
  float sign=(d&1u)?1.0f:-1.0f;
  oq[i]=rte(float(q[i])*cs[f]+(sign*float(q[partner]))*sn[f]);
  ok[i]=rte(float(k[i])*cs[f]+(sign*float(k[partner]))*sn[f]);
}
#pragma clang fp contract(on)
kernel void gn_stats(device const half* x, device float* stats,
constant uint& C,constant uint& HW,constant uint& G,constant float& eps,
uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint threads [[threads_per_threadgroup]]) {
 const uint elements=(C/G)*HW;const ulong offset=ulong(group)*elements;
 float sum=0,ss=0;
 for(uint r=0;r<elements;r+=threads*4) {
  uint base=r+tid*4;
  #pragma unroll
  for(uint i=0;i<4;++i)if(base+i<elements) {float v=float(x[offset+base+i]);sum+=v;ss=fma(v,v,ss);}
 }
 threadgroup float sums[32],squares[32];float a=simd_sum(sum),b=simd_sum(ss);
 if(tid%32==0){sums[tid/32]=a;squares[tid/32]=b;}
 if(threads>32){threadgroup_barrier(mem_flags::mem_threadgroup);if(tid<(threads+31)/32){a=simd_sum(sums[tid]);b=simd_sum(squares[tid]);if(tid==0){sums[0]=a;squares[0]=b;}}}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0){float total=sums[0],sq=squares[0],m=float(elements),mean=total/m;float variance=(sq-2*mean*total)/m+mean*mean;stats[group*2]=mean;stats[group*2+1]=precise::rsqrt(variance+eps);}
}
kernel void gn_apply_fold(device const half* x,device const half* weight,device const half* bias,device const float* stats,device half* out,
constant uint& C,constant uint& HW,constant uint& G,constant uint& K,constant ulong& N,uint i [[thread_position_in_grid]]){
 if(ulong(i)>=N)return;
 uint channel=(i/HW)%C,group=i/((C/G)*HW);ulong batch=ulong(i)/(C*HW),inner=i%(C*HW);
 float norm=(float(x[i])-stats[group*2])*stats[group*2+1];norm=norm*float(weight[channel])+float(bias[channel]);
 float v=float(half(norm));half activated=half(v/(1.0f+precise::exp(-v)));
 for(uint k=0;k<K;++k)out[(batch*K+k)*C*HW+inner]=activated;
}

#pragma clang fp contract(off)
// T=1 causal replicate padding and channel folding, fused with FP16 SiLU.
// Keep the same folded Conv2D inputs; only their construction changes.
kernel void silu_fold(device const half* x, device half* out,
    constant uint& C, constant uint& HW, constant uint& K,
    constant ulong& N, uint i [[thread_position_in_grid]]) {
  if(ulong(i)>=N)return;
  ulong b=ulong(i)/(K*C*HW), inner=ulong(i)%(C*HW);
  float v=float(x[b*C*HW+inner]);
  out[i]=half(v/(1.0f+precise::exp(-v)));
}
// GroupNorm reduction and affine order follow PyTorch's GroupNorm.metal
// (cf30153c); see THIRD_PARTY_NOTICES.md. The FP16 normalization store-round
// remains before SiLU, then causal temporal replicas are written directly.
#pragma clang fp contract(on)
kernel void groupnorm_fold(device const half* x, device const half* weight,
    device const half* bias, device half* out, constant uint& C,
    constant uint& HW, constant uint& G, constant uint& K, constant float& eps,
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const uint cpg=C/G, elements=cpg*HW;
  const ulong offset=ulong(group)*elements;
  float sum=0,ss=0;
  for(uint r=0;r<elements;r+=threads*4) {
    uint base=r+tid*4;
    #pragma unroll
    for(uint i=0;i<4;++i)if(base+i<elements) {
      float v=float(x[offset+base+i]);sum+=v;ss=fma(v,v,ss);
    }
  }
  threadgroup float sums[32],squares[32];
  float a=simd_sum(sum),b=simd_sum(ss);
  if(tid%32==0) { sums[tid/32]=a;squares[tid/32]=b; }
  if(threads>32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<(threads+31)/32) {
      a=simd_sum(sums[tid]);b=simd_sum(squares[tid]);
      if(tid==0) { sums[0]=a;squares[0]=b; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total=sums[0],sq=squares[0],m=float(elements),mean=total/m;
  float variance=(sq-2*mean*total)/m+mean*mean;
  float inv=precise::rsqrt(variance+eps);
  uint cbase=(group%G)*cpg, batch=group/G;
  for(uint r=0;r<elements;r+=threads*4) {
    uint base=r+tid*4;
    #pragma unroll
    for(uint i=0;i<4;++i)if(base+i<elements) {
      uint e=base+i,channel=cbase+e/HW;
      float value=(float(x[offset+e])-mean)*inv;
      value=value*float(weight[channel])+float(bias[channel]);
      const float rounded=float(half(value));
      const half activation=half(rounded/(1.0f+precise::exp(-rounded)));
      for(uint k=0;k<K;++k)
        out[(ulong(batch)*K+k)*C*HW+ulong(cbase)*HW+e]=activation;
    }
  }
}
#pragma clang fp contract(off)
)MTL";

static id<MTLComputePipelineState> pipeline(NSString* name) {
  static NSMutableDictionary* psos=[NSMutableDictionary new];
  static id<MTLLibrary> lib=nil;
  if(psos[name])return psos[name];
  if(!lib) {
    auto opts=[MTLCompileOptions new];
    if(@available(macOS 15.0,*))opts.mathMode=MTLMathModeSafe;else opts.fastMathEnabled=NO;
    NSError* error=nil;
    lib=[MPSDevice::getInstance()->device() newLibraryWithSource:[NSString stringWithUTF8String:source] options:opts error:&error];
    TORCH_CHECK(lib,"SeedVR2 Metal compile: ",error?error.localizedDescription.UTF8String:"");
  }
  NSError* error=nil;auto fn=[lib newFunctionWithName:name];
  TORCH_CHECK(fn,"SeedVR2 missing Metal function");
  auto p=[MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:fn error:&error];
  TORCH_CHECK(p,"SeedVR2 pipeline: ",error?error.localizedDescription.UTF8String:"");
  psos[name]=p;return p;
}
static void check(torch::Tensor x, at::ScalarType dtype) {
  TORCH_CHECK(x.is_mps() && x.scalar_type()==dtype && x.is_contiguous(), "Expected contiguous MPS tensor with requested dtype");
  TORCH_CHECK(x.numel()>0 && x.numel()<(1LL<<32),"SeedVR2 tensor size unsupported");
}
static void bind(id<MTLComputeCommandEncoder> e, torch::Tensor x, uint i) {
  auto b=__builtin_bit_cast(id<MTLBuffer>,x.storage().data());
  [e setBuffer:b offset:x.storage_offset()*x.element_size() atIndex:i];
}

torch::Tensor elementwise(torch::Tensor x, torch::Tensor a, torch::Tensor b,
                          int mode, double eps) {
  check(x,at::kBFloat16);check(a,at::kBFloat16);check(b,at::kBFloat16);
  TORCH_CHECK(x.dim()==2 && a.numel()==x.size(1),"Expected [S,D] input and [D] parameter");
  TORCH_CHECK(mode>=0 && mode<=1,"Unknown SeedVR2 mode");
  TORCH_CHECK(b.numel()==(mode==1?x.numel():x.size(1)),"Parameter shape mismatch");
  uint D=x.size(1);uint64_t N=x.numel();
  auto out=torch::empty_like(x);
  auto p=pipeline(mode==0?@"modulate":@"gate_residual");
  auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:p];
    bind(e,x,0);bind(e,a,1);bind(e,b,2);bind(e,out,3);
    [e setBytes:&D length:sizeof(D) atIndex:4];
    [e setBytes:&N length:sizeof(N) atIndex:5];
    [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});
  return out;
}

std::vector<torch::Tensor> rope(torch::Tensor q, torch::Tensor k, torch::Tensor cs, torch::Tensor sn) {
  check(q,at::kBFloat16);check(k,at::kBFloat16);check(cs,at::kFloat);check(sn,at::kFloat);
  TORCH_CHECK(q.dim()==3 && q.sizes()==k.sizes() && cs.dim()==2 && cs.sizes()==sn.sizes(),"RoPE shapes unsupported");
  uint H=q.size(1),D=q.size(2),R=cs.size(1);uint64_t N=q.numel();
  TORCH_CHECK(cs.size(0)==q.size(0) && R<=D && R%2==0 && D%2==0,"RoPE geometry unsupported");
  auto oq=torch::empty_like(q),ok=torch::empty_like(k);auto p=pipeline(@"rope_pair");auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:p];
    bind(e,q,0);bind(e,k,1);bind(e,cs,2);bind(e,sn,3);bind(e,oq,4);bind(e,ok,5);
    [e setBytes:&H length:sizeof(H) atIndex:6];[e setBytes:&D length:sizeof(D) atIndex:7];
    [e setBytes:&R length:sizeof(R) atIndex:8];[e setBytes:&N length:sizeof(N) atIndex:9];
    [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});return {oq,ok};
}

torch::Tensor silu_fold(torch::Tensor x, int64_t taps) {
  check(x,at::kHalf);TORCH_CHECK(x.dim()==4 && taps>0 && taps<=3,"Expected NCHW FP16 and 1-3 taps");
  uint C=x.size(1),HW=x.size(2)*x.size(3),K=taps;
  auto out=torch::empty({x.size(0),taps*x.size(1),x.size(2),x.size(3)},x.options());
  uint64_t N=out.numel();TORCH_CHECK(N<(1LL<<32),"Folded activation too large");
  auto p=pipeline(@"silu_fold");auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,out,1);
    [e setBytes:&C length:sizeof(C) atIndex:2];[e setBytes:&HW length:sizeof(HW) atIndex:3];
    [e setBytes:&K length:sizeof(K) atIndex:4];[e setBytes:&N length:sizeof(N) atIndex:5];
    [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});return out;
}

torch::Tensor rms_square(torch::Tensor x) {
  check(x,at::kBFloat16);auto out=torch::empty_like(x,x.options().dtype(at::kFloat));
  uint64_t N=x.numel();auto p=pipeline(@"rms_square");auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:p];bind(e,x,0);bind(e,out,1);
    [e setBytes:&N length:sizeof(N) atIndex:2];
    [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});return out;
}
torch::Tensor rms_apply(torch::Tensor x,torch::Tensor inv,torch::Tensor sc,torch::Tensor sh) {
  check(x,at::kBFloat16);check(inv,at::kFloat);check(sc,at::kBFloat16);check(sh,at::kBFloat16);
  TORCH_CHECK(x.dim()==2 && inv.numel()==x.size(0) && sc.numel()==x.size(1) && sh.numel()==x.size(1),"RMS apply shapes");
  auto out=torch::empty_like(x);uint D=x.size(1);uint64_t N=x.numel();
  auto p=pipeline(@"rms_apply");auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:p];
    bind(e,x,0);bind(e,inv,1);bind(e,sc,2);bind(e,sh,3);bind(e,out,4);
    [e setBytes:&D length:sizeof(D) atIndex:5];[e setBytes:&N length:sizeof(N) atIndex:6];
    [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});return out;
}
torch::Tensor groupnorm_fold(torch::Tensor x,torch::Tensor w,torch::Tensor b,
                             int64_t groups,double eps,int64_t taps) {
  check(x,at::kHalf);check(w,at::kHalf);check(b,at::kHalf);
  TORCH_CHECK(x.dim()==4 && groups>0 && x.size(1)%groups==0 && taps>=1 && taps<=3
              && w.numel()==x.size(1) && b.numel()==x.size(1),"GroupNorm geometry");
  uint C=x.size(1),HW=x.size(2)*x.size(3),G=groups,K=taps;float epsilon=eps;
  auto out=torch::empty({x.size(0),C*K,x.size(2),x.size(3)},x.options());
  uint elements=(C/G)*HW;uint threads=std::min(1024u,((elements+3)/4+31)/32*32);
  auto p=pipeline(@"groupnorm_fold");auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:p];
    bind(e,x,0);bind(e,w,1);bind(e,b,2);bind(e,out,3);
    [e setBytes:&C length:sizeof(C) atIndex:4];[e setBytes:&HW length:sizeof(HW) atIndex:5];
    [e setBytes:&G length:sizeof(G) atIndex:6];[e setBytes:&K length:sizeof(K) atIndex:7];
    [e setBytes:&epsilon length:sizeof(epsilon) atIndex:8];
    [e dispatchThreadgroups:MTLSizeMake(x.size(0)*G,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
  }});return out;
}

// Preserve PyTorch's FP32 statistics and reduction order, but distribute the
// rounded FP16 affine + SiLU + causal replicas over the full GPU. The two
// temporary statistics per group never scale with image area.
torch::Tensor groupnorm_split_fold(torch::Tensor x,torch::Tensor w,torch::Tensor b,
                                  int64_t groups,double eps,int64_t taps) {
  check(x,at::kHalf);check(w,at::kHalf);check(b,at::kHalf);
  TORCH_CHECK(x.dim()==4 && groups>0 && x.size(1)%groups==0 && taps>=1 && taps<=3
             && w.numel()==x.size(1) && b.numel()==x.size(1),"GroupNorm geometry");
  uint C=x.size(1),HW=x.size(2)*x.size(3),G=groups,K=taps;float epsilon=eps;
  uint64_t N=x.numel();
  auto stats=torch::empty({x.size(0)*groups,2},x.options().dtype(at::kFloat));
  auto out=torch::empty({x.size(0),C*K,x.size(2),x.size(3)},x.options());
  TORCH_CHECK(out.numel()<(1LL<<32),"Folded activation too large");
  uint elements=(C/G)*HW,threads=std::min(1024u,((elements+3)/4+31)/32*32);
  auto ps=pipeline(@"gn_stats"),pa=pipeline(@"gn_apply_fold");
  auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:ps];bind(e,x,0);bind(e,stats,1);
    [e setBytes:&C length:sizeof(C) atIndex:2];[e setBytes:&HW length:sizeof(HW) atIndex:3];
    [e setBytes:&G length:sizeof(G) atIndex:4];[e setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(x.size(0)*G,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [e setComputePipelineState:pa];bind(e,x,0);bind(e,w,1);bind(e,b,2);bind(e,stats,3);bind(e,out,4);
    [e setBytes:&C length:sizeof(C) atIndex:5];[e setBytes:&HW length:sizeof(HW) atIndex:6];
    [e setBytes:&G length:sizeof(G) atIndex:7];[e setBytes:&K length:sizeof(K) atIndex:8];
    [e setBytes:&N length:sizeof(N) atIndex:9];
    [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});return out;
}

torch::Tensor gather_rows(torch::Tensor a, torch::Tensor b, torch::Tensor ix) {
  TORCH_CHECK(a.is_mps() && b.is_mps() && ix.is_mps()
      && a.dim()>=2 && b.dim()==a.dim() && a.scalar_type()==b.scalar_type()
      && (a.scalar_type()==at::kHalf || a.scalar_type()==at::kBFloat16)
      && ix.scalar_type()==at::kLong && ix.dim()==1 && ix.is_contiguous(), "Gather types");
  uint64_t C=1;
  for(int d=a.dim()-1;d>=1;--d) {
    TORCH_CHECK(a.size(d)==b.size(d) && a.stride(d)==C && b.stride(d)==C,"Gather row layout");
    C*=a.size(d);
  }
  TORCH_CHECK(C<(1LL<<32) && ix.numel()<(1LL<<32),"Gather grid too large");
  TORCH_CHECK(ix.numel()>0 && ix.min().item<int64_t>()>=0
      && ix.max().item<int64_t>()<a.size(0)+b.size(0),"Gather index out of bounds");
  auto shape=a.sizes().vec();shape[0]=ix.numel();
  auto out=torch::empty(shape,a.options());
  uint64_t A=a.size(0),SA=a.stride(0),SB=b.stride(0);
  auto p=pipeline(@"gather_rows");auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto e=stream->commandEncoder();[e setComputePipelineState:p];
    bind(e,a,0);bind(e,b,1);bind(e,ix,2);bind(e,out,3);
    [e setBytes:&A length:sizeof(A) atIndex:4];[e setBytes:&SA length:sizeof(SA) atIndex:5];
    [e setBytes:&SB length:sizeof(SB) atIndex:6];[e setBytes:&C length:sizeof(C) atIndex:7];
    [e dispatchThreads:MTLSizeMake(C,ix.numel(),1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("gather_rows",&gather_rows);
  m.def("elementwise",&elementwise);m.def("rope",&rope);m.def("silu_fold",&silu_fold);
  m.def("rms_square",&rms_square);m.def("rms_apply",&rms_apply);m.def("groupnorm_fold",&groupnorm_fold);
  m.def("groupnorm_split_fold",&groupnorm_split_fold);
}
