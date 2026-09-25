// Krea2 exact memory-pass fusions. RMS reduction order follows PyTorch
// RMSNorm.metal (adapted from MLX, Copyright 2024 Apple Inc.).
// See THIRD_PARTY_NOTICES.md, PYTORCH_LICENSE and MLX_LICENSE.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
using namespace at::mps;
static const char* source=R"MTL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline bfloat rte(float f) {
  uint u=as_type<uint>(f);u+=0x7fffu+((u>>16)&1u);
  return as_type<bfloat>(ushort(u>>16));
}
kernel void norm_mod(device const bfloat* x,device const float* weight,
  device const bfloat* sc,device const bfloat* sh,device bfloat* out,
  constant uint& D,constant float& eps,constant uint& mod,
  uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
  uint nt [[threads_per_threadgroup]],uint lane [[thread_index_in_simdgroup]],
  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[32],inv;
  float acc=0;
  for(uint r=0;r<D;r+=nt*4) {
    for(uint j=0;j<4;j++)if(r+tid*4+j<D) {
      float v=float(x[ulong(row)*D+r+tid*4+j]);acc+=v*v;
    }
  }
  acc=simd_sum(acc);
  if(sg==0)sums[lane]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0)sums[sg]=acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) {
    acc=simd_sum(sums[lane]);
    if(lane==0)inv=precise::rsqrt(acc/float(D)+eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint r=0;r<D;r+=nt*4) {
    for(uint j=0;j<4;j++)if(r+tid*4+j<D) {
      uint d=r+tid*4+j;ulong i=ulong(row)*D+d;
      bfloat value=rte((1.0f+weight[d])*(float(x[i])*inv));
      if(mod)value=rte(float(rte(float(value)*float(rte(1.0f+float(sc[d])))))+float(sh[d]));
      out[i]=value;
    }
  }
}
kernel void point(device const bfloat* x,device const bfloat* a,
  device const bfloat* b,device bfloat* out,constant uint& D,
  constant uint& H,constant uint& S,constant uint& mode,constant ulong& N,
  uint idx [[thread_position_in_grid]]) {
  if(ulong(idx)>=N)return;
  ulong i=idx;
  if(mode==0)out[i]=rte(float(rte(float(x[i])*float(a[i%D])))+float(b[i]));
  else if(mode==1 || mode==2) {
    uint d=i%D,h=(i/D)%H,s=(i/(D*H))%S,batch=i/(D*H*S);
    ulong src=((ulong(batch)*H+h)*S+s)*D+d;
    bfloat gate=mode==1 ? rte(1.0f/(1.0f+precise::exp(-float(a[i])))) : a[i];
    out[i]=rte(float(x[src])*float(gate));
  } else {
    float v=float(x[i]);bfloat silu=rte(v/(1.0f+precise::exp(-v)));
    out[i]=rte(float(silu)*float(a[i]));
  }
}
)MTL";
static id<MTLComputePipelineState> pipeline(NSString* name) {
  static NSMutableDictionary* states=[NSMutableDictionary new];
  static id<MTLLibrary> lib=nil;
  if(states[name])return states[name];
  if(!lib) {
    auto opts=[MTLCompileOptions new];opts.mathMode=MTLMathModeSafe;
    NSError* error=nil;lib=[MPSDevice::getInstance()->device() newLibraryWithSource:[NSString stringWithUTF8String:source] options:opts error:&error];
    TORCH_CHECK(lib,"Krea2 compile: ",error?error.localizedDescription.UTF8String:"");
  }
  NSError* error=nil;auto ps=[MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];
  TORCH_CHECK(ps,"Krea2 pipeline: ",error?error.localizedDescription.UTF8String:"");states[name]=ps;return ps;
}
static void bind(id<MTLComputeCommandEncoder> enc,torch::Tensor t,uint slot) {
  auto b=__builtin_bit_cast(id<MTLBuffer>,t.storage().data());
  [enc setBuffer:b offset:t.storage_offset()*t.element_size() atIndex:slot];
}
static void check(torch::Tensor t,at::ScalarType dtype) {
  TORCH_CHECK(t.is_mps() && t.is_contiguous() && t.scalar_type()==dtype,"Expected contiguous MPS tensor");
  TORCH_CHECK(t.numel()>0 && t.numel()<(1LL<<32),"Unsupported tensor size");
}
torch::Tensor norm_mod(torch::Tensor x,torch::Tensor weight,torch::Tensor scale,torch::Tensor shift,double epsilon,bool mod) {
  check(x,at::kBFloat16);check(weight,at::kFloat);check(scale,at::kBFloat16);check(shift,at::kBFloat16);
  uint D=x.size(-1),rows=x.numel()/D,m=mod;float eps=epsilon;
  TORCH_CHECK(D<=32768 && weight.numel()==D && scale.numel()==D && shift.numel()==D,"Expected channel vectors");
  auto out=torch::empty_like(x);auto ps=pipeline(@"norm_mod");
  uint nt=D<=4096 ? ((D+127)/128)*32 : 1024;
  TORCH_CHECK(nt<=ps.maxTotalThreadsPerThreadgroup,"Unsupported thread count");
  auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto enc=stream->commandEncoder();[enc setComputePipelineState:ps];
    bind(enc,x,0);bind(enc,weight,1);bind(enc,scale,2);bind(enc,shift,3);bind(enc,out,4);
    [enc setBytes:&D length:4 atIndex:5];[enc setBytes:&eps length:4 atIndex:6];[enc setBytes:&m length:4 atIndex:7];
    [enc dispatchThreads:MTLSizeMake(uint64_t(rows)*nt,1,1) threadsPerThreadgroup:MTLSizeMake(nt,1,1)];
  }});return out;
}
torch::Tensor point(torch::Tensor x,torch::Tensor a,torch::Tensor b,int mode) {
  check(x,at::kBFloat16);check(a,at::kBFloat16);check(b,at::kBFloat16);
  uint D=x.size(-1),H=1,S=1,m=mode;uint64_t N=x.numel();
  TORCH_CHECK(mode>=0 && mode<=3,"Unknown mode");
  auto shape=x.sizes().vec();
  if(mode==1 || mode==2) {
    TORCH_CHECK(x.dim()==4 && a.dim()==3 && a.numel()==x.numel(),"Attention layout mismatch");
    H=x.size(1);S=x.size(2);shape={x.size(0),S,H*D};
    TORCH_CHECK(a.size(0)==x.size(0) && a.size(1)==S && a.size(2)==H*D,"Gate shape mismatch");
  } else if(mode==0) {
    TORCH_CHECK(a.numel()==D && b.sizes()==x.sizes(),"Residual shape mismatch");
  } else TORCH_CHECK(a.sizes()==x.sizes(),"SwiGLU shape mismatch");
  auto out=torch::empty(shape,x.options());auto ps=pipeline(@"point");auto stream=getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^(){@autoreleasepool {
    auto enc=stream->commandEncoder();[enc setComputePipelineState:ps];
    bind(enc,x,0);bind(enc,a,1);bind(enc,b,2);bind(enc,out,3);
    [enc setBytes:&D length:4 atIndex:4];[enc setBytes:&H length:4 atIndex:5];[enc setBytes:&S length:4 atIndex:6];
    [enc setBytes:&m length:4 atIndex:7];[enc setBytes:&N length:8 atIndex:8];
    [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("norm_mod",&norm_mod);m.def("point",&point);
  m.def("device_name",[](){return std::string([MPSDevice::getInstance()->device().name UTF8String]);});
}
