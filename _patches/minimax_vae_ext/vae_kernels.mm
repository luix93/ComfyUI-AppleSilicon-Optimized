// FP16 MiniMax VAE glue. Matrix multiplies and attention stay in ComfyUI.
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

// Match PyTorch's weighted MPS RMSNorm: four consecutive values per lane,
// SIMD sum, FP32 rsqrt, FP16 normalization round, FP16 weight multiply.
// The decoder packs [head, qkv, dim], unlike the denoiser's [qkv, head, dim].
kernel void qkv_fp16(
    device const half* x [[buffer(0)]],
    device const half* rope [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* q [[buffer(3)]],
    device half* k [[buffer(4)]],
    device half* v [[buffer(5)]],
    constant uint& S [[buffer(6)]],
    constant uint& H [[buffer(7)]],
    constant uint& R [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint bs = group.x, head = group.y / 3, slot = group.y % 3;
  const uint batch = bs / S, token = bs % S;
  const ulong base = ((ulong)bs * H + head) * 192 + slot * 64;
  const ulong dst = (((ulong)batch * H + head) * S + token) * 64;
  if (slot == 2) {
    for (uint d = lane; d < 64; d += 32) v[dst + d] = x[base + d];
    return;
  }
  float acc = 0.0f;
  if (lane < 16) {
    for (uint i = 0; i < 4; ++i) {
      float a = float(x[base + lane * 4 + i]);
      acc += a * a;
    }
  }
  acc = simd_sum(acc);
  const float inv = precise::rsqrt(acc / 64.0f + eps);
  threadgroup half norm[64];
  for (uint d = lane; d < 64; d += 32)
    norm[d] = weight[d] * half(float(x[base + d]) * inv);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device half* out = slot == 0 ? q : k;
  for (uint d = lane; d < 64; d += 32) {
    half y = norm[d];
    if (d < 2 * R) {
      const uint pair = d % R, side = d / R;
      const ulong f = ((ulong)bs * R + pair) * 4;
      // Kitchen first rounds the diagonal product, then addcmul rounds the
      // sum of that product and the FP32 off-diagonal product back to half.
      const half diagonal = y * rope[f + (side == 0 ? 0 : 3)];
      y = half(float(diagonal) + float(norm[side == 0 ? d + R : d - R])
                                * float(rope[f + (side == 0 ? 1 : 2)]));
    }
    out[dst + d] = y;
  }
}

kernel void swiglu_fp16(
    device const half* x [[buffer(0)]], device half* out [[buffer(1)]],
    constant uint& width [[buffer(2)]], constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  const ulong offset = (ulong)(gid / width) * (width * 2) + gid % width;
  const float gate = float(x[offset]);
  const half activation = half(gate / (1.0f + precise::exp(-gate)));
  out[gid] = activation * x[offset + width];
}

kernel void merge_sanitize_fp16(
    device const half* x [[buffer(0)]], device half* out [[buffer(1)]],
    constant uint& S [[buffer(2)]], constant uint& H [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  const uint d=gid%D, h=(gid/D)%H, s=(gid/(D*H))%S, b=gid/(D*H*S);
  half value=x[(((ulong)b*H+h)*S+s)*D+d];
  if (isnan(value)) value=half(0);
  else if (isinf(value)) value=value>0 ? half(65504) : half(-65504);
  out[gid]=value;
}
)MTL";

static id<MTLComputePipelineState> pipeline(NSString* name) {
  static NSMutableDictionary* pipelines = [NSMutableDictionary new];
  static id<MTLLibrary> lib = nil;
  id<MTLComputePipelineState> cached = pipelines[name];
  if (cached) return cached;
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  if (!lib) {
    MTLCompileOptions* opts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opts.mathMode = MTLMathModeSafe;
    else opts.fastMathEnabled = NO;
    NSError* error = nil;
    lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                               options:opts error:&error];
    TORCH_CHECK(lib, "MiniMax VAE Metal compilation failed: ",
                error ? error.localizedDescription.UTF8String : "unknown");
  }
  id<MTLFunction> function = [lib newFunctionWithName:name];
  TORCH_CHECK(function, "Missing VAE kernel: ", name.UTF8String);
  NSError* error = nil;
  auto result = [device newComputePipelineStateWithFunction:function error:&error];
  TORCH_CHECK(result, "MiniMax VAE pipeline failed: ",
              error ? error.localizedDescription.UTF8String : "unknown");
  pipelines[name] = result;
  return result;
}

static void check_half(const torch::Tensor& t) {
  TORCH_CHECK(t.is_mps() && t.scalar_type() == torch::kHalf && t.is_contiguous(),
              "Expected contiguous FP16 MPS tensor");
}

std::vector<torch::Tensor> qkv(torch::Tensor x, torch::Tensor rope,
                              torch::Tensor weight, int64_t heads, double eps) {
  check_half(x); check_half(rope); check_half(weight);
  TORCH_CHECK(x.dim() == 3 && heads > 0 && x.size(2) == heads * 192,
              "Expected [B,S,H*3*64] QKV");
  TORCH_CHECK(weight.numel() == 64, "Expected 64 norm weights");
  const int64_t B = x.size(0), L = x.size(1);
  TORCH_CHECK(B > 0 && L > 0 && rope.dim() == 6 && rope.size(0) == B &&
              rope.size(1) == L && rope.size(2) == 1 && rope.size(4) == 2 &&
              rope.size(5) == 2 && rope.size(3) > 0 && rope.size(3) <= 32,
              "Expected [B,S,1,R,2,2] rotary table, 0 < R <= 32");
  auto q = torch::empty({B,heads,L,64}, x.options());
  auto k = torch::empty_like(q), v = torch::empty_like(q);
  auto pso = pipeline(@"qkv_fp16");
  auto stream = getCurrentMPSStream();
  std::array<id<MTLBuffer>,6> buffers;
  std::array<NSUInteger,6> offsets;
  torch::Tensor tensors[6] = {x,rope,weight,q,k,v};
  for (int i=0; i<6; ++i) {
    buffers[i] = __builtin_bit_cast(id<MTLBuffer>, tensors[i].storage().data());
    offsets[i] = tensors[i].storage_offset() * tensors[i].element_size();
  }
  uint S=(uint)L, H=(uint)heads, R=(uint)rope.size(3);
  float epsilon=(float)eps;
  dispatch_sync(stream->queue(), ^(){ @autoreleasepool {
    auto enc = stream->commandEncoder();
    [enc setComputePipelineState:pso];
    for (int i=0; i<6; ++i) [enc setBuffer:buffers[i] offset:offsets[i] atIndex:i];
    [enc setBytes:&S length:sizeof(S) atIndex:6];
    [enc setBytes:&H length:sizeof(H) atIndex:7];
    [enc setBytes:&R length:sizeof(R) atIndex:8];
    [enc setBytes:&epsilon length:sizeof(epsilon) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(B*L, heads*3, 1)
        threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  }});
  return {q,k,v};
}

torch::Tensor swiglu(torch::Tensor x) {
  check_half(x);
  TORCH_CHECK(x.dim()==3 && x.size(2)>0 && x.size(2)%2==0 &&
              x.numel()>0 && x.numel()/2 < (1LL<<32), "Expected [B,S,2F] FP16 tensor");
  auto out=torch::empty({x.size(0),x.size(1),x.size(2)/2},x.options());
  auto pso=pipeline(@"swiglu_fp16"); auto stream=getCurrentMPSStream();
  auto xb=__builtin_bit_cast(id<MTLBuffer>,x.storage().data());
  auto ob=__builtin_bit_cast(id<MTLBuffer>,out.storage().data());
  NSUInteger xo=x.storage_offset()*x.element_size();
  uint width=(uint)out.size(2), count=(uint)out.numel();
  dispatch_sync(stream->queue(), ^(){ @autoreleasepool {
    auto enc=stream->commandEncoder(); [enc setComputePipelineState:pso];
    [enc setBuffer:xb offset:xo atIndex:0]; [enc setBuffer:ob offset:0 atIndex:1];
    [enc setBytes:&width length:sizeof(width) atIndex:2];
    [enc setBytes:&count length:sizeof(count) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});
  return out;
}

torch::Tensor merge_sanitize(torch::Tensor x) {
  check_half(x);
  TORCH_CHECK(x.dim()==4 && x.numel()>0 && x.numel()<(1LL<<32),
              "Expected nonempty [B,H,S,D] FP16 tensor");
  auto out=torch::empty({x.size(0),x.size(2),x.size(1)*x.size(3)},x.options());
  auto pso=pipeline(@"merge_sanitize_fp16"); auto stream=getCurrentMPSStream();
  auto xb=__builtin_bit_cast(id<MTLBuffer>,x.storage().data());
  auto ob=__builtin_bit_cast(id<MTLBuffer>,out.storage().data());
  NSUInteger xo=x.storage_offset()*x.element_size();
  uint S=x.size(2),H=x.size(1),D=x.size(3),count=x.numel();
  dispatch_sync(stream->queue(), ^(){ @autoreleasepool {
    auto enc=stream->commandEncoder(); [enc setComputePipelineState:pso];
    [enc setBuffer:xb offset:xo atIndex:0]; [enc setBuffer:ob offset:0 atIndex:1];
    [enc setBytes:&S length:sizeof(S) atIndex:2];
    [enc setBytes:&H length:sizeof(H) atIndex:3];
    [enc setBytes:&D length:sizeof(D) atIndex:4];
    [enc setBytes:&count length:sizeof(count) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }});
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkv", &qkv, "FP16 H3 VAE QK RMSNorm, partial RoPE and attention layout");
  m.def("swiglu", &swiglu, "FP16 SiLU then multiply from adjacent projection halves");
  m.def("merge_sanitize", &merge_sanitize, "Merge attention heads and apply FP16 nan_to_num");
}
