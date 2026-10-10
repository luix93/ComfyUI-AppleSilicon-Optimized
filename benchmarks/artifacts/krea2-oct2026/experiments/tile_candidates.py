from pathlib import Path
import os,sys,json,importlib
MODE='baseline';stats={};original=None;candidate=None
VARIANTS={'tile64':1,'tile64_3d':2,'tile64_nofence':3,'tile64_sw0':4,'tile64_sk32':5,'tile128_3d':6,'tile128_nofence':7,'tile128_sw0':8}
def set_mode(mode):
 global MODE
 MODE=mode;stats.clear()
def build():
 import torch.utils.cpp_extension as cpp
 patches={n.rsplit('.',1)[-1]:m for n,m in list(sys.modules.items()) if '._patches.' in n}
 patch=patches['int8_linear_kernel_mps'];root=Path(patch.__file__).parent
 loader=importlib.import_module(patch.__package__+'.int8_ext.loader')
 src=(root/'int8_ext/int8_gemm.mm').read_text()
 start=src.index('kernel void int8_matmul_fused_cs(');end=src.index('// Long-M channelwise path.',start)
 kernel=src[start:end]
 # Keep the existing epilogue verbatim. Only independent output tile ownership changes.
 extra=[]
 for name,vid in VARIANTS.items():
  k=kernel.replace('int8_matmul_fused_cs(',f'int8_matmul_screen_{vid}(')
  if vid not in (6,7,8):k=k.replace('((128 / 4) / 16)','((64 / 2) / 16)',1).replace('<128, 128, 512, 32, 4, 4, true>','<64, 128, 512, 32, 2, 4, true>')
  if vid in (3,7):k=k.replace('4, true>','4, false>')
  if vid==5:k=k.replace('w8a8_gemm_compute<','w8a8_gemm_screen32<')
  extra.append(k)
 hs=src.index('template <int BM, int BN, int DEFAULT_BK');he=src.index('// ── INT32 entry points',hs)
 helper=src[hs:he].replace('w8a8_gemm_compute(', 'w8a8_gemm_screen32(').replace('constexpr int SK = 16;', 'constexpr int SK = 32;').replace('constexpr int BK = 256;', 'constexpr int BK = 512;')
 src=src[:start]+helper+'\n'.join(extra)+src[start:]
 start=src.index('torch::Tensor i8_matmul2d_nt_fused(');end=src.index('static id<MTLComputePipelineState> g_pso_fused_cs',start)
 host=src[start:end].replace('i8_matmul2d_nt_fused(', 'i8_matmul2d_nt_screen(').replace('c10::optional<torch::Tensor> bias_opt)', 'c10::optional<torch::Tensor> bias_opt, int variant)')
 host=host.replace('Geom g = geom_for(M, N);\n  id<MTLComputePipelineState> pso = pso_for_fused(g.small);', '''Geom g = geom_for(M,N);
  if (M > 128 && variant < 6) {
    g.THREADS=256; g.tiles_m=(M+63)/64;
    g.swizzle_log=variant==4 ? 0 : 2;
    uint tile=1u<<g.swizzle_log;g.grid_x=g.tiles_n*tile;g.grid_y=(g.tiles_m+tile-1)/tile;
  }
  if (variant==8) {g.swizzle_log=0;g.grid_x=g.tiles_n;g.grid_y=g.tiles_m;}
  static id<MTLComputePipelineState> pipelines[9]={nil};
  if(!pipelines[variant])pipelines[variant]=pso_for_name([NSString stringWithFormat:@"int8_matmul_screen_%d",variant]);
  id<MTLComputePipelineState> pso = M<=128 ? pso_for_fused(g.small) : pipelines[variant];''')
 host=host.replace('MTLSizeMake(g.THREADS, 1, 1)', '((M>128 && (variant==2 || variant==6)) ? MTLSizeMake(32,4,g.THREADS/128) : MTLSizeMake(g.THREADS,1,1))')
 host=host.replace('c10::optional<torch::Tensor> bias_opt, int variant)', 'torch::Tensor col_scale, c10::optional<torch::Tensor> bias_opt, int variant)')
 host=host.replace('MPSStream* stream = getCurrentMPSStream();', 'id<MTLBuffer> colBuf=__builtin_bit_cast(id<MTLBuffer>, col_scale.storage().data());\n  const NSUInteger colOff=col_scale.storage_offset()*col_scale.element_size();\n  MPSStream* stream = getCurrentMPSStream();')
 host=host.replace('[enc setBuffer:biasBuf offset:biasOff atIndex:10];', '[enc setBuffer:biasBuf offset:biasOff atIndex:10];\n      [enc setBuffer:colBuf offset:colOff atIndex:12];')
 src=src[:end]+host+src[end:]
 src=src.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {','PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def("screen", &i8_matmul2d_nt_screen);')
 directory=Path(__file__).parent/'tile-build';directory.mkdir(exist_ok=True);(directory/'kernel.mm').write_text(src)
 saved=cpp.TORCH_LIB_PATH
 try:
  cpp.TORCH_LIB_PATH=loader._nospace_torch_lib()
  mod=cpp.load(name='asfp8_krea_tiles_oct10',sources=[str(directory/'kernel.mm')],extra_cflags=['-ObjC++'],extra_ldflags=['-framework','Metal','-framework','Foundation'],build_directory=str(directory),verbose=False)
 finally:cpp.TORCH_LIB_PATH=saved
 return patch,mod

def install():
 global original,candidate
 import torch
 patch,candidate=build();original=patch._kernel.i8_matmul2d_nt_fused_cs
 captured=set()
 def run(*args,**kwargs):
  shape=(args[0].shape[0],args[1].shape[0],args[1].shape[1]);key='x'.join(map(str,shape))
  if MODE=='capture' and key not in captured:
   captured.add(key);torch.save([x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in args],Path(__file__).parent/(key+'-gemm.pt'))
  mode=MODE.removesuffix('_verify')
  if mode=='mixed':mode='tile64_sw0' if shape[1] in (1536,6144) else 'baseline'
  if mode=='selective':mode='tile64_sw0' if shape[1] in (1536,6144) and shape[2]==6144 else 'baseline'
  if mode=='selective_all':mode='tile128_3d' if shape[2]==16384 else 'tile64_sw0'
  if mode not in VARIANTS:return original(*args,**kwargs)
  out=candidate.screen(*args[:5],VARIANTS[mode]);stats[key]=stats.get(key,0)+1
  if MODE.endswith('_verify'):
   ref=original(*args,**kwargs)
   assert torch.equal(out.cpu().view(torch.uint8),ref.cpu().view(torch.uint8)),f'INT8 tile divergence {key}'
  return out
 patch._kernel.i8_matmul2d_nt_fused_cs=run
