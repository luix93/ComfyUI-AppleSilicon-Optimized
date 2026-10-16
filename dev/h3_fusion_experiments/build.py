from pathlib import Path
import os,sys,tempfile,hashlib
D=Path(__file__).resolve().parent;R=D.parents[1]
sys.path.insert(0,str(R))
def build():
 import torch.utils.cpp_extension as cpp
 from _patches.int8_ext import loader
 # Initialize the ordinary loader first, including its compiler/ninja setup.
 os.environ['ASFP8_INT8_EXT']='1';base=loader.module();assert base is not None
 src=(R/'_patches/int8_ext/int8_gemm.mm').read_text()
 start=src.index('template <bool USE_COMPILER_BARRIER>\ninline void int8_matmul_fused_cs_qkv_rms_rope_impl')
 end=src.index('kernel void int8_matmul_fused_cs_qkv_rms_rope(',start)
 body=src[start:end]
 start=src.index('kernel void int8_matmul_fused_cs_qkv_rms_rope_long(')
 end=src.index('\n}',start)+2
 entry=src[start:end]
 extra=[]
 for bm in [32,64]:
  wm=bm//32;groups=wm*4
  h=body.replace('int8_matmul_fused_cs_qkv_rms_rope_impl',f'qkv_{bm}_impl').replace('QKVEpilogueScratch',f'QScratch{bm}')
  h=h.replace('TM = ((128 / 4)',f'TM = (({bm} / {wm})').replace('w8a8_gemm_compute<128, 128, 512, 32, 4, 4,',f'w8a8_gemm_compute<{bm}, 128, 512, 32, {wm}, 4,')
  h=h.replace('row_in_tile < 128u; row_in_tile += 16u',f'row_in_tile < {bm}u; row_in_tile += {groups}u')
  e=entry.replace('int8_matmul_fused_cs_qkv_rms_rope_long',f'qkv_{bm}').replace('int8_matmul_fused_cs_qkv_rms_rope_impl',f'qkv_{bm}_impl').replace('QKVEpilogueScratch',f'QScratch{bm}')
  extra.append(f'union QScratch{bm} {{ bfloat value[{bm}*128]; float sum[{bm}*64]; }};\n'+h+e)
 # A separate 128-row path tests K blocking without affecting other kernels.
 hs=src.index('template <int BM, int BN, int DEFAULT_BK');he=src.index('// ── INT32 entry points',hs)
 helper=src[hs:he].replace('w8a8_gemm_compute(', 'w8a8_bk512(').replace('constexpr int BK = 256;', 'constexpr int BK = 512;')
 h=body.replace('int8_matmul_fused_cs_qkv_rms_rope_impl','qkv_bk512_impl').replace('w8a8_gemm_compute<','w8a8_bk512<')
 e=entry.replace('int8_matmul_fused_cs_qkv_rms_rope_long','qkv_bk512').replace('int8_matmul_fused_cs_qkv_rms_rope_impl','qkv_bk512_impl')
 extra.append(helper+h+e)
 extra.append((D/'fc1.metal').read_text())
 extra.append((D/'fc1_pair.metal').read_text())
 extra.append((D/'fc1_dual.metal').read_text())
 ks=src.index('kernel void i8_swiglu_rot_max_inplace_bf16(');ke=src.index('kernel void i8_quant_rowwise_premax_strided_bf16(',ks)
 rot=src[ks:ke].replace('i8_swiglu_rot_max_inplace_bf16','rot_compact').replace('(2ul * F)','F')
 begin=rot.index('  const float av =');end=rot.index('  threadgroup_barrier',begin)
 rot=rot[:begin]+'  work[lid] = float(xr[i]);\n'+rot[end:]
 extra.append(rot)
 # Same premax/row quantization, now reading compact [M,F] storage.
 ks=src.index('kernel void i8_quant_rowwise_premax_strided_bf16(');ke=src.index('// ── Rowwise int8 activation quant',ks)
 extra.append(src[ks:ke].replace('i8_quant_rowwise_premax_strided_bf16','quant_compact').replace('(2ul * F)','F'))
 src=src.replace(')MTL";', '\n'.join(extra)+'\n)MTL";')
 hs=src.index('std::vector<torch::Tensor> i8_matmul2d_nt_fused_cs_qkv_rms_rope(');he=src.index('torch::Tensor i8_matmul2d_nt_fused_cs_qkv_lora_part(',hs)
 host=src[hs:he].replace('i8_matmul2d_nt_fused_cs_qkv_rms_rope(', 'qkv_variant(')
 host=host.replace('bool long_schedule)', 'bool long_schedule, int variant)')
 host=host.replace('  const bool use_long = long_schedule && M >= 8192;','''  const bool use_long = long_schedule && M >= 8192;
  if(variant==32 || variant==64) {
    g.THREADS=variant*4; g.tiles_m=(M+variant-1)/variant;
    g.grid_y=(g.tiles_m+3)/4; g.swizzle_log=2; g.grid_x=g.tiles_n*4;
  }''')
 a=host.index('  id<MTLComputePipelineState> pso = pso_for_name(');b=host.index('  MPSStream* stream',a)
 host=host[:a]+'''  static id<MTLComputePipelineState> pv[3]={nil};
  int slot=variant==32?0:variant==64?1:2;
  if(!pv[slot]) pv[slot]=pso_for_name(variant==32?@"qkv_32":variant==64?@"qkv_64":@"qkv_bk512");
  id<MTLComputePipelineState> pso=pv[slot];
'''+host[b:]
 src=src[:he]+host+src[he:]
 src=src.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',(D/'fc1_host.cpp').read_text()+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n m.def("qkv_variant", &qkv_variant);\n m.def("fc1_fused", &fc1_fused);')
 out=Path(tempfile.gettempdir())/('asfp8-h3-fusions-'+hashlib.sha256(src.encode()).hexdigest()[:16]);out.mkdir(exist_ok=True);(out/'kernel.mm').write_text(src)
 saved=cpp.TORCH_LIB_PATH
 try:
  cpp.TORCH_LIB_PATH=loader._nospace_torch_lib()
  mod=cpp.load(name='asfp8_h3_fusions_oct14',sources=[str(out/'kernel.mm')],extra_cflags=['-ObjC++'],extra_ldflags=['-framework','Metal','-framework','Foundation'],build_directory=str(out),verbose=False)
 finally:cpp.TORCH_LIB_PATH=saved
 return base,mod
