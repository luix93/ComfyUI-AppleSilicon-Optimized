"""Profiling-guided experiments; never imported by production."""
from pathlib import Path
import sys, functools, torch
D=Path(__file__).resolve().parent
R=next(p for p in D.parents if (p/'_patches/torch_mpp_qcache.py').is_file())
sys.path.insert(0,str(R))
from _patches import torch_mpp_qcache as qc
PACK='''
#include <metal_stdlib>
using namespace metal;
kernel void pack_kv(const device ushort4* src [[buffer(0)]], device ushort4* dst [[buffer(1)]], constant uint& n [[buffer(2)]], constant uint& width [[buffer(3)]], uint i [[thread_position_in_grid]]) {
 uint blocks=(n+31)/32, w=width/4;
 uint col=i%w, row=(i/w)%32, dim=(i/(w*32))%(128/width), tile=(i/(32*32))%blocks, head=i/(blocks*32*32);
 uint token=tile*32+row;
 dst[i]=token<n ? src[(head*n+token)*32+dim*w+col] : ushort4(0);
}
'''
@functools.lru_cache(None)
def source(name):
 s=qc._source(h3_long=True)
 start=s.index('prefill_attention_mpp(');pre,body=s[:start],s[start:]
 if name in ('pack_k','pack_kv'):
  body=body.replace('K += batch_idx * params->K_strides[0] + kv_head_idx * params->K_strides[1];','K += (batch_idx * params->H + kv_head_idx) * NK * BK * BD;')
  body=body.replace('const int k_seq_stride = int(params->K_strides[2]);','const int k_seq_stride = 16;')
  body=body.replace('ik * kU * k_seq_stride + id * kU','id * BK * kU + ik * kU * k_seq_stride')
  body=body.replace('K += BK * k_seq_stride;','K += BK * BD;')
 if name in ('pack_v','pack_kv'):
  body=body.replace('V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1];','V += (batch_idx * params->H + kv_head_idx) * NK * BK * BD;')
  body=body.replace('const int v_seq_stride = int(params->V_strides[2]);','const int v_seq_stride = 32;')
  body=body.replace('ik * kU * v_seq_stride + id * kU','(id / 2) * BK * 32 + ik * kU * v_seq_stride')
  body=body.replace('V += BK * v_seq_stride;','V += BK * BD;')
 if name=='identity_rescale':
  body=body.replace('    Otile.template row_bin_op<MulOp>(factor);','    if (!simd_all(all(factor == metal::vec<AccumType,kRowsPT>(1.0f)))) {\n      Otile.template row_bin_op<MulOp>(factor);\n    }')
 return pre+body
@functools.lru_cache(None)
def library(name):return torch.mps.compile_shader(source(name))
@functools.lru_cache(None)
def packlib():return torch.mps.compile_shader(PACK)
def pack(x,width):
 h,n,d=x.shape[1:];out=torch.empty((h,(n+31)//32,32,128),dtype=x.dtype,device=x.device)
 packlib().pack_kv(x,out,n,width,threads=out.numel()//4,group_size=256)
 return out

def run(q,k,v,name,packed=None):
 if packed is None:
  pk=pack(k,16) if name in ('pack_k','pack_kv') else k
  pv=pack(v,32) if name in ('pack_v','pack_kv') else v
 else:pk,pv=packed
 p=qc._parameters(tuple(q.shape),tuple(k.shape),q.stride(),k.stride(),v.stride(),128**-.5)
 out=torch.empty_like(q)
 library(name).custom_attn(q,pk,pv,out,p,p,p,threads=(q.shape[1]*128,(q.shape[2]+63)//64,q.shape[0]),group_size=(128,1,1))
 return out
