"""Experimental exact matrix-operand paths. Not imported by production."""
from pathlib import Path
import sys,functools,torch
R=next(p for p in Path(__file__).resolve().parents if (p/'_patches/torch_mpp_qcache.py').is_file())
sys.path.insert(0,str(R))
from _patches import torch_mpp_qcache as qc
import importlib.util
_spec=importlib.util.spec_from_file_location("h3_pack_candidates",Path(__file__).resolve().parent.parent/"h3-attention-profile/candidates.py")
_packing=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_packing)

def source(name):
 if name.startswith('pv_transpack'):
  n=64 if '64' in name else 32
  s=source(f'mixedpv_n{n}_k32_trans')
  a=s.index('    if(align_K || !is_last_k)',s.index('    // Online softmax update.'))
  head,body=s[:a],s[a:]
  split=head.index('prefill_attention_mpp(')
  head=head[:split]+head[split:].replace('V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1];','V += (batch_idx * params->H + kv_head_idx) * NK * BK * BD;')
  # The packed final tile is zero padded, so native tensor loads are safe there too.
  body=body.replace('if(align_K || !is_last_k)','if(true)',1)
  body=body.replace('V+ik*kU*v_seq_stride+id*kU','V+id*kU*BK+ik*kU')
  body=body.replace('metal::array<int,2>{v_seq_stride,1}','metal::array<int,2>{1,BK}')
  body=body.replace('V += BK * v_seq_stride;','V += BK * BD;')
  return head+body

 if name in ('qk32_pv16','qk32_pv32'):
  s=source('regpv_n16_k32' if name.endswith('pv16') else 'baseline');q=source('regqk_n32_k32')
  a=s.index('    // S = Q @ K^T.');b=s.index('    // Scale (in log2 space).',a)
  qa=q.index('    using NEW_QK_MARKER');qb=q.index('    // Scale (in log2 space).',qa)
  return s[:a]+q[qa:qb]+s[b:]
 if name.startswith('packed_'):
  plain=name[7:];s=source(plain)
  width=64 if 'n64' in plain else 16 if 'n16_' in plain or plain.endswith('pv16') else 32
  start=s.index('prefill_attention_mpp(');head,body=s[:start],s[start:]
  body=body.replace('K += batch_idx * params->K_strides[0] + kv_head_idx * params->K_strides[1];','K += (batch_idx * params->H + kv_head_idx) * NK * BK * BD;')
  body=body.replace('const int k_seq_stride = int(params->K_strides[2]);','const int k_seq_stride = 16;')
  body=body.replace('ik * kU * k_seq_stride + id * kU','id * BK * kU + ik * kU * k_seq_stride')
  body=body.replace('ik*kU*k_seq_stride+(d+j)*kU','(d+j)*BK*kU+ik*kU*k_seq_stride')
  body=body.replace('K += BK * k_seq_stride;','K += BK * BD;')
  body=body.replace('V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1];','V += (batch_idx * params->H + kv_head_idx) * NK * BK * BD;')
  body=body.replace('const int v_seq_stride = int(params->V_strides[2]);',f'const int v_seq_stride = {width};')
  off=f'(id / {width//16}) * BK * {width} + (id % {width//16}) * kU'
  body=body.replace('ik * kU * v_seq_stride + id * kU',f'ik * kU * v_seq_stride + {off}')
  body=body.replace('ik*kU*v_seq_stride+id*kU',f'ik*kU*v_seq_stride+{off}')
  body=body.replace('(ik+j)*kU*v_seq_stride+id*kU',f'(ik+j)*kU*v_seq_stride+{off}')
  body=body.replace('V += BK * v_seq_stride;','V += BK * BD;')
  return head+body

 if name.endswith('_trans'):
  base=source(name[:-6]);a=base.index('    if(align_K || !is_last_k)',base.index('    // Online softmax update.'))
  # Only the substituted PV branch is changed; fallback keeps its original arithmetic.
  head,body=base[:a],base[a:];width=64 if 'n64' in name else 32;depth=32
  body=body.replace(f'(16,{width},{depth},false,false,true,',f'(16,{width},{depth},false,true,true,')
  body=body.replace(f'metal::extents<int,{width},{depth}>',f'metal::extents<int,{depth},{width}>').replace('metal::array<int,2>{1,v_seq_stride}','metal::array<int,2>{v_seq_stride,1}')
  return head+body
 s=qc._source(h3_long=True)
 if name=='baseline':return s
 if name.startswith('mixedqk') or name=='mixedboth':
  width=32 if name.endswith('32') else 16
  a=s.index('    // S = Q @ K^T.');b=s.index('    // Scale (in log2 space).',a);original=s[a:b]
  body='''
    if(align_K || !is_last_k) {
      constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,WIDTH,false,true,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
      auto left=op.template get_left_input_cooperative_tensor<T,T,AccumType>();
      using RT=metal::tensor<device T,metal::extents<int,WIDTH,32>,metal::tensor_inline>;
      auto acc=op.template get_destination_cooperative_tensor<decltype(left),RT,AccumType>();
      MPP_PRAGMA_UNROLL
      for(short i=0;i<16;i++) acc[i]=0;
      MPP_PRAGMA_UNROLL
      for(short d=0;d<TD;d+=CHUNKS) {
        MPP_PRAGMA_UNROLL
        for(short j=0;j<CHUNKS;j++) {
          MPP_PRAGMA_UNROLL
          for(short i=0;i<8;i++) left[j*8+i]=Qcache[d+j].frag_at(0,0)[i];
        }
        RT right(const_cast<device T*>(K+d*kU),metal::extents<int,WIDTH,32>{},metal::array<int,2>{1,k_seq_stride});
        op.run(left,right,acc);
      }
      MPP_PRAGMA_UNROLL
      for(short i=0;i<8;i++) {Stile.frag_at(0,0)[i]=acc[i];Stile.frag_at(0,1)[i]=acc[8+i];}
    } else {
'''.replace('WIDTH',str(width)).replace('CHUNKS',str(width//16))+original+'\n    }\n'
  s=s[:a]+body+s[b:]
 if name.startswith('mixedpv') or name=='mixedboth':
  n=64 if 'n64' in name else (128 if 'n128' in name else 32)
  k=32 if 'k32' in name else 16
  a=s.index('    // O += P @ V.');b=s.index('\n    K += BK * k_seq_stride;',a);original=s[a:b]
  body='''
    if(align_K || !is_last_k) {
      MPP_PRAGMA_UNROLL
      for(short id=0;id<TD;id+=FRAGS) {
        if(id==BARRIER) threadgroup_barrier(mem_flags::mem_none);
        constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,WIDTH,DEPTH,false,false,true,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
        auto left=op.template get_left_input_cooperative_tensor<AccumType,T,AccumType>();
        using RT=metal::tensor<device T,metal::extents<int,WIDTH,DEPTH>,metal::tensor_inline>;
        auto acc=op.template get_destination_cooperative_tensor<decltype(left),RT,AccumType>();
        MPP_PRAGMA_UNROLL
        for(short c=0;c<FRAGS;c++) {
          MPP_PRAGMA_UNROLL
          for(short i=0;i<8;i++) acc[c*8+i]=Otile.frag_at(0,id+c)[i];
        }
        MPP_PRAGMA_UNROLL
        for(short ik=0;ik<TK;ik+=CHUNKS) {
          MPP_PRAGMA_UNROLL
          for(short j=0;j<CHUNKS;j++) {
            MPP_PRAGMA_UNROLL
            for(short i=0;i<8;i++) left[j*8+i]=Stile.frag_at(0,ik+j)[i];
          }
          RT right(const_cast<device T*>(V+ik*kU*v_seq_stride+id*kU),metal::extents<int,WIDTH,DEPTH>{},metal::array<int,2>{1,v_seq_stride});
          op.run(left,right,acc);
        }
        MPP_PRAGMA_UNROLL
        for(short c=0;c<FRAGS;c++) {
          MPP_PRAGMA_UNROLL
          for(short i=0;i<8;i++) Otile.frag_at(0,id+c)[i]=acc[c*8+i];
        }
      }
    } else {
'''
  for key,value in [('WIDTH',n),('DEPTH',k),('FRAGS',n//16),('CHUNKS',k//16),('BARRIER',6 if n<=32 else 4 if n==64 else 0)]:body=body.replace(key,str(value))
  s=s[:a]+body+original+'\n    }\n'+s[b:]
 if name.startswith('regpv'):
  n=64 if 'n64' in name else 128 if 'n128' in name else 16 if 'n16_' in name else 32
  k=32 if 'k32' in name else 16
  a=s.index('    // O += P @ V.');b=s.index('\n    K += BK * k_seq_stride;',a)
  body="""
    MPP_PRAGMA_UNROLL
    for(short id=0;id<TD;id+=FRAGS) {
      if(id==BARRIER) threadgroup_barrier(mem_flags::mem_none);
      constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,WIDTH,DEPTH,false,false,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
      auto left=op.template get_left_input_cooperative_tensor<AccumType,T,AccumType>();
      auto right=op.template get_right_input_cooperative_tensor<AccumType,T,AccumType>();
      auto acc=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),AccumType>();
      MPP_PRAGMA_UNROLL
      for(short c=0;c<FRAGS;c++) {
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) acc[c*8+i]=Otile.frag_at(0,id+c)[i];
      }
      MPP_PRAGMA_UNROLL
      for(short ik=0;ik<TK;ik+=CHUNKS) {
        MPP_PRAGMA_UNROLL
        for(short j=0;j<CHUNKS;j++) {
          MPPTile<T,1,FRAGS,QFrag_t> vt;
          if(!align_K && is_last_k) vt.load_rows(V+(ik+j)*kU*v_seq_stride+id*kU,v_seq_stride,lim_rows_k-(ik+j)*kU);
          else vt.load(V+(ik+j)*kU*v_seq_stride+id*kU,v_seq_stride);
          MPP_PRAGMA_UNROLL
          for(short i=0;i<8;i++) left[j*8+i]=Stile.frag_at(0,ik+j)[i];
          MPP_PRAGMA_UNROLL
          for(short c=0;c<FRAGS;c++) {
            MPP_PRAGMA_UNROLL
            for(short i=0;i<8;i++) right[j*FRAGS*8+c*8+i]=vt.frag_at(0,c)[i];
          }
        }
        op.run(left,right,acc);
      }
      MPP_PRAGMA_UNROLL
      for(short c=0;c<FRAGS;c++) {
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) Otile.frag_at(0,id+c)[i]=acc[c*8+i];
      }
    }
"""
  for key,value in [('WIDTH',n),('DEPTH',k),('FRAGS',n//16),('CHUNKS',k//16),('BARRIER',6 if n<=32 else 4 if n==64 else 0)]:body=body.replace(key,str(value))
  s=s[:a]+body+s[b:]
 if name.startswith('pvorder'):
  a=s.index('    // O += P @ V.');b=s.index('\n    K += BK * k_seq_stride;',a)
  old=s[a:b];load=old[old.index('          MPPTile<T, 1, 2, QFrag_t> Vtile;'):old.index('\n        }',old.index('          QFrag_t::mma('))]
  barrier='if (ik==1 && id==6) threadgroup_barrier(mem_flags::mem_none);'
  if name=='pvorder_sync':barrier='if (id==6) threadgroup_barrier(mem_flags::mem_none);'
  s=s[:a]+"""
    MPP_PRAGMA_UNROLL
    for(short iq=0;iq<TQ;iq++) {
      MPP_PRAGMA_UNROLL
      for(short ik=0;ik<TK;ik++) {
        MPP_PRAGMA_UNROLL
        for(short id=0;id<TD;id+=2) {
"""+barrier+'\n'+load+'\n        }\n      }\n    }\n'+s[b:]
 if name in ('regqk_n16_k32','regqk_n32_k32'):
  n=32 if 'n32_' in name else 16
  a=s.index('    // S = Q @ K^T.');b=s.index('    // Scale (in log2 space).',a)
  body="""
    using NEW_QK_MARKER = int;
    MPP_PRAGMA_UNROLL
    for(short ik=0;ik<TK;ik+=FRAGS) {
      constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,WIDTH,32,false,true,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
      auto left=op.template get_left_input_cooperative_tensor<T,T,AccumType>();
      auto right=op.template get_right_input_cooperative_tensor<T,T,AccumType>();
      auto acc=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),AccumType>();
      MPP_PRAGMA_UNROLL
      for(short i=0;i<FRAGS*8;i++) acc[i]=0;
      MPP_PRAGMA_UNROLL
      for(short d=0;d<TD;d+=2) {
        MPP_PRAGMA_UNROLL
        for(short j=0;j<2;j++) {
          MPPTile<T,FRAGS,1,KFrag_t> kt;
          if(!align_K && is_last_k) kt.load_rows(K+ik*kU*k_seq_stride+(d+j)*kU,k_seq_stride,lim_rows_k-ik*kU);
          else kt.load(K+ik*kU*k_seq_stride+(d+j)*kU,k_seq_stride);
          MPP_PRAGMA_UNROLL
          for(short i=0;i<8;i++) left[j*8+i]=Qcache[d+j].frag_at(0,0)[i];
          MPP_PRAGMA_UNROLL
          for(short c=0;c<FRAGS;c++) {
            MPP_PRAGMA_UNROLL
            for(short i=0;i<8;i++) right[j*FRAGS*8+c*8+i]=kt.frag_at(c,0)[i];
          }
        }
        op.run(left,right,acc);
      }
      MPP_PRAGMA_UNROLL
      for(short c=0;c<FRAGS;c++) {
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) Stile.frag_at(0,ik+c)[i]=acc[c*8+i];
      }
    }
""".replace("WIDTH",str(n)).replace("FRAGS",str(n//16))
  s=s[:a]+body+s[b:]
 if name.startswith('qk_widen'):
  if name in ('qk_widen_q','qk_widen_both'):
   s=s.replace('MPPTile<T, 1, 1, QFrag_t> Qcache[TD];','MPPTile<AccumType, 1, 1, QFrag_t> Qcache[TD];')
  if name in ('qk_widen_k','qk_widen_both'):
   s=s.replace('MPPTile<T, 2, 1, KFrag_t> Ktile;','MPPTile<AccumType, 2, 1, KFrag_t> Ktile;')
 return s
@functools.lru_cache(None)
def library(name):return torch.mps.compile_shader(source(name))
def run(q,k,v,name):
 p=qc._parameters(tuple(q.shape),tuple(k.shape),q.stride(),k.stride(),v.stride(),128**-.5)
 out=torch.empty_like(q)
 if name.startswith("pv_transpack"):
  v=pack_transpose(v)
 if name.startswith("packed_"):
  k=_packing.pack(k,16);v=_packing.pack(v,64 if "n64" in name else 16 if "n16_" in name or name.endswith("pv16") else 32)
 library(name).custom_attn(q,k,v,out,p,p,p,threads=(q.shape[1]*128,(q.shape[2]+63)//64,q.shape[0]),group_size=(128,1,1))
 return out

TRANSPOSE_PACK = r"""
#include <metal_stdlib>
using namespace metal;
kernel void pack(const device ushort* src [[buffer(0)]],device ushort* dst [[buffer(1)]],constant uint& n [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 uint blocks=(n+31)/32,key=i%32,col=(i/32)%128,tile=(i/4096)%blocks,head=i/(4096*blocks),token=tile*32+key;
 dst[i]=token<n?src[(head*n+token)*128+col]:ushort(0);
}
"""
@functools.lru_cache(None)
def transpacklib():return torch.mps.compile_shader(TRANSPOSE_PACK)
def pack_transpose(v):
 h,n,d=v.shape[1:];out=torch.empty((h,(n+31)//32,128,32),device=v.device,dtype=v.dtype)
 transpacklib().pack(v,out,n,threads=out.numel(),group_size=256)
 return out
