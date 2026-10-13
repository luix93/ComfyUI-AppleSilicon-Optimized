"""Experimental exact-attention candidates; never imported by production."""
import functools,sys,torch
from pathlib import Path
D=Path(__file__).resolve().parent
R=next(p for p in D.parents if (p/'_patches/torch_mpp_qcache.py').is_file())
sys.path.insert(0,str(R))
from _patches import torch_mpp_qcache as qc

def source(name):
 s=qc._source(h3_long=True,cache_query=("noq" not in name or name.startswith("m32")))
 if name=='vectorload':
  a=s.index('struct BaseMPPFrag {');b=s.index('// Second operand of Q@K',a)
  pre,frag,post=s[:a],s[a:b],s[b:]
  old='        MPP_PRAGMA_UNROLL\n        for (short j = 0; j < kElemCols; j++) {\n          dst[i * kElemCols + j] =\n              static_cast<T>(src[r * str_x + c + col_offset(j)]);\n        }'
  new='        if constexpr(IS_APPLE10) {\n          using U = metal::remove_cv_t<pointer_element_t<SrcPtrType>>;\n          auto packed=*reinterpret_cast<const device metal::vec<U,4>*>(src+r*str_x+c);\n          MPP_PRAGMA_UNROLL\n          for(short j=0;j<4;j++) dst[i*4+j]=static_cast<T>(packed[j]);\n        } else {\n'+old+'\n        }'
  assert old in frag;frag=frag.replace(old,new);s=pre+frag+post
 if name.startswith('qk'):
  width=int(name.split('_')[0][2:]); persistent=name.endswith('persistent')
  if persistent:width=16
  body='''
    constexpr auto descq = mpp::tensor_ops::matmul2d_descriptor(16,32,WIDTH,false,true,true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descq,metal::execution_simdgroup> opq;
    auto aq=opq.template get_left_input_cooperative_tensor<T,T,AccumType>();
    auto bq=opq.template get_right_input_cooperative_tensor<T,T,AccumType>();
    auto cq=opq.template get_destination_cooperative_tensor<decltype(aq),decltype(bq),AccumType>();
    MPP_PRAGMA_UNROLL
    for(short i=0;i<16;i++) cq[i]=0;
    MPP_PRAGMA_UNROLL
    for(short d=0;d<TD;d+=CHUNKS) {
      MPP_PRAGMA_UNROLL
      for(short j=0;j<CHUNKS;j++) {
        MPPTile<T,2,1,KFrag_t> kt;
        if (!align_K && is_last_k) kt.load_rows(K+(d+j)*kU,k_seq_stride,lim_rows_k);
        else kt.load(K+(d+j)*kU,k_seq_stride);
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) {
          aq[j*8+i]=Qcache[d+j].frag_at(0,0)[i];
          bq[j*16+i]=kt.frag_at(0,0)[i];
          bq[j*16+8+i]=kt.frag_at(1,0)[i];
        }
      }
      opq.run(aq,bq,cq);
    }
    MPP_PRAGMA_UNROLL
    for(short i=0;i<8;i++) {
      Stile.frag_at(0,0)[i]=cq[i];Stile.frag_at(0,1)[i]=cq[8+i];
    }
'''.replace('WIDTH',str(width)).replace('CHUNKS',str(width//16))
  a=s.index('    // S = Q @ K^T.');b=s.index('    // Scale (in log2 space).',a)
  s=s[:a]+body+s[b:]
 if name.startswith('strides'):
  for cap in 'QKVO':s=s.replace(f'int(params->{cap}_strides[2])','128')
 if name=='tail_split':
  # Main and last key tile have distinct compiled control-flow paths, recurrence unchanged.
  a=s.index('  for (int kb = 0; kb < kb_lim; kb++) {',s.index('prefill_attention_mpp('))
  b=s.index('\n  // Drains the cooperative-tensor pipeline',a)
  loop=s[a:b];body=loop[loop.index('\n')+1:loop.rfind('  }')]
  main=body.replace('const bool is_last_k = (kb == NK_aligned);','const bool is_last_k = false;')
  tail=body.replace('const bool is_last_k = (kb == NK_aligned);','const bool is_last_k = true;')
  s=s[:a]+'  for (int kb=0;kb<NK_aligned;kb++) {\n'+main+'  }\n  if (!align_K) { const int kb=NK_aligned;\n'+tail+'  }\n'+s[b:]
 if name.startswith('unroll'):
  n=int(name[6:]);s=s.replace('  for (int kb = 0; kb < kb_lim; kb++) {',f'  _Pragma("clang loop unroll_count({n})")\n  for (int kb = 0; kb < kb_lim; kb++) {{')
 if name=='pv_persistent':
  a=s.index('    // O += P @ V.');b=s.index('\n    K += BK * k_seq_stride;',a)
  s=s[:a]+'''
    MPP_PRAGMA_UNROLL
    for(short id=0;id<TD;id+=2) {
      if(id==6) threadgroup_barrier(mem_flags::mem_none);
      constexpr auto descp=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,false,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<descp,metal::execution_simdgroup> opp;
      auto ap=opp.template get_left_input_cooperative_tensor<AccumType,T,AccumType>();
      auto bp=opp.template get_right_input_cooperative_tensor<AccumType,T,AccumType>();
      auto cp=opp.template get_destination_cooperative_tensor<decltype(ap),decltype(bp),AccumType>();
      MPP_PRAGMA_UNROLL
      for(short i=0;i<8;i++) { cp[i]=Otile.frag_at(0,id)[i];cp[8+i]=Otile.frag_at(0,id+1)[i]; }
      MPP_PRAGMA_UNROLL
      for(short ik=0;ik<TK;ik++) {
        MPPTile<T,1,2,QFrag_t> vt;
        if(!align_K && is_last_k) vt.load_rows(V+ik*kU*v_seq_stride+id*kU,v_seq_stride,lim_rows_k-ik*kU);
        else vt.load(V+ik*kU*v_seq_stride+id*kU,v_seq_stride);
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) {
          ap[i]=Stile.frag_at(0,ik)[i];bp[i]=vt.frag_at(0,0)[i];bp[8+i]=vt.frag_at(0,1)[i];
        }
        opp.run(ap,bp,cp);
      }
      MPP_PRAGMA_UNROLL
      for(short i=0;i<8;i++) {Otile.frag_at(0,id)[i]=cp[i];Otile.frag_at(0,id+1)[i]=cp[8+i];}
    }
''' + s[b:]
 from more import transform
 s=transform(s,name)
 from lookahead import transform as look
 s=look(s,name)
 from rows32 import transform as rows
 s=rows(s,name)
 from persistent_output import transform as output
 return output(s,name)

@functools.lru_cache(None)
def library(name):return torch.mps.compile_shader(source(name))

def run(q,k,v,name,scale=None):
 w=int(name[5:]) if name.startswith("warps") else 4
 if name.startswith("m32"):w=int(name[-1])
 bq=w*32 if name.startswith("m32") else w*16
 mult=2 if "split" in name and name.startswith("m32") else 1
 p=qc._parameters(tuple(q.shape),tuple(k.shape),q.stride(),k.stride(),v.stride(),128**-.5 if scale is None else scale)
 out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
 library(name).custom_attn(q,k,v,out,p,p,p,threads=(q.shape[1]*w*32*mult,(q.shape[2]+bq-1)//bq,q.shape[0]),group_size=(w*32,1,1))
 return out
