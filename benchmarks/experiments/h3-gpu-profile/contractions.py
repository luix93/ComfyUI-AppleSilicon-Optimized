"""Candidate contractions: exactness must be established, never production."""
import sys,functools,torch
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'h3-exact-attention'))
import kernels
qc=kernels.qc
@functools.lru_cache(None)
def source(name):
 s=kernels.source('qk32' if name in ('qk32','qk32_pv32') else 'baseline')
 if name in ('pv32','qk32_pv32'):
  a=s.index('    // O += P @ V.');b=s.index('\n    K += BK * k_seq_stride;',a)
  body='''
    MPP_PRAGMA_UNROLL
    for(short id=0;id<TD;id+=2) {
      if(id==6) threadgroup_barrier(mem_flags::mem_none);
      constexpr auto descp=mpp::tensor_ops::matmul2d_descriptor(16,32,32,false,false,true,
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
          ap[ik*8+i]=Stile.frag_at(0,ik)[i];bp[ik*16+i]=vt.frag_at(0,0)[i];bp[ik*16+8+i]=vt.frag_at(0,1)[i];
        }
      }
      opp.run(ap,bp,cp);
      MPP_PRAGMA_UNROLL
      for(short i=0;i<8;i++) {Otile.frag_at(0,id)[i]=cp[i];Otile.frag_at(0,id+1)[i]=cp[8+i];}
    }
'''
  s=s[:a]+body+s[b:]
 return s
@functools.lru_cache(None)
def library(name):return torch.mps.compile_shader(source(name))
def run(q,k,v,name):
 p=qc._parameters(tuple(q.shape),tuple(k.shape),q.stride(),k.stride(),v.stride(),128**-.5)
 out=torch.empty_like(q)
 library(name).custom_attn(q,k,v,out,p,p,p,threads=(q.shape[1]*128,(q.shape[2]+63)//64,q.shape[0]),group_size=(128,1,1))
 return out
