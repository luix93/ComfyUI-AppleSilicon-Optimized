"""Keep all four output accumulators as cooperative tensors across the key loop."""
def transform(s,name):
 if not name.startswith('output_ct'):return s
 start=s.index('prefill_attention_mpp(');prefix,s=s[:start],s[start:]
 init='''
  constexpr auto pd=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,false,true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pd,metal::execution_simdgroup> po;
  using PA=decltype(po.template get_left_input_cooperative_tensor<AccumType,T,AccumType>());
  using PB=decltype(po.template get_right_input_cooperative_tensor<AccumType,T,AccumType>());
  using PC=decltype(po.template get_destination_cooperative_tensor<PA,PB,AccumType>());
  PC OC[TD/2];
  MPP_PRAGMA_UNROLL
  for(short id=0;id<TD/2;id++) {
    MPP_PRAGMA_UNROLL
    for(short i=0;i<16;i++) OC[id][i]=0;
  }
'''
 s=s.replace('  otile_t Otile;\n  Otile.clear();',init)
 rescale='''
    MPP_PRAGMA_UNROLL
    for(short id=0;id<TD/2;id++) {
      MPP_PRAGMA_UNROLL
      for(short i=0;i<16;i++) OC[id][i]=OC[id][i]*factor[(i%8)/4];
    }
'''
 s=s.replace('    Otile.template row_bin_op<MulOp>(factor);',rescale)
 a=s.index('    // O += P @ V.');b=s.index('\n    K += BK * k_seq_stride;',a)
 body='''
    MPP_PRAGMA_UNROLL
    for(short id=0;id<TD;id+=2) {
      if(id==6) threadgroup_barrier(mem_flags::mem_none);
      MPP_PRAGMA_UNROLL
      for(short ik=0;ik<TK;ik++) {
        MPPTile<T,1,2,QFrag_t> vt;
        const int off=ik*kU*v_seq_stride+id*kU;
        if(!align_K && is_last_k) vt.load_rows(V+off,v_seq_stride,lim_rows_k-ik*kU);
        else vt.load(V+off,v_seq_stride);
        PA ap;PB bp;
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) {
          ap[i]=Stile.frag_at(0,ik)[i];bp[i]=vt.frag_at(0,0)[i];bp[8+i]=vt.frag_at(0,1)[i];
        }
        po.run(ap,bp,OC[id/2]);
      }
    }
'''
 s=s[:a]+body+s[b:]
 drain='''
  otile_t Otile;
  MPP_PRAGMA_UNROLL
  for(short id=0;id<TD;id+=2) {
    MPP_PRAGMA_UNROLL
    for(short i=0;i<8;i++) {Otile.frag_at(0,id)[i]=OC[id/2][i];Otile.frag_at(0,id+1)[i]=OC[id/2][8+i];}
  }
'''
 s=s.replace('  // Treat fully-masked rows as zero output.',drain+'\n  // Treat fully-masked rows as zero output.')
 # Cooperative tensor types cannot form arrays: emit four named values.
 s=s.replace('PC OC[TD/2];','PC OC0,OC1,OC2,OC3;')
 import re
 for pattern,indices in [('for(short id=0;id<TD/2;id++)',range(4)),('for(short id=0;id<TD;id+=2)',range(0,8,2))]:
  while pattern in s:
   a=s.index(pattern);b=s.index('{',a);depth=1;e=b+1
   while depth:
    depth+=(s[e]=='{')-(s[e]=='}');e+=1
   body=s[b+1:e-1];copies=[]
   for idx in indices:
    copy=re.sub(r'\bid\b',str(idx),body)
    copy=re.sub(r'OC\[([0-9/]+)\]',lambda m:'OC'+str(int(m[1].split('/')[0]) // (int(m[1].split('/')[1]) if '/' in m[1] else 1)),copy)
    copies.append('{'+copy+'}')
   # Remove immediately preceding unroll pragma (the loop no longer exists).
   head=s[:a];head=re.sub(r'MPP_PRAGMA_UNROLL\s*$', '', head)
   s=head+'\n'.join(copies)+s[e:]
 return prefix+s
