"""32-query-row cooperative operations with optional output-column partition."""
def transform(s,name):
 if not name.startswith('m32'):return s
 split='split' in name
 w=int(name[-1]);outcols=4 if split else 8
 s=s.replace('bfloat, 64, 32, 128, 4, 1,',f'bfloat, {w*32}, 32, 128, {w}, 1,')
 s=s.replace('static_assert(TQ == 1, "Check TQ");','')
 s=s.replace('Qcache[TD]','Qcache[TQ*TD]')
 old='''  for (short id = 0; id < TD; id++) {
    if (!align_Q && is_last_q) Qcache[id].load_rows(Q + id*kU,q_seq_stride,lim_rows_q);
    else Qcache[id].load(Q + id*kU,q_seq_stride);
  }'''
 new='''  for(short id=0;id<TQ*TD;id++) {
    const int off=(id/TD)*kU*q_seq_stride+(id%TD)*kU;
    if(!align_Q && is_last_q) Qcache[id].load_rows(Q+off,q_seq_stride,lim_rows_q-(id/TD)*kU);
    else Qcache[id].load(Q+off,q_seq_stride);
  }'''
 assert old in s;s=s.replace(old,new)
 if split:
  s=s.replace('const int head_idx = tid.x;', 'const int head_idx = tid.x / 2;\n  const int ooffset=(tid.x % 2)*64;')
  s=s.replace('using otile_t = MPPTile<AccumType, TQ, TD, QFrag_t>;','using otile_t = MPPTile<AccumType, TQ, 4, QFrag_t>;')
  s=s.replace('O += tm * o_seq_stride;','O += tm * o_seq_stride + ooffset;')
 else:s=s.replace('const int head_idx = tid.x;','const int head_idx = tid.x;\n const int ooffset=0;')
 a=s.index('    // S = Q @ K^T.');b=s.index('    // Scale (in log2 space).',a)
 body='''
    constexpr auto dq=mpp::tensor_ops::matmul2d_descriptor(32,32,16,false,true,true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<dq,metal::execution_simdgroup> oq;
    auto aq=oq.template get_left_input_cooperative_tensor<T,T,AccumType>();
    auto bq=oq.template get_right_input_cooperative_tensor<T,T,AccumType>();
    auto cq=oq.template get_destination_cooperative_tensor<decltype(aq),decltype(bq),AccumType>();
    MPP_PRAGMA_UNROLL
    for(short i=0;i<32;i++) cq[i]=0;
    MPP_PRAGMA_UNROLL
    for(short d=0;d<TD;d++) {
      MPPTile<T,2,1,KFrag_t> kt;
      if(!align_K && is_last_k) kt.load_rows(K+d*kU,k_seq_stride,lim_rows_k);
      else kt.load(K+d*kU,k_seq_stride);
      MPP_PRAGMA_UNROLL
      for(short i=0;i<8;i++) {
        aq[i]=Qcache[d].frag_at(0,0)[i];aq[8+i]=Qcache[TD+d].frag_at(0,0)[i];
        bq[i]=kt.frag_at(0,0)[i];bq[8+i]=kt.frag_at(1,0)[i];
      }
      oq.run(aq,bq,cq);
    }
    MPP_PRAGMA_UNROLL
    for(short r=0;r<2;r++) {
      MPP_PRAGMA_UNROLL
      for(short c=0;c<2;c++) {
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) Stile.frag_at(r,c)[i]=cq[(r*2+c)*8+i];
      }
    }
'''
 if 'col' in name:body=body.replace('(r*2+c)*8','(c*2+r)*8')
 s=s[:a]+body+s[b:]
 a=s.index('    // O += P @ V.');b=s.index('\n    K += BK * k_seq_stride;',a)
 body='''
    MPP_PRAGMA_UNROLL
    for(short id=0;id<OCOLS;id+=2) {
      if(id==BARRIER) threadgroup_barrier(mem_flags::mem_none);
      constexpr auto dp=mpp::tensor_ops::matmul2d_descriptor(32,32,16,false,false,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<dp,metal::execution_simdgroup> op;
      auto ap=op.template get_left_input_cooperative_tensor<AccumType,T,AccumType>();
      auto bp=op.template get_right_input_cooperative_tensor<AccumType,T,AccumType>();
      auto cp=op.template get_destination_cooperative_tensor<decltype(ap),decltype(bp),AccumType>();
      MPP_PRAGMA_UNROLL
      for(short r=0;r<2;r++) {
        MPP_PRAGMA_UNROLL
        for(short c=0;c<2;c++) {
          MPP_PRAGMA_UNROLL
          for(short i=0;i<8;i++) cp[(r*2+c)*8+i]=Otile.frag_at(r,id+c)[i];
        }
      }
      MPP_PRAGMA_UNROLL
      for(short ik=0;ik<TK;ik++) {
        MPPTile<T,1,2,QFrag_t> vt;
        const int off=ik*kU*v_seq_stride+id*kU+ooffset;
        if(!align_K && is_last_k) vt.load_rows(V+off,v_seq_stride,lim_rows_k-ik*kU);
        else vt.load(V+off,v_seq_stride);
        MPP_PRAGMA_UNROLL
        for(short i=0;i<8;i++) {
          ap[i]=Stile.frag_at(0,ik)[i];ap[8+i]=Stile.frag_at(1,ik)[i];
          bp[i]=vt.frag_at(0,0)[i];bp[8+i]=vt.frag_at(0,1)[i];
        }
        op.run(ap,bp,cp);
      }
      MPP_PRAGMA_UNROLL
      for(short r=0;r<2;r++) {
        MPP_PRAGMA_UNROLL
        for(short c=0;c<2;c++) {
          MPP_PRAGMA_UNROLL
          for(short i=0;i<8;i++) Otile.frag_at(r,id+c)[i]=cp[(r*2+c)*8+i];
        }
      }
    }
'''.replace('OCOLS',str(outcols)).replace('BARRIER',str(outcols-2))
 if 'col' in name:body=body.replace('(r*2+c)*8','(c*2+r)*8')
 s=s[:a]+body+s[b:]
 if 'noq' in name:
  old='        aq[i]=Qcache[d].frag_at(0,0)[i];aq[8+i]=Qcache[TD+d].frag_at(0,0)[i];'
  new='        aq[i]=qt0.frag_at(0,0)[i];aq[8+i]=qt1.frag_at(0,0)[i];'
  s=s.replace(old,new)
  anchor='      MPPTile<T,2,1,KFrag_t> kt;'
  load='      MPPTile<T,1,1,QFrag_t> qt0,qt1;\n      if(!align_Q && is_last_q) {\n        qt0.load_rows(Q+d*kU,q_seq_stride,lim_rows_q);\n        qt1.load_rows(Q+16*q_seq_stride+d*kU,q_seq_stride,lim_rows_q-16);\n      } else {\n        qt0.load(Q+d*kU,q_seq_stride);qt1.load(Q+16*q_seq_stride+d*kU,q_seq_stride);\n      }\n'
  assert anchor in s;s=s.replace(anchor,load+anchor)
 return s
