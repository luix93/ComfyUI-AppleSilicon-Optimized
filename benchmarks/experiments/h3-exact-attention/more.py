"""Direct tensor operands and scheduling transformations, experimental only."""
def transform(s,name):
 pos=s.index("prefill_attention_mpp(");prefix,s=s[:pos],s[pos:]
 if name.startswith('directqk'):
  a=s.index('    // S = Q @ K^T.');b=s.index('    // Scale (in log2 space).',a)
  original=s[a:b]
  body='''
    if (align_K || !is_last_k) {
      using dt = metal::tensor<device T,metal::dextents<int,2>,metal::tensor_inline>;
      auto qa=dt(const_cast<device T*>(Q),metal::dextents<int,2>(BD,min(16,int(lim_rows_q))));
      auto ka=dt(const_cast<device T*>(K),metal::dextents<int,2>(BD,32));
      constexpr auto dq=mpp::tensor_ops::matmul2d_descriptor(16,32,128,false,true,true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<dq,metal::execution_simdgroup> oq;
      auto cq=oq.template get_destination_cooperative_tensor<decltype(qa),decltype(ka),AccumType>();
      MPP_PRAGMA_UNROLL
      for(short i=0;i<16;i++) cq[i]=0;
      oq.run(qa,ka,cq);
      MPP_PRAGMA_UNROLL
      for(short i=0;i<8;i++) {Stile.frag_at(0,0)[i]=cq[i];Stile.frag_at(0,1)[i]=cq[8+i];}
    } else {
''' + original + '\n    }\n'
  body=body.replace('min(16,int(lim_rows_q))','((!align_Q && is_last_q) ? max(0,min(16,int(lim_rows_q))) : 16)')
  if name=='directqk_static':
   body=body.replace('if (align_K || !is_last_k)', 'if ((align_K || !is_last_k) && (align_Q || !is_last_q))')
   body=body.replace('auto qa=dt(const_cast<device T*>(Q),metal::dextents<int,2>(BD,((!align_Q && is_last_q) ? max(0,min(16,int(lim_rows_q))) : 16)));','auto qa=metal::tensor<device T,metal::extents<int,128,16>,metal::tensor_inline>(const_cast<device T*>(Q),metal::extents<int,128,16>{});')
   body=body.replace('auto ka=dt(const_cast<device T*>(K),metal::dextents<int,2>(BD,32));','auto ka=metal::tensor<device T,metal::extents<int,128,32>,metal::tensor_inline>(const_cast<device T*>(K),metal::extents<int,128,32>{});')
  s=s[:a]+body+s[b:]
 if name.startswith('prefetchv'):
  count=int(name[9:]);anchor='    // Online softmax update.'
  pre='''
    MPPTile<T,1,2,QFrag_t> Vearly[COUNT*TK];
    MPP_PRAGMA_UNROLL
    for(short id=0;id<COUNT;id++) {
      MPP_PRAGMA_UNROLL
      for(short ik=0;ik<TK;ik++) {
        if(!align_K && is_last_k) Vearly[id*TK+ik].load_rows(V+ik*kU*v_seq_stride+id*32,v_seq_stride,lim_rows_k-ik*kU);
        else Vearly[id*TK+ik].load(V+ik*kU*v_seq_stride+id*32,v_seq_stride);
      }
    }
'''.replace('COUNT',str(count))
  s=s.replace(anchor,pre+anchor)
  a=s.index('          MPPTile<T, 1, 2, QFrag_t> Vtile;',s.index('    // O += P @ V.'))
  b=s.index('          QFrag_t::mma(',a)
  old=s[a:b];new=old.replace('          const int V_load_off',f'          if (id < {count*2}) Vtile = Vearly[(id/2)*TK+ik];\n          else {{\n          const int V_load_off')+'          }\n'
  s=s[:a]+new+s[b:]
 if name.startswith('prefetchk'):
  # Stage all 128 K channels before QK: more ILP, at the cost of registers.
  a=s.index('    // S = Q @ K^T.');b=s.index('    // Scale (in log2 space).',a)
  body=s[a:b]
  pre='''
    MPPTile<T,2,1,KFrag_t> Kearly[TD];
    MPP_PRAGMA_UNROLL
    for(short id=0;id<TD;id++) {
      if(!align_K && is_last_k) Kearly[id].load_rows(K+id*kU,k_seq_stride,lim_rows_k);
      else Kearly[id].load(K+id*kU,k_seq_stride);
    }
'''
  body=body.replace('MPPTile<T, 2, 1, KFrag_t> Ktile;','thread auto& Ktile=Kearly[id];')
  c=body.index('          if (!align_K && is_last_k)');e=body.index('          QFrag_t::mma(',c)
  body=body[:c]+body[e:]
  s=s[:a]+pre+body+s[b:]
 if name.startswith('sync'):
  n=int(name[4:]);old='if (id == 6) {'
  s=s.replace(old,f'if (id == 6 && (kb % {max(1,n)}) == 0 && {"true" if n else "false"}) {{')
 if name.startswith('maxthreads'):
  prefix=prefix.replace('max_total_threads_per_threadgroup(WM* WN * 32)',f'max_total_threads_per_threadgroup({int(name[10:])})')
 if name == 'pv_f32':
  s=s.replace('MPPTile<T, 1, 2, QFrag_t> Vtile;','MPPTile<AccumType, 1, 2, QFrag_t> Vtile;')
 if name == 'rescale_late':
  s=s.replace('    Otile.template row_bin_op<MulOp>(factor);','')
  a=s.index('    // O += P @ V.')
  pre,post=s[:a],s[a:]
  marker='      for (short id = 0; id < TD; id += 2) {'
  extra='\n        MPP_PRAGMA_UNROLL\n        for(short c=0;c<2;c++) QFrag_t::template row_bin_op<MulOp>(Otile.frag_at(iq,id+c),(thread AccumType*)(&factor));\n'
  post=post.replace(marker,marker+extra);s=pre+post
 if name.startswith('inner'):
  n=int(name.split('_')[0][5:])
  marker='        MPP_PRAGMA_UNROLL\n        for (short id = 0; id < TD; id++) {'
  assert marker in s;s=s.replace(marker,f'        _Pragma("clang loop unroll_count({n})")\n        for (short id = 0; id < TD; id++) {{')
 if 'uniform' in name:
  old='  const AccumType scale2 =\n      static_cast<AccumType>(params->scale) * 1.44269504089f;'
  new='  const metal::uniform<float> scale2 = make_uniform(params->scale)*make_uniform(1.44269504089f);'
  assert old in s;s=s.replace(old,new).replace('Stile.elems()[ii] *= scale2;','Stile.elems()[ii] *= float(scale2);')
 if name.startswith('warps'):
  w=int(name[5:]);s=s.replace('bfloat, 64, 32, 128, 4, 1,',f'bfloat, {w*16}, 32, 128, {w}, 1,')
 return prefix+s
