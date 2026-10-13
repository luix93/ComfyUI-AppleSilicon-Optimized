"""Compute independent QK tiles early; consume each softmax tile in original order."""
def transform(s,name):
 if not name.startswith('look'):return s
 n=int(name[4:])
 a=s.index('  for (int kb = 0; kb < kb_lim; kb++) {',s.index('prefill_attention_mpp('))
 end=s.index('\n  // Drains the cooperative-tensor pipeline',a)
 whole=s[a:end];score=whole.index('    // S = Q @ K^T.');soft=whole.index('    // Scale (in log2 space).')
 # Build interleaved dot products: every accumulator retains its original id order.
 qk='''
    MPP_PRAGMA_UNROLL
    for(short id=0;id<TD;id++) {
      MPP_PRAGMA_UNROLL
      for(short t=0;t<COUNT;t++) {
        if(base+t<kb_lim) {
          MPPTile<T,2,1,KFrag_t> kt;
          const device T* kp=K+t*BK*k_seq_stride+id*kU;
          if(!align_K && base+t==NK_aligned) kt.load_rows(kp,k_seq_stride,lim_rows_k);
          else kt.load(kp,k_seq_stride);
          QFrag_t::mma(Ahead[t].frag_at(0,0),Ahead[t].frag_at(0,1),
            Qcache[id].frag_at(0,0),metal::false_type{},
            kt.frag_at(0,0),kt.frag_at(1,0),metal::true_type{});
        }
      }
    }
'''.replace('COUNT',str(n))
 tail=whole[soft:whole.rfind('  }')]
 loop=f'''  for(int base=0;base<kb_lim;base+={n}) {{
    using stile_t = MPPTile<AccumType,TQ,TK,QFrag_t>;
    stile_t Ahead[{n}];
    MPP_PRAGMA_UNROLL
    for(short t=0;t<{n};t++) Ahead[t].clear();
'''+qk+f'''
    MPP_PRAGMA_UNROLL
    for(short t=0;t<{n};t++) {{
      if(base+t<kb_lim) {{
        const int kb=base+t;
        const bool is_last_k=(kb==NK_aligned);
        stile_t Stile=Ahead[t];
'''+tail+'\n      }\n    }\n  }\n'
 return s[:a]+loop+s[end:]
