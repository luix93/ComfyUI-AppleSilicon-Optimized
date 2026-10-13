from pathlib import Path
import sys,struct,json,torch
import argparse
parser=argparse.ArgumentParser(description="Prepare diagnostic shaders and native inputs from an authored public QKV capture.")
parser.add_argument('--capture',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();D=args.output.resolve();D.mkdir(parents=True,exist_ok=True)
R=next(p for p in Path(__file__).resolve().parents if (p/'_patches/torch_mpp_qcache.py').is_file())
sys.path.insert(0,str(R))
from _patches import torch_mpp_qcache as qc
assert torch.__version__.split("+")[0]=="2.14.1" and qc._torch_build_supported() and qc._hardware(), "Validated M5 Ultra / PyTorch 2.14.1 required"
s=qc._source(h3_long=True)
(D/'baseline.metal').write_text(s)
start=s.index('prefill_attention_mpp(');pre,body=s[:start],s[start:]
body=body.replace('    uint simd_group_id [[simdgroup_index_in_threadgroup]],','    constant uint& phase_mask [[buffer(7)]],\n    uint simd_group_id [[simdgroup_index_in_threadgroup]],')
a=body.index('    // S = Q @ K^T.');b=body.index('    // Scale (in log2 space).',a)
body=body[:a]+'    if (phase_mask & 1u) {\n'+body[a:b]+'''    } else {
      // Diagnostic substitute only: dynamic, nonzero input to downstream arithmetic.
      MPP_PRAGMA_UNROLL
      for(short i=0;i<stile_t::kElemsPerTile;i++) Stile.elems()[i]=float(Qcache[0].frag_at(0,0)[i%8])+float(kb%8)*0.001f;
    }
'''+body[b:]
a=body.index('    // Scale (in log2 space).');b=body.index('    // O += P @ V.',a)
body=body[:a]+'    if (phase_mask & 2u) {\n'+body[a:b]+'''    } else {
      // Diagnostic only: keep bounded data-dependent inputs to PV and retain its work.
      MPP_PRAGMA_UNROLL
      for(short i=0;i<stile_t::kElemsPerTile;i++) Stile.elems()[i]=Stile.elems()[i]*0.0001f;
      sum_score=metal::vec<AccumType,kRowsPT>(1.0f);
      max_score=metal::vec<AccumType,kRowsPT>(0.0f);
      simdgroup_barrier(mem_flags::mem_none);
    }
'''+body[b:]
a=body.index('    // O += P @ V.');b=body.index('\n    K += BK * k_seq_stride;',a)
body=body[:a]+'    if (phase_mask & 4u) {\n'+body[a:b]+'    } else {\n      MPP_PRAGMA_UNROLL\n      for(short i=0;i<otile_t::kElemsPerTile;i++) Otile.elems()[i]+=Stile.elems()[i%stile_t::kElemsPerTile]*0.0001f;\n    }\n'+body[b:]
(D/'phases.metal').write_text(pre+body)
# Raw native runner inputs, from public authored fixture only.
cpu=torch.load(args.capture,map_location='cpu',weights_only=True)
assert len(cpu)==3 and all(t.dtype==torch.bfloat16 and t.ndim==4 and t.shape[0]==1 and t.shape[1]==56 and t.shape[3]==128 and 30000<t.shape[2]<=65536 and t.is_contiguous() for t in cpu)
assert cpu[0].shape==cpu[1].shape==cpu[2].shape
meta=[]
for name,t in zip('qkv',cpu):
 assert t.is_contiguous();p=D/(name+'.bin')
 t.view(torch.uint8).numpy().tofile(p)
 meta.append({'shape':list(t.shape),'stride':list(t.stride()),'bytes':t.numel()*t.element_size()})
q,k,v=cpu;b,h,n,d=q.shape
params=struct.pack('=6i2f12q',b,h,d,n,n,1,d**-.5,1.,*q.stride()[:3],*k.stride()[:3],*v.stride()[:3],*q.stride()[:3]);(D/'params.bin').write_bytes(params)
(D/'inputs.json').write_text(json.dumps(meta,indent=2))
