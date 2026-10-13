from pathlib import Path
import torch,time,json,random,statistics,hashlib
from contractions import run,source,qc
import argparse
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if not (qc._hardware() and qc._torch_build_supported() and torch.__version__.split('+')[0]=='2.14.1'):
 raise RuntimeError('This experiment requires the validated M5 Ultra / PyTorch 2.14.1 build')
cpu=torch.load(a.capture,map_location='cpu',weights_only=True)
assert len(cpu)==3 and all(t.dtype==torch.bfloat16 and t.is_contiguous() and t.ndim==4 and t.shape[0]==1 and t.shape[1]==56 and t.shape[3]==128 and 30000<t.shape[2]<=65536 for t in cpu)
assert cpu[0].shape==cpu[1].shape==cpu[2].shape
rows={};q,k,v=[t.to('mps') for t in cpu];del cpu
with torch.inference_mode():
 ref=run(q,k,v,'baseline').cpu()
 for name in ['baseline','qk32','pv32','qk32_pv32']:
  try:
   out=run(q,k,v,name).cpu();rows[name]={'byte_equal':torch.equal(ref.view(torch.uint8),out.view(torch.uint8)),'max_abs':(ref.float()-out.float()).abs().max().item(),'mismatched_elements':int((ref!=out).sum()),'samples_ms':[]};del out
  except Exception as e:rows[name]={'error':str(e)}
  print(name,rows[name],flush=True)
 del ref
 rng=random.Random(194)
 for i in range(10):
  order=[n for n,r in rows.items() if r.get('byte_equal')];rng.shuffle(order)
  for name in order:
   torch.mps.synchronize();t=time.perf_counter();out=run(q,k,v,name);torch.mps.synchronize();rows[name]['samples_ms'].append((time.perf_counter()-t)*1000);del out
 for r in rows.values():
  if r.get('samples_ms'):r['median_ms']=statistics.median(r['samples_ms'])
 a.output.write_text(json.dumps(rows,indent=2)+'\n')
 print({n:r.get('median_ms') for n,r in rows.items()},flush=True)
