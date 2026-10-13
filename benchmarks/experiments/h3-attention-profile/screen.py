from pathlib import Path
import json,random,statistics,time,hashlib,torch
from candidates import run,pack,source,qc
assert torch.__version__.split("+")[0]=="2.14.1" and qc._torch_build_supported() and qc._hardware(), "Validated M5 Ultra / PyTorch 2.14.1 required"
D=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
cpu=torch.load(args.capture,map_location='cpu',weights_only=True)
assert len(cpu)==3 and all(t.dtype==torch.bfloat16 and t.ndim==4 and t.shape[0]==1 and t.shape[1]==56 and t.shape[3]==128 and 30000<t.shape[2]<=65536 and t.is_contiguous() for t in cpu)
assert cpu[0].shape==cpu[1].shape==cpu[2].shape
names=['baseline','pack_k','pack_v','pack_kv','identity_rescale']
rows={};q,k,v=[t.to('mps') for t in cpu];del cpu
with torch.inference_mode():
 ref=run(q,k,v,'baseline').cpu();refhash=hashlib.sha256(ref.view(torch.uint8).numpy().tobytes()).hexdigest()
 for name in names:
  try:
   out=run(q,k,v,name).cpu();equal=torch.equal(ref.view(torch.uint8),out.view(torch.uint8));delta=(ref.float()-out.float()).abs()
   rows[name]={'byte_equal':equal,'max_abs':delta.max().item(),'samples_ms':[],'source_sha256':hashlib.sha256(source(name).encode()).hexdigest()}
   print(name,rows[name],flush=True);del out,delta
  except Exception as e:rows[name]={'error':str(e)};print(name,str(e),flush=True)
 del ref
 rng=random.Random(84)
 for i in range(12):
  order=[n for n in names if rows[n].get('byte_equal')];rng.shuffle(order)
  for name in order:
   torch.mps.synchronize();t=time.perf_counter();out=run(q,k,v,name);torch.mps.synchronize();rows[name]['samples_ms'].append((time.perf_counter()-t)*1000);del out
  print('round',i,flush=True)
 for row in rows.values():
  if row.get('samples_ms'):row['median_ms']=statistics.median(row['samples_ms'])
 # Repacking and kernel-only breakdown; do not mistake kernel-only for practical cost.
 pk,pv=pack(k,16),pack(v,32);torch.mps.synchronize()
 for name,packed in [('pack_k',(pk,v)),('pack_v',(k,pv)),('pack_kv',(pk,pv))]:
  if not rows[name].get('byte_equal'):continue
  samples=[]
  for i in range(6):
   torch.mps.synchronize();t=time.perf_counter();out=run(q,k,v,name,packed=packed);torch.mps.synchronize();samples.append((time.perf_counter()-t)*1000);del out
  rows[name]['kernel_only_ms']=samples
 result={'reference_sha256':refhash,'shape':list(q.shape),'results':rows,'driver_gib':torch.mps.driver_allocated_memory()/2**30}
 args.output.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:{a:b for a,b in v.items() if a not in ('samples_ms','kernel_only_ms')} for k,v in rows.items()},indent=2),flush=True)
