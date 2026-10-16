from build import build,D
import torch,time,json,statistics,random,gc,psutil,urllib.request,argparse
from pathlib import Path
p=argparse.ArgumentParser(description="Diagnostic-only H3 fusion screen; M5 Ultra, captured 19508-token inputs.")
p.add_argument("--captures",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
options=p.parse_args()

def guard():
 assert psutil.virtual_memory().available>48*2**30
 try:
  q=json.load(urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=2))
 except OSError:return
 assert not q['queue_running'] and not q['queue_pending'],'User queue active'

def device(x):
 if isinstance(x,torch.Tensor):return x.to('mps')
 if isinstance(x,(tuple,list)):return type(x)(device(a) for a in x)
 if isinstance(x,dict):return {k:device(v) for k,v in x.items()}
 return x

def equal(a,b):
 return all(torch.equal(x.cpu().contiguous().view(torch.uint8),y.cpu().contiguous().view(torch.uint8)) for x,y in zip(a,b))

guard();base,mod=build();print('BUILT',flush=True)
results={}
for kind,name in [('qkv','i8_matmul2d_nt_fused_cs_qkv_rms_rope'),('mlp','i8_mlp_bridge_prequant')]:
 cap=torch.load(options.captures/(name+'.pt'),weights_only=True);args=device(cap['args']);kw=device(cap['kwargs']);ref=cap['out']
 funs={'current':lambda:getattr(base,name)(*args,**kw)}
 if kind=='qkv':
  for v in [32,64,512]:funs[str(v)]=lambda v=v:mod.qkv_variant(*args,v)
 else:
  for v in [4096,16384,0]:funs['chunk'+str(v)]=lambda v=v:base.i8_mlp_bridge_prequant(*args,v)
  for v in [16,32,128,129,65]:funs['fused'+str(v)]=lambda v=v:mod.fc1_fused(*args,v)
 good={};outcomes={}
 for label,fn in funs.items():
  guard();t=time.perf_counter()
  try:
   out=fn();torch.mps.synchronize();ok=equal(out,ref);del out
   outcomes[label]={'exact':ok,'times_ms':[]};print(kind,label,'exact',ok,'compile/warm',time.perf_counter()-t,flush=True)
   if ok:good[label]=fn
  except Exception as e:
   outcomes[label]={'error':str(e)};print(kind,label,'ERROR',e,flush=True)
 for rep in range(7):
  order=list(good);random.Random(rep).shuffle(order)
  for label in order:
   guard();fn=good[label];torch.mps.synchronize();t=time.perf_counter()
   for _ in range(3):out=fn()
   torch.mps.synchronize();ms=(time.perf_counter()-t)*1000/3;del out
   outcomes[label]['times_ms'].append(ms)
 for label,d in outcomes.items():
  if d.get('times_ms'):d['median_ms']=statistics.median(d['times_ms'])
 results[kind]=outcomes;options.output.write_text(json.dumps(results,indent=2));print(kind,json.dumps(outcomes),flush=True)
 del args,kw,ref,cap;gc.collect();torch.mps.empty_cache()
