import os,sys,inspect,json,time,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[1]/'ComfyUI-AppleSilicon-Optimized'
sys.path.insert(0,str(repo))
os.environ.update(ASFP8_ATTN_BACKEND='pytorch',PYTORCH_MPS_FAST_MATH='0',ASFP8_MPP_QCACHE='1')
import torch
from _patches import torch_mpp_qcache as cache
source=inspect.getsource(cache._supports).replace('not 2048 <= q.shape[2] <= 32768','not 2048 <= q.shape[2] <= (65536 if q.shape[1] == k.shape[1] == 56 else 32768)')
exec(source,cache.__dict__)
records=[]
def sync():
 torch.mps.synchronize()
 assert torch.mps.driver_allocated_memory()<32*2**30
with torch.inference_mode():
 torch.manual_seed(9827)
 for n in (32769,45000,65536):
  storage=torch.randn(1,n+1,3,56,128,dtype=torch.bfloat16,device='mps')
  q,k,v=[storage[:,1:,i].transpose(1,2) for i in range(3)]
  times={'stock':[],'extended':[]};reference=None
  for iteration in range(4):
   for label in ('stock','extended'):
    sync();start=time.perf_counter()
    out=torch.nn.functional.scaled_dot_product_attention(q,k,v) if label=='stock' else cache.maybe_attention(q,k,v)
    assert out is not None
    sync();times[label].append(time.perf_counter()-start)
    if iteration==0:
     raw=out.cpu().contiguous().view(torch.uint8)
     if label=='stock':reference=raw
     else:assert torch.equal(raw,reference),f'Non-exact N={n}'
     del raw
    del out
  record=dict(tokens=n,layout='strided QKV, nonzero offset',exact=True,median={k:statistics.median(v[1:]) for k,v in times.items()},times=times,driver_gib=torch.mps.driver_allocated_memory()/2**30)
  records.append(record);(root/'micro.json').write_text(json.dumps(records,indent=2));print(record,flush=True)
  del storage,q,k,v,reference
  torch.mps.empty_cache()
