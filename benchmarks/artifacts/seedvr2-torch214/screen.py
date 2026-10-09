"""Small attention-only comparison with identical arithmetic and bounded buffers."""
import os,sys,json,time,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
core=Path('/Users/luigis/ComfyUI-Installs/ComfyUI-AppleSilicon-2.14/ComfyUI')
repo=next((p for p in root.parents if (p/'_patches/seedvr2_fast.py').is_file()),root.parents[1]/'ComfyUI-AppleSilicon-Optimized')
sys.path[:0]=[str(core),str(repo)]
import torch
from _patches import seedvr2_fast as fast
from comfy.ldm.seedvr import attention
from comfy.ldm.modules import attention as shared
from candidate import install
# Explicit PyTorch backend, no custom numerical attention implementation.
shared.optimized_attention=shared.attention_pytorch
fast._original_attention=attention.var_attention_optimized_split
base=fast._bounded_attention
install(fast)
new=fast._bounded_attention
result=[]
with torch.inference_mode():
 torch.manual_seed(574)
 lengths=[483]*36+[118,208,483,1]
 offsets=[0]
 for n in lengths:offsets.append(offsets[-1]+n)
 q,k,v=[torch.randn(sum(lengths),24,128,dtype=torch.bfloat16,device='mps') for _ in range(3)]
 reference=None
 for label,fn,batch in [('base8',base,8),('pack8',new,8),('base16',base,16),('pack16',new,16)]:
  os.environ['ASFP8_SEEDVR2_ATTN_BATCH']=str(batch)
  times=[]
  for i in range(8):
   torch.mps.synchronize();t=time.perf_counter()
   out=fn(q,k,v,24,offsets,offsets,skip_reshape=True)
   torch.mps.synchronize();times.append(time.perf_counter()-t)
  raw=out.cpu().view(torch.uint8)
  if reference is None:reference=raw
  result.append(dict(label=label,seconds=statistics.median(times[2:]),exact=torch.equal(reference,raw)))
(root/'attention-screen.json').write_text(json.dumps(result,indent=2))
print(result)
