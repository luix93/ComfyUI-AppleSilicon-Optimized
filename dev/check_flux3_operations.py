"""Validate Flux.3 operations against Comfy Kitchen/PyTorch on the reviewed MPS stack.

Run with ComfyUI's Python:
    python dev/check_flux3_operations.py --comfy /path/to/ComfyUI
"""
import argparse
import json
import sys
from pathlib import Path
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--comfy', type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(args.comfy.resolve())]
# Comfy args must not see the diagnostic script's options.
sys.argv = [sys.argv[0]]
import comfy.ops
from comfy_kitchen.backends.eager.rope import rms_rope
from _patches import flux3_fast as p
from types import SimpleNamespace
rows=[]
p._rope_calls = p._activation_calls = 0
with torch.inference_mode():
 torch.manual_seed(123)
 for B,Bpe,H,N in ((1,1,24,37),(2,1,24,37),(2,2,24,37),(1,1,8,19)):
  qkv=torch.randn(B,N,H*384,device='mps',dtype=torch.bfloat16)
  q,k,_=[a.view(B,N,H,128) for a in qkv.chunk(3,-1)]
  pe=torch.randn(Bpe,N,1,64,2,2,device='mps',dtype=torch.float32)
  norms=[comfy.ops.manual_cast.RMSNorm(128,eps=1e-6,device='mps',dtype=torch.bfloat16) for _ in range(2)]
  for n in norms:n.weight.copy_(torch.randn_like(n.weight))
  module=SimpleNamespace(query_norm=norms[0],key_norm=norms[1])
  def ref(m,q,k,pe):return rms_rope(q,k,pe,m.query_norm.weight,m.key_norm.weight,1e-6)
  a=ref(module,q,k,pe);b=p._qknorm(ref,module,q,k,pe)
  exact=all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(a,b))
  assert exact,(B,Bpe,H,N)
  rows.append({'kind':'rope','shape':[B,N,H,128],'pe_batch':Bpe,'byte_identical':exact})
 for shape in ((1,37,18432),(2,17,18432)):
  x=torch.randn(shape,device='mps',dtype=torch.bfloat16)
  original=lambda v:torch.nn.functional.silu(v.chunk(2,-1)[0])*v.chunk(2,-1)[1]
  token=p._scope.set({'part':'WEIGHT_CACHE'})
  try:y=p._activation(original,x)
  finally:p._scope.reset(token)
  ref=original(x);exact=torch.equal(y.view(torch.uint8),ref.view(torch.uint8));assert exact
  rows.append({'kind':'swiglu','shape':list(shape),'byte_identical':exact})
assert p._rope_calls == 4 and p._activation_calls == 2, 'An accelerated operation fell back'
print(json.dumps(rows, indent=2))
