import sys,inspect,collections
import torch
stats={}
mode='baseline'
def install():
 global cache,old_support,extended_support,original
 cache=next(m for name,m in sys.modules.items() if name.endswith('._patches.torch_mpp_qcache'))
 original_support=cache._supports
 def old_support(q,k,v,kwargs):
  if q.ndim==4 and q.shape[2]>32768:return False
  return original_support(q,k,v,kwargs)
 source=inspect.getsource(original_support).replace('not 2048 <= q.shape[2] <= 32768','not 2048 <= q.shape[2] <= (65536 if q.shape[1] == k.shape[1] == 56 else 32768)')
 ns=dict(cache.__dict__);exec(source,ns);extended_support=ns['_supports']
 original=cache.maybe_attention
 def observe(q,k,v,**kwargs):
  out=original(q,k,v,**kwargs)
  key=f'{q.shape[1]} heads / {q.shape[2]} tokens / '+('cached' if out is not None else 'fallback')
  stats[key]=stats.get(key,0)+1
  if out is not None and mode=='extended-verify' and stats.get('verified',0)<2:
   ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,**{k:v for k,v in kwargs.items() if k!='output_bshd'})
   assert torch.equal(out.cpu().contiguous().view(torch.uint8),ref.cpu().contiguous().view(torch.uint8)), 'Attention bytes differ'
   stats['verified']=stats.get('verified',0)+1
  return out
 cache.maybe_attention=observe
def set_mode(value):
 global mode
 mode=value;stats.clear();cache._supports=extended_support if value.startswith('extended') else old_support
