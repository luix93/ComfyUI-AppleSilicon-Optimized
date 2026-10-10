import torch,time,sys,weakref
MODE='baseline';stats={};cache=weakref.WeakKeyDictionary()
def set_mode(mode):
 global MODE
 MODE=mode;stats.clear();cache.clear()
def install():
 import comfy.ldm.krea2.model as m
 old=m.TextFusionTransformer.forward
 def forward(self,x,mask=None,transformer_options={}):
  if MODE.startswith('cache'):
   cpu=x.detach().cpu().contiguous();entry=cache.get(self)
   if entry is not None and torch.equal(entry[0].view(torch.uint8),cpu.view(torch.uint8)):
    out=entry[1].clone();stats['hits']=stats.get('hits',0)+1
    if MODE=='cache_verify':
     ref=old(self,x,mask,transformer_options)
     assert torch.equal(out.cpu().view(torch.uint8),ref.cpu().view(torch.uint8)), 'Text fusion cache divergence'
     stats['byte_checks']=stats.get('byte_checks',0)+1
    return out
   out=old(self,x,mask,transformer_options);cache[self]=(cpu.clone(),out.detach().clone());stats['misses']=stats.get('misses',0)+1
   return out
  torch.mps.synchronize();begin=time.perf_counter();out=old(self,x,mask,transformer_options);torch.mps.synchronize()
  stats['fusion_seconds']=stats.get('fusion_seconds',0)+(time.perf_counter()-begin);stats['calls']=stats.get('calls',0)+1
  stats['input_dtype']=str(x.dtype);stats['input_shape']=list(x.shape)
  if 'version_errors' not in stats:
   bad=0
   for p in self.parameters():
    try:p._version
    except RuntimeError:bad+=1
   stats['version_errors']=bad;stats['parameters']=len(list(self.parameters()))
   stats['option_keys']=list(transformer_options)
  return out
 m.TextFusionTransformer.forward=forward
