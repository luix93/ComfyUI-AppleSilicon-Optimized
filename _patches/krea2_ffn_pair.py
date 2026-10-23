"""Guarded Krea INT8 paired projection; exact intermediate BF16 stores."""
from pathlib import Path
import importlib
import inspect
import hashlib
import logging
import os

def patch(name):
    return importlib.import_module(__package__+'.'+name)

def _stack_supported():
    qc=patch('torch_mpp_qcache')
    return (os.environ.get('ASFP8_DIT_EXT')=='1'
            and os.environ.get('PYTORCH_MPS_FAST_MATH','0')=='0'
            and torch.__version__.split('+')[0]=='2.14.1'
            and qc._torch_build_supported() and qc._hardware())

_dead=False
def _ready():
    global _library,_dead
    if _dead:return False
    if _library is None:
        try:
            _library=torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
        except Exception as exc:
            _dead=True
            logging.warning('[AppleSilicon-Optimized/paired_ffn] compilation unavailable: %s',exc)
    return _library is not None

import torch

_library=None
calls=0
variant=(128,128,4,4,False)


def project(x,g,u):
    global _library,calls
    assert _library is not None
    i8=patch('int8_linear_kernel_mps')
    q,s=i8._quant_cached(x,True,256)
    m,k=q.shape;n=g._qdata.shape[0]*2
    out=torch.empty((*x.shape[:-1],n//2),device=x.device,dtype=x.dtype)
    bm,bn,wm,wn,barrier=variant;tm,tn=(m+bm-1)//bm,(n+bn-1)//bn
    getattr(_library,f'pair_{bm}_{bn}_{wm}_{wn}_{int(barrier)}')(
        q,g._qdata,u._qdata,out,m,n,k,tm,tn,s.reshape(-1),g._params.scale.reshape(-1),u._params.scale.reshape(-1),
        threads=(tn*wm*wn*32,tm,1),group_size=(wm*wn*32,1,1))
    calls+=1
    return out


def install(selected=None):
    global variant
    if selected is not None:variant=selected
    if not _stack_supported():return lambda mode: None
    import comfy.ldm.krea2.model as krea
    import comfy.ops as ops
    from torch.nn.modules import module as module_hooks
    guards=patch('moss_ffn_mps')
    owner=patch('moss_fast')
    original=krea.SwiGLU.forward
    glue=patch('krea2_glue')
    try: known=original is glue._mlp and hashlib.sha256(inspect.getsource(glue._original_mlp).encode()).hexdigest()=='a96a268020f231c92231f570ce5a041bb535ab37f58588071211e7ba8957ef27'
    except (OSError,TypeError):known=False
    if not known:return lambda mode: None
    def candidate(self,x):
        if (os.environ.get('ASFP8_KREA2_FFN_PAIR','1')!='1' or not glue._enabled() or torch.is_grad_enabled() or not x.is_mps or x.dtype!=torch.bfloat16 or not x.is_contiguous()
                or x.ndim!=3 or x.shape[0]!=1 or not 4096<=x.shape[1]<=14336 or x.shape[-1]!=6144):
            return original(self,x)
        if (module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
                or not owner._owned_dequantizing_cast(ops.cast_bias_weight)):
            return original(self,x)
        for layer in (self.gate,self.up):
            if ('forward' in layer.__dict__ or getattr(layer,'comfy_force_forward',False)
                    or getattr(layer,'_compiled_call_impl',None) is not None
                    or type(layer).__call__ is not torch.nn.Module.__call__
                    or not guards._eligible_layer(layer,(16384,6144),x.device)):
                return original(self,x)
        if not _ready():return original(self,x)
        ops.run_every_op()
        return self.down(project(x,self.gate.weight,self.up.weight))
    def switch(mode):krea.SwiGLU.forward=candidate if mode=='candidate' else original
    switch('candidate')
    return switch
