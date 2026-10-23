"""Guarded MOSS BF16 FC1/SiTU fusion, preserving the down projection."""
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
variant=(64,128,2,4,False)


def project(x,weight,tables):
    global _library,calls
    assert _library is not None
    m=x.numel()//x.shape[-1];k=x.shape[-1];n=weight.shape[0]
    out=torch.empty((*x.shape[:-1],n//2),device=x.device,dtype=x.dtype)
    bm,bn,wm,wn,barrier=variant;tm,tn=(m+bm-1)//bm,(n+bn-1)//bn
    wa,wb=weight.chunk(2,dim=0)
    getattr(_library,f'moss_{bm}_{bn}_{wm}_{wn}_{int(barrier)}')(
        x,wa,wb,out,m,n,k,tm,tn,*tables,threads=(tn*wm*wn*32,tm,1),group_size=(wm*wn*32,1,1))
    calls+=1
    return out


def install():
    if not _stack_supported():return lambda mode: None
    import comfy.ldm.moss_image.model as m
    import comfy.ops as ops
    from torch.nn.modules import module as module_hooks
    original=m.Block.forward;owner=patch('moss_fast')
    if (not owner._installed or original.__globals__ is not owner.__dict__
            or not owner._function_matches(getattr(original,'__wrapped__',None),owner._BLOCK_HASH)):
        return lambda mode: None
    def candidate(self,x,mod,pe,transformer_options={}):
        up=self.proj['ffn_up_gate'];w=up.weight
        state=owner._scope.get()
        if (os.environ.get('ASFP8_MOSS_BF16_FFN_PAIR','1')!='1' or torch.is_grad_enabled() or state is None or not owner._direct_ffn_supported or not owner._enabled('SITU')
                or getattr(m.modulated_rms_norm,'__globals__',None) is not owner.__dict__
                or getattr(ops.linear_input_act,'__globals__',None) is not owner.__dict__
                or getattr(ops.INPUT_ACT_EAGER.get('situ'),'__globals__',None) is not owner.__dict__
                or module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
                or id(self.proj['ffn_down']) not in state['modules']
                or not owner._bf16(x) or x.ndim!=3 or x.shape[0]!=1
                or not 4096<=x.shape[1]<=6656 or not x.is_contiguous() or x.shape[-1]!=3072
                or type(w) not in (torch.Tensor,torch.nn.Parameter) or w.dtype!=torch.bfloat16
                or not w.is_mps or w.shape!=(16384,3072) or not w.is_contiguous()
                or type(up) not in (ops.manual_cast.Linear,ops.disable_weight_init.Linear)
                or up.bias is not None or 'forward' in up.__dict__ or hasattr(up,'_v')
                or getattr(up,'_compiled_call_impl',None) is not None
                or any(getattr(up,key,None) for key in ['_full_precision_mm','comfy_force_forward',
                    'comfy_force_cast_weights','weight_function','bias_function','weight_lowvram_function',
                    'bias_lowvram_function','_forward_hooks','_forward_pre_hooks'])
                or not owner._owned_dequantizing_cast(ops.cast_bias_weight) or not owner._ready() or not _ready()):
            return original(self,x,mod,pe,transformer_options)
        shift_a,scale_a,gate_a,shift_f,scale_f,gate_f=mod
        y=m.modulated_rms_norm(x,self.norm1,scale_a,shift_a)
        if self.fused_qkv:q,k,v=self.proj['qkv'](y).chunk(3,dim=-1)
        else:q,k,v=(self.proj[n](y) for n in ('q','k','v'))
        out=m.attention(q,k,v,self.q_norm,self.k_norm,pe,self.num_heads,self.comfy_attention,transformer_options)
        x=m.gated_residual(x,self.proj['o'](out),gate_a)
        y=m.modulated_rms_norm(x,self.norm2,scale_f,shift_f)
        ops.run_every_op()
        with ops.CastBiasWeightContext(up,y,offloadable=True) as (weight,bias):
            activated=project(y,weight,owner._tables)
        return m.gated_residual(x,self.proj['ffn_down'](activated),gate_f)
    def switch(mode):m.Block.forward=candidate if mode=='candidate' else original
    switch('candidate')
    return switch
