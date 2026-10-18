"""Optional fused MOSS SiTU/ConvRot/INT8 prepass; reassociation is approximate."""
from functools import lru_cache
from pathlib import Path
import torch

@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())


def quantize(x, tables):
    if (type(x) is not torch.Tensor or not x.is_mps or x.dtype!=torch.bfloat16
            or x.ndim!=3 or x.shape[-1]!=16384 or not x.is_contiguous()
            or x.numel()>=2**31 or x.numel()==0):
        raise ValueError('Fused MOSS SiTU requires contiguous MPS BF16 [B,N,16384]')
    rows=x.numel()//16384
    q=torch.empty((rows,8192),device=x.device,dtype=torch.int8)
    scales=torch.empty((rows,1),device=x.device,dtype=torch.float32)
    _library().situ_quant(x,*tables,q,scales,threads=rows*128,group_size=128)
    return q,scales


def linear(linear, x, tables):
    from . import int8_linear_kernel_mps as i8
    weight=linear.weight;params=weight._params
    if (i8._kernel is None or not getattr(params,'convrot',False)
            or getattr(params,'convrot_groupsize',256)!=256
            or getattr(params,'transposed',False)
            or getattr(linear,'_full_precision_mm',False)
            or getattr(linear,'comfy_force_cast_weights',False)
            or getattr(linear,i8._RUNTIME_LORA_ATTR,None) is not None):
        return None
    q,scale=quantize(x,tables)
    return i8._int8_linear_prequant(q,scale,weight._qdata,params.scale,None,
                                  x.dtype,(*x.shape[:-1],8192))
