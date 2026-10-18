"""Experimental Kitchen-style dense INT8 attention. Not CUDA-byte-validated.

D128 BF16, equal head counts, unmasked non-causal attention only. No dispatch
is changed by importing this module. All keys participate; this is quantization,
not sparse attention. Source policy: Kitchen v0.2.38, Apache-2.0.
"""
from functools import lru_cache
from pathlib import Path
import math
import torch

REFERENCE_COMMIT = '42081bf9bb5fad3992c1256aab4f01b27cbb5816'


def requested(config):
    """Honor a checkpoint's INT8 request; never infer it from INT8 weights."""
    configs = config if isinstance(config, list) else [config]
    return any(isinstance(c, dict) and c.get('attention') == 'comfy_kitchen_int8'
               for c in configs)

@lru_cache(maxsize=8)
def _library(nq, nk):
    source = Path(__file__).with_suffix('.metal').read_text()
    return torch.mps.compile_shader(source.replace('@NQ@', str(nq)).replace('@NK@', str(nk)))


def validate(q, k, v):
    from .h3_sol_mps import supported, workspace_budget
    if not supported():
        raise ValueError('Kitchen-style Metal prototype requires M5 and Metal 4')
    if (q.ndim != 4 or k.ndim != 4 or v.shape != k.shape
            or q.shape[:2] != k.shape[:2] or q.shape[-1] != 128 or k.shape[-1] != 128
            or not 1024 < k.shape[2] <= 65536 or not 0 < q.shape[2] <= 65536
            or q.shape[0] < 1 or q.shape[1] < 1
            or any(type(t) is not torch.Tensor or t.dtype != torch.bfloat16
                   or t.device != q.device or not t.is_mps for t in (q, k, v))):
        raise ValueError('Requires BF16 MPS [B,H,N,128], equal heads, 1025..65536 keys')
    # Layout copies + quantized buffers/scales + outputs; no quadratic scratch.
    required = (q.numel() + k.numel() + v.numel()) * 4 + q.numel() * 4 + 2**20
    budget, _ = workspace_budget()
    if required > budget:
        raise ValueError(f'Kitchen-style attention needs {required/2**30:.2f} GiB scratch, budget {budget/2**30:.2f} GiB')


def prequantize(q, k, v):
    validate(q, k, v)
    b, h, nq, _ = q.shape; nk = k.shape[2]; bh = b*h
    qp, kp = (nq+31)//32*32, (nk+127)//128*128
    lib = _library(nq, nk)
    def alloc(shape, dtype): return torch.empty(shape, dtype=dtype, device=q.device)
    qi=alloc((b,h,qp,128),torch.int8);ki=alloc((b,h,kp,128),torch.int8);vi=alloc((b,h,kp,128),torch.int8)
    qs=alloc((b,h,qp//4),torch.float32);ks=alloc((b,h,kp//32),torch.float32);vs=alloc((b,h,128),torch.float32)
    anchors=alloc((b,h),torch.int32); sums=alloc((b,h,kp//128,128),torch.int32)
    q,k,v=(t.contiguous() for t in (q,k,v))
    lib.anchor(k,anchors,threads=bh*128,group_size=128)
    lib.quant_q(q,qi,qs,threads=(qp//32*128,bh),group_size=(128,1))
    lib.quant_k(k,ki,ks,anchors,threads=(kp//128*128,bh),group_size=(128,1))
    lib.quant_v(v,vi,vs,threads=(16*128,bh),group_size=(128,1))
    lib.v_sums(vi,sums,threads=(kp,bh),group_size=(128,1))
    return dict(q=qi,k=ki,v=vi,q_scale=qs,k_scale=ks,v_scale=vs,anchors=anchors,v_sums=sums,nq=nq,nk=nk)


def attend_prequantized(p, scale=None):
    nq,nk=p['nq'],p['nk'];b,h=p['q'].shape[:2]
    scale=128**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale<=0: raise ValueError('Positive finite attention scale required')
    out=torch.empty((b,h,nq,128),dtype=torch.bfloat16,device=p['q'].device)
    _library(nq,nk).attend(*(p[n] for n in ('q','k','v','q_scale','k_scale','v_scale','v_sums')),out,scale,
                         threads=((nq+63)//64*128,b*h),group_size=(128,1))
    return out


def attention(q,k,v,*,scale=None):
    return attend_prequantized(prequantize(q,k,v),scale)
