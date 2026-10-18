#!/usr/bin/env python3
"""Standalone CUDA/Metal attention reference exporter (trusted tensor files only).

CUDA: pip install comfy-kitchen==0.2.38
  python kitchen_attention_reference.py --device cuda --output cuda.pt
Metal (from this repository):
  python benchmarks/tools/kitchen_attention_reference.py --device mps --output metal.pt
Compare on either machine:
  python kitchen_attention_reference.py --compare metal.pt cuda.pt
Use --input captured-qkv.pt to run identical real inputs instead of the seeded
synthetic fixture. No network transfers are performed by this script.
"""
import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path
import torch

VERSION = '0.2.38'
COMMIT = '42081bf9bb5fad3992c1256aab4f01b27cbb5816'


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(a, b):
    x, y = [torch.load(p, map_location='cpu', weights_only=True) for p in (a, b)]
    if x['input_hashes'] != y['input_hashes']:
        raise ValueError('Different inputs: comparison refused')
    result = {}
    for name in ('q', 'k', 'v', 'q_scale', 'k_scale', 'v_scale', 'output'):
        u, v = x['tensors'][name], y['tensors'][name]
        if u.shape != v.shape:
            raise ValueError(f'{name}: shape mismatch {u.shape} vs {v.shape}')
        d = u.float()-v.float()
        result[name] = dict(byte_equal=digest(u)==digest(v),
                            unequal=int((u!=v).sum()), max_abs=float(d.abs().max()),
                            rmse=float(d.square().mean().sqrt()))
    print(json.dumps({'left':x['metadata'],'right':y['metadata'],'comparison':result},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device',choices=('cuda','mps'));p.add_argument('--output',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--tokens',type=int,default=1153)
    p.add_argument('--heads',type=int,default=2);p.add_argument('--seed',type=int,default=1731)
    p.add_argument('--compare',nargs=2,type=Path);p.add_argument('--repeats',type=int,default=20)
    a=p.parse_args()
    if a.compare:return compare(*a.compare)
    if not a.device or not a.output:p.error('--device and --output are required')
    if a.repeats<1 or not 1024<a.tokens<=65536 or not 0<a.heads<=64:p.error('Invalid test size or repeat count')
    if a.input:
        inputs=torch.load(a.input,map_location='cpu',weights_only=True)
    else:
        gen=torch.Generator().manual_seed(a.seed)
        inputs={n:torch.randn((1,a.heads,a.tokens,128),generator=gen,dtype=torch.float32).to(torch.bfloat16) for n in ('q','k','v')}
        # A common component exercises the adaptive K anchor.
        inputs['k']=(inputs['k'].float()+torch.randn((1,a.heads,1,128),generator=gen)*2).to(torch.bfloat16)
    inputs={k:inputs[k].contiguous() for k in ('q','k','v')}
    hashes={k:digest(v) for k,v in inputs.items()}
    q,k,v=[inputs[n].to(a.device) for n in ('q','k','v')]
    metadata={'torch':str(torch.__version__),'device':a.device,'reference_version':VERSION,'reference_commit':COMMIT,'shape':list(q.shape),'seed':a.seed}
    with torch.inference_mode():
        if a.device=='cuda':
            installed=importlib.metadata.version('comfy-kitchen')
            if installed!=VERSION:raise RuntimeError(f'Expected comfy-kitchen {VERSION}, found {installed}; install the pinned version')
            from comfy_kitchen.sage_attention import prequantize_int8_attention,int8_attention_from_prequantized,int8_attention
            packed=prequantize_int8_attention(q,k,v);output=int8_attention_from_prequantized(packed)
            tensors={n:getattr(packed,n).cpu() for n in ('q','k','v','q_scale','k_scale','v_scale')}
            # Undo CUDA V's 16-token MMA permutation into [B,H,K,128].
            vi=tensors['v'];idx=torch.arange(vi.shape[-1]);w=idx%16
            permutation=(idx//16)*16+(w&1)+(((w>>3)&1)<<1)+(((w>>1)&1)<<2)+(((w>>2)&1)<<3)
            tensors['v']=vi[...,permutation].transpose(-1,-2).contiguous()
            # CUDA stores Q/K without padding, unlike the Metal matrix tiles.
            fn=lambda:int8_attention(q,k,v)
            sync=torch.cuda.synchronize;metadata['hardware']=torch.cuda.get_device_name()
        else:
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
            from _patches.kitchen_attention_mps import prequantize,attend_prequantized,attention
            packed=prequantize(q,k,v);output=attend_prequantized(packed)
            tensors={n:packed[n].cpu() for n in ('q','k','v','q_scale','k_scale','v_scale')}
            fn=lambda:attention(q,k,v);sync=torch.mps.synchronize
            metadata['cuda_equivalence']='unverified'
        tensors['q']=tensors['q'][...,:q.shape[2],:].contiguous()
        tensors['k']=tensors['k'][...,:k.shape[2],:].contiguous()
        tensors['v']=tensors['v'][...,:v.shape[2],:].contiguous()
        # Normalize scale shapes across backend packing conventions.
        for n in ('q_scale','k_scale','v_scale'):tensors[n]=tensors[n].reshape(q.shape[0],q.shape[1],-1).contiguous()
        # CUDA reserves complete 128-query scale groups. Its unused trailing
        # groups are not initialized and must not enter a byte comparison.
        tensors['q_scale']=tensors['q_scale'][...,:((q.shape[2]+31)//32)*8].contiguous()
        tensors['output']=output.cpu()
        for _ in range(3):fn()
        sync();times=[]
        for _ in range(a.repeats):
            start=time.perf_counter();fn();sync();times.append((time.perf_counter()-start)*1000)
        metadata['median_ms_including_quantization']=statistics.median(times)
        torch.save(dict(input_hashes=hashes,metadata=metadata,tensors=tensors),a.output)
    print(json.dumps(metadata,indent=2));print(f'Saved {a.output}')

if __name__=='__main__':main()
