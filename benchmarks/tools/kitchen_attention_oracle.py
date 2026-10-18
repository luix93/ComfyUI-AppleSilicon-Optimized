# Quantization/softmax policy adapted from Comfy Kitchen v0.2.38.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Test-only CPU oracle. This is not a substitute for the CUDA reference run.
import torch,math

def rotate(x):
 x=x.float().reshape(*x.shape[:-1],32,4)
 bits=[0x1035997b,0x8087f5ee,0xee2e4e1a,0x71132418]
 signs=torch.tensor([1 if (bits[i//32]>>(i%32))&1 else -1 for i in range(128)]).reshape(32,4)
 x=x*signs
 a=x[...,0]+x[...,1];b=x[...,0]-x[...,1];c=x[...,2]+x[...,3];d=x[...,2]-x[...,3]
 x=torch.stack((a+c,b+d,a-c,b-d),-1)*.5
 for bit in (1,2,4,8,16):
  idx=torch.arange(32);other=x[...,idx^bit,:]
  x=torch.where(((idx&bit)>0)[:,None],other-x,x+other)
 return (x*0.1767766952966369).flatten(-2)

def quant_ref(x,anchors):
 q,k,v=[x[key].float() for key in ('q','k','v')]
 b,h,nq,d=q.shape;nk=k.shape[2];qp=(nq+31)//32*32;kp=(nk+127)//128*128
 k=k.clone()
 for a in range(b):
  for z in range(h):
   if anchors[a,z]>=0:k[a,z]-=k[a,z,anchors[a,z]].clone()
 q=torch.nn.functional.pad(rotate(q),(0,0,0,qp-nq));k=torch.nn.functional.pad(rotate(k),(0,0,0,kp-nk))
 qg=q.reshape(b,h,qp//32,4,8,128);kg=k.reshape(b,h,kp//128,16,4,2,128)
 qs=qg.abs().amax((3,5))/127+1e-7;ks=kg.abs().amax((3,5,6))/127+1e-7
 qi=(qg*(1/qs[:,:,:,None,:,None])).round().clamp(-127,127).to(torch.int8).reshape(b,h,qp,128)
 ki=(kg*(1/ks[:,:,:,None,:,None,None])).round().clamp(-127,127).to(torch.int8).reshape(b,h,kp,128)
 vs=(v.abs().amax(-2)*(1/127)).clamp_min(1e-12)
 vi=torch.nn.functional.pad((v*(1/vs[...,None,:])).round().clamp(-127,127).to(torch.int8),(0,0,0,kp-nk))
 return dict(q=qi,k=ki,v=vi,q_scale=qs.flatten(-2),k_scale=ks.flatten(-2),v_scale=vs)

def attention_ref(p,chunk=32):
 # Scalar policy oracle: different reduction/exp2 implementation from CUDA.
 q,k,v=[p[key].float() for key in ('q','k','v')];nq,nk=p['nq'],p['nk'];b,h=q.shape[:2]
 out=torch.empty((b,h,nq,128));kp=k.shape[2]
 for start in range(0,nq,chunk):
  stop=min(start+chunk,nq);rows=torch.arange(start,stop);qs=p['q_scale'][...,(rows//32)*8+rows%8]
  m=torch.full((b,h,stop-start,1),-float('inf'));den=torch.zeros_like(m);num=torch.zeros((b,h,stop-start,128))
  for c in range(0,kp,128):
   cols=torch.arange(c,c+128);ks=p['k_scale'][...,(cols//128)*4+(cols%8)//2]
   sc=(qs[...,None]*ks[...,None,:])*(torch.tensor(128**-.5)*torch.tensor(math.log2(math.e)))
   scores=q[...,start:stop,:]@k[...,c:c+128,:].transpose(-1,-2)
   # Emulate FP32 FMA with a double intermediate.
   tm=(scores.double()*sc.double()-torch.tensor(7.9943534).double()).float().masked_fill((cols>=nk)[None,None,None,:],-float('inf')).amax(-1,keepdim=True)
   smaller=torch.exp2(-(m-tm).abs());f=torch.where(m<tm,smaller,1.);tf=torch.where(tm<m,smaller,1.);m=torch.maximum(m,tm)
   prob=torch.exp2((scores.double()*sc.double()-tm.double()).float()).round().clamp(0,255).masked_fill((cols>=nk)[None,None,None,:],0)
   den=(prob.sum(-1,keepdim=True).double()*tf.double()+(den*f).double()).float()
   prod=prob@v[...,c:c+128,:]
   num=(prod.double()*tf.double()+(num*f).double()).float()
  out[...,start:stop,:]=(num*(1/den))*p['v_scale'][...,None,:]
 return out.to(torch.bfloat16)

