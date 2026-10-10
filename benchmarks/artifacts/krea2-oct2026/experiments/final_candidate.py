from pathlib import Path
import importlib.util,os,sys,torch
ROOT=Path(__file__).parent
attention=None;stats={};MODE='old'
def install():
 global attention,stats
 spec=importlib.util.spec_from_file_location('final_attention',ROOT/'attention_candidate.py');attention=importlib.util.module_from_spec(spec);spec.loader.exec_module(attention);attention.install();stats=attention.stats
 patches={n.rsplit('.',1)[-1]:m for n,m in list(sys.modules.items()) if '._patches.' in n}
 kernel=patches['int8_linear_kernel_mps']._kernel;old=kernel.i8_matmul2d_nt_fused_cs
 def verify(*args,**kw):
  result=old(*args,**kw)
  if 4096<=args[0].shape[0]<=4352 and args[1].shape[1]==6144 and args[1].shape[0] in (1536,6144):stats['eligible_gemm_calls']=stats.get('eligible_gemm_calls',0)+1
  if MODE=='optimized_verify' and 4096<=args[0].shape[0]<=4352 and args[1].shape[1]==6144 and args[1].shape[0] in (1536,6144):
   os.environ['ASFP8_KREA2_GEMM_TILE']='0'
   try:reference=old(*args,**kw)
   finally:os.environ['ASFP8_KREA2_GEMM_TILE']='1'
   assert torch.equal(result.cpu().view(torch.uint8),reference.cpu().view(torch.uint8)), 'Accepted GEMM changed bytes'
   stats['gemm_byte_checks']=stats.get('gemm_byte_checks',0)+1
  return result
 kernel.i8_matmul2d_nt_fused_cs=verify

def set_mode(mode):
 global MODE
 MODE=mode
 os.environ['ASFP8_KREA2_GEMM_TILE']='0' if mode=='old' else '1'
 attention.set_mode('hadamard_direct' if mode=='relaxed' else 'baseline')
