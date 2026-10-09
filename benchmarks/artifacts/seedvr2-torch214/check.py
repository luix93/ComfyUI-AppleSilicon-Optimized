import os,sys
from pathlib import Path
core=Path('/Users/luigis/ComfyUI-Installs/ComfyUI-AppleSilicon-2.14/ComfyUI')
root=Path(__file__).resolve().parent
repo=next((p for p in root.parents if (p/'_patches/seedvr2_fast.py').is_file()),root.parents[1]/'ComfyUI-AppleSilicon-Optimized')
sys.path[:0]=[str(core),str(repo)]
os.environ['ASFP8_DIT_EXT']='1'
os.environ['ASFP8_INT8_EXT']='1'
import torch
with torch.inference_mode():
 torch.zeros(1,device='mps');torch.mps.synchronize()
 bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
 finite=bits[torch.isfinite(bits)].to('mps')
 assert torch.equal(finite.cpu().view(torch.uint8),(finite/1.0).cpu().view(torch.uint8))
 print('All finite FP16 bit patterns preserve bytes under division by one.',flush=True)
import pytest
raise SystemExit(pytest.main([str(repo/'tests/test_seedvr2_fast.py'),str(repo/'tests/test_seedvr2_loader.py'),str(repo/'tests/test_seedvr2_vae_mps.py'),'-q','-k','not groupnorm_silu_fold']))
