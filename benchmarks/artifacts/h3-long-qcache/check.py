import os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[1]/'ComfyUI-AppleSilicon-Optimized'
sys.path.insert(0,str(repo))
os.environ.update(ASFP8_ATTN_BACKEND='pytorch',PYTORCH_MPS_FAST_MATH='0')
import torch
with torch.inference_mode():torch.zeros(1,device='mps');torch.mps.synchronize()
import pytest
raise SystemExit(pytest.main([str(repo/'tests/test_torch_mpp_qcache.py'),'-q']))
