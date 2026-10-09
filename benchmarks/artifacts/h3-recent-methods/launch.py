import os,runpy,sys
from pathlib import Path
import torch
with torch.inference_mode():torch.zeros(1,device='mps');torch.mps.synchronize()
core=Path('/Users/luigis/ComfyUI-Installs/ComfyUI-AppleSilicon-2.14/ComfyUI')
sys.path.insert(0,str(core));sys.argv[0]=str(core/'main.py');runpy.run_path(str(core/'main.py'),run_name='__main__')
