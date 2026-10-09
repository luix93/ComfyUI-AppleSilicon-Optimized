"""Final installed-node verification at saved 5 s and 10 s settings."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for frames in (120,240):
    label=f'full{frames}-installed'
    command=[sys.executable,str(root/'guard.py'),label,sys.executable,
             str(root/'run.py'),'--label',label,'--frames',str(frames),
             '--video-sync','production','--limit-gib','64','--repeats','2',
             '--fix','--installed','--core-video','--export-video']
    code=subprocess.call(command)
    if code:sys.exit(code)
