"""Sequential post-benchmark regression checks; no concurrent GPU jobs."""
import subprocess,sys,json
from pathlib import Path
root=Path(__file__).resolve().parent
for seed in (42,271828):
 for mode in ('native-base','native'):
  label=f'seed-{seed}-{mode}'
  subprocess.run([sys.executable,str(root/'guard.py'),label,sys.executable,str(root/'run.py'),'--label',label,'--mode',mode,'--fix','--installed','--seed',str(seed),'--input-size','320','--limit-gib','40'],check=True)
 a=json.loads((root/f'seed-{seed}-native-base.json').read_text())['records'][0]
 b=json.loads((root/f'seed-{seed}-native.json').read_text())['records'][0]
 for key in ('encoded_sha256','sampled_sha256','frames_sha256'):assert a[key]==b[key],(seed,key)
subprocess.run([sys.executable,str(root/'guard.py'),'fp16-smoke',sys.executable,str(root/'run.py'),'--label','fp16-smoke','--mode','native','--model','fp16','--fix','--installed','--input-size','320','--repeats','2','--limit-gib','40'],check=True)
print('Extra seeds, FP16 loader smoke and kernel/safety tests completed.',flush=True)
