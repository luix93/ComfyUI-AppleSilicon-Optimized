import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

root=Path(__file__).resolve().parent
label,*command=sys.argv[1:]
env=dict(os.environ,PYTORCH_MPS_HIGH_WATERMARK_RATIO=os.environ.get('SEED_GUARD_HIGH','0.4'),PYTORCH_MPS_LOW_WATERMARK_RATIO=os.environ.get('SEED_GUARD_LOW','0.3'),PYTHONUNBUFFERED='1')
start=time.monotonic();peak=0;reason=None
with (root/(label+'.log')).open('w') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    process=psutil.Process(child.pid)
    while child.poll() is None:
        try:rss=process.memory_info().rss
        except psutil.NoSuchProcess:break
        peak=max(peak,rss)
        if rss>96*2**30:reason='96 GiB RSS limit'
        elif psutil.virtual_memory().available<64*2**30:reason='64 GiB system available floor'
        elif time.monotonic()-start>900:reason='900-second timeout'
        if reason:child.kill();break
        time.sleep(.1)
    code=child.wait()
report=dict(exit_code=code,seconds=time.monotonic()-start,peak_rss_gib=peak/2**30,watchdog_reason=reason)
(root/(label+'-guard.json')).write_text(json.dumps(report,indent=2));print(report)

sys.exit(code if code >= 0 else 128-code)
