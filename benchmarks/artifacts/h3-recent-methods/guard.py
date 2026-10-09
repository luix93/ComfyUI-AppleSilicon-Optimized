import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import psutil

D = Path(__file__).resolve().parent
try:
    with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=2) as response:
        queue = json.load(response)
    assert not queue['queue_running'] and not queue['queue_pending'], 'User server busy'
except urllib.error.URLError:
    pass
script = sys.argv[1]
peak = 0
with (D / (Path(script).stem + '.log')).open('w') as log:
    child = subprocess.Popen([sys.executable, str(D/script)], stdout=log, stderr=subprocess.STDOUT)
    proc = psutil.Process(child.pid)
    start = time.monotonic()
    try:
        while child.poll() is None:
            try:
                rss = proc.memory_info().rss
            except psutil.NoSuchProcess:
                # The worker can finish between poll() and the RSS sample.
                if child.poll() is not None:
                    break
                raise
            peak = max(peak, rss)
            assert rss < 80 * 2**30, '80 GiB process RSS cutoff'
            assert psutil.virtual_memory().available > 48 * 2**30, '48 GiB available cutoff'
            assert time.monotonic() - start < 1800, '30-minute cutoff'
            time.sleep(.25)
        assert child.returncode == 0, f'Exit {child.returncode}; see log'
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        (D / (Path(script).stem + '-guard.json')).write_text(json.dumps(dict(
            peak_rss_gib=peak/2**30, seconds=time.monotonic()-start, exit=child.returncode)))
print('Completed', script, 'peak RSS GiB', peak/2**30)
