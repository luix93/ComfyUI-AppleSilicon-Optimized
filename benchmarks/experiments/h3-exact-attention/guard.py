"""Bounded subprocess runner for the isolated attention experiments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Watchdog JSON report')
    parser.add_argument('--queue-url', default='http://127.0.0.1:8188/queue')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A subprocess command is required after --')
    def queue_busy():
        try:
            with urllib.request.urlopen(args.queue_url, timeout=2) as response:
                state = json.load(response)
            return bool(state['queue_running'] or state['queue_pending'])
        except urllib.error.URLError:
            return False
    if queue_busy():
        parser.error('ComfyUI queue is busy; no test started.')
    if psutil.virtual_memory().available < 64*2**30:
        parser.error('Less than 64 GiB system memory available; no test started.')
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.4',
               PYTORCH_MPS_LOW_WATERMARK_RATIO='0.3')
    start = time.monotonic()
    peak = 0
    reason = None
    last_queue = start
    child = subprocess.Popen(command, env=env)
    process = psutil.Process(child.pid)
    try:
        while child.poll() is None:
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss)
            if rss > 96*2**30:
                reason = '96 GiB process RSS limit'
            elif psutil.virtual_memory().available < 64*2**30:
                reason = '64 GiB system available floor'
            elif time.monotonic()-start > 900:
                reason = '900-second deadline'
            if time.monotonic()-last_queue >= 5:
                last_queue = time.monotonic()
                if queue_busy():
                    reason = 'User ComfyUI queue busy'
            if reason:
                child.kill()
                break
            time.sleep(.1)
        code = child.wait()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    args.output.write_text(json.dumps({'exit_code': code, 'watchdog_reason': reason,
        'seconds': time.monotonic()-start, 'peak_process_rss_gib': peak/2**30}, indent=2)+'\n')
    raise SystemExit(code if code >= 0 else 128-code)

if __name__ == '__main__':
    main()
