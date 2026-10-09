"""Sequential, independently guarded screens; never run beside another GPU job."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
jobs = [
    ('full9-core-before', '--frames 9 --video-sync none --limit-gib 52 --core-video --export-video'),
    ('full25-conv', '--frames 25 --video-sync conv --limit-gib 48 --repeats 2'),
    ('encode120-before', '--frames 120 --video-sync none --limit-gib 48 --stage encode --core-video'),
    ('full25-stock', '--frames 25 --video-sync block --limit-gib 64 --mode stock --repeats 2'),
]
for label, arguments in jobs:
    command = [sys.executable, str(root / 'guard.py'), label, sys.executable,
               str(root / 'run.py'), '--label', label, '--fix', *arguments.split()]
    code = subprocess.call(command)
    print(json.dumps({'label': label, 'exit_code': code}), flush=True)
    if code not in (0, 86):
        sys.exit(code)
