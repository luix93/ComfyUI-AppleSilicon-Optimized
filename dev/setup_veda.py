#!/usr/bin/env python3
"""Download the pinned, hash-verified optional H3 Veda predictor."""

import argparse
import hashlib
import os
from pathlib import Path
import ssl
import tempfile
import urllib.request

NAME = 'minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors'
REVISION = '9a1fd3a41b4a754a7886e64e82edbddf599fd1bd'
SHA256 = '2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a'
URL = f'https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/resolve/{REVISION}/{NAME}'


def checksum(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', required=True, type=Path,
                        help='ComfyUI model directory; creates its veda subdirectory')
    args = parser.parse_args()
    target = args.models_dir.expanduser().resolve() / 'veda' / NAME
    if target.exists():
        if checksum(target) != SHA256:
            parser.error(f'{target} exists with a different checksum; left untouched')
        print(f'Already verified: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.download', delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(URL, context=context, timeout=120) as source:
                while chunk := source.read(4 * 2**20):
                    output.write(chunk)
        if checksum(temporary) != SHA256:
            raise RuntimeError('Veda predictor checksum mismatch; refusing installation')
        if target.exists():
            raise RuntimeError(f'{target} appeared during download; left untouched')
        os.replace(temporary, target)
        print(f'Installed: {target}')
        print('Predictor license: MiniMax Community License (see its Hugging Face model card).')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
