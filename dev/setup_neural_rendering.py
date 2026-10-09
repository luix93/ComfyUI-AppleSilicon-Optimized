#!/usr/bin/env python3
"""Install the optional NR model from the checksum-pinned DLSSMac release."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile


RELEASE = 'https://github.com/Mappsnet7/DLSSMac/releases/download/v0.1.6/DLSSMac-0.1.6-macos.zip'
SHA256 = 'cab2b8abb4c415d888903bad7edc8f88955c44c7279aea5a33bfde6114f24d62'
PACKAGE = 'NeuralRendering.dlssmodel'
MEMBER_ROOT = 'DLSSMac.app/Contents/Resources/models/'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def install(archive, models_dir):
    if digest(archive) != SHA256:
        raise ValueError('DLSSMac archive checksum mismatch')
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / PACKAGE
    with tempfile.TemporaryDirectory(prefix='.nr-install-', dir=models_dir) as folder:
        staged = Path(folder) / PACKAGE
        staged.mkdir()
        with zipfile.ZipFile(archive) as source:
            for name in ['manifest.json', 'weights.safetensors']:
                with source.open(MEMBER_ROOT + PACKAGE + '/' + name) as src, (staged / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        manifest = json.loads((staged / 'manifest.json').read_text())
        if (manifest['architecture'] != 'mlxdlss.neural-rendering-transformer.v1'
                or manifest['weights']['file'] != 'weights.safetensors'
                or manifest['weights']['sha256'] != digest(staged / 'weights.safetensors')):
            raise ValueError('Invalid neural-rendering model package')
        (staged / 'SOURCE.txt').write_text(
            'DLSSMac v0.1.6 / ' + RELEASE + '\nArchive SHA256: ' + SHA256 + '\n'
            'Converted NVIDIA neural-rendering weights. Original ownership applies;\n'
            'these weights are not licensed by the custom node source-code license.\n')
        if destination.exists():
            for name in ['manifest.json', 'weights.safetensors']:
                if digest(destination / name) != digest(staged / name):
                    raise FileExistsError(f'Refusing to overwrite different weights: {destination}')
        else:
            staged.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', required=True, type=Path,
                        help='Your configured models/applesilicon_neural_rendering directory')
    parser.add_argument('--archive', type=Path, help='Reuse the DLSSMac v0.1.6 Mac ZIP')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='as-nr-download-') as folder:
        archive = args.archive
        if archive is None:
            archive = Path(folder) / 'DLSSMac.zip'
            print('Downloading optional neural-rendering model package (305 MiB)...', flush=True)
            urllib.request.urlretrieve(RELEASE, archive)
        print('Installed:', install(archive, args.models_dir.expanduser().resolve()))


if __name__ == '__main__':
    main()
