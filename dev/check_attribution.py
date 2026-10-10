#!/usr/bin/env python3
"""Check release notice hashes and bundled-source provenance; no network/GPU work."""
import hashlib
import json
from pathlib import Path


def check(root):
    index=json.loads((root/'third_party/provenance.json').read_text())
    for rel,expected in index['notice_sha256'].items():
        path=root/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Missing or changed release notice: {rel}')
    vendor=json.loads((root/index['vendor_manifest']).read_text())
    count=0
    for name,entry in vendor.items():
        for rel,expected in entry['files'].items():
            path=root/'_patches/_vendor'/name/rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Changed pinned vendor reference: {name}/{rel}')
            count+=1
    origins=json.loads((root/index['runtime_origins']).read_text())
    for rel,origin in origins['derived_files'].items():
        if not (root/'_patches/h3_runtime'/rel).is_file():
            raise ValueError(f'Missing derived runtime file: {rel}')
        expected=vendor[origin['package']]['files'][origin['source']]
        if origin['source_sha256']!=expected:
            raise ValueError(f'Runtime origin differs from pinned source: {rel}')
    print(f"Validated {len(index['notice_sha256'])} notices, {count} pinned vendor files, "
          f"and {len(origins['derived_files'])} runtime origins.")


if __name__=='__main__':
    try:
        check(Path(__file__).resolve().parents[1])
    except (ValueError,KeyError,OSError) as exc:
        raise SystemExit(str(exc))
