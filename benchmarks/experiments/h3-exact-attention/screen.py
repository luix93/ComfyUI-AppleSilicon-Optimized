"""Isolated M5 Ultra/Torch 2.14.1 attention screen. No production installation.

Provide a torch.save((q.cpu(), k.cpu(), v.cpu()), path) capture from an authored
public fixture. Never point this at untrusted pickle files. weights_only=True
is used. Run through guard.py, which also enforces the MPS allocation watermark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import torch
from kernels import run, source, qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('variants', nargs='*', default=['qk16_persistent', 'vectorload', 'rescale_late', 'sync8'])
    args = parser.parse_args()
    if not qc._hardware() or torch.__version__.split('+')[0] != '2.14.1' or not qc._torch_build_supported():
        parser.error('These experiments target the validated M5 Ultra / PyTorch 2.14.1 build only.')
    if not 1 <= args.rounds <= 100:
        parser.error('--rounds must be between 1 and 100')
    names = list(dict.fromkeys(['baseline'] + args.variants))
    supported = ['directqk', 'directqk_static', 'inner2', 'inner2_noq', 'inner4', 'inner4_noq', 'inner4_uniform_noq', 'look2', 'look4', 'm32col2', 'm32col_split2', 'm32row2', 'm32row_noq2', 'm32row_split2', 'm32row_split4', 'm32row_split_noq1', 'm32row_split_noq2', 'm32row_split_noq4', 'maxthreads256', 'maxthreads512', 'output_ct', 'prefetchk', 'prefetchv1', 'prefetchv2', 'prefetchv4', 'pv_f32', 'pv_persistent', 'qk128', 'qk16_persistent', 'qk64', 'rescale_late', 'strides', 'sync0', 'sync16', 'sync2', 'sync4', 'sync8', 'tail_split', 'uniform', 'unroll2', 'unroll4', 'vectorload', 'warps3', 'warps5', 'warps6']
    if any(name not in supported and name != "baseline" for name in names):
        parser.error("Unknown candidate name; see the recorded benchmark JSON for supported names.")
    rows = {}
    cpu = torch.load(args.capture, map_location='cpu', weights_only=True)
    if (not isinstance(cpu, (tuple, list)) or len(cpu) != 3
        or any(not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16
               or t.ndim != 4 or t.shape[0] != 1 or t.shape[1] != 56
               or t.shape[3] != 128 or not 2048 <= t.shape[2] <= 65536
               or not t.is_contiguous() for t in cpu)
        or not cpu[0].shape == cpu[1].shape == cpu[2].shape):
        parser.error('Expected three contiguous BF16 [1,56,L,128] tensors, L=2048..65536.')
    with torch.inference_mode():
        q, k, v = [t.to('mps') for t in cpu]
        del cpu
        ref = run(q, k, v, 'baseline').cpu()
        ref_hash = hashlib.sha256(ref.view(torch.uint8).numpy().tobytes()).hexdigest()
        for name in names:
            try:
                out = run(q, k, v, name).cpu()
                rows[name] = {'byte_equal': torch.equal(ref.view(torch.uint8), out.view(torch.uint8)),
                              'samples_ms': [], 'source_sha256': hashlib.sha256(source(name).encode()).hexdigest()}
                del out
            except Exception as exc:
                rows[name] = {'error': str(exc)[:2000]}
            print(name, rows[name], flush=True)
        del ref
        rng = random.Random(23)
        for i in range(args.rounds):
            order = [n for n, r in rows.items() if r.get('byte_equal')]
            rng.shuffle(order)
            for name in order:
                torch.mps.synchronize()
                start = time.perf_counter()
                out = run(q, k, v, name)
                torch.mps.synchronize()
                rows[name]['samples_ms'].append((time.perf_counter() - start) * 1000)
                del out
            print('round', i, flush=True)
        for row in rows.values():
            if row.get('samples_ms'):
                row['median_ms'] = statistics.median(row['samples_ms'])
        args.output.write_text(json.dumps({'shape': list(q.shape), 'reference_sha256': ref_hash,
            'results': rows, 'driver_gib_at_end': torch.mps.driver_allocated_memory()/2**30}, indent=2)+'\n')

if __name__ == '__main__':
    main()
