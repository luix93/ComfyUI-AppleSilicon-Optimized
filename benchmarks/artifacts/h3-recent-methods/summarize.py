import json,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]/'ComfyUI-AppleSilicon-Optimized'
rows=json.loads((D/'recent-five-results.json').read_text())
assert len(rows)==12, 'Full campaign still running'
reference=[r for r in rows if r['mode']=='exact' and not r['warmup']]
assert len(reference)==2
assert reference[0]['tensors']==reference[1]['tensors'], 'Exact state leaked'
base=statistics.mean(r['e2e'] for r in reference)
summary={}
for mode in ['exact','spectrum','spectrum-safe','motion0.15','speed3-cal01']:
    matches=[r for r in rows if r['mode']==mode and not r['warmup']]
    assert len(matches)==2
    same=matches[0]['tensors']==matches[1]['tensors']
    assert same, mode+' not self-repeatable'
    times=[r['e2e'] for r in matches];mean=statistics.mean(times)
    summary[mode]={'e2e':times,'mean_e2e':mean,'mean_sampling':statistics.mean(r['timings']['sampling'] for r in matches),'less_time_percent':(1-mean/base)*100,'repeats_byte_identical':same,'actual_evaluations':[r['candidate_stats']['block_0'] for r in matches],'sampled_peak_metal_gib':max(r.get('peak_driver_gib',0) for r in matches)}
quality={}
for case in ['recent-five-3-spectrum','recent-five-7-motion0.15','recent-five-9-speed3-cal01','recent-texture-2-ui-spectrum-protected','recent-motion-2-ui-spectrum-protected']:
    p=D/(case+'-quality.json')
    if p.exists():quality.update(json.loads(p.read_text()))
data={'date':'2026-09-29','hardware':'M5 Ultra 80 GPU / 256 GB','reference':'existing optimized exact path, torch 2.14, res_multistep/simple, 20 steps, 864x480, 124 frames, no LoRA','summary':summary,'quality':quality,'calibration':json.loads((D/'harvest-latest.json').read_text()),'guard':json.loads((D/'recent-five-guard.json').read_text()),'runs':rows}
(R/'benchmarks/data/m5-ultra-h3-recent-methods.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(summary,indent=2))
