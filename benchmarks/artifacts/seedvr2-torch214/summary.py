from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parent
out={}
for p in sorted(root.glob('*.json')):
 d=json.loads(p.read_text())
 if not isinstance(d,dict) or not d.get('records'):continue
 runs=d['records'][1:]
 if not runs:continue
 out[p.stem]=dict(warm_mean_seconds=statistics.mean(r['total_seconds'] for r in runs),timings={k:statistics.mean(r['timings'][k] for r in runs) for k in runs[0]['timings']},state=d['state'],hashes={k:runs[-1][k] for k in ('encoded_sha256','sampled_sha256','frames_sha256') if k in runs[-1]})
print(json.dumps(out,indent=2))
(root/'summary.json').write_text(json.dumps(out,indent=2))
