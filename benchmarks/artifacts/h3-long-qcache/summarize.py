from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parent
records=json.loads((root/'h3-15s-two-step-results.json').read_text())
assert len(records)==6
reference=records[1]['tensors']
assert all(r['tensors']==reference for r in records),'Full output byte mismatch'
result={'frames':362,'width':864,'height':480,'steps':2,'tokens':44600,'all_outputs_exact':True,'compared_runs':len(records),'modes':{}}
for mode in ('baseline','extended'):
 runs=[r for r in records if r['mode']==mode and not r['warmup']]
 assert len(runs)==2
 result['modes'][mode]={'e2e_seconds':statistics.mean(r['e2e'] for r in runs),'sampling_seconds':statistics.mean(r['timings']['sampling'] for r in runs),'step_seconds':statistics.mean(r['timings']['sampling_loop']/2 for r in runs),'step_samples':[v for r in runs for v in r['steps']],'runs':[r['run_id'] for r in runs]}
 for r in runs:
  key=f"56 heads / 44600 tokens / {'cached' if mode=='extended' else 'fallback'}"
  assert r['candidate_stats'][key]==100
result['sampling_reduction_percent']=100*(1-result['modes']['extended']['step_seconds']/result['modes']['baseline']['step_seconds'])
result['max_sampled_driver_gib']=max(r['max_sampled_driver_gib'] for r in records)
result['twenty_step_sampling_saving_estimate']=20*(result['modes']['baseline']['step_seconds']-result['modes']['extended']['step_seconds'])
(root/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
