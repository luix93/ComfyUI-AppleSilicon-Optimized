"""Collect and export Apple macOS 27 hardware counters for one authored trace.

Run under ../h3-exact-attention/guard.py. No production code imports this.
The built-in gpudebug v1.0 needs a fresh session after profile collection.
"""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trace',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--state',choices=['medium','high'],default='high')
p.add_argument('--load-only',action='store_true')
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
trace=str(a.trace.resolve())
if not a.load_only:
 with (out/'collection.jsonlog').open('w') as log:
  subprocess.run(['gpudebug','--oneshot','--json','-t',trace,'-c',f'profile run --gpu-state {a.state} --exec overlapping --embed'],stdout=log,stderr=subprocess.STDOUT,check=True)
# Counter paths observed on the tested M5 Ultra / macOS 27 debugger.
metrics={
 'active_cores':['active_cores'],
 'occupancy':['kernel_occupancy','total_occupancy'],
 'occupancy_manager':['occupancy_manager_target'],
 'occupancy_target_influences':['register_pressure_influence','l1_cache_pressure_influence','memory_request_stalls_influence'],
 'neural_accelerator':['neural_accelerator_limiter','neural_accelerator_utilization'],
 'instruction_throughput':['instruction_throughput_limiter','instruction_throughput_utilization'],
 'alu':['alu_utilization'],
 'bandwidth':['gpu_bandwidth','gpu_read_bandwidth','gpu_write_bandwidth'],
 'l1_cache':['l1_cache_limiter','l1_cache_utilization'],
 'l1_residency':['l1_total_residency','l1_register_residency','l1_stack_residency'],
 'l1_read_bandwidth':['register_l1_read_bandwidth','stack_l1_read_bandwidth','buffer_l1_read_bandwidth'],
 'last_level_cache':['last_level_cache_limiter','last_level_cache_utilization'],
 'mmu':['mmu_limiter','mmu_utilization'],
}
commands=['profile load','go /performance/timeline','info --all','go /performance/shaders/comp0','info --all']
# Fetch into a space-free relative filename; gpudebug v1.0 treats quotes literally.
for group,names in metrics.items():
 commands+=['go /performance/timeline/counters/'+group]
 for name in names:commands+=['info '+name+' --all','fetch '+name+' --out '+(group+'--'+name+'.json')]
with (out/'profile.jsonlog').open('w') as log:
 subprocess.run(['gpudebug','--oneshot','--json','-t',trace,*[v for c in commands for v in ('-c',c)]],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
log=(out/'profile.jsonlog').read_text()
if 'error:' in log:raise RuntimeError('Profiler command errors; inspect profile.jsonlog. Missing counters are not zeros.')
for group,names in metrics.items():
 for name in names:
  result=json.loads((out/(group+'--'+name+'.json')).read_text())
  if not result.get('samples'):raise RuntimeError('Empty counter data: '+name)
print('Exported counters. Check Consistent state and sampled-core count before interpreting them.')
