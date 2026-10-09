import argparse,copy,hashlib,json,os,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
p=argparse.ArgumentParser();p.add_argument('--config',choices=['stock','optimized'],required=True);p.add_argument('--candidate');p.add_argument('--no-lora',action='store_true');p.add_argument('--env',action='append',default=[]);p.add_argument('--model',choices=['sdxl','zimage','krea2','h3'],required=True);p.add_argument('--frames',type=int,default=22);p.add_argument('--steps',type=int);p.add_argument('--modes',default='baseline,baseline,baseline,baseline');p.add_argument('--warmups',type=int,default=1);p.add_argument('--label');p.add_argument('--warmup-steps',type=int);p.add_argument('--krea-glue',action='store_true');p.add_argument('--seed',type=int,default=812764035);p.add_argument('--prompt-text');a=p.parse_args()
label=a.label or f'{a.model}-{a.config}'+(f'-{a.frames}f' if a.model=='h3' else '')
def request(path,data=None):
    req=urllib.request.Request('http://127.0.0.1:8193'+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
    except urllib.error.HTTPError as e:raise RuntimeError(e.read().decode()) from e
try:
    with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=2) as r:q=json.load(r)
    assert not q['queue_running'] and not q['queue_pending'],'User server busy'
except urllib.error.URLError:pass
repo=WORK/'ComfyUI-AppleSilicon-Optimized'
base=ROOT/('base-'+a.config);custom=base/'custom_nodes';custom.mkdir(parents=True,exist_ok=True)
node=repo
links=[('z_matched',ROOT/'bench_node')]
if a.config=='optimized':links.insert(0,('a_optimized',node))
for name,target in links:
    link=custom/name
    if not link.exists():link.symlink_to(target,target_is_directory=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('ASFP8_','MTLFLASHATTN_'))}
env.pop('PYTHONPATH',None)
env.update(PYTORCH_ENABLE_MPS_FALLBACK='1',PYTHONUNBUFFERED='1',PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.4',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.3',BENCH_LABEL=label,BENCH_CONFIG=a.config,MTLFLASHATTN_SDPA='off')
if a.config=='optimized':env.update(ASFP8_NATIVE='1',ASFP8_AUTOFIX='0',ASFP8_DISABLE='ideogram_fast',ASFP8_MINIMAX_RELAXED='0',ASFP8_ATTN_BACKEND='pytorch')
if a.candidate:
    snapshot=ROOT/(label+'-candidate.py');snapshot.write_bytes(Path(a.candidate).read_bytes());env['BENCH_CANDIDATE']=str(snapshot)
for item in a.env:
    k,v=item.split('=',1);env[k]=v
if a.model=='sdxl':graph=json.loads((WORK/'diagnostics/sdxl-ultra/prompt.json').read_text())
else:graph=json.loads((repo/'benchmarks/data/m5-ultra-torch214-workflows.json').read_text())[a.model]
if a.model=='h3':
    graph['140:131']['inputs']['length']=a.frames
    graph['140:129']['inputs']['noise_seed']=a.seed
    if a.prompt_text:graph['140:131']['inputs']['prompt']=a.prompt_text
if a.no_lora:
    assert a.model=='krea2'
    for n in graph.values():
        if 'strength_model' in n['inputs'] and 'lora_name' in n['inputs']:n['inputs']['strength_model']=0.0
for n in graph.values():
    if n['class_type'] in ('KSampler','SamplerCustomAdvanced'):
        n['class_type']='MatchSampler' if n['class_type']=='KSampler' else 'MatchAdvanced'
        if a.steps and n['class_type']=='MatchSampler':n['inputs']['steps']=a.steps
    if n['class_type']=='VAEDecode':n['class_type']='MatchVAE'
if a.steps and a.model=='h3':graph['140:124']['inputs']['steps']=a.steps
if os.environ.get('BENCH_UPSCALE_DECODE')=='1':
    graph=json.loads((repo/'workflows/h3_latent_upscale.api.json').read_text())
    graph['140:125']={'class_type':'SavedH3Latent','inputs':dict(mode='fp16',run_id=label)}
    graph['140:122']['class_type']='MatchVAE'
(ROOT/(label+'-workflow.json')).write_text(json.dumps(graph,indent=2))
cmd=[sys.executable,str(Path('/Users/luigis/ComfyUI-Installs/ComfyUI-AppleSilicon-2.14/ComfyUI/main.py') if os.environ.get('BENCH_NORMAL_LAUNCH')=='1' else ROOT/'launch.py'),'--base-directory',str(base),'--models-directory','/Volumes/OWC NVMe/comfyui/models','--output-directory',str(ROOT/'output'),'--port','8193','--preview-method','none','--cache-classic','--disable-metadata','--use-pytorch-cross-attention']
log=(ROOT/(label+'-server.log')).open('w');child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);proc=psutil.Process(child.pid)
peak=0;begin=time.monotonic();records=[]
def check():
    global peak
    if child.poll() is not None:raise RuntimeError(f'Server exit {child.returncode}; see {label}-server.log')
    rss=proc.memory_info().rss;peak=max(peak,rss)
    if rss>110*2**30:raise RuntimeError('110 GiB RSS watchdog')
    if psutil.virtual_memory().available<48*2**30:raise RuntimeError('48 GiB available watchdog')
    if time.monotonic()-begin>3600:raise TimeoutError('1 hour watchdog')
try:
    for attempt in range(600):
        check()
        try:request('/queue');break
        except urllib.error.URLError:time.sleep(1)
    else:raise TimeoutError('Server startup')
    for i,mode in enumerate(a.modes.split(',')):
        check();run_id=f'{label}-{i}-{mode}';g=copy.deepcopy(graph)
        if i<a.warmups and a.warmup_steps and a.model=='h3':g['140:124']['inputs']['steps']=a.warmup_steps
        for n in g.values():
            if n['class_type'] in ('MatchSampler','MatchAdvanced','SavedH3Latent'):n['inputs'].update(mode=mode,run_id=run_id)
            if n['class_type']=='AppleSiliconH3LatentUpscale':n['inputs']['precision']=mode
            if n['class_type'] in ('SaveImage','SaveVideo'):n['inputs']['filename_prefix']=run_id
        submitted=request('/prompt',{'prompt':g});assert not submitted.get('node_errors'),submitted
        print('START',run_id,'warmup',i<a.warmups,flush=True)
        while True:
            check();entry=request('/history/'+submitted['prompt_id']).get(submitted['prompt_id'])
            if not entry:time.sleep(.3);continue
            (ROOT/(run_id+'-history.json')).write_text(json.dumps(entry,indent=2))
            assert entry['status']['status_str']=='success', [(k,{f:v.get(f) for f in ['node_id','exception_message','exception_type','traceback']}) for k,v in entry['status']['messages'] if k=='execution_error']
            times={k:v['timestamp'] for k,v in entry['status']['messages'] if 'timestamp' in v}
            record=request('/matched/collect',{'run_id':run_id,'save':True})
            assert all(t['finite'] for group in record['tensors'].values() for t in group.values()), 'Non-finite output'
            assert all(v!='True' for group in record.get('patch_status',{}).values() for k,v in group.items() if k in ['_attention_dead','_ff_dead','_na_dead','_qkv_dead']), 'Latched kernel fallback'
            record.update(e2e=(times['execution_success']-times['execution_start'])/1000,warmup=i<a.warmups,outputs=entry['outputs'])
            records.append(record);(ROOT/(label+'-results.json')).write_text(json.dumps(records,indent=2))
            print('DONE',run_id,'e2e',record['e2e'],'timings',record['timings'],flush=True);break
finally:
    child.terminate()
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    log.close();(ROOT/(label+'-guard.json')).write_text(json.dumps({'peak_rss_gib':peak/2**30,'seconds':time.monotonic()-begin,'child_exit':child.returncode},indent=2))
