"""Diagnostic-only matched attention benchmark. Never installed in the live app."""
import collections, hashlib, json, os, platform, sys, time
from pathlib import Path
import torch
import nodes
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from aiohttp import web
from server import PromptServer
ROOT=Path(__file__).resolve().parent.parent
MODE='ours'; STAGE='setup'; CURRENT=None; PENDING={}; COUNTS=collections.Counter(); INSTALLED=False; CANDIDATE=None

def tensors(v,p=''):
    if isinstance(v,torch.Tensor):yield p,v
    elif hasattr(v,'tensors'):yield from tensors(v.tensors,p)
    elif isinstance(v,dict):
        for k,x in v.items():yield from tensors(x,p+'/'+str(k))
    elif isinstance(v,(list,tuple)):
        for i,x in enumerate(v):yield from tensors(x,p+'/'+str(i))
def sync():torch.mps.synchronize()
def count(route,q,k):COUNTS[json.dumps([STAGE,route,str(q.dtype),list(q.shape),list(k.shape)])]+=1

def install():
    global INSTALLED,CANDIDATE
    if INSTALLED:return
    patches={n.rsplit('.',1)[-1]:m for n,m in list(sys.modules.items()) if '._patches.' in n}
    if os.environ['BENCH_CONFIG'] == 'stock':
        assert not patches, 'Custom optimizations leaked into stock'
    else:
        assert patches['attn_router_mps']._installed
        assert patches['int8_linear_kernel_mps']._kernel is not None
        assert patches['lumina_attn_fast']._kernel is not None
    candidate=os.environ.get('BENCH_CANDIDATE')
    if candidate:
        import importlib.util
        spec=importlib.util.spec_from_file_location('candidate',candidate)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.install();CANDIDATE=mod
    def timed_class(cls,method,label,classmethod_=False,capture=False):
        old=getattr(cls,method)
        def wrapped(*a,**kw):
            global STAGE
            STAGE=label;sync();begin=time.perf_counter();out=old(*a,**kw);sync()
            if CURRENT is not None:
                CURRENT['timings'][label]=time.perf_counter()-begin
                if capture:PENDING[label]=out.result if hasattr(out,'result') else out
            STAGE='other';return out
        if classmethod_:
            setattr(cls,method,classmethod(lambda cls,*a,**kw:wrapped(*a,**kw)))
        else:setattr(cls,method,wrapped)
    from comfy_extras.nodes_audio import VAEDecodeAudio
    from comfy_extras.nodes_video import CreateVideo,SaveVideo
    timed_class(VAEDecodeAudio,'execute','audio_decode',True,True)
    timed_class(CreateVideo,'execute','create_video',True)
    timed_class(SaveVideo,'execute','save_video',True)
    timed_class(nodes.SaveImage,'save_images','save_image')
    metadata={'torch':torch.__version__,'torch_git':torch.version.git_version,'python':sys.version,'platform':platform.platform(),'torch_num_threads':torch.get_num_threads(),
      'node_root':str(Path(patches['attn_router_mps'].__file__).parents[1]) if patches else None,'env':{k:v for k,v in os.environ.items() if k.startswith(('ASFP8','PYTORCH','OMP'))},
      'patches':{k:{f:str(getattr(m,f)) for f in ['_installed','_attention_dead','_ff_dead','_na_dead'] if hasattr(m,f)} for k,m in patches.items()}}
    import subprocess
    core=Path(nodes.__file__).parent
    metadata['comfy_commit']=subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip()
    metadata['torch_path']=torch.__file__
    if metadata['node_root']:
        nr=Path(metadata['node_root']);metadata['node_source_sha256']={str(p.relative_to(nr)):hashlib.sha256(p.read_bytes()).hexdigest() for p in nr.rglob('*') if p.is_file() and p.suffix in ('.py','.mm','.metal') and '__pycache__' not in p.parts}
    (ROOT/(os.environ['BENCH_LABEL']+'-runtime.json')).write_text(json.dumps(metadata,indent=2))
    INSTALLED=True

def begin(mode,run_id,run,conditioning):
    global MODE,STAGE,CURRENT,PENDING
    install();MODE=mode;STAGE='sampling';COUNTS.clear();PENDING={}
    if CANDIDATE is not None and hasattr(CANDIDATE,'set_mode'):CANDIDATE.set_mode(mode)
    CURRENT={'run_id':run_id,'mode':mode,'timings':{},'steps':[]}
    PENDING['conditioning']=conditioning
    import comfy.samplers
    original=comfy.samplers.KSAMPLER.sample
    def sample(self,model_wrap,sigmas,extra_args,callback,noise,*a,**kw):
        PENDING['noise']={k:v.detach().cpu().clone() for k,v in tensors(noise)}
        sync();start=previous=time.perf_counter()
        def cb(step,denoised,x,total):
            nonlocal previous
            sync();now=time.perf_counter();CURRENT['steps'].append(now-previous);previous=now
            CURRENT['peak_driver_gib']=max(CURRENT.get('peak_driver_gib',0),torch.mps.driver_allocated_memory()/2**30)
            if callback:callback(step,denoised,x,total)
        out=original(self,model_wrap,sigmas,extra_args,cb,noise,*a,**kw);sync()
        CURRENT['timings']['sampling_loop']=time.perf_counter()-start
        return out
    comfy.samplers.KSAMPLER.sample=sample
    sync();start=time.perf_counter()
    try:out=run();sync()
    finally:comfy.samplers.KSAMPLER.sample=original
    CURRENT['timings']['sampling']=time.perf_counter()-start
    PENDING['latents']=out
    STAGE='other';print('BENCH SAMPLE',run_id,CURRENT['timings'],flush=True)
    return out

def extra():return {'mode':('STRING',{'default':'ours'}),'run_id':('STRING',{'default':'bench'})}
class BenchSampler(nodes.KSampler):
    @classmethod
    def INPUT_TYPES(cls):
        x=super().INPUT_TYPES();x['required'].update(extra());return x
    def sample(self,mode,run_id,**kw):return begin(mode,run_id,lambda:nodes.KSampler.sample(self,**kw),kw['positive'])
class BenchAdvanced:
    @classmethod
    def INPUT_TYPES(cls):return {'required':dict(noise=('NOISE',),guider=('GUIDER',),sampler=('SAMPLER',),sigmas=('SIGMAS',),latent_image=('LATENT',),**extra())}
    RETURN_TYPES=('LATENT','LATENT');FUNCTION='sample';CATEGORY='diagnostics'
    def sample(self,mode,run_id,**kw):
        install()
        if CANDIDATE is not None:kw['guider']=CANDIDATE.configure(kw['guider'],mode)
        return begin(mode,run_id,lambda:CANDIDATE.sample(mode, **kw) if CANDIDATE is not None and hasattr(CANDIDATE, 'sample') else SamplerCustomAdvanced.execute(**kw).result,kw['guider'].original_conds)
class BenchVAE(nodes.VAEDecode):
    def decode(self,vae,samples):
        global STAGE
        STAGE='video_decode';sync();start=time.perf_counter();out=super().decode(vae,samples);sync()
        CURRENT['timings']['video_decode']=time.perf_counter()-start;PENDING['frames']=out[0]
        CURRENT['vae_dtype']=str(vae.vae_dtype);STAGE='other';return out
NODE_CLASS_MAPPINGS={'MatchSampler':BenchSampler,'MatchAdvanced':BenchAdvanced,'MatchVAE':BenchVAE}

@PromptServer.instance.routes.post('/matched/collect')
async def collect(request):
    global MODE,STAGE
    data=await request.json();record=dict(CURRENT);assert record['run_id']==data['run_id']
    record['attention_calls']=dict(COUNTS);record['tensors']={}
    if CANDIDATE is not None and hasattr(CANDIDATE,'stats'):record['candidate_stats']=dict(CANDIDATE.stats)
    record['patch_status']={n.rsplit('.',1)[-1]:{f:str(getattr(m,f)) for f in ['_installed','_attention_dead','_ff_dead','_na_dead','_dense','_qkv_dead','_block_dead','_dead','_calls'] if hasattr(m,f)} for n,m in list(sys.modules.items()) if '._patches.' in n}
    for kind,value in PENDING.items():
        cpu={k:v.detach().cpu().contiguous() for k,v in tensors(value)}
        record['tensors'][kind]={k:{'sha256':hashlib.sha256(memoryview(v.view(torch.uint8).numpy())).hexdigest(),'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(torch.isfinite(v).all())} for k,v in cpu.items()}
        if data.get('save') or kind in ('noise',):torch.save(cpu,ROOT/(data['run_id']+'-'+kind+'.pt'))
    PENDING.clear();MODE='ours';STAGE='setup'
    (ROOT/(data['run_id']+'.json')).write_text(json.dumps(record,indent=2));return web.json_response(record)

# V3 nodes are cloned during prompt validation, before the sampler executes.
# Time their dispatch rather than mutating the already-cloned class methods.
import execution
_old_dispatch=execution.get_output_data
async def timed_dispatch(prompt_id,unique_id,obj,input_data_all,**kw):
    global STAGE
    name=getattr(obj,'__name__',type(obj).__name__)
    label={'VAEDecodeAudio':'audio_decode','CreateVideo':'create_video','SaveVideo':'save_video','AppleSiliconH3LatentUpscale':'latent_upscale'}.get(name)
    if not label:return await _old_dispatch(prompt_id,unique_id,obj,input_data_all,**kw)
    STAGE=label;sync();start=time.perf_counter()
    out=await _old_dispatch(prompt_id,unique_id,obj,input_data_all,**kw);sync()
    if CURRENT is not None:
        CURRENT['timings'][label]=time.perf_counter()-start
        if label in ('audio_decode','latent_upscale'):PENDING[label]=out[0]
    STAGE='other';return out
execution.get_output_data=timed_dispatch

class SavedH3Latent:
    @classmethod
    def INPUT_TYPES(cls):return {'required':extra()}
    RETURN_TYPES=('LATENT',);FUNCTION='load';CATEGORY='diagnostics'
    def load(self,mode,run_id):
        global CURRENT,PENDING,MODE,STAGE
        install();CURRENT={'run_id':run_id,'mode':mode,'timings':{},'steps':[]};PENDING={};MODE=mode;STAGE='other'
        from comfy.nested_tensor import NestedTensor
        raw=torch.load(ROOT/'short-safe-4-exact-latents.pt',weights_only=True,map_location='cpu')
        return ({'samples':NestedTensor([raw['/0/samples/0'],raw['/0/samples/1']])},)
NODE_CLASS_MAPPINGS['SavedH3Latent']=SavedH3Latent
