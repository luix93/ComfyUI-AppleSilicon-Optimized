"""Run the user's SeedVR2 node sequence with bounded diagnostic memory."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--frames',type=int,default=9)
p.add_argument('--video-sync',choices=('none','block','conv','chunk','production'),default='none')
p.add_argument('--seed',type=int,default=959948902156062)
p.add_argument('--model',choices=('int8','fp16'),default='int8')
p.add_argument('--fix', action='store_true')
p.add_argument('--mode', choices=('stock','native','native-base'), default='native')
p.add_argument('--stage', choices=('encode', 'full', 'decode'), default='full')
p.add_argument('--input-size', type=int, default=416)
p.add_argument('--repeats', type=int, default=1)
p.add_argument('--limit-gib', type=float, default=80)
p.add_argument('--sample-file')
p.add_argument('--installed', action='store_true')
p.add_argument('--default-fix', action='store_true')
p.add_argument('--core-video', action='store_true')
p.add_argument('--export-video', action='store_true')
a = p.parse_args()
if os.environ.get('SEED_UNSAFE_SMALL') and (a.mode!='stock' or a.input_size>320 or a.limit_gib>24):
    raise ValueError('Unprotected scheduling probe is limited to stock, 320px input, and 24 GiB.')
root = Path(__file__).resolve().parent
core = Path('/Users/luigis/ComfyUI-Installs/ComfyUI-AppleSilicon-2.14/ComfyUI')
repo = next((parent for parent in root.parents if (parent / '_patches/seedvr2_fast.py').is_file()), root.parents[1] / 'ComfyUI-AppleSilicon-Optimized')
if a.installed:
    repo = core / 'custom_nodes/ComfyUI-AppleSilicon-Optimized'
sys.path[:0] = [str(core), str(repo)]
os.environ.update(ASFP8_NATIVE='1', ASFP8_AUTOFIX='0',
    ASFP8_SEEDVR2_VAE_SYNC='1' if a.fix else '0', PYTORCH_ENABLE_MPS_FALLBACK='1')
os.environ['ASFP8_SEEDVR2_VAE_VIDEO_SYNC']='1' if a.video_sync=='production' else '0'
if a.default_fix:
    os.environ.pop('ASFP8_SEEDVR2_VAE_SYNC',None)
os.environ.update(ASFP8_ATTN_BACKEND='pytorch', MTLFLASHATTN_SDPA='off', PYTORCH_MPS_FAST_MATH='0')
if a.mode=='native-base': os.environ['ASFP8_SEEDVR2_FAST']='0'
import numpy as np
import psutil
from PIL import Image
import torch
with torch.inference_mode():
    initial_mps_tensor=torch.zeros(1,device='mps')
    torch.mps.synchronize()
    del initial_mps_tensor
import comfy.sd
import comfy.utils
import folder_paths
import nodes
from comfy_extras.nodes_post_processing import ResizeImageMaskNode
from comfy_extras.nodes_seedvr import SeedVR2Preprocess, SeedVR2Conditioning, SeedVR2PostProcessing

if a.mode != 'stock':
    spec = importlib.util.spec_from_file_location('seedvr_214_node', repo / '__init__.py', submodule_search_locations=[str(repo)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    if os.environ.get('SEED_CANDIDATE')=='1':
        from candidate import install
        install(sys.modules[spec.name+'._patches.seedvr2_fast'])
    if os.environ.get('SEED_RESIDUAL')=='1':
        from candidate import install_residual
        install_residual(sys.modules[spec.name+'._patches.seedvr2_fast'])
    elif os.environ.get('SEED_RESIDUAL')=='0':
        from candidate import restore_residual
        restore_residual(sys.modules[spec.name+'._patches.seedvr2_fast'])
else:
    # Scheduling-only safety guard; stock arithmetic and backend stay untouched.
    from contextvars import ContextVar
    import comfy.ldm.seedvr.vae as sv
    active=ContextVar('bench_seed_tile',default=None)
    original_tiled=sv.tiled_vae
    def tiled(x, model, *args, **kwargs):
        token=active.set(model)
        try:return original_tiled(x,model,*args,**kwargs)
        finally:active.reset(token)
    def tile_wrap(fn):
        def call(self,x,*args,**kwargs):
            out=fn(self,x,*args,**kwargs)
            if active.get() is self and x.device.type=='mps' and not os.environ.get('SEED_UNSAFE_SMALL'):
                torch.mps.synchronize()
            return out
        return call
    sv.tiled_vae=tiled
    sv.VideoAutoencoderKLWrapper.encode=tile_wrap(sv.VideoAutoencoderKLWrapper.encode)
    sv.VideoAutoencoderKL.decode_=tile_wrap(sv.VideoAutoencoderKL.decode_)
if os.environ.get('SEED_MLP_CHUNK'):
    import comfy.ldm.seedvr.model as seed_model
    seed_model.SEEDVR2_7B_MLP_CHUNK=int(os.environ['SEED_MLP_CHUNK'])
folder_paths.add_model_folder_path('diffusion_models', '/Volumes/OWC NVMe/comfyui/models/diffusion_models')
folder_paths.add_model_folder_path('vae', '/Volumes/OWC NVMe/comfyui/models/vae')

state = dict(stage='init', max_driver_gib=0, max_live_gib=0, max_rss_gib=0)
events = []
records = []
done = threading.Event()
start = time.perf_counter()
def memory():
    return dict(driver_gib=torch.mps.driver_allocated_memory()/2**30,
                live_gib=torch.mps.current_allocated_memory()/2**30,
                rss_gib=psutil.Process().memory_info().rss/2**30)
def save_report():
    (root / (a.label + '.json')).write_text(json.dumps(dict(config=vars(a),state=state,records=records,events=events),indent=2))
def monitor():
    while not done.wait(.1):
        # Keep Metal calls on the execution thread; telemetry must not race
        # first-use command-buffer initialization.
        m = dict(rss_gib=psutil.Process().memory_info().rss/2**30)
        state['max_rss_gib'] = max(state['max_rss_gib'],m['rss_gib'])
        if m['rss_gib'] > a.limit_gib or psutil.virtual_memory().available < 64*2**30:
            state['stopped']='memory watchdog'
            save_report()
            print('WATCHDOG STOP',json.dumps(m),flush=True)
            os._exit(86)
with torch.inference_mode():
    warmup_tensor=torch.zeros(1,device='mps')
    torch.mps.synchronize()
    del warmup_tensor
threading.Thread(target=monitor,daemon=True).start()
def snapshot():
    m=memory()
    for k,v in m.items():state['max_'+k]=max(state['max_'+k],v)
    elapsed=time.perf_counter()-start
    events.append(dict(seconds=elapsed,stage=state['stage'],**m))
    if elapsed-state.get('last_progress',0)>20:
        state['last_progress']=elapsed
        print('PROGRESS',state['stage'],round(elapsed,1),json.dumps(m),flush=True)
    if max(m.values())>a.limit_gib:
        state['stopped']='sampled Metal memory cutoff';save_report()
        print('CONTROLLED STOP',json.dumps(m),flush=True)
        raise SystemExit(86)
    return m
def measured(stage, fn):
    state['stage']=stage
    torch.mps.synchronize()
    t=time.perf_counter()
    out=fn()
    torch.mps.synchronize()
    sec=time.perf_counter()-t
    print('STAGE',stage,sec,json.dumps(snapshot()),flush=True)
    return out,sec
def digest(t):
    t=t.cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

import comfy.ldm.seedvr.vae as sv
from functools import wraps
interval=int(os.environ.get('SEED_SYNC_EVERY','1'))
if interval>1:
    guard=sys.modules[spec.name+'._patches.seedvr2_vae_mps']
    counts={}
    def bounded_tile(fn):
        def call(self,x,*args,**kwargs):
            out=fn(self,x,*args,**kwargs)
            if guard._tiled_model.get() is self and x.device.type=='mps':
                counts[id(self)]=counts.get(id(self),0)+1
                if counts[id(self)]%interval==0:torch.mps.synchronize()
            return out
        return call
    sv.VideoAutoencoderKLWrapper.encode=bounded_tile(sv.VideoAutoencoderKLWrapper.encode.__wrapped__)
    sv.VideoAutoencoderKL.decode_=bounded_tile(sv.VideoAutoencoderKL.decode_.__wrapped__)
    previous_tiled=sv.tiled_vae
    def bounded_tiled(*args,**kwargs):
        try:return previous_tiled(*args,**kwargs)
        finally:
            torch.mps.synchronize()
            counts.clear()
    sv.tiled_vae=bounded_tiled
def monitored_tile(fn):
    @wraps(fn)
    def call(*args,**kwargs):
        out=fn(*args,**kwargs)
        state['tile_calls']+=1
        snapshot()
        return out
    return call
sv.VideoAutoencoderKLWrapper.encode=monitored_tile(sv.VideoAutoencoderKLWrapper.encode)
sv.VideoAutoencoderKL.decode_=monitored_tile(sv.VideoAutoencoderKL.decode_)
state['video_sync']=a.video_sync
state['tile_calls']=0
state['conv_calls']=0
import comfy.ldm.seedvr.model as seed_model
def monitored_block(fn):
    @wraps(fn)
    def call(*args,**kwargs):
        out=fn(*args,**kwargs)
        snapshot()
        return out
    return call
seed_model.NaMMSRTransformerBlock.forward=monitored_block(seed_model.NaMMSRTransformerBlock.forward)
def inside_wrap(fn,kind):
    @wraps(fn)
    def call(self,x,*args,**kwargs):
        out=fn(self,x,*args,**kwargs)
        if x.device.type=='mps':
            state['conv_calls']+=kind=='conv'
            if kind==a.video_sync and x.ndim==5 and x.shape[2]>1:
                torch.mps.synchronize()
            snapshot()
        return out
    return call
sv.InflatedCausalConv3d.forward=inside_wrap(sv.InflatedCausalConv3d.forward,'conv')
sv.ResnetBlock3D.forward=inside_wrap(sv.ResnetBlock3D.forward,'block')
if a.video_sync=='chunk':
    def chunk_wrap(fn):
        @wraps(fn)
        def call(self,x,*args,**kwargs):
            out=fn(self,x,*args,**kwargs)
            if x.device.type=='mps':
                torch.mps.synchronize()
                snapshot()
            return out
        return call
    sv.VideoAutoencoderKL._encode=chunk_wrap(sv.VideoAutoencoderKL._encode)
    sv.VideoAutoencoderKL._decode=chunk_wrap(sv.VideoAutoencoderKL._decode)
state['torch_version']=torch.__version__
state['torch_git']=torch.version.git_version

try:
    with torch.inference_mode():
        import av
        audio=None;fps=24;bit_depth=8
        if a.core_video:
            from comfy_api.input_impl import VideoFromFile
            from comfy_extras.nodes_video import GetVideoComponents
            source=VideoFromFile('/Users/luigis/ComfyUI-Shared/output/video/MiniMax_H3_00024_.mp4')
            source=source.as_trimmed(0,a.frames/24,strict_duration=False)
            pixels,audio,fps,bit_depth,_=GetVideoComponents.execute(source).result
            assert len(pixels)==a.frames,(len(pixels),a.frames)
        else:
            frame_arrays=[]
            with av.open('/Users/luigis/ComfyUI-Shared/output/video/MiniMax_H3_00024_.mp4') as container:
                for index,frame in enumerate(container.decode(video=0)):
                    if index>=a.frames:break
                    arr=frame.to_ndarray(format='rgb24')
                    if a.input_size!=416:
                        arr=np.asarray(Image.fromarray(arr).resize((a.input_size,round(736*a.input_size/416)),Image.Resampling.LANCZOS))
                    frame_arrays.append(arr)
            pixels=torch.from_numpy(np.stack(frame_arrays).astype(np.float32)/255.0)
            del frame_arrays
        vae=None;model=None
        for repeat in range(a.repeats):
            timings={};run_start=time.perf_counter()
            resized,timings['resize']=measured('resize',lambda:ResizeImageMaskNode.execute(
                pixels,'lanczos',
                dict(resize_type='scale by multiplier',multiplier=2))[0])
            prepared,timings['preprocess']=measured('preprocess',lambda:SeedVR2Preprocess.execute(resized)[0])
            if vae is None:
                vae,timings['load_vae']=measured('load_vae',lambda:nodes.VAELoader().load_vae('seedvr2_ema_vae_fp16.safetensors')[0])
            if a.stage=='decode':
                sampled={'samples':torch.load(a.sample_file,map_location='cpu',weights_only=True)}
            else:
                encoded,timings['encode']=measured('encode',lambda:nodes.VAEEncodeTiled().encode(vae,prepared,512,128,64,8)[0])
                if a.stage=='full':
                    if model is None:
                        model,timings['load_model']=measured('load_model',lambda:nodes.UNETLoader().load_unet('SeedVR2/'+('seedvr2_7b_int8_convrot.safetensors' if a.model=='int8' else 'seedvr2_ema_7b_fp16.safetensors'),'default')[0])
                    pos,neg=SeedVR2Conditioning.execute(model,encoded).result
                    sampled,timings['sample']=measured('sample',lambda:nodes.common_ksampler(model,a.seed,1,1,'euler','simple',pos,neg,encoded,denoise=1)[0])
            if a.stage!='encode':
                decoded,timings['decode']=measured('decode',lambda:nodes.VAEDecodeTiled().decode(vae,sampled,512,128,64,8)[0])
                output,timings['postprocess']=measured('postprocess',lambda:SeedVR2PostProcessing.execute(decoded,resized,'none')[0])
                def save_image():
                    im=Image.fromarray(np.clip(output[0].cpu().numpy()*255,0,255).astype(np.uint8))
                    im.save(root/f'{a.label}-{repeat}.png')
                _,timings['save_image']=measured('save_image',save_image)
                if a.export_video:
                    def export():
                        from comfy_extras.nodes_video import CreateVideo, Types
                        video=CreateVideo.execute(output,fps,audio,bit_depth,'sRGB','none')[0]
                        video.save_to(str(root/f'{a.label}-{repeat}.mp4'),
                                      format=Types.VideoContainer('mp4'),codec=Types.VideoCodec('auto'))
                    _,timings['export_video']=measured('export_video',export)
            total=time.perf_counter()-run_start
            state['stage']='verification'
            record=dict(repeat=repeat,timings=timings,total_seconds=total)
            if a.stage!='decode':
                record['encoded_sha256']=digest(encoded['samples'])
                torch.save(encoded['samples'].cpu(),root/f'{a.label}-{repeat}-encoded.pt')
            if a.stage!='encode':
                record['sampled_sha256']=digest(sampled['samples'])
                record['frames_sha256']=digest(output)
                record['frames_finite']=bool(torch.isfinite(output).all())
                record['output_shape']=list(output.shape)
                torch.save(sampled['samples'].cpu(),root/f'{a.label}-{repeat}-sampled.pt')
                if a.frames<=17:torch.save(output.cpu(),root/f'{a.label}-{repeat}-frames.pt')
                assert record['frames_finite']
            records.append(record);save_report()
            print('RESULT',json.dumps(record),flush=True)
        state['completed']=True
finally:
    done.set()
    save_report()
