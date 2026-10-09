import copy,importlib.util,sys
from pathlib import Path
ROOT=Path('/Users/luigis/Documents/ChatGPT/ComfyUI-AppleSilicon/diagnostics/h3-recent-methods')
stats={}
settings={}
def install():pass
def set_mode(mode):
 stats.clear();stats['settings']=dict(settings)
def configure(guider,mode):
 global settings
 unified=next(m for n,m in sys.modules.items() if n.endswith('._patches.minimax_h3_unified'))
 schema=unified.AppleSiliconH3Acceleration.INPUT_TYPES()
 settings={k:v[1]['default'] for group in schema.values() for k,v in group.items() if k!='model'}
 settings.update(cache_mode='disabled',token_reduction=False,sparse_attention=False,sparse_backend='int8',sparse_tau=0.5,sparse_start_percent=0.2,sparse_end_percent=0.9,sparse_min_sequence_tokens=0,sparse_dense_layers='0,1,33,34,35',sparse_window_mode='step_fraction')
 patched=guider.model_patcher
 if mode=='motion-upstream':
  spec=importlib.util.spec_from_file_location('v2_upstream_motion',ROOT/'upstream/motioncache/motion_cache.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  patched=m.MiniMaxH3MotionCache.execute(patched,reuse_threshold=.15,motion_strength=1.,warmup_steps=4,max_consecutive_skips=2,start_percent=.15,end_percent=.95,subsample_factor=8,verbose=True).result[0]
 elif mode=='spectrum-upstream':
  sys.path.insert(0,str(ROOT/'upstream/spectrum'))
  from comfyui_spectrum_h3.nodes import SpectrumApplyMiniMaxH3
  patched=SpectrumApplyMiniMaxH3().apply(model=patched,enabled=True,blend_weight=.5,degree=1,ridge_lambda=.1,window_size=2.,flex_window=.75,warmup_steps=4,tail_actual_steps=2,max_history=8,debug=True,history_storage='vram',bootstrap_first_forecast=False,anchor_residual_feedback=False,selective_rollback_correction=False,offline_smoothing_replay=True,audio_blend_weight=0.,offline_archive_storage='vram',model_aware_mode='off',model_aware_trust_shrinkage=False,model_aware_replay_generic_correction=False)[0]
 else:
  if mode.startswith('motion'):settings['cache_mode']='motion_cache'
  if mode.startswith('spectrum'):settings['cache_mode']='spectrum_protected'
  settings['sparse_attention']=mode.endswith(('-sol','-all'))
  settings['token_reduction']=mode.endswith(('-token','-all'))
  patched=unified.apply_h3_acceleration(patched,**settings)
 clone=copy.copy(guider);clone.model_patcher=patched;clone.model_options=patched.model_options
 return clone
