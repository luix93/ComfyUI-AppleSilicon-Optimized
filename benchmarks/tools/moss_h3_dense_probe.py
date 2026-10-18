"""Diagnostic-only H3 dense backends at MOSS's 24 heads; no SOL routing.

Reuses the pack's CCV-derived kernels and their existing attribution/license.
This is a version-sensitive experiment, not a supported attention dispatcher.
Inputs are matching MPS BF16 [1, 24, sequence, 128] tensors. Both paths retain
all keys, approximate arithmetic, and include layout/quantization costs.
"""
import importlib,re,types
from pathlib import Path

def load(package='_patches'):
 dense=importlib.import_module(package+'.h3_dense_mps')
 original=importlib.import_module(package+'.h3_floatpv_mps')
 path=Path(original.__file__)
 shader=(path.parent/'h3_floatpv_shaders/int8_qk_float_pv.metal').read_text()
 for declaration in ('constant uint Hq = 56;', 'constant uint Hk = 56;', 'tgid.y >= 56'):
  if shader.count(declaration)!=1:
   raise RuntimeError(f'Floating-PV shader changed: expected one {declaration!r}')
 source=path.read_text()
 # Specialize a separate module. Never mutate H3's module, caches or shader.
 source=re.sub(r'\b56\b','24',source)
 source=re.sub(r'\b112\b','48',source)
 needle=".read_text()\n    values ="
 if source.count(needle)!=1:
  raise RuntimeError('Floating-PV wrapper changed: inspect before adapting')
 source=source.replace(needle,".read_text().replace('constant uint Hq = 56;', 'constant uint Hq = 24;').replace('constant uint Hk = 56;', 'constant uint Hk = 24;').replace('tgid.y >= 56', 'tgid.y >= 24')\n    values =")
 clone=types.ModuleType(package+'._moss_floatpv_diagnostic')
 clone.__file__=str(path);clone.__package__=package
 exec(compile(source,str(path)+' (MOSS 24-head diagnostic)','exec'),clone.__dict__)
 def int8(q,k,v):return dense.attention(q,k,v,tau=0,video_start=0,exact_prefix=False)[0]
 def floatpv(q,k,v):return clone.attention(q,k,v,video_start=0,exact_prefix=False)[0]
 return {'int8':int8,'floatpv':floatpv}
