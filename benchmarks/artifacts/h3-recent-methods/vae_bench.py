import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

D = Path(__file__).resolve().parent
R = D.parents[1] / 'ComfyUI-AppleSilicon-Optimized'
C = Path('/Users/luigis/ComfyUI-Installs/ComfyUI-AppleSilicon-2.14/ComfyUI')
sys.path.insert(0, str(C))
os.environ.update(ASFP8_NATIVE='1', ASFP8_AUTOFIX='0', ASFP8_ATTN_BACKEND='pytorch',
                  ASFP8_DISABLE='ideogram_fast', ASFP8_MINIMAX_RELAXED='0',
                  PYTORCH_ENABLE_MPS_FALLBACK='1', PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.3',
                  PYTORCH_MPS_LOW_WATERMARK_RATIO='0.2')
import torch
torch.set_num_threads(2)
with torch.inference_mode():
    torch.zeros(1, device='mps')
    torch.mps.synchronize()
import comfy.sd
import comfy.utils


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


load_module('as_vae_benchmark', R / '__init__.py', True)
fast = load_module('fast_vae_benchmark', D / 'upstream/motion-fastvae/fast_vae_decode.py')
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
    '/Volumes/OWC NVMe/comfyui/models/vae/minimax_h3_video_vae_fp16.safetensors'),
    device=torch.device('mps'), dtype=torch.float16)
records = []
with torch.inference_mode():
    for length, run in [(22, 'recent-screen-1-exact'),
                        (124, '../h3-sol-aggressive/five-window-1-exact')]:
        latent = torch.load(D / (run + '-latents.pt'), map_location='cpu', weights_only=True)['/0/samples/0']
        reference = None
        for index, batch in enumerate([0, 0, 1, 2, 4, 8, 2, 4, 0]):
            torch.mps.synchronize()
            start = time.perf_counter()
            if batch == 0:
                out = vae.decode(latent)
                if out.ndim == 5:
                    out = out.reshape(-1, *out.shape[-3:])
            else:
                out = fast.MiniMaxH3FastVAEDecode.execute({'samples':latent}, vae, batch).result[0]
            torch.mps.synchronize()
            seconds = time.perf_counter() - start
            driver = torch.mps.driver_allocated_memory()/2**30
            cpu = out.detach().cpu().contiguous()
            if reference is None:
                reference = cpu.clone()
            delta = cpu.float() - reference.float()
            row = dict(frames=length, batch=batch, warmup=index==0, seconds=seconds,
                       driver_gib=driver, shape=list(cpu.shape), finite=bool(torch.isfinite(cpu).all()),
                       equal=torch.equal(cpu.view(torch.uint8), reference.view(torch.uint8)),
                       max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt()),
                       sha256=hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest())
            assert row['finite']
            records.append(row)
            (D/'vae-results.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(row), flush=True)
            del out, cpu, delta

