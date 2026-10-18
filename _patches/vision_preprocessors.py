"""Bounded DA3 scheduling, DA3 camera dtype compatibility and SDPose feature-only execution.

DA3 loop adapted from ComfyUI nodes_depth_anything_3.py (GPL-3.0).
The SDPose wrapper stops only after the last feature consumed by its heatmap head;
normal image-generation samplers and other U-Nets are untouched. Source guards
retain stock behavior when the host changes. No resolution/precision is reduced. DA3 batching is experimental and opt-in:
changing the batch shape can change floating-point rounding.
"""
import hashlib
import logging
import os
from pathlib import Path

import torch

_TAG = '[AppleSilicon-Optimized/preprocessors]'
_installed = False
_NODE_HASH = '82f898e2c7e289eff9d228698d1a1344d959eab4a2dbab69d2a065b3233d382a'
_UNET_HASH = '9d27fb036cab8a262ef3d866a643f7fdc40994022616f1b8be14b7d919f57f96'
_DA3_HASH = 'adfce28637b6904a08596aa23e22502d20089bc28fff6bcdaabe0b3c35fb7f02'
_CAMERA_HASH = 'b9c1bc79862c8f2b59a6058da1bf47c1aaef84ca75ec5131805fbdf2f81dca9a'


def _matches(module, expected):
    try:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
    except (OSError, AttributeError, TypeError):
        return False


class _FeatureReady(Exception):
    """Local early return; never escapes the model-function wrapper."""


def _feature_only(model_function, inputs):
    # The original SDPose capture hook runs first, preserving its clone exactly.
    cond = dict(inputs['c'])
    options = dict(cond.get('transformer_options', {}))
    patches = dict(options.get('patches', {}))
    hooks = list(patches.get('output_block_patch', []))
    if not hooks:
        return model_function(inputs['input'], inputs['timestep'], **cond)

    def finish(h, skip, opts):
        if opts.get('block') == ('output', 9) and h.shape[1] == 640:
            raise _FeatureReady()
        return h, skip

    patches['output_block_patch'] = hooks + [finish]
    options['patches'] = patches
    cond['transformer_options'] = options
    try:
        return model_function(inputs['input'], inputs['timestep'], **cond)
    except _FeatureReady:
        # SDPose discards the sampler result; keep normal sampler cleanup intact.
        return torch.zeros_like(inputs['input'])


def _pose_execute(original, model, vae, image, batch_size, bboxes=None):
    unet = getattr(getattr(model, 'model', None), 'diffusion_model', None)
    if (getattr(getattr(model, 'load_device', None), 'type', None) != 'mps'
            or not hasattr(unet, 'heatmap_head')
            or len(getattr(unet, 'output_blocks', ())) != 12
            or model.model_options.get('model_function_wrapper') is not None
            or os.environ.get('ASFP8_SDPOSE_FEATURE_ONLY', '1') == '0'):
        return original(model, vae, image, batch_size, bboxes)
    clone = model.clone()
    clone.set_model_unet_function_wrapper(_feature_only)
    return original(clone, vae, image, batch_size, bboxes)


def _batch_size():
    try:
        return max(1, min(2, int(os.environ.get('ASFP8_DA3_BATCH', '1'))))
    except ValueError:
        return 1


def _da3_run(original, module, model_patcher, image, process_res, method='upper_bound_resize'):
    diff = model_patcher.model.diffusion_model
    # Only independent FP16 mono-depth inference is eligible for optional batching.
    device = module.mm.get_torch_device()
    size = _batch_size()
    target = module.da3_preprocess.compute_target_size(*image.shape[1:3], process_res, method)
    if (device.type != 'mps' or diff.dtype != torch.float16
            or getattr(diff, 'head_type', None) != 'dpt'
            or getattr(diff, 'cam_dec', None) is not None
            or size == 1 or len(image) < 2 or target[0] * target[1] > 512 * 1024):
        return original(model_patcher, image, process_res, method)
    module.mm.load_model_gpu(model_patcher)
    # Do not add a second frame's activations under memory pressure.
    if module.mm.get_free_memory(device) < 4 * 1024**3:
        return original(model_patcher, image, process_res, method)
    _, height, width, _ = image.shape
    depths, confs, skies = [], [], []
    for start in range(0, len(image), size):
        module.mm.throw_exception_if_processing_interrupted()
        single = image[start:start + size].to(device)
        x = module.da3_preprocess.preprocess_image(single, process_res=process_res, method=method)
        with torch.no_grad():
            out = diff(x.to(dtype=diff.dtype))
        for key, values in (('depth', depths), ('depth_conf', confs), ('sky', skies)):
            if key in out:
                values.append(torch.nn.functional.interpolate(
                    out[key].unsqueeze(1).float(), size=(height, width),
                    mode='bilinear', align_corners=False).squeeze(1).cpu())
        del out, x, single
    return torch.cat(depths), torch.cat(confs) if confs else None, torch.cat(skies) if skies else None


def _camera_forward(original, self, feat, *args, **kwargs):
    if feat.device.type != 'mps':
        return original(self, feat, *args, **kwargs)
    # CameraDec intentionally evaluates these heads with FP32 features. Comfy's
    # fast non-casting Linear otherwise sends FP32 x FP16 into Metal and aborts.
    heads = (self.fc_t, self.fc_qvec, self.fc_fov[0])
    if not all(hasattr(head, 'comfy_cast_weights') for head in heads):
        return original(self, feat, *args, **kwargs)
    saved = [head.comfy_cast_weights for head in heads]
    try:
        for head in heads:
            head.comfy_cast_weights = True
        return original(self, feat, *args, **kwargs)
    finally:
        for head, previous in zip(heads, saved):
            head.comfy_cast_weights = previous


def install():
    global _installed
    if _installed or os.environ.get('ASFP8_PREPROCESSORS', '1') == '0' or not torch.backends.mps.is_available():
        return
    from comfy_extras import nodes_sdpose as pose, nodes_depth_anything_3 as da3
    from comfy.ldm.modules.diffusionmodules import openaimodel
    from comfy.ldm.depth_anything_3 import camera
    installed = []
    if _matches(pose, _NODE_HASH) and _matches(openaimodel, _UNET_HASH):
        original_pose = pose.SDPoseKeypointExtractor.execute
        def execute(cls, model, vae, image, batch_size, bboxes=None):
            return _pose_execute(original_pose, model, vae, image, batch_size, bboxes)
        pose.SDPoseKeypointExtractor.execute = classmethod(execute)
        installed.append('SDPose feature-only execution')
    if _matches(da3, _DA3_HASH):
        original_run = da3._run_da3
        def run(model_patcher, image, process_res, method='upper_bound_resize'):
            return _da3_run(original_run, da3, model_patcher, image, process_res, method)
        da3._run_da3 = run
        installed.append('bounded DA3 mono batches')
    if _matches(camera, _CAMERA_HASH):
        original_camera = camera.CameraDec.forward
        def forward(self, feat, *args, **kwargs):
            return _camera_forward(original_camera, self, feat, *args, **kwargs)
        camera.CameraDec.forward = forward
        installed.append('DA3 FP32 camera-head compatibility')
    _installed = True
    logging.info('%s enabled: %s; unmatched host sources retain stock paths.', _TAG, ', '.join(installed) or 'none')
