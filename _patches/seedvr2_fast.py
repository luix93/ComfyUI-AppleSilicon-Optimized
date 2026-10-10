"""SeedVR2 exact scheduling/layout paths; MPS attention views stay bounded."""
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import sys

import torch

TAG = '[AppleSilicon-Optimized/seedvr2_fast]'
_installed = False
_original_attention = None
_kernel = None
_original_rope = None
_original_block = None
_original_resnet = None
_current_vae_fusions = False
_CURRENT_MODEL_SHA = 'f88d32b1295cd1064db48a72b22f619a779ffdd6cee64a942aab6101abfb60f8'
_CURRENT_VAE_SHA = 'e01dd80282d357e183e1c3b732f26747e5caf8f5778cf9eca1f26ab015132532'


def _current_contract_supported(model, vae):
    from . import torch_mpp_qcache as qc
    if (torch.__version__.split('+')[0] != '2.14.1'
            or not qc._torch_build_supported() or not qc._hardware()):
        return False
    try:
        return (hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest() == _CURRENT_MODEL_SHA
                and hashlib.sha256(Path(vae.__file__).read_bytes()).hexdigest() == _CURRENT_VAE_SHA)
    except (AttributeError, OSError, TypeError):
        return False


def _enabled(name):
    # The legacy single-dispatch GN fusion was neutral. The reviewed current
    # split-statistics fusion has its own switch and preserves both store rounds.
    default = '0' if name == 'VAE_GN' else '1'
    return os.environ.get('ASFP8_SEEDVR2_' + name, default) == '1'


def _bf16(x):
    return (isinstance(x, torch.Tensor) and x.device.type == 'mps'
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and 0 < x.numel() < 2**32 and not torch.is_grad_enabled())


def _rope(self, q, k, shape, cache):
    if (_kernel is None or not _enabled('ROPE') or not _bf16(q)
            or not _bf16(k) or q.dim() != 3 or q.shape != k.shape):
        return _original_rope(self, q, k, shape, cache)
    freqs = cache('rope_freqs_3d', lambda: self.get_freqs(shape)).to(q.device)
    matrix = freqs.ndim == 4 and freqs.shape[-2:] == (2, 2)
    if freqs.shape[0] != q.shape[0] or freqs.shape[-1] > q.shape[-1]:
        return _original_rope(self, q, k, shape, cache)
    if matrix:
        if freqs.shape[-3] * 2 > q.shape[-1]:
            return _original_rope(self, q, k, shape, cache)
        cs, sn = cache('seedfast_trig', lambda: (
            freqs[..., 0, 0].repeat_interleave(2, -1).contiguous(),
            freqs[..., 1, 0].repeat_interleave(2, -1).contiguous()))
    else:
        cs, sn = cache('seedfast_trig', lambda: (freqs.float().cos().contiguous(),
                                              freqs.float().sin().contiguous()))
    if q.numel() >= 2**30:
        oq, ok = torch.empty_like(q), torch.empty_like(k)
        for start in range(0, q.shape[0], 65536):
            end = min(start + 65536, q.shape[0])
            aq, ak = _kernel.rope(q[start:end], k[start:end], cs[start:end], sn[start:end])
            oq[start:end].copy_(aq); ok[start:end].copy_(ak)
    else:
        oq, ok = _kernel.rope(q, k, cs, sn)
    if matrix:
        # Current core rotates in place; preserve both values and ownership.
        q.copy_(oq); k.copy_(ok)
        return q, k
    return oq, ok


def _branch(mm, branch):
    return mm.all if mm.shared_weights else getattr(mm, branch)


def _vectors(ada, emb, layer, x):
    import comfy.ops
    index = ada.layers.index(layer)
    table = emb.reshape(emb.shape[0], -1, len(ada.layers), 3)[0, :, index]
    def parameter(suffix):
        return comfy.ops.cast_to_input(getattr(ada, layer + '_' + suffix), x)
    return ((table[:, 1] + parameter('scale')).contiguous(),
            (table[:, 0] + parameter('shift')).contiguous(),
            (table[:, 2] + parameter('gate')).contiguous())


def _block(self, vid, txt, vid_shape, txt_shape, emb, cache):
    if (_kernel is None or not _enabled('MOD') or not self.version
            or not _bf16(vid) or not _bf16(txt) or vid.dim() != 2
            or txt.dim() != 2 or emb.shape[0] != 1 or vid_shape.shape[0] != 1
            or txt_shape.shape[0] != 1 or emb.dtype != torch.bfloat16
            or vid.shape[-1] != 3072 or txt.shape[-1] != 3072):
        return _original_block(self, vid, txt, vid_shape, txt_shape, emb, cache)
    norms = [_branch(getattr(self, name), branch)
             for name in ('attn_norm', 'mlp_norm') for branch in ('vid', 'txt')]
    if any(not isinstance(norm, torch.nn.RMSNorm) or norm.weight is not None
           or norm.eps is None or norm._forward_hooks or norm._forward_pre_hooks
           or tuple(norm.normalized_shape) != (3072,) for norm in norms):
        return _original_block(self, vid, txt, vid_shape, txt_shape, emb, cache)
    import comfy.ldm.seedvr.model as model
    if any(type(_branch(self.ada, b)) is not model.AdaSingle for b in ('vid', 'txt')):
        return _original_block(self, vid, txt, vid_shape, txt_shape, emb, cache)
    cache('vid_len', lambda: vid_shape.prod(-1))
    cache('txt_len', lambda: txt_shape.prod(-1))
    gates = {}
    modern = hasattr(self, '_norm_ada_in')
    def rows(fn, x):
        if x.numel() < 2**30:
            return fn(x, 0, x.shape[0])
        out = torch.empty_like(x)
        for start in range(0, x.shape[0], 65536):
            end = min(start + 65536, x.shape[0])
            out[start:end].copy_(fn(x[start:end], start, end))
        return out
    def normalize(x, branch, layer):
        norm = _branch(getattr(self, layer + '_norm'), branch)
        scale, shift, gate = _vectors(_branch(self.ada, branch), emb, layer, x)
        if modern:
            # Current AdaSingle sends scale-1 to Kitchen, which adds 1 again.
            # Both BF16 rounding boundaries are part of the reference arithmetic.
            scale = ((scale - 1.0) + 1.0).contiguous()
        gates[(branch, layer)] = gate
        def apply(part, start, end):
            if _enabled('RMS'):
                # Preserve the unweighted MPS composite reduction and rsqrt.
                square = _kernel.rms_square(part)
                inv = square.mean(dim=-1, keepdim=True).add_(norm.eps).rsqrt_()
                return _kernel.rms_apply(part, inv, scale, shift)
            return _kernel.elementwise(norm(part).contiguous(), scale, shift, 0, norm.eps)
        return rows(apply, x)
    def residual(x, original, branch, layer):
        return rows(lambda part, start, end: _kernel.elementwise(
            part.contiguous(), gates[(branch, layer)],
            original[start:end].contiguous(), 1, 0.), x)
    va, ta = normalize(vid, 'vid', 'attn'), normalize(txt, 'txt', 'attn')
    va, ta = self.attn(va, ta, vid_shape, txt_shape, cache)
    va, ta = residual(va, vid, 'vid', 'attn'), residual(ta, txt, 'txt', 'attn')
    vm, tm = normalize(va, 'vid', 'mlp'), normalize(ta, 'txt', 'mlp')
    vm, tm = self._chunked_mlp(vm, tm) if modern else self._seedvr2_7b_mlp(vm, tm)
    return (residual(vm, va, 'vid', 'mlp'), residual(tm, ta, 'txt', 'mlp'),
            vid_shape, txt_shape)


def _silu_conv(conv, x, state, memory_cache, norm=None):
    from comfy.ldm.seedvr.vae import MemoryState, causal_norm_wrapper, get_norm_limit
    from . import conv3d_mps
    import comfy.ops
    import torch.nn.functional as F
    eligible = (state == MemoryState.DISABLED and conv3d_mps._installed
        and x.ndim == 5 and x.shape[2] == 1 and x.is_contiguous()
        and x.dtype == torch.float16 and x.device.type == 'mps'
        and conv.groups == 1 and not conv.transposed and conv.dilation == (1, 1, 1)
        and conv.stride[0] == 1 and conv.kernel_size[0] in (1, 3)
        and conv.temporal_padding * 2 == conv.kernel_size[0] - 1
        and x.numel()*x.element_size()*conv.kernel_size[0]
            <= conv.memory_limit*1024**3
        and not conv._forward_hooks and not conv._forward_pre_hooks)
    split_norm = (_current_vae_fusions and _enabled('VAE_GN_SPLIT')
        and hasattr(_kernel, 'groupnorm_split_fold'))
    norm_fusion_enabled = (split_norm or (_enabled('VAE_GN')
        and torch.__version__.split('+')[0].startswith('2.13.')))
    fused_norm = (norm is not None and norm_fusion_enabled
        and isinstance(norm, torch.nn.GroupNorm) and norm.affine
        and not norm._forward_hooks and not norm._forward_pre_hooks
        and x.numel()*x.element_size()/1024**3 <= get_norm_limit())
    if norm is not None and not (eligible and fused_norm):
        x = causal_norm_wrapper(norm, x)
    if not eligible:
        return conv(F.silu(x), memory_state=state, memory_cache=memory_cache)
    if memory_cache is not None:
        memory_cache.pop(conv, None)
    comfy.ops.run_every_op()
    with comfy.ops.CastBiasWeightContext(conv, x, offloadable=True) as (weight, bias):
        co, ci, kt, kh, kw = weight.shape
        if fused_norm:
            comfy.ops.run_every_op()
            with comfy.ops.CastBiasWeightContext(norm, x, offloadable=True) as (nw, nb):
                normalize_fold = (_kernel.groupnorm_split_fold if split_norm
                                  else _kernel.groupnorm_fold)
                folded = normalize_fold(x[:, :, 0], nw.contiguous(),
                    nb.contiguous(), norm.num_groups, norm.eps, kt)
        else:
            folded = _kernel.silu_fold(x[:, :, 0], kt)
        packed = conv3d_mps._pack_weight(weight, co, kt, ci, 1, kh, kw)
        out = F.conv2d(folded, packed, bias, conv.stride[1:], conv.padding[1:],
                       conv.dilation[1:], 1)
        return out.unsqueeze(2)


def _resnet(self, input_tensor, temb, memory_state=None, memory_cache=None):
    from comfy.ldm.seedvr.vae import MemoryState
    if (_kernel is None or not _enabled('VAE_SILU') or torch.is_grad_enabled()
            or input_tensor.device.type != 'mps' or input_tensor.dtype != torch.float16
            or input_tensor.ndim != 5 or input_tensor.shape[2] != 1
            or memory_state != MemoryState.DISABLED or temb is not None
            or self.time_emb_proj is not None or type(self.nonlinearity) is not torch.nn.SiLU):
        return _original_resnet(self, input_tensor, temb, memory_state, memory_cache)
    hidden = _silu_conv(self.conv1, input_tensor, memory_state, memory_cache, self.norm1)
    hidden = _silu_conv(self.conv2, hidden, memory_state, memory_cache, self.norm2)
    if self.conv_shortcut is not None:
        input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state,
                                           memory_cache=memory_cache)
    residual = input_tensor + hidden
    # SeedVR2's image VAE uses unit scaling: avoid a redundant full-tensor pass.
    return residual if self.output_scale_factor == 1.0 else residual / self.output_scale_factor


def _resnet_current(self, input_tensor, memory_state, memory_cache):
    from comfy.ldm.seedvr.vae import MemoryState
    if (_kernel is None or not _enabled('VAE_SILU') or torch.is_grad_enabled()
            or input_tensor.device.type != 'mps' or input_tensor.dtype != torch.float16
            or input_tensor.ndim != 5 or input_tensor.shape[2] != 1
            or memory_state != MemoryState.DISABLED):
        return _original_resnet(self, input_tensor, memory_state, memory_cache)
    hidden = _silu_conv(self.conv1, input_tensor, memory_state, memory_cache, self.norm1)
    hidden = _silu_conv(self.conv2, hidden, memory_state, memory_cache, self.norm2)
    if self.conv_shortcut is not None:
        input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state,
                                          memory_cache=memory_cache)
    return hidden.add_(input_tensor)


def _batch_size():
    try:
        default = '8' if _enabled('FAST') and os.environ.get('ASFP8_NATIVE') == '1' else '1'
        return max(1, min(16, int(os.environ.get('ASFP8_SEEDVR2_ATTN_BATCH', default))))
    except ValueError:
        return 1


def _gather_rows(a, indices, b=None):
    large = max(a.numel(), indices.numel() * a[0].numel()) >= 2**31
    if large and _kernel is not None and a.device.type == 'mps' and a.dtype in (torch.float16, torch.bfloat16):
        return _kernel.gather_rows(a, a if b is None else b, indices.contiguous())
    source = a if b is None else torch.cat((a, b))
    return torch.index_select(source, 0, indices)


def _window_idx(shape, window_fn):
    import comfy.ldm.seedvr.model as model
    indices = torch.arange(shape.prod(-1).sum(), device=shape.device).unsqueeze(-1)
    target, target_shape, windows, lengths, counts = model.window(indices, shape, window_fn)
    target = target.squeeze(-1)
    source = torch.argsort(target)
    result = (lambda x: _gather_rows(x, target), lambda x: _gather_rows(x, source),
              target_shape, windows)
    return result + (lengths, counts) if hasattr(model, 'repeat_concat') else result


def _repeat_concat_idx(vid_len, txt_len, txt_repeat):
    import comfy.ldm.seedvr.model as model
    device = vid_len.device
    vi = torch.arange(vid_len.sum(), device=device)
    ti = torch.arange(len(vi), len(vi) + txt_len.sum(), device=device)
    repeats = txt_repeat.tolist()
    legacy = hasattr(model, 'repeat_concat')
    if legacy:
        target = model.repeat_concat(vi, ti, vid_len, txt_len, repeats)
    else:
        video_blocks = vi.split(vid_len.tolist())
        text_blocks = [block for part, repeat in zip(ti.split(txt_len.tolist()), repeats)
                       for block in [part] * repeat]
        lengths = [video.numel() + text.numel() for video, text in zip(video_blocks, text_blocks)]
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        target = torch.cat([part for i in order for part in (video_blocks[i], text_blocks[i])])
    source = torch.argsort(target)
    text_count = len(target) - len(vi)
    text_lengths = (txt_len * txt_repeat).tolist()
    def unconcat(all_tokens):
        video, text = _gather_rows(all_tokens, source).split((len(vi), text_count))
        parts = [part.reshape(-1, repeat, *part.shape[1:]).mean(1)
                 for part, repeat in zip(text.split(text_lengths), repeats)]
        return video, torch.cat(parts)
    result = (lambda video, text: _gather_rows(video, target, text), unconcat)
    return result if legacy else result + (model.cumulative_lengths([lengths[i] for i in order]),)


def _bounded_attention(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, *args,
                       skip_reshape=False, skip_output_reshape=False, **kwargs):
    if not isinstance(q, torch.Tensor):
        from comfy.ldm.modules.attention import AttentionTensorContainer
        if isinstance(q, AttentionTensorContainer):
            q, k, v = q.take(), k.take(), v.take()
    if (q.device.type != 'mps' or q.dtype not in (torch.float16, torch.bfloat16)
            or torch.is_grad_enabled()):
        return _original_attention(q, k, v, heads, cu_seqlens_q, cu_seqlens_k,
            *args, skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape, **kwargs)
    # The small-image screen loses to the original loop. Keep it when the
    # backing buffers are safe; larger images amortize grouping and packing.
    if (q.shape[0] < 16384 and all(
            t.untyped_storage().nbytes() // t.element_size() < 2**31
            for t in (q, k, v))):
        return _original_attention(q, k, v, heads, cu_seqlens_q, cu_seqlens_k,
            *args, skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape, **kwargs)
    from comfy.ldm.seedvr.attention import _var_attention_qkv, _var_attention_output
    from comfy.ldm.modules import attention as attention
    q, k, v, dim = _var_attention_qkv(q, k, v, heads, skip_reshape)
    qs, ks = list(cu_seqlens_q), list(cu_seqlens_k)
    if len(qs) != len(ks) or k.shape[0] != v.shape[0]:
        raise ValueError('SeedVR2 attention sequence lengths do not match')
    qs, ks = [int(i) for i in qs], [int(i) for i in ks]
    if (not qs or qs[0] != 0 or ks[0] != 0
            or qs[-1] != q.shape[0] or ks[-1] != k.shape[0]):
        raise ValueError('SeedVR2 attention sequence endpoints do not match tensors')
    groups = defaultdict(list)
    for i in range(len(qs) - 1):
        groups[(qs[i+1]-qs[i], ks[i+1]-ks[i])].append(i)
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    for (lq, lk), indices in groups.items():
        batch = _batch_size() if 0 < max(lq, lk) <= 512 else 1
        for start in range(0, len(indices), batch):
            subset = indices[start:start+batch]
            # Materialize only the window batch. MPSGraph can reject a small
            # view whose underlying storage spans more than INT_MAX elements.
            q4 = torch.stack([q[qs[i]:qs[i+1]].transpose(0, 1) for i in subset])
            k4 = torch.stack([k[ks[i]:ks[i+1]].transpose(0, 1) for i in subset])
            v4 = torch.stack([v[ks[i]:ks[i+1]].transpose(0, 1) for i in subset])
            container = getattr(attention, 'AttentionTensorContainer', None)
            if container is not None:
                q4, k4, v4 = container(q4), container(k4), container(v4)
            result = attention.optimized_attention(q4, k4, v4, heads,
                skip_reshape=True, skip_output_reshape=True, **kwargs)
            for j, i in enumerate(subset):
                out[qs[i]:qs[i+1]].copy_(result[j].transpose(0, 1))
    return _var_attention_output(out, heads, dim, skip_output_reshape)


def install():
    global _installed, _original_attention, _kernel
    global _original_rope, _original_block, _original_resnet, _current_vae_fusions
    if (_installed or sys.platform != 'darwin' or not torch.backends.mps.is_available()):
        return
    try:
        import comfy.ldm.seedvr.attention as attention
        import comfy.ldm.seedvr.model as model
    except ImportError:
        return
    _original_attention = attention.optimized_var_attention
    if _enabled('ATTN_FIX'):
        attention.optimized_var_attention = _bounded_attention
        model.optimized_var_attention = _bounded_attention
        print(f'{TAG} bounded MPS window attention; batch={_batch_size()}.')
    _installed = True
    if os.environ.get('ASFP8_DIT_EXT') != '1':
        return
    from .seedvr2_ext.loader import module
    _kernel = module()
    if _kernel is None:
        return
    if _enabled('ATTN_FIX'):
        model.window_idx = _window_idx
        model.repeat_concat_idx = _repeat_concat_idx
    if not _enabled('FAST'):
        return
    import comfy.ldm.seedvr.vae as vae
    modern = _current_contract_supported(model, vae)
    _current_vae_fusions = modern
    if hasattr(model, 'repeat_concat') or modern:
        _original_rope = model.NaRotaryEmbedding3d.forward
        _original_block = model.NaMMSRTransformerBlock.forward
        model.NaRotaryEmbedding3d.forward = _rope
        model.NaMMSRTransformerBlock.forward = _block
    if hasattr(vae, 'VideoAutoencoderKL'):
        _original_resnet = vae.ResnetBlock3D.forward
        vae.ResnetBlock3D.forward = _resnet
        print(f'{TAG} SeedVR2 BF16 sampling and FP16 image VAE fusions installed.')
    elif modern:
        _original_resnet = vae.ResnetBlock3D.forward
        vae.ResnetBlock3D.forward = _resnet_current
        print(f'{TAG} current-core exact BF16 sampling and FP16 image VAE fusions installed'
              f"; split GroupNorm/SiLU/folding {'on' if _enabled('VAE_GN_SPLIT') else 'off'}.")
    else:
        # The refactored core has a new block contract; keep its implementation.
        print(f'{TAG} bounded SeedVR2 sampling installed; core sampling/VAE implementation retained.')
