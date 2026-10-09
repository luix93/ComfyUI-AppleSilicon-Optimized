"""Diagnostic: coalesce adjacent equal-length window layout copies."""
import inspect

def install(fast):
    source=inspect.getsource(fast._bounded_attention)
    old='''            q4 = torch.stack([q[qs[i]:qs[i+1]].transpose(0, 1) for i in subset])
            k4 = torch.stack([k[ks[i]:ks[i+1]].transpose(0, 1) for i in subset])
            v4 = torch.stack([v[ks[i]:ks[i+1]].transpose(0, 1) for i in subset])'''
    new='''            consecutive = subset[-1] - subset[0] + 1 == len(subset)
            if consecutive:
                def pack(x, offsets, length):
                    return x[offsets[subset[0]]:offsets[subset[-1]+1]].reshape(len(subset), length, heads, dim).transpose(1,2).contiguous()
                q4, k4, v4 = pack(q, qs, lq), pack(k, ks, lk), pack(v, ks, lk)
            else:
                q4 = torch.stack([q[qs[i]:qs[i+1]].transpose(0, 1) for i in subset])
                k4 = torch.stack([k[ks[i]:ks[i+1]].transpose(0, 1) for i in subset])
                v4 = torch.stack([v[ks[i]:ks[i+1]].transpose(0, 1) for i in subset])'''
    assert old in source
    source=source.replace(old,new)
    old='''            for j, i in enumerate(subset):
                out[qs[i]:qs[i+1]].copy_(result[j].transpose(0, 1))'''
    new='''            if consecutive:
                out[qs[subset[0]]:qs[subset[-1]+1]].copy_(result.transpose(1,2).reshape(-1,heads,dim))
            else:
                for j, i in enumerate(subset):
                    out[qs[i]:qs[i+1]].copy_(result[j].transpose(0, 1))'''
    assert old in source
    exec(source.replace(old,new),fast.__dict__)
    import comfy.ldm.seedvr.attention as attention
    import comfy.ldm.seedvr.model as model
    attention.optimized_var_attention=model.optimized_var_attention=fast._bounded_attention

def install_residual(fast):
    source=inspect.getsource(fast._resnet)
    old='    return (input_tensor + hidden) / self.output_scale_factor'
    if old not in source:
        return  # Already applied in the current production source.
    source=source.replace(old,'''    residual = input_tensor + hidden
    return residual if self.output_scale_factor == 1.0 else residual / self.output_scale_factor''')
    exec(source,fast.__dict__)
    import comfy.ldm.seedvr.vae as vae
    vae.ResnetBlock3D.forward=fast._resnet

def restore_residual(fast):
    source=inspect.getsource(fast._resnet)
    old='''    residual = input_tensor + hidden
    # SeedVR2's image VAE uses unit scaling: avoid a redundant full-tensor pass.
    return residual if self.output_scale_factor == 1.0 else residual / self.output_scale_factor'''
    assert old in source
    exec(source.replace(old,'    return (input_tensor + hidden) / self.output_scale_factor'),fast.__dict__)
    import comfy.ldm.seedvr.vae as vae
    vae.ResnetBlock3D.forward=fast._resnet
