"""Patch #31 — VAE slice_attention: respect MPSGraph's INT_MAX tensor limit.

comfy's VAE spatial attention (normal_attention -> slice_attention in
diffusionmodules/model.py — NOT the optimized_attention stack, so neither the
attention router nor flash kernels see it) sizes its query slices from
get_free_memory ALONE. MPSGraph additionally refuses any tensor over INT_MAX
(~2.1e9) elements, a structural limit comfy never models. At Wan-VAE 1024²
video decode the score tensor is [~21 frames, 16384, 16384] ≈ 5.6e9 elements:
memory says "one slice fits" (it does — ~11 GB on a 128 GB box), MPSGraph
raises, and the retry loop re-raises because the error is not an OOM.

Smaller machines dodge this by accident: a pessimistic free-memory figure
forces more slices. Honest accounting (patch #30) exposes it — the more RAM
you have, the more likely stock comfy picks the crashing slice count.

Fix, numerically free (softmax is per-query-row over the full key length, so
the result is identical at ANY slice count):
  * floor the step count so every score slice stays under a safe element
    budget (1<<30, half of INT_MAX — same headroom as patch #26's conv2d
    budget);
  * treat the "tensor dims larger than INT_MAX" RuntimeError like an OOM in
    the retry loop (slice harder) instead of re-raising.

Everything else is byte-for-byte the stock implementation, including the
memory-based step estimate, so behavior is unchanged wherever the floor
doesn't bind. Kill switch: ASFP8_VAE_SLICE_FIX=0.
"""

import logging
import math
import os
import sys

import torch

TAG = "[AppleSilicon-Optimized/vae_slice_attn]"

_installed = False

# Per-slice score-tensor element budget: half of INT_MAX, matching the
# conv3d patch's headroom against MPSGraph's limit.
_MAX_SCORE_ELEMS = 1 << 30


def _is_dim_limit_error(e):
    return isinstance(e, RuntimeError) and "INT_MAX" in str(e)


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_VAE_SLICE_FIX", "1") != "1":
        return  # kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.modules.diffusionmodules.model as M
        import comfy.model_management as MM
    except Exception as e:
        print(f"{TAG} comfy VAE attention module unavailable ({e!r}); not installed.")
        return

    def slice_attention(q, k, v):
        # Stock body (torch213-era comfy) with two additions marked NEW.
        r1 = torch.zeros_like(k, device=q.device)
        scale = (int(q.shape[-1]) ** (-0.5))

        mem_free_total = MM.get_free_memory(q.device)

        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

        # NEW: MPSGraph structural floor — each score slice
        # [B, Lq/steps, Lk] must stay under the element budget no matter how
        # much RAM is free.
        forced = False
        if q.device.type == "mps":
            score_elems = int(q.shape[0]) * int(q.shape[1]) * int(k.shape[2])
            if score_elems > _MAX_SCORE_ELEMS:
                min_steps = -(-score_elems // _MAX_SCORE_ELEMS)  # ceil
                if min_steps > steps:
                    steps = 2 ** math.ceil(math.log2(min_steps))
                    forced = True

        while True:
            try:
                if forced and q.shape[1] % steps != 0:
                    # Stock's non-divisible fallback is "one full-width slice",
                    # which is exactly the crashing shape when the floor is
                    # binding — use a ceil-sized slice instead (tensor slicing
                    # clamps the ragged tail; per-row math is unchanged).
                    slice_size = -(-q.shape[1] // steps)
                else:
                    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
                for i in range(0, q.shape[1], slice_size):
                    end = i + slice_size
                    s1 = torch.bmm(q[:, i:end], k) * scale

                    s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
                    del s1

                    r1[:, :, i:end] = torch.bmm(v, s2)
                    del s2
                break
            except Exception as e:
                # NEW: the MPSGraph INT_MAX refusal is retryable exactly like
                # an OOM — slice harder. Everything else keeps stock behavior
                # (raise_non_oom re-raises non-OOM errors).
                if not _is_dim_limit_error(e):
                    MM.raise_non_oom(e)
                MM.soft_empty_cache(True)
                steps *= 2
                if steps > 128:
                    raise e
                forced = True
                logging.warning("out of memory error, increasing steps and trying again {}".format(steps))

        return r1

    M.slice_attention = slice_attention
    _installed = True
    print(f"{TAG} VAE slice_attention capped to MPSGraph's INT_MAX tensor "
          f"limit (fixes Wan-VAE 1024p video decode on large-RAM machines).")
