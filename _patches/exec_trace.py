"""Dev tracer — per-node and per-sampler-phase wall timing (opt-in ASFP8_EXEC_TRACE=1).

Attributes where a ComfyUI prompt's wall time goes outside the sampler step
loop: node execution (execution.get_output_data), sampler setup vs inner loop
(CFGGuider.sample / outer_sample / inner_sample), model residency checks
(prepare_sampling / load_models_gpu), cond processing (process_conds), and the
KSAMPLER loop itself. Everything logs to STDERR (ComfyUI swallows stdout during
execution).

Diagnostic tool, not a perf patch: adds a few time.perf_counter() calls per
run. Not implied by ASFP8_NATIVE.
"""

import functools
import os
import sys
import time

TAG = "[AppleSilicon-Optimized/exec_trace]"

_installed = False


def _log(depth, name, ms, extra=""):
    sys.stderr.write(f"{TAG} {'  ' * depth}{name}: {ms:8.1f} ms{extra}\n")


def _wrap_sync(mod, attr, label, depth=0):
    orig = getattr(mod, attr)

    @functools.wraps(orig)
    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            _log(depth, label, (time.perf_counter() - t0) * 1000)

    setattr(mod, attr, timed)


def install():
    global _installed
    if _installed or os.environ.get("ASFP8_EXEC_TRACE") != "1":
        return

    try:
        import execution
        import comfy.samplers
        import comfy.sampler_helpers
        import comfy.model_management
    except Exception as e:
        print(f"{TAG} comfy modules unavailable ({e!r}); tracer off.")
        return

    # Per-node timing: wrap the async node dispatch.
    orig_god = execution.get_output_data

    @functools.wraps(orig_god)
    async def timed_god(prompt_id, unique_id, obj, input_data_all, **kw):
        t0 = time.perf_counter()
        try:
            return await orig_god(prompt_id, unique_id, obj, input_data_all, **kw)
        finally:
            _log(0, f"node {obj.__class__.__name__}",
                 (time.perf_counter() - t0) * 1000, f"  (id {unique_id})")

    execution.get_output_data = timed_god

    _wrap_sync(comfy.samplers.CFGGuider, "sample", "CFGGuider.sample", 1)
    _wrap_sync(comfy.samplers.CFGGuider, "outer_sample", "outer_sample", 2)
    _wrap_sync(comfy.samplers.CFGGuider, "inner_sample", "inner_sample", 3)
    _wrap_sync(comfy.samplers, "process_conds", "process_conds", 4)
    _wrap_sync(comfy.samplers.KSAMPLER, "sample", "KSAMPLER.sample (step loop)", 4)
    _wrap_sync(comfy.sampler_helpers, "prepare_sampling", "prepare_sampling", 2)
    _wrap_sync(comfy.sampler_helpers, "cleanup_models", "cleanup_models", 2)
    _wrap_sync(comfy.model_management, "load_models_gpu", "load_models_gpu", 3)

    _installed = True
    print(f"{TAG} execution tracer on (stderr).")
