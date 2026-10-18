"""
ComfyUI-AppleSilicon-Optimized
(github.com/luix93/ComfyUI-AppleSilicon-Optimized — grew out of, and still
installs from, the ComfyUI-AppleSilicon-FP8 directory/package name)

Apple Silicon acceleration layer for ComfyUI: makes FP8/INT8-quantized models
(FLUX, SD3.5, Wan, LTX, Krea, Z-Image, ...) run fast on Metal/MPS via runtime
patches and JIT-compiled Metal 4.1 kernels, and works around a psutil bug on
recent macOS betas that otherwise kills renders mid-way.

ComfyUI imports this at startup, before any model loads, so the patches are active
for the whole session. Each patch is a no-op on machines that don't need it
(non-macOS, non-MPS, or where the relevant library isn't installed).

Patches applied:
  1. psutil.virtual_memory() vm_stat fallback   (macOS 26/27 beta crash)
  2. comfy_kitchen eager FP8 + NVFP4/MXFP8 decode (Ideogram 4, LTX, other ck/mixed-precision models)
  3. torch._scaled_mm FP8 on MPS                (FLUX, SD3.5, FP8 _scaled_mm path)
  4. F.rms_norm manual fp32 path on MPS         (PiD >=2048px: black image / NaN)
  5. flash_attn drop-in + fast SDPA on MPS      (mtlflashattn: 3-4x over fused SDPA,
                                                 fixes the large-attention OOM/cliff)
  6. cast_bias_weight FP8 weight+bias decode    (FP8 UNETLoader dtype: weight/bias cast crash)
  7. stochastic_rounding FP8 CPU reroute        (LoRA + FP8 base model: re-quant crash)
  8. torch.Tensor.to FP8<->float on MPS         (3rd-party fp8 Linears: WanVideo custom_linear, etc.)
  9. WanVideo block-swap neutralizer on MPS      (block swap is CUDA-VRAM-only; breaks on MPS)
 10. F.linear FP8 operand decode on MPS         (T5 encoder + any FP8 Linear using F.linear directly)
 11. text_encoder_device CPU->MPS on MPS         (text/LLM encoders default to CPU on Apple Silicon)
 12. torch._int_mm GPU path on MPS               (INT8 models: _int_mm has no Metal kernel -> CPU-fallback freeze)
 13. int8-fast Linear via MPS bf16 GEMM           (INT8 wide-batch matmul was 3-5x too slow in fp32 _int_mm)
 14. MLX-backed TextGenerate on MPS (Qwen3-VL + Gemma3)  (Krea2 ~50s & LTX2 ~13h eager generate -> MLX)
 16. strided FP8 ops -> CPU on MPS (global)        (NVFP4/MXFP8 quant+dequant: reshape/contiguous of fp8 crashes MPS)
 17. INT8 W8A8 via bit-exact Metal kernel on MPS   (opt-in ASFP8_INT8_EXT=1: int8 matmul ~1.75x over bf16,
                                                    bypasses per-step fp32 weight dequant/un-rotation; Cider-derived kernel)
 18. standard-fp8 Linear via W8A16 Metal kernel    (opt-in ASFP8_FP8_EXT=1: half×fp8 matmul2d skips the
                                                    per-forward fp8->bf16 weight decode; 1.1-4.2x on MLP shapes, M5)
 19. per-call MPS attention routing                (opt-in ASFP8_ATTN_ROUTER=1: unmasked long-seq attention ->
                                                    mtlflashattn flash kernel, masked long-seq -> sub-quad; the
                                                    memory-unsafe stock-SDPA fallback path is never taken)
 20. fp8 fill/factory ops -> CPU on MPS            (zeros/ones/full/eye/_like/fill_: no Metal fp8 fill kernel;
                                                    torch>=2.13 additionally regressed torch.zeros for fp8 on MPS)
 21. Lumina/Z-Image fused attention pre-pass       (opt-in ASFP8_DIT_EXT=1: qkv-split + QK RMSNorm + RoPE +
                                                    head transpose in one Metal pass; ~7.5ms -> ~0.5ms per layer)
 22. comfy_quant missing-'format' repair            (community int8-convrot checkpoints: default int8+scale
                                                    layers to int8_tensorwise instead of failing to load)
 23. fast apply_rope on MPS                          (opt-in ASFP8_DIT_EXT=1: flux-family q/k RoPE via one
                                                    Metal pass; ~7.3ms -> ~0.7ms per attention at Flux2 shapes)
 (#15 fp8-native F.linear retired — wrong seam; the fp8-native win is patch #3's opt-in _scaled_mm fast path)

See README.md for details. MIT licensed.
"""

# ComfyUI looks for these on every custom node. Empty = side effects only, no nodes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
# Frontend extension: "Free RAM" topbar button (web/asfp8_topbar.js) — one
# click unloads all models and releases caches + the MPS allocator pool.
WEB_DIRECTORY = "./web"

if __spec__ is not None and __spec__.parent:
    # Loaded as part of a package (normal ComfyUI startup) — install patches.
    # When imported without a real parent package (e.g. pytest --import-mode=importlib
    # imports __init__.py directly as a top-level module), __spec__.parent is an empty
    # string and relative imports would fail.  Checking __spec__.parent is CPython-
    # guaranteed behaviour (see importlib docs) and avoids inspecting ImportError
    # message strings that are implementation details liable to change.
    from ._patches import attention_backend, comfykitchen_fp8, linear_fp8, linear_fp8_w8a16, ops_bias_fp8, psutil_vmstat, rmsnorm_mps_large, scaled_mm_fp8, flash_attn_mtl, attn_router_mps, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, te_device_mps, int_mm_mps, int8_linear_mps, int8_linear_kernel_mps, mlx_textgen, fp8_mps_strided, fp8_fill_mps, lumina_attn_fast, quant_meta_fix, int8_embedding, rope_fast, flux_attn_fast, krea2_attn_fast, conv3d_mps, ltx_fast, ltx25_fast, wan_fast, wan_vae_fast, vae_slice_attn_mps, seedvr2_vae_mps, seedvr2_fast, seedvr2_loader, ideogram_fast, minimax_fast, minimax_relaxed, minimax_vae_fast, minimax_h3cache, minimax_vae_tiling, minimax_vae_batch, free_mem_mps, exec_trace, optrace, mps_profile, sdxl_geglu, qwen21_fast, yue2_fast, qwen3vl_decode, qwen_vl_vision, vision_preprocessors, flux3_fast

    # Acceleration controls and optional image/video/latent upscalers.
    # Old visual graphs migrate in web/h3_migration.js.
    from ._patches import minimax_h3_turbo, minimax_h3_unified, minimax_h3_presets, super_resolution, h3_latent_upscaler
    for _mod in (attention_backend, minimax_h3_turbo, minimax_h3_unified, minimax_h3_presets, super_resolution, h3_latent_upscaler):
        NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

    from . import prompt_studio
    from server import PromptServer
    NODE_CLASS_MAPPINGS.update(prompt_studio.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(prompt_studio.NODE_DISPLAY_NAME_MAPPINGS)
    prompt_studio.install_routes(PromptServer.instance)

    # Identity banner: every subsequent "[AppleSilicon-Optimized/...]" console
    # line comes from this project.
    print("[AppleSilicon-Optimized] ComfyUI-AppleSilicon-Optimized — Apple "
          "Silicon acceleration layer (runtime patches + Metal 4.1 kernels); "
          "lines tagged [AppleSilicon-Optimized/...] are ours.")

    # Default to native acceleration on macOS/MPS. Explicit master/per-feature
    # values always win, and each route retains its own guards and fallbacks.
    from ._patches._common import apply_native_master_switch as _anms
    from ._patches._common import apply_default_attention_backend as _adab
    from ._patches._common import ensure_native_prereqs as _enp
    _implied = _anms()
    if _implied:
        print(f"[AppleSilicon-Optimized] ASFP8_NATIVE=1 -> {', '.join(v + '=1' for v in _implied)}")
    _attention_implied = _adab()
    if _attention_implied:
        import os as _default_os
        print("[AppleSilicon-Optimized] native PyTorch attention defaults -> "
              + ', '.join(v + '=' + _default_os.environ[v] for v in _attention_implied))
    # Self-heal JIT prerequisites BEFORE any extension tries to build
    # (fresh Desktop venvs: no ninja, stripped torch headers). ASFP8_AUTOFIX=0
    # disables; the loud banner below still reports anything left broken.
    for _act in _enp():
        print(f"[AppleSilicon-Optimized] autofix: {_act}")

    # Bisection switches (for debugging a regression to a single patch):
    #   ASFP8_ENABLE_ONLY=psutil_vmstat,comfykitchen_fp8   install ONLY these (by module name)
    #   ASFP8_DISABLE=flash_attn_mtl,rmsnorm_mps_large     install everything EXCEPT these
    # ENABLE_ONLY wins if both are set. Names are the patch module names (last path segment).
    import os as _os
    _only = {n.strip() for n in _os.environ.get("ASFP8_ENABLE_ONLY", "").split(",") if n.strip()}
    _disabled = {n.strip() for n in _os.environ.get("ASFP8_DISABLE", "").split(",") if n.strip()}

    # optrace installs LAST (opt-in ASFP8_TRACE_OPS=1) so it sees every matmul the
    # model dispatches, on top of all the seams the other patches wrapped.
    for _patch in (psutil_vmstat, fp8_mps_strided, fp8_fill_mps, comfykitchen_fp8, scaled_mm_fp8, ops_bias_fp8, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, rmsnorm_mps_large, flash_attn_mtl, attn_router_mps, attention_backend, linear_fp8, linear_fp8_w8a16, te_device_mps, int_mm_mps, int8_linear_mps, int8_linear_kernel_mps, mlx_textgen, lumina_attn_fast, sdxl_geglu, quant_meta_fix, int8_embedding, rope_fast, flux_attn_fast, krea2_attn_fast, conv3d_mps, ltx_fast, ltx25_fast, wan_fast, wan_vae_fast, vae_slice_attn_mps, seedvr2_vae_mps, seedvr2_fast, seedvr2_loader, ideogram_fast, minimax_fast, minimax_relaxed, minimax_vae_fast, minimax_vae_tiling, minimax_vae_batch, free_mem_mps, qwen21_fast, yue2_fast, qwen3vl_decode, qwen_vl_vision, vision_preprocessors, flux3_fast, exec_trace, optrace, mps_profile):
        _short = _patch.__name__.rsplit(".", 1)[-1]
        if (_only and _short not in _only) or _short in _disabled:
            print(f"[AppleSilicon-Optimized] skipping {_short} (ASFP8_ENABLE_ONLY/ASFP8_DISABLE)")
            continue
        try:
            _patch.install()
        except Exception as _e:  # never take ComfyUI down because of us
            import traceback
            print(f"[AppleSilicon-Optimized] patch {_patch.__name__} failed: {_e}")
            traceback.print_exc()

    # Loud prerequisite check: a fresh Desktop-provisioned venv ships without
    # ninja and with torch's C++ headers stripped — every JIT Metal kernel
    # then silently disables and renders run 2x+ slower on eager fallbacks
    # (each loader logs one quiet line that is easy to miss). Say it once,
    # unmissably, with the exact fix.
    if _os.environ.get("ASFP8_INT8_EXT") == "1" or _os.environ.get("ASFP8_DIT_EXT") == "1":
        _missing = []
        try:
            import ninja as _ninja  # noqa: F401
        except Exception:
            _missing.append("ninja (fix: <venv>/bin/python3 -m pip install ninja)")
        try:
            import torch as _torch
            import os.path as _osp
            if not _osp.exists(_osp.join(_osp.dirname(_torch.__file__),
                                         "include", "torch", "extension.h")):
                _missing.append("torch C++ headers (Desktop strips them; copy "
                                "site-packages/torch/include/ from a venv with "
                                "the same torch version, or pip reinstall torch)")
        except Exception:
            pass
        if _missing:
            print("[AppleSilicon-Optimized] " + "!" * 62)
            print("[AppleSilicon-Optimized] NATIVE METAL KERNELS DISABLED — renders will be MUCH slower.")
            for _m in _missing:
                print(f"[AppleSilicon-Optimized]   missing: {_m}")
            print("[AppleSilicon-Optimized] " + "!" * 62)
