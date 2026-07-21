"""Individual MPS/FP8 patches, each exposing an idempotent install()."""

# Patch numbers match the README "What it fixes" table (#1–#10).
# Install order (the tuple in the top-level __init__.py) differs from patch
# number order; the comments here use patch numbers, not install positions.
__all__ = [
    "psutil_vmstat",        # patch #1  — psutil vm_stat fallback (macOS beta crash)
    "comfykitchen_fp8",     # patch #2  — comfy_kitchen eager FP8 dequant (Ideogram 4)
    "scaled_mm_fp8",        # patch #3  — torch._scaled_mm FP8 → bf16 + matrix-unit matmul
    "rmsnorm_mps_large",    # patch #4  — F.rms_norm large-row fp32 path (PiD black image)
    "flash_attn_mtl",       # patch #5  — mtlflashattn SDPA (large attention OOM/correctness)
    "ops_bias_fp8",         # patch #6  — cast_bias_weight FP8 weight+bias decode (UNETLoader)
    "stochastic_round_fp8", # patch #7  — stochastic_rounding FP8 re-quant via CPU (LoRA)
    "tensor_to_fp8",        # patch #8  — torch.Tensor.to FP8↔float shim (Python-level .to())
    "wan_blockswap_mps",    # patch #9  — WanModel.forward block-swap neutralizer
    "linear_fp8",           # patch #10 — F.linear / nn.Linear FP8 operand decode
    "te_device_mps",        # patch #11 — text_encoder_device CPU→MPS on Apple Silicon
    "int_mm_mps",           # patch #12 — torch._int_mm on GPU on MPS (INT8 models, no CPU fallback)
    "int8_linear_mps",      # patch #13 — int8-fast wide-batch Linear via MPS native bf16 GEMM
    "mlx_textgen",          # patch #14 — MLX-backed Qwen3-VL TextGenerate (prompt expansion)
    # patch #15 (fp8_linear_mps, opt-in fp8-native F.linear) RETIRED — wrong seam:
    #   real ComfyUI fp8 routes through torch._scaled_mm (QuantizedTensor hides the fp8
    #   dtype so the F.linear gate never fired), and weight-only fp8 only won at tiny M.
    #   The fp8-native win lives in patch #3's _scaled_mm fast path (ASFP8_FP8_EXT=1).
    #   Patch #18 revisits the idea at the RIGHT seam (forward_comfy_cast_weights) after
    #   removing the kernel's per-call COMMIT_AND_WAIT stall — see linear_fp8_w8a16.
    "linear_fp8_w8a16",     # patch #18 — standard-fp8 Linear via half×fp8 W8A16 kernel (opt-in)
    "attn_router_mps",      # patch #19 — per-call MPS attention routing (opt-in ASFP8_ATTN_ROUTER=1:
                            #   unmasked long-seq → mtlflashattn flash, masked long-seq → sub-quad, never stock SDPA)
    "fp8_fill_mps",         # patch #20 — fp8 fill/factory ops (zeros/ones/full/eye/_like/fill_) via CPU on MPS
                            #   (no Metal fp8 fill kernel; torch>=2.13 regressed fp8 torch.zeros on MPS)
    "lumina_attn_fast",     # patch #21 — Lumina/Z-Image fused qkv-split+QKNorm+RoPE+transpose (opt-in
                            #   ASFP8_DIT_EXT=1; dit_ext Metal kernel, ~7.5ms -> ~0.5ms per layer)
    "quant_meta_fix",       # patch #22 — default missing comfy_quant "format" to int8_tensorwise
                            #   (community int8-convrot checkpoints that omit it fail to load otherwise)
    "rope_fast",            # patch #23 — flux-family apply_rope via dit_ext Metal kernel (opt-in
                            #   ASFP8_DIT_EXT=1; ~7.3ms -> ~0.7ms per attention call at Flux2 shapes)
    "flux_attn_fast",       # patch #24 — Flux double/single block fused qkv+QKNorm+RoPE+seq-cat
                            #   (opt-in ASFP8_DIT_EXT=1; kills 4 RMSNorms + 3 seq-cats per double block)
    "krea2_attn_fast",      # patch #25 — Krea2 GQA-native attention: fused q/k/v norm+RoPE at native
                            #   head counts + flash41 grouped-KV (no repeat_interleave; ASFP8_DIT_EXT=1)
    "conv3d_mps",           # patch #26 — Conv3d -> sliced conv2d on MPS (~10x at video-VAE shapes;
                            #   opt-in ASFP8_CONV3D=1, implied by ASFP8_NATIVE)
    "ltx_fast",             # patch #27 — LTX one-pass split-RoPE + fused rms_norm+modulate with
                            #   per-frame timestep rows (opt-in ASFP8_DIT_EXT=1)
    "wan_fast",             # patch #28 — Wan block fusion: LN+modulate/gates with per-frame rows (no
                            #   repeat_e copies) + wide RMSNorm+RoPE via fp16/bf16 wan_ext kernels
    "free_mem_mps",         # patch #30 — MPS get_free_memory counts torch's reusable allocator
                            #   pool (stops needless model eviction / 14s Wan re-uploads per run)
    "wan_vae_fast",
    "ideogram_fast",        # patch #32 — Ideogram 4 fused block glue (norm+modulate,
                            # gate+residual via dit_ext; FFN silu-gating via Lumina class)         # patch #29 — Wan 2.1 VAE chunk batching: decode/encode causal loops and
                            #   run_up sub-chunking stream 8 frames per pass instead of 2 (ASFP8_CONV3D=1)
    # Internal helpers (not patches):
    # "_common"             — decode_fp8, fp8_to_float_lut, FP8_DTYPES
    # "na_gemm"             — optional NA matmul2d backend (not wired into hot path)
    # "_mlx_qwen3vl"        — MLX Qwen3-VL load/generate backend for patch #14
    # "fp8_ext"             — Metal 4.1 fp8 matmul2d ObjC++ extension (patch #3 fp8-native scaled_mm backend)
    # "fp8s_ext"            — staged Metal 4.1 half×fp8 W8A16 extension (patch #18 preferred backend,
    #                         opt-in ASFP8_FP8_SGEMM=1: tiled/swizzled geometry + fused bias epilogue)
]
