# Environment variables

Set startup switches before launching ComfyUI, then restart. Explicit
per-feature overrides take precedence over the automatic native defaults.
Optional workflow approximations remain independently selectable.

| Var | Default | Meaning |
|---|---|---|
| `ASFP8_SETUP_KEEP_TORCH` | 0 | installer/first-launch opt-out: `1` retains installed Torch/vision while installing pack dependencies; unchanged launches skip setup |
| `ASFP8_NATIVE` | on on macOS/MPS; otherwise unset | Master switch: enables the six native feature groups below; `0` disables automatic implication; explicit per-feature settings win; it does not apply LoRAs or approximation nodes |
| `ASFP8_FP8_EXT` | implied by native | fp8-native `_scaled_mm` kernel, validated on M5 / Metal 4.1 |
| `ASFP8_FP8_SGEMM` | implied by native | staged half×fp8 W8A16 backend |
| `ASFP8_INT8_EXT` | implied by native | int8 W8A8 kernel + fused epilogue, validated on M5 |
| `ASFP8_ATTN_ROUTER` | implied by native | per-call attention routing (flash41 / bmm / sub-quad) |
| `ASFP8_ATTN_BACKEND` | `pytorch` on pinned tested 2.14.0/2.14.1 + M5 Ultra with native enabled; otherwise unset | selects the tested PyTorch MPS route; an explicitly empty value retains the earlier router; explicit backend settings win |
| `ASFP8_MPP_QCACHE` | on | exact query-load hoisting under the PyTorch attention route; guarded to pinned validated M5 Ultra / 2.14.0 and 2.14.1 builds, headers and shapes; H3 supports up to 65,536 tokens, other families up to 32,768; `0` restores the unmodified PyTorch kernel |
| `ASFP8_H3_LONG_ATTN` | on | exact long-H3 attention scheduling on validated M5 Ultra / PyTorch 2.14.1, 30,001–65,536 tokens; changes dispatch order and one scheduling barrier, retaining arithmetic and output bytes; `0` restores the preceding query-cache schedule. See [long-clip measurements](../benchmarks/m5-ultra-h3-long-exact.md) |
| `ASFP8_ATTN_NA` | 1 | production independent-simdgroup flash41 kernel; `0` restores the older cooperative kernel |
| `ASFP8_DIT_EXT` | implied by native | DiT block-fusion kernels (all families) |
| `ASFP8_INT8_EMBEDDING` | on | preserve already-dequantized INT8 embedding weights across CPU/MPS casts; compatibility repair independent of native kernels |
| `ASFP8_QWEN21_FAST` | on (under DIT_EXT) | exact Qwen-Image 2.1 fusions on the pinned validated M5 Ultra / PyTorch 2.14.0 and 2.14.1 builds; `0` retains the encoder fix and generic native paths |
| `ASFP8_SOL_LAYOUT` | `head_major` | optional SOL physical layout; `legacy` selects the prior wrapper/shaders for diagnosis. New layout preserves prior SOL output bytes, not exact-native attention bytes; restart after changing it |
| `ASFP8_YUE2_FAST` | on | exact phase-local cast-weight reuse, guarded vocabulary projection and single-token BF16 RoPE for the MPS YuE2 encoder; scoped acoustic attention under the existing PyTorch query-cache route; `0` restores the preceding YuE2 paths |
| `ASFP8_YUE2_WEIGHT_CACHE` / `ASFP8_YUE2_VOCAB_HEAD` / `ASFP8_YUE2_ATTN` / `ASFP8_YUE2_ROPE` | on | independent YuE2 switches; exact against the existing GPU stack |
| `ASFP8_YUE2_CACHE_GIB` / `ASFP8_YUE2_VERBOSE` | 8 / off | temporary weight-cache cap (also limited by device memory/headroom) / optional phase counters; cache size 0 disables storage |
| `ASFP8_QWEN21_ROPE` / `ASFP8_QWEN21_MODULATE` / `ASFP8_QWEN21_RESIDUAL` | on (under QWEN21_FAST) | independent switches for Q/K RMSNorm + RoPE, modulation, and gated residual fusions |
| `ASFP8_SDXL_GEGLU` | on (under DIT_EXT) | exact FP16 GELU + multiplication fusion for SDXL on PyTorch 2.14 MPS; `0` restores eager GEGLU |
| `ASFP8_CONV3D` | implied by native | conv3d → sliced conv2d + Wan VAE fast path |
| `ASFP8_FLUX_FAST` / `KREA2_FAST` / `LTX_FAST` / `WAN_FAST` / `WAN_VAE_FAST` / `IDEOGRAM_FAST` / `MINIMAX_FAST` | on (under DIT_EXT/CONV3D) | per-architecture kill switches |
| `ASFP8_KREA2_GLUE` | on | exact Krea2 normalization/modulation, residual, SwiGLU and attention-layout fusions; guarded to M5 Ultra / PyTorch 2.13–2.14 under KREA2_FAST |
| `ASFP8_KREA2_GEMM_TILE` | 1 | exact M5 Ultra BF16 columnwise INT8 output tile for 1MP Krea2 shapes (M=4096…4352, K=6144, N=1536/6144); 0 restores the previous schedule |
| `ASFP8_IDEOGRAM_ATTN_TILE` | 1 | exact M5 Ultra D=256 attention output-channel tiling for S=4096…4608; 0 restores the previous full-width output tile |
| `ASFP8_IDEOGRAM_GEMM_TILE` | 1 | exact M5 Ultra projection-specific INT8 scheduling for 1MP Ideogram shapes; 0 restores the previous schedule |
| `ASFP8_KREA2_ATTN_SCHEDULE` | on | exact synchronized Morton attention schedule for M5 Ultra BF16 Krea2, B=1, 48Q/12KV, D=128, 8192–12287 tokens; other geometries keep their existing schedule |
| `ASFP8_MINIMAX_ATTN` / `ASFP8_MINIMAX_BLOCK` | 1 | MiniMax direct QKV/attention-output layout and segmented block-glue fusions |
| `ASFP8_MINIMAX_GRAPH` | 1 | exact guarded H3 final-layer graph rewrite (target RMSNorm + modulation + fp32 casts) |
| `ASFP8_MINIMAX_VAE_FAST` | 1 (under DIT_EXT) | FP16 H3 video-decoder QK normalization/RoPE/layout and elementwise kernels; `0` restores core forwards |
| `ASFP8_MINIMAX_VAE_SWIGLU` / `ASFP8_MINIMAX_VAE_MERGE` | 1 | per-feature switches for exact FP16 SwiGLU and attention-output merge/sanitization |
| `ASFP8_MINIMAX_RELAXED` | 0 (off) | explicit experimental H3 kernel preset; independent of `ASFP8_NATIVE`; per-feature settings override it; restart to change |
| `ASFP8_MINIMAX_ATTN_MODE` | exact (tuned under RELAXED) | `exact`, original relaxed `tail`, or `tuned`; the relaxed preset supplies its selected mode when unset; long H3 only, workflow attention overrides take precedence |
| `ASFP8_MINIMAX_VAE_DENSE_ATTENTION` | 0 (1 under RELAXED) | optional non-bit-exact dense FP16 H3 VAE attention; requires native VAE/Metal attention; independent of sampling |
| `ASFP8_MINIMAX_VAE_ATTN_TUNING` | 0 (1 under RELAXED) | tuned batch-dependent tiles within optional dense VAE attention; `0` retains the original dense kernel settings |
| `ASFP8_MINIMAX_VAE_QKV` | 0 (1 under RELAXED) | FP16 QKV projection with FP32 accumulation and fused normalization/RoPE/layout; `0` retains the existing GEMM; independent of dense attention |
| `ASFP8_MINIMAX_VAE_INT8` | 1 (under native INT8/VAE kernels) | native W8A8 for an explicitly loaded H3 INT8 ConvRot VAE; `0` restores the prior weight-only MPS decoder; FP16 VAE and sampler unchanged |
| `ASFP8_INT8_QKV_SCATTER` | 1 | exact H3 INT8 QKV epilogue directly to three attention-layout tensors; `0` restores the full fused-QKV buffer |
| `ASFP8_INT8_QKV_NORM_ROPE` | 1 | exact D=128 H3 INT8 projection + optional standard runtime-LoRA delta + Q/K RMSNorm + partial-RoPE epilogue |
| `ASFP8_MINIMAX_CROSS_BLOCK` | 1 | exact MLP-gate → next-attention-AdaLN carry when no workflow block replacement is active |
| `ASFP8_INT8_ACT_CACHE` | 1 | one-slot reuse of identical rotate + row-quant inputs shared by projection groups |
| `ASFP8_INT8_LONG_SCHEDULE` / `ASFP8_INT8_LONG_QKV_TAIL` | 1 | exact long-M INT8 schedule / exact QKV suffix dispatch (active only at M≥8192) |
| `ASFP8_INT8_OPROJ_FUSE` | 1 | fused o-proj pre-pass: `1` exact D=256 rotation or full cross-head FWHT for D<256, `2` always full FWHT, `0` off |
| `ASFP8_INT8_SWIGLU` | 1 | keep H3's SwiGLU down-projection in native W8A8; `0` restores ComfyUI's stock dequantized W8A16 seam |
| `ASFP8_INT8_SWIGLU_ROT` | 1 | fuse SwiGLU + convrot before H3's INT8 FC2 (rare ±1 quantized-byte drift from FWHT reassociation) |
| `ASFP8_INT8_MLP_BRIDGE` | 1 | tiled H3 FC1 → SwiGLU/convrot/premax quant → FC2 bridge; exact relative to the native SwiGLU/convrot route, `0` restores its materialized tensors |
| `ASFP8_INT8_MLP_LORA_TILE_ROWS` | 2048 | row scratch size for the exact standard-LoRA H3 FC1/SwiGLU/FC2 bridge; minimum/alignment 128 |
| `ASFP8_MINIMAX_VAE_OVERLAP` | unset (reference 64) | approximate H3 video-VAE tile overlap; 32 is the tested faster setting |
| `ASFP8_MINIMAX_VAE_DECODE_BATCH` | 1 (off) | exact experimental tile batching; slower in the real 124-frame test, with peak decoder memory scaling by batch |
| `ASFP8_SOL_EXT` | 1 | set `0` to force-disable the Sol sparse-attention Metal kernel (the H3 sparse node then refuses to apply) |
| `ASFP8_SOL_ROUTER` | `diag64` | set `legacy_empirical` to route H3 sparse attention with the **rejected** 32-row-tile router (proxy-row mean/variance threshold), for diagnostics only. `tau` is not comparable between the two |
| `ASFP8_MINIMAX_VAE_TILE_SIZE` | unset (reference 256) | experimental H3 video-VAE tile size (multiple of 16) |
| `ASFP8_AUTOFIX` | on | self-heal JIT prerequisites at boot (install ninja, restore stripped torch headers) |
| `ASFP8_WAN_EXACT` | on | Wan 14B exact FP16 INT8 output fusion and batched attention query caching on the validated M5 Ultra / PyTorch 2.14 build; `0` restores the preceding native paths |
| `ASFP8_WAN_FP16_GEMM` / `ASFP8_WAN_ATTN` | on | independent switches for the Wan FP16 projection epilogue and exact attention optimization, under WAN_EXACT |
| `ASFP8_WAN_CROSS_ATTN` | on | also use the exact Wan query-cache/layout kernel for 512-token text cross-attention; `0` keeps only its self-attention extension |
| `ASFP8_WAN_VAE_CHUNK` | 16 | Wan VAE frames per causal pass (even, 2–32; lower caps decode memory peak) |
| `ASFP8_FREE_MEM` | on | MPS free-memory accounting fix (patch #30) |
| `ASFP8_VAE_SLICE_FIX` | on | VAE slice_attention INT_MAX cap (patch #31) |
| `ASFP8_SEEDVR2_VAE_SYNC` | on | wait between SeedVR2 VAE spatial tiles and bound pending video residual blocks on MPS; scheduling-only protection remains enabled independently of native/relaxed flags |
| `ASFP8_SEEDVR2_VAE_VIDEO_SYNC` | on | include bounded residual-block waits inside SeedVR2 video tiles; setting 0 restores the previous spatial-only schedule and its video memory growth |
| `ASFP8_SEEDVR2_VAE_BLOCK_BATCH` | 4 on reviewed 2.14.1/M5 Ultra/current-core FP16 video; otherwise 1 | cap pending residual blocks at 1–4 while retaining every spatial-tile fence; early fences at 2 GiB driver growth or 40 GiB total (an early-fence threshold, not a hard allocation cap); `1` restores every-block waits, invalid values use 1, FAST=0 retains 1 |
| `ASFP8_SEEDVR2_LOADER` | on | supply official missing conditioning for raw original SeedVR2 7B checkpoints on macOS; existing conditioning and weights are preserved |
| `ASFP8_SEEDVR2_FAST` | on | SeedVR2 exact performance additions; native sampling/SiLU kernels require `ASFP8_DIT_EXT=1` (implied by `ASFP8_NATIVE=1`); `0` at startup retains the memory/error safeguards |
| `ASFP8_SEEDVR2_ATTN_FIX` | on | pack MPS attention windows into bounded buffers to avoid oversized-backing-storage `INT_MAX` failures; independent of the native kernel flag |
| `ASFP8_SEEDVR2_ATTN_BATCH` | 8 with native/FAST, otherwise 1 | maximum equal-size windows per call, clamped to 1–16; small safe images keep the original loop |
| `ASFP8_SEEDVR2_VAE_PACK` | on | reuse folded convolution weights within a tiled SeedVR2 VAE call; requires FAST and VAE_SYNC wrappers |
| `ASFP8_SEEDVR2_VAE_SILU` | on | native FP16 single-frame SiLU + causal padding + channel packing, retaining MPS Conv2D |
| `ASFP8_SEEDVR2_MOD` / `ASFP8_SEEDVR2_RMS` / `ASFP8_SEEDVR2_ROPE` | on | independent native SeedVR2 BF16 sampling fusions; RMS retains PyTorch's existing mean/rsqrt reduction |
| `ASFP8_SEEDVR2_VAE_GN_SPLIT` | on for the reviewed 2.14.1/M5 Ultra/current-core path | exact two-dispatch FP16 GroupNorm + SiLU + causal folding in image VAE residual blocks; preserves intermediate rounding, existing convolutions and memory fences; `0` restores the preceding exact implementation |
| `ASFP8_SEEDVR2_VAE_GN` | off | experimental exact GroupNorm + SiLU + packing on tested PyTorch 2.13; whole-decoder timing was neutral, so it is not enabled by default |
| `ASFP8_FREE_INTERLEAVE` | on | drain the MPS pool between models on explicit free-all requests |
| `ASFP8_MPS_RESERVE_GB` | max(16, 15%) | OS memory reserve held back from comfy's estimate |
| `ASFP8_FA41_F32S` | off | fp32 score tiles in flash41 |
| `ASFP8_FP8_EXT_MIN_DIM` | 8192 | fp8 kernel size gate |
| `ASFP8_ENABLE_ONLY` / `ASFP8_DISABLE` | — | bisection: install only / all except these patch modules |
| `ASFP8_ATTN_DEBUG` / `ASFP8_EXEC_TRACE` / `ASFP8_PROFILE` | off | route logging / per-node timing / profiling (stderr) |
| `ASFP8_EXEC_TRACE_SYNC` | off | with EXEC_TRACE: time the GPU tail drain after the sampler loop (splits real overhead from queued GPU work) |
| `APPLESILICON_FP8_PSUTIL` | auto | psutil fix: `auto` / `off` / `force` |
| `MTLFLASHATTN_SDPA*` | — | patch #5 gates (see mtlflashattn docs) |

## Flux.3 (ComfyUI v0.41.2)

See [Flux.3 controls and memory budgets](flux3.md). `ASFP8_FLUX3_FAST=0`
disables the new paths; sampling cache, prompt cache, RoPE and SwiGLU have
independent switches. Enabled automatically only on the reviewed M5 Ultra /
PyTorch 2.14.1 / Comfy Kitchen 0.2.37 stack.
