# Flux.3 on Apple Silicon

Flux.3 acceleration is automatic on the reviewed M5 Ultra / PyTorch 2.14.1 stack.
Use the normal ComfyUI model, text encoder, reference-image and sampler nodes;
no extra acceleration node is needed.

Validated configuration: ComfyUI **v0.41.2**, PyTorch **2.14.1**, Comfy Kitchen
**0.2.37**, macOS 27, M5 Ultra 80 GPU / 256 GB. See the
[measurements and rejected candidates](../benchmarks/m5-ultra-flux3.md).

## Models and settings

The tested checkpoints are `flux3_image_klein_base_int8_convrot.safetensors`
and `flux3_image_klein_sd_int8_convrot.safetensors`, with
`qwen3vl_4b_int8_convrot.safetensors` and `flux2-vae.safetensors`.
The supplied edit graphs also use
`qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors` to expand the prompt.

| Model | Tested sampler | Steps | CFG |
|---|---|---:|---:|
| Base | Euler / simple | 50 | 3.5 |
| Step-distilled | Euler / simple | 16 | 1 |

These settings follow the [official model guidance](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/README.md).
Prompt generation keeps the saved sampling parameters, with MTP disabled.
Text-to-image tests use 1024×1024; edit dimensions follow the saved reference
workflow's aspect ratio and latent alignment.

## What changes

- **Sampling-local prepared MLP weights:** reuse owned copies of the floating
  weights produced by ComfyUI's existing cast/ConvRot operation. Keep the same
  floating matrix multiply, activation and residual arithmetic. The tested
  cache uses about 2.0 GiB for text-to-image and 2.27 GiB with reference images.
- **RoPE:** retain PyTorch RMS normalization, then combine the Q/K FP32 rotary
  operations into one Metal pass. Keep separate multiplication/addition and
  the original BF16 rounding. The more aggressive complete normalization
  fusion and explicit fused multiply-add failed exactness checks and were
  rejected.
- **SwiGLU:** reuse the exact BF16 SiLU lookup implementation to combine the
  activation and gate multiplication while preserving intermediate rounding.
- **Prompt expansion:** reuse prepared Qwen3.5-9B weights within one generation
  call. The tested cache uses about 16.7 GiB temporarily. Token generation,
  sampling parameters and RNG remain unchanged. The existing pack already
  places this text encoder on MPS instead of stock's CPU placement.

The weight-cast hook exists only while a Flux.3 sampling or prompt-generation
phase is active; the original cast function is restored afterward. Unrelated
workflows do not pass through a permanent extra cast wrapper.

Caches are cleared after each sampling/generation call, including interruptions
and failures. Released allocations may remain in PyTorch's reusable allocator
pool. No activations, conditioning states or model predictions are reused
across denoising steps. In particular, the distilled model's timestep-dependent
conditioning is still recomputed.

The new changes are **byte-exact to the preceding optimized implementation**
on the tested inputs. This is not a claim that the pack's existing INT8 W8A8
route is identical to vanilla ComfyUI's floating dequantization route.

## Controls and memory limits

All controls are environment variables read when ComfyUI runs:

| Setting | Default | Effect |
|---|---:|---|
| `ASFP8_FLUX3_FAST` | `1` | Set `0` to disable all new Flux.3/Qwen3.5 changes. |
| `ASFP8_FLUX3_WEIGHT_CACHE` | `1` | Sampling-local prepared-weight cache. |
| `ASFP8_FLUX3_ROPE` | `1` | Exact Q/K rotary fusion. |
| `ASFP8_FLUX3_SWIGLU` | `1` | Exact BF16 activation fusion. |
| `ASFP8_FLUX3_PROMPT_CACHE` | `1` | Qwen3.5-9B generation-local weight cache. |
| `ASFP8_FLUX3_CACHE_GIB` | `4` | Requested sampling-cache ceiling. |
| `ASFP8_FLUX3_PROMPT_CACHE_GIB` | `24` | Requested prompt-cache ceiling. |
| `ASFP8_FLUX3_VERBOSE` | `0` | Set `1` to log cache counts and allocated bytes. |

Actual cache budgets are capped at 32 GiB, 12% of Metal's recommended working
set, and 20% of available system memory at phase start. Cache admission also
checks Metal working-set headroom. A zero/invalid budget retains the original
weight-preparation path; partial caches do not evict and repeatedly rebuild
large weights. Smaller Macs may admit fewer weights and achieve smaller gains.

Weight caches fall back for dynamic weight/bias functions, low-VRAM transforms,
VBAR-backed layers and customized sampling hooks/wrappers. The Qwen cache also
falls back for active MTP draft generation. Unsupported framework/library
versions and hardware retain the earlier implementation; M1–M4 performance
and parity have not been validated.
