# Flux.3: exact incremental acceleration on M5 Ultra

October 16, 2026 · M5 Ultra 80 GPU / 256 GB · macOS 27 · ComfyUI v0.41.2 · PyTorch 2.14.1 · Comfy Kitchen 0.2.37.

## Complete diffusion workflows

Seconds, with decoding and PNG export included. Stock and previous-pack columns use the second run. Both final new-path runs are shown; run 1 includes initial loading/encoding and run 2 supplies the warm speedup. Unchanged text conditioning is cached. **Edit comparisons use the identical frozen expanded prompt**, so these rows exclude prompt generation.

| Model / task | Dimensions | Steps / CFG | Stock | Previous pack | New run 1 / run 2 | Stock speedup |
|---|---|---|---:|---:|---:|---:|
| Distilled / t2i | 1024×1024 | 16 / 1 | 12.115 | 7.904 | **8.342 / 6.166** | **1.96×** |
| Base / t2i | 1024×1024 | 50 / 3.5 | 69.154 | 45.442 | **37.782 / 35.470** | **1.95×** |
| Distilled / edit | 1376×752 | 16 / 1 | 21.939 | 16.073 | **15.616 / 13.347** | **1.64×** |
| Base / edit | 896×1152 | 50 / 3.5 | 118.772 | 87.894 | **79.072 / 75.353** | **1.58×** |

All four full-step tests produced byte-identical conditioning, noise, sampler latents and decoded floating frame tensors across previous-pack and new-path runs. This establishes incremental parity on these fixtures, **not stock equivalence**: the pack already uses its existing INT8 W8A8 path instead of stock floating dequantization.

## What was implemented

1. Keep owned copies of ComfyUI-prepared MLP output weights within a sampling call. Avoid repeating ConvRot/cast preparation while retaining the same floating GEMM.
2. Preserve PyTorch RMS normalization and fuse only Q/K rotary multiplication/addition and output layout. Separate FP32 operations and BF16 rounding are retained.
3. Fuse BF16 SwiGLU using the existing exact SiLU lookup path, including its intermediate rounding.
4. Cache prepared Qwen3.5 prompt-expander weights within one generation call. No predictions, activations or timesteps are skipped.

The synchronized two-step profile put about 350 ms in MLP output/activation/residual, including roughly 158 ms attributable to repeated weight preparation, and 131 ms in Q/K normalization/rotary operations. Inclusive block time was 1,141 ms. Profile overhead is not used as the final performance measurement.

## Prompt expansion

The saved edit graphs include a Qwen3.5-9B prompt expander with MTP disabled. Stock places it on CPU: the attempted stock edit generated only four tokens in about 86 seconds before the isolated run was stopped at 97 seconds. There is no completed stock end-to-end prompt-generation baseline; no extrapolated total is presented. Our existing pack already places this encoder on MPS.

| Prompt fixture | Previous pack | Prepared-weight cache | Validation |
|---|---:|---:|---|
| Distilled edit | 116.75–117.44 s | 9.81–9.94 s | Same 210 token IDs, expanded text, short-run latents and pixels |
| Base edit, CFG 3.5 | 116.39 s | 9.98 s | Same expanded text, short-run conditioning, latents and pixels |

These are complete prompt-expander node times. The phase improves about **12×**. Full-step diffusion validation above uses those saved expanded strings to keep inputs identical.

## Rejected and retained paths

- Complete Q/K normalization fusion changed BF16 results (maximum absolute difference 0.25 in the screened operations). Rejected.
- Explicit fused multiply-add in RoPE changed outputs (maximum absolute difference 0.0625); final latents also differed. Rejected. The accepted shader disables contraction and keeps separate products/addition.
- Existing fused QKV checkpoints, attention and matrix-multiply implementations are retained. They are not claimed as new work.
- Distilled timestep-dependent conditioning remains recomputed each step.

## Memory and compatibility

Sampling caches used approximately 2.0 GiB for text-to-image and 2.27 GiB for reference-image edits. Prompt-generation cache used approximately 16.7 GiB. Caches are phase-local and released on success, interruption or failure; PyTorch may retain freed allocations in its reusable pool. Budgets and Metal headroom checks limit admission. The complete edit runs observed about 52.4 GiB peak Metal driver allocation. No memory guard fired.

New paths are gated to the reviewed M5 Ultra / PyTorch 2.14.1 / Comfy Kitchen 0.2.37 stack with fast math disabled. Dynamic transforms, custom sampling hooks and active MTP draft generation retain applicable original paths. Independent switches and memory limits are in the [Flux.3 guide](../docs/flux3.md).

## Reproduction and scope

- Base and step-distilled INT8 ConvRot Klein checkpoints, Qwen3-VL 4B INT8 ConvRot text encoder and Flux.2 VAE. Euler/simple, base 50 steps / CFG 3.5; distilled 16 steps / CFG 1.
- Both supplied saved graphs were edit workflows. T2I variants use a fixed authored bicycle-shop prompt at 1024². Edit tests preserve the public template references and their dimensions; the base CFG was changed to 3.5 as requested. Original saved workflows were not edited.
- Base seed 1106932821006959; distilled seed 563228241303057. Prompt generation preserves saved parameters and seed 0, with MTP off.
- Serial isolated servers; 110 GiB RSS ceiling, 48 GiB available-memory floor, bounded MPS watermarks. Main interactive server left unchanged. Tensor capture/hashing runs outside workflow timing.
- UI-only comparison branch excluded; PNG output retained. No private reference media or SATA files accessed.
- Two renders per stock case, two per prior-pack case and two per new-path case. These are repeatability checks, not broad statistical estimates or a guarantee across all prompts/hardware.

[Machine-readable measurements](data/m5-ultra-flux3.json) · [ComfyUI regression results](m5-ultra-comfy0412-smoke.md)

## Complete edit graphs with fresh prompt generation

The final validation regenerates the expanded prompt, encodes the reference, runs the full step count, decodes and exports. Prompt generation is forced on both runs; these are not cached-prompt timings. Both runs and the matched frozen-prompt runs have identical conditioning, noise, latents and decoded tensors.

| Model | First complete run | Warm complete run | Warm prompt phase |
|---|---:|---:|---:|
| Distilled edit | 27.044 s | **24.005 s** | 9.968 s |
| Base edit | 88.354 s | **85.947 s** | 9.989 s |

These complete runs are not compared against the incomplete stock CPU prompt-expander attempt.

Visual inspection of the final T2I and edit outputs found coherent images and the requested edits (red gown; openwork crescent earring). The new path preserves the prior optimized pixels exactly. This is a functional sanity check, not a broad stock-versus-optimized quality study.

## Verification

- 19 CPU unit tests passed for ownership, invalidation, limits, fallback and cleanup.
- Six accelerated-operation fixtures passed byte comparison against Comfy Kitchen/PyTorch, including batch size two, strided Q/K and shared/per-batch positional embeddings. Fast-path counters confirm the kernels actually ran.
- Reproduce operation checks with ComfyUI’s interpreter: `python dev/check_flux3_operations.py --comfy /path/to/ComfyUI`.
- Full-step workflow tests, independently reset caches on every sampling/generation call; no approximate Flux.3 option introduced.

## Final hook lifetime and unrelated-model checks

The final implementation installs the prepared-weight hook only during an
active Flux sampling/prompt phase and restores the original function afterward.
All four full-step cases were rerun twice with this final implementation; the
table above uses those final warm measurements. Every captured tensor matched
the previous-pack references, and restoration was asserted after every phase.
A further two-step base edit with fresh prompt generation also matched its
saved reference, including both active caches.

The full fresh-prompt totals above were measured before this hook-lifetime
refinement; they are retained as actual measurements, not reconstructed by
adding phase times. Their source hash is recorded separately in the data file.
The refinement changes dispatch lifetime, not arithmetic.

An earlier permanently installed cast wrapper showed a small 0.1–0.2 s penalty
in alternating short YuE tests. The final version removes that wrapper entirely
outside Flux phases. SDXL and YuE post-install captures retained identical
outputs. Nested-phase, interruption and foreign-hook preservation checks are
included in the 19 unit tests.

The final installed YuE check completed twice (6.785 s first / 5.875 s warm),
with all token IDs, conditioning, sampler and decoded audio tensors unchanged.
Runtime assertions confirmed Flux.3 was registered while its cast hook was
absent throughout this unrelated workflow. Short-run timing variation remains;
these checks are not evidence of a YuE speed improvement.
