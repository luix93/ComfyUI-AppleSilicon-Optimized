# Ideogram 4 on M5 Ultra

**2026-09-26 · M5 Ultra / 80 GPU cores / 256 GB · macOS 27.0 · PyTorch 2.14.0.**

The 1024×1024, 12-step workflow improves from **43.89 s stock to 18.23 s optimized**, including VAE decode and PNG export. The previous native stack took **24.64 s**. The new additions reduce that by **26.0%**, preserving every final latent and decoded pixel byte.

The user's **41 s M5 Max** reference implies a 20.5 s target. The Ultra result is **2.25× as fast** as that reference and beats the target by **2.27 s**. The Max was unavailable: its timing is user supplied, and its exact prompt, seed and software build were not revalidated. The stock/native Ultra comparison below uses identical inputs.

## Frozen workflow

The source is `ComfyUI-Shared/workflows/ideagram.json`. The connected text prompt is replaced with a benign **botanical poster**: white daisies in an amber vase, a green pear, and the lettering “BOTANIC STUDY” / “FIELD NOTES · 04”. It follows the workflow's caption template: minified JSON with `aspect_ratio`, `high_level_description`, and `compositional_deconstruction`, including normalized object/text boxes. The disconnected caption-writing utility branch is retained.

- **1024×1024**, exactly 1,048,576 pixels; **12 Euler steps**.
- Ideogram Turbo scheduler: **mu 0.5 / std 1.75**.
- Dual-model CFG **7**, overridden to **3** from 70% through sampling.
- Original seed **71584314815009**.
- Models: `Ideogram4/ideogram4_int8_convrot.safetensors` and `Ideogram4/ideogram4_unconditional_int8_convrot.safetensors`.
- Text encoder: `qwen3vl_8b_fp8_scaled.safetensors`, type `ideogram4`.
- VAE: `flux2-vae.safetensors`, BF16 decode in the measured runs.

Both INT8 checkpoints already contain valid format metadata. Stock runs with the acceleration node absent and requires **no compatibility repair**. Both stacks use PyTorch 2.14 attention. The native stack retains its existing INT8 ConvRot projections and BF16 intermediates; the new work changes no quantization settings, steps, CFG or model weights.

ComfyUI commit: `b33e2b55cae074eca5aec96283cceac19aa249ba`.
PyTorch git: `08187d9e0fba026dc8217405802ab5381dc88d90`.

## Measurements

| Configuration | End to end | Sampling | VAE decode |
|---|---:|---:|---:|
| Stock PyTorch 2.14 | 43.893 s | 43.539 s | 0.303 s |
| Previous native stack | 24.638 s | 24.374 s | 0.210 s |
| Restored preparation + direct attention output layout | 22.479 s | 22.231 s | 0.193 s |
| **New exact stack, including Q/K loop scheduling** | **18.230 s** | **17.982 s** | **0.197 s** |

Warm full-workflow samples:

- Stock: **43.873 / 43.913 s**.
- Previous native: **24.561 / 24.715 s**.
- Preparation and output layout: **22.407 / 22.550 s**.
- Final exact stack: **18.260 / 18.199 s**.
- Reverting the final additions in the same process: **24.370 s**.

The first full render is excluded for model loading and compilation. Warm results include allocation, layout conversion, activation quantization, all 12 sampling steps, complete VAE decode and PNG saving. Unchanged text encoding is cached. Verification, profiling and tensor hashing are outside the accepted timings. Separate processes run stock and native configurations, using an isolated server, one GPU job at a time, preview disabled, and memory watchdogs. The user's normal server is left running.

## What changed

1. **Restore the current Q/K normalization and split-half RoPE path.** Current ComfyUI supplies a rotation matrix; the older Ideogram wrapper only accepted a three-tensor tuple, so it fell back. The new BF16 kernel reads the packed QKV projection, reproduces PyTorch 2.14's four-value-per-thread RMS reduction and weighted FP32 arithmetic, rounds the normalized values to BF16, performs the same FP32 rotation, then writes attention's layout. Cast/offload contexts and projection hooks remain active. Unsupported calls use the existing implementation.
2. **Write attention output directly in the following projection's layout.** Instantiate the installed, hash-checked PyTorch MPP source at D=256, retaining its matrix arithmetic, softmax and reductions. Output allocation uses BSHD strides, removing the following head-merge copy.
3. **Stop forcing the D=256 Q/K inner loop to unroll.** The default source forces sixteen head-dimension fragments to unroll. Letting the Metal compiler schedule that loop improves this Ultra workload without changing the dot-product iteration order or floating-point expressions. Register pressure/temporary lifetime was the working hypothesis; compiler register/spill counts were not measured, so the timing result does not prove a specific hardware cause.

The key saved-tensor measurements include output allocation:

| Operation | Reference | Candidate |
|---|---:|---:|
| Complete Q/K normalization + RoPE + V layout | 2.183 ms | **0.276 ms** |
| D=256 attention, original loop and direct output layout | ~13.2 ms | ~12.5 ms |
| D=256 attention, compiler-scheduled Q/K loop and direct output layout | 13.249 ms | **8.241 ms** |

Actual sampling has **18 heads, D=256 and 4,414 tokens** for both model passes with this 318-token caption. There are **816 attention calls** over 12 steps. Attention preparation and the attention kernel were the useful targets; VAE decode was already around 0.2 s and was not changed.

## Exactness and controls

**57 tests pass**, covering current-API normalization/RoPE, random values, epsilon variants, nonzero storage offsets, aligned/ragged sequence lengths, attention output layout, unsupported calls, opt-outs, and shared PyTorch attention routes. The generated D=128 BF16 and FP16 shader sources are byte-identical to their prior versions; changes are scoped to Ideogram's D=256 calls.

The two-step integration checks compare every complete block directly against the preceding native implementation: **136 block executions per case**, spanning both models. Three cases pass: the main caption/seed, the same caption with seed **981247**, and a shorter teapot/cups caption with seed **42**. All three preserve conditioning, noise, latents and decoded images. Earlier screening also checks Q/K/V and complete attention outputs. Full 12-step runs preserve conditioning, initial noise, both sampler outputs and final FP32 decoded images byte for byte across all repeats. No tensor-error tolerance is substituted for byte equality.

**Stock and native are not byte-identical to one another.** Existing native quantization and text-encoder execution already change the generated result. New versus previous native error is zero. For context, stock/native latent RMSE is **0.427586** and decoded-image RMSE **0.090819** (PSNR **20.84 dB**). These whole-stack numbers are not an error introduced by the new kernels or a quality guarantee. Both saved images have readable lettering and the requested watercolor appearance; flower arrangement and pear placement differ. The new optimized image is identical to the previous native image.

## Other experiments

| Experiment | Result / decision |
|---|---|
| Full query-load caching at D=256 | Exact, but ~25.86 ms versus ~13.47 ms PyTorch. Rejected. The D=128 win does not transfer to this width. |
| Partial query caching (1/2/4 fragments) and 32/64/128-row query tiles | Exact; no clear gain beyond direct output layout. Keep the existing 64-row geometry. |
| Q/K and P/V barriers | Exact; roughly 12.4–12.9 ms with direct layout. No material additional gain. |
| Disable forced P/V unrolling | Exact but ~16.54 ms. Rejected. This is a different loop from the accepted Q/K change. |
| Split output columns into two/four workgroups | Exact, but ~12.68 / 16.17 ms. Repeated Q/K work offsets the lower accumulator footprint. Rejected. |
| Apple8 attention variant | Different bytes and slower (~13.98 ms). Rejected. |
| INT8 workgroup ordering / compiler fences | Exact; small improvements at these shapes. Initial cold controls were misleading, so subsequent screens interleave warmed controls. |
| Alternate INT8 matrix tiles | Smaller tiles improve some projections by a few percent and hurt others. A combined short-workflow prototype saves about 0.06 s in two sampling steps, before the final attention change. Kept as an experiment; not part of the reported final implementation. Larger register tiles were substantially slower. |

These are measured screens, not a claim that all possible exact optimizations have been exhausted.

## Defaults and artifacts

The new route requires the validated **M5 Ultra / PyTorch 2.14 build**, native mode and fast math disabled. Attention source hashes must match the installed PyTorch headers. Other devices/builds, training, unsupported shapes and attention masks retain their preceding paths. The selected attention interface still handles third-party overrides.

```text
ASFP8_NATIVE=1
ASFP8_ATTN_BACKEND=pytorch
MTLFLASHATTN_SDPA=off

ASFP8_IDEOGRAM_EXACT=0       # disable all new Ideogram additions
ASFP8_IDEOGRAM_PREP=0        # disable fused Q/K normalization, RoPE and layout
ASFP8_IDEOGRAM_ATTN_LAYOUT=0 # disable D=256 output-layout / loop changes
```

The off switches are optional; normal native defaults enable the accepted changes. No new workflow node or relaxed mode is required.

The five runtime files are installed in `ComfyUI-AppleSilicon-2.14`; their hashes match the validated repository files. A fresh process using that installed node, ordinary defaults, and no candidate wrapper renders in **18.178 / 18.195 s** warm. Its first prompt, including model/text setup, takes **22.686 s** (process startup excluded). Each render executes all **816 preparation and 816 attention calls** on the new routes and preserves the reference tensors exactly. [Installed replay records](data/m5-ultra-ideogram4/installed-full-results.json) and [file hashes](data/m5-ultra-ideogram4/installation.json) are retained.

The tested caption is saved into the shared `ideagram.json`; its original is backed up under local `diagnostics/ideogram4/original-ideagram.json`. The optimized image is also saved as `ComfyUI-Shared/output/Ideogram4_M5Ultra_optimized.png`. **Restart the 2.14 ComfyUI installation to load the new code in its normal server.**

- [Matched workflow](data/m5-ultra-ideogram4/source.json) and [benign caption](data/m5-ultra-ideogram4/prompt.json).
- [Summary, samples and byte checks](data/m5-ultra-ideogram4/summary.json), [additional prompt/seed checks](data/m5-ultra-ideogram4/case-validation.json), and [test results](data/m5-ultra-ideogram4/tests.json).
- [Model headers and sizes](data/m5-ultra-ideogram4/models.json), [source provenance](data/m5-ultra-ideogram4/source-provenance.json), and [runtime manifest](data/m5-ultra-ideogram4/qk-full-runtime.json).
- [Full final run records](data/m5-ultra-ideogram4/qk-full-results.json), [stock records](data/m5-ultra-ideogram4/stock-full-results.json), and [short block verification](data/m5-ultra-ideogram4/qk-screen-results.json).
- [Attention scheduling experiments](data/m5-ultra-ideogram4/attention-qk.json), [INT8 tile screens](data/m5-ultra-ideogram4/int8-tiles.json), and [shared-source regression](data/m5-ultra-ideogram4/shared-source-regression.json).

Optimized output (identical to previous native):

![Botanical poster, optimized](data/m5-ultra-ideogram4/optimized.png)

[Stock comparison image](data/m5-ultra-ideogram4/stock.png).
