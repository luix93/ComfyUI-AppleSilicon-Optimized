# Krea2 Turbo: exact M5 Ultra sampling improvements

2026-09-24. M5 Ultra, 256 GB unified memory, macOS 27, PyTorch 2.13.0
(`cf30153c4c131c8164ee7798e5022d810682e2cb`), ComfyUI 0.37.0.

## Reference workflow

Extracted from `Krea2_turbo_00019_.png`, including its executable prompt:

- 1448×1448, batch 1, eight Euler steps, simple scheduler, CFG 1, denoise 1.
- `Krea2/krea2_turbo_int8_convrot.safetensors`.
- `Krea2/snofs_krea_v1_3D.safetensors`, strength 0.8. This file contains LoKr
  adapters. ComfyUI merges them into the quantized weights; all eight major
  block projections still use `TensorWiseINT8Layout`, with no per-call
  weight functions or fallback to dense floating-point projections.
- `qwen3vl_4b_fp8_scaled.safetensors`, Krea2 text-encoder type.
- `qwen_image_vae.safetensors`, default BF16 VAE on this setup.
- Prompt expansion disabled. The effective prompt includes the trailing
  comma/space from the workflow's empty trigger-word concatenation.
- Seed `535122089142682`.

The baseline is the existing native accelerated implementation, including
W8A8 INT8 matmuls and native GQA attention, not unpatched ComfyUI. The replay
matches **every pixel of the user's original PNG**. Model, LoKr, prompt,
resolution, seed, steps, sampler and VAE are preserved throughout.

## Where the time goes

The first warm replay took 18.508 s: 18.076 s sampling, 0.363 s decode,
0.065 s PNG saving. Text encoding and model loading are outside this warm
measurement, matching ComfyUI's reuse of those unchanged nodes. GPU work is
synchronized at the sampling/decode boundaries; iteration-bar submission
times underreport sampling completion.

A separate two-step synchronized profile (not a benchmark timing) measured
4.968 s sampling. Nested buckets include 2.467 s INT8 projection calls,
of which 0.201 s is activation rotation/quantization and 2.262 s is the
GEMM/epilogue; attention itself is 1.598 s. The remaining cost includes
normalization, gating, residuals, layout conversion and model glue.
VAE decode is already too small to offer a meaningful end-to-end gain here.

## Accepted changes

1. **Normalization/modulation:** read BF16 inputs directly, follow the tested
   PyTorch FP32 RMS reduction and weighting order, preserve the BF16 norm
   store, then the separate BF16 scale multiplication and shift addition.
   This removes FP32 input/output materializations and separate memory passes.
   Captured 8306×6144 block inputs measured about 2.86 ms → 0.30 ms for this
   complete chain. Block/final-layer normalization is guarded to the original
   classes, no overridden norm hooks, batch 1 and no timestep-zero references.
2. **Elementwise/layout fusions:** retain rounding after the gated product
   before residual addition; retain BF16 SiLU rounding before multiplication;
   fuse BF16 sigmoid gating with the attention BHS-D → BS-HD layout copy.
   The original projections and their LoKr-modified INT8 weights remain.
3. **GQA attention scheduling:** reuse the synchronized Morton traversal
   that improved Ultra H3, retaining Krea2's dynamic dimensions, score tiles,
   softmax arithmetic and reduction order. Captured attention improved from
   about 28.63 ms to 21.80 ms (1.319× throughput, about 24% less time), with
   identical bytes. Full query tiles are padded so inactive groups safely
   participate in synchronization. Dispatch is restricted to M5 Ultra,
   BF16, B=1, 48 query / 12 KV heads, D=128, and 8192–12287 tokens.

The normalization fusions are restricted to the tested M5 Ultra / PyTorch
2.13 combination. Other devices/releases retain the previous path.
The attention wrapper also now honors current ComfyUI `attn1_patch` and
`attn1_output_patch` hooks by returning those calls to the original forward.

Default under the existing native/Krea2 flags. Independent switches:

```bash
ASFP8_KREA2_GLUE=0
ASFP8_KREA2_ATTN_SCHEDULE=0
```

Set both before startup to restore the previous accelerated implementation.
These are exact improvements relative to that implementation; they are not
new relaxed-precision modes.

## Interleaved complete-workflow acceptance

One process, same loaded models and conditioning, first baseline excluded
as model-load warmup. Allocation, rotation, quantization, layout conversion,
GPU completion, VAE and PNG saving are included. Interpreter/JIT startup,
post-run tensor hashing and diagnostic tensor saves are excluded.

| Warm run | Sampling | Complete workflow |
|---|---:|---:|
| Baseline A | 18.056 s | 18.487 s |
| Optimized A | 14.665 s | 15.102 s |
| Optimized B | 14.669 s | 15.083 s |
| Baseline B | 18.094 s | 18.510 s |
| Mean | **18.075 → 14.667 s** | **18.498 → 15.092 s** |

**3.406 s saved, 18.4% less total time, about 1.23× throughput.** Fusions
alone measured 16.559 s warm before adding the attention schedule.

A final replay from the installed custom-node directory measured 18.779 s
for the first render and **15.081 s warm** (14.644 s sampling, 0.368 s
decode, 0.064 s saving). Both runs preserved all three hashes below and
matched the original PNG pixels. Interpreter/native compilation startup
is outside these render timings.

All runs, including warmup, have the same bytes:

| Output | SHA-256 |
|---|---|
| Sampler latents | `b8f7c6a25f9bb278fda696d9291648937b168b822b7d7a3f337a090dd9ce2452` |
| Decoded tensor | `4c233cb7ad07a74db68ec6e08756c65d4cf2e176e58c33c8b998a2a3a92874b1` |
| RGB pixels | `0c37173257b88e4d22f2abbeb77aeaaa2b77354ba167cfb8df28ef9b081ac9cb` |

The RGB pixels also match `Krea2_turbo_00019_.png`. PNG file bytes are not
compared because the diagnostic outputs omit ComfyUI workflow metadata.

## Numerical and compatibility checks

- 24 focused tests passed, including FP32 RMS reduction/BF16 rounding at
  several widths/seeds, all finite BF16 sigmoid/SiLU inputs, ragged attention
  layout, residual product rounding, norm-hook fallbacks and attention hooks.
- A two-step real-model diagnostic compared each fused normalization,
  modulation, residual, SwiGLU and attention-tail operation to its original
  operation; complete latents and decoded tensors matched the baseline.
- Two additional prompts/seeds passed paired short two-step runs with
  byte-identical latents, decoded tensors and RGB pixels; their results
  are recorded in the JSON. The full eight-step comparison
  above uses the original image workflow.
- The final attention extension was compared to a frozen pre-change build
  at Krea2 lengths 1024, 8192 and 8351 with its new flag both on and off,
  plus H3's 56-head/15787-token shape. All eight comparisons are byte-exact.

## Screened but not promoted

- Attention groups of 1/2/4/8 versus 16 preserved bytes but were slower.
- Shared-memory probability tiles instead of the existing register path
  preserved bytes but were slower.
- INT8 swizzle 0/3/4, K-block 128/512, and the previous compiler-barrier
  schedule were tested on all four dominant Krea2 matrix shapes. Most were
  neutral or slower. Swizzle 0 helped the narrow K/V projection by about 9%,
  but the larger projections improved only around 1–2%; this was not promoted
  without a demonstrated complete-workflow benefit.

Further substantial gains would need improvements to the INT8 projection
or attention kernels. Combining compatible Q/K/V/gate projections and
extending exact projection epilogues are reasonable next experiments.
They are not implemented or assigned an estimated end-to-end gain here.

Machine-readable results: [JSON](data/m5-ultra-krea2-exact.json).
