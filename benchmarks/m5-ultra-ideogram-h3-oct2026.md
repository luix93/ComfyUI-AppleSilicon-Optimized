# Ideogram 4 and MiniMax H3 kernel follow-up — October 10, 2026

M5 Ultra, 80 GPU cores, 256 GB; macOS 27.0; PyTorch 2.14.1 (`5c4886908584029761b579af026dcfb627c84070`); ComfyUI `v0.39.0-33-g1d2ea294`. Tests use the isolated **2.14-Update** instance. Loading/JIT runs are excluded; unchanged conditioning is cached. Complete-operator timing includes allocation, quantization, centering/rotation and layout work.

Only generated public fixtures and NVMe models were accessed. GPU jobs run serially, with MPS watermarks 0.4/0.3, a 110 GiB host-RSS cap and at least 48 GiB available memory. Host RSS does not include all Metal allocation; allocation snapshots are recorded separately. No private inputs, SATA files or raw checkpoint tensors are included in the published artifacts.

[Machine-readable results](data/m5-ultra-ideogram-h3-oct2026.json) contain measured runs, tensor hashes, operator samples and source hashes.

## Accepted exact Ideogram changes

The controlled workload is **1024×1024, 12 steps, dual-model CFG, INT8 ConvRot models, BF16 image VAE**, with a public botanical-poster prompt. Attention has **4414 tokens, 18 heads, D=256**. The previous optimized implementation is the comparison baseline; this is a follow-up to the October 9 stock-versus-pack test, not a new stock test.

Two narrowly selected changes compose usefully:

1. **128-channel attention output tiles.** Q/K still compute the complete D=256 dot product. The key tile, softmax and PV reduction order stay unchanged. Splitting the output channels halves the live output accumulator width; it duplicates Q/K work across the output tiles. Standalone attention improves **7.664 → 7.328 ms**. No hardware register/spill counters were collected, so reduced live state is an implementation fact, not proof of reduced spilling.
2. **Projection-specific INT8 scheduling.** QKV and MLP input projections use 64×128 output tiles; the output projection and MLP down projection retain 128×128 tiles. Both use linear tile traversal without the volatile compiler scheduling fence. Actual threadgroup barriers remain. INT32 accumulation and BF16 scale/round/bias operations stay unchanged.

The diagnostic combined implementation improves three paired warm complete renders from **18.451 → 17.716 s**: controls 18.442/18.468/18.442; candidates 17.609/17.590/17.948. All sampler latents and decoded frame tensor bytes match. The production implementation then passes a fresh paired benchmark:

| Production path | Warm E2E samples | Mean E2E | Mean sampling |
|---|---|---:|---:|
| Previous optimized tiles | 18.370 / 18.404 / 18.460 s | **18.411 s** | 18.163 s |
| New exact tiles | 17.624 / 17.612 / 18.097 s | **17.778 s** | 17.528 s |

This saves **0.634 s / 3.4% E2E**, or 3.5% sampling time. All eight runs, including warmups, have identical conditioning, noise, sampler latents and decoded frame tensor hashes. The slower third candidate is included in the average. Relative to the separately measured October 9 stock result (43.54 s), the table now reports **2.45×** speedup; stock was not remeasured in this follow-up.

Both accepted source files were installed into the Update instance after verifying its pre-change hashes matched our backups. A fresh installed-instance teapot fixture (seed 42, two steps) preserves sampler and decoded bytes in all four runs, with no attention/preparation fallback. Warm control/candidate E2E is 3.235/3.056 s; this is a compatibility check, not a substitute for the 12-step timing table. The production full render uses about 28.06 GiB allocated / 30.00 GiB driver memory at its recorded end-of-run snapshot, not a continuous peak measurement.

Eligibility is restricted to validated **4096–4608-token/row** shapes on the existing supported Ultra/runtime path. Other resolutions retain the previous schedule. Small-row modulation projections are unchanged. The Krea2 specialization is also unchanged.

Independent opt-outs restore the previous optimized paths:

```bash
ASFP8_IDEOGRAM_ATTN_TILE=0  # previous D=256 attention output tile
ASFP8_IDEOGRAM_GEMM_TILE=0  # previous INT8 matrix schedule
```

Two short prompt/seed runs verify **136 attention calls and 680 affected matrix calls each** against the preceding exact implementations, then compare full sampler and decoded tensors. Native tests also exercise random tensors, nonzero storage offsets, both attention layouts, bias/no-bias, negative INT8 values, zero scales, tile tails and selection boundaries. The native suite passes **71 tests**, including regression checks for Krea2.

Rejected exact candidates:

| Candidate | Finding |
|---|---|
| D=256 query tile 32 / 128 | 7.938 / 8.176 ms vs 7.664 ms current |
| 64-channel output tile | 11.561 ms; duplicates too much Q/K work |
| Linear matrix traversal alone | 18.334 → 18.346 s complete render; rejected |
| Blanket 64-row matrix policy | Hurts some projections; use per-projection selection |
| Larger K tiles / multiple INT32 partial accumulators | Mixed or substantially slower; no blanket adoption |

## Optional arithmetic research: Ideogram INT8 attention

A prototype derived from newer Metal quantized attention uses INT8 Q/K/P/V, FP32 accumulation, K centering, Hadamard Q/K rotation and 512-row K/V mean chunks. This is **non-bit-exact** and is not installed as a default or exposed as an accepted quality preset.

| Public fixture | Previous exact optimized | Experimental INT8 | Less total time |
|---|---:|---:|---:|
| Botanical poster, 12 steps, three paired warm runs | 18.260 s | 15.223 s | 16.6% |
| Blue teapot, different prompt and seed, one warm pair | 17.941 s | 15.050 s | 16.1% |

These comparisons precede the newly accepted exact tiles. Combining the relaxed attention with the new matrix policy has not been benchmarked; the savings cannot be added.

At one captured attention operation, centered/rotated INT8 has **2.48% relative RMSE** and cosine **0.99969** against exact. Through 12 denoising steps, final latent relative RMSE grows to **95.3% / 63.0%** in the two fixtures. Decoded frame RMSE on [0,1] is **0.133 / 0.0616**, with cosine **0.98736 / 0.99694**. Small operator errors do not establish unchanged image quality.

Visual review: the poster retains readable text and a coherent scene, but flower geometry, vase appearance, pear size/position and texture change. The teapot retains text and overall layout, but its proportions, reflections, cups and color change. There is no video motion test for this image model. These are visibly different renderings, not a quality-preservation claim.

![Exact and experimental botanical poster](artifacts/ideogram-h3-oct2026/poster-comparison.jpg)

![Exact and experimental teapot](artifacts/ideogram-h3-oct2026/teapot-comparison.jpg)

An initial diagnostic wrapper mistakenly returned logical BSHD instead of BHSD and produced striped images. That wrapper was corrected, and both complete experiments were rerun. The invalid first batch is excluded from timing and quality conclusions.

## MiniMax H3: current exact kernels remain preferable

The long fixture is **864×480, 124 frames, audio included**, using identical saved noise/conditioning. Main attention is **15458 tokens, 56 heads, D=128**: 14985 video tokens and a 473-token text/audio prefix. Short screens use 22 frames and 20 steps; longer dense-kernel screens use 124 frames and **two steps**, not the full 20-step quality/performance test.

The current exact long-fixture control measures **24.650 s E2E for two steps**, including about **10.27 s video decode** and **13.40 s sampling**. The October 9 full 20-step reference remains **141.65 s**; it was not rerun in this kernel screen and is not a new measurement.

| Exact experiment | Complete-operator result | Decision |
|---|---:|---|
| Query tile 32 | 63.560 vs 63.632 ms | Essentially tied; keep current 64 |
| Query tile 128 | 69.850 ms | Slower |
| Remove forced Q/K loop unroll | 152.445 ms | Much slower; D=256 findings do not transfer to D=128 |
| 64-row INT8 matrix tiles | Approximately 6–10% slower on captured output/down projections | Reject |
| SK32 matrix inner block | Approximately 3× slower | Keep SK16 |
| BK512 with SK16 | Down 12.041→11.901 ms; output 5.892→5.843 ms | Small isolated gain; no accepted workflow change |
| BK1024 / unroll4 | Slower | Reject |
| Two / four INT32 partial accumulators | Down 14.598 / 100.475 ms vs 12.041 ms | Reject |
| Remove actual threadgroup synchronization | Down 17.399 ms | Reject |
| VAE attention query cache | Helps batch1, hurts batch4; tiny/noisy differences | No complete-decoder gain established; retain VAE |

The matrix screen covers the direct output and MLP-down matrix calls. H3's QKV and fused FC1/SwiGLU bridges are separate native paths; these measurements do not establish that every matrix kernel is optimal.

The startup `final-rms-mod-cast skipped ... missing ['norm']` message is intentional for current ComfyUI's nested/PDD final-layer semantics. This wrapper was already adapted and byte-checked in [sampling sprint 2](m5-ultra-sampling-sprint2.md), but complete sampling gains were within variance. It is not a newly discovered performance regression.

## H3 optional dense / SOL experiments

The newer dense INT8 implementation is faster in isolation on the long attention geometry: **51.827 → 47.616 ms**, including quantization, means, layout conversion and allocation. Relative operator error is **1.45%** with cosine **0.999895**; the old dense implementation is 1.54%.

The protected integration retains exact prefix attention and protected transformer layers. For the two-step long screen only, the usual first/last step protection is disabled so both tested steps actually use the candidate; **90 calls per run** take the tested backend. This differs from the ordinary 20-step preset window.

| Matched integrated test | Existing dense INT8 | New dense INT8 |
|---|---:|---:|
| 22 frames / 20 steps, E2E | 19.103 s | 19.130 s |
| 124 frames / 2 steps, sampling | 12.446 s | 12.202 s |
| 124 frames / 2 steps, E2E | 23.718 s | 23.591 s |

The short screen temporarily bypasses the usual 8192-token eligibility threshold solely to check compatibility; the normal threshold is retained in the pack. The long integrated gain is only **2.0% sampling / 0.5% E2E**, with overlapping E2E samples. This does not justify replacing the default exact kernel or stable SOL presets. No full 20-step motion/temporal quality qualification was performed for the new dense kernel.

Changing SOL's FMA/rescaling operand order also fails the short workflow screen: **18.352 → 18.361 s**, with 630 sparse calls in each run. It is rejected. The stable SOL INT8 Fast/Hybrid presets and Top-k Wide morphing warning remain unchanged.

There is still room to investigate QKV/FC1 bridge scheduling, large-sequence attention and VAE matrix/fusion paths. The tested changes do not support an additional large exact speedup. Current SOL Hybrid already has meaningful optional gains in the [separate full-resolution camera-scene comparison](m5-ultra-torch2141-retest.md); those are distinct from this dense-kernel screen.

## Upstream evidence

[Draw Things 26.1008.0 release notes](https://drawthings.ai/downloads/) list M5 Ultra and quantized-attention changes. The tested ideas were read from the project's pinned/open CCV Metal sources:

- [D>192 output tiling](https://github.com/liuliu/ccv/commit/0a4c93f) and [head-major layout](https://github.com/liuliu/ccv/commit/16ec14d).
- [FMA rescaling](https://github.com/liuliu/ccv/commit/8d5a3cf), [Hadamard Q/K](https://github.com/liuliu/ccv/commit/36c168ed) and [K centering](https://github.com/liuliu/ccv/commit/3af6541d).
- [Latest R1 attention work](https://github.com/liuliu/ccv/commit/6510c96) targets one/two-query autoregressive attention; it does not directly cover H3's full self-attention or its VAE's 1797-query attention.

Matching an implementation idea does not imply matching Draw Things' end-to-end timing: model representation, graph and workflow still matter. The newer attention uses relaxed arithmetic, whereas the accepted Ideogram output split retains PyTorch's arithmetic.
