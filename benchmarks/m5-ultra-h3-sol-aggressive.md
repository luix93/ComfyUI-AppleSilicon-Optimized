# M5 Ultra: 14-step H3 SOL, cache and token reduction

The previous optional seven-step SOL configuration saved about 10 seconds on the five-second workflow. This follow-up tests exactly **14 of 20 steps**, keeping the first four and final two dense, then tests the node's cache and token-reduction mechanisms separately and together. All timings are against the **already optimized dense PyTorch 2.14 reference**, not stock ComfyUI.

## Test configuration

M5 Ultra, 80 GPU cores, 256 GB unified memory; PyTorch 2.14; 864×480, 124 frames at 24 fps, 20 steps, `res_multistep` / `simple`. The existing INT8 convrot H3 diffusion checkpoint, FP16 video VAE and FP32 audio VAE remain unchanged. No LoRA, legacy relaxed environment preset, or approximations other than those named in each row. Fixed red-paper-boat prompt, seed 812764035, identical saved initial noise and conditioning. The 22-frame compatibility and additional-prompt runs keep the same spatial resolution and step count.

All measured executions include sampling, video/audio VAE decoding and export. Model loading and unchanged prompt encoding are warm. First executions are excluded. The combination series uses a four-step full-resolution warmup before its 20-step measurements; warmup outputs are not compared with 20-step outputs. CPU quality measurements run outside the repeated performance measurements. Brief single-thread contact-sheet extraction was used during screening only.

The isolated diagnostic server checks the user's queue before launch, enforces a 110 GiB process-RSS watchdog and 48 GiB available-memory floor, and sets the MPS high-watermark ratio to 0.4. Process RSS is not total unified GPU memory; sampled Metal driver allocation is reported separately. No models or user workflows are overwritten.

## Compatibility defect and fix

FirstBlockCache immediately reproduced:

```
TypeError: _install_block_fusion.<locals>.fast_forward() got an unexpected keyword argument 'attention'
```

Current ComfyUI passes `attention=None` through every DiT block-replacement call. The fused block signature omitted this argument. Ordinary sampling did not take that dispatcher path, while cache/token/block-oracle did. The wrapper now accepts the keyword, preserves its ordinary fusion arithmetic, and honors actual custom attention functions through upstream.

Both FirstBlockCache variants, token reduction, the composed cache/token/INT8-SOL route, the old combined BF16 defaults, and both oracle modes subsequently completed. On 22 frames / 20 steps, the exact control took 19.07 s; cumulative FirstBlockCache at 0.20 took 11.66 s (9 reused evaluations), per-step FirstBlockCache took 10.80 s (10 reused), and token reduction took 14.66 s. These are compatibility/screening observations, not final full-clip recommendations. The first call to the BF16 default combination also paid setup overhead.

`oracle` and `block_oracle` are diagnostics, not speed modes. Both preserved sampler-latent, decoded-frame and decoded-audio bytes with other approximations disabled. An exact run after the approximate modes also matched. Fresh full-clip exact results matched the reference saved before this work, including initial noise and conditioning.

## Selecting precisely 14 steps

A new optional **Window Schedule = step_fraction** derives boundaries from the actual sampler sigmas. Start/end 0.20/0.90 selects zero-based indices [4,18), i.e. steps 5–18. The tests confirm **630 sparse main-attention calls out of 1000**: 14 steps × 45 eligible layers. Layers 0,1,33,34,35 stay dense. Prefix queries retain exact attention. Video-query attention still uses quantization and summaries, so indirect changes to both streams remain possible.

Existing workflows retain `sigma_percent`. Model diffusion percentages and sampler-step fractions are not interchangeable for every schedule; float32 values at an inclusive sigma boundary can also shift the selected steps. The new mode groups repeated CFG evaluations by sigma rather than counting each model call as a step. Nonintegral boundaries round toward additional protected steps.

## Whole-clip results

| Configuration | Warm runs | Sampling mean | End-to-end mean (range) | Time saved vs 144.80 s |
|---|---:|---:|---:|---:|
| SOL14, tau 0.25 | 1 | 110.93 s | **122.28 s** (single run) | 15.6% |
| SOL14, tau 0.5 | 1 | 108.92 s | **120.18 s** (single run) | 17.0% |
| SOL14, tau 1.0 | 1 | 105.52 s | **116.80 s** (single run) | 19.3% |
| SOL14, tau 1.3 | 2 | 103.87 s | **115.23 s** (115.06–115.40) | 20.4% |
| SOL14 tau 1.3 + FBC 0.08 | 2 | 95.82 s | **107.27 s** (106.72–107.81) | 25.9% |
| SOL14 tau 0.5 + FBC 0.10 | 2 | 86.95 s | **98.44 s** (98.43–98.45) | 32.0% |
| FirstBlockCache 0.20 only | 2 | 76.45 s | **87.83 s** (87.22–88.43) | 39.3% |
| Token reduction only | 1 | 87.47 s | **98.91 s** (single run) | 31.7% |
| SOL14 + FBC 0.08 + token reduction | 1 | 67.52 s | **79.08 s** (single run) | 45.4% |

The initial warmed exact controls were **145.116 and 144.490 s** (mean **144.803 s**). Later exact controls were **150.839 s**, **149.258 s**. Savings above deliberately use the faster initial mean. All exact outputs remained byte-identical; a fresh final process recorded all **980 cross-block fusions**. No fused path latched off. The later timing drift is not attributed to a specific cause.

The moderate hybrid reuses five tails while the low-threshold 0.08 combination reuses two. Cache 0.20 alone reuses nine. Cache combinations at thresholds 0.08/0.10 use the cumulative policy, window **0.25–0.85**, maximum **one** consecutive skip, and audio weight **1.0**. Cache 0.20 alone uses its existing **0.15–0.95** window and maximum **two** consecutive skips. All 20 sampler steps remain; a reused evaluation still computes block 0. The SOL window is eligible on 14 steps, but a cached tail omits its attention calls. Token reduction affects all evaluations, including the steps whose sparse attention is disabled.

The strongest all-three result is retained as experimental. It has larger composition changes and is not promoted as a quality-preserving default. The same applies to treating a single timing as a repeatable performance claim.


## Quality assessment

All saved outputs are finite. Numerical error, frame PSNR, RGB shifts and temporal residuals are diagnostics rather than quality guarantees. Pixel-aligned error penalizes changes in geometry and timing, even when both videos look plausible. Conversely, a small error does not guarantee preserved detail or temporal stability.

Eight positions spanning each five-second clip were inspected as frame sequences. The wider SOL results retain crisp paper edges and a coherent boat/ripple progression, but boat geometry, reflected highlights, ripple placement and timing change. The 22-frame boat scene shifts composition more strongly than the five-second scene. These are visible changes, not merely different tensor bytes. There is no dialogue or lip-sync validation in this set.

| Five-second setting | Video-latent relative L2 | Frame PSNR |
|---|---:|---:|
| SOL14, tau 0.25 | 0.2255 | 22.85 dB |
| SOL14, tau 0.5 | 0.2576 | 21.54 dB |
| SOL14, tau 1.3 | 0.4133 | 17.78 dB |
| SOL14 tau 0.5 + FBC 0.10 | 0.2548 | 21.55 dB |
| FirstBlockCache 0.20 only | 0.1485 | 25.94 dB |
| Token reduction only | 0.6375 | 13.79 dB |
| SOL14 + FBC 0.08 + token reduction | 0.6358 | 14.03 dB |

Two additional 22-frame prompts/seeds checked a blue teacup on fine fabric (431927) and a walking woman in a green jacket (957130). They explicitly forced sparse routing below the normal 8192-token guard to test the numerical behavior. Those short-clip timings are not evidence of useful sparse acceleration at short sequence lengths.

The sampled frame sequences retain sharp, recognizable objects and coherent progression, with no obvious catastrophic corruption. They also show real differences: cloth patterns, cup outline, steam shape, facial details, pose, head-turn timing and background placement. Cache alone has a better numerical fidelity/speed tradeoff on the boat and texture cases, while the moderate SOL/cache combination has lower aligned error on the motion case. There is no universally best quality setting across these examples. Token reduction and the all-three combination change framing, identity and composition most strongly. These limited visual checks do **not** establish equivalent perceptual quality across other prompts, longer clips, I2V or dialogue.

- [Five-second exact vs cache](artifacts/h3-sol-aggressive/boat-cache.jpg)
- [Five-second exact vs SOL14](artifacts/h3-sol-aggressive/boat-sol14.jpg)
- [Five-second exact vs moderate hybrid](artifacts/h3-sol-aggressive/boat-hybrid.jpg)
- [Texture: exact vs cache](artifacts/h3-sol-aggressive/texture-cache.jpg)
- [Motion: exact vs cache](artifacts/h3-sol-aggressive/motion-cache.jpg)
- [Motion: exact vs moderate hybrid](artifacts/h3-sol-aggressive/motion-hybrid.jpg)
- [Motion: exact vs all three](artifacts/h3-sol-aggressive/motion-all.jpg)

Silent side-by-side videos put the exact reference on the left:

- [Five-second boat vs FirstBlockCache](artifacts/h3-sol-aggressive/boat-cache-silent.mp4)
- [Five-second boat vs moderate SOL/cache](artifacts/h3-sol-aggressive/boat-hybrid-silent.mp4)
- [Walking subject vs FirstBlockCache](artifacts/h3-sol-aggressive/motion-cache-silent.mp4)
- [Walking subject vs moderate SOL/cache](artifacts/h3-sol-aggressive/motion-hybrid-silent.mp4)


## Reproduction and controls

Use [the H3 acceleration controls guide](../docs/h3-acceleration.md). The exact path remains available by disabling all three approximation sections. The older seven-step SOL workflow is retained. Examples and per-run machine-readable measurements accompany this report. Every named combination remains optional; token reduction and cache are not silently enabled by selecting INT8 SOL.

**Validation:** 119 CPU tests passed; 41 GPU-only unit tests were skipped in that CPU invocation. Eight frontend tests passed. The real MPS workflow matrix exercises every exposed cache mode, token reduction and composed BF16/INT8 routes. All saved tensors were finite. Exact controls and repeated same-configuration outputs matched bytes. Sampled Metal driver allocation peaked at **58.31 GiB** in the full-clip tests; the memory guards did not fire. Process RSS was approximately 11 GiB and is not a substitute for the driver measurement.

[Machine-readable results, settings, hashes and errors](data/m5-ultra-h3-sol-aggressive.json). Example API workflows: [SOL14](../workflows/h3_sol14.api.json), [FirstBlockCache](../workflows/h3_first_block_cache.api.json), [moderate SOL14/cache](../workflows/h3_sol14_cache.api.json), [experimental all-three](../workflows/h3_all_approximate.api.json). Keep the supplied minimum-token guard for actual use.
