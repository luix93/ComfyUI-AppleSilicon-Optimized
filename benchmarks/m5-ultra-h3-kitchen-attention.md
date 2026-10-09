# H3 attention: Kitchen-inspired Metal experiments and temporal audit

October 8, 2026. M5 Ultra, **80 GPU cores**, 256 GB unified memory, macOS 27,
PyTorch 2.14 (`08187d9e0fba026dc8217405802ab5381dc88d90`), ComfyUI
`b33e2b55cae074eca5aec96283cceac19aa249ba`.

This pass compares optional attention against **our best exact optimized stack**,
not a fresh vanilla-Comfy baseline. The earlier stock/optimized table remains
unchanged. No private input media was used. An authored text-to-video scene
makes the background objects and camera movement inspectable.

## Controlled clip

- 124 frames at 24 fps, approximately 5.17 seconds; 20 steps, `res_multistep`,
  `simple`, seed `812764035`.
- 864×480: 15,549 packed tokens. 1344×768: 37,860 packed tokens.
- Same prompt, conditioning and initial noise within each size. Different sizes
  necessarily have different noise shapes and are compared against their own exact control.
- INT8 diffusion weights, original FP16 video VAE and FP32 audio VAE. Accepted
  exact projection, normalization, VAE and attention-query-cache improvements
  stay enabled. Relaxed VAE, Turbo LoRA, temporal caches and token reduction are off.
- Each attention path receives a warm-up before measured complete sampling,
  video/audio decode and file export. Kernel measurements include preprocessing,
  allocations and output layout. Model loading and unchanged text encoding are cached.
- An isolated server uses port 8193 and stops if the user's 8188 queue becomes
  busy. RSS limit 110 GiB, available-RAM floor 48 GiB, MPS watermarks 0.4/0.3.

Prompt:

> A continuous cinematic shot inside a quiet sunlit study. The camera slowly
> dollies forward toward a wooden desk, producing a gentle steady push-in with
> clear natural parallax. On the desk stand a single red ceramic vase and a
> small brass globe. In the background, a round white wall clock with dark
> hands hangs above a wooden bookshelf containing upright blue, green and red
> books. Three distinct rectangular framed landscape photographs hang in a
> straight horizontal row on the wall. A leafy potted plant stands beside the
> bookshelf. The square checkerboard floor and all furniture remain rigid and
> stationary throughout. No people, no object movement, no scene cuts. Natural
> daylight, realistic fine detail, stable geometry and consistent colors.
> Quiet room ambience, no music.

## Findings

### Native dense INT8 attention

Comfy Kitchen's [CUDA attention API](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/cuda/__init__.py)
provides full INT8 attention, chunked production and optional SOL token augmentation.
Those CUDA kernels cannot be called on MPS. Our implementation adapts the
existing BSD-licensed CCV-derived Metal path, retaining every key token in the
video-query attention loop. It uses INT8 QK and PV, FP32 running softmax/output
state, and exact PyTorch attention for the packed prefix queries.

**Retaining every connection does not make this bit-exact.** Q/K/V and softmax
weights are quantized; small softmax contributions can round to zero. Changes to video information can propagate through
later layers into both video and audio. Protected steps/layers reduce the
scope of the approximation, but do not guarantee preserved output quality.

Dense scheduling screens covered query blocks of 16/32/64/128, head-major
INT8 scratch, and periodic SIMD-group synchronization. A 64-row workgroup
with the existing long-sequence synchronization was best. Removing that
synchronization made the longer operator about 26% slower. The head-major
refinement retained the dense prototype's output bytes.

### Fewer SOL layout copies

The native layout route quantizes from head-major Q/K/V, stores INT8 scratch
head-major, and writes BHSD output directly. The arithmetic, route selection
and reduction order are unchanged. It is **byte-identical to our prior SOL**,
including complete sampler latents, decoded frame tensors and audio in the
480p and higher-resolution clips. SOL itself remains an approximate attention method.

On captured 15,458-token inputs, complete SOL attention decreased from
18.46 to 16.32 ms; on a **synthetically extended** 37,768-token input,
78.94 to 70.66 ms. These operator results are separate from the real 37,860-token
higher-resolution clip measurements below. `ASFP8_SOL_LAYOUT=legacy` provides
a diagnostic rollback to the previous physical layout.

The exact H3 stack already fuses INT8 QKV projection with RMSNorm and RoPE.
Kitchen's generic producer opportunity therefore overlaps existing work.
This pass improves the remaining layout/quantization path; it **does not**
implement a complete projection-to-INT8 chunked producer or prior-step V-scale reuse.

### Token rescue and cumulative-mass routing: screened, not new defaults

A bounded rescue prototype promotes 64 individual tokens from the commonly
omitted blocks shared by four query groups. Its residual tail excludes the
promoted tokens, avoiding double counting. A float64 reference verifies the
partition, full-rescue dense limit and constant-value property.

At the current Fast threshold, complete rescue attention reduced relative
operator error from 0.163 to 0.136 at the smaller captured shape, but increased
16.56 to 22.57 ms. At the longer synthetic shape it increased 71.19 to 113.26 ms.
More aggressive thresholds recovered runtime by sacrificing considerably more
accuracy: tau 3 reached 70.90 ms with relative error 0.380. The native token
selector was also slower than the bounded PyTorch alternative. **Rejected for
presets; no complete-clip speed claim is made for rescue.** The experiments remain
available under the local diagnostics folder.

A separate cumulative-proxy-mass router retained enough pooled-centroid
softmax mass to cover 85% of its proxy distribution. This is not a guarantee
about true attention mass. Its 480p clip took 119.73 seconds, approximately the
existing Balanced runtime, without a decisive quality improvement. The larger
clip took 402.77 seconds, again approximately Balanced, without a decisive quality advantage. It is not exposed as a preset.

## Timings and numerical comparisons

Warm complete-clip means, in seconds. `n` is the number of measured full runs.
Existing recipes were re-audited once per size, except Exact and Fast controls.
Existing Conservative/Balanced/Top-k audit timings use the frozen legacy layout;
operator byte parity covers their new layout too, but their new wall times are
not inferred from the Fast measurements. New modes repeat at both sizes. Dense 480p means use the final head-major public
recipe, not the slower earlier dense scratch layout. Previous prototype timings
remain in the JSON; matching output bytes do not justify averaging different
implementations. Baselines average matching exact controls at each size.

| Mode | 480 sampling | 480 total (n) | 1 MP sampling | 1 MP total (n) | Less total time, 480 / 1 MP |
|---|---:|---:|---:|---:|---:|
| Exact optimized | 131.98 | 143.20 (2) | 567.24 | 587.48 (2) | 0.0% / 0.0% |
| SOL Conservative | 121.85 | 133.06 (1) | 478.32 | 498.46 (1) | 7.1% / 15.2% |
| SOL Balanced | 108.19 | 119.33 (1) | 382.14 | 402.29 (1) | 16.7% / 31.5% |
| Previous SOL Fast | 103.22 | 114.40 (5) | 358.51 | 378.92 (2) | 20.1% / 35.5% |
| SOL Fast, new layout | 102.45 | 113.74 (4) | 352.46 | 372.87 (2) | 20.6% / 36.5% |
| Top-k Wide | 97.11 | 108.28 (1) | 312.02 | 332.20 (1) | 24.4% / 43.5% |
| Top-k All-step Draft | 84.14 | 95.35 (1) | 227.29 | 247.46 (1) | 33.4% / 57.9% |
| Veda Protected | 111.80 | 123.04 (1) | 358.53 | 378.83 (1) | 14.1% / 35.5% |
| Veda Draft | 98.89 | 110.15 (1) | 244.96 | 265.17 (1) | 23.1% / 54.9% |
| Dense INT8 Protected | 126.61 | 137.86 (2) | 472.88 | 493.25 (2) | 3.7% / 16.0% |
| Dense INT8 Full Draft | 122.40 | 133.58 (2) | 423.65 | 443.89 (2) | 6.7% / 24.4% |
| SOL Hybrid Fast | 101.83 | 113.04 (2) | 345.15 | 365.41 (2) | 21.1% / 37.8% |
| 85% proxy mass (rejected) | 108.55 | 119.72 (1) | 382.43 | 402.77 (1) | 16.4% / 31.4% |

The longer exact control varied from 581.61 to 593.36 seconds (about 2%).
Small improvements should be read with the repeated ranges, not as precise
hardware-independent gains. Matched old/new Fast runs and 480p repetitions supplement
the isolated operator measurements.

Video decoding stays around 10.5 seconds at 480p and 18.9 seconds at 1 MP;
audio decoding and export are included. These are attention-only comparisons
with cache and token reduction off, not claims about combining accelerators.

### Repeated-run ranges

| Mode | 480 total range | 1 MP total range |
|---|---:|---:|
| Exact optimized | 142.80–143.60 s | 581.61–593.36 s |
| Previous SOL Fast | 113.91–114.78 s | 375.05–382.80 s |
| SOL Fast, new layout | 113.63–113.86 s | 370.07–375.67 s |
| Dense INT8 Protected | 137.22–138.49 s | 492.74–493.77 s |
| Dense INT8 Full Draft | 133.47–133.69 s | 443.40–444.37 s |
| SOL Hybrid Fast | 112.87–113.20 s | 364.50–366.32 s |

A separate same-server alternating legacy/new/legacy/new public-layout test
measured **114.776 → 113.667 seconds**, saving
1.109 s (0.97%). All saved output bytes match.
The diagnostic rollback includes an extra wrapper budget check; this is a practical
whole-pipeline comparison, not an attribution of every millisecond to GPU copies.

### Complete-output changes

Frame RMSE is on decoded float tensors in [0,1]. Latent relative L2 concerns
the video and audio sampler outputs separately. These quantify change, not
quality or temporal fidelity. All non-exact recipes change video and audio
latents; exact prefix attention does not make final audio byte-identical.

| Mode | Frame RMSE 480 / 1 MP | Video latent relative L2 480 / 1 MP | Audio latent relative L2 480 / 1 MP |
|---|---:|---:|---:|
| Exact optimized | 0.00000 / 0.00000 | 0.00000 / 0.00000 | 0.00000 / 0.00000 |
| SOL Conservative | 0.02450 / 0.02205 | 0.08298 / 0.09538 | 0.06952 / 0.05137 |
| SOL Balanced | 0.12879 / 0.08061 | 0.39924 / 0.25495 | 0.16131 / 0.17485 |
| Previous SOL Fast | 0.13451 / 0.09786 | 0.42650 / 0.30754 | 0.23600 / 0.24602 |
| SOL Fast, new layout | 0.13451 / 0.09786 | 0.42650 / 0.30754 | 0.23600 / 0.24602 |
| Top-k Wide | 0.14473 / 0.16609 | 0.45936 / 0.58567 | 0.23190 / 0.46967 |
| Top-k All-step Draft | 0.27131 / 0.22292 | 0.91387 / 0.71138 | 0.72890 / 0.86874 |
| Veda Protected | 0.13675 / 0.10700 | 0.43161 / 0.33882 | 0.34119 / 0.16943 |
| Veda Draft | 0.20961 / 0.23211 | 0.70932 / 0.73842 | 0.38800 / 0.76818 |
| Dense INT8 Protected | 0.07472 / 0.05602 | 0.23387 / 0.16457 | 0.15523 / 0.09129 |
| Dense INT8 Full Draft | 0.09776 / 0.27004 | 0.32141 / 0.83417 | 0.19686 / 0.82369 |
| SOL Hybrid Fast | 0.13540 / 0.10000 | 0.42876 / 0.31232 | 0.25527 / 0.25203 |
| 85% proxy mass (rejected) | 0.12316 / 0.08926 | 0.38234 / 0.28000 | 0.14811 / 0.19306 |

## Temporal assessment: every attention preset

The authored study deliberately includes a clock, colored books, framed
pictures, a plant, furniture and a checkerboard floor during a slow dolly-in.
Assessment distinguishes a changed scene from within-clip morphing. All 124
frames were saved, full-frame strips were inspected, and manually initialized
clock crops follow detail through the sequence without warping the pixels.
Black crop padding marks the clock leaving the frame naturally.

| Preset | 480p inspection | Higher-resolution inspection |
|---|---|---|
| Exact optimized | Reference room and object shapes remain stable during the push-in. Fine generated textures are not photographic ground truth. | Reference clock, bookshelf, framed pictures and floor remain stable. Feature tracking estimates approximately 20% camera magnification. |
| SOL Conservative | Closest matched appearance among existing approximate recipes; no gross background replacement observed. Small fine-detail changes. | Closest matched room layout, color and camera path; no gross background replacement observed. |
| SOL Balanced | Globe/desk and plant detail differ from exact, while rigid background objects persist through the clip. | Small plant, vase and shelf-detail changes; major background objects persist. |
| Previous SOL Fast | Similar temporal stability in this scene; changed foreground placement and fine background detail. | Background object identity and camera path are stable; shelf/vase/foliage differ from exact. |
| Top-k Wide | No severe morphing reproduced in this authored scene; larger appearance changes. Prior user-reported background morphing remains a warning. | Bookshelf shape/size, clock rendering and plant differ substantially from exact. No gross transient replacement observed in this scene; prior morphing warning remains. |
| Top-k All-step Draft | Substantial room composition and lighting changes, including more frontal framing. Draft only, regardless of stable object persistence. | Different desk/bookcase/plant proportions and lighting; room framing differs substantially. Fine-detail stability is not established generally. |
| Veda Protected | Chair/material and fine-detail changes from exact; major background objects persist. | Main room layout and camera path retained; plant/clock/bookshelf detail changes. Preview predictor chooses an approximate geometry plan. |
| Veda Draft | Substantial layout changes, including a moved plant; changed appearance and camera path. | Different plant, bookcase, desk and framing, and stronger magnification than exact (~29% vs ~20%). Draft only. |
| Dense INT8 Protected | Room layout closest to exact among the new candidates, with changed door/vase/fine texture details. No gross background replacement observed. | Main exact room composition and camera path retained, with longer/narrower plant leaves and changed vase/clock/texture details. No gross transient object replacement observed; fine generated markings vary. |
| Dense INT8 Full Draft | All token connections retained, but desk/vase/perspective/fine detail change more than protected. No gross background replacement observed. | Substantial composition change from exact: window switches sides, clock/desk/bookcase/plant rendering differs. Objects persist through the clip, but this is a draft-quality tradeoff; small operator error did not predict preserved composition. |
| SOL Hybrid Fast | Similar to SOL Fast with additional changed details. No gross background replacement observed in this scene. | Close to SOL Fast in room geometry and camera path. Plant/desk/books and clock markings differ from exact; no gross transient object replacement observed in this scene. |

**No abrupt gross background replacement was reproduced in this study. This
is not a no-morphing guarantee.** Fine clock markings are malformed or fluctuate
in exact and approximate output. Top-k Wide retains the user-reported morphing
warning. Dense Full changes window placement and reduces three framed pictures
to two at 1 MP; retaining all connections did not preserve composition.

Adjacent-frame feature tracking supplies warning signals, not a quality score.
Perspective/parallax, smoothing or a changed scene can lower affine residuals
without improving quality. At 1 MP, estimated camera magnification is about
20% for Exact/Fast/Hybrid, 18% for Wide and 29% for Veda Draft. This changed
camera path is a real temporal difference even without an object disappearing.

[480p first/middle/last comparison](artifacts/h3-kitchen/all-attention-480.jpg) ·
[1 MP first/middle/last comparison](artifacts/h3-kitchen/all-attention-1mp.jpg) ·
[480p clock detail](artifacts/h3-kitchen/clock-detail-480.jpg) ·
[1 MP clock detail](artifacts/h3-kitchen/clock-detail-1mp.jpg).
The complete synchronized video player and every-frame sheets are in the local
`diagnostics/h3-kitchen/comparisons.html`; run `serve_player.py` for byte-range
video playback at `http://127.0.0.1:8194/comparisons.html`.

### Additional prompts/seeds

Two additional authored 22-frame scenes use seeds 812764036/37: a greenhouse
with pots, watering cans, a toolbox, clock and rigid roof supports, and a toy
store with cars, boxes, robots, framed pictures and checkerboard floor. Each
new mode uses identical noise/conditioning to its scene-specific exact control,
20 complete sampling steps, and a two-step warm-up. The 8192-token gate is
explicitly lowered for these diagnostic runs; ordinary short clips retain
native attention by default. They are quality stress checks, not short-clip
speedup claims.

| Scene | Exact | Dense Protected | Dense Full Draft | Hybrid Fast |
|---|---:|---:|---:|---:|
| Greenhouse | 18.66 s | 18.74 s | 18.76 s | 18.06 s |
| Toy store | 18.99 s | 19.00 s | 19.04 s | 18.34 s |

The greenhouse Protected output retains the reference roof, central clock,
walkway and pot arrangement, with small background ornament and foliage
changes. Full Draft changes roof proportions, framing and pot arrangement.
Hybrid retains rigid roof supports and a continuous push-in, while changing
fine foliage and background contents. No abrupt object replacement was
observed across these 22-frame outputs.

In the toy store, Protected retains the aisle, shelf geometry, cars and rear
pictures most closely. Full Draft changes shelf contents, car shapes and floor
perspective. Hybrid visibly changes the yellow robots and shelf contents from
exact, but those objects persist during the short push-in. Tiny labels and toy
textures fluctuate in the reference too; they are not evidence of preserved
fine detail in an approximate mode. Feature tracking flags parallax-heavy
pairs in the exact toy clip as well as its variants, so those flags alone
cannot establish morphing. These sub-second clips do not replace the full
5.17-second temporal tests or qualify general stability.


## Acceptance, safety and reproduction

- 59 complete measured samplings, plus excluded warm-ups, are
  recorded in the [machine-readable results](data/m5-ultra-h3-kitchen-attention.json).
- GPU checks cover ragged lengths, both physical layouts, exact prefix rows,
  finite output and constant values. Frozen research and public-recipe outputs
  match at the sampler latents, decoded frames and audio bytes. Complete SOL
  layout comparisons match the prior SOL at both clip sizes.
- Relevant Python tests: 468 passed / 35 skipped; frontend tests: 18 passed.
  Skipped GPU pytest cases are supplemented by the explicit native-MPS checks
  and complete-clip validations in the result JSON.
- Highest measured driver allocation: 66.61 GiB; highest server RSS:
  16.13 GiB. Driver/RSS overlap on unified memory and must not be added.
  No OOM occurred. Auto/manual workspace ceilings remain; transient budget
  rejection does not disable later calls that fit.

Available optional recipes: **Dense INT8 · protected**, **Dense INT8 · full draft**,
and **SOL INT8 · hybrid fast**, in the Simple node and as advanced starting
points. At 20 steps, Protected keeps the first four/last two steps and layers
0, 1, 33, 34, 35 exact. Full Draft uses dense INT8 video queries throughout;
packed prefix queries remain exact. Hybrid retains exact attention outside the
14-step active window, uses SOL Fast in ordinary layers and dense INT8 in
protected layers inside that window. The head-major layout improvement becomes the normal SOL route. Its shader
files have a separate path, leaving the old wrapper/shader ABI intact for an
already-running ComfyUI process. Restart ComfyUI and refresh the browser after
installation to load the new wrapper and preset choices together. Exact
remains available, and cache/token/attention defaults are unchanged. Full Draft
is an experimental composition tradeoff, not a quality-preserving mode. The
hybrid offers a small additional gain over Fast, not another 30–40% improvement.
Conservative remains the closest visual match in this study; Fast is the
practical speed/fidelity starting point, especially after the prior Wide report.

The [480p workflow](../workflows/h3_attention_quality_480.api.json) and
[1 MP workflow](../workflows/h3_attention_quality_1mp.api.json) preserve the
prompt, seed, frame count, steps and sampler. Both start with cache/attention/
tokens Off; select only the attention recipe being assessed. They use the
public INT8 H3 checkpoint and original FP16 video/FP32 audio VAEs. Complete
noise/conditioning/latents/frames, runtime/source hashes, scripts and guard
logs remain under `diagnostics/h3-kitchen`. Token rescue, mass routing and
schedule experiments remain available there without being installed as presets.

The study does not qualify RefMod/I2V inputs, people, fast camera motion,
4/6/8-step Turbo, or arbitrary cache/token combinations. A full projection-to-
INT8 chunked producer, prior-step V-scale reuse and larger token-rescue budgets
remain unimplemented/unqualified; no speed claim is made for them. The current
64-token rescue and native selector failed screening, while mass routing failed
to improve the speed/fidelity frontier. Further work would need a cheaper
selector or a quantization design that better preserves complete outputs.
