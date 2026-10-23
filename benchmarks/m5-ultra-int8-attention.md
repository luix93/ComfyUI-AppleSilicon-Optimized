# Opt-in INT8 attention — M5 Ultra, October 22, 2026

**A general dense INT8 attention kernel, selectable per workflow, saves 21–24%
end to end on long video workloads**: a 15-second H3 clip went from 799.65 to
604.97 s, and Wan 2.2 20-step I2V from 285.72 to 225.52 s. Every key is retained,
but arithmetic is quantized, so outputs change. It is never a default.
[Usage and limits](../docs/int8-attention.md).

M5 Ultra (80 GPU cores, 256 GB), macOS 27.0, PyTorch 2.14.1 (`5c488690`),
ComfyUI core `v0.42.0-19-g7a283320`. Authored public prompts only; no private
media. Isolated servers with 110 GiB RSS / 48 GiB available-memory guards. The
user's ComfyUI install was not modified.

## Why a new kernel

Hardware counters (workspace `diagnostics/int8-attention-profile-oct21`) measured matrix-unit busy time at a fixed 428 ms (exact BF16),
322 ms (INT8 QK / BF16 PV) and 214 ms (dense INT8) for H3 44.6k-token attention:
**INT8 runs at exactly 2× BF16** on M5 neural accelerators. The earlier dense
INT8 kernel reached only 61% matrix utilization. Its per-tile INT32→float PV
epilogue alone cost 13%.

The new kernel accumulates PV in INT32 across all key tiles and rescales in
place only for rows whose maximum rose. Probabilities use 255 levels (P−128)
with the offset restored from V prefix sums. V is mean-centered with one scale
per channel. Padded keys need no inner-loop mask. Inputs/outputs accept
ComfyUI's strided layouts, unequal query/key lengths, batches, BF16 and FP16.

Screened and rejected along the way (kernel-level, 44.6k tokens):

| Candidate | Result |
|---|---|
| 127-level P with per-channel V, midrange center | 2.3× the old kernel's error (mass lost to rounding biased toward an arbitrary center) |
| Normalizing by the quantized probability sum | Much worse for diffuse attention (0.345 vs 0.049 relative error) |
| Inner-loop tail mask | +80 ms from code generation alone; replaced by exact end-of-run correction |
| Per-simdgroup (instead of per-row) rescale | +60–100 ms |
| Headroom on raised maxima (0.5–2 log2) | ≤7% faster, 20–100% more error; left at 0 |
| 128-key tiles, 32-row queries, wider PV blocks | 1.1–1.8× slower |
| Lane-local deferred max/sum | Slower and incorrect |
| Overwrite-mode first QK (no zeroing) | No change |

The accepted configuration uses 64-deep QK contractions (2 per tile).

## v3: Kitchen numerics at v2 speed

`int8_dense_accurate` (backend node) and the **Dense INT8 v3** H3 presets add
the two numerical ideas of Comfy Kitchen's INT8 attention to the same fast
structure: randomized Hadamard Q/K rotation, and probabilities quantized against
each 64-key tile's own maximum with FP32 accumulation. Real H3 activations:

| Variant | Error · 15.5k | Error · 44.7k | Worst head · 44.7k | ms · 44.7k |
|---|---:|---:|---:|---:|
| v2 (running-max P) | 0.0171 | 0.0238 | 0.058 | 337 |
| Tiled P, no rotation | 0.0105 | 0.0105 | 0.014 | 365 |
| **v3: tiled P + rotation** | **0.0099** | **0.0100** | **0.015** | 364 |
| Tiled 127-level P + rotation | 0.0110 | 0.0110 | 0.017 | 344 |
| Kitchen v0.2.38 policy, Metal port | 0.0105 | 0.0103 | 0.016 | 1758 (untuned port) |

Per-tile probability scaling is the main factor; rotation adds a little.
v3's error no longer grows with sequence length.

Complete workflows at **Kitchen-style coverage** (every call, every step and
layer, prefix/text/audio/reference tokens included), frame RMSE against exact:

| Workflow | Exact | v3 all calls | Kitchen policy all calls | v2 all calls | v3 protected |
|---|---:|---:|---:|---:|---:|
| H3 5 s / 20 steps | 142.09 s | 123.40 s · 0.123 | port · 0.099 | 123.29 s · 0.195 | 132.19 s · 0.047 |
| H3 15 s / 20 steps | 799.65 s | **588.57 s** · 0.124 | port · 0.104 | — | 662.04 s · **0.017** |
| FastH3 OmniRef, image reference, 8 steps | 264.77 s | 208.90 s · **0.102** | port · 0.114 | 200.76 s · 0.205 | — |

v3 all calls runs at v2's end-to-end speed and lands at Kitchen-policy
difference levels: closer on OmniRef, slightly further on the H3 clips. v3
protected on the 15 s clip reaches 35.5 dB frame PSNR against exact.
[5 s](artifacts/int8-attention/v3-h3-5s-keyframes.jpg) ·
[15 s](artifacts/int8-attention/v3-h3-15s-keyframes.jpg) ·
[OmniRef](artifacts/int8-attention/v3-omniref-keyframes.jpg) keyframes.

## Kernel timings, synthetic inputs

Medians of 5 shuffled rounds, including quantization; errors are relative L2
against FP32 attention on sampled rows.

| Shape | INT8 v2 | Best exact | Speedup | v2 error | Earlier dense INT8 |
|---|---:|---:|---:|---:|---:|
| H3 15 s `[1,56,44600,128]` BF16 | 372 ms | 541 ms | 1.45× | 0.035 | 372 ms / 0.057 |
| H3 5 s `[1,56,15458,128]` | 51.1 ms | 64.2 ms | 1.26× | 0.028 | 52.4 ms / 0.045 |
| Wan 480p `[1,40,21504,128]` FP16 | 67.0 ms | 102.4 ms | 1.53× | 0.028 | — |
| LTX `[1,32,24576,128]` | 69.9 ms | 93.7 ms | 1.34× | 0.032 | — |
| Flux / Qwen ~4.5k tokens | 3.0–3.2 ms | 2.9–3.0 ms | none | 0.022 | — |

Short image-model sequences gain nothing, hence the 8192-key default gate.

## Real H3 activations

Captured attention inputs (20th eligible call, authored study-room prompt):

| Tokens | INT8 v2 | Earlier dense INT8 | INT8 QK / BF16 PV | BF16 SDPA |
|---|---:|---:|---:|---:|
| 15,549 | **0.0167** (worst head 0.040) | 0.0247 (0.060) | 0.0086 | 0.0017 |
| 44,691 | **0.0238** (worst head 0.058) | 0.0353 (0.086) | 0.0086 | 0.0017 |

About one third lower error than the earlier dense INT8 kernel at equal kernel
speed. INT8 QK / BF16 PV is more accurate but limited to 322 ms by BF16 PV.

## Complete workflows

Warm end to end, including sampling, decode and export; model loading cached.

### MiniMax H3, 864×480, 20 steps, `res_multistep`, seed 812764035

| Clip | Mode | E2E | vs exact | Frame RMSE | Audio rel. RMSE | Latent rel. L2 |
|---|---|---:|---:|---:|---:|---:|
| 5 s / 124 frames | Exact | 142.73 / 145.07 s | — | 0 | 0 | 0 |
| | **v2 full** | 127.56 / 131.27 s | −10.1% | 0.115 | 0.216 | 0.347 |
| | Earlier dense full | 132.23 s | −8.1% | 0.098 | 0.227 | 0.321 |
| | **v2 protected** | 135.65 s | −5.7% | 0.066 | 0.152 | 0.196 |
| | Earlier dense protected | 138.20 s | −4.0% | 0.075 | 0.150 | 0.234 |
| 15 s / 362 frames | Exact | 799.65 s | — | 0 | 0 | 0 |
| | **v2 full** | **604.97 s** | **−24.3%** | 0.124 | 0.952 | 0.445 |
| | **v2 protected** | 666.40 s | −16.7% | 0.046 | 0.308 | 0.154 |

Every v2 run reproduced its own output bytes. Call counts confirm routing:
1000 quantized calls in full mode, 630 in protected.
[5 s keyframes](artifacts/int8-attention/h3-5s-keyframes.jpg) ·
[15 s keyframes](artifacts/int8-attention/h3-15s-keyframes.jpg).

Visual assessment:
- **5 s:** all modes keep the room. Both v2 renders retain the exact render's
  ceiling lamp and door framing, which both earlier-kernel renders lose.
- **15 s, protected:** very close to exact in layout, window, light patches and
  clock.
- **15 s, full:** keeps the room and camera move, but replaces the right-hand
  window with white doors and drops the ceiling beams and floor light. Audio
  changes substantially in full mode.

### Wan 2.2 I2V 14B×2 INT8, 512×512 / 81 frames (Attention Backend node)

| Workflow | Exact | INT8 backend | Speedup | Frame RMSE | Frame cosine |
|---|---:|---:|---:|---:|---:|
| 4-step LoRAs, CFG 1 | 31.23 / 31.98 s | 26.72 / 26.48 s | 1.19× | 0.063 | 0.9942 |
| 20 steps, CFG 3.5 | 285.72 s | 225.52 s | 1.27× | 0.060 | 0.9945 |

Wan output is nearly indistinguishable from exact: same pose, motion and
composition, with small changes in glare and fur detail.
[Keyframes](artifacts/int8-attention/wan-keyframes.jpg).

## Coverage recipes: balanced, calibrated layers, reference rows

All rows use v3 numerics, seed 812764035, 864×480, 20 steps (OmniRef: 1376×768,
8 steps). Times are the sampling loop; PSNR is frame PSNR against exact.

| Recipe | Exact steps (of 20) | Exact layers | Quantized calls |
|---|---|---|---:|
| All calls | none | none | 1000 |
| Protected | first 4, last 2 | 0, 1, 33–35 | 630 |
| Balanced | first 2, last 1 | 0, 1, 33–35 | 765 |
| Balanced, calibrated layers | first 2, last 1 | 0, 1, 2, 3, 5 | 765 |

| Clip | Exact | All calls | Protected | Balanced | Calibrated layers |
|---|---:|---:|---:|---:|---:|
| Fox 5 s | 130.5 s | 115.5 s · 21.9 dB | 122.6 s · 23.9 dB | 121.4 s · 22.9 dB | 122.2 s · 20.5 dB |
| Skate 5 s | 131.5 s | 114.5 s · 19.0 dB | 122.9 s · 22.1 dB | 120.9 s · 17.4 dB | 123.0 s · 16.8 dB |
| Train 15 s | 756.6 s | 572.6 s · 10.1 dB | 640.8 s · 27.9 dB | 621.4 s · 24.4 dB | 631.7 s · 23.7 dB |
| Dance 15 s | 762.0 s | 574.1 s · 15.6 dB | 644.3 s · 18.7 dB | 631.9 s · 16.8 dB | 644.8 s · 15.9 dB |

- **Balanced** sits between protected and all calls. On the train it keeps the
  composition that all calls loses, and it saves 10–20 s over protected on
  15-second clips. On skate it scored below all calls: single-seed drift is
  noisy at this level.
- **Calibration.** At five sampler steps (skate) and two (fox), the model was
  re-run once per block with only that block's attention in INT8. The change in
  the prediction measures that block's sensitivity. Operator error is flat across
  blocks (about 0.009), but sensitivity falls steadily with depth: blocks 0–14
  score about 1.4–1.6× the mean and block 49 scores 0.23×. The ranking is the
  same across prompts and steps (Spearman 0.90–0.99). Blocks 33–35 rank 34th–36th.
  Per-step sensitivity peaks early: mean 0.042 at step 5 vs 0.009 at step 18.
- **Calibrated layers did not help.** Swapping 33–35 for the top-ranked blocks
  2, 3 and 5 was 0.7–1.6 dB lower on the motion clips and 1.2 dB higher on
  OmniRef. Sensitivity is spread across all 50 blocks: the best five hold
  about 15% of the total, and 0, 1, 33–35 hold about 10%. Steps matter more
  than which five layers stay exact.

**Reference rows, OmniRef image reference, 8 steps (balanced: exact steps 0
and 7, 270 quantized calls):**

| Rows quantized | Sampling | PSNR |
|---|---:|---:|
| Exact (reference) | 248.2 s | — |
| Balanced, video rows only | 233.2 s | 21.4 dB |
| Balanced, reference rows too | 232.5 s | 21.5 dB |
| Balanced, all rows (Kitchen-style) | 223.6 s | 22.0 dB |
| Balanced, calibrated layers | 234.7 s | 22.6 dB |
| Protected | 234.9 s | 24.3 dB |
| All calls (Attention Backend node) | 192.4 s | 19.9 dB |

Keys and values from references are quantized in every INT8 mode; these rows
change only which queries are quantized. Quantizing the reference or prefix
queries did not increase the difference from exact. This matches Kitchen
staying close without row protection.

**Preset-path cost on OmniRef (fixed).** Earlier per-call estimates (about
0.06 s saved per call through the presets vs 0.14 s through the Attention
Backend node) compared runs from different sessions. Exact OmniRef sampling
drifts from 248 s to 264 s between sessions on this machine, so only
same-session numbers are comparable.

In-model timing with GPU syncs around each 41k-token attention call, in one
session:

| Path | Per call |
|---|---:|
| Exact | 435–470 ms |
| Attention Backend node, INT8 v3 | 311–328 ms |
| Preset path | 353–356 ms |
| Preset path, all rows | 320–330 ms |

The one real overhead was that the preset path computed the 2,816 prefix rows
(text, reference image, audio: 7% of OmniRef's rows) in INT8, discarded them,
then recomputed them exactly. A single leading exact range is now excluded from
the INT8 call. Video rows are bit-identical and OmniRef renders are
byte-identical to before.

| OmniRef, 8 steps, one session | Sampling | Saved per quantized call |
|---|---:|---:|
| Exact (run before and after) | 264.0 / 262.0 s | — |
| All calls (Attention Backend node) | 202.8 s | 0.150 s |
| Balanced (presets, fixed) | 225.1 s | 0.140 s |
| Protected (presets, fixed) | 229.7 s | 0.148 s |

The H3 15 s prefix is 1,356 rows of 44,691 (3%), so the fix saves about
10 ms per call there.

## Scope and caveats

- One scene per model. The frame metrics measure difference from exact, not
  quality, and diffusion amplifies small numerical changes unpredictably.
- Measured on M5 Ultra only; the kernel requires an M5 GPU.
- The DGX Spark comparison in earlier discussion (user-supplied 1075 s for the
  15-second clip) is not a matched measurement here.
- Remaining headroom: the kernel reaches about 60% of the 214 ms matrix-unit
  floor at 44.6k tokens. The remaining cost is issue and ALU work around
  16-row matrix operations.

[Raw data](data/m5-ultra-int8-attention.json). Harnesses, captured activations
and full outputs are in the workspace's `diagnostics/int8-v2-oct22`.
