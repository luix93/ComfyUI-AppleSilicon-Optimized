# PyTorch 2.15.0 on the dev instance

November 9, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27.0.1 · ComfyUI v0.44.2
(`e6515b16`) · PyTorch 2.15.0 (`d0f2af8b`) / torchvision 0.29.2 / Comfy Kitchen
0.2.40 · frontend 1.57.6.

**All 35 cases run on PyTorch 2.15.0 with every pack route active. Every
build-gated route is byte-exact against PyTorch's own path on the same stack
(H3's DiT kernels were corrected separately; see below), and timings match
2.14.1 in a same-session A/B. Two speedups come from 2.15: FP8 weight decode is
2.3–44× faster through the new native MPS cast (byte-identical), and YuE2's
language model is about 7% faster from faster native GEMV.**

## What 2.15 changes for the pack

| Area | Without changes | Now |
|---|---|---|
| Installer | 2.15.0 is not a reviewed build, so first startup **re-pins 2.14.1** | 2.15.0 + torchvision 0.29.2 is a reviewed `supported` build and is kept. 2.14.1 remains the preferred install |
| Fusion gates | Routes gated on the exact 2.14.1 release or build (MPP query cache, H3 schedules, Fibo 2, Flux.3, MOSS, Krea2, SeedVR2, SDXL GEGLU) stand down. Results stay correct but slower | Gates admit the reviewed 2.15.0 build through one check (`torch_mpp_qcache.current_torch()`) |
| MPP attention source check | The expanded `MppAttention.h` hash changes, which disables the query-cache attention | The 2.15.0 hash is admitted. The only differences are include guards and new FP8/complex helpers in `c10/metal`; tile sizes, MMA arithmetic and softmax are unchanged |
| FP8 decode | LUT gather, with a CPU round trip for transposed weights | Native MPS `float8_e4m3fn` cast after a one-time all-256-codes bit check (below) |
| Unit tests | 18 failures: version literals in gates and tests | 1,797 pass (default), 1,994 (GPU extensions), 157 (router/flash/FP8 integration) |

The 7 GPU-extension failures (`test_minimax_fast` gate/final-layer and
`test_minimax_vae_fast` QKV-strided references) fail identically on 2.14.1,
so 2.15 didn't cause them.

**Correction (same day):** this report first called them stale test
references. That was wrong for H3's DiT. Since PyTorch 2.14, native MPS
`F.rms_norm(x, weight)` rounds once, but the H3 norm and Q/K RoPE kernels
still followed 2.13's double rounding. On 2.14+ they differed from the eager
ops in about 21% of norm outputs and 18% of rotated outputs, by 1 BF16 ULP.
The VAE kernels were exact for production's unit Q/K scale.

All of them are now fixed, and the 7 tests pass. See the
[nightly report](m5-ultra-comfy-nightly-fca60fe7-nov09.md#h3-norm-and-rope-rounding).
The routes-off reference below closed only the build-gated routes. H3's
`minimax_fast` kernels aren't gated on the torch build, so they ran in both
runs, and that comparison doesn't establish their exactness. The H3 timings
below predate the fix.

## Exactness

Many native ops changed bytes between 2.14.1 and 2.15.0. In an op-by-op A/B,
the changes are:

- **Changed:** single-row linear (FP16 with or without bias, BF16 with bias),
  softmax (all dtypes), and BF16/FP16 `addcmul` and `var`.
- **Unchanged:** SDPA (all shapes, including GQA, masked, causal and single
  query), large linears, bmm/baddbmm, layer/RMS/group norm, GELU/SiLU,
  reductions, conv2d/3d, interpolate, casts and every FP32 op.

As a result, most workflow latents differ from the 2.14.1 runs. This is
expected for a framework upgrade.

The pack's standard is that a fusion must reproduce the bytes of PyTorch's
compatibility path **on the same stack**. To test it, the suite was run twice
on 2.15.0:

1. **Routes on.** Repo state. All 35 cases succeed and repeat their own bytes.
2. **Compatibility reference.** Every 2.15 fusion gate is closed in the
   instance copy only. Native SDPA stays selected.

**Every compared tensor in all 35 cases is byte-identical between the two
runs** (`prev_diff=0` everywhere), so each build-gated fusion is exact on
2.15.0. With the
routes off, the reference is 2–310% slower; Flux.3 prompt expansion
10.17 → 41.65 s is the largest gap. That confirms each route was active.

## Timings

**Same-session A/B** (the 2.14.1 venv clone and the 2.15.0 venv, run
back to back, routes on):

| Workflow | 2.14.1 | 2.15.0 |
|---|---:|---:|
| Qwen-Image 2.1 T2I | 12.74 | 12.61 |
| Krea2 1024² | 6.07 | 6.06 |
| H3 SOL Metal · fast | 117.83 | 119.43 |
| H3 Dense Metal · balanced | 134.66 | 133.77 |
| Wan 2.2 Turbo | 30.67 | 30.77 |

The H3 sampling-loop spread within one setup is 102–108 s, so the H3 rows are
within noise. Today's machine state is about 2% slower on H3 than November 1
for **both** versions. That difference, not the upgrade, is the gap between
the full-suite run below and November 1.

**Full suite, routes on, against November 1 (2.14.1, clean machine):**

| Workflow | Nov 1 warm | 2.15.0 warm |
|---|---:|---:|
| SDXL · 896×1152 · 12 steps | 2.53 | 2.52 |
| Z-Image Turbo · 1024² · 8 steps | 3.53 | 3.47 |
| Krea2 · 1448² · LoKr / no LoKr | 13.39 / 13.53 | 13.38 / 13.44 |
| Krea2 · 1024² | 5.98 | 6.05 |
| Flux.2 Klein 9B · T2I / edit | 2.37 / 4.64 | 2.34 / 4.63 |
| Ideogram 4 · 1024² · 12 steps | 17.16 | 17.38 |
| Qwen-Image 2.1 · T2I / edit | 12.48 / 22.80 | 12.71 / 22.97 |
| **YuE2 · 8 s / full song / Mothersuperior** | 5.86 / 34.43 / 22.23 | **5.38 / 31.83 / 20.16** |
| LTX 2.5 · diffusion / convolutional VAE | 44.81 / 30.92 | 45.00 / 30.97 |
| H3 exact | 144.88 | 148.15 |
| H3 SOL Metal · fast / Dense Metal · balanced | 112.42 / 129.32 | 114.77 / 131.78 |
| Wan 2.2 Turbo | 30.45 | 30.72 |
| SeedVR2 image / 120-frame video | 44.76 / 123.02 | 45.00 / 123.22 |
| Flux.3 distilled · T2I / edit | 5.80 / 12.80 | 5.71 / 12.77 |
| Flux.3 Base · T2I / edit / 64-token expansion | 33.38 / 70.82 / 10.44 | 33.32 / 71.45 / 10.17 |
| MOSS SFT / Base 1024², SFT 1248² | 18.92 / 19.53 / 29.83 | 18.88 / 19.54 / 29.94 |
| Fibo 2.0 Dev / Edit Turbo | 31.65 / 6.32 | 31.60 / 6.30 |
| Preprocessors (BiRefNet, SDPose, DA3 mono / multi-view) | 0.77 / 1.47 / 0.41 / 0.12 | 0.74 / 1.46 / 0.41 / 0.12 |

**YuE2** gains from 2.15's faster native GEMV: semantic decoding goes from
95 to 105 tokens/s. The changed single-row linear bytes alter the sampled
token stream (the full song decodes 2,095 semantic tokens instead of 2,080).
The 8-second case shows the same −8% with a near-identical token count.

Op-level checks behind these results (identical shapes and inputs on both
builds):

| Op | 2.14.1 → 2.15.0 |
|---|---|
| Large GEMMs (16384×{3072…28672}), norms, GELU/SiLU, conv3d, SDPA 56×16384 | ±1% |
| Pack MPP query-cache attention, H3 shapes 8k–41k tokens | ±0.3% |
| GEMV (single-row linear) | 10–55% faster; BF16 5120² without bias +11% |
| softmax, BF16 attention-sized rows | 17–33% slower; the default attention routes (native SDPA, MPP query cache) do not call it |
| `var`, BF16 | 41–50% faster |
| `addcmul` | ±5% |

## New native paths

**FP8 decode (enabled by default on 2.15).** 2.15 adds MPS kernels for
`float8_e4m3fn` casts:

- FP8 → BF16 and FP8 → FP32 are bit-identical to the pack's LUT for all 256
  codes, including NaNs, and work on non-contiguous tensors.
- FP8 → FP16 differs only in the NaN payload (native `0x7E00`), so it keeps
  the LUT.
- `float8_e5m2` has no MPS cast and keeps the LUT.

The pack now checks every code once per dtype pair and uses the native cast
where the check passes; any other build keeps the LUT. On 2.14.1 the check
fails closed and the full suite passes. Opt out with
`ASFP8_FP8_NATIVE_CAST=0`.

| Pack path (BF16 output) | LUT | Native | Speedup | Same bytes |
|---|---:|---:|---:|---|
| `torch._scaled_mm` 1×3072×3072 | 5.61 ms | 0.36 ms | 15.4× | ✓ |
| `torch._scaled_mm` 256×3072×3072 | 5.61 ms | 0.43 ms | 13.1× | ✓ |
| `torch._scaled_mm` 4096×3072×12288 | 26.51 ms | 4.46 ms | 5.9× | ✓ |
| `torch._scaled_mm` 8192×5120×5120 | 19.39 ms | 5.17 ms | 3.8× | ✓ |
| `torch._scaled_mm` 16384×3072×12288 | 37.92 ms | 16.28 ms | 2.3× | ✓ |
| Kitchen per-tensor dequant 3072² / 12288×3072 / 5120×13824 | 0.28 / 1.19 / 2.24 ms | 0.08 / 0.28 / 0.54 ms | 3.4–4.3× | ✓ |
| Transposed weight decode, same shapes | 5.2 / 21.7 / 39.9 ms | 0.12 / 0.66 / 0.90 ms | 33–44× | ✓ |

No full FP8 diffusion checkpoint is on this machine, so these are path
measurements, not a workflow timing. FP8-weight models (Kitchen per-tensor
FP8, `_scaled_mm` checkpoints, FP8-stored biases) are the beneficiaries.

**Evaluated, not adopted:**

- **Native `F.scaled_mm` (`_scaled_mm_v2`).** It runs FP8 GEMMs on MPS, but
  its accumulation differs from the decode path. It is now matched by the
  decode path up to M≈256 and slower at M=8192. The legacy `torch._scaled_mm`
  has no MPS kernel in 2.15.
- **Native `_int_mm`, `_weight_int8pack_mm`, `_weight_int4pack_mm` and
  `_grouped_mm`.** These are now on MPS. The pack's INT8 routes are integer-
  exact and already faster at their shapes, so they are unchanged.

## Data

Harnesses are in `diagnostics/`:

- `comfy0442-torch215-smoke/`: routes on, previous = November 8.
- `comfy0442-torch215-compat/`: gates closed, previous = the routes-on run.
- `comfy0442-torch215-ab2141/` and `comfy0442-torch215-ab2150/`: the
  same-session A/B.

The 2.14.1 environment is preserved as an APFS clone,
`ComfyUI-AppleSilicon-dev/venv-backup-torch2141`.
