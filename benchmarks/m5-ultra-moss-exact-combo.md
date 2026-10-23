# MOSS SFT: combined exact speedups — October 23, 2026

M5 Ultra 80 GPU / 256 GB · macOS 27 · PyTorch 2.14.1 · ComfyUI v0.41.3 core ·
Comfy Kitchen 0.2.38. SFT INT8 ConvRot, CFG 2.5, ER-SDE/simple, 25 steps, seed
`856771198829162`, the same theatre-marquee fixture as the earlier MOSS reports.
Implemented as the pack's default exact MOSS route (see the final section); not
yet installed into the live ComfyUI instance.

## Result

Three byte-exact changes, combined in one resident server, interleaved with
exact controls:

| Resolution | Exact E2E (2 runs) | Combined E2E (2 runs) | Less time |
|---|---:|---:|---:|
| 1024×1024 | 20.480 / 20.449 s | 19.298 / 19.306 s | **1.16 s / 5.7%** |
| 1248×1248 | 32.088 / 32.267 s | 31.377 / 31.021 s | **0.98 s / 3.0%** |

| Single change, one run each | 1024² E2E | 1248² E2E |
|---|---:|---:|
| INT8 tile schedule | 19.934 s | 31.721 s |
| Joined CFG rows | 19.978 s | 31.599 s |
| 32-row exact attention tiles | 20.193 s | 31.670 s |

Every render's sampler latents and decoded frames match the same-resolution
exact control **byte for byte**. A two-step verification run also compared all
42 joined blocks with separate-branch blocks at both sizes (84 checks each).

## The three changes

- **Joined CFG rows.** The Oct 18 diagnostic adapter, persistent variant:
  positive and negative hidden states stay joined through all four INT8
  projections, while attention, positions, text embedding and modulation stay
  per branch. It is limited to the reviewed batch-one, global-conditioning
  fixture and requires identical timestep modulation for both branches.
- **INT8 tile schedule.** The Oct 18 schedule (128-row tiles, no swizzle,
  N-major grid), extended to K=8192 and to the joined 8,298-row shapes. Integer
  accumulation makes the output independent of tile order.

  | Matrix (ms) | Production schedule | New schedule |
  |---|---:|---:|
  | QKV | 1.262 | 1.203 |
  | O | 0.445 | 0.402 |
  | FFN up/gate | 2.162 | 2.094 |
  | FFN down | 1.177 | 1.043 |
  | Joined, per layer-step | 9.39 | 9.15 (separate branches: 10.10) |

- **32-row exact attention tiles.** The existing `ASFP8_MOSS_ATTN_MODE=tile32`.

## Fused rotate + quantize: exact, but slower

ConvRot rotates every activation with Kitchen's BF16 Hadamard matmul before
rowwise INT8 quantization. On 12 captured projection inputs (QKV, O, FFN
up/gate and FFN down; early, middle and late layers; 218 M values):

| Rotation, FP32 summation order | Bytes differing from the MPS matmul |
|---|---:|
| Sequential, blocked (2–128) and pairwise ALU orders (15 variants) | 215–915 |
| Matrix units (MPP `matmul2d`), any K chunk ≥ 16 | **0** |
| Matrix units, K chunk 8 | 215 |

A fused kernel built on the matrix units makes two passes: an order-independent
row absmax, then rotate and quantize. It reproduces every INT8 value and scale of
the current path. It is slower (fused 0.16–0.20 ms vs current 0.13 ms at
K=3072, kernel time only), because each pass costs about as much as the MPS
rotation itself. The ceiling for any exact fusion is about 0.4 s per 25-step
render: the saved BF16 round trip. **Not pursued.**

## Context

At 1024² one pass is about 4.1k tokens. Projections run at 175–190 TOPS INT8
(about 70% of the estimated matrix-unit peak). Attention uses about 4.5 s of the
20.5 s; rotation and quantization use about 2.1 s. The remaining exact headroom
is small. Beyond about 19.3 s, the remaining speedups are approximate (for
example the tuned INT8 attention modes).

Harness, captures and scripts: workspace `diagnostics/moss-int8v3-oct22`
(`exact_candidate.py`, `gemm_sched.py`, `rot_mpp.py`, `rotq.py`, `ex-results.json`).

## Implemented in the pack (October 23)

- `moss_fast` joins CFG rows inside the guarded MOSS sample (`ASFP8_MOSS_CFG_JOIN`).
- It sets the INT8 tile schedule for that sample only (`ASFP8_MOSS_GEMM_SCHEDULE`).
  The schedule is limited to the measured sizes: up to 8,704 rows, covering joined
  1024² and single-branch 1248². Joined 1248² projections (12,274 rows) measured
  slower with it and keep the default schedule.
- `tile32` is now the default `ASFP8_MOSS_ATTN_MODE`.

Pack defaults vs the previous route (all three disabled), alternating in one
server:

| Resolution | Previous | New | Less time |
|---|---:|---:|---:|
| 1024² | 20.432 / 20.680 s | 19.309 / 19.251 s | 1.28 s / 6.2% |
| 1248² | 32.387 / 32.403 s | 31.374 / 31.344 s | 1.04 s / 3.2% |

All eight renders' latents and decoded frames are byte-identical to the
October 23 exact references at their resolution. Each new run recorded 25
joined model calls and 2,100 tile32 attention calls.

## Beta-tester fusions (October 23)

A beta tester's handoff patch was reviewed and its MOSS and Krea2 parts were
merged. Its H3 BF16 changes were not merged: they apply only to plain-BF16 H3
weights.

**Independent byte checks, with no mismatches:**

| Check | Values compared |
|---|---:|
| Q/K RMS + RoPE fusion (24 seeds, 4,119–6,656 rows, magnitudes 1e-3 to 1e3) | 779 M |
| RMS/modulation fusion (up to 12,274 rows) | 570 M |
| MOSS BF16 FC1/SiTU pair | 118 M |
| H3 BF16 8192-row chunked FC1 | 3.2 B |

**Full renders**, SFT INT8, 25 steps, fusions off vs on in one server:

| Resolution | Previous (this report's route) | With fusions |
|---|---:|---:|
| 1024² | 19.516 / 19.420 s | 17.588 / 17.588 s (−9.7%) |
| 1248² | 31.651 / 31.655 s | 28.840 / 28.611 s (−9.3%) |

After merging into the pack, the 1024² render took 17.467 / 17.487 s and the
1248² render 28.425 / 28.440 s. All latents and decoded frames are
byte-identical to the exact references.

**Krea2 Turbo INT8** with LoRA, 1448², 8 steps: the paired gate/up FFN averaged
13.09 → 12.96 s sampling (three runs each, within noise). All outputs are
byte-identical, with 224 paired calls per render.

## README table pairs (October 23)

These pairs use the README table's boundary: warm end-to-end renders with both
prompts freshly encoded, sampling, decode and PNG save, and prompt enhancement
off. They were measured on core `7a283320` (`v0.42.0-19`) with Kitchen 0.2.39
and PyTorch 2.14.1. Stock omits this pack. Base keeps its saved Skip Layer
Guidance.

| Workflow | Stock | Optimized | Speedup |
|---|---:|---:|---:|
| SFT 1024², 25 steps | 69.251 / 69.427 s | 19.093 / 19.095 s | 3.63× |
| Base 1024², 25 steps | 70.435 / 70.317 s | 19.777 / 19.735 s | 3.56× |
| SFT 1248², 25 steps | 88.411 s | 30.212 / 30.286 s | 2.92× |
| SFT 1248², 50 steps | 155.494 s | 58.494 / 58.850 s | 2.65× |

Stock text encoding alone takes about 21–22 s per render; the pack's takes about 1.7 s.
