# Fibo 2.0: stock baseline and exact route — October 23, 2026

M5 Ultra 80 GPU / 256 GB · macOS 27 · PyTorch 2.14.1 · ComfyUI core `7a283320`
(`v0.42.0-19`, Fibo 2.0 support from Comfy-Org/ComfyUI#17240) · Comfy Kitchen
0.2.39. Comfy-Org `fibo2_dev_int8_convrot` / `fibo2_edit_turbo_int8_convrot`,
`qwen3vl_4b_int8_convrot` text encoder, `flux2-vae`. Euler/simple, seed
`812764035`.

- **Dev text-to-image:** 1024×1024 (4,096 image + 256 gist tokens), 30 steps,
  CFG 5, the lighthouse prompt from the bundled workflow.
- **Edit Turbo:** a 1024×1024 Dev render as the reference (8,512 tokens), 4
  steps, CFG 1, the bundled winter edit instruction.

Each configuration ran in its own server: one load run, then two warm runs.
Every job re-encodes its prompt (and its reference image). The timing covers
encoding, sampling, decode and PNG save. Stock omits this pack.

## Result

| Workflow | Stock | Pack before | Pack now | Speedup vs stock |
|---|---:|---:|---:|---:|
| Dev T2I, 1024² / 30 steps / CFG 5 | 88.49 s | 38.47 s | **31.87 s** | **2.78×** |
| Edit Turbo, 1024² reference / 4 steps | 16.19 s | 7.38 s | **6.61 s** | **2.45×** |

Two-run warm means (stock Dev 88.86 / 88.11 s; now 31.98 / 31.77 s; stock
Edit 16.25 / 16.14 s; now 6.69 / 6.53 s). Sampling alone: Dev 85.88 → 37.51 →
**30.90 s**, Edit 11.29 → 5.97 → **5.21 s**.

The new route's sampler latents and decoded frames are **byte-identical** to the
previous pack for both workflows (latents `43142a09…` / `73c6d534…`). The pack
is not stock-byte-identical, as for every INT8 ConvRot model: its native INT8
kernels replace Kitchen's eager emulation. Outputs were checked visually
against stock.

## Where the time goes (previous pack, Dev, one step = two CFG passes, 1.24 s)

- INT8 projections: about 0.76 s at 140–185 TOPS (qkv 10752×3584, out
  3584², w13 19456×3584, w2 3584×9728).
- Self-attention, 28 heads × 128: about 0.28 s on PyTorch SDPA (3.09 ms per
  call at 4,352 tokens). Our hash-checked query-cache kernel did not cover 28
  heads.
- Norms, modulation, gates and residuals: about 0.16 s. Each RMSNorm,
  slice-wise modulate and gate is a separate full memory pass.

## The exact changes

| Probe, real captured inputs | Dev step (2 passes) | Edit pass |
|---|---:|---:|
| Previous pack | 1.239 s | 1.406 s |
| + fused glue and tile32 attention | 1.106 s | 1.219 s |
| + joined CFG rows | **1.019 s** | n/a (CFG 1) |

All outputs in this probe matched the previous pack byte for byte.

1. **tile32 attention.** This is the PyTorch 2.14 MPP prefill kernel with 32-row
   query tiles and 2 SIMD groups, the same re-tiling MOSS uses. Key tiles, MMA
   arithmetic, softmax and reduction order are unchanged. It is byte-identical
   to SDPA at 4,352–8,448 tokens and for batch 2. Per call: 3.09 → 2.69 ms at
   4,352 tokens, 10.83 → 8.85 ms at 8,448. Of the other tilings tried, 64/4
   and 128/8 are slower and 16/1 is much slower. Three other shapes do not
   compile.
2. **Fused glue.** RMSNorm + modulate and RMSNorm + gate + in-place residual
   each run as one pass. The kernel reproduces PyTorch's `RMSNorm.metal`
   reduction: N_READS 4, ceil(D/128)·32 threads, two simd_sum levels, precise
   rsqrt, and `w·(x·inv)` in FP32. It keeps each BF16 rounding of the upstream
   `modulate` / `apply_gate` / `add_` chain. Gist and reference tokens keep
   their zero-timestep rows. A probe of alternative formulations is in
   `diagnostics/fibo2-perf-oct23/rms_probe.py`: `(x·inv)·w` also matches,
   while rounding `x·inv` first or folding `inv·w` does not.
3. **Joined CFG rows.** Fibo's positive and negative prompts have different
   lengths, so upstream runs them as two passes. The gist count is the same
   for both whenever the prompts give the same count (256 up to 256 text
   tokens). In that case both branches run as one batch-2 forward. The
   Perceiver, text cross-attention, BF16 `x_embedder`, final layer and the
   per-block modulation GEMMs stay per branch, so the BF16 matmuls keep their
   single-branch row counts. The INT8 projections use integer accumulation and
   run at M = 8,704, reaching 192–199 TOPS instead of 167–186. RMSNorm, RoPE,
   attention and the elementwise ops are row- and batch-independent.

INT8 GEMM schedules were also swept at M = 4,352 / 8,448 / 8,704. The options
were the MOSS 128-row N-major order, the Ideogram and Krea 64-row tiles, and
the long-M kernel. All are byte-identical. At the joined size none beats the
default by more than 1%. At 4,352 rows the MOSS order saves about 3%, but
joined CFG removes that shape from Dev. No schedule was adopted.

## Reproduce

`diagnostics/fibo2-perf-oct23/`:

- `run.py` runs the server renders: `--task t2i|edit --config stock|native`.
- `probe.py capture|bench` captures real inputs and runs the per-mode byte
  checks and profile.
- `attn_bench.py`, `tile_sweep.py`, `gemm_bench.py` and `rms_probe.py` hold the
  kernel measurements.

Data: [m5-ultra-fibo2.json](data/m5-ultra-fibo2.json).
