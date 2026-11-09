# ComfyUI nightly `fca60fe7` (RelayVSR core): regression on the dev instance

November 9, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27.0.1 · PyTorch 2.15.0 /
torchvision 0.29.2 / **Comfy Kitchen 0.2.41** · frontend 1.58.5 · templates
0.11.103.

**34 jobs (35 workflow/mode/preprocessor cases) completed twice. Every case
repeats its own bytes, and every compared tensor is byte-identical to the
[PyTorch 2.15 run on v0.44.2](m5-ultra-torch215-nov09.md).** The nightly makes
no numerical change to any tested workflow.

Five routes stood down or slowed on this core and were repaired. Each repaired
case is byte-identical to its unrepaired run on the same core. After the
repairs, timings match v0.44.2.

A separate H3 fix was then made: the DiT norm and RoPE kernels now match
PyTorch 2.14+ rounding, and the DiT fast path is byte-identical to the
upstream DiT path. That fix intentionally changes H3's output bytes; see
[H3 norm and RoPE rounding](#h3-norm-and-rope-rounding).

## Tested source and method

- **Core:** master `fca60fe7` (`v0.44.0-33`, November 9), which adds RelayVSR
  support (`2f894332`). The v0.44.2 release-branch commit is not an ancestor;
  the relevant source changes are covered by the diff review below.
- **Packages:** Kitchen 0.2.40 → 0.2.41 and frontend 1.57.6 → 1.58.5. Torch is
  unchanged.
- **Pack:** repository `main` at `ef595be` plus the repairs below.
- **References:** each case's routes-on run on v0.44.2 with PyTorch 2.15.0 (same
  torch, so bytes are directly comparable). Fixtures, prompts, seeds, steps and
  inputs are the same.
- **Scope:** as before. H3 covers exact, *SOL Metal · fast* and *Dense Metal ·
  balanced*. Wan 2.2 covers the four-step Turbo workflow.
- **Order:** as-is run → static guard review → repairs → rerun of the affected
  cases. As-is results are kept in `asis/`.

## Repairs

| Route | What changed upstream | Repair | As-is → repaired |
|---|---|---|---|
| Flux.3 | Kitchen 0.2.41: adds `window_attention`, `memmove` and CUDA/HIP kernels; eager RoPE/SwiGLU unchanged | Admit 0.2.41 | Base T2I 42.88 → **33.15 s**; 64-token expansion 41.32 → **10.10 s** |
| MOSS-Image | Kitchen 0.2.41, plus `ops.py` adds `CombinedLinear` for INT8-quantized Llama/Qwen 3.5 projections. Uncombined modules and the `linear_input_act`/cast paths are unchanged | Admit 0.2.41 and the new `ops.py` hash | SFT 1024² 34.19 → **18.99 s** |
| YuE2 exact RoPE | `llama.py`: Q/K/V and gate/up go through `CombinedLinear` first; for BF16 weights it returns `NotImplemented` and the separate projections run. Attention and RoPE unchanged | Admit the new `llama.py` hash | RoPE active again (`rope_calls` 0 → 5,600 per 8 s phase) |
| YuE2 INT8 DiT and LM | `llama.py`: a merged gate/up projection without a pre-norm now calls `linear_input_act(gate_up_proj, x, None)` instead of `gate_up_proj(x)`. On MPS the INT8 branch adds a requant cast and an eager INT8 attempt that bypasses the pack's INT8 Metal kernel | When the INT8 kernel is installed, the pack's `linear_input_act` wrapper keeps the module call for that exact three-argument form. Calls with norms or residuals are unaffected. `ASFP8_INT8_PLAIN_LIA=0` restores the upstream call | DiT 8 → 22 it/s; 8 s 9.41 → **5.39 s**, full song 44.40 → **32.26 s**, Mothersuperior 28.62 → **20.19 s** |
| SeedVR2 | `seedvr/vae.py`: adaptive slice planning and memory-compiler scopes. On MPS the plan keeps the previous one-frame head/tail chunks and slice sizes. `ResnetBlock3D`, causal convolutions, norms and `CausalMemoryCache.pop` are unchanged | Admit the new VAE hash; the guard is now a reviewed set | Image 55.15 → **45.03 s**; 120-frame video 128.90 → **123.55 s** |

How the YuE2 regression was found: after the hash admission, YuE2 still ran
57% slower with identical bytes. Restoring v0.44.2's `model_prefetch.py` alone
changed nothing. Restoring v0.44.2's `llama.py` alone brought back the old
speed and cache size. Logging then showed the combined weights were never
built on MPS. That pointed to the changed merged-MLP call. Harness:
`diagnostics/comfy-nightly-fca60fe7-bisect/`.

The other core changes were reviewed with no effect on tested routes:

- **Model management, patcher and prefetch:** new stream and VBAR helpers. The
  functions the pack replaces (`text_encoder_device`, `CLIP.generate`) are
  unchanged.
- **Qwen-VL / Qwen 3.5:** video input. Image preprocessing builds the same
  tensors, and the Flux.3 prompt cache only wraps `generate`.
- **RelayVSR model and nodes:** new.

## Timings

| Workflow | v0.44.2 (2.15) | Nightly as-is | Nightly repaired | Same bytes |
|---|---:|---:|---:|---|
| SDXL · 896×1152 · 12 steps | 2.52 | 2.49 | — | 5/5 |
| Z-Image Turbo · 1024² · 8 steps | 3.47 | 3.50 | 3.49 | 5/5 |
| Krea2 · 1448² · LoKr / no LoKr | 13.38 / 13.44 | 13.39 / 13.39 | — | 4/4 each |
| Krea2 · 1024² | 6.05 | 6.02 | 6.07 | 4/4 |
| Flux.2 Klein 9B · T2I / edit | 2.34 / 4.63 | 2.37 / 4.63 | 2.30 / 4.67 | 9/9 · 11/11 |
| Ideogram 4 · 1024² · 12 steps | 17.38 | 17.08 | — | 9/9 |
| Qwen-Image 2.1 · T2I / edit | 12.71 / 22.97 | 12.54 / 22.80 | 12.62 / — | 4/4 · 5/5 |
| YuE2 · 8 s / full song / Mothersuperior | 5.38 / 31.83 / 20.16 | 9.41 / 44.40 / 28.62 | **5.39 / 32.26 / 20.19** | 3/3 each |
| LTX 2.5 · diffusion / convolutional VAE | 45.00 / 30.97 | 44.94 / 30.89 | — | 21/21 each |
| H3 exact | 148.15 | 148.95 | — | 9/9 |
| H3 SOL Metal · fast / Dense Metal · balanced | 114.77 / 131.78 | 113.09 / 131.72 | — | 9/9 each |
| Wan 2.2 Turbo | 30.72 | 30.36 | — | 19/19 |
| SeedVR2 image / 120-frame video | 45.00 / 123.22 | 55.15 / 128.90 | **45.03 / 123.55** | 3/3 each |
| Flux.3 distilled · T2I / edit | 5.71 / 12.77 | 7.36 / 15.13 | **5.90 / 12.79** | 6/6 · 8/8 |
| Flux.3 Base · T2I / edit / 64-token expansion | 33.32 / 71.45 / 10.17 | 42.88 / 83.06 / 41.32 | **33.15 / 70.85 / 10.10** | 6/6 · 8/8 · 8/8 |
| MOSS SFT / Base 1024², SFT 1248² | 18.88 / 19.54 / 29.94 | 34.19 / 34.33 / 50.57 | **18.99 / 19.59 / 29.95** | 8/8 each |
| Fibo 2.0 Dev / Edit Turbo | 31.60 / 6.30 | 31.32 / 6.28 | — | 8/8 · 10/10 |
| Preprocessors (BiRefNet, SDPose, DA3 mono / multi-view) | 0.74 / 1.46 / 0.41 / 0.12 | 0.73 / 1.45 / 0.41 / 0.12 | — | ✓ |

A dash means the case was not affected and kept its as-is result. Text-encoder
cases (Z-Image, Flux.2, Krea2 1024², Qwen T2I, Flux.3 distilled T2I and
64-token expansion) were rerun after the INT8 call repair; their bytes are
unchanged. Flux.3 distilled T2I ran 5.77–5.90 s across the reruns (November 8:
5.79 s).

**Not covered:** RelayVSR itself (new in this core), and Qwen3-VL prompt
generation, which is not part of the suite.

## H3 norm and RoPE rounding

Investigating the 7 long-failing MiniMax GPU tests showed a real exactness
bug in H3's DiT fast path. It predates this core and is not specific to it.

- **RMSNorm:** since PyTorch 2.14, native MPS `F.rms_norm(x, weight)` with a
  same-dtype weight runs a fused MLX-derived kernel. It rounds
  `(x · rsqrt(mean + eps)) · w` once. PyTorch 2.13 rounded the normalized
  value before the weight multiply, and the pack's H3 kernels still did.
- **RoPE:** Kitchen's eager split-half RoPE rounds the diagonal product, then
  `addcmul_` adds the exact off-diagonal product with one rounding. The pack's
  kernels rounded both products.
- **Effect on 2.14.1 and 2.15.0:** the norm outputs differed in about 21% of
  elements and the rotated Q/K values in about 18%, each by 1 BF16 ULP. H3
  still repeated its own bytes, so pack-vs-pack regressions never showed it.

**Fix.** A shared `torch_rms_sumsq` helper reproduces native's
sum-of-squares order: a virtual threadgroup of 1024 lanes (or
32·⌈⌈D/4⌉/32⌉ for D ≤ 4096), 4 contiguous values per lane, then two
`simd_sum` levels. Its output was 0 mismatches against native for
D = 64–5376. The rotation now follows Kitchen's `addcmul` order. The fix
covers:

- `rms_mod_seg`, `rms_mod_seg_f32` and `gate_norm_mod_seg`;
- the three Q/K norm + partial-RoPE kernels;
- the INT8 QKV epilogue (Ultra, default and LoRA schedules);
- the weighted-norm multiply in the two VAE QKV kernels. Production uses a
  unit Q/K scale there, so VAE output is unchanged.

**Verification:**

- **Kernel level:** each kernel against the eager calls production makes, on
  16M-element H3 tensors, on both 2.14.1 and 2.15.0: 0 mismatches. The new
  `tests/test_h3_kernels_eager_parity.py` keeps this checked.
- **Tests:** the 7 MiniMax tests pass. The GPU suite reports 2,017 passed, 0
  failed.
- **End to end:** 4-step H3 exact, compared with `ASFP8_MINIMAX_FAST=0` (the
  upstream DiT path), with `ASFP8_INT8_OPROJ_FUSE=0`:

| Kernels | Tensors identical to the upstream DiT path |
|---|---|
| Before the fix (rebuilt from `ef595be`) | 3 / 9 (latents, audio and frames differ) |
| After the fix | **9 / 9** |

The default o-projection fusion (`ASFP8_INT8_OPROJ_FUSE=1`, full FWHT at
H3's D = 128) remains a documented rounding-class route. It is not
byte-identical, and it keeps H3's default output different from the
upstream DiT path. The final-layer graph rewrite (`rms_mod_seg_f32`) stands
down on every recent core, so it is not in the production path.

**Timing** (full regression workflows, same session, before → after the fix):

| Mode | Warm | Sampling loop |
|---|---:|---:|
| H3 exact | 148.95 → 148.87 | 137.38 → 137.31 |
| SOL Metal · fast | 113.10 → 114.03 | 101.91 → 102.81 |
| Dense Metal · balanced | 131.72 → 133.19 | 120.48 → 121.80 |

The SOL and Dense changes are within H3's run-to-run spread. H3 output bytes
change with this fix, as intended, and each mode repeats its own bytes.
Harness: `diagnostics/comfy-nightly-h3-exactness/`. The pre-fix H3 results
are in `comfy-nightly-fca60fe7-smoke/pre-norm-fix/`.

## Exactness audit against stock

The regression compares each pack run with the previous pack run, so it
cannot tell whether a fast route equals ComfyUI's own path. To check that,
every family not already covered by the PyTorch 2.15 gates-closed run was run
twice on this core: once with defaults, once with its fast modules disabled
(`ASFP8_DISABLE=<module>`). Each was a single run of the regression workflow
with the same seed and inputs. Families that differed were then bisected one
module at a time. The comparison is the final 8-bit output; video uses all
decoded frames.

| Workflow | Module disabled | Same bytes as default | Output vs default |
|---|---|---|---|
| SDXL | `sdxl_geglu`, `vae_slice_attn_mps` | 5/5 | identical |
| Qwen-Image 2.1 T2I | `qwen21_fast` | 4/4 | identical |
| Z-Image Turbo | `rope_fast` / `vae_slice_attn_mps` | 5/5 each | identical |
| Z-Image Turbo | `lumina_attn_fast` | 3/5 | 27.8 dB PSNR |
| Flux.2 Klein T2I | `rope_fast` / `vae_slice_attn_mps` | 9/9 each | identical |
| Flux.2 Klein T2I | `flux_attn_fast` | 6/9 | 22.0 dB |
| Ideogram 4 | `rope_fast` | 9/9 | identical |
| Ideogram 4 | `ideogram_fast` | 6/9 | 20.5 dB |
| Krea2 1024² | `krea2_ffn_pair` / `rope_fast` | 4/4 each | identical |
| Krea2 1024² | `krea2_attn_fast` | 2/4 | 23.0 dB |
| LTX 2.5 | `ltx25_fast` | 21/21 | identical |
| LTX 2.5 | `ltx_fast` | 8/21 | 27.7 dB |
| LTX 2.5 | `conv3d_mps` | 13/21 | 40.4 dB (VAE only) |
| Wan 2.2 Turbo | `wan_fast` | 15/19 | 27.0 dB |
| Wan 2.2 Turbo | `wan_vae_fast` / `conv3d_mps` | 11/19 each | 25.0 / 27.5 dB |

What this means:

- **The routes in the lower rows are not byte-exact against ComfyUI's own
  path on the current stack.** Their documentation already describes them as
  per-op exact-or-1-ULP (`lumina_attn_fast`, `flux_attn_fast`, `ltx_fast`,
  `wan_fast`, `rope_fast`) or exact up to accumulation order (`conv3d_mps`).
  `ideogram_fast` and `krea2_attn_fast` describe their fusions as exact.
- **Over 4–20 sampling steps a per-op ULP difference grows into a different
  but comparable render,** hence 20–28 dB rather than near-identical output.
  The VAE-only convolution change stays at 40 dB.
- **Not yet established:** which kernel in each DiT module causes the
  difference. Several of them share the dit_ext norm kernels that rounded
  like PyTorch 2.13 (the H3 bug above), but that cause is not verified for
  these models.
- **The README numerics policy now lists both groups.**

Harness: `diagnostics/comfy-nightly-exactness-audit/`:

- `cmp_pairs.py` compares the default and all-off runs;
- `cmp_bisect.py` compares the single-module runs;
- its family runners keep explicit `--env` overrides, which the regression
  runners strip.

## Data

Harness: `diagnostics/comfy-nightly-fca60fe7-smoke/`, prepared from the PyTorch
2.15 v0.44.2 run by `prepare.py`. It holds:

- `asis/`: the unrepaired first pass;
- `repair1/`: the run after the hash and Kitchen admissions;
- `repair2-unused-split/`: an intermediate run with a discarded hypothesis;
- the final results in place.

The bisect runs are in `diagnostics/comfy-nightly-fca60fe7-bisect/`.
