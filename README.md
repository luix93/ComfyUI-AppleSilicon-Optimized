<img width="1684" height="934" alt="banner" src="https://github.com/user-attachments/assets/b2c78072-5f28-4f65-8ee5-72dc55b5db0b" />


# ComfyUI-AppleSilicon-Optimized

**A full Apple Silicon inference stack for ComfyUI: quantized-model
compatibility plus custom Metal 4.1 kernels — flash attention, int8/fp8
matmuls, per-architecture DiT block fusions, and video-VAE kernels — tuned on
M5 Max.**

Private continuation of
[ComfyUI-AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8).
What began as an FP8 compatibility shim has diverged into a general
optimization layer: 29 runtime patches and five JIT-compiled Metal extensions
that take image models to roughly half their stock render time and make long
video generation on a MacBook competitive with — and in places faster than —
the fastest native Mac apps.

No model conversion, no forked ComfyUI: one custom node applies everything at
startup. Every fast path checks its inputs per call and **falls back to the
eager implementation** on anything it doesn't recognize, and every kernel is
held to **exact-or-1-ulp parity** against the eager math it replaces.

## Results

Measured on an M5 Max (128 GB), macOS 27 dev beta, warm model cache. The
"before" column is this node's compatibility layer alone (already several
times faster than a crash, which is the stock experience for most of these
models); "after" is with `ASFP8_NATIVE=1`.

| Workflow | Before | After | Reference point |
|---|---|---|---|
| Z-Image (Lumina) int8, 1MP | 18.0 s | **7.6 s** | — |
| Flux2 Klein 9B int8, 1MP | ~24 s | **~13 s** | — |
| Krea2 Turbo int8 convrot, 1MP / 8 steps | ~28 s | **15.5 s** | DrawThings ~13.5–14 s |
| LTX 2.3 video, 512² × 5 s | 63 s | **22 s** | DrawThings ~35 s |
| LTX 2.3 video, 1024² × 5 s | 295 s | **~100 s** | DrawThings ~120 s |
| Wan 2.2 i2v 14B×2 int8, 512² × 81 f | 445 s | **~135 s** | DGX-Spark-class ≈ 100 s (in progress) |

The LTX numbers beat DrawThings outright; Krea2 is within ~10% of it; the Wan
attention rewrite to close the last ~35 s is the active work item.

## The two layers

**1. Compatibility (automatic, any M1→M5).** FP8/INT8 checkpoints load and
render instead of crashing: LUT-based bit-exact FP8 decode, CPU reroutes for
the casts MPS can't do, `torch._int_mm` on GPU, flash attention for
long-context correctness/OOM, text encoders moved off the CPU, plus fixes for
macOS-beta psutil, the PiD ≥2048px black image, and WanVideo block swap. No
env vars needed. See [the compatibility layer](#the-compatibility-layer-patches-116).

**2. Native kernels (opt-in, `ASFP8_NATIVE=1`).** Five ObjC++/Metal
extensions, JIT-built on first use against **Metal 4.1** (developed on the
macOS 27 dev beta; the int8 and fp8 matmuls additionally need an **M5**-class
GPU for cooperative TensorOps / the fp8 format type). Anything that can't
build or doesn't apply falls back silently, so the switch is safe to leave on
everywhere.

```bash
xcode-select --install        # Metal toolchain
pip install ninja
export ASFP8_NATIVE=1         # before launching ComfyUI
```

`ASFP8_NATIVE=1` implies `ASFP8_FP8_EXT`, `ASFP8_FP8_SGEMM`,
`ASFP8_INT8_EXT`, `ASFP8_ATTN_ROUTER`, `ASFP8_DIT_EXT`, `ASFP8_CONV3D`.
Explicit per-feature values always win (e.g. `ASFP8_NATIVE=1 ASFP8_DIT_EXT=0`).

## The native fast paths

### flash41 — Metal 4.1 flash attention (`mfa4_ext`)

The workhorse. A cooperative-matmul flash-attention forward kernel written
for Metal 4.1 TensorOps:

- **Dtype-templated bf16/fp16 I/O** with fp32 math inside — Wan computes in
  fp16, the flux family in bf16; both take the same kernel. Low-precision
  score tiles by default (`ASFP8_FA41_F32S=1` restores fp32 tiles).
- **mergeP**: the P (softmax) tile overwrites the score tile in threadgroup
  memory, which is what lets the large-tile geometries fit.
- **Sequence-adaptive tiling**: BR=32 below ~12.8k tokens (compute-bound
  regime, image models), BR=64 above (K/V-bandwidth-bound regime, video) —
  the switch is worth ~15% at Wan's 21.5k tokens.
- **GQA-native grouped KV**: Krea2's 48Q/12KV heads run without
  `repeat_interleave`, bit-identical to the repeated form.
- ~37 TF/s at Flux shapes (8.7k tokens, bf16), ~26–28 TF/s at video length.

### Attention router (patch #19)

Per-call routing on MPS: unmasked long-sequence attention → flash41; masked
attention above a real-score-size threshold → memory-safe sub-quadratic;
mid-size → batched bmm (gated on the *actual* `Lq×Lk` score bytes, not the
max-sequence square — that distinction is what keeps tiny cross-attention off
the slow path). Stock fused SDPA — silently inaccurate past ~4k tokens and
O(L²) memory — is never taken at length. `ASFP8_ATTN_DEBUG=1` logs each
distinct route taken.

### Quantized matmuls (patches #3, #17, #18)

- **int8 W8A8** (`ASFP8_INT8_EXT`, M5+): convrot checkpoints run the path the
  format was designed for — rotate the activation online, per-row quantize,
  then a bit-exact INT8×INT8→INT32 kernel (~105 TF/s at diffusion shapes,
  ~1.85× over bf16) with the rescale + bias fused into the store epilogue.
  Ported from [Cider](https://github.com/Mininglamp-AI/cider) (MIT). This is
  the backbone of the Z-Image / Flux2 / Krea2 / Wan int8 numbers above.
- **fp8-native `_scaled_mm`** (`ASFP8_FP8_EXT`): fp8×fp8 matmuls read fp8
  operands directly on the matrix units — no bf16 materialization —
  bit-exact, ~1.2–2.1× on FLUX/SD3.5-class shapes.
- **W8A16 staged GEMM** (`ASFP8_FP8_SGEMM`): half×fp8 kernel with
  tiled/swizzled geometry and fused bias for standard-fp8 Linears, skipping
  the per-forward weight decode.

### DiT block fusions (`ASFP8_DIT_EXT` — patches #21, #23–#25, #27, #28)

Per-architecture Metal kernels for the elementwise/copy-heavy glue between
matmuls, which on MPS costs as much as the matmuls themselves. Each patch
mirrors eager rounding op-for-op (exact-or-1-ulp) and falls back per call:

- **Lumina / Z-Image (#21)**: fused qkv-split + QK-RMSNorm + RoPE + head
  transpose (~7.5 ms → ~0.5 ms per layer), fused norm+modulate,
  norm+gate+residual, silu-mul.
- **RoPE (#23)**: flux-family `apply_rope` in one pass (~7.3 ms → ~0.7 ms per
  attention at Flux2 shapes).
- **Flux2 (#24)**: double/single-block fused qkv+QKNorm+RoPE+seq-cat — kills
  4 RMSNorms and 3 sequence cats per double block; handles Klein's global
  modulation.
- **Krea2 (#25)**: fused q/k/v norm+RoPE at native GQA head counts feeding
  flash41's grouped-KV path (the `repeat_interleave` and its 4× KV traffic
  are gone).
- **LTX (#27)**: one-pass split-RoPE (zero transposed copies) + fused
  rms_norm+modulate with per-frame timestep rows.
- **Wan (#28)** (`wan_ext`, fp16/bf16-templated): fused LayerNorm+modulate
  and gated residuals with *per-frame modulation rows* — the eager path
  materializes [B,F,C]→[B,L,C] repeats (220 MB a use, six uses per block);
  here they never exist — plus wide RMSNorm+RoPE written straight into
  attention layout.

### Video VAE (patches #26, #29)

- **Conv3d → sliced conv2d (#26, `ASFP8_CONV3D`)**: MPS's conv3d lowering is
  ~10× slower than the equivalent per-slice conv2d. Temporal taps fold into
  conv2d's *channel* dimension, so one conv2d accumulates all taps in fp32
  internally and rounds once — **exactly** conv3d's numerics at full speed.
  (Summing separately-rounded per-tap outputs looks harmless in one decode,
  but recursive causal VAEs like Wan's feed each chunk's output into the next
  chunk's cache, and the per-tap rounding compounds into horizontal banding
  that grows over the clip. Found the hard way.) 7× on LTX's video VAE.
- **Wan 2.1 VAE fast path (#29)**: decode profiling showed the convs already
  at MPS's ~60 TF/s ceiling; the time was in memory patterns around them.
  `chan_rms` runs the channelwise RMS_norm (+fused SiLU) — a 42 MB-strided
  reduction MPS handles at ~80 GB/s — at ~400 GB/s via threadgroup tiling;
  `fold_taps` gathers each causal conv's folded input straight from
  [zeros | cache | x], eliminating the pad-cat/tap-copy/channel-cat chain
  (~14 tensor volumes per conv → ~4); and the causal chunk loops stream 16
  frames per pass instead of 2. Decode 16.8 s → 6.1 s, encode 6.5 s → ~4 s at
  512²×81f, parity 1–2 ulp.

### MLX text generation (patch #14)

Autoregressive prompt-expansion encoders (Krea2's Qwen3-VL, LTX's Gemma3) run
token-by-token in Python on MPS (~1 s/token). With `mlx-vlm` installed, the
generation loop routes through MLX's fused decode loop with GPU KV-cache;
conditioning encode is untouched.

## The compatibility layer (patches #1–#16)

Condensed — each is automatic, MPS-scoped, and a no-op where not needed:

| # | Fixes |
|---|---|
| 1 | psutil `host_statistics64` crash on macOS betas (vm_stat fallback; self-disabling on healthy systems) |
| 2 | comfy_kitchen eager FP8 + NVFP4/MXFP8 dequant (Ideogram 4, LTX's Gemma3 TE) |
| 3 | `torch._scaled_mm` FP8 → LUT decode + bf16 matrix-unit matmul (FLUX, SD3.5, fp8_scaled) |
| 4 | `F.rms_norm` garbage above ~2²² rows — PiD ≥2048px black image |
| 5 | `F.scaled_dot_product_attention` → [mtlflashattn](https://github.com/pawel-mazurkiewicz/mtlflashattn) at length (correctness ≥4k tokens, OOM rescue, 3–4× over fused SDPA) |
| 6 | FP8 `UNETLoader` weight+bias decode in `cast_bias_weight` |
| 7 | LoRA-on-FP8 re-quant (`stochastic_rounding`) via CPU |
| 8 | `torch.Tensor.to` FP8↔float shim (third-party FP8 Linears, WanVideoWrapper) |
| 9 | WanVideo block-swap neutralizer (unified memory makes it pure breakage on MPS) |
| 10 | `F.linear` FP8 operand decode (T5 and friends) |
| 11 | Text/LLM encoders redirected CPU→MPS (autoregressive encoders were ~1.7 s/token on CPU) |
| 12 | `torch._int_mm` on GPU instead of the silent CPU-fallback freeze |
| 13 | int8-fast wide-batch Linear via native bf16 GEMM (~3.5–4.7×) |
| 14 | MLX-backed TextGenerate (above) |
| 16 | Strided-FP8 materialization (`reshape`/`contiguous`/`clone`) via CPU — fixes NVFP4/MXFP8 block-scale swizzles everywhere at once |

(#15 was retired: wrong seam — ComfyUI's fp8 routes through `_scaled_mm`, not
`F.linear`. #20 and #22 round out the set: fp8 fill/factory ops via CPU, and
defaulting missing `comfy_quant` format metadata so community int8-convrot
checkpoints load.)

### How the FP8 trick works

MPS can't cast to/from `float8_e4m3fn`/`e5m2` on the GPU — but it *can* move
FP8 tensors to the device, bit-view them as `uint8`, and gather. So a
256-entry table maps every FP8 byte to its float value (decoded once on CPU),
and any FP8 tensor decodes as `lut[x.view(uint8)]`: bit-exact with a real
cast, entirely on GPU, feeding MPS's fast float matmuls.

## Numerics policy

Three tiers, stated per patch:

- **Bit-exact**: all FP8 decodes, the int8 W8A8 kernel, `fold_taps`, the
  conv3d channel fold (single fp32 accumulation = conv3d's own numerics).
- **Exact-or-1-ulp**: the DiT fusions and VAE norm kernels mirror eager
  rounding at every point eager rounds (fp32 math between roundings); images
  shift imperceptibly but deterministically.
- **Regime-gated**: flash41's low-precision score tiles (default; parity
  ~5e-4, `ASFP8_FA41_F32S=1` for fp32 tiles).

Everything ships with parity tests against the eager path (104+ tests, run
against torch 2.10 and 2.13).

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `ASFP8_NATIVE` | off | Master switch: enables all six below |
| `ASFP8_FP8_EXT` | off | fp8-native `_scaled_mm` kernel (M5+, Metal 4.1) |
| `ASFP8_FP8_SGEMM` | off | staged half×fp8 W8A16 backend |
| `ASFP8_INT8_EXT` | off | int8 W8A8 kernel + fused epilogue (M5+) |
| `ASFP8_ATTN_ROUTER` | off | per-call attention routing (flash41 / bmm / sub-quad) |
| `ASFP8_DIT_EXT` | off | DiT block-fusion kernels (all families) |
| `ASFP8_CONV3D` | off | conv3d → sliced conv2d + Wan VAE fast path |
| `ASFP8_FLUX_FAST` / `KREA2_FAST` / `LTX_FAST` / `WAN_FAST` / `WAN_VAE_FAST` | on (under DIT_EXT/CONV3D) | per-architecture kill switches |
| `ASFP8_WAN_VAE_CHUNK` | 16 | Wan VAE frames per causal pass (even, 2–32) |
| `ASFP8_FA41_F32S` | off | fp32 score tiles in flash41 |
| `ASFP8_FP8_EXT_MIN_DIM` | 8192 | fp8 kernel size gate |
| `ASFP8_ENABLE_ONLY` / `ASFP8_DISABLE` | — | bisection: install only / all except these patch modules |
| `ASFP8_ATTN_DEBUG` / `ASFP8_EXEC_TRACE` / `ASFP8_PROFILE` | off | route logging / per-node timing / profiling (stderr) |
| `APPLESILICON_FP8_PSUTIL` | auto | psutil fix: `auto` / `off` / `force` |
| `MTLFLASHATTN_SDPA*` | — | patch #5 gates (see mtlflashattn docs) |

## Install

```bash
cd <your ComfyUI>/custom_nodes
git clone https://github.com/luix93/ComfyUI-AppleSilicon-Optimized
pip install -r ComfyUI-AppleSilicon-Optimized/requirements.txt
```

Restart ComfyUI. The only required dependency (`mtlflashattn`) installs
automatically; the native kernels additionally want `ninja` + Xcode
command-line tools. For ComfyUI Desktop, install into the app's own venv
(`.venv/bin/python3 -m pip ...`) — the bundled torch is the one that matters.

Startup prints one line per active patch
(`[AppleSilicon-FP8/<patch>] ...`); with `ASFP8_NATIVE=1` you'll also see the
master-switch line listing what it implied. Kernels JIT-build on first use
(a one-time delay per dtype) and print nothing when they fall back — set the
debug vars above to see routing decisions.

> Developed and tested on: M5 Max / 128 GB, macOS 27 dev beta, PyTorch
> 2.10 & 2.13, Python 3.12, ComfyUI Desktop. M1–M4 get the compatibility
> layer and flash attention; the int8/fp8 matmul kernels need M5-class
> TensorOps and fall back cleanly below that.

## Field notes (what the benchmarks taught us)

A few load-bearing lessons encoded in this repo, for future work:

- **Check the model's compute dtype per family.** Wan runs fp16 where the
  flux family runs bf16; a bf16-only kernel gate silently dropped Wan's 21.5k
  token attention onto a chunked-python path that was ~70% of the render.
  Dtype-gated routes degrade *silently* — `ASFP8_ATTN_DEBUG=1` exists for this.
- **Size gates on actual bytes, not worst-case shapes.** Two separate bugs
  came from gating on max-sequence² instead of the real `Lq×Lk`.
- **Check the sampler for hidden work.** LTX's `euler_ancestral_cfg_pp`
  disables comfy's cfg=1 optimization and double-evaluates the model; plain
  euler is frame-identical and half the cost.
- **"Launch-bound" is a hypothesis, not a diagnosis.** The Wan VAE looked
  launch-bound (dozens of tiny decoder passes); batching them changed
  nothing. Attribution showed strided reductions and cat chains — memory
  patterns, not launches — and the convs already at the platform ceiling.
- **MPS traps**: conv2d silently miscomputes with non-contiguous weights;
  per-tap rounding compounds through recursive causal VAE caches; tensor
  prints during execution go to a swallowed stdout (use stderr); the first
  call after a kernel build pays PSO compilation — benchmark run 2.

## Credits

- [ComfyUI-AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8)
  (Paweł Mazurkiewicz, MIT) — the original compatibility layer this project
  grew from, and [`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn),
  which still backs patch #5.
- [Cider](https://github.com/Mininglamp-AI/cider) (Mininglamp, MIT) — the
  register-tiled cooperative-TensorOps int8 `matmul2d` that patch #17's GEMM
  is ported from, and the design reference for the staged/swizzled fragment
  work.

## License

MIT — see [LICENSE](LICENSE).
