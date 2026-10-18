# MOSS SFT with the existing H3 dense INT8 kernels

Measured 2026-10-18 on M5 Ultra, 80 GPU cores, 256 GiB, macOS 27,
PyTorch 2.14.1 and ComfyUI nightly `c06bf5f569d268b80076e5f44126f87cdf835869`
(after v0.41.3). These are optional diagnostic adaptations, not new defaults.

## Scope and implementation

Tested the pack's existing CCV-derived dense INT8 QK/PV and INT8 QK/BF16 PV
kernels against the current optimized exact MOSS path. No SOL, sparse routing,
step protection, text-query protection or token reduction was used. Both dense
paths retain every key. This is not a CUDA Kitchen equivalence claim.

The full INT8 wrapper already supports MOSS's 24 heads. The floating-PV wrapper
needed a diagnostic specialization from 56 to 24 heads, including allocations,
scratch accounting, dispatch dimensions and three shader head constants/guards.
It runs in a separate module without changing the original H3 runtime or shader.
The [adapter](tools/moss_h3_dense_probe.py) is deliberately diagnostic-only and
fails if its expected source declarations change.

The initial experiment replaces all 2,100 joint-attention calls in a 25-step
render, across every layer and step, including their text query rows. Four short
text-embedder attention calls initially remain native; the exact text reuse
optimization reuses their results on later steps. A follow-up replaces those
short calls as well. The Qwen text encoder and VAE are outside this MOSS DiT
attention experiment. Both candidates explicitly use `exact_prefix=False` and
`video_start=0`; full INT8 uses `tau=0`, so no attention blocks are skipped.

## Workflow and measurement

- SFT `moss_image_sft_int8_convrot.safetensors`;
  `qwen3.5_9b_int8_convrot.safetensors`; `flux2-vae_small.safetensors`.
- 25 steps, CFG 2.5, ER-SDE/simple, seed `856771198829162`.
- 1024×1024 and 1248×1248 (1.558 MP); prompt enhancer and SLG disabled.
- Synthetic night-street fashion prompt with an illuminated “MOSS” theatre
  marquee, a white wool coat, wet asphalt and warm/cool lighting.
- Same prompt, negative prompt, conditioning, initial latents and noise per
  resolution. Full tensor hashes verify the identical inputs.
- End-to-end includes repeated text encoding, complete sampling, VAE and PNG
  saving. CPU tensor capture/hashing is outside this timed interval. Sampling
  includes layout conversion, quantization, allocation and all model work.
- Exact control before/after candidates; model-loading run excluded. All
  approximation repeats produced byte-identical latents and decoded frames
  within their own mode. Restoring exact recovered the original exact bytes.
- Isolated benchmark server, guarded memory, no changes to the user's server.
  Peak Metal driver allocation was approximately 26.45 GiB at 1024 and
  27.37 GiB at 1248 in the initial comparison. No memory guard tripped.

## Initial joint-attention results

Warm complete renders, seconds (two runs per mode):

| Resolution | Exact | Dense INT8 QK/PV | INT8 QK / BF16 PV |
|---|---:|---:|---:|
| 1024×1024 | 22.241 | 22.065 | 22.174 |
| 1248×1248 | 33.815 | 33.447* | 33.290 |

At 1024, differences are under 0.2 seconds and do not establish a useful
improvement. Floating PV saves approximately 0.53 seconds / 1.6% at 1248.

\* The first full-INT8 1248 render includes a shape-specific startup cost
(33.886 s); its repeat takes 33.007 s. Do not treat the average as a clean
steady-state comparison. The unrestricted follow-up explicitly warms each
backend/shape with a one-step render before its full-step measurements.

## Fully unrestricted follow-up

The user requested removal of all remaining attention protection. This follow-up
also sends the text-embedder's 83-token and 23-token calls through the selected
kernel: **2,104 calls per render**, consisting of 1,050 calls at 6,167 tokens,
1,050 at 6,107 tokens, two at 83 and two at 23. All 42 joint-attention layers,
all 25 steps and all text/image query rows use the chosen dense mode. No native
MOSS attention remains in these candidate runs. Exact text-result reuse stays
enabled, as it is in the control; the external Qwen encoder is unchanged.

Each mode first receives a one-step warm-up. Then two full 25-step renders are
measured per mode, with exact controls bracketing the candidates. This removes
the earlier full-INT8 shape compilation cost from the timed comparison.

| Mode | End-to-end mean | Repeated E2E range | Sampling mean | E2E speedup |
|---|---:|---:|---:|---:|
| Optimized exact | 33.808 s | 33.666–33.950 s | 31.990 s | 1.000× |
| Dense INT8 QK/PV | 32.978 s | 32.954–33.001 s | 30.993 s | 1.025× |
| INT8 QK / BF16 PV | 33.210 s | 33.193–33.227 s | 31.222 s | 1.018× |

Full INT8 saves 0.83 s end-to-end (2.46% less time); floating PV saves 0.60 s
(1.77% less time). The sampling-only reductions are approximately 3.12% and
2.40%. Inputs are identical; repeated latents and decoded outputs are byte-equal
within each mode. Restoring exact recovers the exact reference. Peak Metal
driver allocation during the follow-up is approximately 29.89 GiB; no guard
trips or nonfinite outputs occurred.

The first test already had no prefix or step protection on the expensive joint
attention. Extending to four short cached text-attention calls does not unlock a
large performance gain. Full INT8 decoded frame RMSE is 0.08123; floating PV is
0.05023. Both images are coherent, with changed clothing/pose and background
lighting details. The lower error than the initial experiment on this one seed
is not evidence that removing protection generally improves quality.

![1248 unrestricted: exact, dense INT8, floating PV](artifacts/moss-h3-dense/1248_unrestricted-comparison.jpg)

## Numerical and visual findings

Actual MOSS tensors, including their original strided layout, were screened
with layout/quantization/allocation included. At 4,179 tokens the warmed exact
operator took 2.084 ms, full INT8 2.107 ms and floating PV 2.043 ms. At 4,119
tokens they took 1.908, 1.967 and 1.976 ms respectively. The first exact timing
series had GPU ramp-up and is retained in the data, but not used to claim a gain.
This shows why H3's attention win does not automatically transfer to MOSS's
shorter sequences: the complete quantized operation is close to exact here.

| Compared with exact | Full INT8 | INT8 QK / BF16 PV |
|---|---:|---:|
| Attention RMSE, 4,179 tokens | 0.10031 | 0.04361 |
| Attention maximum absolute error, 4,179 tokens | 3.375 | 1.08594 |
| Decoded frame RMSE, 1024 | 0.14071 | 0.13939 |
| Decoded frame RMSE, 1248 | 0.09063 | 0.06004 |

Frame RMSE is over decoded float RGB; it is not a perceptual quality score.
The images remain coherent and the marquee text remains legible. Approximation
changes pose, clothing, framing, lighting/reflections and background details.
At 1248 the floating-PV image stays visibly closer to exact than full INT8.
At 1024, both alter composition noticeably despite floating PV's lower operator
error. One prompt/seed at two sizes cannot certify general quality preservation.

![1024: exact, dense INT8, floating PV](artifacts/moss-h3-dense/1024-comparison.jpg)
![1248: exact, dense INT8, floating PV](artifacts/moss-h3-dense/1248-comparison.jpg)

## Validation and disposition

The 24-head floating-PV specialization matches the first 24 heads of the
unchanged 56-head H3 backend byte-for-byte at 65 and 1,153 tokens (real captured
heads, remaining heads zero padded). Both full-INT8 repeat checks also pass.
Existing H3 dense/floating-PV unit tests: **26 passed**.

The exact implementation remains the default. These are experiments, not
accepted replacement kernels: the tested tradeoff is visible output changes
for small or negligible time savings. No new production preset is enabled.

[Raw measurements, hashes and source manifest](data/m5-ultra-moss-h3-dense.json).
