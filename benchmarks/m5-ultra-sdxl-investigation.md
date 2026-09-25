# SDXL: M5 Ultra scaling investigation

2026-09-25. M5 Ultra, 256 GB, macOS 27, PyTorch 2.13.0, ComfyUI 0.37.0.
No new runtime optimization was accepted or installed in this investigation.

**Follow-up:** an isolated PyTorch 2.14 comparison subsequently measured
**3.15 → 2.77 s** with the same router, and **2.39 s** when using raw PyTorch
SDPA. The original operation-shape explanation did not establish the cause
of the Max/Ultra gap and missed this available software improvement. See
[the version comparison](#follow-up-pytorch-213-vs-214) below. The live
installation is still on 2.13; outputs differ across versions/backends.

## Reproduction

The executable workflow was extracted from `sdxl_simple_00006_.png`:

- `SDXL/intorealismUltra_v90.safetensors`, no LoRA or model modifiers.
- 896×1152, batch 1, 12 Euler ancestral steps, Karras scheduler.
- CFG 1.5, denoise 1, seed 137740203719215.
- Original positive/negative conditioning and checkpoint VAE preserved.
- UNet runs FP16 on MPS; the checkpoint VAE runs BF16 on MPS.
- Native package settings enabled, as in the installed configuration.

The reference replay matches every RGB pixel of the original PNG. Repeated
reference runs also preserve complete sampler latents and decoded tensors.
PNG file bytes are not compared because diagnostic PNGs omit workflow metadata.

The first run took 7.128 s (5.320 s sampling and 1.770 s decode), with initial
GPU preparation included. The following warm runs took 3.175 and 3.194 s;
sampling took 3.008 and 2.991 s. A later interleaved control measured 3.117–3.127 s.
This is **roughly 3.1–3.2 s warm**, including decode and saving. Interpreter,
native compilation, initial checkpoint/CLIP loading and unchanged text encoding
are outside warm timing, matching reuse of unchanged ComfyUI nodes. Hashing and
diagnostic tensor saves are also outside timing. GPU completion is synchronized
at stage boundaries. The ComfyUI queue was checked before each diagnostic render.

The user's M5 Max reference is 3.5–4 s. That machine was not remeasured, and its
precise historical software state was not reconstructed. These data therefore
do **not** establish a macOS/Metal regression or quantify Max/Ultra hardware
scaling under identical software.

## Where time goes

Warm sampling is approximately 2.96–3.01 s; decode is only 0.12–0.15 s and
saving about 0.03–0.05 s. VAE changes cannot account for a large speedup here.

A separate two-step profile synchronized each measured operation. It took
1.353 s rather than the normal approximately 0.50 s because synchronization
serializes execution. The buckets below identify candidates, not additive
unprofiled GPU costs:

| Sampling operation | Calls over 2 steps | Synchronized seconds |
|---|---:|---:|
| Attention, including layout/scaling/softmax/matmuls | 280 | 0.463 |
| Linear projections | 1,486 | 0.438 |
| Convolutions | 102 | 0.106 |
| LayerNorm | 420 | 0.072 |
| GELU | 140 | 0.041 |
| GroupNorm | 92 | 0.021 |
| SiLU | 108 | 0.019 |

The bmm sub-bucket (0.179 s) is inside attention and must not be added again.
The full 12-step run has 8,916 linear calls, 1,680 attention calls and 840
transformer blocks. CFG 1.5 evaluates positive and negative conditioning
together: the observed UNet batch is 2 for a single output image.

This is a floating-point UNet workload. Its projections do not use this
package's INT8 ConvRot kernels, and the Krea2/H3 architecture-specific fusions
do not apply. The global attention router does accelerate it. The dominant
attention shapes are:

| Query / KV tokens | Heads | Head width | Dtype |
|---|---:|---:|---|
| 4032 / 4032 | 10 | 64 | FP16 |
| 4032 / 77 | 10 | 64 | FP16 |
| 1008 / 1008 | 20 | 64 | FP16 |
| 1008 / 77 | 20 | 64 | FP16 |

All take the router's short-sequence bmm path. The 4032-token shape is just
below the 4096-token flash threshold. One such self-attention score tensor is
about 620 MiB; the 1008-token score tensor is about 77.5 MiB. They are repeatedly
materialized, normalized and consumed. The many smaller projections, memory
passes and dispatches make core count alone an insufficient latency predictor.
This is an inference from the workload and controls, not a hardware utilization
trace proving a particular number of idle GPU cores.

A CPU cProfile puts substantial waiting inside `upsample_nearest2d` and
`softmax`. It does not identify that waiting as time executing those GPU
kernels: asynchronous preceding work can be charged at a later blocking call.
Synchronizing first measured only 2.3 ms total in the four sampling interpolation
calls over two steps. Replacing nearest resize did not produce a sustained win.

## Controlled experiments

| Candidate | Outcome | Decision |
|---|---|---|
| Combined self-attention Q/K/V projection | All 420 checked projection outputs and complete two-step latents/frames matched; warm sampling 0.496–0.505 s versus 0.498 s control | Neutral; not installed |
| Repeat-based nearest-neighbor resize | Exact two-step latents/frames; 0.492–0.493 s sampling versus 0.494 s final control | Neutral; not installed |
| CPU threads 12 → 1 | Exact; 0.499 s versus 0.494 s control | No gain |
| Query-chunked bmm attention | Mostly slower; 4032-token self-attention also changed bytes | Rejected |
| Stock PyTorch SDPA | Warm short run 0.488 s; changed latents; no compelling gain | Not installed |
| Native FlashAttention, all four shapes | Full warm run 3.122 → 2.900 s; changed latents and images | Diagnostic only |
| `PYTORCH_MPS_PREFER_METAL=1` | Final warm short sampling 0.497 s, same bytes | No gain |
| `PYTORCH_MPS_FAST_MATH=1` | Final warm short sampling 0.498 s, same bytes in this test | No gain; no precision-policy change adopted |

The two environment settings were tested separately in fresh processes. These
results do not establish that either setting changed the kernel selected by
every operator. See the [PyTorch MPS environment documentation](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)
for their intended scope; the measured runtime here is 2.13.0.

Batch 2 took 5.827–5.856 s warm, or about 2.92 s/image. This improves throughput
only modestly and does not lower single-image latency or validate a 2× scaling
claim. Changing batch size was a diagnostic, not a workflow change.

## Full FlashAttention comparison

Same loaded model, prompt, seed and 12 steps; first baseline excluded as warmup:

| Run | Sampling | Total |
|---|---:|---:|
| Baseline A | 2.966 s | 3.127 s |
| Flash A | 2.743 s | 2.904 s |
| Flash B | 2.735 s | 2.896 s |
| Baseline B | 2.956 s | 3.116 s |

Mean total decreases by about **0.222 s / 7.1%**. Thus, crossing the router
threshold helps, but does not explain a missing 2× speedup on its own.

Against the reference, the second Flash run has latent max absolute error
10.679 and RMSE 0.13985; decoded-frame MAE 0.00348, maximum error 0.32227 and
PSNR 42.24 dB on the [0,1] frame scale. Numerical similarity is not a quality
acceptance test. No multi-prompt quality assessment was performed, so this
route remains a local experiment and does not replace the exact default.

## Next kernel work

The useful targets are the FP16 projection/GEGLU path and attention layout and
softmax memory traffic. Candidate kernels should combine projection epilogues
with activation or layout work, or preserve the current bmm attention's
rounding while removing intermediate passes. Simply concatenating Q/K/V weights
was already tested and did not yield a practical gain by itself. A replacement
FP16 GEMM would need intermediate checks because accumulation order can change.

The investigation found no accepted new bit-exact speedup and no demonstrated
route to 1.75–2 s. It narrows the next work to SDXL-specific FP16 kernels rather
than another INT8 scheduling adjustment. Runtime defaults remain unchanged.

Machine-readable timings, profile shapes, errors and guards:
[JSON](data/m5-ultra-sdxl-investigation.json).

## Follow-up: PyTorch 2.13 vs 2.14

The user's version question prompted a controlled test on the same Ultra,
macOS, workflow, checkpoint, seed and ComfyUI source. Official PyPI wheels
for torch 2.14.0 and matching torchvision 0.29.0 were installed into a separate
diagnostic directory. The custom node was copied with only its native build
cache paths changed, so 2.14 extensions did not overwrite the 2.13 caches.
Other application dependencies were reused. The live environment was not changed.

Subsequent cross-model testing found that this first 2.14 copy's extension
builds failed: the loaders requested C++17, but 2.14 headers require C++20.
The SDXL measurements above still exercise the reported short-sequence BMM
and raw SDPA routes; this FP16 SDXL workflow does not use the failed INT8/DiT
extensions. They do **not** validate native-extension compatibility with 2.14.
The later Z-Image/Krea2/H3 comparison rebuilds the isolated extensions with
C++20 and asserts that Flash41, INT8 and DiT kernels actually load before
benchmarking.

| PyTorch | SDXL attention | Warm sampling | Warm total |
|---|---|---:|---:|
| 2.13.0 | Current package router | 2.975 s | **3.148 s** |
| 2.13.0 | Raw PyTorch SDPA | 2.926 s | 3.091 s |
| 2.14.0 | Current package router | 2.582 s | **2.773 s** |
| 2.14.0 | Raw PyTorch SDPA | 2.217 s | **2.394 s** |
| 2.13.0, repeated afterward | Current package router | 2.976 s | 3.147 s |

Each row averages two warm full 12-step renders. Within the first two
processes the order was baseline warmup, baseline, SDPA, SDPA, baseline.
A fresh 2.13 process afterward reproduced its original timing. Thus the
approximately **0.754 s / 24% reduction** is not explained by a warmup-only
comparison. Upgrading while preserving the router yielded approximately 12%
less total time; obtaining 2.39 s required changing attention routing too.

"Raw PyTorch SDPA" explicitly bypassed both this package's bmm router and
the separate mtlflashattn interception. Selecting a nominal PyTorch backend
without accounting for that second wrapper is not necessarily the same test.

PyTorch 2.14 adds an MPP prefill-attention implementation for these FP16/head-64
shapes. This was verified in the released `Attention.mm` source, not inferred
solely from the headline benchmark. The current router's explicit bmm/softmax
sequence cannot select it. Sources: [release announcement](https://pytorch.org/blog/pytorch-2-14-release-blog/),
[attention implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/Attention.mm),
[merged implementation PR](https://github.com/pytorch/pytorch/pull/182256).

With the current router unchanged, a synchronized two-step profile also showed
attention including layout work improving from 0.463 to 0.377 s and GELU from
0.041 to 0.029 s. Linear calls changed only from 0.438 to 0.426 s, while
convolution was nearly unchanged (0.106 to 0.105 s). These profiled times include
per-operation synchronization and are not additive predictions of the warm run.
They locate the improvement primarily in attention and adjacent operations;
they do not demonstrate doubled projection throughput on the Ultra.

### Numerical results and rollout status

The initial CPU noise was byte-identical across versions:
`fc733c7d27549b38225575b933b591ddd415da511c1c670de789def23671fa73`.
Every repeated configuration reproduced its own complete latent, frame and
pixel hashes. **Neither 2.14 configuration matched the 2.13 reference bytes.**
The upgrade itself is therefore not an exact-preserving change to this workflow.

| Against 2.13 current-router reference | Frame MAE | Frame maximum error | Frame PSNR |
|---|---:|---:|---:|
| 2.14, current router | 0.00490 | 0.31982 | 39.10 dB |
| 2.14, raw SDPA | 0.00645 | 0.49316 | 36.34 dB |

Frames are on the [0,1] scale. These metrics do not establish perceptual quality
across prompts. No live upgrade or new default was installed. A wider migration
also needs other models revalidated: for example, the accepted Krea2 exact
normalization fusions explicitly require torch 2.13 and decline 2.14 today.

### What the older records establish

The pre-Ultra commit `c0560df` (2026-08-15) already documented M5 Max / 128 GB,
macOS 27 developer beta, Python 3.13 and **PyTorch 2.13.0** in its release-validation
section. It does not record the specific 3.5–4 s SDXL run together with a workflow,
checkpoint hash, exact runtime commit and attention route. That omission prevents
a controlled reconstruction and should not be replaced with an assumed version.
The evidence does not show that the old Max result used 2.14.

The new diagnostic harness automatically records the PyTorch version and source
commit, import path, Python version, macOS build, ComfyUI commit, relevant package
versions, node source path and performance environment variables for each run.
The workflow and source PNG references, stage timings and output hashes remain
recorded separately. Full data: [version comparison JSON](data/m5-ultra-sdxl-torch-comparison.json).

The result establishes a real improvement available through the runtime and
attention backend. It does not prove why the original Max/Ultra ratio was small,
nor that a matched Max running 2.14 would retain its historical 3.5–4 s time.
