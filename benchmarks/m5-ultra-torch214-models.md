# M5 Ultra: matched PyTorch 2.13 / 2.14 model benchmarks

2026-09-25. M5 Ultra, 80 GPU cores, 256 GB; macOS 27.0 build 26A428;
Python 3.13.12; ComfyUI 0.37.0, commit `b33e2b55cae074eca5aec96283cceac19aa249ba`.
PyTorch 2.13.0 (`cf30153c4c131c8164ee7798e5022d810682e2cb`) versus 2.14.0
(`08187d9e0fba026dc8217405802ab5381dc88d90`, torchvision 0.29.0).
All other application dependencies and model files are shared.

**Outcome:** PyTorch 2.14 raw SDPA improves Z-Image and H3, but simply upgrading
regresses Krea2 because its accepted exact glue is version-gated. The isolated
Krea2 compatibility experiment below separates that missing optimization from
the attention comparison. No production runtime, attention default or version
guard was changed.

## Matched warm results

Seconds, means of two measured warm runs per row. `ours` retains the package's
attention router and Flash41/GQA kernels. `torch` calls the real PyTorch SDPA,
bypassing both our router and mtlflashattn interception. All projection,
normalization, RoPE and quantization optimizations eligible for that runtime
remain enabled. Krea2 glue is enabled on 2.13 and guarded off on 2.14 in this
table; both 2.14 attention modes therefore have the same fusion setting.

| Workflow | PyTorch | Attention | Sampling loop | Video/image VAE | E2E | E2E range |
|---|---|---|---:|---:|---:|---:|
| Z-Image, 1024² / 8 steps | 2.13.0 | ours | 3.621 | 0.146 | **3.821** | 3.766–3.877 |
| Z-Image, 1024² / 8 steps | 2.14.0 | ours | 3.524 | 0.129 | **3.694** | 3.690–3.699 |
| Z-Image, 1024² / 8 steps | 2.14.0 | torch | 3.314 | 0.137 | **3.505** | 3.501–3.510 |
| Krea2, 1448² / 8 steps | 2.13.0 | ours | 14.580 | 0.391 | **15.055** | 15.039–15.070 |
| Krea2, 1448² / 8 steps | 2.14.0 | ours | 16.363 | 0.357 | **16.812** | 16.805–16.820 |
| Krea2, 1448² / 8 steps | 2.14.0 | torch | 15.436 | 0.360 | **15.902** | 15.885–15.919 |
| H3, 864×480 / 22 frames / 20 steps | 2.13.0 | ours | 19.338 | 1.879 | **21.576** | 21.554–21.599 |
| H3, 864×480 / 22 frames / 20 steps | 2.14.0 | ours | 19.363 | 1.913 | **21.659** | 21.626–21.692 |
| H3, 864×480 / 22 frames / 20 steps | 2.14.0 | torch | 17.422 | 1.497 | **19.286** | 19.273–19.299 |
| H3, 864×480 / 124 frames / 20 steps | 2.13.0 | ours | 148.719 | 13.209 | **163.003** | 162.971–163.035 |
| H3, 864×480 / 124 frames / 20 steps | 2.14.0 | ours | 149.113 | 13.199 | **163.371** | 163.170–163.573 |
| H3, 864×480 / 124 frames / 20 steps | 2.14.0 | torch | 141.062 | 10.259 | **152.322** | 152.048–152.596 |

Full H3 with 2.14 raw SDPA averages **7.053
s/step**, saving **10.68 s E2E
(6.55%)** versus the fresh 2.13 control.
The 2.14 attention-only comparison saves
**11.05 s E2E**.
Audio decode and MP4 export are included; their individual times are in the data.
The long clip is 124 / 24 = 5.167 seconds. Short-clip gains must not be
extrapolated to the longer attention sequence.

## Krea2 fusion compatibility experiment

Only the isolated copy bypasses the 2.13 guard, using
`ASFP8_BENCH_KREA214_GLUE=1`; it is not a shipped user flag. Both attention
modes are rerun with the same 1448², LoKr 0.8 workflow.

| Attention, with glue enabled | Sampling (s) | E2E (s) | Latents and frames identical to corresponding 2.14 glue-off mode |
|---|---:|---:|---|
| ours | 14.590 | **15.052** | True |
| torch | 13.548 | **14.014** | True |

The saved real-block normalization/modulation probe and exhaustive finite-BF16
SiLU/sigmoid probes report:

```json
{
  "torch": "2.14.0",
  "norm_mod": {
    "bit_equal": true,
    "different_bytes": 0,
    "max_abs": 0.0
  },
  "silu_all_finite_bf16": {
    "bit_equal": true,
    "different_bytes": 0,
    "max_abs": 0.0
  },
  "sigmoid_all_finite_bf16": {
    "bit_equal": true,
    "different_bytes": 0,
    "max_abs": 0.0
  }
}
```

Complete-output parity is scoped to this workflow; the version guard remains
unchanged in the repository and live installation. The additional runs are
reported separately from the existing-stack comparison.

## Workflows and reference validation

The [frozen API graphs](data/m5-ultra-torch214-workflows.json) preserve every
active input. Inactive Krea2 utility/prompt-expansion nodes are replaced with
their resolved values, as in the previous Krea2 benchmark.

- Z-Image: kitten/windowsill prompt, seed 630651083309421, 1024×1024,
  8 res_multistep/simple steps, CFG 1, flow shift 3; INT8 ConvRot Turbo,
  `qwen_3_4b.safetensors`, `ae.safetensors`.
- Krea2: original perfume/nature prompt, including trailing comma-space;
  seed 535122089142682; 1448×1448, 8 Euler/simple steps, CFG 1;
  INT8 ConvRot Turbo, Qwen3VL 4B FP8, Qwen image VAE,
  `snofs_krea_v1_3D.safetensors` LoKr at 0.8. Prompt expansion stays off.
- H3: original red-paper-boat prompt, seed 812764035; 864×480,
  22 or 124 frames, 20 res_multistep/simple steps; INT8 ConvRot denoiser,
  Qwen3VL 32B INT8 ConvRot encoder, **FP16 video VAE and FP32 audio VAE**.
  No relaxed preset, sparse attention, cache approximation or INT8 video VAE.

Initial noise **and text conditioning match byte for byte across runtimes**
for all four cases. The fresh Krea2 2.13 latents/frames match the previous
accepted reference. H3 2.13 historical latent match:
**True**; historical decoded-frame match:
**True**, SHA-256
`538f33de59395e714e0fb87ade8011dfe61f4131d11f34ba5c142d72dd71d5be`.
All rendered tensors are finite. Repeats within each configuration reproduce
their own bytes. All full videos have 124 frames at 24 fps and an audio stream.

## Numerical changes

Krea2 with our attention remains byte-identical across these runtimes, with or
without its exact glue. Z-Image and H3 change when upgrading. Switching to raw
PyTorch attention changes outputs for all three models.
`upgrade` compares 2.13/ours with 2.14/ours; `attention214` compares the two
attention implementations under 2.14. Frame tensors are FP32 on [0,1].

| Workflow | Comparison | Byte-identical frames | Mean absolute error | Maximum absolute error | PSNR (dB) |
|---|---|---|---:|---:|---:|
| Z-Image | upgrade | False | 0.008150 | 0.675781 | 34.64 |
| Z-Image | attention214 | False | 0.010170 | 0.707031 | 32.18 |
| Krea2 | upgrade | True | 0.000000 | 0.000000 | inf |
| Krea2 | attention214 | False | 0.021365 | 0.876953 | 25.22 |
| H3 short | upgrade | False | 0.036968 | 0.703497 | 23.61 |
| H3 short | attention214 | False | 0.028093 | 0.695286 | 26.06 |
| H3 full | upgrade | False | 0.055052 | 1.000000 | 18.26 |
| H3 full | attention214 | False | 0.058089 | 0.932336 | 18.57 |

Representative exported frames were visually inspected. H3 retains the paper-boat
scene, but boat geometry/position, water ripples and reflections visibly differ.
Krea2 retains the perfume scene, but generated label text and flower placement
change with PyTorch attention. The local `h3-comparison.jpg` (frames 0, 60, 123)
and `krea-comparison.jpg` in the diagnostic folder preserve those comparisons.
This inspection does not establish motion stability or quality across prompts.

These are performance and numerical checks on the established workflows, not a
multi-prompt perceptual-quality qualification. Latent and audio errors and all
individual hashes are in the [measurement data](data/m5-ultra-torch214-models.json).

## Build compatibility and attention audit

The original loaders explicitly request **C++17**. PyTorch 2.14's headers
require **C++20**, causing native extension builds to fail. The isolated copy
uses C++20 and a separate `/tmp/asfp8_models214` build cache. Flash41, INT8 and
DiT availability is asserted before benchmarking; no latched runtime fallback
is allowed. The earlier SDXL result used BMM/SDPA and does not itself establish
that these native extensions work under 2.14; its report now states this limit.

The full H3 ours mode makes 1,000 Flash41 calls at `[1,56,15458,128]` per
20-step run. The torch mode instead makes raw SDPA calls at the same shape.
H3 video VAE calls use `[4,32,1797,64]` and `[1,32,1797,64]` and switch with
the attention mode. Z-Image uses 4,096/4,192-token denoiser attention. Krea2
uses 224 direct GQA calls with 48 query heads / 12 KV heads at 8,306 tokens;
the raw route preserves GQA instead of changing QK preparation or repeating KV.
Z-Image and Krea2 spatial VAE attention stays on its existing implementation.

## Reproduction and timing boundaries

The local harness is `../diagnostics/torch214-models/` relative to the repository.
`METHOD.md`, `run.py`, `bench_node/__init__.py`, `launch.py`, `series.py`,
`krea_glue_probe.py`, `summarize.py` and `validate.py` preserve the test procedure.
The data archive includes source/harness hashes, asset filenames, sizes, mtimes
and safetensors-header hashes (not full weight hashes), runtime flags, warmup
markers, every measured run, route counts and validation results.

Each process uses an isolated ComfyUI server on port 8193. GPU runs are serial.
For 2.14: ours warmup, torch warmup, ours, torch, torch, ours. For 2.13:
ours warmup, ours, ours. Unique sampler inputs force execution while loader and
text-conditioning nodes remain cached. MPS synchronization surrounds sampling
and decode, and occurs at each sampling callback as in the previous H3 tests.
E2E is Comfy's execution_start through execution_success, including decode and
PNG/MP4 export. Preview and embedded output metadata are disabled. Reference
tensor dumps, byte hashes and comparisons happen after execution_success.
The older H3 harness wrote small latent dumps inside E2E, so the fresh matched
control is the primary baseline, rather than subtracting from an old headline.

V3 audio/export dispatch timing was added after the 2.14 short H3 screen;
that screen still has valid sampling, video decode and E2E values, but lacks
separate audio/export timers. All full-clip rows use the corrected instrument.
Loading, compilation and first-use warmups are excluded from table means.
