# SeedVR2: bounded tiled VAE memory on MPS

2026-09-24. M5 Ultra, 80 GPU cores, 256 GiB unified memory; macOS 27,
PyTorch 2.13.0, ComfyUI `b33e2b55` / 0.37.0. Baseline custom node: `db2a013`.

The SeedVR2 spatial tile loop grows Metal temporary allocations between tiles
even while live tensor memory stays nearly flat. Waiting for MPS completion
after each VAE tile bounds those allocations. The change preserves the tested
encoder latents, sampler latents and complete decoded frame bytes.

## Requested 2400×1600 → 7200×4800 run

The original photograph was downsampled to 2400×1600 with Lanczos, then passed
through the workflow's 3× Lanczos resize, SeedVR2 preprocessing, tiled VAE
encode, INT8 7B sampling, tiled VAE decode and postprocessing. Output is
7200×4800 RGBA, 34.56 megapixels. VAE tiles are 512 pixels with 128-pixel
overlap: 19×13 = 247 tiles for encoding and 247 for decoding.

Checkpoints: `seedvr2_7b_int8_convrot.safetensors` and
`seedvr2_ema_vae_fp16.safetensors`. Sampler: one Euler step, simple schedule,
CFG 1, denoise 1, seed 959948902156062. Color correction remains `none`.

| Stage | First run (s) | Warm repeat (s) |
|---|---:|---:|
| Encode | 16.761 | 16.352 |
| Sample, including device model loading when needed | 24.717 | 18.907 |
| Decode | 36.035 | 35.860 |
| Save PNG | 1.900 | 1.959 |
| **Complete node sequence** | **80.148** | **73.413** |

Total includes resize, preprocessing, checkpoint construction when needed,
conditioning, postprocessing and PNG writing. The harness invokes the actual
node implementations and synchronizes stage timers; app startup and post-run
hashing/tensor saves are excluded. These are direct node-sequence timings, not
server-history execution timestamps.

Maximum observed Metal driver allocation across both runs was **40.264 GiB**;
maximum sampled live GPU tensors were **27.473 GiB**. Metal allocation includes
allocator pools and framework allocations, and is not total system RAM usage.
Driver allocation, live tensors and CPU RSS overlap and must not be added.
The second run reused the resident model and allocator pool.

With the fix disabled, this requested resolution still showed the original
encoder growth. The diagnostic watchdog stopped it at **17.58 GiB** observed
Metal allocation, with only about **0.98 GiB** in live tensors. It was deliberately
stopped before OOM; there is no complete disabled-run timing at this resolution.

## Evidence from the original 104 MP failure

The original workflow resized a 6240×4160 image by 2×, producing
12480×8320 and 726 encoder tiles. Its crash log ended immediately after loading
the FP16 VAE, before the SeedVR2 7B diffusion model was loaded.

Bounded tests at those original dimensions reached **17.44 GiB after nine
encoder tiles** with native patches enabled, and **16.69 GiB** with the entire
custom node disabled. Live tensors stayed near 1.18 GiB. A synthetic-pixel test
of all 726 encoder tiles with per-tile synchronization completed below
**5.22 GiB** observed Metal allocation. This establishes an encoder scheduling
problem in the tested stack, not an INT8 checkpoint-size problem. It does not
measure the full 7B workflow at that larger resolution or reproduce its 250 GB
crash peak.

## Scope and activation

`ASFP8_SEEDVR2_VAE_SYNC` defaults to **1** on MPS, independently of
`ASFP8_NATIVE` and MiniMax relaxed flags. Restart ComfyUI after installation.
Set `ASFP8_SEEDVR2_VAE_SYNC=0` before startup to restore original scheduling.

The custom node wraps SeedVR2's spatial tiling entry point and its encoder and
decoder tile calls. A context-local model identity restricts synchronization to
that model's active MPS tile operation. Context is restored on normal return or
exceptions; non-tiled calls, unrelated models, CPU and CUDA retain their
original behavior. No core ComfyUI file is edited. Precision, weights, tile
overlap, temporal caching and arithmetic are unchanged. The cost is reduced
overlap between CPU submission and GPU execution; this is a memory safeguard,
not a new numerical approximation or a claimed universal speedup.

## Verification

- Both complete 7200×4800 runs produced finite output and identical raw-frame
  hashes; encoded and sampled latents also matched between repetitions.
- A separate small full workflow (320×213 input, 960×638 final output after
  the existing even-dimension crop) compared the fix enabled and disabled.
  Saved encoder latents, sampler latents and complete FP32 RGBA frames are
  byte-identical, verified from tensors as well as SHA-256 hashes.
- A fresh process loaded the installed node with the new flag **unset** and
  reproduced all three small reference hashes, confirming default activation.
- Seven focused tests pass: argument/result preservation, per-tile waiting,
  CPU/CUDA bypass, non-tiled and unrelated-model bypass, exception cleanup,
  nested context restoration and disabled/idempotent startup guards.
- The prepared shared-folder API workflow passed ComfyUI prompt validation.
- The full output preview was inspected; no blank or corrupted image was seen.

Three standalone validation attempts hit a Metal command-buffer assertion
before image processing or any VAE tile call. Moving memory telemetry off the
background thread alone did not resolve it. Explicitly initializing MPS before
custom-node imports in the harness allowed the subsequent numerical comparison
and installed-default check to pass. No production workaround for that separate
startup assertion was added; the responsible dependency was not isolated.

The successful large runs used independent RSS/system-memory watchdogs and an
80 GiB observed-memory stop. The disabled baseline used a 16 GiB stop. These
diagnostic limits are not installed as user launch settings.

Machine-readable measurements: [m5-ultra-seedvr2-memory.json](data/m5-ultra-seedvr2-memory.json).
The workspace's `diagnostics/seedvr2-memory/` and `diagnostics/seedvr2-3x/`
retain scripts, raw logs, memory events, original workflow, snapshots, output
images, small reference tensors and installation backups. Original user files
and model checkpoints are unchanged; the test input/workflow use new filenames.
