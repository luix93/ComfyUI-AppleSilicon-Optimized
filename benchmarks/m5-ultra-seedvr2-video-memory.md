# SeedVR2 video memory investigation — M5 Ultra, PyTorch 2.14

2026-09-27–28. M5 Ultra 80 GPU cores, 256 GB unified memory, macOS 27.0.
PyTorch 2.14.0, ComfyUI `b33e2b55cae074eca5aec96283cceac19aa249ba`.

## Reproduction and cause

The saved `utility_seedvr2_3b_int8_upscale_video.json` actually selects
`SeedVR2/seedvr2_7b_int8_convrot.safetensors`. The user confirmed keeping 7B.
Input is `video/MiniMax_H3_00024_.mp4`: 416×736, 24 fps, 243 frames, with
audio. The saved workflow trims five seconds and resizes 2× to 832×1472.
It uses the FP16 SeedVR2 VAE, 512-pixel spatial tiles, 128-pixel overlap,
one Euler/simple step, CFG 1, seed 959948902156062, and no color correction.
“Split Latent” is disabled and remains disabled.

Our existing MPS safeguard waits after each **spatial tile**. For a video,
one spatial tile processes the entire clip, including many internal temporal
slices. Queued Metal/MPSGraph temporary buffers can therefore accumulate
before the first tile finishes. The issue appears during VAE encoding,
before the 7B sampler is loaded; model size is not its immediate cause.

The workflow's temporal settings read 64/8, but this ComfyUI revision's
SeedVR2-owned `encode_tiled`/`decode_tiled` deliberately ignore those external
settings to preserve causal memory. The internal encoder slice size is four
frames, and the decoder slice size is one latent frame. The fix preserves
those choices, temporal caches, spatial overlap, and blending order.

The original nine-frame encoder crossed a deliberate 24 GiB cutoff while
live tensors peaked at only 2.68 GiB. With residual-block waits the same
encode completed at 10.04 GiB measured Metal allocation. The old schedule
also reached 50.99 GiB on a complete nine-frame run. The new schedule
completed two matching native-ComfyUI video runs below 31.26 GiB.
The saved five-second input crossed the 48 GiB cutoff during encoding,
before finishing the first spatial tile: 48.13 GiB driver allocation versus
3.36 GiB live tensors at that instant. The process was stopped deliberately;
this is not a completed baseline or a system OOM.

## Fix and exactness

`_patches/seedvr2_vae_mps.py` extends the existing scoped synchronization
safeguard inside video tiles. It waits after residual blocks only during an
MPS tiled SeedVR2 video encode/decode. Image scheduling, other VAEs, CPU,
CUDA, and non-tiled calls retain their existing behavior. The scope resets
after normal completion, nested calls, and exceptions.

No arithmetic, weights, precision, kernel selection, tile size, temporal
context, or sampler settings change. The complete nine-frame comparison
using ComfyUI's own video reader preserves the encoded latents, sampled
latents, final decoded frame tensor, and exported H.264/AAC file byte for byte.
The PyAV-only early
screens use a different input conversion and are compared only against
each other, not against the native reader's hashes.

The safeguard remains enabled by default, independently of native/relaxed
kernel flags. `ASFP8_SEEDVR2_VAE_SYNC=0` disables the entire safeguard;
`ASFP8_SEEDVR2_VAE_VIDEO_SYNC=0` restores the earlier spatial-only schedule
for controlled diagnostics. Neither setting is recommended for this video
workflow.

## Safe test method

GPU tests run sequentially, with execution-thread memory probes after VAE
convolutions/residual blocks and sampler blocks. Baseline reproductions are
stopped at 24–52 GiB rather than allowed to reach system OOM. Longer runs
use a 64 GiB sampled-memory cutoff. Independent RSS/system-availability
watchdogs retain at least 64 GiB system-available memory and terminate a
process after 900 seconds. MPS high/low watermarks are 0.4/0.3 for these
diagnostic processes only. These are test safeguards, not global settings
installed into the user's ComfyUI launcher.

“Metal allocation” is sampled `torch.mps.driver_allocated_memory()` and
includes allocator-pool/driver buffers. It is not a precise total-system
RAM measurement or a mathematical upper bound on every transient allocation.

## Scheduling experiments

On the same 25-frame input, waiting only after an existing temporal slice
preserved all output bytes but saved just 0.23 s (27.81 → 27.58 s warm),
while increasing measured Metal allocation from 31.51 to 43.39 GiB. Rejected:
the small timing difference does not justify the larger memory footprint.

Waiting after each convolution was also rejected: 28.53 s warm and 32.82 GiB,
versus 27.81 s / 31.51 GiB for block waits. Every completed candidate retained
the same encoded/sample/frame hashes on this screen.

For context, the **25-frame processing screen** took 34.18 s with stock
2.14 arithmetic plus the scheduling safety guards, versus 27.81 s with our
existing kernels plus block waits (18.6% less time). Sampling was 6.02 →
2.44 s; encoding 7.94 → 7.36 s; decoding 19.89 → 17.69 s. These screen
totals include resize, encoding, sampling, decoding and a first-frame PNG,
but not video export. They are not full five/ten-second stock benchmarks.
The stock and optimized stacks do not produce identical bytes; the new
scheduling fix is exact relative to the existing optimized stack.

## Installed-node full video validation

The five-second run uses ComfyUI's own trimmed-video reader, the original
7B checkpoint and full sampler temporal context, and ComfyUI's default
H.264/AAC export. Input loading and model-independent initialization occur
before the timed processing loop. Each loop recomputes resize, preprocessing,
VAE encoding, the sampler, VAE decoding, postprocessing, a preview PNG, and
video export. No tensor-hash verification or extension compilation is charged
to processing time.

| Clip / output | First processing run | Warm processing run | Maximum sampled Metal allocation |
|---|---:|---:|---:|
| 5 s / 120 frames / 832×1472 | 139.63 s | **138.41 s** | **36.61 GiB** |
| 10 s / 240 frames / 832×1472 | 279.27 s | **278.60 s** | **47.53 GiB** |

The five-second warm breakdown is encode **35.69 s**, sample **14.34 s**,
decode **85.98 s**, export **0.93 s**, with the remainder in resize/preprocessing
and the preview PNG. Both runs preserve identical encoded/sample/frame hashes;
the saved videos contain exactly 120 frames and five seconds of video/audio.
The two exported five-second MP4 files also have identical SHA256 hashes.
The external watchdog observed at least **166.27 GiB system-available memory**.
This is a measured reserve during these runs, not a universal memory guarantee.

The ten-second warm breakdown is encode **70.65 s**, sample **32.24 s**,
decode **171.01 s**, export **1.87 s**. Both runs have identical encoded,
sampled and decoded-frame hashes and finite output tensors. The exported
videos contain 240 frames and ten seconds of video/audio. The external
watchdog observed at least **144.92 GiB system-available memory**. Peak process
RSS was approximately 25.9 GiB; RSS and Metal allocations are distinct
measurements on unified memory and should not simply be added as a total.

Doubling duration increased warm processing time by **2.01×**, while measured
Metal allocation rose by 10.92 GiB. The existing bounded attention/gather
and INT8 routes handle this longer full-context sampler without enabling
latent splitting. This establishes these two tested clip sizes, not an
unlimited-length guarantee.

The pre-fix five-second run was deliberately stopped at its allocation cutoff,
so no full-workflow before/after speed percentage is claimed. This change
principally fixes memory scheduling; it does not make the video VAE much faster.

## Image regression and artifacts

The existing **2400×1600 → 7200×4800** 7B INT8 image workflow completed in
**49.62 s first / 48.76 s warm**, with **32.76 GiB** maximum sampled Metal
allocation. Encoded latents, sampled latents and final RGBA frame bytes match
the previous accepted implementation exactly. The prior warm mean was
49.01 s, so this check shows no meaningful timing regression; the 0.25 s
difference is not claimed as a new optimization. The image schedule is unchanged.

The installed fix is automatic on restart of the ComfyUI 2.14 installation.
The original workflow was not edited. Local verified warm exports were also
copied into the shared output/video folder as
`SeedVR2_7B_memoryfix_5s_20260928.mp4` and
`SeedVR2_7B_memoryfix_10s_20260928.mp4`.

The [compact evidence archive](artifacts/seedvr2-video-memory/README.md)
contains the original workflow, reproducible guarded harness, timing and
memory records, exactness checks, installation hashes, and the 14 passing
scheduling-scope tests. Large tensors and exported media remain local.
