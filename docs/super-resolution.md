# Apple Silicon Super Resolution

Optional **RTX VSR 1.8.2, High Bitrate Low / mode 16, 2×** upscaling through
[MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS). This is a separate model and
runtime from SeedVR2. An optional **DLSS neural-rendering stage runs after VSR**
to enhance detail and appearance at the enlarged resolution. Both stages use
Swift/MLX/Metal; PyTorch supplies the ComfyUI tensor interface.

| Node | Input → output | Use |
|---|---|---|
| Apple Silicon Super Resolution · Image 2x | IMAGE → IMAGE | A still or a small frame batch |
| Apple Silicon Super Resolution · Video 2x | VIDEO → VIDEO | Clips streamed to a file, with audio and timing |

Both nodes are under **Apple Silicon / Upscaling**. Final output is always 2× on
each axis. VSR operates on RGB8 SDR independently per frame. Optional neural
rendering can use temporal history in the Video node. HDR is rejected; neither
stage adds frames or offers native 3× output. Image batches still require memory
for the complete returned float32 tensor; use the Video node for longer sequences.

## Optional neural rendering: VSR first, detail second

The stages run in one native worker, without an encoded intermediate video:

```text
Input W × H → VSR 2W × 2H → neural rendering → final 2W × 2H
```

`nr_processing_scale=2` makes NR work internally at **4W × 4H**, then return to
**2W × 2H**. It does not enlarge the final output again. Scale 1 works directly
at VSR's output size. Scale 2 uses four times as many neural-processing pixels.

Install the optional model from the checksum-pinned DLSSMac v0.1.6 release:

```sh
python dev/setup_neural_rendering.py \
  --models-dir /path/to/your/shared/models/applesilicon_neural_rendering
```

The explicit setup command downloads the 305 MiB Mac release, verifies its SHA-256,
and extracts only `NeuralRendering.dlssmodel/manifest.json` and `weights.safetensors`.
It also verifies the model checksum. Use `--archive /path/to/DLSSMac-0.1.6-macos.zip`
to reuse a download. No rented GPU, NVIDIA DLL, or Python-package installation is
needed. Rebuild the native runtime with the updated setup script and restart
ComfyUI after upgrading. Model weights retain their original ownership and are
not included in this repository.

| Control | Meaning |
|---|---|
| `neural_model` | `disabled` keeps VSR alone; select `NeuralRendering.dlssmodel` to enable the second stage |
| `nr_profile` | Natural, standard, or cinematic appearance; these are styles, not different model checkpoints |
| `nr_intensity` | Strength of the neural effect; zero bypasses NR completely, including loading its model |
| `nr_detail` | Strength of the high-frequency part of the model's changes; 1 leaves it unchanged |
| `nr_colour` | Strength of the low-frequency colour/tone changes; 1 leaves it unchanged |
| `nr_processing_scale` | Internal scale 1 or 2; final dimensions remain 2× input |
| `nr_temporal` | Video only: estimated optical flow, previous-frame history and automatic cut resets |

Existing workflows keep NR disabled by default. Example workflows enable it with
natural style, intensity/detail/colour 1, internal scale 2, and a 32 GiB budget.
Lower `nr_colour` if you prefer less tonal change, or use scale 1 for less memory
and faster execution. Model-specific changes are approximate, can alter image
content, and are not guaranteed to improve every source. Zero strength and
disabled mode preserve the original VSR image tensor byte-for-byte.

This integration uses DLSSMac's converted neural-rendering weights with the
existing MLX-DLSS FP16 runtime. It does not enable DLSSMac's separate experimental
FP8/NAX kernel profile. Video uses MLX-DLSS's native temporal pipeline, including
VideoToolbox/Vision optical flow, rather than DLSSMac studio's independent-frame
preview path. This is separate from temporal DLSS Super Resolution (`.srmodel`).

Ready-to-load workflows: [image](../workflows/apple_silicon_vsr_neural_image.json)
and [video](../workflows/apple_silicon_vsr_neural_video.json). Matching `.api.json`
files are provided for API execution. Select your own source image/video when
using these outside the original shared install.

### Two-stage validation, 2026-09-28

M5 Ultra 80 GPU, 256 GB RAM, macOS 27; natural profile, intensity/detail/colour 1.
Times include complete Comfy node calls and worker/model startup. Warm averages
use runs two and three. [Machine-readable results](../benchmarks/data/m5-ultra-vsr-neural-rendering.json).

| Input → output | Mode | Warm time | Peak active MLX memory |
|---|---|---:|---:|
| 1024² → 2048² image | VSR only | 0.71 s | 1.07 GiB |
| Same image | VSR → NR, internal scale 1 | 1.40 s | 2.11 GiB |
| Same image | VSR → NR, internal scale 2 | 3.07 s | 7.33 GiB |
| 416×736 → 832×1472, 5 s / 120 frames / 24 fps | VSR → temporal NR, internal scale 2, HEVC + audio | 44.11 s | 3.09 GiB |

The five-second video runs were 43.97 / 44.13 / 44.09 seconds. Active MLX memory
was also 3.09 GiB on a one-second sequence; no full video tensor was materialized.
Allocator cache is additional, limited to 512 MiB. PNG encoding after a Save Image
node is outside the image node timings above.

Checks cover VSR-only byte equality with the saved pre-change reference, exact
zero-intensity bypass without NR weights, identical repeated enhanced image
tensors, fixed output dimensions at both internal scales, temporal on/off video,
frame count, timestamps, audio and one reset at a deliberate scene cut. Both
installed model menus and the image workflow were also checked through a running
ComfyUI server. Twenty-two unit tests cover memory, controls, bypass and worker
failure/cancellation. Visual comparisons show appearance changes; these checks do
not establish universal quality improvement or NVIDIA output parity.

A standalone native check confirms that the combined image pipeline matches
explicit VSR followed by NR in memory byte-for-byte. Saving and reloading the
intermediate PNG introduces a small half-float/colour-conversion difference before
NR, so separate file-based passes are not a byte-exact reference for this direct
pipeline. The original command-line-only Swift installation lacks XCTest; the
native composition check was compiled and run as a standalone Swift executable.

## Setup

Requires Apple Silicon, macOS 26+, Swift 6.3+ command-line tools, Git, and Python
3.10+ with pip. Normal node execution makes no network requests.

Build the optional native runtime from the custom-node directory:

```sh
python dev/setup_super_resolution.py
```

This fetches MLX-DLSS revision `0ca2deab092fe6f3e331bf4f616271dbc64521d0`, builds its
CLI using Swift's native build system, and obtains the matching precompiled
`mlx-metal==0.31.1` shader library. The Python wheel is placed under `_runtime`,
without installing packages into ComfyUI. Swift's native build system is deprecated
in the current toolchain but was tested successfully with Swift 6.4. All build
artifacts stay under the ignored `_runtime` directory.

Advanced setup can reuse a checkout and matching shader library:

```sh
python dev/setup_super_resolution.py \
  --source /path/to/MLX-DLSS \
  --metallib /path/to/mlx-0.31.1/mlx.metallib
```

Local CLI changes add allocator/cache limits, a memory watchdog, peak-memory
reporting, and the optional `--vsr-first on` order. The upstream VSR and NR model
operations are unchanged. The order patch is tracked in
`dev/patches/vsr-neural-rendering.patch` and applied idempotently by setup.

## Model

Prepare your own weights using the
[upstream extraction instructions](https://github.com/iamwavecut/MLX-DLSS/blob/main/docs/super-resolution.md).
The supported Linux library is `libnvidia-ngx-vsr.so.1.8.2`, SHA-256:

```text
c7f2387a565e41b77a624634c102c2254d5285d28f886bd5db7491df8b1b037e
```

It is present inside the Linux CPython 3.10 wheel for `nvidia-vfx==0.1.0.1` on
[NVIDIA's package index](https://pypi.nvidia.com/nvidia-vfx/), at
`nvvfx/libs/libnvidia-ngx-vsr.so.1.8.2`. The library's bytes are extracted and read;
the Linux code need not run on the Mac, and no rented GPU is required for VSR.

With NumPy and safetensors installed, run this from the MLX-DLSS checkout:

```sh
python python/mlxdlss/tools/extract_vsr_weights.py \
  /path/to/libnvidia-ngx-vsr.so.1.8.2 \
  /path/to/ComfyUI/models/applesilicon_super_resolution/vsr.safetensors
```

The model contains 66 tensors / 442,144 parameters, approximately 0.85 MiB. The node
validates its model identifier and upstream validates tensor names/shapes. Restart
ComfyUI after setup, then select `vsr.safetensors`. Extra model path configurations
can register the `applesilicon_super_resolution` category. With Comfy Desktop,
place the model in `applesilicon_super_resolution` inside your configured shared
models directory. Shared paths retain their configured priority; the install's
local model folder is also searched. Refresh ComfyUI after adding a model to
update the dropdown. Vendor libraries and
generated weights are separate from the public custom-node repository.

## Workflows

Image: **Load Image → Apple Silicon Super Resolution · Image 2x → Save Image**.

Video: **Load Video → Apple Silicon Super Resolution · Video 2x → Save Video**.
Use Save Video's automatic codec/container options to preserve the already encoded
stream where supported. The VSR node offers HEVC, H.264, or ProRes; ProRes produces
a MOV file and requires more disk space.

The Video input is exported through ComfyUI's streaming video API to honor trims
and crops. Compatible unedited MP4 input is remuxed. Edited or incompatible input
may be transcoded by ComfyUI before upscaling. The native pipeline then processes
frames sequentially and returns a file-backed VIDEO; it never builds an upscaled
IMAGE batch. Output encoding is separate from the neural inference and can change
pixels. The nodes do not claim NVIDIA byte parity or SeedVR2-equivalent detail.

## Memory and cancellation

The default budget is **8 GiB**, adjustable from 1–32 GiB. A conservative preflight
includes padded input dimensions and the complete returned IMAGE batch. Insufficient
free system memory or an oversized job is rejected before execution. Input dimensions
above 8192 pixels on either side are currently rejected; this wrapper does not tile.
With NR enabled, the estimate includes both the VSR enlargement and NR's internal
scale, plus model/history/workspace allowance. Oversized combinations are rejected
before the worker starts; decrease the internal scale or image size. The estimate
is conservative and cannot replace the worker's enforced allocation budget.

The worker applies an MLX allocation limit and a cache limit of at most 512 MiB.
A watchdog checks active plus cached MLX memory every 100 ms; the host also checks
worker RSS and ComfyUI cancellation. These are sampled safeguards, not a promise
that no transient allocation can exceed the budget. Per-job temporary directories
are cleaned on success, cancellation, or failure, including partial native exports.
The returned video remains in ComfyUI's temporary directory for downstream nodes.

Cancellation is checked around input preparation and during native execution.
The current ComfyUI video-export call controls cancellation responsiveness during
input preparation. Image batches launch one native job per frame; video uses one
worker for the entire clip. A persistent tensor interface is a possible future
image-batch optimization.

## Validation, 2026-09-28

M5 Ultra, 80 GPU cores, 256 GB RAM, macOS 27:

Machine-readable measurements: [m5-ultra-vsr.json](../benchmarks/data/m5-ultra-vsr.json).

- 384×256 image: decoded RGB output matches the upstream CLI reference byte for
  byte; three repeated runs match each other.
- 2400×1600 → 4800×3200: **2.23 / 2.22 seconds** for complete node calls, including
  tensor/image conversion, worker startup, inference, PNG I/O, and returned tensor.
  Peak active MLX memory: **2.40 GiB**. Repeated output tensors are byte-identical.
- One-second video: HEVC, H.264 and ProRes preserve 24 frames, 24 fps, audio and
  increasing timestamps. A trimmed/cropped case returns 12 frames at 512×384 and
  preserves its half-second duration and audio.
- The same H3 input used in the SeedVR2 memory investigation, 416×736 → 832×1472,
  24 fps with audio, completes five seconds / 120 frames in **1.88 s** and ten
  seconds / 240 frames in **3.11 s**. These are means of runs two and three, including
  input preparation, inference and HEVC export. The respective warm runs are
  1.90 / 1.85 s and 3.45 / 2.77 s. Peak active MLX memory stays at **0.61 GiB** for
  both durations; up to 0.50 GiB of allocator cache is additional. File-backed input
  was tested with `get_components()` disabled to catch accidental materialization.
- Twelve unit tests cover memory preflight, large IMAGE batches, failures, worker
  termination and cancellation. A real 512 MiB GPU-budget test stops cleanly with
  the expected memory-limit error.

These are integration checks and timings, not a complete visual-quality study.
They do not compare a PyTorch port or establish equality with NVIDIA's proprietary
runtime. The stock/optimized diffusion benchmark table remains a separate comparison.

## Temporal DLSS SR and PyTorch

Temporal DLSS SR is not exposed by these nodes. Upstream's supported model needs a
first-frame CUDA capture from DLSS SDK 310.7.0, preset K, LDR, 2×. Its public
packager consumes captured network weights and reconstruction coefficients; the
capture-producing harness is absent from the inspected repository. Obtain or build
that harness before renting a GPU. A prepared `.srmodel` is a directory containing
weights, a manifest, and a reconstruction shader.

A PyTorch/MPS VSR implementation is feasible but is not included here. It would
need validation of padding, channel order, interpolation and rounding against this
native reference, followed by a complete-node speed comparison.
