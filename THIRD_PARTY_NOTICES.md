# Third-party notices

`_patches/torch_mpp_qcache.py` expands the locally installed PyTorch 2.14
`MppAttention.h` / `PrefillAttention.h` and their Metal utility headers, checks
their combined source hash, and moves query loads outside the key loop.
The validated PyTorch commit is `08187d9e0fba026dc8217405802ab5381dc88d90`.
These attention kernels are adapted from MLX. PyTorch and MLX license notices
are reproduced in [_patches/krea2_ext/PYTORCH_LICENSE](_patches/krea2_ext/PYTORCH_LICENSE)
and [_patches/krea2_ext/MLX_LICENSE](_patches/krea2_ext/MLX_LICENSE).

The GroupNorm reduction/affine ordering in
`_patches/seedvr2_ext/seedvr2_kernels.mm` is adapted from PyTorch's
`aten/src/ATen/native/mps/kernels/GroupNorm.metal` and
`c10/metal/reduction_utils.h` at commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`.

PyTorch's copyright notices and license terms are reproduced in
[_patches/seedvr2_ext/PYTORCH_LICENSE](_patches/seedvr2_ext/PYTORCH_LICENSE).

The fixed SeedVR2 conditioning tensors in
`_patches/seedvr2_assets/conditioning.safetensors` come from
[ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185)
(`pos_emb.pt` and `neg_emb.pt`). Only the container and tensor names changed;
the BF16 tensor bytes are preserved. Their Apache 2.0 license is reproduced
in [_patches/seedvr2_assets/LICENSE](_patches/seedvr2_assets/LICENSE).

The RMSNorm reduction ordering in `_patches/krea2_ext/krea2_kernels.mm`
follows PyTorch's `aten/src/ATen/native/mps/kernels/RMSNorm.metal`
at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`, which is adapted
from MLX's `mlx/backend/metal/kernels/rms_norm.metal`
(Copyright © 2024 Apple Inc.). The original license notices are included in
[_patches/krea2_ext/PYTORCH_LICENSE](_patches/krea2_ext/PYTORCH_LICENSE)
and [_patches/krea2_ext/MLX_LICENSE](_patches/krea2_ext/MLX_LICENSE).

The Qwen-Image 2.1 RMSNorm reduction in `_patches/qwen21_fast.metal`
follows the same PyTorch/MLX reduction convention as the Krea2 kernels above.
The PyTorch and MLX license notices are included in `_patches/krea2_ext`.

The tile grouping and attention loop in `_patches/ltx25_vae.py` are adapted
from Comfy Kitchen 0.2.35 `backends/eager/na.py`, Copyright (c) 2025 Comfy Org.
All rights reserved. Changes batch independent MPS tiles under a bounded
workspace, fuse neighborhood copies with bounded integer row maps, and build
identical masks without GPU readback. Apache 2.0 license
terms are reproduced in [_patches/seedvr2_assets/LICENSE](_patches/seedvr2_assets/LICENSE).
The LTX RMSNorm reduction in `_patches/ltx25_fast.metal` follows the same
PyTorch/MLX convention and license notices referenced above.

## Pack-owned H3 accelerator derivatives

The M5 INT8 SOL and dense attention shaders under `_patches/h3_sol_shaders`,
`h3_sol_head_major_shaders`, `h3_sol_legacy_shaders` and `h3_dense_shaders` derive from CCV's
`NAInt8SolAttentionKernel.cpp`, revision
`6a611be1aab6470ae279115ae4eaf7e01bc87135`, by Liu Liu and contributors.
Changes include BF16 external carriers, head-major INT8 scratch/output,
an optional all-token dense loop, and protected PyTorch prefix queries.
The BSD-3-Clause notice is preserved in each source and reproduced in
[third_party/ccv-BSD-3-Clause.txt](third_party/ccv-BSD-3-Clause.txt).
Comfy Kitchen's full INT8 attention, chunked producer and token augmentation
were research references; its CUDA implementation is not bundled by this pass.

MotionCache, Spectrum and the Veda predictor runtime are locally maintained
under `_patches/h3_runtime`. The combined node calls direct internal APIs;
these modules neither register upstream nodes nor import the vendor references.
They load only when selected and perform no runtime downloads.

The pinned, unmodified `_patches/_vendor` files remain comparison references.
Their `manifest.json` records original commits and SHA-256 hashes.
`_patches/h3_runtime/ORIGINS.json` identifies the input sources used for each
copied/adapted file; these source hashes do not describe current fork contents.
Future upstream changes must be reviewed, ported and parity-tested explicitly.

- **MiniMax-H3-MotionCache**, copyright (c) 2026 starsFriday, commit
  `bc2894102b2486661884371259a27080b0b137bf`:
  https://github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache — MIT.
  The state/reuse arithmetic is retained. Our direct model API replaces the
  upstream V3 node; keyword inputs and preparation-error cleanup are supported. Detector scalar
  readbacks are batched without changing reductions or Python scalar arithmetic.
  The upstream Fast VAE node is excluded. Full license: `h3_runtime/motion/LICENSE`.
- **ComfyUI-Spectrum-MiniMax-H3**, copyright (C) 2026 xmarre, v0.2.28,
  commit `5161f0457bc8c52535212d6783eee73f439e1537`:
  https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3 — GPL-3.0-or-later.
  Forecasting, solver/replay arithmetic and compatibility hook order are retained.
  Our direct preset API replaces public node registration. Unused objective-media
  nodes/tools are excluded; the isolated worker has a pack-specific package name.
  Full notices: `h3_runtime/spectrum/COPYRIGHT` and `h3_runtime/spectrum/LICENSE`.
- **Veda-on-ComfyUI**, Veda contributors, v0.2.0, commit
  `fd59c7277ccc37ebf1a8f6823474b8c2ef2e33a0`:
  https://github.com/veda-sparse/Veda-on-ComfyUI — MIT.
  Checkpoint dequantization, trained plans, TripPool scoring and tile selection
  retain upstream arithmetic. Our explicit Apple runtime replaces backend
  discovery and the public node API; bounded Metal/MLX execution lives in
  `h3_runtime/veda/runtime.py`. Projection-memory accounting is cached and cleared
  per run. Native Metal kernels remain in `h3_veda_mps.py`. Full license:
  `h3_runtime/veda/LICENSE`. Predictor weights remain a separate model artifact
  under the MiniMax H3 Community License; they are not bundled or retrained.

This combined distribution includes GPL-covered Spectrum code and is distributed
under GPL-3.0-or-later (see `LICENSE-BUNDLED`). The project's original code retains
its existing MIT notices in `LICENSE`; those notices do not relicense Spectrum.
The complete corresponding source is supplied in this repository. Preserve these
notices, licenses and corresponding source when redistributing this bundle.
