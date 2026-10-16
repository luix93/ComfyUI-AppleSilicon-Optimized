# Third-party notices

## Distribution scope and inventory

The complete bundle is distributed under **GPL-3.0**, selecting version 3 of
Spectrum's GPL-3.0-or-later grant and retaining ComfyUI's GPL-3.0 terms. Root `LICENSE` is the full GPL text; `LICENSE-BUNDLED`
is retained as a compatibility copy. Original permissive notices below remain
in force for their respective portions. Neither moving code behind an internal
API nor making a runtime optional removes these notices.

| Upstream / authors | Use in this repository | Notice / terms |
|---|---|---|
| [ComfyUI-AppleSilicon-FP8 / Paweł Mazurkiewicz](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8) | Inherited compatibility layer and original repository code | [Original MIT notice](third_party/original-FP8-MIT.txt) |
| [mtlflashattn / Paweł Mazurkiewicz](https://github.com/pawel-mazurkiewicz/mtlflashattn) | Installed dependency; adapted V2 kernel in `_patches/mfa4_ext/flash41.mm`, reused by `sol_ext/sol_attn.mm` | MIT declared by upstream package metadata; Paweł's [full MIT notice](third_party/original-FP8-MIT.txt) and source copyright retained |
| [Cider / Mininglamp contributors](https://github.com/Mininglamp-AI/cider) | Adapted INT8 GEMM helpers, compute and dequantization epilogues in `int8_ext/int8_gemm.mm`; diagnostic derivatives in `dev/h3_fusion_experiments` | [MIT](third_party/cider-MIT.txt) |
| [CCV / Liu Liu and contributors](https://github.com/liuliu/ccv) | NA/dense/SOL attention in `mfa4_ext`, H3 shader directories; generated research shaders under `benchmarks/artifacts/krea2-oct2026/experiments` | [BSD-3-Clause](third_party/ccv-BSD-3-Clause.txt); source notice applies to code, CCV documentation has separate CC-BY terms |
| [PyTorch / contributors](https://github.com/pytorch/pytorch) | MPP headers expanded at runtime; adapted reduction, normalization and rounding kernels, including Ideogram/Qwen/LTX/Wan/H3/SeedVR paths | [PyTorch full notices](_patches/krea2_ext/PYTORCH_LICENSE) and [SeedVR copy](_patches/seedvr2_ext/PYTORCH_LICENSE) |
| [MLX / Apple and contributors](https://github.com/ml-explore/mlx) | PyTorch MPP/normalization ancestry; installed MLX integrations | [MIT](_patches/krea2_ext/MLX_LICENSE) |
| [ComfyUI / contributors](https://github.com/Comfy-Org/ComfyUI) | Host model/decoder integrations and adapted forwards | GPL-3.0; [full text](LICENSE), [upstream license](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) |
| [Comfy Kitchen / Comfy Org](https://github.com/Comfy-Org/comfy-kitchen) | Adapted LTX neighborhood-attention grouping; quantization, ConvRot, INT8 VAE and Flux.3 RMS/RoPE compatibility references | [Apache-2.0](third_party/Comfy-Kitchen-Apache-2.0.txt); copyright (c) 2025 Comfy Org on the adapted LTX file |
| [SeedVR / ByteDance-Seed](https://github.com/ByteDance-Seed/SeedVR) | Bundled fixed positive/negative conditioning tensors | [Apache-2.0](_patches/seedvr2_assets/LICENSE), [asset provenance](_patches/seedvr2_assets/README.md) |
| [MotionCache / starsFriday](https://github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache) | Bundled reference and maintained derivative | [MIT](_patches/h3_runtime/motion/LICENSE) |
| [Spectrum H3 / xmarre](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) | Bundled reference and maintained derivative | [GPL-3.0-or-later](_patches/h3_runtime/spectrum/LICENSE), [copyright](_patches/h3_runtime/spectrum/COPYRIGHT) |
| [Veda / veda-sparse](https://github.com/veda-sparse/Veda-on-ComfyUI) | Bundled predictor/scoring/selection source and maintained derivative | [MIT](_patches/h3_runtime/veda/LICENSE); predictor weights separately use MiniMax Community terms |
| [H3 latent upscaler / LBH-123-AI](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | Adapted `h3_upscale_network_2d.py` / `3d.py` and model constants | [MIT](third_party/h3-latent-upscaler-MIT.txt) |
| [h3.c / Salvatore Sanfilippo](https://github.com/antirez/h3.c) | Horizontal token-reduction policy adapted into our Python coordinator; C implementation is not bundled | [MIT](third_party/h3.c-MIT.txt) |
| [Sol-Attn / NVIDIA and paper authors](https://arxiv.org/abs/2607.24027), [Sana reference](https://github.com/NVlabs/Sana/tree/sol-engine) | Adapted H3 `diag64` routing policy and centroid correction; CUDA/Triton/CuTe/FlashAttention kernels are not bundled | [Sana Apache-2.0](third_party/NVIDIA-Sana-Apache-2.0.txt), [branch licensing declaration](https://github.com/NVlabs/Sana/tree/sol-engine#-license) |
| [MLX-DLSS / contributors](https://github.com/iamwavecut/MLX-DLSS) | Optional separately built native runtime; `dev/patches/vsr-neural-rendering.patch` contains modified upstream source/context | [Apache-2.0](third_party/MLX-DLSS-Apache-2.0.txt), [upstream NOTICE](third_party/MLX-DLSS-NOTICE.txt) |
| [MLX Swift / ml-explore](https://github.com/ml-explore/mlx-swift) | Dependency of separately built MLX-DLSS runtime | MIT; runtime setup preserves its and resolved dependencies' license/notice files |
| [DLSSMac / Mappsnet7](https://github.com/Mappsnet7/DLSSMac) | Research/integration reference; optional downloaded v0.1.6 model package | Source is Apache-2.0; no DLSSMac app/source/binary is bundled. Converted NVIDIA weights retain vendor terms |

Exact original notices are retained rather than replaced by this summary.
A [release audit](docs/release-attribution.md) describes the corrections,
validation and source-release boundaries.
A [machine-readable provenance index](third_party/provenance.json) records audit
sources, revisions and notice hashes. Auditing an upstream license revision does
not imply that every derivative was copied from that revision; pinned source
revisions are identified separately below or in the adjacent source headers.

## Additional adapted source and research provenance

The optional H3 INT8 QK / BF16 PV shader in `h3_floatpv_shaders` is generated
from CCV revision `e499321b29b5c9c31329a8efc6960d84856d3315`
(`NAInt8AttentionKernel`, `quantizePV=false`). The upstream BSD-3-Clause
notice is retained in the shader. Native BHSD addressing, PyTorch buffer
bindings, memory guards and H3 preset integration are local adaptations.


The H3 latent-upscaler networks derive from revision
`40316cf008b2fd8663263270669eb4da23f89d2c` of LBH-123-AI's project.
Training architecture, normalization constants and ordering are retained;
Apple device/precision selection, memory bounds and integration are local changes.

The optional MLX-DLSS build pins revision
`0ca2deab092fe6f3e331bf4f616271dbc64521d0`. Our patch changes image/video stage
ordering and adds a VSR-before-neural-rendering option; the setup script adds
allocator/cache limits and a watchdog. The patch is an Apache-covered derivative,
and carries the upstream notice through this bundle. Runtime builds retain
upstream LICENSE/NOTICE and resolved Swift/MLX dependency notices. The generated
`_runtime` directory is ignored by Git and is not part of the source release.

Kijai's [ConvRot VAE integration](https://github.com/Comfy-Org/ComfyUI/pull/16187)
and [Comfy Kitchen work](https://github.com/Comfy-Org/comfy-kitchen/pull/167)
were implementation references. CUDA kernels from those changes are not bundled.
The exact MPP/reduction/normalization ancestry notices above also cover the
Ideogram output-channel split and subsequent exact fusion/layout derivatives;
our changes retain upstream arithmetic and add platform-specific schedules.

Other research/integration references, not copied runtime dependencies:

- [yshenaw/ComfyUI-SolAttn-MPS](https://github.com/yshenaw/ComfyUI-SolAttn-MPS): Apple sparse-attention and H3 packed-prefix integration reference.
- [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) and [KingGore's Blackwell node](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell): sparse routing/prefix protection and comparison references; their CUDA/Triton/flex implementations are not bundled.
- [TheStageAI/qlip-mlx](https://github.com/TheStageAI/qlip-mlx): quantized/Sage-style attention research, screened in external diagnostics. The released Krea experimental shaders are CCV-derived, not qlip source.
- [StanLukuvka/ComfyUI-MiniMax-H3-SPEED](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED): external benchmarking/calibration only. Its PolyForm Noncommercial implementation is not bundled or relicensed by this pack.
- [ParaAttention](https://github.com/chengzeyi/ParaAttention): FirstBlockCache residual-reuse method reference; the H3 state/dispatcher implementation is local.
- [Draw Things community](https://github.com/drawthingsai/draw-things-community): H3/Krea graph, quantization, scheduling and performance references; CCV-derived kernel code is covered separately above.

The old name `ComfyUI-FlashVSR-AppleSilicon` in `sol_ext/sol_attn.mm` referred
to this maintainer's local prototype, not a third-party repository. The maintainer
confirmed this during the October 10 audit; the misleading source header has
been corrected. The underlying mtlflashattn provenance remains credited.

Model checkpoints, LoRAs, Veda predictors and NVIDIA-derived VSR/neural-rendering
weights are separate artifacts. This pack's source license grants no rights to
those weights. `dev/setup_neural_rendering.py` records the separately obtained
DLSSMac v0.1.6 package's origin and checksum; availability of that ZIP is not a
redistribution permission. Do not include externally obtained model weights or
optional runtime binaries in a source release merely because they are present
locally. The sole bundled conditioning asset and its declared Apache provenance
are listed explicitly above. Credits do not imply upstream endorsement.

## Kernel and conditioning notices


`_patches/torch_mpp_qcache.py` expands the locally installed PyTorch 2.14
`MppAttention.h` / `PrefillAttention.h` and their Metal utility headers, checks
their combined source hash, and moves query loads outside the key loop.
The validated PyTorch commits are `08187d9e0fba026dc8217405802ab5381dc88d90`
(2.14.0) and `5c4886908584029761b579af026dcfb627c84070` (2.14.1).
These attention kernels are adapted from MLX. PyTorch and MLX license notices
are reproduced in [_patches/krea2_ext/PYTORCH_LICENSE](_patches/krea2_ext/PYTORCH_LICENSE)
and [_patches/krea2_ext/MLX_LICENSE](_patches/krea2_ext/MLX_LICENSE).

The GroupNorm reduction/affine ordering in
`_patches/seedvr2_ext/seedvr2_kernels.mm` is adapted from PyTorch's
`aten/src/ATen/native/mps/kernels/GroupNorm.metal` and
`c10/metal/reduction_utils.h` at commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`. The split-statistics FP16
GroupNorm/SiLU/folding path follows the same reduction and rounding order
reviewed at commit `5c4886908584029761b579af026dcfb627c84070` (PyTorch 2.14.1).

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
under GPL-3.0 (see `LICENSE`). Spectrum itself retains GPL-3.0-or-later terms. The project's original code retains
its existing MIT notices in `third_party/original-FP8-MIT.txt`; those notices do not relicense Spectrum.
The complete corresponding source is supplied in this repository. Preserve these
notices, licenses and corresponding source when redistributing this bundle.
