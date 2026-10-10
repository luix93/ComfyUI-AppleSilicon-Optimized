# Credits and implementation provenance

Full project and author credits moved from the README. Keep
[third-party notices](../THIRD_PARTY_NOTICES.md) and the referenced license
files with any distribution.

This pack includes adapted third-party source as well as independently written
integrations. Replacing an upstream node wrapper with an internal runtime does
not remove the upstream authors' credit or license. The [component/source map,
full notices and audited revisions](../THIRD_PARTY_NOTICES.md) is the authoritative
inventory; keep it and the referenced license files when redistributing.

- [ComfyUI-AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8)
  and [`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn)
  (Paweł Mazurkiewicz, MIT) — the original compatibility layer this project
  grew from, and the flash-attention-on-Metal-4-TensorOps kernel design.
  mtlflashattn backs the compatibility-tier SDPA reroute (patch #5) directly;
  our `flash41` extension (patch #19's fast path) is built *around* his kernel
  skeleton — we make it compile on macOS 27, add the wide-tile/mergeP/static-
  reduction/seq-adaptive performance work (~2× over the straight port), and
  extend it to fp16, video sequence lengths, and native GQA. The QK→softmax→PV
  cooperative-tensor structure is his; the compile path, host dispatch, and
  tuning are ours. His MIT copyright is preserved in `flash41.mm`.
- [Cider](https://github.com/Mininglamp-AI/cider) (Mininglamp, MIT) — the
  register-tiled cooperative-TensorOps int8 `matmul2d` that patch #17's GEMM
  is ported from, and the design reference for the staged/swizzled fragment
  work.
- [ccv / NNC](https://github.com/liuliu/ccv) (Liu Liu, BSD-3-Clause) — the
  `NAAttentionKernel` structure DrawThings runs is the recipe our `na`
  attention kernel follows (independent simdgroups, direct-device K/V through
  static-extent slices, row-reduction softmax, relaxed precision). The Metal
  4.1 port, the register-chained P variant, the ragged-tail handling, and the
  torch host integration are ours.
- [h3.c](https://github.com/antirez/h3.c) (Salvatore Sanfilippo) — the released
  MiniMax H3 horizontal token-reduction schedule adapted by the structural
  workflow node. The ComfyUI implementation retains its own full-resolution
  bypass and composes cache and sparse-attention metadata around that schedule.
- [Sol-Attn](https://arxiv.org/abs/2607.24027) and NVIDIA's released H3 routing
  policy — the block-centroid sparse-attention method behind `diag64`. The
  Metal router, corrected sparse kernel integration, ragged handling, and
  ComfyUI ownership/fallback plumbing in this repository are local work.

- **[MotionCache](https://github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache)**
  (starsFriday, MIT), **[Spectrum H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)**
  (xmarre, GPL-3.0-or-later), and **[Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI)**
  (veda-sparse, MIT) — the copied/adapted cache, forecast and predictor runtimes
  in `_patches/h3_runtime`, with pinned originals in `_patches/_vendor`.
  The local APIs and Apple backends retain their algorithms and attribution.
- **[PyTorch](https://github.com/pytorch/pytorch)** (PyTorch contributors, BSD-style
  notices) and **[MLX](https://github.com/ml-explore/mlx)** (Apple / MLX contributors,
  MIT) — exact MPP attention, reduction, normalization and rounding references,
  including derived kernel code. Full notices are bundled with the native sources.
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** (ComfyUI contributors, GPL-3.0)
  and **[Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen)** (Comfy Org,
  Apache-2.0) — host model/decoder contracts and kernel references; Comfy Kitchen's
  neighborhood-attention tile grouping is adapted in the LTX VAE path.
  **[Kijai](https://github.com/kijai)** is credited for ConvRot/INT8 VAE integration
  and sparse-attention work used as implementation/research references.
- **[SeedVR](https://github.com/ByteDance-Seed/SeedVR)** (ByteDance-Seed contributors,
  Apache-2.0 for the bundled fixed conditioning assets) — the positive/negative
  conditioning tensors converted to our safetensors container. Main model weights
  are not bundled and retain their own terms.
- **[MiniMax H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)**
  (LBH-123-AI, MIT) — copied/adapted 2D/3D network architecture and constants;
  our MPS device/precision controls and bounded execution wrap that architecture.
- **[MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS)** (MLX-DLSS contributors,
  Apache-2.0) and **[MLX Swift](https://github.com/ml-explore/mlx-swift)** (MIT) —
  the optional external native runtime used by Apple Silicon Super Resolution.
  Our source patch adds VSR-before-rendering ordering and memory limits.
  **[DLSSMac](https://github.com/Mappsnet7/DLSSMac)** (Mappsnet7 / Artem Pankratov)
  supplied integration/research references and the separately fetched v0.1.6
  neural-rendering package. NVIDIA owns those converted weights; the pack's
  source license does not authorize redistributing them.

Research references also include [SolAttn MPS](https://github.com/yshenaw/ComfyUI-SolAttn-MPS),
[Kijai's Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton),
[KingGore's Blackwell Sol-Attn](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell),
[qlip-mlx](https://github.com/TheStageAI/qlip-mlx), and
[MiniMax H3 SPEED](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED).
They are not required external nodes for our bundled acceleration runtimes;
SPEED's noncommercial implementation is not bundled. FirstBlockCache's residual
reuse method is also credited to [ParaAttention](https://github.com/chengzeyi/ParaAttention).
These references are distinguished from copied source in the notices inventory.
