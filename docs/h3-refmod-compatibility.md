# MiniMaxH3Mod / RefMod on Apple Silicon

Checked September 29, 2026 against upstream commit
[`6131c1f`](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/6131c1f62c612a074d6aac8435c26ffa19c00563).
Environment: M5 Ultra 80 GPU, macOS 27, ComfyUI 2.14 install, PyTorch 2.14.0.
External source was loaded in isolated diagnostic processes, not installed into
the user's live custom_nodes directory. No production code was changed.

## Upstream fixes submitted

[PR #29: Fix H3 reference frame sampling and MPS preprocessing](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/pull/29)
implements the corrections identified below in a separate upstream checkout.
The original findings and measurements below describe unpatched `6131c1f`.

- Shared node/CLI video sampling now uses 17k+5, preserving valid 22/39-frame
  clips and both endpoints when uniformly reducing a clip. Limits below five
  frames retain RefMod's first-image fallback.
- MPS-resident pooling and visual/audio strength blurring explicitly execute
  on CPU and return to the original device/dtype, without global MPS fallback.
- The refinement timing harness now synchronizes MPS around measurements.

Validation: 18 regression tests passed; real H3 FP16 VAE encodes of 22 and 39
320×320 frames matched direct encoding of the same preprocessed pixels byte
for byte. MPS VAE output placement, compressed creation with three refinement
steps, weakened Apply and byte-preserving save/load also passed. The guarded
codec process peaked at 10.58 GiB RSS (not total GPU/unified-memory use).
All MPS checks kept global fallback disabled. The PR is a compatibility fix,
not a new perceptual-quality or complete-sampling speed qualification.

The separate checkout is `research/ComfyUI-MiniMaxH3Mod` in the parent workspace;
reproducer/results are `diagnostics/h3-refmod/fix_validation.py` and
`fix-codec-results.json`. This does not install RefMod into the live ComfyUI.

## Finding

The normal RefMod workflow runs on Apple Silicon without a CUDA-to-Metal port.
The pack uses native H3 reference conditioning and ordinary PyTorch operations;
it does not require Triton, FlashAttention, SageAttention or a CUDA extension.
Upstream reports Windows-only validation; the checks below add bounded local MPS
coverage, not identity/motion/voice-quality qualification.

| Path | Local result |
|---|---|
| Package import | All 16 node IDs loaded |
| Full visual creation | H3 FP16 VAE encoded a 320×320 image on MPS; finite 100-token latent |
| Compressed visual creation | 8×8 latent, 16 tokens, three refinement steps; finite |
| Save/load | Latent bytes preserved in both modes |
| Apply + preview | Weakened reference attached successfully; native VAE previews completed |
| Short H3 generation | Full/compressed refs, Step Curve, and Bridge completed with exact native kernels |
| Audio creation | 0.25 s mono at 32 kHz encoded on MPS with FP32 audio VAE; [1,32,2,10], 20 tokens |
| Video creation | Runs, but an incorrect temporal-grid assumption discards valid frames |

All checks used `PYTORCH_ENABLE_MPS_FALLBACK=0`. Normal ComfyUI stores VAE outputs
and loaded RefMods on CPU, where pooling and default refinement execute. The VAE
and H3 diffusion model still use MPS. The Step Curve probe recorded CPU reference
latents on all four evaluations, so it also avoided unsupported GPU pooling.

## Matched generation check

864×480, 22 frames, four sampling steps, res_multistep/simple, seed 812764035,
INT8-convrot H3, FP16 video VAE, no Turbo LoRA. A two-step warmup was discarded.
These are compatibility runs, not four-step quality or acceleration benchmarks.
A synthetic reference was used; perceptual identity preservation was not assessed.

| Reference path | Sampling | End to end |
|---|---:|---:|
| Native reference block | 3.57 s | 5.40 s |
| RefMod Apply, same stored latent | 3.56 s | 5.27 s |
| Compressed RefMod | 3.61 s | 5.46 s |
| RefMod Step Curve | 3.53 s | 5.23 s |
| RefMod Bridge, same stored latent | 3.56 s | 5.39 s |
| Native reference again | 3.55 s | 5.25 s |

Apply and Bridge both produced byte-identical sampler latents, decoded video and
decoded audio to the native reference with the **same stored FP16 latent**. The
native reference repeated after the other modes also matched. This does not
claim that RefMod's entire preprocessing pipeline is byte-identical to a native
reference-image workflow. Compression and strength curves changed outputs as
expected. All outputs were finite. Small timing differences are single-run noise;
there is no demonstrated sampling speed gain for this tiny reference.

Memory watchdog: 110 GiB process RSS limit and 48 GiB minimum available RAM.
Sampled peak process RSS: 11.48 GiB; this is not total unified/GPU memory.

## Work needed

1. **Video frame policy — actual correctness issue, all platforms.**
   [`common.py`](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/6131c1f62c612a074d6aac8435c26ffa19c00563/common.py#L182)
   snaps references to 4k+1 frames. Current native H3 uses 17k+5 (5,22,39,…),
   plus its special still-image path. With 22 input frames and a limit of 22,
   RefMod passed only 21 to the VAE. Both produced seven latent frames, but the
   stored latent differed from directly encoding all 22 frames. The probe changed
   the final input frame deliberately; maximum FP16 latent difference was 5.802734375.
   This demonstrates lost content, not a typical quality-error estimate.
   Replace the stale temporal policy with current H3-aware sampling/padding that
   preserves the selected clip's endpoints, then test motion and timestamps.

2. **Explicit MPS/CPU placement for pooling.**
   PyTorch 2.14 build `08187d9e0fba026dc8217405802ab5381dc88d90` does not implement
   `aten::_adaptive_avg_pool3d` on MPS. Both `pool_latent` and visual `_blur_latent`
   fail when handed MPS tensors with fallback disabled. Normal intermediate
   placement works; `--gpu-only` or GPU-resident latents expose this failure.
   Keep those small preprocessing operations explicitly on CPU, or add a tested
   MPS implementation. Do not depend on users enabling a global silent fallback.
   Trilinear interpolation/backward and the three-step Adam refinement probes
   did execute on MPS; this is not a requirement to rewrite all refinement math.

3. **Optional performance work, not compatibility blockers.**
   Refinement currently inherits the CPU intermediate device. Benchmark any MPS
   variant with synchronization and realistic latent sizes before changing it.
   Step Curve can precompute its fixed blurred target within a sampler run,
   subject to preserving output arithmetic and cleanup. Upstream's CUDA timing
   harness does not synchronize MPS, so its raw MPS times would be misleading.

The optional RefMod Text Encode/Qwen reference presentation, long video/audio,
non-bit-exact acceleration combinations, and perceptual reference fidelity were
not qualified here. Source inspection shows the text node reconstructs visuals
for Qwen and can add work. RefMod compression minimizes latent reconstruction
error; it is not LoRA training or an identity-specialized learning objective.

[Machine-readable evidence](../benchmarks/data/h3-refmod-compatibility.json).
Local reproducer scripts, logs and saved test RefMods are in
`diagnostics/h3-refmod` in the parent workspace; matched rendering artifacts are
under `diagnostics/h3-recent-methods/refmod-mps-compat-*`.
