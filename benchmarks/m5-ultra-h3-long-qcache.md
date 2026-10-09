# MiniMax H3: restore exact query caching for longer clips

2026-09-27 · M5 Ultra 80 GPU cores, 256 GB · macOS 27.0 · validated PyTorch
2.14.0 build `08187d9e0fba026dc8217405802ab5381dc88d90`.

## Finding

The existing query-cache wrapper rejected sequences longer than **32,768
tokens**. The saved 5-second H3 workflow has 15,458 tokens and benefits from
query caching. The same 864×480 workflow at 362 frames (~15 seconds, 24 fps)
has **44,600 tokens**: every main attention call fell back to ordinary PyTorch
SDPA. The other native kernels remained active.

This was our validation ceiling, not a 32,768-token restriction in the MPP
kernel. The installed header advances batch/head/block pointers using 64-bit
strides; the remaining narrowed offsets apply within tiles. The extension
keeps the existing stride guards and validates the larger supported range.

The wrapper now accepts **up to 65,536 tokens for the H3 BF16, 56-head, D=128
geometry**. Other model families retain their existing 32,768-token ceiling.
Build/header checks, dtype/layout/mask guards, tile sizes, matrix arithmetic,
softmax and reduction order are unchanged. Only the eligibility guard changes.
`ASFP8_MPP_QCACHE=0` still disables the optimization. No new flag is required.

## Matched 15-second workflow

The canonical 2.14 H3 workflow retained its prompt, seed 812764035, resolution,
24 fps, audio, res_multistep/simple sampler and existing native kernels.
Relaxed H3 options stayed off. Duration changed to 362 frames and sampling
was reduced to **two steps for this controlled comparison**.

| Measurement | Previous 32,768-token limit | Extended H3 limit |
|---|---:|---:|
| Sampling loop, per step | **41.656 s** | **36.849 s** |
| Two-step sampling, including setup | 83.422 s | 73.809 s |
| Complete two-step workflow, including video/audio decode and export | **116.303 s** | **106.672 s** |
| Main attention dispatches per render | 100 PyTorch fallbacks | 100 query-cache calls |

This is **11.5% less sampling time**, or 4.807 seconds saved per step. The
per-step difference projects to approximately **96 seconds over 20 steps**;
that is an extrapolation, **not a measured 20-step end-to-end result**. The
existing README's 124-frame / 20-step measurement remains unchanged.

Six renders ran in one process: baseline warmup, baseline, extended with
intermediate verification, extended, baseline, extended. Timing means use the
two ordinary warm baseline renders and two ordinary extended renders only.
Warmup and verification timings are excluded. The two control sampling loops
were 83.900 / 82.724 s; the two extended loops were 73.366 / 74.030 s.

All six runs have identical conditioning, initial noise, complete sampler
latents, decoded frame tensors and decoded audio tensors. Two real long
attention operations were also compared directly against PyTorch output bytes.
All main calls used the expected route; no failure latch was set.

Maximum sampled Metal allocation across the sequence was **60.633 GiB**,
including the intermediate verification run. External sampled process RSS
peaked at **11.166 GiB**. These overlapping unified-memory measurements must
not be added. Guards enforced 96 GiB RSS, 64 GiB available system memory and
an 80 GiB sampled Metal limit at sampler callbacks. No guard stopped the runs.

## Larger-shape kernel checks

Interleaved BF16 Q/K/V storage with nonzero offsets, 56 heads, D=128; median of
three timed repeats after each implementation's first call:

| Tokens | PyTorch SDPA | Query cache | Output bytes |
|---|---:|---:|---|
| 32,769 | 344.758 ms | 300.827 ms | Identical |
| 45,000 | 659.450 ms | 566.549 ms | Identical |
| 65,536 | 1417.633 ms | 1252.080 ms | Identical |

**24 regression tests pass**, including direct byte checks at 32,769, 44,600
and 65,536 tokens, existing ragged/GQA/scale cases, CPU/build fallbacks and
Qwen editing contracts. Boundary tests verify that H3 above 65,536 tokens
falls back and other families still stop at 32,768. Long tests do not allocate
quadratic-sized diagnostic masks.

## Interpreting the DGX Spark comparison

The user reports identical-workflow 20-step totals of 263 / 649 / 1075 s for
5 / 10 / 15 seconds: 2.47× and 4.09× the 5-second time. These are user-supplied
measurements, not Spark tests performed here.

The comparison highlighted a real lost optimization on the Ultra, now fixed.
It does not imply that all devices must have the same duration scaling.
H3 combines approximately linear tokenwise projections/MLPs with dense
attention whose arithmetic grows approximately quadratically with token count.
The measured 15-second sequence has 2.885× as many tokens as the 5-second
sequence, corresponding to about 8.32× the dense attention pair count.
Different relative costs of attention and the linear work produce different
overall scaling. The exact reason for Spark's relative cost balance cannot be
established from its total times alone. Also, Spark end-to-end totals should
not be directly compared with a sampler progress bar's seconds per iteration.

The extended cache recovers a measured part of the gap. It does not remove the
quadratic attention work or establish that the remaining implementation is
optimal.

## Reproduction

[Raw records, workflow, guard reports and harness](artifacts/h3-long-qcache/)
retain the six runs, dispatch counts, tensor hashes, microbenchmarks and test
output. Large saved tensors and videos remain in local
`diagnostics/h3-long-qcache`. The harness uses this machine's install/model
paths and is intended to live under the workspace's `diagnostics` directory.
The `baseline` diagnostic mode explicitly restores the old 32,768-token guard;
`extended` enables the validated H3 range. It does not change model arithmetic.

The tested runtime file was installed into `ComfyUI-AppleSilicon-2.14` and its
hash verified. Restart ComfyUI to load it. The exact route is automatic with
the existing native/PyTorch-attention configuration; relaxed modes remain off.
