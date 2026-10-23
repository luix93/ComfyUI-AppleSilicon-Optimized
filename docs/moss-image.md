# MOSS-Image on Apple Silicon

MOSS-Image SFT and Base use the existing INT8 ConvRot paths plus new exact
BF16 SiTU, split-half RoPE and modulation kernels, exact PyTorch query reuse,
and sampling-local caches. No extra workflow node is required.

## Tested configuration

- M5 Ultra, 80 GPU cores / 256 GB, macOS 27.
- ComfyUI nightly `c06bf5f5`, PyTorch 2.14.1, Comfy Kitchen 0.2.38.
- SFT/Base INT8 ConvRot diffusion checkpoints, Qwen3.5-9B INT8 ConvRot encoder,
  `flux2-vae_small.safetensors`, 1024×1024, 25 steps, CFG 2.5, ER-SDE/simple.
- SFT also validated at 1248×1248 / 25 and 50 steps / CFG 2.5: warm means
  **34.07 s / 66.73 s** after the exact attention range extension.
  [1.5 MP follow-up](../benchmarks/m5-ultra-moss-15mp.md).
- Prompt enhancement disabled for benchmarking; the original saved workflows
  are unchanged. Base retains its saved SLG Simple configuration.

The new module requires the reviewed Torch build and matching MOSS/Kitchen
source hashes, including the SiTU reference, plus Kitchen 0.2.38. Other
devices/builds retain existing paths. Updating ComfyUI or
Kitchen can disable the new module until the changed code has been validated.
The new attention path additionally requires BF16, batch one, 24 heads,
128 channels/head and 4096–6656 tokens. Other shapes retain current attention;
the elementwise and cache paths have separate guards.

## Controls

Set environment variables before starting ComfyUI, then restart it.
The startup log confirms activation with
`[AppleSilicon-Optimized/MOSS] exact SiTU/RoPE/modulation, query reuse, joined CFG rows, INT8 tile schedule and bounded sampling caches enabled.`

| Variable | Default | Effect |
|---|---:|---|
| `ASFP8_MOSS_FAST` | `1` | Master switch; `0` restores the preceding pack's MOSS execution. |
| `ASFP8_MOSS_SITU` | `1` | Exact BF16 soft-capped gated activation using PyTorch-generated lookup tables. |
| `ASFP8_MOSS_ROPE` | `1` | Split-half rotary operations after unchanged native RMS normalization. |
| `ASFP8_MOSS_QK_FUSION` | `1` | Weighted Q/K RMS normalization and split-half RoPE in one kernel, keeping the BF16 store before RoPE. Byte-identical on full renders and 779 M stress values. |
| `ASFP8_MOSS_MODULATION` | `1` | Fuse the operations after unchanged native RMS normalization, preserving intermediate BF16 rounding. |
| `ASFP8_MOSS_RMS_MODE` | `exact` | `exact` keeps PyTorch's FP32 RMS reduction while removing full-size intermediates; `original` restores the previous route; `native` uses weighted native RMSNorm and changes rounding. |
| `ASFP8_MOSS_BF16_FFN_PAIR` | `1` | BF16 MOSS checkpoints only: paired FC1 columns with the exact SiTU lookup in the epilogue, removing the wide intermediate. |
| `ASFP8_MOSS_ATTENTION` | `1` | Hoist repeated query loads in the existing PyTorch dense-attention algorithm. No sparsity or attention quantization. |
| `ASFP8_MOSS_CFG_JOIN` | `1` | Run the positive and negative CFG branches through every INT8 projection as one joined matrix; attention, positions, text and modulation stay per branch. Exact: INT8 accumulation is independent of the row count. |
| `ASFP8_MOSS_GEMM_SCHEDULE` | `1` | 128-row, N-major INT8 tile order for MOSS projection shapes (up to 8,704 rows). Exact: integer accumulation is order-independent. |
| `ASFP8_MOSS_FFN_DISPATCH` | `1` | Skip unused FFN weight preparation on the reviewed core and owned cast-wrapper chain; same SiTU and INT8 module execution. |
| `ASFP8_MOSS_FFN_EPILOGUE` | `0` | Optional exact FC1/SiTU matrix-output fusion; reviewed SFT 1024/1248 tests. No approximate rotation or weight cache. |
| `ASFP8_MOSS_WEIGHT_CACHE` | `1` | Reuse owned, prepared FFN output weights within a sampling call. |
| `ASFP8_MOSS_REUSE` | `1` | Reuse identical positional embeddings and timestep-independent text features within a sampling call. |
| `ASFP8_MOSS_CACHE_GIB` | `4` | Requested weight-cache budget; `0` disables it. Capped by available RAM, Metal's recommended memory limit and an 8 GiB absolute ceiling. |

The current exact FFN dispatch avoids the formerly retained **1.97 GiB** weight
cache on the reviewed configuration. The cache remains as a fallback when that
dispatch is unavailable or disabled. Full-step measurements show less than 1%
E2E change; this cleanup is not a material speed claim.
[Execution audit and 25/50-step results](../benchmarks/m5-ultra-moss-execution.md).
Positional/text caches
share a separate 32 MiB ceiling. Caches are released on completion, failure or
interruption; the allocator can retain freed storage for reuse. Live hooks,
custom transformer replacements and unsupported cases retain upstream paths.
Explicit attention backend selections are respected.

## Numerical contract

The default MOSS refinements preserve the preceding optimized implementation's
conditioning, initial noise, complete sampler latents and decoded float tensors
on both full-step fixtures. This is not a claim that the entire existing pack
matches stock: its pre-existing MPS encoder placement and native INT8 W8A8
execution already change stock outputs.

A separate comparison enables only this module with
`ASFP8_ENABLE_ONLY=moss_fast`, leaving stock placement and linear operations in
place. This is a **global diagnostic selection**, disabling the pack's other
patch modules, not a per-workflow production preset. See the report for its
stock-byte checks and separate timings.

[Measurements, rejected experiments and test boundaries](../benchmarks/m5-ultra-moss-image.md).


## Research-only Kitchen/FFN switches

Two independent experiments are disabled by default:
`ASFP8_MOSS_KITCHEN_INT8=1` and `ASFP8_MOSS_SITU_QUANT=1`.
Both change output bytes and both lost to the exact path in full 25-step tests;
**they are not recommended acceleration presets**. The first honors eligible
MOSS checkpoint requests for Kitchen INT8; it does not globally enable Kitchen
or redirect other models. No current H3 preset changes.

[Implementation, timings, quality limits and CUDA reference instructions](../benchmarks/m5-ultra-moss-kitchen-int8.md).

## Optional exact FC1/SiTU fusion

`ASFP8_MOSS_FFN_EPILOGUE=1` enables the tested matrix-output fusion. It preserves
projection and activation rounding and leaves attention, ConvRot, FC2 and
sampling unchanged. The measured complete-render benefit is modest; the
previous exact route remains the default. The option requires the default
SiTU/FFN-dispatch switches and the reviewed BF16 INT8 ConvRot configuration.
Custom layer behavior and unsupported inputs retain the existing path.

[Paired full-render results and rejected larger fusion](../benchmarks/m5-ultra-moss-ffn-epilogue.md).


## MOSS attention modes

`ASFP8_MOSS_ATTN_MODE` selects the dense attention path:

| Value | Behavior |
|---|---|
| `tile32` (default) | Smaller query groups and direct projection-ready output. Same QK/PV arithmetic and reduction order as `exact`; complete-render byte checks. |
| `exact` | The previous 64-row exact attention. |
| `dense_int8` | INT8 QK and PV, with MOSS-specific preparation and direct output. Approximate versus exact. |
| `dense_int8_floatpv` | INT8 QK with BF16 PV, with MOSS-specific preparation and direct output. Approximate versus exact. |
| `int8_v3` | The pack's general M5 INT8 attention with Kitchen-style numerics (Q/K rotation, per-tile probabilities). Approximate versus exact. |

These modes retain every key and every sampling step. There is no SOL,
sparsity, token reduction or step skipping. The INT8 modes preserve their
preceding H3-derived dense algorithms' output bytes on the tested inputs;
that does **not** make them exact versus PyTorch. They are not CUDA Kitchen
byte-equivalence implementations.

The new modes are limited to the tested M5 Ultra / PyTorch 2.14.1 MOSS BF16
configuration, batch 1, 24 heads, D128 and 4,096–6,656 tokens. Unsupported
layouts, memory-budget failures and unavailable kernels retain exact attention.
Explicit backend/transformer overrides are respected. Restart after setting
the variable; the log identifies the activated mode. Keep
`ASFP8_MOSS_KITCHEN_INT8=0` for these comparisons. The optional FC1/SiTU fusion
is independently controlled by `ASFP8_MOSS_FFN_EPILOGUE`.

All INT8 modes, and `ASFP8_MOSS_KITCHEN_INT8=1`, are off by default and quantize
every attention call: all steps, blocks and both CFG branches. At 1248², 50
steps (three prompts, against the exact default):

| Mode | Sampling | Mean PSNR vs exact |
|---|---:|---:|
| exact default | 62.5 s | — |
| `int8_v3` | 61.2 s | 25.1 dB |
| `dense_int8_floatpv` | 59.8 s | 28.1 dB |
| Kitchen policy port | 105.6 s | 24.6 dB |

The Kitchen port's time reflects an untuned Metal port, not Kitchen's CUDA
speed. The gains are small, so the exact route remains the default.

[Dense attention tuning, practical timings and limitations](../benchmarks/m5-ultra-moss-attention-tuning.md).

## Joined CFG rows (default)

The October 18 grouping experiment is now the default `ASFP8_MOSS_CFG_JOIN`
route, in its joined-state form. Positive and negative hidden states stay joined
through all four INT8 projections in every block; each branch keeps its own
prompt length, positions, text embedding, attention and options.

It joins only:
- plain text-prompt CFG pairs (no ControlNet, area, hooks, patches or extra
  conditioning);
- at equal timesteps;
- with INT8-weight projections;
- inside the guarded MOSS sample.

Anything else runs the two branches separately, exactly as before. The opt-in
FFN epilogue keeps separate branches.

Together with the INT8 tile schedule and tile32 attention (October 23), with
every latent and decoded frame byte-identical to the previous route:

| Resolution | Previous exact route | New exact route | Less time |
|---|---:|---:|---:|
| 1024², 25 steps | 20.43 / 20.68 s | 19.31 / 19.25 s | 1.28 s / 6.2% |
| 1248², 25 steps | 32.39 / 32.40 s | 31.37 / 31.34 s | 1.04 s / 3.2% |

Setting `ASFP8_MOSS_CFG_JOIN=0`, `ASFP8_MOSS_GEMM_SCHEDULE=0` and
`ASFP8_MOSS_ATTN_MODE=exact` restores the previous route.
[Measurements](../benchmarks/m5-ultra-moss-exact-combo.md) ·
[original grouping experiment](../benchmarks/m5-ultra-moss-cfg-projections.md).

## October 19 core compatibility repair

ComfyUI `8f04fd6e` (`v0.42.0-3-g8f04fd6e`, reported by the instance as the
0.42.1 update) adds a `transformer_options` argument and custom RoPE controls.
Kitchen also moves to 0.2.39. The previous guards disabled the reviewed MOSS
path, increasing the matched 1024² SFT / 25-step warm render to 34.39 s.

The compatibility repair accepts the reviewed source and Kitchen version,
forwards the new options unchanged, and bypasses position reuse whenever custom
RoPE controls are present. The default path returns to about 22.3 s with the
same conditioning, noise, latent and decoded-frame bytes. No attention mode or
approximation is enabled by this repair.

[Updated-stack regression and numerical scope](../benchmarks/m5-ultra-comfy0421-smoke.md).
