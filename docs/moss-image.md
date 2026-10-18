# MOSS-Image on Apple Silicon

MOSS-Image SFT and Base use the existing INT8 ConvRot paths plus new exact
BF16 SiTU, split-half RoPE and modulation kernels, exact PyTorch query reuse,
and sampling-local caches. No extra workflow node is required.

## Tested configuration

- M5 Ultra, 80 GPU cores / 256 GB, macOS 27.
- ComfyUI nightly `c06bf5f5`, PyTorch 2.14.1, Comfy Kitchen 0.2.38.
- SFT/Base INT8 ConvRot diffusion checkpoints, Qwen3.5-9B INT8 ConvRot encoder,
  `flux2-vae_small.safetensors`, 1024×1024, 25 steps, CFG 2.5, ER-SDE/simple.
- Prompt enhancement disabled for benchmarking; the original saved workflows
  are unchanged. Base retains its saved SLG Simple configuration.

The new module requires the reviewed Torch build and matching MOSS/Kitchen
source hashes, including the SiTU reference, plus Kitchen 0.2.38. Other
devices/builds retain existing paths. Updating ComfyUI or
Kitchen can disable the new module until the changed code has been validated.
The new attention path additionally requires BF16, batch one, 24 heads,
128 channels/head and 4096–4608 tokens. Other shapes retain current attention;
the elementwise and cache paths have separate guards.

## Controls

Set environment variables before starting ComfyUI, then restart it.
The startup log confirms activation with
`[AppleSilicon-Optimized/MOSS] exact SiTU/RoPE/modulation, query reuse and bounded sampling caches enabled.`

| Variable | Default | Effect |
|---|---:|---|
| `ASFP8_MOSS_FAST` | `1` | Master switch; `0` restores the preceding pack's MOSS execution. |
| `ASFP8_MOSS_SITU` | `1` | Exact BF16 soft-capped gated activation using PyTorch-generated lookup tables. |
| `ASFP8_MOSS_ROPE` | `1` | Split-half rotary operations after unchanged native RMS normalization. |
| `ASFP8_MOSS_MODULATION` | `1` | Fuse the operations after unchanged native RMS normalization, preserving intermediate BF16 rounding. |
| `ASFP8_MOSS_ATTENTION` | `1` | Hoist repeated query loads in the existing PyTorch dense-attention algorithm. No sparsity or attention quantization. |
| `ASFP8_MOSS_WEIGHT_CACHE` | `1` | Reuse owned, prepared FFN output weights within a sampling call. |
| `ASFP8_MOSS_REUSE` | `1` | Reuse identical positional embeddings and timestep-independent text features within a sampling call. |
| `ASFP8_MOSS_CACHE_GIB` | `4` | Requested weight-cache budget; `0` disables it. Capped by available RAM, Metal's recommended memory limit and an 8 GiB absolute ceiling. |

The tested weight cache uses approximately **1.97 GiB**. Positional/text caches
share a separate 32 MiB ceiling. Caches are released on completion, failure or
interruption; the allocator can retain freed storage for reuse. Live hooks,
custom transformer replacements and unsupported cases retain upstream paths.
Explicit attention backend selections are respected.

## Numerical contract

All new MOSS refinements preserve the preceding optimized implementation's
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
