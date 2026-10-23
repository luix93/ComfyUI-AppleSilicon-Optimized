# Fibo 2.0 on Apple Silicon

Fibo 2.0 Dev (text to image) and Edit Turbo (image editing) use the INT8
ConvRot and Lumina/Z-Image paths. They also get an exact Fibo route: joined
CFG rows, fused norm/modulation/residual passes and tile32 attention. Output is
byte-identical to the preceding pack. No extra workflow node is required. Use
core's `TextEncodeFibo2` workflows.

## Tested configuration

- M5 Ultra, 80 GPU cores / 256 GB, macOS 27.
- ComfyUI core `7a283320`, PyTorch 2.14.1, Comfy Kitchen 0.2.39.
- Comfy-Org INT8 ConvRot checkpoints: `fibo2_dev_int8_convrot`,
  `fibo2_edit_turbo_int8_convrot` and `qwen3vl_4b_int8_convrot`, plus
  `flux2-vae`.
- Dev at 1024², 30 steps, CFG 5: **88.49 s stock → 31.87 s**. Edit Turbo at a
  1024² reference, 4 steps: **16.19 → 6.61 s**.
  [Report](../benchmarks/m5-ultra-fibo2.md).

The route requires the reviewed Torch build on an M5 Ultra and the reviewed
`comfy/ldm/fibo2/model.py` source. It also needs matching Lumina `modulate` /
`apply_gate` sources and the pack's Lumina fast path (`ASFP8_DIT_EXT=1`, the
default). Updating ComfyUI can disable it until the changed code is validated.
Hooks, block replacements, attention overrides and any other unexpected case
keep the upstream forward. LoRA-patched or non-INT8 projections disable only
the CFG join.

## Controls

Set environment variables before starting ComfyUI, then restart it. The
startup log confirms activation with
`[AppleSilicon-Optimized/fibo2] exact joined CFG rows, fused norm/modulation/residual and tile32 attention enabled.`

| Variable | Default | Effect |
|---|---:|---|
| `ASFP8_FIBO2_FAST` | `1` | Master switch; `0` restores the preceding pack's Fibo execution. |
| `ASFP8_FIBO2_CFG_JOIN` | `1` | Run positive and negative CFG branches as one batch-2 forward when both prompts give the same gist count. The Perceiver, text cross-attention, embedder, final layer and modulation stay per branch. Needs unpatched INT8 projections. |
| `ASFP8_FIBO2_GLUE` | `1` | RMSNorm+modulate and RMSNorm+gate+residual in one pass each, with PyTorch's RMSNorm reduction order and every BF16 rounding. |
| `ASFP8_FIBO2_ATTN` | `1` | 32-row query tiles of the hash-checked PyTorch MPP attention kernel. Byte-identical to SDPA. |

Prompts longer than 256 text tokens raise the gist count, so a long positive
prompt with an empty negative runs the two branches separately. The glue and
attention parts still apply.
