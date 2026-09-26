# Qwen-Image 2.1: INT8 ConvRot compatibility and exact fusions

This records the initial text-to-image optimization pass. The subsequent
[image-edit investigation](m5-ultra-qwen21-edit.md) adds exact attention
scheduling for long reference prefixes and exact Qwen VAE scheduling.
The README's top table tracks the latest installed measurements.

Measured 2026-09-25 on **M5 Ultra, 80 GPU cores, 256 GB**, macOS 27.0
(26A428), Python 3.13.12, PyTorch 2.14.0
(`08187d9e0fba026dc8217405802ab5381dc88d90`), and ComfyUI
`b33e2b55cae074eca5aec96283cceac19aa249ba`.

## Final measurements

| Configuration | Warm end to end, mean | Measured range | Sampling loop | VAE decode |
|---|---:|---:|---:|---:|
| Stock 2.14, acceleration node absent | 29.283 s | 29.262–29.317 s | 28.529 s | 0.555 s |
| Existing native kernels + encoder repair | 15.287 s | 15.238–15.314 s | 14.180 s | 0.902 s |
| Installed native kernels + new exact Qwen fusions | **13.152 s** | 13.118–13.192 s | **12.068 s** | 0.889 s |

The final stack takes **55.1% less total time than stock (2.23× speed)**.
The new fusions save **2.135 s / 14.0%** over the otherwise unchanged,
repaired native stack. Paired development runs also measured about 2.1 s
saved. These experiments establish a repeatable improvement, not an
absolute hardware limit.

Each final series discards two complete renders, then measures three.
End-to-end time includes sampling, VAE decode and PNG export. Model loading
and unchanged text encoding are cached during measured repeats; first-render
times were 35.121 s stock, 18.912 s repaired native, and 16.936 s final.
Only one benchmark GPU job runs at a time. The harness synchronizes MPS at
sampler callbacks and stage boundaries, and hashes/saves reference tensors
outside the execution timer. Final confirmation uses the installed custom
node with no separate diagnostic compatibility shim.

## Frozen workflow

The base is the user's saved `image_qwen_image_2_1_t2i.json` workflow:

- 1024×1024, 25 steps, Euler / simple, CFG 1, denoise 1.
- Seed `593103825222985`; unchanged long fashion-portrait prompt.
- DiT: `qwen/qwen_image_2.1_int8_convrot.safetensors`.
- Text encoder: `qwen3vl_8b_int8_convrot.safetensors`.
- VAE: `qwen_image_2.1_vae_bf16.safetensors`.
- Original SaveImageAdvanced PNG export, 8-bit sRGB.

Both configurations select `--use-pytorch-cross-attention`. Native uses
`ASFP8_NATIVE=1`, `ASFP8_ATTN_BACKEND=pytorch`, `MTLFLASHATTN_SDPA=off`;
the relaxed H3 preset remains off. ComfyUI's existing exact Qwen prefix cache
is preserved in both. No prompt, seed, steps or resolution shortcut is used.
Initial noise is byte-identical across all configurations.

The [raw results](data/m5-ultra-qwen21.json) include the saved workflow,
checkpoint SHA-256 hashes, per-run timings, runtime source hashes, tensor
hashes, numerical comparisons and experiments. Larger reference tensors and
the execution harness remain in the workspace's `diagnostics/qwen21-ultra`.

## Why the text encoder failed

The node's existing text-encoder placement patch moves this encoder from CPU
to MPS, exposing an upstream embedding-cast issue. ComfyUI's cast helper can
return an already-dequantized BF16 weight tensor after a device transfer.
The mixed-precision embedding path then treats it as INT8 data and attempts
another dequantization, raising `NoCapableBackendError` because the decoder
requires INT8 input. The failure was reproduced before applying the repair.

`int8_embedding.py` retains ComfyUI's cast/offload context and gathers rows
directly when the result is floating point. If the cast preserves a
QuantizedTensor, it retains the ordinary quantized embedding decoder.
Weight hooks and other formats retain their original path. No checkpoint
or ComfyUI core files are modified. Stock ComfyUI keeps this encoder on CPU
and completes without this repair, so the stock benchmark has the custom
node completely absent.

## Accepted kernel work

- **Q/K RMSNorm + RoPE:** one Metal dispatch retains FP32 reduction order,
  weighted BF16 rounding, and rotary arithmetic. Projection modules, weight
  casting, output layout and the attention callback remain unchanged.
- **Modulation:** keep native LayerNorm and combine the following BF16
  scale/add operations in one pass, choosing the prefix or target scale per
  row. This also removes repeated normalization of prefix rows.
- **Gated residual:** one in-place pass selects the correct row gate and
  preserves the original multiply/add arithmetic and final BF16 rounding.

The model-specific paths require the validated M5 Ultra / Torch build,
supported BF16 geometry and inference mode. Unsupported inputs and attention
patch hooks retain the original path. Matrix multiplies, attention and VAE
kernels were not changed by these fusions.

## Exactness and visual comparison

**The new fusions are byte-identical to the repaired native stack.** Checks
cover affected operations, complete transformer blocks, conditioning,
initial noise, complete sampler latents and final decoded RGBA tensors.
The full 25-step workflow matched on every final repeat. Two additional
four-step cases (a teapot with seed 42 and a Prague canal with seed 901)
matched too; each verified 160 complete transformer-block calls against
the unfused path. The original short case verified another 96 block calls.

**123 targeted tests passed; 4 skipped.** Tests cover the cast repair,
Parameter weights, nonzero storage offsets, sequence/prefix boundaries,
multiple random inputs, opt-outs/fallbacks and the existing attention,
SDXL and Krea2 routes.

Stock and native are **not** byte-identical stacks: existing W8A8 compute
and CPU-versus-MPS text encoding differ. This is not a new relaxed Qwen mode.
Stock-to-final conditioning RMSE is 0.06207, latent RMSE 0.29559, and decoded
RGBA RMSE 0.05673 (maximum absolute pixel difference 1.0). Both outputs are
finite. Repaired-native-to-final errors are zero for every checked tensor.

| Stock 2.14 | Final optimized |
|---|---|
| ![Stock portrait](data/m5-ultra-qwen21-stock.png) | ![Optimized portrait](data/m5-ultra-qwen21-optimized.png) |

Composition and the monochrome/lime palette remain similar, while fine hair,
fabric and background-line details differ between the complete stacks. One
portrait does not establish general visual-quality parity; the new exact
fusions themselves introduce no pixel differences in the validated cases.

## Experiments not promoted

| Candidate | Finding |
|---|---|
| Cached-query MPP attention for Qwen's 32 heads / Q4096 / K4431 | Byte checks passed, but two-step sampling was slower: 1.278/1.242 s control versus 1.306/1.261 s candidate. Qwen stays on ordinary 2.14 attention. |
| INT8 MLP bridge, 1024-row tile | Short screens looked promising; full renders regressed to 13.26–13.37 s versus about 13.14 s with the accepted fusions alone. |
| MLP bridge, 2048/4096-row tiles | About 13.26 s / 13.18–13.19 s; no repeatable complete-workflow gain. Outputs remained exact. |
| Scoped native-PyTorch VAE convolution routing | Ten repeated decodes of the same latent averaged 0.784361 s versus 0.784330 s: effectively identical, with matching bytes. |
| Stock VAE normalization / broader stock VAE routing | Approximately 0.40–0.43 s decode, but decoded pixels changed. Not included in this exact optimization stack. |

At the end of this initial pass, no MLP bridge, VAE change or new Qwen
attention specialization was installed. The later image-edit pass accepts
a different VAE scheduling change and restricts the attention specialization
to longer image-reference contexts; text-only attention stays unchanged.

## Controls

Restart the 2.14 installation after updating the custom node. The encoder
repair is on by default; `ASFP8_INT8_EMBEDDING=0` disables it independently.
With native kernels enabled, the new Qwen fusions default on when supported.
`ASFP8_QWEN21_FAST=0` disables these (and the later image-edit additions), retaining the encoder repair and
existing generic kernels. Set any of `ASFP8_QWEN21_ROPE`,
`ASFP8_QWEN21_MODULATE`, or `ASFP8_QWEN21_RESIDUAL` to `0` to disable that
individual fusion. Other installations and model files are unchanged.
