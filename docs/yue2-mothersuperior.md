# YuE2 Mothersuperior instrumental workflow

Import [the visual workflow](../workflows/audio_yue2_mothersuperior_instrumental.json)
into ComfyUI. An [API version](../workflows/audio_yue2_mothersuperior_instrumental.api.json)
is also provided. This uses native ComfyUI nodes and the existing Apple Silicon
optimization pack; it adds no third-party node dependency.

## Model connections

| File | Location | Connection / starting strength |
|---|---|---|
| `yue2_3b_int8_convrot.safetensors` | `models/checkpoints/yue2/` | Existing checkpoint loader |
| `ar_lora_inst_v3abc_comfyui.safetensors` | `models/loras/yue2/` | Standard LoRA Loader: CLIP 1, MODEL 0 |
| `nar_lora_joint_v4_comfyui.safetensors` | `models/loras/yue2/` | LoRA Loader Model Only: MODEL 1 |

The instrumental LoRA changes the autoregressive planner in the checkpoint's
**CLIP** branch. Feed its CLIP output to **both** Generate ABC and Generate Music.
The NAR LoRA changes the acoustic **MODEL** branch; feed its MODEL output to
KSampler. The checkpoint's audio VAE goes directly to VAE Decode Audio.
This follows the [instrumental model card's ComfyUI pairing](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras).

The real-audio repository contains two distinct components. Its **tokenizer head**
converts MERT features from existing recordings into semantic tokens for
round-tripping/training. That is not a replacement for the checkpoint's text
tokenizer or audio VAE, and is unnecessary for this text-to-music graph. This
workflow uses the repository's **NAR decoder LoRA**. Its four projection/bias
diffs must load as well as its transformer LoRAs; a partial load is insufficient.
See the [real-audio model card](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).

## Starting settings

Keep both Generate nodes in **full** mode, and keep Generate ABC connected to
Generate Music. Style, section plan and seed are shared. Write genre, instruments,
mood and BPM in Style. The plan field contains actual newlines:

```text
[intro]
[verse]
[chorus]
[bridge]
[chorus]
[outro]
```

Use `[instrumental]` alone to let the model choose the structure. Timed sections
are another supported option; durations are guidance, not an exact clock. Keep
production descriptions in Style rather than inventing new section tag names.
The [author's prompting guide](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras#prompting-three-caption-styles)
describes all three forms.

The prepared graph retains 8192 ABC tokens, maximum music duration 120 s, CFG 1,
and the tested 32-step `dpm_2` / `sgm_uniform` acoustic sampler. Empty latent audio
receives **generated seconds** from Generate Music, not a separate fixed length.
The 120 s cap can cut a song off. Raise it to 240–300 s if you want more room
for the model to finish the section plan; this also increases runtime and memory.
These are starting settings, not a newly optimized LoRA-specific sampler recipe.

## Versions and alternatives

The v4 decoder is the reproducible first pairing explicitly named by the
instrumental author. The real-audio repository also offers ComfyUI v5/v8/v9
decoder adapters. Its author prefers v9 in their listening tests; that is not
our quality qualification for this instrumental workflow. Swap the MODEL-only
adapter to compare later, keeping everything else fixed. Never stack different
versions of these decoder adapters on top of each other.

For actual **input-recording tokenization**, additional MERT extraction and
semantic-token input nodes would be required. This workflow does not implement
that path and does not load user media. The optional merged audio VAE mentioned
upstream is also a separate experiment; the prepared graph keeps the tested VAE.

## Local validation

**October 8 RoPE follow-up:** the same text-only 20-second compatibility graph
now improves **30.947 → 21.181 s** in two paired warm runs, **31.6% less total
time**. All score/music IDs, conditioning, sampler latents, waveform and FLAC
bytes match the preceding GPU implementation and the frozen reference below.
The independently switchable `ASFP8_YUE2_ROPE` route is enabled by default on the
validated runtime; CPU RNG and attention remain unchanged.
[Paired benchmark and exactness checks](../benchmarks/m5-ultra-yue2-followup.md).

The following measurements are the original compatibility check:

On this M5 Ultra, the original 8192-token score allowance produces a complete
1254-token score. A 20-second music cap then gives two warm full-graph runs of
**31.781 / 30.995 s (31.388 s mean)** at 32 acoustic steps. All 112 AR patched
weights and 116 NAR patched weights load, including the four projection/bias
patches. Tokens, conditioning, latents, waveform and saved FLAC bytes repeat
identically. Peak measured MPS driver memory is **13.18 GiB**. Our phase-local
weight cache remains active with these standard loaders.

The supplied graph uses a 120-second maximum; its full-length run has not been
timed here. The short check deliberately caps semantic generation at 20 seconds,
so it checks compatibility rather than a naturally completed song. Determinism
does not establish musical quality or byte parity against an unoptimized LoRA
stack; this pass did not conduct a listening comparison.

Download file hashes, node mapping checks, repeated short-run timings and output
hashes are recorded in [the validation data](../benchmarks/data/yue2-mothersuperior-workflow.json).
The two adapter files are public safetensors and are verified against their
published SHA-256 values. The graph's links and widget order were constructed
from this installed ComfyUI's `/object_info` schema. Native loader/runtime
compatibility is checked with our existing INT8 checkpoint on MPS; no private
clips, images or recordings are involved.

The LoRAs intentionally change the generated music relative to the base model.
Our YuE2 runtime remains enabled; weight-hook or low-VRAM cases retain its
fallback. Restart/refresh ComfyUI if the downloaded files do not appear in the
LoRA selectors. The provided workflow requires no changes to the original saved
`audio_yue2_text2music.json`.
