# YuE2 Hum-to-Song — separate Apple Silicon workflow

Import [audio_yue2_hum_to_song.json](../workflows/audio_yue2_hum_to_song.json).
The existing Mothersuperior instrumental workflow is unchanged.
An [API version](../workflows/audio_yue2_hum_to_song.api.json) is also provided;
set its audio filename to an uploaded recording before submitting it.

## Required nodes and models

This graph requires [FS Audio Suite](https://github.com/KytraScript/ComfyUI-FS_Audio_Suite),
revision `26fe2e91d20ebf3531cf053c0c654fd3b849e5de` for the validation below.
It is a separate third-party installation, not bundled in this pack.
Install it in the instance's `custom_nodes` folder and install its requirements
using that instance's Python. The dependencies are librosa, scipy and soundfile.
Restart ComfyUI and refresh the browser after installation.

| Model | Location |
|---|---|
| `yue2_3b_int8_convrot.safetensors` | `models/checkpoints/yue2/` |
| `humsong_yue2_adapter_v1_comfy.safetensors` | `models/loras/yue2/` |
| `sheetsage2_bf16.safetensors` | `models/audio_encoders/` |

The graph contains no downloader node. These files are selected explicitly.
The hum adapter includes the real-audio decoder adaptation: do not stack the
old NAR v4 decoder LoRA on top of it. The instrumental planner LoRA is not
loaded in this new sung-song starter. The original instrumental graph keeps
its existing adapters.

## Use

1. In **Record or upload YOUR hum**, use Upload or Record for 10–30 seconds of
   real humming. No existing user recording is preselected. Start is 0 seconds;
   at most 30 seconds are used. You can change both values.
2. Edit **Style** and **Lyrics** in **Compose and render**. The included heavy-rock
   style and original sample lyrics are editable starting points.
3. Queue. The output node saves a FLAC plus the ABC score and settings under
   `output/audio/YuE2_HumToSong…`.

Defaults are 32 acoustic steps, `dpm_2` / `sgm_uniform`, Weirdness/CFG 1,
120-second maximum, hum influence 1, offset 0 and **continue from hum**.
The cap is a maximum, not a requested exact duration. With a connected hum,
FS Audio Suite uses a melody-mode score continued from the transcription.
**Hum only** restricts the score to the transcribed phrase; **ignore hum** lets
the planner write its own score while the adapter can still guide phrasing.
Keeping Weirdness/CFG and hum influence at 1 avoids their extra guidance passes.

This is melody/phrasing conditioning, not voice cloning. The adapter's additional
hum projections require the dedicated adapter loader and sampler; an ordinary
LoRA loader alone does not activate them.

## Validation and limits

The new graph was validated against the installed FS node schemas on ComfyUI
core `8f04fd6e` / PyTorch 2.14.1. All three model selectors resolve locally,
and all six graph links and widget orders were generated from those schemas.
The input validation used an owned synthetic file without queueing generation.
A real-weight MPS check loaded all 116 decoder patches, the transcriber and all
four injection projections. An eight-frame synthetic BF16 decoder call was
finite and responded to the hum condition. This is a compatibility check, not
a complete song or quality benchmark. Audio loading, 24→48 kHz resampling
and pYIN carrier extraction also passed on a three-second synthetic harmonic
fixture. No real-hum transcription was run.
No user recordings, images or SATA files were opened. Full-song generation and
perceptual melody-following quality have not been tested for this graph.

FS Audio Suite's custom decoder forward is a separate path from our usual
acoustic wrapper. Existing native YuE2 planner hooks can still apply, but the
text-only workflow's benchmark timings are not a hum-to-song speed promise.

## Credits

The hum adapter and its design are by Mothersuperior / Make the Robot Do It:
[model card](https://huggingface.co/Mothersuperior/YuE2-hum-to-song).
The inference nodes and frontend are from
[KytraScript/ComfyUI-FS_Audio_Suite](https://github.com/KytraScript/ComfyUI-FS_Audio_Suite).
This workflow is an original configuration for those nodes, not a bundled copy
of their runtime. The upstream suite and model list CC BY-NC 4.0 terms.
