# H3 Latent Upscale · Apple Silicon

This node runs the learned MiniMax H3 latent upscaler on PyTorch Metal/MPS.
It is a port of [LBH-123-AI's upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/tree/40316cf008b2fd8663263270669eb4da23f89d2c), with the training architecture and channel-normalization constants retained.
The MIT notice is in `third_party/h3-latent-upscaler-MIT.txt`.

## Setup and use

1. Put an H3 **safetensors** upscaler checkpoint in `models/latent_upscale_models/` (subfolders are supported).
2. Add **H3 Latent Upscale · Apple Silicon**, under **Apple Silicon / Minimax H3**.
3. Connect a sampled H3 LATENT. Both plain video and combined video/audio latents work. Audio is carried through unchanged.
4. Select **mps**, **fp16**, and **scale = 2**. Connect the result to a refinement sampler or to H3 video VAE decode.

The picker recognizes H3's 24-channel input weights from checkpoint headers and excludes unrelated LTX upscalers in the same directory. It accepts the upstream 2D and 3D network formats; the real-checkpoint platform test uses `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`.

The node changes width and height, preserving batch size and frame count. Dimensions align to 32 pixels. Scale 1 returns the original latent without loading a model. Noise masks resize with nearest-neighbor interpolation, while other metadata survives. It does not automatically run a high-resolution diffusion refinement pass or reproduce the upstream repository's separate split-resampling node family.

## Precision and memory

**FP16 is the default.** Compute precision is independent of the checkpoint filename: the same FP16 checkpoint can be evaluated in FP16, BF16 or FP32. Casting FP16 weights to FP32 does not recover additional weight precision. BF16 has more exponent range but less mantissa precision. These compute choices are not bit-exact to one another.

The node uses strict safetensors loading into a metadata-only model, caches at most one CPU checkpoint, and releases the active GPU model after every call. It asks ComfyUI to make room before allocating GPU weights. A conservative **8 GiB working-memory admission budget**, index-size checks, per-block MPS synchronization, cancellation checks and low-system-memory checks bound execution. The estimate is conservative, not a hard limit on the entire ComfyUI process.

**Temporal chunk = 32** retains the upstream 3D default. It counts *latent* frames, not output video frames. Set **0** for full-clip processing; this is faster (1.85 versus 3.02 s) for the tested five-second 2× upscale on the M5 Ultra and avoids chunk-induced changes. A memory refusal suggests reducing output resolution or the 3D chunk length; the node does not silently change chunking. Temporal chunking changes normalization statistics and temporal context, so it is not bit-exact to a full-clip pass. Spatial tiling is not substituted for the network's global normalization.

The 2D checkpoint format runs its complete time sequence; its temporal chunk control is ignored. Real 2D checkpoint quality and speed have not been measured on this machine. No pickle checkpoint loader or automatic downloads are included.

## Platform validation

Measurements and numerical comparisons are recorded in [the H3 hybrid/upscaler benchmark report](../benchmarks/m5-ultra-h3-sol-hybrid.md).

## Distorted output after refinement

Decode the low-resolution latent and the upscaler's output separately before changing the upscaler. If both are intact but the refinement is damaged, check the refinement sampler, noise schedule and LoRA together. A high starting sigma gives the sampler much more freedom to redraw the scene; it is not just a sharpening control.

The [TaoMate H3 three-step conversion](https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI) recommends **Euler**, CFG 1 and LoRA strength 1. A tested three-step refinement using `res_multistep` and sigmas `0.9035, 0.6316, 0.3158, 0` produced broken textures even with cache, sparse attention and token reduction disabled. Switching only the refinement sampler to Euler removed that texture failure in the short comparison; a 124-frame, 1088×1600 confirmation also produced clean textures. Give the refinement its own sampler selector so the first pass stays unchanged. High-noise Euler refinement still redraws appearance and detail.

For stronger source preservation, the tested alternative `0.3158, 0.1579, 0` made a gentler two-step refinement with `res_multistep`, also checked on the complete 124-frame clip. That changes both the noise schedule and step count: this sampler uses Euler updates for its first and final steps, so this two-step case never enters its multistep correction. This is a workflow-specific starting point, not a universal schedule for every distilled LoRA. Compare motion and detail on the complete clip before adopting either recipe. Keep optional approximations off when establishing a quality baseline.

The October 12 regression check used identical saved input latents on the older ComfyUI/PyTorch 2.14.0 and updated ComfyUI/PyTorch 2.14.1 installs, with the same current patch pack. Both the damaged `res_multistep` refinement and the clean Euler refinement produced byte-identical video and audio sampler latents across those stacks. The upscaler also matched its earlier saved reference byte-for-byte. This case did not require a framework downgrade or an upscaler kernel change.
