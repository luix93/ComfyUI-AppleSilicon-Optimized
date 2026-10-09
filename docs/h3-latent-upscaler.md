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
