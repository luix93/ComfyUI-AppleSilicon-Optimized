# SeedVR2 fixed conditioning

`conditioning.safetensors` contains ByteDance SeedVR's `pos_emb.pt` and
`neg_emb.pt`, renamed to ComfyUI's `positive_conditioning` and
`negative_conditioning`. Tensor bytes and BF16 dtype are unchanged.

Source: [ByteDance-Seed/SeedVR, e4de8c24441a67e1b7df56abea10645059bb1185](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185).
The original Apache 2.0 [LICENSE](LICENSE) is included here.

SHA-256 of the safetensors file:
`41faea5bb586cf806237b660e80fc66c3d8472268806863fd1e2d5ebec00ba3a`.

SHA-256 of each tensor's contiguous raw bytes:

- Positive, BF16 `[58, 5120]`: `92ff754e036702f20f1782b0f5327c3d120aac6370ddc2ef0cdfdb06988f70ab`.
- Negative, BF16 `[64, 5120]`: `f7279b7df52f3eb2e290019528c9794a5d02999c804f02f44a2e2a0d029c5410`.

Both tensors were also verified byte-identical to the conditioning embedded
in the tested ComfyUI normal and sharp 7B INT8 ConvRot checkpoints. The loader
supplies them only when both are absent from a recognized original 7B model.
No checkpoint is modified and no network access is needed at runtime.
