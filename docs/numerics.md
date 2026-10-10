# Numerical behavior and reference paths

Exactness is stated against a particular reference implementation, dtype,
shape and stack. It is not a blanket guarantee that every optimized output
matches vanilla ComfyUI. The retained test counts below are historical;
see the release notes for release-candidate validation.

## Numerics policy

Numerical contracts, stated per patch:

- **Bit-exact**: all FP8 decodes, the int8 W8A8 kernel (including long-M
  scheduling and suffix dispatches), `fold_taps`, the conv3d channel fold
  (single fp32 accumulation = conv3d's own numerics), and the D=256 fused
  o-projection pre-pass in default mode. H3's LoRA-aware QKV partition
  epilogue and tiled FC1/FC2 bridge are byte-exact against their corresponding
  unfused native runtime-LoRA chains.
- **Exact-or-1-ulp**: the DiT fusions and VAE norm kernels mirror eager
  rounding at every point eager rounds (fp32 math between roundings); H3's
  D=128 cross-head convrot and optional fused SwiGLU use a reassociated FWHT
  and can move rare quantized values by one integer step. Images shift
  imperceptibly but deterministically.
- **Quantization-mode change**: H3's SwiGLU down-projection stays in the
  checkpoint's W8A8 path, while ComfyUI's generic `linear_input_act` helper
  dequantizes that seam to W8A16. The measured local error class is about 1.3%;
  `ASFP8_INT8_SWIGLU=0` restores the stock W8A16 route. The tiled MLP and LoRA
  bridges do not add further error: they are byte-exact against the selected
  native W8A8 SwiGLU/convrot chain.
- **Regime-gated**: flash41's low-precision score tiles (default; parity
  ~5e-4, `ASFP8_FA41_F32S=1` for fp32 tiles).

Patch #30 (memory accounting) changes *scheduling*, not numerics — outputs are
identical, only fewer reloads.

**October 9 publication check:** the current CPU regression suite passes
**934 tests**, with **441 GPU tests deliberately skipped**, and all **23 frontend
tests pass**. This check leaves the user's GPU/inference session untouched;
complete GPU parity, memory and workflow timing evidence is retained in the
individual benchmark reports. No new performance measurements were made for
this documentation/publishing pass.

Everything ships with parity tests against its stated reference path. The
historical full suite in the dedicated `ComfyUI-torch213-latest` environment
reported **397 passed, 19 skipped** on PyTorch 2.13.0. The Metal harnesses include a dirty-pool
test that poisons the MPS allocator with NaN pages before checking ragged
attention shapes, because parity on fresh, zeroed allocations structurally
cannot catch out-of-bounds reads.
