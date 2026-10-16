# H3 projection fusion experiments

Diagnostic-only prototypes from the [October 14 report](../../benchmarks/m5-ultra-h3-fusions.md).
None is imported by ComfyUI or changes its defaults. These are rejected or
inconclusive performance candidates, not supported acceleration settings.

Requires the tested M5 Ultra / Metal 4.1 / PyTorch 2.14.1 toolchain, a quiet GPU,
and at least 48 GiB available RAM. Build outputs go to a temporary cache.

Use the ComfyUI Python environment:

```sh
python dev/h3_fusion_experiments/screen.py --captures /path/to/captures --output /path/to/results.json
```

Captures are local `torch.save` dictionaries with `args`, `kwargs`, and `out`
from the native operations, moved to CPU. Expected filenames are
`i8_matmul2d_nt_fused_cs_qkv_rms_rope.pt` and `i8_mlp_bridge_prequant.pt`.
The report lists the tested shapes. Model-derived tensors are not distributed.
The script compares output bytes before timing, includes allocations and the
complete FC1 activation/rotation/quantization bridge, and guards the standard
local ComfyUI queue and available RAM. Stop other GPU work first.

`fc1_fused` prototype identifiers: 16/32 fuse through ConvRot and group maxima;
128 uses sequential gate/up GEMMs with fused SwiGLU; 129/65 use shared input
loads with separate gate/up accumulators (128/64-row tiles). Other shapes,
LoRA branches and hardware are not validated. Production dispatch is unchanged.

The INT8 GEMM derives from Cider; its MIT notice is in
[third_party/cider-MIT.txt](../../third_party/cider-MIT.txt). The prototypes
reuse the package's existing core and preserve its arithmetic and rounding.
