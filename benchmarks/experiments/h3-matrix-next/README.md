# Exact QK/PV execution screens

These are diagnostic candidates, **not production acceleration modes**. None
improved on the preceding packed K/V experiment. See the
[report](../../m5-ultra-h3-matrix-execution.md) and
[measurements](../../data/m5-ultra-h3-matrix-execution.json).

Use the validated M5 Ultra / PyTorch 2.14.1 environment. From the repository root,
set `PY` to that environment's Python and provide the trusted public-authored
BF16 `[1,56,44600,128]` Q/K/V tuple used in the previous attention investigations:

```sh
"$PY" benchmarks/experiments/h3-exact-attention/guard.py \
  --output /tmp/h3-matrix-guard.json -- \
  "$PY" benchmarks/experiments/h3-matrix-next/screen.py \
  --capture /path/to/authored-qkv.pt --output /tmp/h3-matrix.json \
  --rounds 11 mixedpv_n64_k32 pvorder packed_baseline packed_qk32_pv32
```

The screen includes baseline automatically; checks bytes before timing; excludes
nonmatching/unsupported candidates; randomizes measurement order; and includes
output allocation plus packing costs. Failed output checks are never offered as
approximate modes. Every measured batch is retained separately in the report's
data file, including its own baseline. `screen.py` lists all selectable variants.

The module builds on the installed hash-checked PyTorch shader through the
existing query-cache helper and reuses the earlier standalone packing helper.
No third-party SDK/header sources or captured tensors are bundled. Its generated
Metal source retains the current softmax/reduction code and precision flags.

`mixed*` tests cooperative left / device-backed right operands. `regpv*` and
`regqk*` test register-only shapes. `pvorder*` reorder independent output columns.
`qk_widen*` use lossless BF16-to-FP32 input widening. `packed_*` include physical
K/V packing. `pv_transpack*` include a physical V transpose into contiguous
tiles. The two `*_trans` view-only prototypes are **invalid** and intentionally
fail the output-byte gate; they do not demonstrate numerical approximation.
N64/N128 `regpv*` variants intentionally reproduce unsupported MPP shape errors.

For hardware profiling, use the
[native capture/profiler tools](../h3-gpu-profile/README.md). The two profiles in
this report use `source('mixedpv_n64_k32')` or `source('pvorder')` as the native
input directory's `baseline.metal`, with the original contiguous Q/K/V buffers.
The native runner's usual 128-thread group and BQ64 grid apply. Verify captured
output bytes before interpreting its counters. **Packed variants require their
packed input buffers; do not feed them the original native buffers.**

The guard aborts for a busy main ComfyUI queue, low available system memory,
excessive process RSS or timeout. Run GPU jobs serially. The guard's RSS field
is not total GPU/profiler memory. No sampler is automatically launched by these
scripts, and they never install or enable a production patch.
