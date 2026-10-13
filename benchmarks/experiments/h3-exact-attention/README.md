# Exact H3 attention experiments — October 13, 2026

These are **rejected diagnostic candidates**, not an acceleration mode. They
are never imported by the production node. The current production kernel
remains the byte-validated long-query-cache implementation.

See [the report](../../m5-ultra-h3-exact-attention-pipeline.md) and
[all measurements](../../data/m5-ultra-h3-exact-attention-pipeline.json).
The shader builder transforms the installed, hash-checked PyTorch MPP source;
it does not bundle PyTorch or Apple SDK headers. Existing repository third-party
notices apply. The partial-unroll/uniform experiments were informed by Apple's
MLX attention implementation, linked in the report.

## Reproduce

Use the validated M5 Ultra / PyTorch 2.14.1 Comfy Python environment with
`psutil`. Supply a trusted local `torch.save((q.cpu(), k.cpu(), v.cpu()), path)`
fixture containing contiguous BF16 tensors `[1,56,L,128]`. The measured public
T2V fixture used L=44,600; its multi-gigabyte tensors are not bundled. Other
inputs do not reproduce its reported timings or byte-comparison result.

From this directory, using that environment's `python`:

```sh
python guard.py --output /tmp/h3-attention-guard.json -- \
  python screen.py --capture /path/to/trusted-public-qkv.pt \
  --output /tmp/h3-attention-screen.json --rounds 7 \
  qk16_persistent vectorload rescale_late sync8
```

The guard stops the test when the main Comfy queue on port 8188 becomes busy,
process RSS exceeds 96 GiB, system available memory drops below 64 GiB, or
900 seconds elapse. MPS high/low allocation watermarks are 0.4/0.3. RSS is
**not** GPU peak allocation; the kernel screen reports only end-of-run driver
allocation. All GPU benchmarks must run serially.

Baseline is included automatically. Every candidate is compiled/warmed and
checked against its output bytes before seven shuffled timing rounds.
Nonmatching candidates are not timed. Output allocation and the Python launch
wrapper are included. The published paired repeat additionally used twenty
randomized baseline/candidate pairs per finalist; its raw pairs and bootstrap
intervals are in the data file. No candidate passed the improvement gate, so
none was installed or evaluated through a new complete sampler.

`m32col*` are intentionally incorrect fragment-packing controls. `qk64` and
`qk128` expose unsupported register-only Metal shapes and fail compilation;
they are not usable kernels. The data file also preserves initial interface
errors separately from the subsequently corrected tests.
