# SOL Metal hybrid on Dense Metal (v3): October 23, 2026

M5 Ultra 80 GPU / 256 GiB · macOS 27 · PyTorch 2.14.1 · ComfyUI nightly
`7a283320` · MiniMax H3, 864×480, 124 frames, 20 steps, seed 812764035, the
regression's study-room prompt. One server, one warm-up per mode, then one
measured run. PSNR is over all 124 decoded frames (uint8) against the exact run.

**SOL Metal · hybrid fast** runs SOL Fast (tau 1.3) in ordinary layers. It runs
dense INT8 attention in the protected layers 0, 1, 33, 34 and 35, only inside
the 20–90% step window (70 calls per run). Those protected-layer calls now use
the **Dense Metal (v3)** kernel instead of the original v1 kernel.

| Mode | End to end | Sampling | vs exact | PSNR vs exact | Worst frame |
|---|---:|---:|---:|---:|---:|
| Exact | 145.50 s | 134.03 s | — | — | — |
| SOL Metal · fast | 114.32 s | 102.89 s | −21.4% | 17.42 dB | 15.77 dB |
| Hybrid fast, v1 protected layers (previous) | 113.90 s | 102.51 s | −21.7% | 17.37 dB | 15.74 dB |
| **Hybrid fast, Dense Metal v3 (new default)** | **113.51 s** | **102.08 s** | **−22.0%** | **17.43 dB** | 15.71 dB |

- **Speed:** v3 is as fast as v1 here: 0.4 s apart, within run-to-run spread.
  The hybrid stays about 0.8 s (0.7%) faster than plain SOL Fast. Only 70 of
  1,000 attention calls change.
- **Difference from exact:** v3 brings the hybrid level with SOL Fast (17.43 vs
  17.42 dB), slightly closer than v1 (17.37 dB). The approximation is dominated
  by SOL's sparse routing in the ordinary layers, so the protected-layer kernel
  moves the result very little.
- **Exactness checks:** `ASFP8_H3_HYBRID_DENSE=v1` reproduces the October 19
  hybrid fast output byte for byte (9/9 captured tensors). SOL Fast matches its
  October 19 bytes as well. The v3 hybrid output differs from v1, as expected.

Harness: `diagnostics/h3-hybrid-v3-oct23/`, derived from the regression's H3
runner, with decoded frames saved for the PSNR comparison.
