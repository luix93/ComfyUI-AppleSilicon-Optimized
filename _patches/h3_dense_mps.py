"""Optional dense INT8 H3 attention for M5.

All token connections are retained. QK/PV use INT8 and FP32 running state;
this changes arithmetic and is not an exact PyTorch replacement. Derived from
our BSD-licensed CCV Metal path, inspired by Kitchen's full INT8 attention.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path
import re
import struct
import subprocess

import psutil
import torch


@functools.lru_cache(maxsize=1)
def supported():
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        return False
    try:
        brand = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            timeout=2,
        )
        return "M5" in brand
    except (OSError, subprocess.SubprocessError):
        return False


def scratch_bytes(tokens, heads, layout_copies=3):
    """Temporary allocations only, including worst-case protected-prefix copies.

    Dense attention never constructs token-by-token scores. Its current
    quantizer still writes pooled carriers; routing maps are allocated for the
    shared ABI but never read or cast to INT64. Count those actual buffers,
    rather than SOL's much larger route-reduction allowance.
    """
    blocks=(tokens+63)//64; padded_blocks=(blocks+63)//64*64
    padded_tokens=blocks*64
    return (heads*tokens*128*(2*layout_copies+2+4)
            + heads*padded_tokens*128*3
            + heads*blocks*128*4 + heads*padded_blocks*128*9
            + heads*128*12 + heads*blocks*blocks
            + heads*padded_tokens//4 + heads*blocks*16
            + heads*padded_blocks//8
            + heads*blocks*((blocks+31)//32)*4)


class WorkspaceBudgetExceeded(ValueError):
    """This attention shape exceeds the selected budget or current headroom."""


def workspace_budget(memory_budget_gib=0.0):
    """Return (effective budget, headroom) in bytes without GPU synchronization.

    Auto uses at most 1/8 of physical RAM and half the safe headroom. Manual
    ceilings retain an OS reserve and respect Metal's recommended working set.
    Allocator-cached memory is not assumed to be freely reclaimable.
    """
    if not math.isfinite(memory_budget_gib) or memory_budget_gib < 0:
        raise ValueError("Dense INT8 temporary memory must be finite and non-negative; 0 means Auto")
    vm = psutil.virtual_memory()
    reserve = max(2 * 2**30, int(vm.total * 0.05))
    metal_free = max(0, torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory())
    headroom = max(0, min(vm.available, metal_free) - reserve)
    budget = (min(vm.total // 8, headroom // 2) if memory_budget_gib == 0
              else min(int(memory_budget_gib * 2**30), headroom))
    return budget, headroom


def check_workspace(tokens, heads, memory_budget_gib=0.0, layout_copies=3):
    required = scratch_bytes(tokens, heads, layout_copies)
    budget, headroom = workspace_budget(memory_budget_gib)
    if required > budget:
        setting = "Auto" if memory_budget_gib == 0 else f"manual {memory_budget_gib:g} GiB"
        raise WorkspaceBudgetExceeded(
            f"Dense INT8 needs {required / 2**30:.2f} GiB temporary memory "
            f"({tokens:,} tokens, {heads} heads); budget {budget / 2**30:.2f} GiB "
            f"({setting}), safe headroom {headroom / 2**30:.2f} GiB. "
            "Set Attention acceleration > Settings > Temporary memory (GiB), 0=Auto. "
            "If headroom is limiting, free memory or reduce the sequence.")


@functools.lru_cache(maxsize=12)
def _library(tokens, heads, exact_prefix, schedule):
    source = (Path(__file__).parent / "h3_dense_shaders/int8.metal").read_text()
    for index, (name, value) in enumerate(
        (("N", 1), ("T", tokens), ("H", heads), ("QB", 64))
    ):
        needle = f"constant uint SOL_{name} [[function_constant({index})]];"
        if source.count(needle) != 1:
            raise RuntimeError("H3 SOL shader constants have changed")
        source = source.replace(needle, f"constant constexpr uint SOL_{name} = {value};")
    if exact_prefix:
        needle = "  const uint n = group.z, h = tile.y, nh = n * SOL_H + h;"
        if source.count(needle) != 1:
            raise RuntimeError("H3 SOL shader dispatch has changed")
        source = source.replace(needle, "  if (tile.x * SOL_QB < p.begin) return;\n" + needle)
    if schedule == "linear":
        start = source.index("  const uint2 tile = morton_decode_rectangular_2d(group.x,",
                             source.index("kernel void sol_attention"))
        end = source.index(";", start) + 1
        source = (source[:start] +
                  "  const uint2 tile = uint2(group.x % query_groups, group.x / query_groups);" +
                  source[end:])
    # CCV uses a common sparse buffer-index ABI. PyTorch binds positional
    # arguments, so compact the declared buffers independently for each kernel.
    bindings = {}
    for match in reversed(list(re.finditer(r"kernel void (\w+)\(", source))):
        start, end = match.start(), source.index("{", match.start())
        signature = source[start:end]
        indices = [int(m[1]) for m in re.finditer(r"\[\[buffer\((\d+)\)\]\]", signature)]
        bindings[match[1]] = indices
        remap = {old: new for new, old in enumerate(indices)}
        signature = re.sub(
            r"\[\[buffer\((\d+)\)\]\]",
            lambda m: f"[[buffer({remap[int(m[1])]})]]", signature,
        )
        source = source[:start] + signature + source[end:]
    return torch.mps.compile_shader(source), bindings


@functools.lru_cache(maxsize=128)
def _parameters(tokens, heads, video_start, scale, tau):
    raw = struct.pack("=6I2f2I", 1, tokens, heads, 64, video_start,
                      tokens, scale, tau, 64, 1)
    return torch.tensor(list(raw), dtype=torch.uint8, device="mps")


def attention(q, k, v, *, tau, video_start, scale=None, exact_prefix=True, schedule="linear",
              memory_budget_gib=0.0, topk_ratio=0.0):
    """Return BHSD output and on-device selected-block counts.

    Boundary query blocks are fully protected, rounded outward to 64 rows.
    This protects their direct attention calculation; later layers can still
    propagate changed video information into audio or conditioning streams.
    """
    if not supported():
        raise RuntimeError("H3 dense INT8 requires an M5 GPU and torch.mps.compile_shader")
    if schedule not in ("linear", "morton"):
        raise ValueError("Unknown H3 SOL workgroup schedule")
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
            or q.shape[0] != 1 or q.shape[-1] != 128
            or any(t.device.type != "mps" or t.dtype != torch.bfloat16 for t in (q, k, v))):
        raise ValueError("H3 dense INT8 requires matching MPS BF16 [1,H,S,128] tensors")
    _, heads, tokens, _ = q.shape
    scale = 128 ** -0.5 if scale is None else float(scale)
    if not (0 <= video_start <= tokens and tokens > 0 and heads > 0
            and math.isfinite(tau) and tau >= 0 and math.isfinite(scale)):
        raise ValueError("Invalid H3 SOL bounds or attention parameters")
    if topk_ratio != 0:
        raise ValueError("Dense INT8 retains all tokens; topk_ratio must be zero")
    check_workspace(tokens, heads, memory_budget_gib, sum(not x.is_contiguous() for x in (q, k, v)))

    # H3 often already supplies this physical layout; otherwise include the
    # conversion cost. Model tensors retain BF16 range throughout preparation.
    inputs = [x.contiguous() for x in (q, k, v)]
    blocks = (tokens + 63) // 64
    padded_blocks = (blocks + 63) // 64 * 64
    padded_tokens = blocks * 64

    def alloc(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device=q.device)

    output = alloc(q.shape, torch.bfloat16)
    pooled_q = alloc((heads, blocks, 128))
    pooled_k = alloc((heads, padded_blocks, 128), torch.float16)
    pooled_v = alloc((heads, padded_blocks, 128))
    statistics = alloc((heads, 128, 2))
    routes = alloc((heads, blocks, blocks), torch.uint8)
    quantized = [alloc((1, padded_tokens, heads, 128), torch.int8) for _ in range(3)]
    query_scales = alloc((heads, padded_tokens // 16))
    key_scales, value_scales = [alloc((heads, blocks)) for _ in range(2)]
    value_mean = alloc((heads, 128))
    summary_keys = alloc((heads, padded_blocks, 128), torch.int8)
    summary_values = alloc((heads, padded_blocks, 128), torch.float16)
    summary_scales = [alloc((heads, padded_blocks // 64)) for _ in range(2)]
    route_bits = alloc((heads, blocks, (blocks + 31) // 32), torch.int32)
    buffers = [*inputs, output, pooled_q, pooled_k, pooled_v, statistics,
               routes, *quantized, query_scales, key_scales, value_scales,
               value_mean, summary_keys, summary_values, *summary_scales,
               route_bits, _parameters(tokens, heads, int(video_start), scale, float(tau))]
    library, bindings = _library(tokens, heads, exact_prefix, schedule)
    tile = 1
    while tile < (4 if tokens <= 20480 else 8) and heads >= tile * 8:
        tile *= 2
    mean_threads = max(256 if tokens <= 20480 else 128, 32 * tile)
    padded_heads = 1 << (heads - 1).bit_length()
    padded_queries = 1 << (blocks - 1).bit_length()
    stages = (
        ("sol_v_mean", padded_heads * (32 // tile), 1, mean_threads),
        ("sol_quantize", blocks, heads, 128),
        ("sol_attention", blocks * heads if schedule == "linear" else padded_queries * padded_heads, 1, 128),
    )
    for name, width, height, threads in stages:
        getattr(library, name)(
            *[buffers[i] for i in bindings[name]],
            threads=(width * threads, height, 1), group_size=(threads, 1, 1),
        )
    if exact_prefix and video_start:
        end = min(tokens, (video_start + 63) // 64 * 64)
        protected = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, :end], k, v, scale=scale,
        )
        output[:, :, :end].copy_(protected)
    return output, torch.full((heads, blocks), blocks, device=q.device, dtype=torch.int64), routes.numel()
