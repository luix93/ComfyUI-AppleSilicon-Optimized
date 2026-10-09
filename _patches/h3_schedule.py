"""Windows derived from sampler sigmas, shared by optional H3 accelerators."""
import math
import torch


def step_window(sigmas, start, end):
    if not 0 <= start <= end <= 1:
        raise ValueError("step fractions must satisfy 0 <= start <= end <= 1")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("step_fraction requires the sampler's sigma schedule")
    values = sigmas.detach().float().cpu().tolist()
    if (not all(math.isfinite(v) for v in values)
            or any(a <= b for a, b in zip(values, values[1:]))):
        raise ValueError("step_fraction requires finite, strictly decreasing sigmas")
    count = len(values) - 1
    first = min(count, math.ceil(count * start - 1e-9))
    stop = min(count, math.floor(count * end + 1e-9))
    if start == end:
        stop = first
    upper = float("inf") if first == 0 else (values[first - 1] + values[first]) * .5
    lower = float("-inf") if stop == count else (values[max(0, stop - 1)] + values[stop]) * .5
    return (first, stop, count), upper, lower
