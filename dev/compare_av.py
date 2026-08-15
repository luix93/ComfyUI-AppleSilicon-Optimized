#!/usr/bin/env python3
"""Compare matched H3 video/audio artifacts produced with the same seed.

The report intentionally includes exact decoded equality as well as perceptual
and temporal metrics.  A high spatial SSIM alone can hide changed motion, and
video-only comparison can hide H3's audio divergence.
"""

import argparse
import hashlib
import json
import math
from collections import namedtuple
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

# Rec.601 luma weights, matching the coefficients used for rgb24 -> gray.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# A ROI smaller than this many pixels on either axis carries too little edge
# structure for the alignment search to mean anything.
MIN_ROI_PIXELS = 16

# Apparent background translation beyond this many pixels per frame pair is not
# a static background, so the phase-correlation search never looks that far.
MAX_BACKGROUND_SHIFT = 16

# Guards the normalisation for a completely flat (edge-free) reference crop.
_EDGE_EPSILON = 1.0e-6

BackgroundRoi = namedtuple("BackgroundRoi", "name x0 y0 x1 y1")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_video(path):
    with av.open(str(path)) as container:
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(video=0)
        ]
    if not frames:
        raise ValueError(f"no video frames in {path}")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError(f"variable video frame shape in {path}")
    return np.stack(frames)


def _decode_audio(path, sample_rate=48000):
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None, None
        resampler = av.AudioResampler(
            format="fltp", layout="stereo", rate=sample_rate)
        chunks = []
        for frame in container.decode(audio=0):
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            for output in converted:
                if output is not None:
                    chunks.append(output.to_ndarray())
        flushed = resampler.resample(None)
        if not isinstance(flushed, list):
            flushed = [flushed]
        for output in flushed:
            if output is not None:
                chunks.append(output.to_ndarray())
    if not chunks:
        return np.empty((2, 0), dtype=np.float32), sample_rate
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False), sample_rate


def _ssim_per_frame(reference, candidate):
    """Classic 11x11 Gaussian-window RGB SSIM for uint8 frame batches."""
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError("SSIM inputs must be matching [frames,height,width,3]")
    if reference.shape[-1] != 3 or min(reference.shape[1:3]) < 11:
        raise ValueError("SSIM requires RGB frames at least 11x11")

    x = torch.from_numpy(reference).permute(0, 3, 1, 2).float().div_(255.0)
    y = torch.from_numpy(candidate).permute(0, 3, 1, 2).float().div_(255.0)
    coords = torch.arange(11, dtype=torch.float32) - 5.0
    gaussian = torch.exp(-(coords * coords) / (2.0 * 1.5 * 1.5))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :]).view(1, 1, 11, 11)
    window = window.expand(3, 1, 11, 11).contiguous()

    mu_x = F.conv2d(x, window, groups=3)
    mu_y = F.conv2d(y, window, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, groups=3) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.mean(dim=(1, 2, 3)).numpy()


def _psnr(mse, peak=1.0):
    return math.inf if mse == 0.0 else 10.0 * math.log10((peak * peak) / mse)


def parse_background_roi(spec, index=0):
    """Parse one ``[name:]x0,y0,x1,y1`` ROI declared in normalized coordinates.

    Normalized coordinates keep a single ROI string valid at any resolution.
    Every failure names the offending ROI; nothing is silently clamped.
    """
    default_name = f"roi{index}"
    if not isinstance(spec, str):
        raise ValueError(
            f"background ROI '{default_name}': expected a string, "
            f"got {type(spec).__name__}")
    text = spec.strip()
    name = default_name
    if ":" in text:
        name, _, text = text.partition(":")
        name = name.strip()
        text = text.strip()
        if not name:
            raise ValueError(
                f"background ROI '{default_name}' ({spec!r}): empty ROI name "
                "before ':'")

    def fail(reason):
        return ValueError(f"background ROI '{name}' ({spec!r}): {reason}")

    fields = [field.strip() for field in text.split(",")]
    if len(fields) != 4 or any(not field for field in fields):
        raise fail(
            "expected '[name:]x0,y0,x1,y1' with 4 comma-separated numbers, "
            f"got {len(fields)} field(s)")
    values = []
    for label, field in zip(("x0", "y0", "x1", "y1"), fields):
        try:
            value = float(field)
        except ValueError:
            raise fail(f"{label}={field!r} is not a number") from None
        if not math.isfinite(value):
            raise fail(f"{label}={field!r} is not finite")
        if not 0.0 <= value <= 1.0:
            raise fail(f"{label}={value!r} is outside the normalized range [0,1]")
        values.append(value)
    x0, y0, x1, y1 = values
    if x0 >= x1:
        raise fail(f"empty or inverted ROI on x: x0={x0!r} must be < x1={x1!r}")
    if y0 >= y1:
        raise fail(f"empty or inverted ROI on y: y0={y0!r} must be < y1={y1!r}")
    return BackgroundRoi(name, x0, y0, x1, y1)


def parse_background_rois(specs):
    """Parse a repeatable ``--background-roi`` list into `BackgroundRoi`s."""
    if not specs:
        return []
    return [parse_background_roi(spec, index)
            for index, spec in enumerate(specs)]


def resolve_background_roi(roi, height, width, min_pixels=MIN_ROI_PIXELS):
    """Map a normalized ROI onto integer pixel bounds for one frame size."""
    x0 = int(round(roi.x0 * width))
    x1 = int(round(roi.x1 * width))
    y0 = int(round(roi.y0 * height))
    y1 = int(round(roi.y1 * height))
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if (x1 - x0) < min_pixels or (y1 - y0) < min_pixels:
        raise ValueError(
            f"background ROI '{roi.name}' resolves to "
            f"{max(x1 - x0, 0)}x{max(y1 - y0, 0)} pixels at {width}x{height}, "
            f"below the {min_pixels}x{min_pixels} minimum")
    return x0, y0, x1, y1


def _to_gray(frames):
    """Rec.601 luma for a [frames,height,width,3] uint8 batch."""
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            f"expected [frames,height,width,3] video, got {list(array.shape)}")
    return array.astype(np.float32) @ _LUMA


def _bounded_shift_candidates(size, max_shift):
    """Integer shifts the search is allowed to consider along one axis."""
    # Beyond (size-1)//2 a "shift" is ambiguous with its circular alias and
    # would leave no mutually valid pixels to compare.
    limit = max(0, min(int(max_shift), (size - 1) // 2))
    return np.arange(-limit, limit + 1)


def phase_correlation_shift(reference, moved, max_shift=MAX_BACKGROUND_SHIFT):
    """Integer (dy, dx) translation between two 2D crops, bounded by max_shift.

    The bound is applied to the *search*: only candidate shifts within
    +-max_shift are scored, so an unreliable far-away correlation peak can
    never be picked and then clipped into range.
    """
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moved, dtype=np.float32)
    if ref.ndim != 2 or ref.shape != mov.shape:
        raise ValueError("phase correlation needs two matching 2D crops")
    height, width = ref.shape

    # Mean removal plus a separable Hann window suppresses the spectral leakage
    # caused by the crop borders not being a true circular shift of each other.
    window = (np.hanning(height)[:, None] * np.hanning(width)[None, :]
              ).astype(np.float32)
    a = (ref - ref.mean()) * window
    b = (mov - mov.mean()) * window

    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = np.conj(fa) * fb
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross),
                      where=magnitude > 1.0e-12)
    surface = np.fft.irfft2(cross, s=(height, width))

    dys = _bounded_shift_candidates(height, max_shift)
    dxs = _bounded_shift_candidates(width, max_shift)
    if dys.size == 0 or dxs.size == 0:
        return 0, 0
    window_surface = surface[np.ix_(dys % height, dxs % width)]
    row, col = np.unravel_index(int(np.argmax(window_surface)),
                                window_surface.shape)
    return int(dys[row]), int(dxs[col])


def _aligned_slices(size, shift):
    """Mutually valid slices for (reference, moved) after an integer shift."""
    if shift >= 0:
        return slice(0, size - shift), slice(shift, size)
    return slice(-shift, size), slice(0, size + shift)


def _sobel_magnitude(crops):
    """Sobel gradient magnitude for a stack of 2D float crops."""
    tensor = torch.from_numpy(np.ascontiguousarray(
        np.asarray(crops, dtype=np.float32)))[:, None]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel = torch.stack((kernel_x, kernel_x.t()))[:, None]
    gradients = F.conv2d(tensor, kernel)
    return gradients.square().sum(dim=1).sqrt().numpy()


def edge_instability_pair(reference, moved, max_shift=MAX_BACKGROUND_SHIFT,
                          align=True):
    """Alignment-invariant edge instability between two 2D grayscale crops.

    The pair is aligned by a bounded phase-correlation translation, the border
    invalidated by that shift is cropped from *both* crops, and the Sobel edge
    magnitudes of what is left are compared.  The mean absolute edge difference
    is normalized by the mean edge magnitude of the reference crop so the score
    is scale-comparable across ROIs of different contrast and size.
    """
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moved, dtype=np.float32)
    if ref.ndim != 2 or ref.shape != mov.shape:
        raise ValueError("edge instability needs two matching 2D crops")

    shift_y, shift_x = (
        phase_correlation_shift(ref, mov, max_shift=max_shift) if align
        else (0, 0))
    ref_rows, mov_rows = _aligned_slices(ref.shape[0], shift_y)
    ref_cols, mov_cols = _aligned_slices(ref.shape[1], shift_x)
    ref_crop = ref[ref_rows, ref_cols]
    mov_crop = mov[mov_rows, mov_cols]
    if min(ref_crop.shape) < 3:
        raise ValueError(
            "aligned background crop is smaller than the 3x3 Sobel window")

    edges = _sobel_magnitude(np.stack((ref_crop, mov_crop)))
    difference = float(np.mean(np.abs(edges[0] - edges[1]), dtype=np.float64))
    scale = float(np.mean(edges[0], dtype=np.float64))
    return difference / (scale + _EDGE_EPSILON), (shift_y, shift_x)


def background_instability(video, roi, max_shift=MAX_BACKGROUND_SHIFT,
                           align=True, min_pixels=MIN_ROI_PIXELS):
    """Background edge instability of one ROI over a decoded clip.

    `video` is a [frames,height,width,3] uint8 array; `roi` a `BackgroundRoi`
    in normalized coordinates.  Returns the mean and 95th percentile of the
    per-frame-pair instability.
    """
    gray = _to_gray(video)
    frames, height, width = gray.shape
    x0, y0, x1, y1 = resolve_background_roi(
        roi, height, width, min_pixels=min_pixels)
    crops = gray[:, y0:y1, x0:x1]

    scores = []
    shifts = []
    for index in range(frames - 1):
        score, shift = edge_instability_pair(
            crops[index], crops[index + 1], max_shift=max_shift, align=align)
        scores.append(score)
        shifts.append(list(shift))
    values = np.asarray(scores, dtype=np.float64)
    return {
        "name": roi.name,
        "normalized": [roi.x0, roi.y0, roi.x1, roi.y1],
        "pixels": [x0, y0, x1, y1],
        "pairs": len(scores),
        "aligned": bool(align),
        "instability_mean": float(values.mean()) if values.size else 0.0,
        "instability_p95": (
            float(np.percentile(values, 95.0)) if values.size else 0.0),
        "per_pair": [float(value) for value in values],
        "shifts": shifts,
    }


def background_metrics(reference, candidate, rois,
                       max_shift=MAX_BACKGROUND_SHIFT):
    """Per-ROI reference/candidate instability plus the acceptance ratio."""
    report = []
    for roi in rois:
        ref_stats = background_instability(reference, roi, max_shift=max_shift)
        cand_stats = background_instability(candidate, roi, max_shift=max_shift)
        ref_mean = ref_stats["instability_mean"]
        cand_mean = cand_stats["instability_mean"]
        if ref_mean > 0.0:
            ratio = cand_mean / ref_mean
        else:
            ratio = 1.0 if cand_mean == 0.0 else math.inf
        report.append({
            "name": roi.name,
            "normalized": [roi.x0, roi.y0, roi.x1, roi.y1],
            "pixels": ref_stats["pixels"],
            "pairs": ref_stats["pairs"],
            "instability_ratio": ratio,
            "reference": {
                "instability_mean": ref_mean,
                "instability_p95": ref_stats["instability_p95"],
            },
            "candidate": {
                "instability_mean": cand_mean,
                "instability_p95": cand_stats["instability_p95"],
            },
        })
    return report


def video_metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"video shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.astype(np.float32) / 255.0
    cand = candidate.astype(np.float32) / 255.0
    error = ref - cand
    mse = float(np.mean(error * error, dtype=np.float64))
    per_frame_ssim = _ssim_per_frame(reference, candidate)

    temporal_mse = 0.0
    temporal_psnr = math.inf
    if len(reference) > 1:
        ref_delta = np.diff(ref, axis=0)
        cand_delta = np.diff(cand, axis=0)
        delta_error = ref_delta - cand_delta
        temporal_mse = float(np.mean(delta_error * delta_error, dtype=np.float64))
        # Frame differences span [-1,1], hence a peak-to-peak error range of 2.
        temporal_psnr = _psnr(temporal_mse, peak=2.0)

    return {
        "shape": list(reference.shape),
        "decoded_bit_exact": bool(np.array_equal(reference, candidate)),
        "mse": mse,
        "psnr_db": _psnr(mse),
        "ssim_mean": float(per_frame_ssim.mean()),
        "ssim_min": float(per_frame_ssim.min()),
        "ssim_per_frame": [float(value) for value in per_frame_ssim],
        "temporal_delta_mse": temporal_mse,
        "temporal_delta_psnr_db": temporal_psnr,
        "max_abs_pixel_delta": int(np.abs(
            reference.astype(np.int16) - candidate.astype(np.int16)).max()),
    }


def audio_metrics(reference, candidate, sample_rate):
    if reference is None or candidate is None:
        if reference is candidate:
            return {"present": False, "decoded_bit_exact": True}
        raise ValueError("only one artifact contains audio")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"audio shape mismatch: {reference.shape} != {candidate.shape}")
    error = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(error * error)) if error.size else 0.0
    signal_power = float(np.mean(reference.astype(np.float64) ** 2)) if error.size else 0.0
    snr = math.inf if mse == 0.0 else 10.0 * math.log10(
        max(signal_power, 1.0e-30) / mse)
    correlations = []
    for ref_channel, cand_channel in zip(reference, candidate):
        if np.array_equal(ref_channel, cand_channel):
            correlations.append(1.0)
        elif np.std(ref_channel) == 0.0 or np.std(cand_channel) == 0.0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(ref_channel, cand_channel)[0, 1]))
    return {
        "present": True,
        "sample_rate": int(sample_rate),
        "shape": list(reference.shape),
        "decoded_bit_exact": bool(np.array_equal(reference, candidate)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "snr_db": snr,
        "max_abs_delta": float(np.max(np.abs(error))) if error.size else 0.0,
        "channel_correlation": correlations,
    }


def compare(reference_path, candidate_path, background_rois=None):
    reference_path = Path(reference_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    reference_video = _decode_video(reference_path)
    candidate_video = _decode_video(candidate_path)
    reference_audio, reference_rate = _decode_audio(reference_path)
    candidate_audio, candidate_rate = _decode_audio(candidate_path)
    if reference_rate != candidate_rate:
        raise ValueError(
            f"audio sample-rate mismatch: {reference_rate} != {candidate_rate}")
    report = {
        "reference": {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
        },
        "container_bit_exact": _sha256(reference_path) == _sha256(candidate_path),
        "video": video_metrics(reference_video, candidate_video),
        "audio": audio_metrics(reference_audio, candidate_audio, reference_rate),
    }
    # Accept already-parsed ROIs, raw "[name:]x0,y0,x1,y1" strings, or a single
    # string; with no ROIs the report is unchanged.
    if isinstance(background_rois, str):
        background_rois = [background_rois]
    rois = [roi if isinstance(roi, BackgroundRoi)
            else parse_background_roi(roi, index)
            for index, roi in enumerate(background_rois or ())]
    if rois:
        report["background"] = background_metrics(
            reference_video, candidate_video, rois)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--background-roi",
        action="append",
        dest="background_rois",
        default=None,
        metavar="[NAME:]X0,Y0,X1,Y1",
        help="repeatable static-background ROI in normalized [0,1] coordinates, "
             "e.g. --background-roi wall:0.0,0.0,0.25,0.4 (name optional, "
             "defaults to roi0, roi1, ...)",
    )
    args = parser.parse_args()
    rois = parse_background_rois(args.background_rois)
    report = compare(args.reference, args.candidate, background_rois=rois)
    rendered = json.dumps(report, indent=2, allow_nan=True)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
