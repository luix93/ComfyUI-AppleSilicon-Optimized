"""Tests for matched H3 video/audio artifact metrics."""

import math

import numpy as np
import pytest

from dev import compare_av


def _frames():
    y, x = np.mgrid[:16, :18]
    frame = np.stack((x * 7, y * 9, (x + y) * 5), axis=-1).astype(np.uint8)
    return np.stack((frame, np.roll(frame, 1, axis=1)))


def _rigid_world(seed=0, height=160, width=176):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3)).astype(np.uint8)


def _translating_clip(frames=6, height=96, width=112, step=(3, 2), noise=1):
    """A rigid textured world seen through a panning camera."""
    world = _rigid_world()
    rng = np.random.default_rng(1234)
    offsets = [(20 + step[0] * i, 24 + step[1] * i) for i in range(frames)]
    clip = np.stack([world[oy:oy + height, ox:ox + width]
                     for oy, ox in offsets]).astype(np.int16)
    if noise:
        clip += rng.integers(-noise, noise + 1, size=clip.shape)
    return np.clip(clip, 0, 255).astype(np.uint8)


def _changing_geometry_clip(frames=6, height=96, width=112):
    """The same static texture, but with edges appearing and geometry growing."""
    world = _rigid_world()
    clip = np.tile(world[20:20 + height, 24:24 + width][None], (frames, 1, 1, 1))
    clip = clip.copy()
    for index in range(frames):
        side = 5 + 8 * index
        clip[index, 30:30 + side, 30:30 + side] = 255
        if index % 2 == 0:
            clip[index, 5:25, 70:100] = 0
    return clip


_BACKGROUND_ROI = compare_av.BackgroundRoi("bg", 0.1, 0.1, 0.9, 0.9)


def test_identical_video_is_exact_with_unit_ssim_and_infinite_psnr():
    frames = _frames()
    metrics = compare_av.video_metrics(frames, frames.copy())

    assert metrics["decoded_bit_exact"]
    assert metrics["ssim_mean"] == 1.0
    assert metrics["ssim_min"] == 1.0
    assert math.isinf(metrics["psnr_db"])
    assert math.isinf(metrics["temporal_delta_psnr_db"])
    assert metrics["max_abs_pixel_delta"] == 0


def test_changed_motion_is_visible_in_spatial_and_temporal_metrics():
    reference = _frames()
    candidate = reference.copy()
    candidate[1] = np.roll(candidate[1], 2, axis=0)
    metrics = compare_av.video_metrics(reference, candidate)

    assert not metrics["decoded_bit_exact"]
    assert metrics["ssim_mean"] < 1.0
    assert metrics["ssim_min"] < metrics["ssim_mean"]
    assert metrics["temporal_delta_mse"] > 0.0
    assert metrics["max_abs_pixel_delta"] > 0


def test_audio_metrics_report_exactness_snr_and_channel_correlation():
    reference = np.stack((
        np.linspace(-1.0, 1.0, 128, dtype=np.float32),
        np.linspace(1.0, -1.0, 128, dtype=np.float32),
    ))
    exact = compare_av.audio_metrics(reference, reference.copy(), 48000)
    assert exact["decoded_bit_exact"]
    assert math.isinf(exact["snr_db"])
    assert exact["channel_correlation"] == [1.0, 1.0]

    candidate = reference.copy()
    candidate[:, 64:] *= 0.9
    changed = compare_av.audio_metrics(reference, candidate, 48000)
    assert not changed["decoded_bit_exact"]
    assert changed["rmse"] > 0.0
    assert changed["snr_db"] > 0.0
    assert all(value < 1.0 for value in changed["channel_correlation"])


def test_background_roi_parser_names_by_index_and_honours_explicit_names():
    default, named = compare_av.parse_background_rois(
        ("0.0,0.0,0.5,0.5", " wall : 0.25,0.5,1.0,1.0 "))

    assert default == compare_av.BackgroundRoi("roi0", 0.0, 0.0, 0.5, 0.5)
    assert named == compare_av.BackgroundRoi("wall", 0.25, 0.5, 1.0, 1.0)
    assert compare_av.parse_background_roi("0,0,1,1", index=7).name == "roi7"
    assert compare_av.parse_background_rois(None) == []


@pytest.mark.parametrize("spec", [
    "0.2,0.2,0.2,0.6",       # empty on x
    "0.2,0.5,0.6,0.5",       # empty on y
    "0.8,0.2,0.3,0.6",       # inverted on x
    "0.2,0.9,0.6,0.4",       # inverted on y
    "-0.1,0.0,0.5,0.5",      # below the normalized range
    "0.0,0.0,1.5,0.5",       # above the normalized range
    "0.1,0.2,0.3",           # too few fields
    "0.1,0.2,0.3,0.4,0.5",   # too many fields
    "0.1,left,0.3,0.4",      # non-numeric field
    "0.1,nan,0.3,0.4",       # non-finite field
    "wall:",                 # no coordinates at all
    ":0.0,0.0,1.0,1.0",      # empty explicit name
])
def test_malformed_background_rois_are_rejected_with_a_named_error(spec):
    with pytest.raises(ValueError) as failure:
        compare_av.parse_background_roi(spec, index=3)

    message = str(failure.value)
    assert "background ROI" in message
    assert ("'wall'" in message) if spec.startswith("wall") else ("roi3" in message)


def test_background_roi_below_sixteen_pixels_is_rejected_not_clamped():
    tiny = compare_av.BackgroundRoi("corner", 0.0, 0.0, 0.05, 0.05)

    with pytest.raises(ValueError) as failure:
        compare_av.resolve_background_roi(tiny, height=96, width=112)
    assert "corner" in str(failure.value)
    assert "16x16" in str(failure.value)

    with pytest.raises(ValueError, match="corner"):
        compare_av.background_instability(_translating_clip(), tiny)

    # The same normalized ROI is fine once the frame is large enough.
    assert compare_av.resolve_background_roi(tiny, height=640, width=640) == (
        0, 0, 32, 32)


def test_pure_camera_translation_is_stable_once_aligned():
    clip = _translating_clip()

    aligned = compare_av.background_instability(clip, _BACKGROUND_ROI)
    unaligned = compare_av.background_instability(
        clip, _BACKGROUND_ROI, align=False)

    assert aligned["pairs"] == len(clip) - 1
    # The bounded phase correlation recovers the exact per-frame camera pan.
    assert all(shift == [-3, -2] for shift in aligned["shifts"])
    assert aligned["instability_mean"] < 0.05
    assert aligned["instability_p95"] < 0.10
    # Alignment is what makes it stable: without it the same clip looks broken.
    assert unaligned["instability_mean"] > 10.0 * aligned["instability_mean"]
    assert unaligned["instability_mean"] > 0.3


def test_changing_geometry_scores_far_more_unstable_than_a_camera_pan():
    translation = compare_av.background_instability(
        _translating_clip(), _BACKGROUND_ROI)
    geometry = compare_av.background_instability(
        _changing_geometry_clip(), _BACKGROUND_ROI)

    assert geometry["instability_mean"] > 5.0 * translation["instability_mean"]
    assert geometry["instability_p95"] > translation["instability_p95"]
    assert geometry["instability_mean"] > 0.1


def test_phase_correlation_search_is_bounded_to_sixteen_pixels():
    world = _rigid_world()
    reference = compare_av._to_gray(world[None, 20:116, 24:136])[0]
    moved = compare_av._to_gray(world[None, 22:118, 29:141])[0]

    assert compare_av.phase_correlation_shift(reference, moved) == (-2, -5)

    # A 24px displacement is outside the trusted range; the search never even
    # scores it, so it cannot be accepted and then clipped to the boundary.
    far = compare_av._to_gray(world[None, 44:140, 24:136])[0]
    shift_y, shift_x = compare_av.phase_correlation_shift(reference, far)
    assert abs(shift_y) <= 16 and abs(shift_x) <= 16
    assert (shift_y, shift_x) != (-24, 0)


def test_background_metrics_report_per_roi_instability_ratio():
    reference = _translating_clip()
    rois = compare_av.parse_background_rois(
        ("left:0.05,0.05,0.45,0.95", "0.55,0.05,0.95,0.95"))

    exact = compare_av.background_metrics(reference, reference.copy(), rois)
    assert [entry["name"] for entry in exact] == ["left", "roi1"]
    assert all(entry["instability_ratio"] == 1.0 for entry in exact)
    assert all(entry["reference"]["instability_mean"]
               == entry["candidate"]["instability_mean"] for entry in exact)

    degraded = compare_av.background_metrics(
        reference, _changing_geometry_clip(), rois)
    assert any(entry["instability_ratio"] > 1.10 for entry in degraded)


def test_perfectly_static_reference_does_not_divide_by_zero():
    static = np.tile(_rigid_world()[None, :96, :112], (4, 1, 1, 1))
    rois = [_BACKGROUND_ROI]

    assert compare_av.background_instability(
        static, _BACKGROUND_ROI)["instability_mean"] == 0.0
    assert compare_av.background_metrics(
        static, static.copy(), rois)[0]["instability_ratio"] == 1.0
    assert math.isinf(compare_av.background_metrics(
        static, _changing_geometry_clip(4), rois)[0]["instability_ratio"])
