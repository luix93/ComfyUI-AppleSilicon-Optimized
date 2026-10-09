"""Optional VSR 2x image/video nodes using the native MLX-DLSS executable."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

import numpy as np
from PIL import Image
import psutil
from safetensors import safe_open
import torch


MODEL_ID = "rtx-vsr-1.8.2-highbitrate-low-2x"
MODEL_FOLDER = "applesilicon_super_resolution"
NR_FOLDER = "applesilicon_neural_rendering"
GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]


def _folders():
    import folder_paths
    # Keep the local folder even when Desktop has registered shared paths first.
    # ComfyUI deduplicates this addition and preserves configured path priority.
    folder_paths.add_model_folder_path(
        MODEL_FOLDER, str(Path(folder_paths.models_dir) / MODEL_FOLDER))
    return folder_paths


def _models():
    return [name for name in _folders().get_filename_list(MODEL_FOLDER)
            if name.endswith(".safetensors")]


def _model_path(name):
    folders = _folders()
    if name not in _models():
        raise ValueError(f"Choose a VSR model from models/{MODEL_FOLDER}.")
    path = folders.get_full_path_or_raise(MODEL_FOLDER, name)
    with safe_open(path, framework="numpy") as weights:
        if (weights.metadata() or {}).get("model") != MODEL_ID:
            raise ValueError("This node needs the extracted VSR 1.8.2 High Bitrate Low 2x model.")
    return path


def _neural_folders():
    folders = _folders()
    # Desktop can discover the new category only on its next launch. Reuse the
    # configured VSR model roots as well, keeping shared models discoverable now.
    for parent in [Path(p).parent for p in folders.get_folder_paths(MODEL_FOLDER)]:
        folders.add_model_folder_path(NR_FOLDER, str(parent / NR_FOLDER))
    return folders


def _neural_models():
    names = _neural_folders().get_filename_list(NR_FOLDER)
    return ["disabled"] + sorted({str(Path(name).parent) for name in names
                                  if name.endswith(".dlssmodel/manifest.json")})


def _neural_model_path(name):
    if name == "disabled" or name not in _neural_models():
        raise ValueError(f"Choose a neural-rendering model from models/{NR_FOLDER}.")
    manifest_path = Path(_neural_folders().get_full_path_or_raise(NR_FOLDER, name + "/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("architecture") != "mlxdlss.neural-rendering-transformer.v1"
            or manifest.get("weights", {}).get("file") != "weights.safetensors"
            or not (manifest_path.parent / "weights.safetensors").is_file()):
        raise ValueError("Invalid neural-rendering model package; run dev/setup_neural_rendering.py.")
    return str(manifest_path.parent)


def _neural_inputs(video=False):
    result = {
        "neural_model": (_neural_models(), {"tooltip": "Optional DLSS neural rendering AFTER VSR. Disabled preserves VSR-only output."}),
        "nr_profile": (["natural", "standard", "cinematic"],),
        "nr_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "Zero bypasses neural rendering completely."}),
        "nr_detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
        "nr_colour": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
        "nr_processing_scale": ([1, 2], {"tooltip": "Internal NR scale. 2 uses four times the pixels; output stays at VSR's 2x dimensions."}),
    }
    if video:
        result["nr_temporal"] = ("BOOLEAN", {"default": True,
            "tooltip": "Use estimated motion and previous-frame history. Resets at scene cuts."})
    return result


def _neural_arguments(model, profile, intensity, detail, colour, scale, temporal=None):
    if not math.isfinite(intensity) or not 0 <= intensity <= 1:
        raise ValueError("NR intensity must be between 0 and 1.")
    if model == "disabled" or intensity == 0:
        return []
    if (profile not in {"natural", "standard", "cinematic"} or scale not in {1, 2}
            or not math.isfinite(detail) or not 0 <= detail <= 4
            or not math.isfinite(colour) or not 0 <= colour <= 2):
        raise ValueError("Invalid neural-rendering controls.")
    result = ["--model", _neural_model_path(model), "--vsr-first", "on",
              "--profile", profile, "--intensity", str(intensity),
              "--detail-strength", str(detail), "--colour-strength", str(colour),
              "--processing-scale", str(scale)]
    if temporal is not None:
        result += ["--temporal", "on" if temporal else "off"]
    return result


def _executable():
    path = Path(os.environ.get("ASFP8_SR_EXECUTABLE", ROOT / "_runtime/mlx-dlss/mlxdlss"))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("VSR runtime is missing. Run dev/setup_super_resolution.py; see docs/super-resolution.md.")
    return str(path)


def _memory_budget(width, height, frames, memory_limit_gb, image_output, nr_scale=None):
    if min(width, height, frames) < 1 or max(width, height) > 8192:
        raise ValueError("VSR needs nonempty images with input dimensions no larger than 8192 pixels.")
    if not math.isfinite(memory_limit_gb) or not 1 <= memory_limit_gb <= 32:
        raise ValueError("Memory limit must be between 1 and 32 GiB.")
    budget = int(memory_limit_gb * GIB)
    padded_pixels = ((width + 31) // 32 * 32) * ((height + 31) // 32 * 32)
    working = padded_pixels * 1280 + 256 * 1024 ** 2
    if nr_scale is not None:
        # NR runs at the UPSCALED size, optionally supersampled again. Include
        # model, history and workspaces; the worker also enforces a hard MLX limit.
        nr_width = max(320, (width * 2 * nr_scale + 63) // 64 * 64)
        nr_height = max(320, (height * 2 * nr_scale + 63) // 64 * 64)
        working = max(working, nr_width * nr_height * 1536 + GIB)
    output = frames * height * width * 4 * 3 * 4 if image_output else 0
    if working + output > budget:
        advice = " Use the Video node to stream long sequences." if image_output and frames > 1 else " Reduce the input size or raise the memory limit."
        if nr_scale is not None:
            advice += " NR processing scale 2 greatly increases memory use; try scale 1."
        raise ValueError(f"Upscale estimates {(working + output) / GIB:.1f} GiB, above the {memory_limit_gb:g} GiB limit.{advice}")
    available = psutil.virtual_memory().available
    if working + output > available * 0.7:
        raise MemoryError("Not enough free memory for this upscale. Unload other models first.")
    return budget - output


def _run(arguments, memory_bytes, progress=None):
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()
    environment = dict(os.environ, ASFP8_SR_MEMORY_BYTES=str(memory_bytes))
    with tempfile.TemporaryFile(mode="w+b") as errors:
        process = subprocess.Popen([_executable(), *arguments], stdout=subprocess.PIPE,
                                   stderr=errors, env=environment)
        resident = psutil.Process(process.pid)
        peak = 0
        try:
            while True:
                mm.throw_exception_if_processing_interrupted()
                try:
                    rss = resident.memory_info().rss
                    peak = max(peak, rss)
                    if rss > memory_bytes:
                        raise MemoryError("VSR exceeded its memory limit; the worker was stopped.")
                except psutil.NoSuchProcess:
                    pass
                try:
                    output, _ = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if progress is not None:
                        size = os.fstat(errors.fileno()).st_size
                        lines = os.pread(errors.fileno(), 65536, max(0, size - 65536)).decode("utf-8", errors="replace")
                        matches = re.findall(r"(\d+)/(\d+) input frames", lines)
                        if matches:
                            done, total = map(int, matches[-1])
                            progress.update_absolute(done, max(done, total, 1))
            size = os.fstat(errors.fileno()).st_size
            details = os.pread(errors.fileno(), 65536, max(0, size - 65536)).decode("utf-8", errors="replace")
            if process.returncode:
                raise RuntimeError(f"VSR worker failed (exit {process.returncode}): {details[-4000:]}")
            result = json.loads(output)
            result["worker_peak_rss_bytes"] = peak
            metal = re.search(r"ASFP8_SR_METAL_PEAK_BYTES=(\d+)", details)
            result["metal_peak_bytes"] = int(metal[1]) if metal else None
            return result
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()


def _limits():
    return ("FLOAT", {"default": 8.0, "min": 1.0, "max": 32.0, "step": 1.0,
                      "tooltip": "GiB budget for this upscale. Includes the returned image batch; video streams frames to disk."})


def _report(timing, label):
    metal = timing.get("metal_peak_bytes")
    memory = f", {metal / GIB:.2f} GiB peak active MLX memory" if metal is not None else ""
    print(f"[AppleSilicon-Optimized/VSR] {label}: {timing['elapsedSeconds']:.3f}s native{memory}")


class AppleSiliconSuperResolution:
    CATEGORY = "Apple Silicon/Upscaling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    DESCRIPTION = "VSR 2x, optionally followed by DLSS neural rendering for detail and appearance. Final size stays 2x. Enhancement can change content. SDR, independent frames; use Video for long clips."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "model_name": (_models(),),
                             "memory_limit_gb": _limits()}, "optional": _neural_inputs()}

    def upscale(self, image, model_name, memory_limit_gb=8.0, neural_model="disabled",
                nr_profile="natural", nr_intensity=1.0, nr_detail=1.0,
                nr_colour=1.0, nr_processing_scale=1):
        import comfy.utils
        import comfy.model_management as mm
        if image.ndim != 4 or image.shape[-1] != 3 or not image.is_floating_point():
            raise ValueError("VSR expects floating-point RGB images [frames, height, width, 3].")
        frames, height, width, _ = image.shape
        weights = _model_path(model_name)
        _executable()
        neural = _neural_arguments(neural_model, nr_profile, nr_intensity, nr_detail,
                                   nr_colour, nr_processing_scale)
        budget = _memory_budget(width, height, frames, memory_limit_gb, True,
                                nr_processing_scale if neural else None)
        result = torch.empty((frames, height * 2, width * 2, 3), dtype=torch.float32, device="cpu")
        progress = comfy.utils.ProgressBar(frames)
        with tempfile.TemporaryDirectory(prefix="as-vsr-", dir=_folders().get_temp_directory()) as directory:
            source, destination = Path(directory) / "input.png", Path(directory) / "output.png"
            for index, frame in enumerate(image):
                mm.throw_exception_if_processing_interrupted()
                pixels = frame.detach().to(device="cpu", dtype=torch.float32).numpy()
                if not np.isfinite(pixels).all():
                    raise ValueError("VSR input contains NaN or infinite values.")
                pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(pixels).save(source)
                timing = _run(["process-image", str(source), "--output", str(destination),
                               "--vsr-weights", weights, *neural], budget)
                with Image.open(destination) as decoded:
                    if decoded.size != (width * 2, height * 2):
                        raise RuntimeError("VSR returned an unexpected image size.")
                    result[index].copy_(torch.from_numpy(np.array(decoded.convert("RGB"))).float().div_(255))
                destination.unlink()
                progress.update(1)
                _report(timing, f"Frame {index + 1}/{frames}")
        return (result,)


class AppleSiliconSuperResolutionVideo:
    CATEGORY = "Apple Silicon/Upscaling"
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "upscale"
    DESCRIPTION = "Streamed VSR 2x, optionally followed by DLSS neural rendering with temporal history. Final size stays 2x. Preserves audio and timing; SDR. Enhancement can change content."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "model_name": (_models(),),
                             "codec": (["hevc", "h264", "prores"],),
                             "memory_limit_gb": _limits()}, "optional": _neural_inputs(video=True)}

    def upscale(self, video, model_name, codec="hevc", memory_limit_gb=8.0,
                neural_model="disabled", nr_profile="natural", nr_intensity=1.0,
                nr_detail=1.0, nr_colour=1.0, nr_processing_scale=1, nr_temporal=True):
        import comfy.utils
        import comfy.model_management as mm
        from comfy_api.latest import InputImpl
        from comfy_api.latest._util import VideoContainer, VideoCodec
        if codec not in {"hevc", "h264", "prores"}:
            raise ValueError("Choose HEVC, H.264 or ProRes.")
        if video.get_color_space() in {"HDR", "HDR PQ"}:
            raise ValueError("VSR supports SDR only. Convert HDR to SDR before upscaling.")
        weights = _model_path(model_name)
        _executable()
        width, height = video.get_dimensions()
        neural = _neural_arguments(neural_model, nr_profile, nr_intensity, nr_detail,
                                   nr_colour, nr_processing_scale, nr_temporal)
        budget = _memory_budget(width, height, 1, memory_limit_gb, False,
                                nr_processing_scale if neural else None)
        extension = "mov" if codec == "prores" else "mp4"
        temporary = Path(_folders().get_temp_directory())
        destination = temporary / f"applesilicon-vsr-{uuid.uuid4().hex}.{extension}"
        try:
            with tempfile.TemporaryDirectory(prefix="as-vsr-video-", dir=temporary) as directory:
                source = Path(directory) / "input.mp4"
                staged = Path(directory) / f"output.{extension}"
                # ComfyUI applies trims/crops here and streams file-backed input.
                # Unedited compatible MP4 streams are remuxed without re-encoding.
                video.save_to(str(source), format=VideoContainer.MP4, codec=VideoCodec.AUTO)
                mm.throw_exception_if_processing_interrupted()
                progress = comfy.utils.ProgressBar(1)
                timing = _run(["process-video", str(source), "--output", str(staged),
                               "--vsr-weights", weights, "--codec", codec,
                               "--audio", "on", *neural], budget, progress)
                progress.update_absolute(timing["inputFrames"], timing["inputFrames"])
                staged.replace(destination)
            _report(timing, f"{timing['inputFrames']} frames")
            return (InputImpl.VideoFromFile(str(destination)),)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


NODE_CLASS_MAPPINGS = {
    "AppleSiliconSuperResolution": AppleSiliconSuperResolution,
    "AppleSiliconSuperResolutionVideo": AppleSiliconSuperResolutionVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconSuperResolution": "Apple Silicon Super Resolution · Image 2x",
    "AppleSiliconSuperResolutionVideo": "Apple Silicon Super Resolution · Video 2x",
}
