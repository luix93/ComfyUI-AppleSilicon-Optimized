"""Bounded reference preparation; video is sampled before upload."""
import base64
import binascii
import io
import math
import uuid
import wave

from PIL import Image, ImageOps

MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_AUDIO_BYTES = 4 * 1024 * 1024
MAX_FRAMES = 8


def validate_reference(item):
    kind = item.get("kind")
    if kind not in ("image", "frame", "audio"):
        raise ValueError("Reference must be an image, sampled video frame or WAV audio clip.")
    data = item.get("data", "")
    limit = MAX_AUDIO_BYTES if kind == "audio" else MAX_IMAGE_BYTES
    if not isinstance(data, str) or len(data) > (limit * 4 // 3 + 8):
        raise ValueError("Reference is too large; use the reference uploader to prepare a smaller copy.")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as e:
        raise ValueError("Invalid reference data.") from e
    if kind == "audio":
        try:
            with wave.open(io.BytesIO(raw)) as audio:
                frames, rate = audio.getnframes(), audio.getframerate()
                if audio.getsampwidth() != 2 or audio.getnchannels() != 1 or rate != 24000 or frames > rate * 30:
                    raise ValueError("Audio references must be mono 24 kHz PCM16 WAV, up to 30 seconds.")
                if len(audio.readframes(frames)) != frames * 2:
                    raise ValueError("Incomplete audio reference.")
        except (wave.Error, EOFError) as e:
            raise ValueError("Invalid WAV audio reference.") from e
        item["mime"] = "audio/wav"
    else:
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if max(image.size) > 1024 or image.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError("Image references must be JPEG, PNG or WebP, up to 1024 pixels per side.")
                item["mime"] = Image.MIME[image.format]
                image.verify()
        except (OSError, Image.DecompressionBombError) as e:
            raise ValueError("Invalid image reference.") from e
    return item


def image_reference(image, name, kind="image", timestamp=None):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88)
    return {"id": uuid.uuid4().hex, "name": name, "kind": kind, "mime": "image/jpeg",
            "data": base64.b64encode(output.getvalue()).decode(), "timestamp": timestamp}


def _tensor_image(image, name, kind="image", timestamp=None):
    import torch
    import torch.nn.functional as F
    # Resize one selected frame before copying to CPU or converting to bytes.
    # A large IMAGE batch must never become a second full-resolution batch.
    image = image.detach().movedim(-1, 0).unsqueeze(0)
    h, w = image.shape[-2:]
    scale = min(1.0, 1024 / max(h, w))
    if scale < 1:
        image = F.interpolate(image.float(), size=(max(1, round(h * scale)), max(1, round(w * scale))),
                              mode="bilinear", align_corners=False, antialias=True)
    array = image[0].movedim(0, -1).clamp(0, 1).mul(255).to(dtype=torch.uint8).cpu().numpy()
    return image_reference(Image.fromarray(array), name, kind, timestamp)


def tensor_references(images=None, audio=None, frame_count=4, fps=24, batch_mode="auto"):
    import numpy as np
    refs = []
    if images is not None:
        count = len(images)
        if not count:
            raise ValueError("The connected image/frame batch is empty.")
        if batch_mode not in ("auto", "images", "video"):
            raise ValueError("Unknown connected image-batch mode.")
        is_video = batch_mode == "video" or (batch_mode == "auto" and count > MAX_FRAMES)
        if not is_video and count > MAX_FRAMES:
            raise ValueError("Connect at most 8 images, or select Video frames for a video batch.")
        indices = np.linspace(0, count - 1, min(count, frame_count)).round().astype(int) if is_video else range(count)
        for index in indices:
            refs.append(_tensor_image(images[int(index)], f"Input {'frame' if is_video else 'image'} {index + 1}",
                                      "frame" if is_video else "image", float(index / fps) if is_video else None))
    if audio is not None:
        waveform = audio["waveform"]
        rate = int(audio["sample_rate"])
        if rate <= 0:
            raise ValueError("Audio sample rate must be positive.")
        samples = waveform[0, :, :rate * 30].detach().float().cpu().numpy().mean(axis=0)
        if not len(samples):
            raise ValueError("Audio reference is empty.")
        positions = np.arange(round(len(samples) * 24000 / rate)) * rate / 24000
        samples = np.interp(positions, np.arange(len(samples)), samples)
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        refs.append({"id": uuid.uuid4().hex, "name": "Input audio · first 30s", "kind": "audio", "mime": "audio/wav", "data": base64.b64encode(out.getvalue()).decode()})
    return refs


def video_references(video, frame_count=4):
    """Sample native ComfyUI VIDEO inputs without materializing a whole file."""
    from comfy_api.latest._input_impl.video_types import VideoFromComponents, VideoFromList

    if isinstance(video, VideoFromComponents):
        parts = video.get_components()  # Existing tensors; no decoding or copying.
        return tensor_references(parts.images, frame_count=frame_count,
                                 fps=float(parts.frame_rate), batch_mode="video")

    duration = float(video.get_duration())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The connected video has no readable duration.")
    rate = float(video.get_frame_rate())
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("The connected video has no readable frame rate.")
    count = min(frame_count, max(1, round(duration * rate)))
    refs = []
    clips = video.videos if isinstance(video, VideoFromList) else [video]
    durations = [float(clip.get_duration()) for clip in clips]
    for index in range(count):
        timestamp = duration * (index + 0.5) / count
        offset = timestamp
        for clip, seconds in zip(clips, durations):
            if offset < seconds:
                break
            offset -= seconds
        clip_rate = float(clip.get_frame_rate())
        # Use the VIDEO API's trimmed view so existing trims, crops and display
        # rotation are honored. File-backed video decodes only this short window.
        start = max(0, math.floor(offset * clip_rate) / clip_rate)
        sample = clip.as_trimmed(start, 1 / clip_rate, strict_duration=False)
        if sample is None:
            raise ValueError("Cannot sample the connected video at the requested time.")
        parts = sample.get_components()
        if not len(parts.images):
            raise ValueError("The connected video has no frame at the requested time.")
        refs.append(_tensor_image(parts.images[0], f"Video frame {index + 1}", "frame", timestamp))
        del parts, sample
    return refs
