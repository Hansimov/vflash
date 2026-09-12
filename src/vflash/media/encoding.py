"""Bounded-memory, atomic H.264/AAC delivery from decoded H3 tensors."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from vflash.media.audio_delivery import (
    AUDIO_DELIVERY_PROFILES,
    WEB_AUDIO_PROFILE,
    web_loudness_filter,
)
from vflash.native.errors import VflashNativeError


class MediaError(VflashNativeError):
    """Decoded tensors or the local media encoder violated the output contract."""


def media_executables() -> tuple[str, str]:
    """Check the local binaries before acquiring a model or CUDA context."""
    paths = tuple(shutil.which(name) for name in ("ffmpeg", "ffprobe"))
    if any(path is None for path in paths):
        raise MediaError("install ffmpeg and ffprobe before generating video")
    return str(paths[0]), str(paths[1])


def probe_mp4(path: Path, *, ffprobe: str | None = None) -> dict[str, Any]:
    """Read only local media metadata; this adapter has no URL or network input."""
    if not path.is_file():
        raise MediaError("the encoded MP4 is missing")
    executable = ffprobe or media_executables()[1]
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path.resolve(strict=True)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaError("ffprobe could not inspect the encoded MP4") from exc
    if result.returncode:
        raise MediaError(f"ffprobe failed: {result.stderr[-2000:].strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("streams"), list):
        raise MediaError("ffprobe returned no media streams")
    # ffprobe's filename and tags are neither necessary nor a public identity.
    return {
        "streams": value["streams"],
        "duration_seconds": float(value.get("format", {}).get("duration", 0)),
        "size_bytes": path.stat().st_size,
    }


def encode_mp4(
    video: Any,
    audio: Any,
    output_path: Path,
    *,
    fps: int = 24,
    audio_sample_rate: int = 32000,
    preset: str = "medium",
    crf: int = 18,
    timeout_seconds: float = 600,
    audio_delivery_profile: str = "unchanged",
) -> dict[str, Any]:
    """Encode CPU ``[1,3,F,H,W]`` RGB and ``[1,2,S]`` stereo without overwriting.

    RGB floats are quantized eight frames at a time. Temporary RGB, PCM and MP4
    files share the output filesystem and are removed on success or failure.
    A completed and probed MP4 is published with an atomic, no-clobber link.
    """
    import torch

    if getattr(video, "ndim", None) != 5 or tuple(video.shape[:2]) != (1, 3):
        raise MediaError("video must have shape [1,3,frames,height,width]")
    if getattr(audio, "ndim", None) != 3 or tuple(audio.shape[:2]) != (1, 2):
        raise MediaError("audio must have shape [1,2,samples]")
    if video.device.type != "cpu" or audio.device.type != "cpu":
        raise MediaError("the media encoder requires owned CPU tensors")
    frames, height, width = (int(value) for value in video.shape[2:])
    samples = int(audio.shape[2])
    if min(frames, height, width, samples) <= 0 or height % 2 or width % 2:
        raise MediaError("H.264 delivery requires nonempty media and even dimensions")
    if type(fps) is not int or not 1 <= fps <= 60:
        raise MediaError("fps must be an integer between 1 and 60")
    if type(audio_sample_rate) is not int or audio_sample_rate <= 0:
        raise MediaError("audio_sample_rate must be a positive integer")
    if type(crf) is not int or not 0 <= crf <= 51:
        raise MediaError("crf must be an integer between 0 and 51")
    if audio_delivery_profile not in AUDIO_DELIVERY_PROFILES:
        raise MediaError("unknown audio delivery profile")
    if preset not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }:
        raise MediaError("unknown H.264 preset")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise MediaError("the encoding deadline must be positive and finite")
    if abs(samples / audio_sample_rate - frames / fps) > 1 / audio_sample_rate:
        raise MediaError("audio and video must share the same delivery duration")
    ffmpeg, ffprobe = media_executables()
    # Resolve the parent, but not a pre-existing symlink at the destination.
    output_path = output_path.parent.resolve() / output_path.name
    if output_path.exists() or output_path.is_symlink():
        raise MediaError("the output path already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".vflash-media-") as tmp:
        directory = Path(tmp)
        video_path, audio_path = directory / "video.rgb24", directory / "audio.wav"
        encoded_path = directory / "output.mp4"
        with video_path.open("wb") as destination:
            for start in range(0, frames, 8):
                block = video[0, :, start : start + 8].permute(1, 2, 3, 0).float()
                if not torch.isfinite(block).all().item():
                    raise MediaError("decoded video contains nonfinite values")
                rgb = block.mul(255).round().clamp(0, 255).to(torch.uint8).contiguous()
                rgb.numpy().tofile(destination)
        if not torch.isfinite(audio).all().item():
            raise MediaError("decoded audio contains nonfinite values")
        audio_filters: list[str] = []
        audio_delivery = None
        if audio_delivery_profile == WEB_AUDIO_PROFILE:
            audio_path = directory / "audio.f32le"
            pcm = audio[0].T.float().contiguous()
            pcm.numpy().astype("<f4", copy=False).tofile(audio_path)
            audio_input = [
                "-f",
                "f32le",
                "-ar",
                str(audio_sample_rate),
                "-ac",
                "2",
                "-i",
                str(audio_path),
            ]
            normalization, audio_delivery = web_loudness_filter(ffmpeg, audio_input)
            if normalization:
                audio_filters.append(normalization)
        else:
            pcm = (
                audio[0]
                .T.float()
                .clamp(-1, 1)
                .mul(32767)
                .round()
                .to(torch.int16)
                .contiguous()
            )
            with wave.open(str(audio_path), "wb") as destination:
                destination.setnchannels(2)
                destination.setsampwidth(2)
                destination.setframerate(audio_sample_rate)
                destination.writeframes(pcm.numpy().tobytes())
            audio_input = ["-i", str(audio_path)]
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            str(video_path),
            *audio_input,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            *(["-af", ",".join(audio_filters)] if audio_filters else []),
            *(
                ["-ar", str(audio_sample_rate)]
                if audio_delivery_profile == WEB_AUDIO_PROFILE
                else []
            ),
            "-movflags",
            "+faststart",
            str(encoded_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaError("ffmpeg did not complete the video") from exc
        if completed.returncode or not encoded_path.is_file():
            raise MediaError(f"ffmpeg failed: {completed.stderr[-2000:].strip()}")
        probe = probe_mp4(encoded_path, ffprobe=ffprobe)
        streams = probe["streams"]
        visual = [row for row in streams if row.get("codec_type") == "video"]
        sound = [row for row in streams if row.get("codec_type") == "audio"]
        if (
            len(visual) != 1
            or len(sound) != 1
            or visual[0].get("width") != width
            or visual[0].get("height") != height
            or int(visual[0].get("nb_frames", 0)) != frames
            or int(sound[0].get("sample_rate", 0)) != audio_sample_rate
            or sound[0].get("channels") != 2
        ):
            raise MediaError("the encoded streams differ from the requested delivery")
        try:
            os.link(encoded_path, output_path)
        except FileExistsError as exc:
            raise MediaError("the output path already exists") from exc
    result = {
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "audio_sample_rate": audio_sample_rate,
        "audio_samples": samples,
        "duration_seconds": frames / fps,
        "preset": preset,
        "crf": crf,
        "size_bytes": output_path.stat().st_size,
        "probe": probe,
    }
    if audio_delivery is not None:
        result["audio_delivery"] = audio_delivery
    return result
