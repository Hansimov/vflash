"""Atomic, opt-in delivery of restored RGB while copying original audio packets."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vflash.media.encoding import MediaError, media_executables, probe_mp4


def encode_restored_video(
    frames: Sequence[Any], source_video: Path, output_path: Path
) -> dict[str, Any]:
    """Encode a new 24 fps MP4 without overwriting or re-encoding source audio.

    Frames must cover the complete original clock, even for local repairs. Source
    and restored results remain separate. H.264 output is lossy; this function
    does not claim unchanged RGB pixels or repair quality in the delivered MP4.
    """
    from PIL import Image

    if not 1 <= len(frames) <= 240:
        raise MediaError("restoration delivery requires one to 240 frames")
    if any(not isinstance(frame, Image.Image) or frame.mode != "RGB" for frame in frames):
        raise MediaError("restoration delivery requires RGB images")
    width, height = frames[0].size
    if (
        width < 32
        or height < 32
        or width % 32
        or height % 32
        or width * height > 1024**2
        or any(frame.size != (width, height) for frame in frames)
    ):
        raise MediaError("restoration frames require one 32-aligned canvas up to one mebipixel")
    output_path = output_path.parent.resolve() / output_path.name
    if output_path.exists() or output_path.is_symlink():
        raise MediaError("the output path already exists")
    source_video = source_video.resolve(strict=True)
    ffmpeg, ffprobe = media_executables()
    source = probe_mp4(source_video, ffprobe=ffprobe)
    visual = [row for row in source["streams"] if row.get("codec_type") == "video"]
    sound = [row for row in source["streams"] if row.get("codec_type") == "audio"]
    if (
        len(visual) != 1
        or len(sound) > 1
        or visual[0].get("width") != width
        or visual[0].get("height") != height
        or visual[0].get("avg_frame_rate") != "24/1"
        or visual[0].get("r_frame_rate") != "24/1"
        or int(visual[0].get("nb_frames", 0)) != len(frames)
    ):
        raise MediaError(
            "restoration must preserve the source canvas and complete 24 fps clock"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".vflash-restored-") as tmp:
        directory = Path(tmp)
        raw = directory / "frames.rgb24"
        encoded = directory / "output.mp4"
        with raw.open("wb") as destination:
            for frame in frames:
                destination.write(frame.tobytes())
        command = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            "24",
            "-i",
            str(raw),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-map_metadata",
            "-1",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(encoded),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise MediaError(
                "restoration encoding failed; the original video is unchanged"
            ) from exc
        probe = probe_mp4(encoded, ffprobe=ffprobe)
        videos = [row for row in probe["streams"] if row.get("codec_type") == "video"]
        audios = [row for row in probe["streams"] if row.get("codec_type") == "audio"]
        if (
            len(videos) != 1
            or len(audios) != len(sound)
            or int(videos[0].get("nb_frames", 0)) != len(frames)
            or videos[0].get("width") != width
            or videos[0].get("height") != height
            or videos[0].get("avg_frame_rate") != "24/1"
        ):
            raise MediaError("restored MP4 does not match the source clock or streams")
        if sound and any(
            audios[0].get(key) != sound[0].get(key)
            for key in ("codec_name", "sample_rate", "channels", "nb_frames")
        ):
            raise MediaError("restored MP4 changed the original audio stream")
        try:
            os.link(encoded, output_path)
        except FileExistsError as exc:
            raise MediaError("the output path already exists") from exc
    return {
        "frames": len(frames),
        "width": width,
        "height": height,
        "fps": 24,
        "duration_seconds": len(frames) / 24,
        "size_bytes": output_path.stat().st_size,
        "audio_delivery": "source-packet-copy" if sound else "no-source-audio",
        "source_retained": True,
        "visual_encoding": "h264-crf16-lossy",
    }
