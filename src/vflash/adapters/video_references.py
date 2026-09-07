"""Bounded local video decoding, with no model imports or source-audio conditioning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from vflash.adapters.video_geometry import reference_video_geometry
from vflash.contracts import ContractError
from vflash.media.encoding import media_executables

MAX_VIDEO_REFERENCE_BYTES = 20 * 1024 * 1024
_DEMUXERS = "mov,matroska,webm"


@dataclass
class DecodedVideoReference:
    """Own source-sized uint8 THWC frames until capture completes or fails."""

    frames: Any
    metadata: dict[str, Any]

    def require_frames(self) -> Any:
        if self.frames is None:
            raise ContractError("the decoded video reference is closed")
        return self.frames

    def close(self) -> None:
        self.frames = None

    def __enter__(self) -> DecodedVideoReference:
        self.require_frames()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _run_media(command: list[str], work: Path, *, output: Path, timeout: int) -> None:
    # File-backed output keeps a corrupt input's diagnostics off the heap.
    # On timeout, interrupt or failure, kill/reap before TemporaryDirectory exits.
    with output.open("wb") as stdout, (work / "stderr").open("wb+") as stderr:
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
            )
        except OSError as exc:
            raise ContractError("the video reference decoder could not start") from exc
        try:
            code = process.wait(timeout=timeout)
        except BaseException as exc:
            process.kill()
            process.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise ContractError("video reference decoding timed out") from exc
            raise
        if code:
            # Do not echo paths, container tags or source metadata in public errors.
            raise ContractError("the video reference is not a decodable MP4, MOV or WebM")


def _probe_metadata(value: Any) -> tuple[int, int, float, int]:
    if not isinstance(value, dict) or not isinstance(value.get("streams"), list):
        raise ContractError("the video reference has no readable stream metadata")
    streams = [row for row in value["streams"] if row.get("codec_type") == "video"]
    if len(streams) != 1 or streams[0].get("disposition", {}).get("attached_pic"):
        raise ContractError("a video reference must contain exactly one moving video stream")
    video = streams[0]
    try:
        width, height = int(video["width"]), int(video["height"])
        fps = Fraction(video["avg_frame_rate"])
        duration = float(video.get("duration") or value.get("format", {}).get("duration"))
        rotations = [
            float(row["rotation"])
            for row in video.get("side_data_list", [])
            if "rotation" in row
        ]
        if "rotate" in video.get("tags", {}):
            rotations.append(float(video["tags"]["rotate"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ContractError("the video reference has invalid geometry or timing") from exc
    if (
        len(rotations) > 1
        or any(not math.isfinite(rotation) or rotation % 90 for rotation in rotations)
        or video.get("sample_aspect_ratio") not in {None, "N/A", "0:1", "1:1"}
        or not 0 < fps <= 240
        or not math.isfinite(duration)
        or not 2 <= duration <= 5
    ):
        raise ContractError(
            "video references require a complete 2-5s clip, square pixels, a readable "
            "frame rate up to 240 fps, and an unambiguous right-angle rotation"
        )
    if rotations and int(rotations[0]) % 360 in {90, 270}:
        width, height = height, width
    frames = math.ceil(duration * 24)
    reference_video_geometry(width, height, frames)
    return width, height, duration, frames


def read_video_reference(path: Path) -> DecodedVideoReference:
    """Decode an immutable copy of one complete 2-5s MP4/MOV/WebM on the CPU.

    Audio/subtitles are discarded. CFR24 with EOF pass preserves fractional-rate
    final frames; no output ``-t`` is used. The official H3 setup later performs
    its spatial normalization once, independently of the requested output size.
    """
    import numpy as np

    if not isinstance(path, Path):
        raise ContractError("reference_video must be a local pathlib.Path")
    ffmpeg, ffprobe = media_executables()
    # Nonblocking open plus fstat also rejects FIFOs/devices without waiting.
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or not 0 < info.st_size <= MAX_VIDEO_REFERENCE_BYTES
            ):
                raise ContractError(
                    "a video reference must be a nonempty local file up to 20 MiB"
                )
            data = source.read(MAX_VIDEO_REFERENCE_BYTES + 1)
    except OSError as exc:
        raise ContractError("the local video reference could not be read") from exc
    if not data or len(data) > MAX_VIDEO_REFERENCE_BYTES:
        raise ContractError("a video reference must contain at most 20 MiB")
    size_bytes = len(data)
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix="vflash-reference-") as tmp:
        work = Path(tmp)
        snapshot = work / "source.media"
        snapshot.write_bytes(data)
        del data
        probe = work / "probe.json"
        _run_media(
            [
                ffprobe,
                "-v",
                "error",
                "-threads",
                "2",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                _DEMUXERS,
                "-show_entries",
                "stream=codec_type,width,height,avg_frame_rate,duration,sample_aspect_ratio:"
                "stream_disposition=attached_pic:stream_tags=rotate:stream_side_data=rotation:format=duration",
                "-of",
                "json",
                str(snapshot),
            ],
            work,
            output=probe,
            timeout=15,
        )
        if probe.stat().st_size > 65536:
            raise ContractError("the video reference has excessive stream metadata")
        try:
            width, height, duration, frame_count = _probe_metadata(
                json.loads(probe.read_bytes())
            )
        except (ValueError, UnicodeError) as exc:
            raise ContractError("the video reference metadata is invalid") from exc
        raw = work / "frames.rgb"
        _run_media(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-threads",
                "2",
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                _DEMUXERS,
                "-i",
                str(snapshot),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                "fps=24:eof_action=pass,setsar=1",
                "-filter_threads",
                "2",
                "-frames:v",
                str(frame_count),
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-fs",
                str(frame_count * width * height * 3 + 1),
                "pipe:1",
            ],
            work,
            output=raw,
            timeout=60,
        )
        # Bound the actual output as well as the declared clock. Maximum RGB
        # is 120 * 475136 * 3 bytes; no normalized 768p frame copies live here.
        if raw.stat().st_size != frame_count * width * height * 3:
            raise ContractError("decoded video frames differ from the complete source clock")
        pixels = np.fromfile(raw, dtype=np.uint8).reshape(frame_count, height, width, 3)
    metadata = {
        "kind": "video",
        "index": 1,
        "role": "reference",
        "size_bytes": size_bytes,
        "sha256": digest,
        "source_width": width,
        "source_height": height,
        "duration_seconds": duration,
        "fps": 24,
        "frames": frame_count,
        **reference_video_geometry(width, height, frame_count),
        "audio_conditioning": False,
        "decoded_rgb_sha256": hashlib.sha256(memoryview(pixels).cast("B")).hexdigest(),
    }
    return DecodedVideoReference(pixels, metadata)
