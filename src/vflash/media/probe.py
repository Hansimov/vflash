"""Bounded local MP4 inspection for explicit complete-pipeline windows."""

import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from fractions import Fraction
from pathlib import Path

from vflash.cancellation import GenerationCancelled

# Do not let a source container open external data references or another protocol.
MP4_INPUT = (
    "-max_alloc",
    "268435456",
    "-max_streams",
    "2",
    "-protocol_whitelist",
    "file,pipe",
    "-f",
    "mov",
    "-enable_drefs",
    "0",
    "-use_absolute_path",
    "0",
    "-probesize",
    "8388608",
    "-analyzeduration",
    "5000000",
)


def probe_video(path: Path, cancel_check=None) -> tuple[dict, list[dict]]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("A regular owned local video is required")
    result = _run_probe(
        [
            "ffprobe",
            "-v",
            "error",
            *MP4_INPUT,
            "-show_streams",
            "-of",
            "json",
            str(path.absolute()),
        ],
        timeout=30,
        cancel_check=cancel_check,
    )
    streams = json.loads(result.stdout)["streams"]
    videos = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    if len(videos) != 1 or len(audio) > 1 or len(streams) != len(videos) + len(audio):
        raise ValueError("One video and optional single audio stream are required")
    if videos[0].get("codec_name") != "h264" or videos[0].get("disposition", {}).get(
        "attached_pic"
    ):
        raise ValueError("A moving H.264 source is required")
    return videos[0], audio


def probe_normalized_video(path: Path, cancel_check=None) -> tuple[dict, list[dict]]:
    video, audio = probe_video(path, cancel_check)
    if (
        Fraction(video["r_frame_rate"]) != 24
        or Fraction(video["avg_frame_rate"]) != 24
        or abs(float(video.get("start_time", 0))) > 0.000001
        or video.get("sample_aspect_ratio", "1:1") not in {"1:1", "N/A"}
        or any(s.get("rotation", 0) for s in video.get("side_data_list", []))
    ):
        raise ValueError("Normalize frame clock, rotation and pixel aspect before assembly")
    return video, audio


def _run_probe(argv, *, timeout, cancel_check=None):
    """Keep the owned probe in its own process group, including observer failures."""
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        while True:
            if cancel_check is not None and cancel_check():
                raise GenerationCancelled("Media inspection was cancelled")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Media inspection exceeded its deadline")
            try:
                stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise ValueError("The owned MP4 could not be inspected")
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    finally:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        process.communicate()
