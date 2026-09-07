from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vflash.adapters.video_references import (
    MAX_VIDEO_REFERENCE_BYTES,
    _run_media,
    read_video_reference,
)
from vflash.contracts import ContractError
from vflash.native.h3_conditioning_bundle import _validate_video_reference
from vflash.pipeline.contracts import VideoRequest

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is required"
)


def make_clip(path: Path, *, seconds="2", fps="24", size="128x72", audio=False):
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={fps}:duration={seconds}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += ["-c:v", "libvpx-vp9" if path.suffix == ".webm" else "libx264", "-threads", "1"]
    if audio:
        args += ["-c:a", "aac"]
    subprocess.run([*args, str(path)], check=True, capture_output=True, timeout=20)
    return path


@pytest.mark.parametrize(
    "seconds,fps,suffix,expected",
    [
        ("2", "24", ".mp4", 48),
        ("5", "24", ".mp4", 120),
        ("2.1", "30", ".mov", 51),
        ("2.002", "30000/1001", ".mp4", 49),
        ("4.971633", "30000/1001", ".mp4", 120),
        ("2", "24", ".webm", 48),
    ],
)
def test_real_decoder_preserves_fractional_eof_and_schema2(
    tmp_path, seconds, fps, suffix, expected
):
    source = make_clip(tmp_path / f"reference{suffix}", seconds=seconds, fps=fps)
    with read_video_reference(source) as decoded:
        assert decoded.frames.shape == (expected, 72, 128, 3)
        assert decoded.frames.dtype == np.uint8
        assert decoded.metadata["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert decoded.metadata["size_bytes"] == source.stat().st_size
        assert decoded.metadata["frames"] == math.ceil(
            decoded.metadata["duration_seconds"] * 24
        )
        assert _validate_video_reference(decoded.metadata) == decoded.metadata
        # The decoder keeps source-sized RGB. Spatial normalization belongs to
        # the official setup, independently of the requested output canvas.
        assert decoded.metadata["width"] == 1344 and decoded.metadata["height"] == 768
        if fps == "24":
            original = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(source),
                    "-an",
                    "-vf",
                    "fps=24,setsar=1",
                    "-filter_threads",
                    "1",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
                timeout=20,
            ).stdout
            assert decoded.frames.tobytes() == original
    assert decoded.frames is None
    with pytest.raises(ContractError, match="closed"):
        decoded.require_frames()
    decoded.close()


def test_source_audio_is_discarded_and_owned_copy_avoids_reopening_input(tmp_path, monkeypatch):
    source = make_clip(tmp_path / "with-audio.mp4", audio=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = _run_media

    def change_original(*args, **kwargs):
        # The input copy was already frozen before either external process.
        source.write_bytes(b"the caller changed its source after reading")
        return original(*args, **kwargs)

    monkeypatch.setattr("vflash.adapters.video_references._run_media", change_original)
    with read_video_reference(source) as decoded:
        assert decoded.metadata["audio_conditioning"] is False
        assert decoded.metadata["sha256"] == digest
        assert decoded.frames.shape[0] == 48


@pytest.mark.parametrize(
    "seconds,size", [("1", "128x72"), ("6", "128x72"), ("2", "960x512"), ("2", "256x64")]
)
def test_actual_media_outside_duration_source_or_normalized_budget_fails(
    tmp_path, seconds, size
):
    source = make_clip(tmp_path / "invalid.mp4", seconds=seconds, size=size)
    with pytest.raises(ContractError):
        read_video_reference(source)


@pytest.mark.parametrize("kind", ["empty", "large", "corrupt", "directory", "fifo"])
def test_non_media_files_are_rejected_before_model_loading(tmp_path, kind):
    source = tmp_path / "invalid"
    if kind == "directory":
        source.mkdir()
    elif kind == "fifo":
        import os

        os.mkfifo(source)
    else:
        with source.open("wb") as handle:
            if kind == "large":
                handle.truncate(MAX_VIDEO_REFERENCE_BYTES + 1)
            elif kind == "corrupt":
                handle.write(b"not a video")
    with pytest.raises(ContractError):
        read_video_reference(source)


def test_timeout_reaps_actual_cpu_child(tmp_path):
    pid_file = tmp_path / "child.json"
    command = [
        sys.executable,
        "-c",
        "import os,time,json,sys; "
        "open(sys.argv[1],'w').write(json.dumps(os.getpid())); time.sleep(30)",
        str(pid_file),
    ]
    with pytest.raises(ContractError, match="timed out"):
        _run_media(command, tmp_path, output=tmp_path / "out", timeout=1)
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(json.loads(pid_file.read_text()), 0)


def test_decode_error_and_interrupt_clean_temporary_files(tmp_path, monkeypatch):
    source = make_clip(tmp_path / "reference.mp4")
    original = _run_media
    directories = []

    def interrupt(command, work, **kwargs):
        directories.append(work)
        if command[0].endswith("ffmpeg"):
            raise KeyboardInterrupt
        return original(command, work, **kwargs)

    monkeypatch.setattr("vflash.adapters.video_references._run_media", interrupt)
    with pytest.raises(KeyboardInterrupt):
        read_video_reference(source)
    assert len(directories) == 2 and not any(path.exists() for path in directories)


@pytest.mark.parametrize(
    "values",
    [
        {"reference_video": "video.mp4"},
        {"reference_video": Path("video.mp4"), "reference": Path("image.png")},
        {"reference_video": Path("video.mp4"), "references": (Path("image.png"),)},
        {"reference_video": Path("video.mp4"), "prompt": "<Picture 1> moves"},
        {"reference_video": Path("video.mp4"), "prompt": "<Video 2> moves"},
        {"reference_video": Path("video.mp4"), "prompt": "<Video1> moves"},
        {"prompt": "<Video 1> without a video"},
    ],
)
def test_video_request_rejects_ambiguous_modality_and_labels(values):
    with pytest.raises(ContractError):
        VideoRequest(**{"prompt": "A scene", **values})
