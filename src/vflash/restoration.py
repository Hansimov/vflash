"""Complete local-video restoration; optional and independent of H3 generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vflash.adapters.restoration_motion import restoration_segments
from vflash.adapters.stcdit_attention import ATTENTION_BACKENDS
from vflash.adapters.stcdit_runtime import LocalStcditTiny
from vflash.media.encoding import MediaError, media_executables, probe_mp4
from vflash.media.restoration import encode_restored_video


def restore_video(
    source_video: Path,
    output_path: Path,
    *,
    source: Path,
    weights: Path,
    caption: str,
    seed: int = 42,
    trust_local_code: bool = False,
    device: str = "cuda:0",
    memory_policy: str = "offload",
    attention_backend: str = "torch",
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Decode, segment, enhance and atomically publish a separate complete video."""
    from PIL import Image

    if trust_local_code is not True:
        raise ValueError("restoration requires trust_local_code=True")
    if memory_policy not in {"offload", "resident"}:
        raise ValueError("restoration memory_policy must be offload or resident")
    if attention_backend not in ATTENTION_BACKENDS:
        raise ValueError("restoration attention must be torch or sage-int8-fp16")
    if not isinstance(caption, str) or not caption.strip() or len(caption) > 10000:
        raise ValueError("supply an observation caption of one to 10000 characters")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if output_path.exists() or output_path.is_symlink():
        raise MediaError("the output path already exists")
    ffmpeg, ffprobe = media_executables()
    metadata = probe_mp4(source_video, ffprobe=ffprobe)
    videos = [row for row in metadata["streams"] if row.get("codec_type") == "video"]
    if len(videos) != 1:
        raise MediaError("restoration requires one video stream")
    if sum(row.get("codec_type") == "audio" for row in metadata["streams"]) > 1:
        raise MediaError("restoration supports at most one original audio stream")
    video = videos[0]
    count = int(video.get("nb_frames", 0))
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if (
        not 1 <= count <= 240
        or width < 32
        or height < 32
        or width % 32
        or height % 32
        or width * height > 1024**2
        or video.get("avg_frame_rate") != "24/1"
        or video.get("r_frame_rate") != "24/1"
    ):
        raise MediaError(
            "restoration requires a 32-aligned, 24 fps video up to 10s/one mebipixel"
        )
    began = time.monotonic()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    enhanced = ()
    try:
        if on_progress is not None:
            on_progress("preparing", 0, 10)
        with tempfile.TemporaryDirectory(
            prefix=".vflash-restore-input-", dir=output_path.parent
        ) as tmp:
            directory = Path(tmp)
            subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-nostdin",
                    "-n",
                    "-i",
                    str(source_video.resolve()),
                    "-map",
                    "0:v:0",
                    "-fps_mode",
                    "passthrough",
                    "-frames:v",
                    str(count + 1),
                    "-start_number",
                    "0",
                    "-threads",
                    "2",
                    str(directory / "frame-%03d.png"),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            paths = sorted(directory.glob("frame-*.png"))
            if len(paths) != count:
                raise MediaError("decoded input does not match the declared source clock")
            for path in paths:
                with Image.open(path) as frame:
                    frames.append(frame.convert("RGB"))
        segments = restoration_segments(frames)
        prepared = time.monotonic()
        with LocalStcditTiny(
            source=source,
            weights=weights,
            device=device,
            trust_local_code=trust_local_code,
            memory_policy=memory_policy,
            attention_backend=attention_backend,
        ) as restorer:
            loaded = time.monotonic()
            result = restorer.restore(
                frames, caption=caption, segments=segments, seed=seed, on_progress=on_progress
            )
            enhanced = result.frames
        if on_progress is not None:
            on_progress("delivery", 10, 10)
        media = encode_restored_video(enhanced, source_video, output_path)
        return {
            "profile": result.profile,
            "memory_policy": memory_policy,
            "attention_backend": attention_backend,
            "media": media,
            "segments": len(segments),
            "stages": {
                "prepare_seconds": prepared - began,
                "load_seconds": loaded - prepared,
                "restore_seconds": result.elapsed_seconds,
            },
            "total_seconds": time.monotonic() - began,
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise MediaError("restoration input decoding failed or timed out") from error
    finally:
        for frame in (*frames, *enhanced):
            frame.close()


def add_restoration_command(commands: Any) -> None:
    parser = commands.add_parser(
        "restore-video", help="opt-in STCDiT enhancement; retain source"
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-code", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--caption-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-policy", choices=("offload", "resident"), default="offload")
    parser.add_argument("--attention-backend", choices=ATTENTION_BACKENDS, default="torch")
    parser.add_argument("--trust-local-code", action="store_true", required=True)


def run_restoration_command(args: argparse.Namespace) -> int:
    if args.caption_file.stat().st_size > 40000:
        raise ValueError("the caption file exceeds 40000 UTF-8 bytes")
    result = restore_video(
        args.source_video,
        args.output,
        source=args.runtime_code,
        weights=args.weights,
        caption=args.caption_file.read_text(),
        seed=args.seed,
        device=args.device,
        memory_policy=args.memory_policy,
        attention_backend=args.attention_backend,
        trust_local_code=args.trust_local_code,
    )
    print(json.dumps(result, indent=2))
    return 0
