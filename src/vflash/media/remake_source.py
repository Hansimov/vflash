"""Prepare one display-correct H.264 input window for an original Remake graph."""

import os
from fractions import Fraction
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

from vflash.media.probe import MP4_INPUT, probe_normalized_video, probe_video, run_media_command


def valid_canvas(width: int, height: int) -> bool:
    return (
        all(
            type(edge) is int and 480 <= edge <= 3072 and edge % 32 == 0
            for edge in (width, height)
        )
        and min(width, height) <= 1440
        and width * height <= 2048**2
        and max(width, height) <= 4 * min(width, height)
    )


def prepare_remake_source(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    start_seconds=0,
    duration_seconds=None,
    soundtrack="preserve",
    cancel_check=None,
) -> dict:
    """Trim, contain and align one window. Publish a new file after verifying it.

    The returned content frame count excludes repeat padding. A preserved track is
    synchronized to the selected video origin, then padded with silence to model length.
    """
    if not valid_canvas(width, height):
        raise ValueError("Choose a 32-aligned canvas within the Remake request limits")
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise ValueError("Choose a new MP4 output path")
    if soundtrack not in {"preserve", "silent"}:
        raise ValueError("Select preserve or silent sound")
    video, audio = probe_video(source, cancel_check)
    tick = Fraction(video["time_base"])
    origin = int(video["start_pts"]) * tick
    length = int(video["duration_ts"]) * tick
    start = Fraction(str(start_seconds))
    duration = length - start if duration_seconds is None else Fraction(str(duration_seconds))
    if start < 0 or duration <= 0 or start + duration > length:
        raise ValueError("Select a positive interval within the source video")
    active = ceil(duration * 24)
    if active > 345:
        raise ValueError("Select at most 345 frames (14.375 seconds) per source window")
    if soundtrack == "preserve" and not audio:
        raise ValueError("Source has no soundtrack; select silent delivery")
    frames = max(90, 5 + 17 * ceil(Fraction(active - 5, 17)))
    offset = origin + start

    def seconds(value):
        return f"{float(value):.12f}"

    sar = video.get("sample_aspect_ratio", "1:1")
    sar = Fraction(sar.replace(":", "/")) if sar != "N/A" else Fraction(1)
    if sar <= 0:
        raise ValueError("A positive source sample aspect ratio is required")
    rotation = sum(float(row.get("rotation", 0)) for row in video.get("side_data_list", []))
    if rotation % 90:
        raise ValueError(
            "Normalize non-quarter-turn display rotation before preparing a window"
        )
    aspect = Fraction(video["width"], video["height"]) * sar
    if rotation % 180:
        aspect = 1 / aspect
    fit_width = min(width, max(2, int(height * aspect) // 2 * 2))
    fit_height = min(height, max(2, int(width / aspect) // 2 * 2))
    left = (width - fit_width) // 4 * 2
    top = (height - fit_height) // 4 * 2
    # Sample the picture already on screen at the selected instant. Removing
    # packets before the start would drop that picture on an off-grid cut.
    vf = (
        f"trim=end={seconds(offset + duration)},setpts=PTS-{seconds(offset)}/TB,"
        "tpad=stop_mode=clone:stop_duration=1,fps=24:start_time=0:round=up,"
        f"trim=end_frame={active},scale={fit_width}:{fit_height},setsar=1,"
        f"pad={width}:{height}:{left}:{top},"
        f"tpad=stop_mode=clone:stop_duration={seconds(Fraction(frames, 24))},"
        f"trim=end_frame={frames},settb=1/24,setpts=N"
    )
    # Seek near a late selection, keeping decoder preroll and absolute timestamps.
    seek = (
        ["-seek_timestamp", "1", "-noaccurate_seek", "-ss", seconds(origin + start - 2)]
        if start > 2
        else []
    )
    ffmpeg = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-n"]
    source_input = ["-copyts", *seek, *MP4_INPUT, "-threads", "2", "-i", str(source.absolute())]
    with TemporaryDirectory(prefix=".vflash-source-", dir=output.parent) as temp:
        picture = Path(temp) / "picture.mp4"
        run_media_command(
            [
                *ffmpeg,
                *source_input,
                "-map",
                "0:v:0",
                "-filter_threads",
                "2",
                "-vf",
                vf,
                "-frames:v",
                str(frames),
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "17",
                "-threads",
                "2",
                "-map_metadata",
                "-1",
                "-movflags",
                "+faststart",
                str(picture),
            ],
            timeout=180,
            cancel_check=cancel_check,
        )
        staged = picture
        if soundtrack == "preserve":
            track = audio[0]
            audio_start = int(track["start_pts"]) * Fraction(track["time_base"])
            audio_end = audio_start + int(track["duration_ts"]) * Fraction(track["time_base"])
            sound = Path(temp) / "sound.m4a"
            samples = frames * 2000
            if audio_start >= offset + duration or audio_end <= offset:
                inputs = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
                af = f"atrim=end_sample={samples}"
            else:
                inputs = source_input
                # Retain the source audio clock, including silence before a delayed track.
                af = (
                    f"atrim=start={seconds(offset)}:end={seconds(offset + duration)},"
                    f"asetpts=PTS-{seconds(offset)}/TB,aresample=48000:async=1:first_pts=0,"
                    f"apad=whole_len={samples},atrim=end_sample={samples}"
                )
            run_media_command(
                [
                    *ffmpeg,
                    *inputs,
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-af",
                    af,
                    "-t",
                    seconds(Fraction(frames, 24)),
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(sound),
                ],
                timeout=60,
                cancel_check=cancel_check,
            )
            staged = Path(temp) / "window.mp4"
            run_media_command(
                [
                    *ffmpeg,
                    "-i",
                    str(picture),
                    "-i",
                    str(sound),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(staged),
                ],
                timeout=30,
                cancel_check=cancel_check,
            )
        result, tracks = probe_normalized_video(staged, cancel_check)
        if (int(result["nb_frames"]), result["width"], result["height"], bool(tracks)) != (
            frames,
            width,
            height,
            soundtrack == "preserve",
        ):
            raise ValueError("Prepared output differs from the requested window or soundtrack")
        os.link(staged, output)
    return {
        "version": 1,
        "content_rect": [left, top, fit_width, fit_height],
        "source_start_seconds": float(start),
        "source_duration_seconds": float(duration),
        "content_frames": active,
        "model_frames": frames,
        "padding_frames": frames - active,
        "width": width,
        "height": height,
        "fps": 24,
        "soundtrack": soundtrack,
    }


def finish_remake_window(
    model_output: Path, output: Path, receipt: dict, *, cancel_check=None
) -> dict:
    """Remove repeat padding and letterboxing from a generated window on its 24 fps clock."""
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise ValueError("Choose a new MP4 output path")
    if not isinstance(receipt, dict) or receipt.get("version") != 1:
        raise ValueError("Use the receipt from prepare-remake-source")
    width, height = receipt.get("width"), receipt.get("height")
    active, frames = receipt.get("content_frames"), receipt.get("model_frames")
    rect = receipt.get("content_rect")
    if (
        not valid_canvas(width, height)
        or type(active) is not int
        or type(frames) is not int
        or not 1 <= active <= frames <= 345
        or frames < 90
        or (frames - 5) % 17
        or receipt.get("fps") != 24
        or receipt.get("soundtrack") not in {"preserve", "silent"}
        or not isinstance(rect, list)
        or len(rect) != 4
        or any(type(n) is not int or n < 0 or n % 2 for n in rect)
    ):
        raise ValueError("Invalid prepared window receipt")
    x, y, w, h = rect
    if not w or not h or x + w > width or y + h > height:
        raise ValueError("Content rectangle must be inside the prepared canvas")
    video, audio = probe_normalized_video(model_output, cancel_check)
    if (video["width"], video["height"], int(video["nb_frames"])) != (width, height, frames):
        raise ValueError("Generated window differs from the receipt's canvas or clock")
    preserve = receipt["soundtrack"] == "preserve"
    if preserve and not audio:
        raise ValueError("The generated window is missing its preserved soundtrack")
    with TemporaryDirectory(prefix=".vflash-result-", dir=output.parent) as temp:
        staged = Path(temp) / "result.mp4"
        args = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-n",
            *MP4_INPUT,
            "-threads",
            "2",
            "-i",
            str(model_output.absolute()),
            "-map",
            "0:v:0",
            "-filter_threads",
            "2",
            "-vf",
            f"trim=end_frame={active},crop={w}:{h}:{x}:{y},setsar=1",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
        ]
        if preserve:
            args += [
                "-map",
                "0:a:0",
                "-af",
                f"aresample=48000,atrim=end_sample={active * 2000}",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
            ]
        else:
            args += ["-an"]
        run_media_command(
            [*args, "-map_metadata", "-1", "-movflags", "+faststart", str(staged)],
            timeout=180,
            cancel_check=cancel_check,
        )
        complete, tracks = probe_normalized_video(staged, cancel_check)
        if (
            int(complete["nb_frames"]),
            complete["width"],
            complete["height"],
            bool(tracks),
        ) != (active, w, h, preserve):
            raise ValueError(
                "Finished clip did not retain the intended geometry, clock and sound"
            )
        os.link(staged, output)
    return {
        "frames": active,
        "fps": 24,
        "width": w,
        "height": h,
        "soundtrack": receipt["soundtrack"],
    }
