"""Extract paintable source frames or finish an edit with the prepared soundtrack.

The standalone CLI works with the Vflash 0.29.0 wheel, FFmpeg and FFprobe.
"""

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from vflash.media.probe import MP4_INPUT, probe_normalized_video, run_media_command


def require_audio_coverage(track, frames, *, path=None, cancel_check=None):
    """Check the selected picture interval against the audio presentation clock."""
    tick = Fraction(track["time_base"])
    start = int(track["start_pts"]) * tick
    end = start + int(track["duration_ts"]) * tick
    tolerance = max(tick, Fraction(1, 1000))
    if abs(start) > tolerance or end + tolerance < Fraction(frames, 24):
        raise ValueError(
            "The preserved soundtrack does not cover the selected picture interval"
        )
    if path is None:
        return
    # In MP4 a missing run of AAC packets can be represented as a single
    # long packet duration. Only decoded sample counts prove audible coverage.
    stop = Fraction(frames, 24)
    sample_rate = int(track["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("A positive audio sample rate is required")
    decoded = run_media_command(
        [
            "ffprobe",
            "-v",
            "error",
            *MP4_INPUT,
            "-select_streams",
            "a:0",
            "-read_intervals",
            f"%+{float(stop) + 0.1:.6f}",
            "-show_frames",
            "-show_entries",
            "frame=pts,nb_samples",
            "-of",
            "json",
            str(path.absolute()),
        ],
        timeout=30,
        cancel_check=cancel_check,
    )
    covered = Fraction(0)
    for frame in json.loads(decoded.stdout)["frames"]:
        pts, count = frame.get("pts"), frame.get("nb_samples")
        if not isinstance(pts, int) or not isinstance(count, int) or count <= 0:
            raise ValueError("Decoded audio lacks a usable sample clock")
        begin = pts * tick
        if begin > covered + tolerance:
            break
        covered = max(covered, begin + Fraction(count, sample_rate))
        if covered + tolerance >= stop:
            return
    raise ValueError(
        f"Decoded audio covers only {float(covered):.3f}s of continuous picture time"
    )


def window_geometry(path, receipt):
    from vflash.media.remake_source import valid_canvas

    video, tracks = probe_normalized_video(path)
    active, frames = receipt.get("content_frames"), receipt.get("model_frames")
    if (
        receipt.get("version") != 1
        or receipt.get("fps") != 24
        or type(active) is not int
        or type(frames) is not int
        or not 1 <= active <= frames <= 345
        or frames < 90
        or (frames - 5) % 17
        or not valid_canvas(receipt.get("width"), receipt.get("height"))
        or (video["width"], video["height"], int(video["nb_frames"]))
        != (receipt["width"], receipt["height"], frames)
    ):
        raise ValueError("The receipt and prepared window must share a valid canvas and clock")
    return tracks


def export_frames(source, directory, receipt, frames):
    window_geometry(source, receipt)
    if (
        not 1 <= len(frames) <= 4
        or len(set(frames)) != len(frames)
        or any(type(n) is not int or not 0 <= n < receipt["content_frames"] for n in frames)
    ):
        raise ValueError("Choose one to four distinct frames inside the selected content")
    if directory.exists():
        raise FileExistsError(directory)
    with TemporaryDirectory(prefix=".vflash-pins-", dir=directory.parent) as tmp:
        root = Path(tmp)
        selected = sorted(frames)
        expression = "+".join(f"eq(n\\,{n})" for n in selected)
        run_media_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-xerror",
                *MP4_INPUT,
                "-threads",
                "2",
                "-i",
                str(source.absolute()),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "select=" + expression,
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(selected)),
                "-threads",
                "1",
                str(root / "picture-%02d.png"),
            ],
            timeout=60,
        )
        files = {frame: root / f"picture-{i:02d}.png" for i, frame in enumerate(selected, 1)}
        rows = [
            {"frame": n, "file": f"frame-{n:03d}.png", "window_seconds": n / 24} for n in frames
        ]
        if any(not path.is_file() or not path.stat().st_size for path in files.values()):
            raise ValueError("The decoder did not produce every requested frame")
        manifest = root / "frames.json"
        manifest.write_text(json.dumps(rows, indent=2) + "\n")
        directory.mkdir()
        published = []
        try:
            for row in rows:
                output = directory / row["file"]
                os.link(files[row["frame"]], output)
                published.append(output)
            os.link(manifest, directory / "frames.json")
        except BaseException:
            for output in published:
                output.unlink()
            directory.rmdir()
            raise
    return rows


def finish_window(model_output, output, receipt, original_audio=None):
    from vflash.media.remake_source import finish_remake_window

    tracks = window_geometry(model_output, receipt)
    if original_audio is not None:
        if receipt.get("soundtrack") != "preserve":
            raise ValueError("The preparation receipt must request preserved audio")
        tracks = window_geometry(original_audio, receipt)
    if receipt.get("soundtrack") == "preserve":
        if not tracks:
            raise ValueError("A preserved soundtrack is required")
        require_audio_coverage(
            tracks[0], receipt["content_frames"], path=original_audio or model_output
        )
    if output.exists() or output.is_symlink() or output.suffix.lower() != ".mp4":
        raise ValueError("Choose a new MP4 file")
    with TemporaryDirectory(prefix=".vflash-delivery-", dir=output.parent) as tmp:
        root = Path(tmp)
        source = model_output
        if original_audio is not None:
            source = root / "window.mp4"
            run_media_command(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-n",
                    *MP4_INPUT,
                    "-i",
                    str(model_output.absolute()),
                    *MP4_INPUT,
                    "-i",
                    str(original_audio.absolute()),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c",
                    "copy",
                    "-map_metadata",
                    "-1",
                    str(source),
                ],
                timeout=60,
            )
        result = root / "export.mp4"
        details = finish_remake_window(source, result, receipt)
        if receipt["soundtrack"] == "preserve":
            require_audio_coverage(
                probe_normalized_video(result)[1][0], details["frames"], path=result
            )
        os.link(result, output)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    frames = commands.add_parser("frames", help="Export full-canvas PNGs to a new directory.")
    finish = commands.add_parser("finish", help="Remove padding and verify preserved sound.")
    for command in (frames, finish):
        command.add_argument("window", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--receipt", type=Path, required=True)
    frames.add_argument("--frames", type=int, nargs="+", required=True)
    finish.add_argument(
        "--original-audio", type=Path, help="Use sound from the matching prepared source."
    )
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if args.action == "frames":
        result = export_frames(args.window, args.output, receipt, args.frames)
    else:
        result = finish_window(args.window, args.output, receipt, args.original_audio)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
