"""Extract paintable source frames or finish an edit with the prepared soundtrack.

The standalone CLI works with the Vflash 0.29.0 wheel, FFmpeg and FFprobe.
"""

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from vflash.media.probe import MP4_INPUT, probe_normalized_video, run_media_command


def prepared_identity(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_original(path, receipt):
    expected = receipt.get("prepared_window_sha256")
    if expected is not None and (
        not isinstance(expected, str)
        or len(expected) != 64
        or prepared_identity(path) != expected
    ):
        raise ValueError("Use the exact source window saved with this preparation receipt")


def prepare_window(
    source, directory, *, width, height, start="0", duration=None, soundtrack="preserve"
):
    """Publish the model window together with its original-source identity and crop receipt."""
    from vflash.media.remake_source import prepare_remake_source

    if directory.exists() or directory.is_symlink():
        raise FileExistsError(directory)
    with TemporaryDirectory(prefix=".vflash-prepare-", dir=directory.parent) as tmp:
        staged = Path(tmp) / "prepared"
        staged.mkdir()
        window = staged / "window.mp4"
        receipt = prepare_remake_source(
            source,
            window,
            width=width,
            height=height,
            start_seconds=start,
            duration_seconds=duration,
            soundtrack=soundtrack,
        )
        receipt["prepared_window_sha256"] = prepared_identity(window)
        (staged / "window.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Reserve the name exclusively before publishing both completed files.
        directory.mkdir()
        try:
            os.replace(staged, directory)
        except OSError:
            directory.rmdir()
            raise
    return receipt


def prepare_appearance(source, output, receipt):
    """Fit one upright appearance photo to the unpinned Remake input contract."""
    from PIL import Image, ImageOps

    from vflash.media.remake_source import valid_canvas

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    width, height = receipt.get("width"), receipt.get("height")
    if type(width) is not int or type(height) is not int or not valid_canvas(width, height):
        raise ValueError("Use the canvas from the prepared video receipt")
    edge = min(width, height)
    with Image.open(source) as original:
        if getattr(original, "n_frames", 1) != 1 or original.width * original.height > 16777216:
            raise ValueError("Provide one still image of at most 16 megapixels")
        image = ImageOps.exif_transpose(original).convert("RGBA")
    w, h = image.size
    if min(w, h) < 32 or not 0.25 <= w / h <= 4:
        raise ValueError("The upright photo needs an aspect ratio between 1:4 and 4:1")
    scale = min(edge / min(w, h), 4096 / max(w, h))
    real = (2 * round(w * scale / 2), 2 * round(h * scale / 2))
    canvas = tuple(max(edge, ((v + 31) // 32) * 32) for v in real)
    rgb = Image.new("RGB", image.size, "white")
    rgb.paste(image, mask=image.getchannel("A"))
    fitted = rgb.resize(real, Image.Resampling.LANCZOS)
    # Only the grid margin extends beyond the original composition.
    result = Image.new("RGB", canvas)
    x, y = ((outer - inner) // 4 * 2 for outer, inner in zip(canvas, real, strict=True))
    result.paste(fitted, (x, y))
    if x:
        result.paste(fitted.crop((0, 0, 1, real[1])).resize((x, real[1])), (0, y))
    if right := canvas[0] - x - real[0]:
        result.paste(
            fitted.crop((real[0] - 1, 0, real[0], real[1])).resize((right, real[1])),
            (x + real[0], y),
        )
    if y:
        result.paste(result.crop((0, y, canvas[0], y + 1)).resize((canvas[0], y)))
    if bottom := canvas[1] - y - real[1]:
        end = y + real[1]
        result.paste(
            result.crop((0, end - 1, canvas[0], end)).resize((canvas[0], bottom)), (0, end)
        )
    with TemporaryDirectory(prefix=".vflash-appearance-", dir=output.parent) as tmp:
        finished = Path(tmp) / "appearance.png"
        result.save(finished)
        os.link(finished, output)
    return {"width": canvas[0], "height": canvas[1], "content_rect": [x, y, *real]}


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
    verify_original(source, receipt)
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
        verify_original(original_audio, receipt)
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
    prepare = commands.add_parser(
        "prepare", help="Save a normalized video and its matching receipt together."
    )
    prepare.add_argument("source", type=Path)
    prepare.add_argument("directory", type=Path, help="A new directory in an existing parent.")
    prepare.add_argument("--width", type=int, required=True)
    prepare.add_argument("--height", type=int, required=True)
    prepare.add_argument("--start", default="0")
    prepare.add_argument("--duration")
    prepare.add_argument("--soundtrack", choices=("preserve", "silent"), default="preserve")
    photo = commands.add_parser("appearance", help="Prepare an unpinned appearance photo.")
    photo.add_argument("source", type=Path)
    photo.add_argument("output", type=Path)
    photo.add_argument("--receipt", type=Path, required=True)
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
    if args.action == "prepare":
        result = prepare_window(
            args.source,
            args.directory,
            width=args.width,
            height=args.height,
            start=args.start,
            duration=args.duration,
            soundtrack=args.soundtrack,
        )
    elif args.action == "appearance":
        result = prepare_appearance(
            args.source, args.output, json.loads(args.receipt.read_text())
        )
    elif args.action == "frames":
        receipt = json.loads(args.receipt.read_text())
        result = export_frames(args.window, args.output, receipt, args.frames)
    else:
        result = finish_window(
            args.window, args.output, json.loads(args.receipt.read_text()), args.original_audio
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
