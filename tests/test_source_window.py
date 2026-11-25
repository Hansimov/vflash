"""Exercise a complete source-window exchange using only FFmpeg and the CPU."""

import json
import shutil
import subprocess

import pytest

from vflash.cli import main
from vflash.media.probe import probe_normalized_video
from vflash.media.remake_source import finish_remake_window, prepare_remake_source

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)


def encode(path, *args):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", *args, "-threads", "1", str(path)], check=True
    )


def test_prepare_and_finish_preserve_real_content_and_delayed_audio(tmp_path, capsys):
    source, window, result = (
        tmp_path / name for name in ("source.mp4", "window.mp4", "result.mp4")
    )
    encode(
        source,
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=192x108:r=30:d=2",
        "-itsoffset",
        "0.35",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=600:duration=1.5",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
    )
    assert (
        main(
            [
                "prepare-remake-source",
                str(source),
                str(window),
                "--width",
                "512",
                "--height",
                "512",
                "--start",
                "0.05",
                "--duration",
                "1.10",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["content_rect"] == [0, 112, 512, 288]
    assert (receipt["content_frames"], receipt["model_frames"], receipt["padding_frames"]) == (
        27,
        90,
        63,
    )
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(window),
            "-map",
            "0:a",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "s16le",
            "-",
        ]
    )
    assert not any(data[:1600]) and any(data[8000:10000])
    assert finish_remake_window(window, result, receipt)["frames"] == 27
    video, audio = probe_normalized_video(result)
    assert (video["width"], video["height"], video["nb_frames"]) == (512, 288, "27")
    assert audio and abs(float(audio[0]["duration"]) - 27 / 24) < 0.01
    with pytest.raises(ValueError, match="new MP4"):
        finish_remake_window(window, result, receipt)


def test_select_last_displayed_picture_and_reject_outside_interval(tmp_path):
    source, window = tmp_path / "silent.mp4", tmp_path / "window.mp4"
    encode(source, "-f", "lavfi", "-i", "color=red:s=120x160:r=25:d=3", "-c:v", "libx264")
    receipt = prepare_remake_source(
        source,
        window,
        width=480,
        height=640,
        start_seconds="2.991",
        duration_seconds=".009",
        soundtrack="silent",
    )
    assert receipt["content_frames"] == 1
    assert probe_normalized_video(window)[0]["nb_frames"] == "90"
    with pytest.raises(ValueError, match="within the source"):
        prepare_remake_source(
            source, tmp_path / "outside.mp4", width=480, height=640, start_seconds=4
        )


def test_long_prefix_checks_only_its_consumed_picture_clock(tmp_path):
    source, irregular = tmp_path / "source.mp4", tmp_path / "irregular.mp4"
    encode(
        source,
        "-f",
        "lavfi",
        "-i",
        "color=s=96x64:r=24:d=20",
        "-g",
        "24",
        "-bf",
        "0",
        "-c:v",
        "libx264",
    )
    encode(
        irregular,
        "-i",
        str(source),
        "-c",
        "copy",
        "-bsf:v",
        "setts=pts='if(eq(N,390),PTS+96,PTS)'",
    )
    assert probe_normalized_video(irregular, history_interval=(30, 17))[0]["nb_frames"] == "480"
    with pytest.raises(ValueError, match="presentation timestamps"):
        probe_normalized_video(irregular, history_interval=(380, 34))
