import subprocess

import pytest

from vflash.media.probe import probe_normalized_video
from vflash.media.remake_footage import export_frames, finish_window
from vflash.media.remake_source import finish_remake_window, prepare_remake_source


def test_export_real_frames_and_recover_interrupted_soundtrack(tmp_path):
    source = tmp_path / "phone.mp4"
    window = tmp_path / "window.mp4"
    truncated = tmp_path / "truncated.mp4"
    dest = tmp_path / "delivery.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x96:r=30:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=d=3",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    receipt = prepare_remake_source(source, window, width=480, height=480, duration_seconds=2)
    directory = tmp_path / "pins"
    rows = export_frames(window, directory, receipt, [31, 0])
    assert [row["frame"] for row in rows] == [31, 0]
    for row in rows:
        expected = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(window),
                "-vf",
                f"select=eq(n\\,{row['frame']})",
                "-frames:v",
                "1",
                "-threads",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
        )
        assert (directory / row["file"]).read_bytes() == expected
    with pytest.raises(FileExistsError):
        export_frames(window, directory, receipt, [2])
    with pytest.raises(ValueError, match="inside"):
        export_frames(window, tmp_path / "padding", receipt, [48])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(window),
            "-f",
            "lavfi",
            "-i",
            "sine=d=0.2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(truncated),
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="cover"):
        finish_remake_window(truncated, dest, receipt)
    with pytest.raises(ValueError, match="cover"):
        finish_window(truncated, dest, receipt)
    assert not dest.exists()
    gap = tmp_path / "audio-gap.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(window),
            "-map",
            "0",
            "-c",
            "copy",
            "-bsf:a",
            "noise=drop='between(pts*tb,0.4,0.9)'",
            str(gap),
        ],
        check=True,
    )
    for operation in (finish_remake_window, finish_window):
        with pytest.raises(ValueError, match="continuous"):
            operation(gap, dest, receipt)
        assert not dest.exists()
    assert finish_window(truncated, dest, receipt, window)["frames"] == 48
    video, audio = probe_normalized_video(dest)
    assert int(video["nb_frames"]) == 48 and float(audio[0]["duration"]) == pytest.approx(
        2, abs=0.002
    )
