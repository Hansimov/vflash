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


def test_paired_preparation_survives_retry_and_declines_a_different_soundtrack(tmp_path):
    from vflash.media.remake_footage import export_frames, finish_window, prepare_window

    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x96:r=24:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=d=4",
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
    first, other = tmp_path / "first", tmp_path / "other"
    receipt = prepare_window(source, first, width=480, height=480, duration="2")
    raw = (first / "window.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_window(source, first, width=480, height=480, start="1", duration="2")
    assert (first / "window.json").read_bytes() == raw
    prepare_window(source, other, width=480, height=480, start="1", duration="2")
    with pytest.raises(ValueError, match="exact source window"):
        export_frames(other / "window.mp4", tmp_path / "wrong-pins", receipt, [0])
    with pytest.raises(ValueError, match="exact source window"):
        finish_window(
            first / "window.mp4", tmp_path / "wrong.mp4", receipt, other / "window.mp4"
        )
    assert not (tmp_path / "wrong.mp4").exists()
    assert (
        finish_window(
            first / "window.mp4", tmp_path / "valid.mp4", receipt, first / "window.mp4"
        )["frames"]
        == 48
    )


def test_unpinned_high_resolution_photo_keeps_its_displayed_shape_and_old_file(tmp_path):
    from PIL import Image

    from vflash.media.remake_footage import prepare_appearance

    photo = tmp_path / "wide.jpg"
    metadata = Image.Exif()
    metadata[274] = 6
    Image.new("RGB", (1600, 400), "blue").save(photo, exif=metadata)
    output = tmp_path / "appearance.png"
    receipt = {"width": 1440, "height": 1440}
    result = prepare_appearance(photo, output, receipt)
    assert result["content_rect"] == [208, 0, 1024, 4096]
    with Image.open(output) as rendered:
        assert rendered.size == (1440, 4096)
    with pytest.raises(FileExistsError):
        prepare_appearance(photo, output, receipt)
