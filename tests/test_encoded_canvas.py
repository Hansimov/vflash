import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from vflash.media.canvas import crop_geometry, crop_rgb
from vflash.media.encoding import MediaError, encode_mp4
from vflash.media.runtime import OfficialMediaDecoder

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "crop", [(1, 0, 0, 0), (False, 0, 0, 0), (-2, 0, 0, 0), (64, 0, 32, 0), [2, 0, 4, 0]]
)
def test_invalid_crop_is_rejected(crop):
    with pytest.raises(MediaError):
        crop_geometry(96, 160, crop)


def test_media_decoder_crops_only_the_rgb_pixels_before_encoding(tmp_path, monkeypatch):
    video = torch.arange(1 * 3 * 124 * 64 * 96, dtype=torch.float32).reshape(1, 3, 124, 64, 96)
    audio = torch.ones(1, 2, 160512)
    decoder = OfficialMediaDecoder.__new__(OfficialMediaDecoder)
    decoder._closed = False
    decoder._cuda_active = True
    decoder.device = "cuda:0"
    decoder.audio_sample_rate = 32000
    decoder._torch = SimpleNamespace(
        cuda=SimpleNamespace(
            reset_peak_memory_stats=lambda _: None, max_memory_allocated=lambda _: 123
        )
    )
    decoder._decode_cpu = lambda *a, **kw: (video, audio, {"decode": 1.0})
    calls = []

    def encode(v, a, out, **kwargs):
        calls.append((v, a))
        return {"frames": v.shape[2]}

    monkeypatch.setattr("vflash.media.runtime.encode_mp4", encode)
    result = decoder.generate_mp4(
        tmp_path / "latents",
        tmp_path / "out.mp4",
        width=96,
        height=64,
        duration_seconds=5,
        video_crop=(2, 2, 4, 4),
    )
    assert len(calls) == 1
    assert torch.equal(calls[0][0], video[:, :, :120, 2:60, 2:92])
    assert torch.equal(calls[0][1], audio[:, :, :160000])
    assert calls[0][0].untyped_storage().data_ptr() == video.untyped_storage().data_ptr()
    assert result.media["video_crop"] == crop_geometry(96, 64, (2, 2, 4, 4))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg")
def test_single_encoded_crop_preserves_audio_and_physical_dimensions(tmp_path):
    torch.set_num_threads(2)
    video = (
        torch.linspace(0, 1, 24 * 160 * 96).reshape(1, 1, 24, 160, 96).expand(1, 3, -1, -1, -1)
    )
    audio = torch.sin(torch.arange(32000) * 0.04).reshape(1, 1, -1).expand(1, 2, -1) * 0.1
    original = tmp_path / "full.mp4"
    cropped = tmp_path / "cropped.mp4"
    encode_mp4(video, audio, original, preset="ultrafast", video_threads=1)
    encode_mp4(
        crop_rgb(video, crop_geometry(96, 160, (2, 0, 4, 0))),
        audio,
        cropped,
        preset="ultrafast",
        video_threads=1,
    )
    waves = [
        subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(p),
                "-vn",
                "-c:a",
                "copy",
                "-f",
                "adts",
                "pipe:1",
            ]
        )
        for p in [original, cropped]
    ]
    assert waves[0] == waves[1]
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-select_streams",
                "v:0",
                "-of",
                "json",
                str(cropped),
            ]
        )
    )
    assert data["streams"][0] == {"width": 90, "height": 160, "nb_frames": "24"}


def test_declared_square_pixels_preserve_decoded_rgb_and_audio(tmp_path):
    video = (
        torch.linspace(0, 1, 24 * 160 * 90).reshape(1, 1, 24, 160, 90).expand(1, 3, -1, -1, -1)
    )
    audio = torch.sin(torch.arange(32000) * 0.04).reshape(1, 1, -1).expand(1, 2, -1) * 0.1
    original, declared = tmp_path / "original.mp4", tmp_path / "square.mp4"
    encode_mp4(video, audio, original, preset="ultrafast", video_threads=1, video_input="pipe")
    encode_mp4(
        video,
        audio,
        declared,
        preset="ultrafast",
        video_threads=1,
        video_input="pipe",
        square_pixels=True,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,sample_aspect_ratio",
                "-of",
                "json",
                str(declared),
            ]
        )
    )
    assert probe["streams"][0] == {"width": 90, "height": 160, "sample_aspect_ratio": "1:1"}

    def decoded(path):
        result = subprocess.check_output(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0", "-f", "framemd5", "-"]
        ).decode()
        return [line for line in result.splitlines() if not line.startswith("#")]

    assert decoded(original) == decoded(declared)


@pytest.mark.parametrize("width,height", [(1366, 768), (768, 1366), (1280, 720)])
def test_exact_nonstandard_display_ratio_has_unit_pixel_aspect(tmp_path, width, height):
    video = torch.zeros(1, 3, 2, height, width)
    target = tmp_path / "canvas.mp4"
    encode_mp4(
        video,
        None,
        target,
        square_pixels=True,
        video_threads=1,
        preset="ultrafast",
        video_input="pipe",
    )
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,sample_aspect_ratio",
                "-of",
                "json",
                str(target),
            ],
            timeout=20,
        )
    )["streams"][0]
    assert data == {"width": width, "height": height, "sample_aspect_ratio": "1:1"}
