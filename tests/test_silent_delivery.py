"""Explicit silence changes only delivery, never the native AV trajectory."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vflash.media.encoding import MediaError, encode_mp4  # noqa: E402
from vflash.media.runtime import OfficialMediaDecoder  # noqa: E402
from vflash.pipeline.contracts import VideoRequest  # noqa: E402


def test_silence_requires_an_explicit_request_parameter():
    assert VideoRequest("An entirely silent scene.").audio_delivery_profile == "unchanged"
    assert VideoRequest("A scene.", audio_delivery_profile="silent-v1").mode == "t2va"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg")
def test_real_silent_aac_preserves_every_video_frame_and_the_callers_audio(tmp_path):
    video = torch.linspace(0, 1, 24 * 16 * 32).reshape(1, 1, 24, 16, 32)
    video = video.expand(1, 3, -1, -1, -1)
    audio = torch.sin(torch.arange(32000) * 0.04).reshape(1, 1, -1).repeat(1, 2, 1)
    original = audio.clone()
    control, silent = tmp_path / "control.mp4", tmp_path / "silent.mp4"
    encode_mp4(video, audio, control, preset="ultrafast")
    result = encode_mp4(
        video, audio, silent, preset="ultrafast", audio_delivery_profile="silent-v1"
    )

    def read(path, *options):
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), *options, "pipe:1"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout

    samples = read(silent, "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le")
    assert samples and not any(samples)  # Every delivered decoded sample is digital zero.
    # Bitstream equality is stronger than a visual similarity score here. No hashes.
    options = ("-map", "0:v:0", "-c", "copy", "-f", "h264")
    assert read(silent, *options) == read(control, *options)
    assert torch.equal(audio, original)
    assert result["audio_delivery"] == {"profile": "silent-v1", "status": "silent"}
    assert result["audio_samples"] == 32000 and result["frames"] == 24
    assert sorted(p.name for p in tmp_path.iterdir()) == ["control.mp4", "silent.mp4"]


def test_silent_decode_does_not_read_or_decode_audio_latents(monkeypatch):
    decoder = OfficialMediaDecoder.__new__(OfficialMediaDecoder)
    decoder._torch, decoder.device = torch, "unused"
    decoder.video = object()
    requested = []
    latent = torch.zeros(1, 4, 2, 4, 4)
    pixels = torch.rand(1, 3, 124, 32, 64)

    def load(_path, names):
        requested.append(names)
        return {"video_latents": latent}

    def decode(component, tensor, **_kwargs):
        assert component is decoder.video and tensor is latent
        return pixels

    def unexpected(*_args, **_kwargs):
        raise AssertionError("explicit silence decoded the unused audio")

    monkeypatch.setattr("vflash.media.runtime.load_safetensor_tensors", load)
    monkeypatch.setattr("vflash.media.runtime.decode_official_h3_video_latents", decode)
    monkeypatch.setattr("vflash.media.runtime.decode_official_h3_audio_latents", unexpected)
    video, audio, stages = decoder._decode_cpu(
        Path("latents.safetensors"),
        height=32,
        width=64,
        silent_samples=160000,
    )
    assert requested == [("video_latents",)]
    assert torch.equal(video, pixels)
    assert audio.shape == (1, 2, 160000) and torch.count_nonzero(audio).item() == 0
    assert stages["audio_decode_and_copy"] == 0


def test_silence_does_not_hide_corrupt_decoded_media(tmp_path):
    video = torch.zeros(1, 3, 24, 16, 32)
    audio = torch.full((1, 2, 32000), float("nan"))
    with pytest.raises(MediaError, match="nonfinite"):
        encode_mp4(video, audio, tmp_path / "invalid.mp4", audio_delivery_profile="silent-v1")
    assert not list(tmp_path.iterdir())
