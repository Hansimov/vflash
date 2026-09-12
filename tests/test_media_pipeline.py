from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vflash.adapters.official_vae import (  # noqa: E402
    H3NativeVAEError,
    load_h3_vae_config,
    load_official_h3_vae_component,
)
from vflash.media.audio_delivery import _loudness_measurement  # noqa: E402
from vflash.media.encoding import MediaError, encode_mp4  # noqa: E402
from vflash.media.runtime import OfficialMediaDecoder  # noqa: E402
from vflash.pipeline.residency import capture_cpu_master, restore_cpu_master  # noqa: E402


def test_cpu_master_restores_parameters_and_nonpersistent_buffers() -> None:
    module = torch.nn.Linear(3, 2)
    module.register_buffer("offset", torch.ones(2), persistent=False)
    weight, offset = module.weight.detach(), module.offset
    master = capture_cpu_master(module)
    module.to_empty(device="meta")
    restore_cpu_master(module, master)
    assert module.weight.data_ptr() == weight.data_ptr()
    assert module.offset.data_ptr() == offset.data_ptr()
    assert master.tensor_bytes == (6 + 2 + 2) * 4


def _vae_fixture(path: Path, *, std: float = 1.0) -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {"AutoModel": "entry.TestVAE"},
                "latent_channels": 2,
                "latents_mean": [0.0, 0.0],
                "latents_std": [std, std],
            }
        )
    )
    (path / "entry.py").write_text(
        "import torch\n"
        "class TestVAE(torch.nn.Module):\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, path):\n"
        "        model = cls()\n"
        "        model.weight = torch.nn.Parameter(torch.tensor([3., 4.]))\n"
        "        model.register_buffer('scale', torch.ones(1), persistent=False)\n"
        "        return model\n"
    )


def test_official_adapter_requires_explicit_trust_and_does_not_patch_torch(
    tmp_path: Path,
) -> None:
    _vae_fixture(tmp_path)
    with pytest.raises(H3NativeVAEError, match="trust_local_code"):
        load_official_h3_vae_component(tmp_path, torch_module=torch)
    register = torch.nn.Module.register_parameter
    loader = torch.nn.Module.load_state_dict
    component = load_official_h3_vae_component(
        tmp_path,
        torch_module=torch,
        trust_local_code=True,
    )
    assert torch.nn.Module.register_parameter is register
    assert torch.nn.Module.load_state_dict is loader
    assert component.module.weight.device.type == "cpu"
    torch.testing.assert_close(component.module.weight, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(component.module.scale, torch.ones(1))


@pytest.mark.parametrize("std", [float("nan"), float("inf"), 0, -1])
def test_invalid_normalization_is_rejected_before_local_code(
    tmp_path: Path, std: float
) -> None:
    _vae_fixture(tmp_path, std=std)
    with pytest.raises(H3NativeVAEError, match="latents_std"):
        load_h3_vae_config(tmp_path)


def _media() -> tuple[object, object]:
    video = (
        torch.linspace(0, 1, 24 * 16 * 32).reshape(1, 1, 24, 16, 32).expand(1, 3, -1, -1, -1)
    )
    audio = torch.sin(torch.arange(32000) * 0.04).reshape(1, 1, -1).expand(1, 2, -1) * 0.1
    return video, audio


def _delivery_media(*, peaked: bool = False) -> tuple[object, object]:
    seconds, sample_rate = 3, 32000
    video = torch.zeros(1, 3, seconds * 24, 16, 32)
    clock = torch.arange(seconds * sample_rate) / sample_rate
    audio = (0.03 * torch.sin(2 * math.pi * 440 * clock)).repeat(2, 1).unsqueeze(0)
    if peaked:
        audio[:, :, 8000::16000] = 1.25
    return video, audio


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg")
def test_real_mp4_has_expected_clocks_and_preserves_existing_output(tmp_path: Path) -> None:
    video, audio = _media()
    target = tmp_path / "output.mp4"
    result = encode_mp4(video, audio, target, preset="ultrafast")
    assert result["frames"] == 24
    assert result["audio_samples"] == 32000
    assert result["duration_seconds"] == 1
    before = target.read_bytes()
    with pytest.raises(MediaError, match="already exists"):
        encode_mp4(video, audio, target)
    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output.mp4"]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg")
def test_default_audio_delivery_does_not_invoke_web_processing(
    tmp_path: Path, monkeypatch
) -> None:
    video, audio = _media()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("default delivery invoked web processing")

    monkeypatch.setattr("vflash.media.encoding.web_loudness_filter", unexpected)
    default_path = tmp_path / "default.mp4"
    explicit_path = tmp_path / "unchanged.mp4"
    result = encode_mp4(video, audio, default_path, preset="ultrafast")
    encode_mp4(
        video,
        audio,
        explicit_path,
        preset="ultrafast",
        audio_delivery_profile="unchanged",
    )
    assert "audio_delivery" not in result
    assert default_path.read_bytes() == explicit_path.read_bytes()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg")
@pytest.mark.parametrize("peaked,expected_mode", [(False, "linear"), (True, "limited")])
def test_web_audio_delivery_applies_bounded_gain_and_peak_control(
    tmp_path: Path, peaked: bool, expected_mode: str
) -> None:
    video, audio = _delivery_media(peaked=peaked)
    output = tmp_path / f"web-{expected_mode}.mp4"
    result = encode_mp4(
        video,
        audio,
        output,
        preset="ultrafast",
        audio_delivery_profile="web-v1",
    )
    processing = result["audio_delivery"]
    assert processing["status"] == "normalized"
    assert processing["mode"] == expected_mode
    assert processing["gain_db"] <= 18
    measured, _ = _loudness_measurement("ffmpeg", ["-i", str(output)])
    if expected_mode == "linear":
        assert measured["input_i"] == pytest.approx(-18, abs=1.2)
    else:
        assert processing["input_true_peak_dbtp"] > 0
        # AAC can introduce a small inter-sample overshoot after limiting.
        assert measured["input_tp"] <= processing["true_peak_limit_dbtp"] + 1


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg")
def test_web_audio_delivery_bypasses_audio_below_the_quiet_floor(tmp_path: Path) -> None:
    video, audio = _delivery_media()
    audio.mul_(1e-4)
    output = tmp_path / "quiet.mp4"
    result = encode_mp4(
        video,
        audio,
        output,
        preset="ultrafast",
        audio_delivery_profile="web-v1",
    )
    assert result["audio_delivery"]["status"] == "quiet_bypass"
    measured, _ = _loudness_measurement("ffmpeg", ["-i", str(output)])
    assert measured["input_i"] < -55


def test_failed_encoding_leaves_no_partial_output(tmp_path: Path) -> None:
    video, audio = _media()
    video = video.clone()
    video[0, 0, 9, 0, 0] = float("nan")
    with pytest.raises(MediaError, match="nonfinite"):
        encode_mp4(video, audio, tmp_path / "output.mp4")
    assert list(tmp_path.iterdir()) == []


def test_close_fences_resources_before_retirement_and_can_retry() -> None:
    decoder = OfficialMediaDecoder.__new__(OfficialMediaDecoder)
    decoder.video = object()
    decoder.audio = object()
    decoder._video_master = object()
    decoder._audio_master = object()
    decoder._cuda_touched = decoder._cuda_active = True
    decoder._closed = decoder._released = False
    decoder.device = "cuda:0"
    calls = []

    def failed_fence(device: str) -> None:
        calls.append(device)
        raise RuntimeError("CUDA completion unavailable")

    decoder._torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=failed_fence))
    with pytest.raises(RuntimeError, match="completion unavailable"):
        decoder.close()
    assert decoder._closed and not decoder._released
    assert decoder.video is not None and decoder._video_master is not None
    with pytest.raises(MediaError, match="closed"):
        decoder.resume_cuda()
    decoder._torch.cuda.synchronize = calls.append
    decoder.close()
    decoder.close()
    assert calls == ["cuda:0", "cuda:0"]
    assert decoder._released and decoder.video is None and decoder._video_master is None


def test_media_decoder_delivers_the_exact_ten_second_audio_video_window(
    tmp_path, monkeypatch
) -> None:
    decoder = OfficialMediaDecoder.__new__(OfficialMediaDecoder)
    decoder._closed = False
    decoder._cuda_active = True
    decoder.device = "cuda:0"
    decoder.audio_sample_rate = 32000
    decoder._torch = SimpleNamespace(
        cuda=SimpleNamespace(
            reset_peak_memory_stats=lambda _device: None,
            max_memory_allocated=lambda _device: 123,
        )
    )
    decoded_video = torch.zeros(1, 3, 243, 2, 2)
    decoded_audio = torch.zeros(1, 2, 320512)
    decoder._decode_cpu = lambda *_args, **_kwargs: (
        decoded_video,
        decoded_audio,
        {"decode": 1.0},
    )
    observed = {}

    def encode(video, audio, output, **kwargs):
        observed.update(
            video_frames=int(video.shape[2]),
            audio_samples=int(audio.shape[2]),
            output=output,
            **kwargs,
        )
        return {"frames": int(video.shape[2]), "audio_samples": int(audio.shape[2])}

    monkeypatch.setattr("vflash.media.runtime.encode_mp4", encode)
    result = decoder.generate_mp4(
        tmp_path / "latents.safetensors",
        tmp_path / "ten-seconds.mp4",
        height=32,
        width=32,
        duration_seconds=10,
    )
    assert observed == {
        "video_frames": 240,
        "audio_samples": 320000,
        "output": tmp_path / "ten-seconds.mp4",
        "fps": 24,
        "audio_sample_rate": 32000,
    }
    assert result.peak_allocated_bytes == 123

    observed.clear()
    decoder.generate_mp4(
        tmp_path / "latents.safetensors",
        tmp_path / "ten-seconds-web.mp4",
        height=32,
        width=32,
        duration_seconds=10,
        audio_delivery_profile="web-v1",
    )
    assert observed["audio_delivery_profile"] == "web-v1"
