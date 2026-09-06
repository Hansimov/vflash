from __future__ import annotations

import json
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
