import errno
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vflash.media.encoding import MediaError
from vflash.media.rgb_array import create_rgb_array
from vflash.media.rgb_delivery import generate_rgb, validate_rgb_selection


@pytest.fixture
def decoder(monkeypatch):
    from vflash.media import runtime

    pixels = torch.linspace(-0.1, 1.1, 3 * 124 * 32 * 32).reshape(1, 3, 124, 32, 32)
    monkeypatch.setattr(
        runtime,
        "load_safetensor_tensors",
        lambda *args: {"video_latents": torch.zeros(1, 24, 37, 1, 1)},
    )
    monkeypatch.setattr(runtime, "decode_official_h3_video_latents", lambda *args, **kw: pixels)
    for name in ["reset_peak_memory_stats", "synchronize"]:
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
    value = SimpleNamespace(
        _torch=torch,
        device=torch.device("cpu"),
        video=object(),
        _require_open=lambda: None,
        _cuda_active=True,
        _closed=False,
    )
    return value, pixels


def options():
    return dict(model_frames=124, width=32, height=32, duration_seconds=5)


def test_rgb_uses_original_quantization_without_mp4_or_audio(tmp_path, decoder):
    value, pixels = decoder
    path = tmp_path / "output.npy"
    result = generate_rgb(value, tmp_path / "latent", path, **options())
    expected = (
        (pixels[0].permute(1, 2, 3, 0).float() * 255).round().clamp(0, 255).byte().numpy()
    )
    assert np.array_equal(np.load(path), expected)
    assert result.media["format"] == "rgb8-npy"
    assert result.media["model_frames"] == 124 and result.media["frames"] == 120
    assert result.media["complete_model_timeline"] and result.media["audio_samples"] == 0
    assert not result.media["intermediate_video_encoded"]


def test_rejects_nonfinite_and_preserves_existing_files(tmp_path, decoder):
    value, pixels = decoder
    path = tmp_path / "existing.npy"
    path.write_bytes(b"saved")
    with pytest.raises(MediaError, match="already exists"):
        generate_rgb(value, tmp_path / "latent", path, **options())
    assert path.read_bytes() == b"saved"
    path = tmp_path / "new.npy"
    pixels[0, 0, 17, 0, 0] = float("nan")
    with pytest.raises(MediaError, match="nonfinite"):
        generate_rgb(value, tmp_path / "latent", path, **options())
    assert not path.exists()


@pytest.mark.parametrize("frames", [(), (1, 0), (0, 0), (124,), (False,), [0]])
def test_selection_is_explicit_and_matches_native_timeline(frames):
    with pytest.raises(MediaError, match="required_frames"):
        validate_rgb_selection(124, frames)


def test_array_exclusive_creation_preserves_symlinks_and_saved_data(tmp_path):
    saved = tmp_path / "saved.npy"
    saved.write_bytes(b"saved")
    link = tmp_path / "link.npy"
    link.symlink_to(saved)
    for p in [saved, link]:
        with pytest.raises(FileExistsError):
            create_rgb_array(p, shape=(2, 3, 4, 3))
    assert saved.read_bytes() == b"saved" and link.is_symlink()


def test_reservation_failure_is_recoverable_before_page_write(tmp_path, monkeypatch):
    import vflash.media.rgb_array as module

    def full(*args):
        raise OSError(errno.ENOSPC, "test capacity exhausted")

    monkeypatch.setattr(module.os, "posix_fallocate", full)
    target = tmp_path / "out.npy"
    with pytest.raises(OSError) as error:
        create_rgb_array(target, shape=(2, 3, 4, 3))
    assert error.value.errno == errno.ENOSPC and not target.exists()


def test_unsupported_reservation_uses_real_writes(tmp_path, monkeypatch):
    import vflash.media.rgb_array as module

    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, "test unsupported")

    monkeypatch.setattr(module.os, "posix_fallocate", unsupported)
    target = tmp_path / "out.npy"
    a = create_rgb_array(target, shape=(2, 3, 4, 3))
    a[:] = 42
    a.flush()
    a._mmap.close()
    assert np.all(np.load(target) == 42)
