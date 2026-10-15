from types import SimpleNamespace

import pytest

from vflash.media.face_encoder import face_encoder, frame_group_norm

torch = pytest.importorskip("torch")


class Encoder:
    config = SimpleNamespace(clip_length=17, token_drop=3)

    def _encode(self, x):
        return x[..., :1, :1]

    def _encode_clip(self, x):
        # Encode temporal frame identities without mixing independent clips.
        return x[:, :, ::4, :1, :1]


def test_batch_keeps_temporal_order_padding_and_exact_tail():
    torch.set_num_threads(2)
    vae = Encoder()
    x = torch.arange(124).float().view(1, 1, 124, 1, 1).expand(-1, -1, -1, 512, 512)
    with face_encoder(vae, "batch2-fp32") as report:
        actual = vae._encode(x).flatten().tolist()
    expected = [
        min(start + offset, 123) for start in range(0, 136, 17) for offset in (0, 4, 8, 12, 16)
    ][:-3]
    assert actual == expected
    assert report["temporal_clips"] == 8 and report["calls"] == 1
    assert "_encode" not in vae.__dict__


def test_scope_restores_existing_owner_on_failure():
    vae = Encoder()

    def previous(x):
        return x

    vae._encode = previous
    with pytest.raises(ValueError, match="one complete"), face_encoder(vae, "batch2-fp32"):
        vae._encode(torch.zeros(1, 1, 124, 2, 2))
    assert vae._encode is previous


def test_default_does_not_override_encoder_or_precision():
    vae = Encoder()
    x = torch.ones(1, 1, 1, 2, 2)
    with face_encoder(vae) as report:
        assert "_encode" not in vae.__dict__
        assert torch.equal(vae._encode(x), x[..., :1, :1])
    assert not report["approximate"]


def test_rejects_unknown_geometry_and_cpu_amp():
    vae = Encoder()
    with pytest.raises(ValueError, match="unknown"), face_encoder(vae, "fp8"):
        pass
    x = torch.zeros(1, 1, 124, 1, 1).expand(-1, -1, -1, 512, 512)
    with pytest.raises(ValueError, match="CUDA"), face_encoder(vae, "batch2-fp16"):
        vae._encode(x)
    vae.config = SimpleNamespace(clip_length=16, token_drop=3)
    with pytest.raises(ValueError, match="temporal geometry"), face_encoder(vae, "batch2-fp32"):
        pass


@pytest.mark.parametrize("channels_last", (False, True))
def test_half_norm_uses_separate_frames_and_preserves_original_fp32_parameters(channels_last):
    torch.manual_seed(10)
    module = torch.nn.GroupNorm(2, 4)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.317, 1.213, 4))
        module.bias.copy_(torch.linspace(-0.321, 0.419, 4))
    original = {k: v.clone() for k, v in module.state_dict().items()}
    x = torch.randn(2, 4, 3, 5, 7).half()
    x[:, :, 1] += 15
    expected = torch.stack([module(x[:, :, t].float()) for t in range(3)], dim=2).half()
    actual = frame_group_norm(module, x, channels_last=channels_last)
    assert actual.dtype == torch.float16 and module.weight.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
    changed = x.clone()
    changed[:, :, 1] *= 3
    second = frame_group_norm(module, changed, channels_last=channels_last)
    assert torch.equal(actual[:, :, 0], second[:, :, 0])
    assert torch.equal(actual[:, :, 2], second[:, :, 2])
    for name, value in module.state_dict().items():
        assert torch.equal(value, original[name])


def test_norm_overrides_and_parameters_survive_a_failed_context():
    class MiniMaxH3VideoGroupNorm(torch.nn.GroupNorm):
        pass

    vae = Encoder()
    norm = MiniMaxH3VideoGroupNorm(2, 4)
    vae.encoder = torch.nn.Sequential(norm)
    initial = {name: tensor.clone() for name, tensor in norm.state_dict().items()}
    with pytest.raises(ValueError, match="one complete"), face_encoder(vae, "batch2-fp16-norm"):
        assert "forward" in norm.__dict__
        vae._encode(torch.zeros(1, 4, 124, 2, 2))
    assert "forward" not in norm.__dict__ and "_encode" not in vae.__dict__
    for name, tensor in norm.state_dict().items():
        assert torch.equal(tensor, initial[name])


@pytest.mark.parametrize("initial_tiling", (False, True))
def test_whole_crop_restores_the_original_tile_policy_and_norms(initial_tiling):
    class MiniMaxH3VideoGroupNorm(torch.nn.GroupNorm):
        pass

    vae = Encoder()
    vae.use_tiling = initial_tiling
    norm = MiniMaxH3VideoGroupNorm(2, 4)
    vae.encoder = torch.nn.Sequential(norm)
    original = vae._encode
    with (
        pytest.raises(RuntimeError, match="failed crop"),
        face_encoder(vae, "batch2-fp16-norm-reuse-untiled") as report,
    ):
        assert not vae.use_tiling and report["spatial_context_changed"]
        assert report["approximate"] and "forward" in norm.__dict__
        raise RuntimeError("failed crop")
    assert vae.use_tiling is initial_tiling and vae._encode == original
    assert "forward" not in norm.__dict__


@pytest.mark.parametrize("frames,side", ((124, 384), (242, 512)))
def test_whole_crop_rejects_unqualified_geometry_before_cuda(frames, side):
    class MiniMaxH3VideoGroupNorm(torch.nn.GroupNorm):
        pass

    vae = Encoder()
    vae.use_tiling = True
    vae.encoder = torch.nn.Sequential(MiniMaxH3VideoGroupNorm(1, 1))
    pixels = torch.zeros(1, 1, 1, 1, 1).expand(1, 1, frames, side, side)
    with (
        pytest.raises(ValueError, match="124 frames at 512 square"),
        face_encoder(vae, "batch2-fp16-norm-reuse-untiled"),
    ):
        vae._encode(pixels)
    assert vae.use_tiling and "_encode" not in vae.__dict__
