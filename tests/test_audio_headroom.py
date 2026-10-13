from types import SimpleNamespace

import pytest
import torch

from vflash.media.audio_headroom import audio_decoder_headroom


class BigVGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_post = torch.nn.Identity()
        self.use_tanh_at_final = False

    def forward(self, value):
        return self.conv_post(value).clamp(-1, 1)


@pytest.mark.parametrize("peak", [0.0, 0.2, 0.99, 2.0])
def test_headroom_precedes_clamp_and_links_stereo_without_boost(peak):
    decoder = BigVGAN()
    component = SimpleNamespace(module=SimpleNamespace(model=SimpleNamespace(decoder=decoder)))
    values = torch.tensor([[[peak, -peak / 2]], [[-peak / 4, peak / 8]]])
    observation = {}
    with audio_decoder_headroom(component, observation):
        result = decoder(values)
    gain = min(1.0, 0.98 / peak) if peak else 1.0
    assert torch.allclose(result, values * gain)
    assert observation["gain"] <= 1 and observation["linked_stereo"]
    assert torch.equal(decoder(values), values.clamp(-1, 1))
    assert not decoder.conv_post._forward_hooks


def test_invalid_or_aborted_audio_restores_decoder_hook():
    decoder = BigVGAN()
    component = SimpleNamespace(module=SimpleNamespace(model=SimpleNamespace(decoder=decoder)))
    with (
        pytest.raises(ValueError, match="nonfinite"),
        audio_decoder_headroom(component, {}),
    ):
        decoder(torch.full((2, 1, 8), float("nan")))
    assert not decoder.conv_post._forward_hooks
    with (
        pytest.raises(RuntimeError, match="decode stopped"),
        audio_decoder_headroom(component, {}),
    ):
        raise RuntimeError("decode stopped")
    assert not decoder.conv_post._forward_hooks


def test_diagnostic_sees_unclipped_values_and_missed_decode_fails():
    decoder = BigVGAN()
    component = SimpleNamespace(module=SimpleNamespace(model=SimpleNamespace(decoder=decoder)))
    values = torch.tensor([[[2.0, -1.5]], [[-0.5, 0.1]]])
    seen = []
    with audio_decoder_headroom(component, {}, on_unclamped=lambda x: seen.append(x.clone())):
        decoder(values)
    assert len(seen) == 1 and torch.equal(seen[0], values)
    with (
        pytest.raises(ValueError, match="did not reach"),
        audio_decoder_headroom(component, {}),
    ):
        pass
    assert not decoder.conv_post._forward_hooks


def test_duplicate_decode_and_callback_failure_restore_hook():
    decoder = BigVGAN()
    component = SimpleNamespace(module=SimpleNamespace(model=SimpleNamespace(decoder=decoder)))
    values = torch.zeros((2, 1, 8))
    with (
        pytest.raises(ValueError, match="one complete"),
        audio_decoder_headroom(component, {}),
    ):
        decoder(values)
        decoder(values)

    def fail(value):
        raise RuntimeError("diagnostic failed")

    with (
        pytest.raises(RuntimeError, match="diagnostic failed"),
        audio_decoder_headroom(component, {}, on_unclamped=fail),
    ):
        decoder(values)
    assert not decoder.conv_post._forward_hooks
