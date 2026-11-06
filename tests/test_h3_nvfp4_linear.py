# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")
from vflash.native import h3_nvfp4_linear as nv


def weights(scale=4.0):
    return nv.NVFP4Weight(
        torch.zeros((128, 32), dtype=torch.uint8),
        torch.zeros((128, 4), dtype=torch.uint8),
        torch.tensor(2.0, dtype=torch.float32),
        torch.tensor(scale, dtype=torch.float32),
    )


def test_calibrated_input_scale_and_weight_alpha_reach_the_same_kernel(monkeypatch):
    seen = {}
    w = weights()

    def quantize(x, sf, **kwargs):
        seen["input"] = x
        assert sf is w.input_scale
        assert kwargs == {"sfLayout": "128x4", "do_shuffle": False}
        return torch.zeros((6, 32), dtype=torch.uint8), torch.zeros((128, 4), dtype=torch.uint8)

    def matmul(a, b, a_scale, b_scale, alpha, dtype, out, **kwargs):
        assert b.data_ptr() == w.packed.data_ptr() and b.shape == (32, 128)
        assert b_scale.data_ptr() == w.scales.data_ptr() and b_scale.shape == (4, 128)
        assert dtype == torch.bfloat16 and out is None
        assert alpha.item() == 0.5
        assert kwargs == dict(
            block_size=16, use_8x4_sf_layout=False, backend="cutlass", use_nvfp4=True
        )
        return torch.full((6, 128), 3.0, dtype=torch.bfloat16)

    monkeypatch.setattr(nv, "_require_device", lambda _: None)
    monkeypatch.setattr(nv, "_operators", lambda: ("128x4", quantize, matmul))
    engine = nv.NVFP4Execution()
    value = torch.ones((2, 3, 64), dtype=torch.float32)
    output = engine.linear(value, w, torch.ones(128, dtype=torch.bfloat16))
    assert seen["input"].dtype == torch.bfloat16
    assert output.shape == (2, 3, 128) and torch.equal(output, torch.full_like(output, 4.0))
    assert engine.calls == 1 and engine.rows == 6


def test_unsupported_device_refuses_before_optional_backend_import(monkeypatch):
    def unexpected():
        raise AssertionError("backend import should not happen")

    monkeypatch.setattr(nv, "_operators", unexpected)
    with pytest.raises(ValueError, match="SM120"):
        nv.NVFP4Execution().linear(torch.ones((1, 64), dtype=torch.bfloat16), weights())


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_calibration_cannot_become_a_valid_model(scale):
    with pytest.raises(ValueError, match="positive and finite"):
        weights(scale)


def test_unpadded_quantizer_does_not_silently_select_another_arithmetic(monkeypatch):
    monkeypatch.setattr(nv, "_require_device", lambda _: None)

    def quantize(*args, **kwargs):
        return torch.zeros((3, 32), dtype=torch.uint8), torch.zeros((3, 4), dtype=torch.uint8)

    def unexpected(*args, **kwargs):
        raise AssertionError("kernel must not consume invalid scale layout")

    monkeypatch.setattr(nv, "_operators", lambda: ("128x4", quantize, unexpected))
    engine = nv.NVFP4Execution()
    with pytest.raises(ValueError, match="scale padding"):
        engine.linear(torch.ones((3, 64), dtype=torch.bfloat16), weights())
    assert engine.calls == 0
