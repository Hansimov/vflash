"""CPU layout/reference and independent-owner checks; not GPU/media evidence."""

import pytest

torch = pytest.importorskip("torch")
from vflash.native.h3_fp8_linear import (  # noqa: E402
    RowFP8Execution,
    RowFP8Linear,
    _quantize_activation_torch,
)


def weights():
    codes = torch.arange(128).reshape(16, 8).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.01, 0.1, 16).unsqueeze(1)
    return codes, scales


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8), (0, 8)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_weight_only_is_exact_restored_weight_expression(shape, dtype):
    codes, scales = weights()
    values = torch.randn(shape, dtype=dtype)
    bias = torch.randn(16, dtype=dtype)
    execution = RowFP8Execution()
    actual = execution.linear(values, codes, scales, bias)
    restored = (codes.float() * scales).to(dtype)
    expected = torch.nn.functional.linear(values, restored, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert execution.calls == {"weight_only": 1, "w8a8": 0}


def test_module_assigns_weight_scales_without_bias_copy():
    codes, scales = weights()
    original = torch.nn.Linear(8, 16)
    execution = RowFP8Execution()
    module = RowFP8Linear(original, execution)
    assert module.bias is original.bias and module.weight.is_meta and module.row_scale.is_meta
    module.load_state_dict(
        {"weight": codes, "row_scale": scales, "bias": original.bias}, assign=True
    )
    value = torch.randn(2, 8)
    torch.testing.assert_close(
        module(value), execution.linear(value, codes, scales, module.bias), rtol=0, atol=0
    )
    assert not module.weight.requires_grad


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_w8a8_cpu_fails_without_loading_triton_or_fallback(backend):
    codes, scales = weights()
    execution = RowFP8Execution("w8a8", activation_quantizer=backend)
    with pytest.raises(ValueError, match="Hopper"):
        execution.linear(torch.ones(3, 8, dtype=torch.bfloat16), codes, scales)
    assert execution.calls == {"weight_only": 0, "w8a8": 0}


@pytest.mark.parametrize(
    "bad", ["codes", "scale_dtype", "scale_shape", "input_shape", "rank", "bias"]
)
def test_bad_layout_never_enters_gemm(bad, monkeypatch):
    codes, scales = weights()
    value = torch.ones(3, 8)
    bias = None
    if bad == "codes":
        codes = codes.float()
    elif bad == "scale_dtype":
        scales = scales.bfloat16()
    elif bad == "scale_shape":
        scales = scales.T
    elif bad == "input_shape":
        value = torch.ones(3, 7)
    elif bad == "rank":
        codes = codes.flatten()
    elif bad == "bias":
        bias = torch.ones(15)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid layout reached GEMM")

    monkeypatch.setattr(torch.nn.functional, "linear", forbidden)
    with pytest.raises(ValueError, match="declared"):
        RowFP8Execution().linear(value, codes, scales, bias)


def test_executions_keep_precision_observation_and_counts_independent():
    codes, scales = weights()
    a, b = RowFP8Execution(), RowFP8Execution("w8a8")
    observed = []
    a.observer = lambda: observed.append(dict(a.calls))
    a.linear(torch.ones(8), codes, scales)
    assert observed == [{"weight_only": 0, "w8a8": 0}]
    assert b.mode == "w8a8" and b.calls == {"weight_only": 0, "w8a8": 0}
    with pytest.raises(ValueError):
        a.select("silent-fallback")
    assert a.mode == "weight-only"


@pytest.mark.parametrize("rows", [1, 16, 17])
def test_activation_reference_pads_zeros_and_respects_strided_rows(rows):
    # The value half of an FFN projection has a doubled physical row stride.
    base = torch.full((rows, 128), float("nan"), dtype=torch.bfloat16)
    values = base[:, :64]
    values.copy_(torch.linspace(-448, 448, 64))
    values[0].zero_()
    if rows > 1:
        values[1].fill_(torch.finfo(torch.bfloat16).tiny)
    codes, scales = _quantize_activation_torch(values)
    padded = (rows + 15) // 16 * 16
    assert codes.shape == (padded, 64) and scales.shape == (padded, 1)
    assert codes.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert torch.isfinite(codes.float()).all() and (scales > 0).all()
    assert (codes[0].float() == 0).all() and scales[0].item() == 1
    assert (codes[rows:].float() == 0).all() and (scales[rows:] == 1).all()
    assert torch.isnan(base[:, 64:].float()).all()  # input/guard lanes were not changed
    expected_scale = (values.float().abs().amax(dim=1, keepdim=True) / 448).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    expected_scale[0] = 1
    assert torch.equal(scales[:rows], expected_scale)
