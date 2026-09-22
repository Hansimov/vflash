import sys
from types import SimpleNamespace

import pytest

from vflash.adapters.stcdit_attention import configure_attention


def test_default_leaves_native_attention_and_optional_dependencies_alone(monkeypatch):
    original = object()
    module = SimpleNamespace(flash_attention=original)
    monkeypatch.setitem(sys.modules, "sageattention", None)
    configure_attention([module], "torch")
    assert module.flash_attention is original
    with pytest.raises(ValueError, match="attention"):
        configure_attention([module], "automatic")
    with pytest.raises(RuntimeError, match="ABI-compatible"):
        configure_attention([module], "sage-int8-fp16")
    assert module.flash_attention is original


def test_only_supplied_restoration_modules_are_changed(monkeypatch):
    original = object()
    modules = [SimpleNamespace(flash_attention=original) for _ in range(3)]
    monkeypatch.setitem(
        sys.modules,
        "sageattention",
        SimpleNamespace(sageattn_qk_int8_pv_fp16_triton=lambda *a, **k: None),
    )
    configure_attention(modules[:2], "sage-int8-fp16")
    assert modules[0].flash_attention is modules[1].flash_attention
    assert modules[2].flash_attention is original


@pytest.mark.parametrize(
    "capability,maximum", [((8, 6), 1), ((8, 9), 65504), ((8, 9), float("nan"))]
)
def test_unsupported_gpu_or_values_fail_before_quantized_kernel(
    monkeypatch, capability, maximum
):
    import math

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=lambda _: capability),
            isfinite=math.isfinite,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sageattention",
        SimpleNamespace(
            sageattn_qk_int8_pv_fp16_triton=lambda *a, **k: pytest.fail("unsafe kernel call"),
        ),
    )
    module = SimpleNamespace()
    configure_attention([module], "sage-int8-fp16")
    query = SimpleNamespace(is_cuda=True, device="cuda:0")
    value = SimpleNamespace(abs=lambda: SimpleNamespace(amax=lambda: maximum))
    with pytest.raises(ValueError):
        module.flash_attention(query, query, value, num_heads=12)
