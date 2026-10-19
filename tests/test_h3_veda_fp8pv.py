from types import SimpleNamespace

import pytest

from vflash.native.h3_veda_fp8pv import FP8PVBackend


def test_fp8pv_wraps_only_veda_triton_int8_and_delegates_the_rest():
    base = SimpleNamespace(name="triton-int8", tolerance=0.05, display="Triton INT8 (x)")
    wrapped = FP8PVBackend(base)
    assert wrapped.name == "triton-int8-fp8pv"
    assert wrapped.tolerance == 0.05 and wrapped.display == "Triton INT8 (x)"
    with pytest.raises(ValueError):
        FP8PVBackend(SimpleNamespace(name="mlx-gather"))
