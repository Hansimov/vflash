"""Address arithmetic regression checks; no CUDA allocation or model is needed."""

from __future__ import annotations

import itertools
import math
import sys
from types import SimpleNamespace

import pytest

from vflash.native import h3_fused_ops as fused


class MetadataTensor:
    """Only the metadata read by the real fused-kernel admission and launchers."""

    _ids = itertools.count(1)
    device = SimpleNamespace(type="cuda")
    dtype = "bf16"
    requires_grad = False

    def __init__(self, shape, strides=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self._strides = strides or tuple(
            math.prod(self.shape[index + 1 :]) for index in range(self.ndim)
        )
        self._storage_id = next(self._ids)

    def numel(self):
        return math.prod(self.shape)

    def stride(self, index=None):
        return self._strides if index is None else self._strides[index]

    def is_contiguous(self):
        return self._strides == tuple(
            math.prod(self.shape[index + 1 :]) for index in range(self.ndim)
        )

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self._storage_id)


def test_modulation_stride_crosses_limit_before_logical_size():
    # The six AdaLN rows are unbound views, not contiguous [tokens, hidden].
    small = MetadataTensor((66_577, 5_376), (32_256, 1))
    large = MetadataTensor((66_578, 5_376), (32_256, 1))
    assert large.numel() < 2**31
    assert not fused._use_wide_index(small)
    assert fused._use_wide_index(large)
    actual_offset = 66_577 * 32_256
    signed_i32 = (actual_offset + 2**31) % 2**32 - 2**31
    assert actual_offset > 2**31 - 1
    assert signed_i32 < 0
    assert (signed_i32 - actual_offset) * 2 == -(2**33)


def test_packed_ffn_address_can_overflow_with_small_output():
    packed = MetadataTensor((1, 80_000, 28_672))
    output = MetadataTensor((1, 80_000, 14_336))
    assert not fused._use_wide_index(output)
    assert fused._use_wide_index(packed)


def test_small_strided_and_contiguous_views_keep_narrow_indices():
    assert not fused._use_wide_index(
        MetadataTensor((2, 128, 64)), MetadataTensor((128, 64), (384, 1))
    )
    assert not fused._use_wide_index(MetadataTensor((2**31 - 1,)))
    assert fused._use_wide_index(MetadataTensor((2**31,)))


@pytest.fixture
def launches(monkeypatch):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    kernel = Kernel()
    triton = SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d)
    torch = SimpleNamespace(
        Tensor=MetadataTensor,
        bfloat16="bf16",
        empty=lambda shape, **_: MetadataTensor(shape),
        empty_like=lambda tensor: MetadataTensor(tensor.shape),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", triton)
    for factory in ("_strict_bf16_kernels", "_strict_bf16_adapter_kernels"):
        monkeypatch.setattr(fused, factory, lambda: (triton, kernel, kernel))
    for factory in ("_strict_bf16_silu_mul_kernel", "_strict_bf16_rotary_kernel"):
        monkeypatch.setattr(fused, factory, lambda: (triton, kernel))
    for factory in (
        "_strict_bf16_ffn_adapter_silu_kernel",
        "_strict_bf16_qkv_direct_merge_kernel",
    ):
        monkeypatch.setattr(fused, factory, lambda: kernel)
    return calls


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        "modulate",
        "gate",
        "silu",
        "ffn_adapter",
        "merge",
        "qkv",
        "direct_qkv",
        "adapter_gate",
        "rotary",
    ],
)
def test_real_launchers_select_width_from_all_accessed_tensors(launches, operation, wide):
    tokens = 120_000 if wide else 16
    hidden = 5_376
    value = MetadataTensor((1, tokens, hidden))
    modifier = MetadataTensor((tokens, hidden), (6 * hidden, 1))
    packed = MetadataTensor((1, tokens, 28_672))
    qkv = MetadataTensor((1, tokens, 21_504))
    if operation == "modulate":
        fused.triton_strict_bf16_modulate(value, modifier, modifier)
    elif operation == "gate":
        fused.triton_strict_bf16_gate_residual(value, modifier, value)
    elif operation == "silu":
        fused.triton_strict_bf16_silu_mul(packed)
    elif operation == "ffn_adapter":
        fused.triton_strict_bf16_ffn_adapter_silu(
            packed, packed, scaling=0.0625, block_size=1024
        )
    elif operation == "merge":
        fused.triton_strict_bf16_adapter_merge(packed, packed, scaling=0.0625)
    elif operation == "qkv":
        fused.triton_strict_bf16_qkv_adapter_merge(qkv, qkv, scalings=(0.0625,) * 3)
    elif operation == "direct_qkv":
        adapters = tuple(MetadataTensor((1, tokens, 7_168)) for _ in range(3))
        fused.triton_strict_bf16_qkv_direct_merge(qkv, adapters, scalings=(0.0625,) * 3)
    elif operation == "adapter_gate":
        fused.triton_strict_bf16_adapter_gate_residual(
            value, value, value, modifier, scaling=0.0625
        )
    else:
        states = MetadataTensor((1, tokens, 56, 128))
        rotation = MetadataTensor((tokens, 96), (32_256, 1))
        fused.triton_strict_bf16_rotary(states, rotation, rotation)
    assert len(launches) == 1
    assert launches[0][2]["WIDE_INDEX"] is wide
