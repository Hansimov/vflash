import gc
import weakref
from types import SimpleNamespace

import pytest

from vflash.native import h3_fused_ops as fused
from vflash.native.h3_native_denoiser import H3NativeBlockBF16Resident


def block(**changes):
    result = H3NativeBlockBF16Resident.__new__(H3NativeBlockBF16Resident)
    result.artifact = SimpleNamespace(
        target=SimpleNamespace(compute_capability="sm89"),
        weight_profile="lightx-ref-turbo4-v0.1",
    )
    result.elementwise_backend = result.adapter_fusion_backend = "triton-strict"
    result.elementwise_block_size = 1024
    result.weights = SimpleNamespace(
        ffn_in=object(), ffn_in_residual=SimpleNamespace(scaling=0.0625)
    )
    for name, value in changes.items():
        owner = result
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        setattr(owner, parts[-1], value)
    return result


def test_qualified_dispatch_keeps_gemms_and_releases_temporary_values(monkeypatch):
    instance = block()
    states, output, calls, references = object(), object(), [], []

    class Temporary:
        pass

    def project(value, weight):
        assert value is states
        kind = "base" if weight is instance.weights.ffn_in else "adapter"
        calls.append(kind)
        temporary = Temporary()
        references.append(weakref.ref(temporary))
        return temporary

    def merge(base, adapter, *, scaling, block_size):
        assert base is references[0]() and adapter is references[1]()
        assert scaling == 0.0625 and block_size == 1024
        calls.append("merge-silu")
        return output

    instance._linear = instance._residual_linear_unscaled = project
    original_fields = vars(instance).copy()
    monkeypatch.setattr(fused, "triton_strict_bf16_ffn_adapter_silu", merge)
    assert instance._ffn_input(states) is output
    gc.collect()
    assert calls == ["base", "adapter", "merge-silu"]
    assert all(reference() is None for reference in references)
    assert vars(instance) == original_fields


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact.target.compute_capability": "sm86"},
        {"artifact.weight_profile": "lightx-turbo4-v1.0"},
        {"artifact.weight_profile": "lightx-turbo8-v1.0"},
        {"elementwise_backend": "torch-eager"},
        {"adapter_fusion_backend": "torch-eager"},
        {"elementwise_block_size": 512},
        {"weights.ffn_in_residual": None},
        {"weights.ffn_in_residual.scaling": 1.0},
    ],
)
def test_other_contracts_keep_separate_merge_and_activation(monkeypatch, changes):
    instance = block(**changes)
    states, merged, output, calls = object(), object(), object(), []

    def adapted(value, weight, adapter):
        assert value is states and weight is instance.weights.ffn_in
        assert adapter is instance.weights.ffn_in_residual
        calls.append("adapted-linear")
        return merged

    def silu(value):
        assert value is merged
        calls.append("silu")
        return output

    instance._adapted_linear, instance._silu_mul = adapted, silu
    monkeypatch.setattr(
        fused,
        "triton_strict_bf16_ffn_adapter_silu",
        lambda *_args, **_kwargs: pytest.fail("unexpected fused dispatch"),
    )
    assert instance._ffn_input(states) is output
    assert calls == ["adapted-linear", "silu"]


def test_kernel_failure_is_not_silently_retried(monkeypatch):
    instance = block()
    instance._linear = instance._residual_linear_unscaled = lambda *_: object()
    error = RuntimeError("kernel failed")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(fused, "triton_strict_bf16_ffn_adapter_silu", fail)
    with pytest.raises(RuntimeError) as caught:
        instance._ffn_input(object())
    assert caught.value is error


def test_cpu_tensor_is_rejected_without_cuda_initialization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    initialized = torch.cuda.is_initialized()
    value = torch.zeros((1, 3, 8), dtype=torch.bfloat16)
    with pytest.raises(fused.H3FusedOpsError, match="CUDA BF16"):
        fused.triton_strict_bf16_ffn_adapter_silu(value, value, scaling=0.0625, block_size=1024)
    assert torch.cuda.is_initialized() == initialized
