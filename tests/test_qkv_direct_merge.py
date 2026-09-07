import gc
import weakref
from types import SimpleNamespace

import pytest

from vflash.native import h3_fused_ops as fused
from vflash.native import h3_native_denoiser as native


class Value:
    def untyped_storage(self):
        return self

    def data_ptr(self):
        return id(self)


def block(**changes):
    instance = native.H3NativeBlockBF16Resident.__new__(native.H3NativeBlockBF16Resident)
    instance.artifact = SimpleNamespace(
        target=SimpleNamespace(compute_capability="sm89"),
        weight_profile="lightx-ref-turbo4-v0.1",
    )
    instance.elementwise_backend = instance.adapter_fusion_backend = "triton-strict"
    instance.elementwise_block_size = 1024
    instance.weights = SimpleNamespace(
        qkv=object(),
        qkv_residuals=tuple(SimpleNamespace(scaling=0.0625) for _ in range(3)),
    )
    for name, value in changes.items():
        owner = instance
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        setattr(owner, parts[-1], value)
    return instance


def test_direct_dispatch_recomputes_base_and_releases_adapters(monkeypatch):
    instance = block()
    states, calls, references = Value(), [], []

    def project(value, weight):
        assert value is states
        calls.append("base" if weight is instance.weights.qkv else "adapter")
        result = Value()
        references.append(weakref.ref(result))
        return result

    def direct(base, adapters, *, scalings, block_size):
        assert tuple(scalings) == (0.0625,) * 3 and block_size == 1024
        assert base is references[-4]()
        assert all(
            value is reference()
            for value, reference in zip(adapters, references[-3:], strict=True)
        )
        return base

    def unexpected(*_args, **_kwargs):
        pytest.fail("qualified direct merge must not concatenate or run the old merge")

    instance._linear = instance._residual_linear_unscaled = project
    original_fields = vars(instance).copy()
    monkeypatch.setattr(native, "_torch", lambda: SimpleNamespace(cat=unexpected))
    monkeypatch.setattr(fused, "triton_strict_bf16_qkv_direct_merge", direct)
    monkeypatch.setattr(fused, "triton_strict_bf16_qkv_adapter_merge", unexpected)
    first = instance._qkv_linear(states)
    second = instance._qkv_linear(states)
    assert first is not second and first is references[0]() and second is references[4]()
    assert calls == ["base", "adapter", "adapter", "adapter"] * 2
    assert vars(instance) == original_fields
    assert all(reference() is None for i, reference in enumerate(references) if i not in (0, 4))
    del first, second
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact.target.compute_capability": "sm86"},
        {"artifact.weight_profile": "lightx-turbo4-v1.0"},
        {"artifact.weight_profile": "lightx-turbo8-v1.0"},
        {"elementwise_backend": "torch-eager"},
        {"elementwise_block_size": 512},
        {
            "weights.qkv_residuals": tuple(
                SimpleNamespace(scaling=s) for s in (1.0, 0.0625, 0.3)
            )
        },
    ],
)
def test_other_fused_contracts_keep_concatenation(monkeypatch, changes):
    instance = block(**changes)
    states, base, packed, output = Value(), Value(), Value(), Value()
    adapters = tuple(Value() for _ in range(3))
    calls = []
    instance._linear = lambda *_: base
    instance._residual_linear_unscaled = lambda _, row: adapters[
        next(i for i, value in enumerate(instance.weights.qkv_residuals) if value is row)
    ]

    def concatenate(values, dim):
        assert values == adapters and dim == -1
        calls.append("cat")
        return packed

    def merge(value, update, *, scalings, block_size):
        assert value is base and update is packed
        assert scalings == tuple(row.scaling for row in instance.weights.qkv_residuals)
        assert block_size == instance.elementwise_block_size
        calls.append("existing-merge")
        return output

    monkeypatch.setattr(native, "_torch", lambda: SimpleNamespace(cat=concatenate))
    monkeypatch.setattr(fused, "triton_strict_bf16_qkv_adapter_merge", merge)
    monkeypatch.setattr(
        fused,
        "triton_strict_bf16_qkv_direct_merge",
        lambda *_args, **_kwargs: pytest.fail("unqualified direct merge"),
    )
    assert instance._qkv_linear(states) is output
    assert calls == ["cat", "existing-merge"]


@pytest.mark.parametrize(
    "changes", [{"adapter_fusion_backend": "torch-eager"}, {"weights.qkv_residuals": ()}]
)
def test_eager_and_no_adapter_paths_remain_delegated(monkeypatch, changes):
    instance = block(**changes)
    states, output = Value(), Value()

    def eager(owner, value):
        assert owner is instance and value is states
        return output

    monkeypatch.setattr(native._H3BlockOperations, "_qkv_linear", eager)
    assert instance._qkv_linear(states) is output


def test_fresh_base_cannot_alias_states_and_kernel_errors_are_not_retried(monkeypatch):
    instance = block()
    states = Value()
    monkeypatch.setattr(native, "_torch", lambda: None)
    instance._linear = lambda *_: states
    with pytest.raises(native.H3NativeDenoiserError, match="alias"):
        instance._qkv_linear(states)
    instance._linear = instance._residual_linear_unscaled = lambda *_: Value()
    error = RuntimeError("merge failure")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(fused, "triton_strict_bf16_qkv_direct_merge", fail)
    with pytest.raises(RuntimeError) as caught:
        instance._qkv_linear(states)
    assert caught.value is error


def test_branch_metadata_and_cpu_launch_rejection():
    torch = pytest.importorskip("torch")
    initialized = torch.cuda.is_initialized()
    base = torch.zeros((1, 3, 12), dtype=torch.bfloat16)
    adapters = tuple(torch.ones((1, 3, 4), dtype=torch.bfloat16) for _ in range(3))
    scales = (0.0625, 1.0, -0.3)
    assert fused._validate_qkv_direct_merge(base, adapters, scales, block_size=1024) == (
        12,
        4,
        scales,
    )
    alias = base.view(-1)[:12].view(1, 3, 4)
    for invalid in (
        base.float(),
        base.transpose(1, 2),
        base.clone().requires_grad_(True),
    ):
        with pytest.raises(fused.H3FusedOpsError):
            fused._validate_qkv_direct_merge(invalid, adapters, scales, block_size=1024)
    grad_adapter = adapters[0].clone().requires_grad_(True)
    with pytest.raises(fused.H3FusedOpsError, match="separate"):
        fused._validate_qkv_direct_merge(
            base, (grad_adapter, *adapters[1:]), scales, block_size=1024
        )
    with pytest.raises(fused.H3FusedOpsError, match="separate"):
        fused._validate_qkv_direct_merge(base, (alias, *adapters[1:]), scales, block_size=1024)
    for invalid in ((0.0625,) * 2, (float("nan"), 1.0, 1.0), (True, 1.0, 1.0)):
        with pytest.raises(fused.H3FusedOpsError, match="scalings"):
            fused._validate_qkv_direct_merge(base, adapters, invalid, block_size=1024)
    with pytest.raises(fused.H3FusedOpsError, match="CUDA BF16"):
        fused.triton_strict_bf16_qkv_direct_merge(base, adapters, scalings=scales)
    assert torch.cuda.is_initialized() == initialized
