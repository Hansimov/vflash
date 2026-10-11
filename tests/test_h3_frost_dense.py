import importlib.metadata
import sys
from types import SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.native import h3_frost_dense as frost
from vflash.native import h3_veda_dense as dense


@pytest.mark.parametrize("capability", ["8.6", "8.9", "9.0", "10.3"])
def test_frost_rejects_unqualified_architecture(capability):
    with pytest.raises(ContractError, match="SM120"):
        dense.validate_dense_backend(
            "frost-flash", attention_backend="veda-triton", capability=capability
        )


def test_frost_pins_optional_dependency_and_requires_explicit_veda(monkeypatch):
    dense.validate_dense_backend(
        "frost-flash", attention_backend="veda-triton", capability="12.0"
    )
    with pytest.raises(ContractError, match="SM120"):
        dense.validate_dense_backend(
            "frost-flash", attention_backend="torch-flash", capability="12.0"
        )

    def missing(_):
        raise dense.PackageNotFoundError("nvidia-cudnn-frontend")

    monkeypatch.setattr(dense, "version", missing)
    with pytest.raises(ContractError, match="installed"):
        dense.require_dense_dependencies("frost-flash")
    monkeypatch.setattr(dense, "version", lambda _: "1.30.0")
    with pytest.raises(ContractError, match="qualified"):
        dense.require_dense_dependencies("frost-flash")
    monkeypatch.setattr(dense, "version", lambda _: "1.31.0")
    monkeypatch.setitem(
        sys.modules,
        "cudnn.sdpa.fwd.api_dsl",
        SimpleNamespace(SdpaFwdDslSm120=lambda: None, TensorDesc=lambda: None),
    )
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    with pytest.raises(ContractError, match="CUTE_DSL_ARCH=sm_120a"):
        dense.require_dense_dependencies("frost-flash")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120")
    with pytest.raises(ContractError, match="CUTE_DSL_ARCH=sm_120a"):
        dense.require_dense_dependencies("frost-flash")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    dense.require_dense_dependencies("frost-flash")
    torch = pytest.importorskip("torch")
    q = torch.empty(1, 2, 56, 128, dtype=torch.bfloat16)
    with pytest.raises(ContractError, match="native SM120 BF16"):
        frost.frost_dense(q, q, q)


@pytest.mark.parametrize("fused_value", [False, True])
def test_frost_native_layout_and_current_stream_without_whole_input_copies(
    monkeypatch, fused_value
):
    torch = pytest.importorskip("torch")
    from torch._subclasses.fake_tensor import FakeTensorMode

    calls = []

    def prepare(shape, strides, device, tile):
        assert shape == (1, 56, 32, 128) and tile == (128, 128)
        assert strides[2][2] == (3 if fused_value else 1) * 56 * 128
        assert strides[0][2] == strides[3][2] == 56 * 128

        def execute(*views, current_stream):
            calls.append(current_stream)
            assert views[2].stride(2) == strides[2][2]

        return SimpleNamespace(execute=execute)

    binding = SimpleNamespace(driver=SimpleNamespace(CUstream=lambda value: value))
    monkeypatch.setitem(sys.modules, "cuda", SimpleNamespace(bindings=binding))
    monkeypatch.setitem(sys.modules, "cuda.bindings", binding)
    monkeypatch.setattr(frost, "_prepare", prepare)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.31.0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=731)
    )
    with FakeTensorMode():
        q, k = [
            torch.empty(1, 32, 56, 128, dtype=torch.bfloat16, device="cuda") for _ in range(2)
        ]
        v = (
            (
                torch.empty(1, 32, 3 * 56 * 128, dtype=torch.bfloat16, device="cuda")
                .chunk(3, dim=-1)[2]
                .unflatten(-1, (56, 128))
            )
            if fused_value
            else q
        )
        out = frost.frost_dense(q, k, v)
        assert out.shape == q.shape and out.is_contiguous()

        def fail(*args):
            raise RuntimeError("candidate kernel failure")

        monkeypatch.setattr(frost, "_prepare", fail)
        with pytest.raises(RuntimeError, match="kernel failure"):
            frost.frost_dense(q, k, v)
    assert calls == [731]


@pytest.mark.parametrize("workspace", [0, 128])
def test_frost_caches_only_metadata_and_rejects_scratch_layout(monkeypatch, workspace):
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    torch = pytest.importorskip("torch")
    descriptors, plans = [], []

    def descriptor(dtype, shape, strides, order, device):
        assert not any(
            isinstance(x, torch.Tensor) for x in (dtype, shape, strides, order, device)
        )
        result = (dtype, shape, strides, order, device)
        descriptors.append(result)
        return result

    class Plan:
        def __init__(self, *args, **kwargs):
            assert len(args) == 4 and kwargs == dict(tile_m=128, tile_n=128, split_kv=1)
            plans.append(self)

        def check_support(self):
            pass

        def compile(self):
            pass

        def scratch_workspace_bytes(self):
            return workspace

    monkeypatch.setitem(
        sys.modules,
        "cudnn.sdpa.fwd.api_dsl",
        SimpleNamespace(SdpaFwdDslSm120=Plan, TensorDesc=descriptor),
    )
    frost._prepare.cache_clear()
    shape, strides, device = (
        (1, 56, 32, 128),
        ((229376, 128, 7168, 1),) * 4,
        torch.device("cuda"),
    )
    try:
        if workspace:
            with pytest.raises(ContractError, match="without scratch"):
                frost._prepare(shape, strides, device, (128, 128))
        else:
            first = frost._prepare(shape, strides, device, (128, 128))
            assert frost._prepare(shape, strides, device, (128, 128)) is first
            assert len(plans) == 1 and len(descriptors) == 4
    finally:
        frost._prepare.cache_clear()
