import sys
from types import SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.native import h3_veda_dense as dense
from vflash.native.h3_veda_attention import VedaVideoAttention


@pytest.mark.parametrize(
    "backend,capability",
    [
        ("torch-flash", "12.0"),
        ("veda-sm89", "8.9"),
        ("veda-triton", "9.0"),
        ("veda-triton", "10.3"),
    ],
)
def test_sage_rejects_unqualified_or_unrelated_contract(backend, capability):
    with pytest.raises(ContractError, match="SM120"):
        dense.validate_dense_backend(
            "sageattention2", attention_backend=backend, capability=capability
        )


def test_explicit_choice_and_pinned_optional_dependency(monkeypatch):
    dense.validate_dense_backend(
        "sageattention2", attention_backend="veda-triton", capability="12.0"
    )
    with pytest.raises(ContractError, match="unknown"):
        dense.validate_dense_backend("auto", attention_backend="veda-triton", capability="12.0")

    def missing(_):
        raise dense.PackageNotFoundError("sageattention")

    monkeypatch.setattr(dense, "version", missing)
    dense.require_dense_dependencies("torch-flash")
    with pytest.raises(ContractError, match="installed"):
        dense.require_dense_dependencies("sageattention2")
    monkeypatch.setattr(dense, "version", lambda _: "2.1.0")
    with pytest.raises(ContractError, match="qualified"):
        dense.require_dense_dependencies("sageattention2")


def test_only_ten_dense_layers_change_and_kernel_failure_never_falls_back(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def kernel(q, k, v, **options):
        calls.append(options)
        return q + 2

    monkeypatch.setitem(
        sys.modules, "sageattention", SimpleNamespace(sageattn_qk_int8_pv_fp8_cuda=kernel)
    )
    monkeypatch.setattr(dense, "version", lambda _: "2.2.0")
    dense.require_dense_dependencies("sageattention2")
    op = VedaVideoAttention.__new__(VedaVideoAttention)
    op.dense_backend = "sageattention2"
    op.dense_attention = dense.sage_dense
    op.layout = SimpleNamespace(seq_len=2)
    op.choice = SimpleNamespace(plan=object())
    sparse = []
    op.engine = SimpleNamespace(attention=lambda q, k, v, i, *_: (sparse.append(i), q)[1])
    op.sparse_calls = op.dense_calls = 0
    q = torch.zeros(1, 2, 56, 128, dtype=torch.bfloat16)

    def forbidden(*_):
        raise AssertionError("Flash fallback executed")

    for index in range(50):
        result = op(index, forbidden, q, q, q)
        assert bool((result == (2 if index < 5 or index >= 45 else 0)).all())
    op.validate_calls(1)
    assert sparse == list(range(5, 45)) and len(calls) == 10
    assert all(
        opts
        == dict(
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_warp",
            pv_accum_dtype="fp32+fp16",
            smooth_k=True,
            smooth_v=False,
        )
        for opts in calls
    )

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA kernel failure")

    monkeypatch.setitem(
        sys.modules, "sageattention", SimpleNamespace(sageattn_qk_int8_pv_fp8_cuda=fail)
    )
    with pytest.raises(RuntimeError, match="kernel failure"):
        op(0, forbidden, q, q, q)
