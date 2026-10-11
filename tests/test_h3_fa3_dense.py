import sys
from types import SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.native import h3_fa3_dense as fa3
from vflash.native import h3_veda_dense as dense


@pytest.mark.parametrize("capability", ["8.6", "8.9", "10.3", "12.0"])
def test_fa3_rejects_unqualified_architecture(capability):
    with pytest.raises(ContractError, match="SM90"):
        dense.validate_dense_backend(
            "fa3-flash", attention_backend="veda-triton", capability=capability
        )


def test_fa3_requires_explicit_veda_and_pinned_dependency(monkeypatch):
    dense.validate_dense_backend("fa3-flash", attention_backend="veda-triton", capability="9.0")
    with pytest.raises(ContractError, match="SM90"):
        dense.validate_dense_backend(
            "fa3-flash", attention_backend="torch-flash", capability="9.0"
        )
    fa3.implementation.cache_clear()
    try:

        def missing(_):
            raise fa3.PackageNotFoundError("flash-attn-3")

        monkeypatch.setattr(fa3, "version", missing)
        with pytest.raises(ContractError, match="installed"):
            dense.require_dense_dependencies("fa3-flash")
        monkeypatch.setattr(fa3, "version", lambda _: "2.0.0")
        with pytest.raises(ContractError, match="qualified"):
            dense.require_dense_dependencies("fa3-flash")
        monkeypatch.setattr(fa3, "version", lambda _: "3.0.0")
        monkeypatch.setitem(
            sys.modules, "flash_attn_interface", SimpleNamespace(flash_attn_func=lambda: None)
        )
        dense.require_dense_dependencies("fa3-flash")
        torch = pytest.importorskip("torch")
        q = torch.empty(1, 2, 56, 128, dtype=torch.bfloat16)
        with pytest.raises(ContractError, match="SM90"):
            fa3.fa3_dense(q, q, q)
    finally:
        fa3.implementation.cache_clear()


@pytest.mark.parametrize("fused_value", [False, True])
def test_fa3_preserves_views_and_full_context_and_propagates_failures(monkeypatch, fused_value):
    torch = pytest.importorskip("torch")
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vflash.native.h3_veda_attention import VedaVideoAttention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    calls = []

    def kernel(q, k, v, **options):
        calls.append((q, k, v, options))
        return torch.empty_like(q)

    monkeypatch.setattr(fa3, "implementation", lambda: kernel)
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
        op = VedaVideoAttention.__new__(VedaVideoAttention)
        op.dense_backend = "fa3-flash"
        op.dense_attention = fa3.fa3_dense
        op.layout = SimpleNamespace(seq_len=32)
        op.choice = SimpleNamespace(plan=object())
        sparse = []
        op.engine = SimpleNamespace(attention=lambda q, k, v, i, *_: (sparse.append(i), q)[1])
        op.sparse_calls = op.dense_calls = 0

        def forbidden(*_):
            raise AssertionError("fallback invoked")

        for index in (*range(5), *range(45, 50)):
            out = op(index, forbidden, q, k, v)
            assert out.shape == q.shape
        assert op.dense_calls == 10 and op.sparse_calls == 0 and len(calls) == 10
        assert all(
            a is q
            and b is k
            and c is v
            and options
            == dict(causal=False, num_splits=1, pack_gqa=False, return_attn_probs=False)
            for a, b, c, options in calls
        )

        def fail(*a, **k):
            raise RuntimeError("kernel failed")

        monkeypatch.setattr(fa3, "implementation", lambda: fail)
        with pytest.raises(RuntimeError, match="kernel failed"):
            op(0, forbidden, q, k, v)
        monkeypatch.setattr(fa3, "implementation", lambda: lambda *a, **k: (q, None))
        with pytest.raises(ContractError, match="incompatible"):
            op(0, forbidden, q, k, v)
