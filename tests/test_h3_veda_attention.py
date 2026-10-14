from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.native.h3_veda_attention import (
    VedaVideoAttention,
    dense_layers_for_edges,
    target_layout,
    validate_predictor,
)


def layout():
    torch = pytest.importorskip("torch")
    grid = (2, 2, 3)
    positions = torch.stack(
        torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij"), dim=-1
    ).reshape(-1, 3)
    tensors = {
        "video_indices": torch.tensor([0, *range(3, 15)]),
        "text_indices": torch.tensor([1]),
        "audio_indices": torch.tensor([2]),
        "position_ids": torch.cat([torch.zeros(3, 3, dtype=torch.int64), positions]),
    }
    return tensors, SimpleNamespace(frames=5, height=64, width=96, num_condition_video_rows=1)


def test_layout_validates_spatial_order_and_protected_rows():
    tensors, profile = layout()
    assert target_layout(tensors, profile) == (3, 15, (2, 2, 3))
    tensors["position_ids"][[3, 4]] = tensors["position_ids"][[4, 3]]
    with pytest.raises(ValueError, match="row-major"):
        target_layout(tensors, profile)
    tensors, profile = layout()
    tensors["video_indices"][-1] = 1
    with pytest.raises(ValueError, match="contiguous"):
        target_layout(tensors, profile)


def test_predictor_required_and_unrelated_backends_reject_it(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    path = tmp_path / "predictor.safetensors"
    with pytest.raises(ContractError, match="local"):
        validate_predictor("veda-sm89", path)
    save_file({"x": torch.zeros(1)}, path, metadata={"format": "unrelated"})
    with pytest.raises(ContractError, match="50-layer"):
        validate_predictor("veda-sm89", path)
    with pytest.raises(ContractError, match="requires attention_backend"):
        validate_predictor("torch-flash", path)
    validate_predictor("torch-flash", None)


@pytest.mark.parametrize("edge_layers", [0, 3, 5])
def test_attention_routes_every_logical_layer_and_rejects_missing_work(
    edge_layers, monkeypatch
):
    torch = pytest.importorskip("torch")
    calls = []
    op = VedaVideoAttention.__new__(VedaVideoAttention)
    op.dense_layers = dense_layers_for_edges(edge_layers, attention_backend="veda-triton")
    monkeypatch.setattr("vflash.native.h3_veda_attention.DENSE_LAYERS", frozenset())
    op.dense_attention = None
    op.dense_backend = "torch-flash"
    op.layout = SimpleNamespace(seq_len=2)
    op.choice = SimpleNamespace(plan=object())
    op.engine = SimpleNamespace(
        attention=lambda q, k, v, index, *_: (calls.append(index), q)[1]
    )
    op.sparse_calls = op.dense_calls = 0
    q = torch.zeros(1, 2, 56, 128, dtype=torch.bfloat16)

    def dense(q, k, v):
        return q + 1

    for index in range(50):
        out = op(index, dense, q, q, q)
        assert out.shape == q.shape
        expected = 1 if index < edge_layers or index >= 50 - edge_layers else 0
        assert bool((out == expected).all())
    assert calls == list(range(edge_layers, 50 - edge_layers))
    op.validate_calls(1)
    assert op.dense_calls == 2 * edge_layers
    assert op.sparse_calls == 50 - 2 * edge_layers
    with pytest.raises(RuntimeError, match="coverage"):
        op.validate_calls(4)
    with pytest.raises(ValueError, match="logical block"):
        op(50, dense, q, q, q)
    with pytest.raises(ValueError, match="matching BF16"):
        op(10, dense, q.float(), q, q)


@pytest.mark.parametrize("value", [True, False, 0.0, 1, 2, 4, 6, 7, None, "3"])
def test_invalid_dense_partition_fails_before_loading(value):
    with pytest.raises(ContractError, match="0, 3 or 5"):
        dense_layers_for_edges(value, attention_backend="veda-triton")


def test_dense_partition_requires_veda_and_remains_session_local():
    with pytest.raises(ContractError, match="requires Veda"):
        dense_layers_for_edges(3, attention_backend="torch-flash")
    a = dense_layers_for_edges(3, attention_backend="veda-triton")
    b = dense_layers_for_edges(5, attention_backend="veda-triton")
    assert a == frozenset((0, 1, 2, 47, 48, 49))
    assert b == frozenset((*range(5), *range(45, 50)))
    assert len(a) == 6 and len(b) == 10


def test_actual_two_slot_loop_assigns_logical_indices_on_every_forward(monkeypatch):
    from vflash.native import h3_native_denoiser as native

    seen = []
    stream = SimpleNamespace(wait_event=lambda _: None)
    event = SimpleNamespace(record=lambda _: None)
    fake = SimpleNamespace(
        cuda=SimpleNamespace(current_stream=lambda _: stream, stream=lambda _: nullcontext())
    )
    monkeypatch.setattr(native, "_torch", lambda: fake)
    ring = native.H3NativeDenoiserBF16Ring.__new__(native.H3NativeDenoiserBF16Ring)
    ring.device = "test"
    ring.host_blocks = tuple(range(50))

    class Slot:
        def forward_prevalidated(self, states, invocation):
            seen.append(self._attention_block_index)
            return states

    ring.slots = (Slot(), Slot())
    ring.copy_stream = stream
    ring.ready_events = ring.compute_done_events = (event, event)
    ring._queue_initial_slots = lambda _: None
    ring._copy_ring_block = lambda *_: None
    for _ in range(2):
        assert ring.forward_prevalidated("states", None) == ("states", {})
    assert seen == list(range(50)) * 2


def test_installer_retains_only_core_and_licenses(tmp_path, monkeypatch):
    import io
    import tarfile

    from vflash import install_veda

    names = [
        "LICENSE",
        "NOTICE.md",
        "veda_comfy/core/engine.py",
        "veda_comfy/backends/triton_int8.py",
        "veda_comfy/nodes.py",
        "example_workflows/private.json",
    ]
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name in names:
            member = tarfile.TarInfo(f"Veda-on-ComfyUI-{install_veda.REVISION}/{name}")
            member.size = 5
            archive.addfile(member, io.BytesIO(b"hello"))
    monkeypatch.setattr(install_veda, "urlopen", lambda *a, **kw: io.BytesIO(data.getvalue()))
    install_veda.fetch_package(tmp_path)
    assert not (tmp_path / "veda_comfy/nodes.py").exists()
    assert not (tmp_path / "example_workflows").exists()
    assert "BSD 3-Clause" in (tmp_path / "SAGE-LICENSE").read_text()
    assert install_veda.REVISION in (tmp_path / "veda_comfy/_vflash_pin.py").read_text()
