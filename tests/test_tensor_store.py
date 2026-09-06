import gc
import json
import struct
import weakref
from pathlib import Path

import pytest

from vflash.native import h3_tensor_file as files


def write_tensors(path, rows):
    """A small independent format fixture, including scalar and empty tensors."""
    torch = pytest.importorskip("torch")
    header, payload = {}, bytearray()
    names = files._safetensors_dtypes()
    for name, tensor in rows.items():
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_owned_group_reads_are_exact_and_independent_of_file(tmp_path):
    torch = pytest.importorskip("torch")
    tensors = {}
    for dtype in files._torch_dtypes().values():
        tensors[str(dtype)] = torch.arange(12).reshape(3, 4).to(dtype)
    tensors["scalar"] = torch.tensor(-3.5)
    tensors["empty"] = torch.empty(0, 7, dtype=torch.bfloat16)
    path = tmp_path / "weights.safetensors"
    write_tensors(path, tensors)
    store = files.H3SingleTensorStore(path)
    loaded = store.load_many(tensors)
    path.unlink()
    del store
    for name, actual in loaded.items():
        expected = tensors[name]
        assert actual.device.type == "cpu" and actual.is_contiguous()
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
        assert actual is not expected


def test_store_reuses_index_but_rejects_changed_files(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.safetensors"
    write_tensors(path, {"a": torch.ones(2), "b": torch.zeros(3)})
    original = files._inspect_safetensors
    inspected = []

    def inspect(candidate):
        inspected.append(candidate)
        return original(candidate)

    monkeypatch.setattr(files, "_inspect_safetensors", inspect)
    store = files.H3SingleTensorStore(path)
    assert torch.equal(store.load("a"), torch.ones(2))
    assert torch.equal(store.load_many(("b", "a"))["b"], torch.zeros(3))
    assert inspected == [path]
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(files.H3TensorFileError, match="changed after indexing"):
        store.load("a")


@pytest.mark.parametrize("name,shape", [("a", [5]), ("a", [1, 2])])
def test_shape_byte_mismatch_fails_before_payload_load(tmp_path, name, shape):
    pytest.importorskip("torch")
    header = json.dumps(
        {name: {"dtype": "F32", "shape": shape, "data_offsets": [0, 4]}}
    ).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
    with pytest.raises(files.H3TensorFileError, match="shape and payload size"):
        files.load_safetensor_tensor(path, name)


def test_short_read_closes_file_and_releases_partial_tensors(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.safetensors"
    write_tensors(path, {"a": torch.ones(5), "b": torch.zeros(6)})
    store = files.H3SingleTensorStore(path)
    original_open, original_empty = Path.open, torch.empty
    handles, tensors = [], []

    class ShortRead:
        def __init__(self, handle):
            self.handle = handle
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def seek(self, offset):
            return self.handle.seek(offset)

        def readinto(self, view):
            self.reads += 1
            return self.handle.readinto(view) if self.reads == 1 else 0

    def opened(candidate, *args, **kwargs):
        result = ShortRead(original_open(candidate, *args, **kwargs))
        handles.append(result.handle)
        return result

    def empty(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        tensors.append(weakref.ref(value))
        return value

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(files.H3TensorFileError, match="truncated") as failure:
        store.load_many(("a", "b"))
    gc.collect()
    assert failure.value.__traceback__ is not None
    assert handles and all(handle.closed for handle in handles)
    assert all(reference() is None for reference in tensors)


def test_group_validates_all_names_before_reading(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.safetensors"
    write_tensors(path, {"a": torch.ones(2)})
    with pytest.raises(files.H3TensorFileError, match="missing"):
        files.load_safetensor_tensors(path, ("a", "missing"))
    with pytest.raises(files.H3TensorFileError, match="duplicate"):
        files.load_safetensor_tensors(path, ("a", "a"))
