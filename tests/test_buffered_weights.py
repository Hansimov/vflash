"""Sequential I/O preserves exact tensor storage and rejects malformed budgets."""

import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

SequentialWeights = pytest.importorskip("vflash.native.h3_buffered_weights").SequentialWeights


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_sequential_values_lifetime_and_source_identity(tmp_path, workers):
    path = tmp_path / "weights.safetensors"
    values = {
        "codes": torch.arange(24).reshape(3, 8).to(torch.float8_e4m3fn),
        "scale": torch.tensor([[0.1], [0.2], [0.3]]),
        "norm": torch.randn(8).bfloat16(),
        "index": torch.arange(3),
        "empty": torch.empty((0, 3), dtype=torch.float16),
    }
    save_file(values, path)
    before = path.stat()
    observation = {"files": 0, "bytes": 0, "read_seconds": 0.0}
    with SequentialWeights(path, observation=observation, read_workers=workers) as source:
        names = source.keys()
        loaded = {name: source.get_tensor(name) for name in names}
    for name, actual in loaded.items():
        assert actual.dtype == values[name].dtype and actual.shape == values[name].shape
        assert torch.equal(actual.float(), values[name].float())
    assert observation["bytes"] == sum(t.numel() * t.element_size() for t in values.values())
    assert observation["files"] == 1
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def test_sequential_rejects_capacity_and_truncated_payload(tmp_path, monkeypatch):
    from vflash.native import h3_buffered_weights as io

    path = tmp_path / "weights.safetensors"
    save_file({"w": torch.ones(4)}, path)
    monkeypatch.setattr(io, "host_headroom", lambda: 1)
    with pytest.raises(ValueError, match="memory budget"), SequentialWeights(path):
        pass
    with path.open("r+b") as f:
        f.truncate(path.stat().st_size - 1)
    with (
        pytest.raises(ValueError, match="invalid immutable tensor header"),
        SequentialWeights(path),
    ):
        pass


def test_parallel_short_reads_and_eof(tmp_path, monkeypatch):
    from vflash.native import h3_buffered_weights as io

    path = tmp_path / "weights.safetensors"
    values = torch.arange(3001, dtype=torch.int32)
    save_file({"w": values}, path)
    real = io.os.preadv

    def short(fd, buffers, offset):
        return real(fd, [buffers[0][:7]], offset)

    monkeypatch.setattr(io.os, "preadv", short)
    with SequentialWeights(path, read_workers=4) as source:
        assert torch.equal(source.get_tensor("w"), values)
    monkeypatch.setattr(io.os, "preadv", lambda *args: 0)
    with (
        pytest.raises(ValueError, match="ended during parallel"),
        SequentialWeights(path, read_workers=4),
    ):
        pass


def test_parallel_reads_reject_changed_source_and_invalid_concurrency(tmp_path, monkeypatch):
    from vflash.native import h3_buffered_weights as io

    path = tmp_path / "weights.safetensors"
    save_file({"w": torch.ones(101)}, path)
    for workers in (0, True, 3, 16):
        with pytest.raises(ValueError, match="concurrency"):
            SequentialWeights(path, read_workers=workers)
    real = io.os.preadv

    def altered(fd, buffers, offset):
        count = real(fd, buffers, offset)
        # Fixture-only metadata mutation must be noticed after readers join.
        io.os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000))
        return count

    monkeypatch.setattr(io.os, "preadv", altered)
    with (
        pytest.raises(ValueError, match="source changed"),
        SequentialWeights(path, read_workers=2),
    ):
        pass


@pytest.mark.parametrize("compressed", [False, True])
def test_complete_shard_retains_values_after_scope(tmp_path, compressed):
    if compressed:
        pytest.importorskip("zstandard")
        from vflash.native.h3_weight_codec import prepare
    path = tmp_path / "weights.safetensors"
    values = {
        "codes": torch.arange(1024).remainder(41).to(torch.float8_e4m3fn),
        "scalar": torch.tensor(3.5),
        "empty": torch.empty((0, 7)),
    }
    save_file(values, path)
    sidecar = tmp_path / "sidecar" if compressed else None
    if sidecar is not None:
        prepare(path, sidecar, chunk_bytes=4096)
    observation = {"files": 0, "bytes": 0, "read_seconds": 0.0}
    with SequentialWeights(
        path, compressed_directory=sidecar, read_workers=4, observation=observation
    ) as source:
        retained = {name: source.get_tensor(name) for name in values}
    path.unlink()
    for name, tensor in retained.items():
        assert torch.equal(
            tensor.reshape(-1).view(torch.uint8), values[name].reshape(-1).view(torch.uint8)
        )
    assert observation["files"] == 1
    assert ("compressed_bytes" in observation) is compressed


def test_failed_parallel_load_does_not_retain_payload_in_traceback(tmp_path, monkeypatch):
    import gc
    import weakref

    from vflash.native import h3_buffered_weights as io

    path = tmp_path / "weights.safetensors"
    save_file({"w": torch.ones(1000)}, path)
    original = io.torch.empty
    allocations = []

    def allocate(*args, **kwargs):
        value = original(*args, **kwargs)
        allocations.append(weakref.ref(value))
        return value

    monkeypatch.setattr(io.torch, "empty", allocate)
    monkeypatch.setattr(io.os, "preadv", lambda *args: 0)
    with (
        pytest.raises(ValueError, match="ended during parallel") as failure,
        SequentialWeights(path, read_workers=4),
    ):
        pass
    gc.collect()
    assert failure.value.__traceback__ is not None
    assert allocations and all(ref() is None for ref in allocations)
