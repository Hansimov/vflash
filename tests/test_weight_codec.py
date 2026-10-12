"""Explicit lossless sidecar: exact storage, bounded allocation and immutable reads."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zstandard")
save_file = pytest.importorskip("safetensors.torch").save_file

compressed = pytest.importorskip("vflash.native.h3_weight_codec")
header = pytest.importorskip("vflash.native.h3_buffered_weights").payload_header


def fixture(tmp_path):
    source = tmp_path / "weights.safetensors"
    sidecar = tmp_path / "compressed"
    values = {
        "codes": torch.arange(32768).remainder(51).to(torch.float8_e4m3fn),
        "scale": torch.randn(511),
        "norm": torch.randn(333).bfloat16(),
    }
    save_file(values, source)
    offset, _ = header(source)
    expected = source.read_bytes()[offset:]
    before = source.stat()
    manifest = compressed.prepare(source, sidecar, chunk_bytes=4096)
    return source, sidecar, offset, expected, before, manifest


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_exact_payload_and_original_metadata(tmp_path, workers):
    source, sidecar, offset, expected, before, manifest = fixture(tmp_path)
    output = bytearray(len(expected))
    result = compressed.read_into(
        sidecar,
        memoryview(output),
        source_name=source.name,
        source_size=before.st_size,
        header_bytes=offset,
        workers=workers,
    )
    assert output == expected and result["decoded_bytes"] == len(expected)
    assert compressed.identity(source.stat()) == compressed.identity(before)
    assert manifest["verified_roundtrip"]
    with pytest.raises(ValueError, match="new destination"):
        compressed.prepare(source, sidecar)


@pytest.mark.parametrize(
    "change", ["overlap", "oversize", "source", "name", "truncated", "symlink"]
)
def test_rejects_inconsistent_or_unsafe_sidecars(tmp_path, change):
    source, sidecar, offset, expected, before, manifest = fixture(tmp_path)
    if change == "overlap":
        manifest["frames"][1][0] -= 1
    if change == "oversize":
        manifest["frames"][0][2] = compressed.MAX_FILE
    if change == "source":
        manifest["source_size"] += 1
    if change == "name":
        manifest["source_name"] = "different-block.safetensors"
    if change == "truncated":
        with (sidecar / "payload.zst").open("r+b") as f:
            f.truncate(5)
    if change == "symlink":
        original = sidecar / "payload.zst"
        original.rename(sidecar / "original")
        original.symlink_to(sidecar / "original")
    (sidecar / "index.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        compressed.read_into(
            sidecar,
            memoryview(bytearray(len(expected))),
            source_name=source.name,
            source_size=before.st_size,
            header_bytes=offset,
        )


def test_short_reads_and_premature_eof(tmp_path, monkeypatch):
    source, sidecar, offset, expected, before, _manifest = fixture(tmp_path)
    actual = compressed.os.pread
    monkeypatch.setattr(compressed.os, "pread", lambda fd, n, at: actual(fd, min(n, 7), at))
    target = bytearray(len(expected))
    compressed.read_into(
        sidecar,
        memoryview(target),
        source_name=source.name,
        source_size=before.st_size,
        header_bytes=offset,
    )
    assert target == expected
    monkeypatch.setattr(compressed.os, "pread", lambda *args: b"")
    with pytest.raises(ValueError, match="ended early"):
        compressed.read_into(
            sidecar,
            memoryview(target),
            source_name=source.name,
            source_size=before.st_size,
            header_bytes=offset,
        )


def test_trailing_frame_and_bomb_declaration_rejected(tmp_path):
    source, sidecar, offset, expected, before, manifest = fixture(tmp_path)
    payload = sidecar / "payload.zst"
    raw = payload.read_bytes()
    first = manifest["frames"][0][1]
    extra = compressed.zstandard.ZstdCompressor().compress(b"not a declared frame")
    payload.write_bytes(raw[:first] + extra + raw[first:])
    manifest["frames"][0][1] += len(extra)
    for record in manifest["frames"][1:]:
        record[0] += len(extra)
    manifest["compressed_bytes"] += len(extra)
    (sidecar / "index.json").write_text(json.dumps(manifest))
    with pytest.raises(compressed.zstandard.ZstdError):
        compressed.read_into(
            sidecar,
            memoryview(bytearray(len(expected))),
            source_name=source.name,
            source_size=before.st_size,
            header_bytes=offset,
        )


@pytest.mark.parametrize("kind", ["symlink", "fifo", "list"])
def test_index_must_be_bounded_regular_and_object(tmp_path, kind):
    import os

    source, sidecar, offset, expected, before, _manifest = fixture(tmp_path)
    index = sidecar / "index.json"
    if kind == "symlink":
        original = sidecar / "original.json"
        index.rename(original)
        index.symlink_to(original)
    elif kind == "fifo":
        index.unlink()
        os.mkfifo(index)
    else:
        index.write_text("[]")
    with pytest.raises((ValueError, OSError)):
        compressed.read_into(
            sidecar,
            memoryview(bytearray(len(expected))),
            source_name=source.name,
            source_size=before.st_size,
            header_bytes=offset,
        )
