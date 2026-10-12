"""Lossless framed sidecars for explicitly trusted immutable tensor artifacts.

Original safetensors remain authoritative. These helpers validate layout and
immutable reads; they do not authenticate an artifact or select a model revision.
The caller must bind source and sidecar to its trusted prepared-asset manifest.
No payload hashes or numerical transformations are performed.
"""

from __future__ import annotations

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import zstandard

from vflash.native.h3_tensor_file import inspect_dense_safetensors as header

CHUNK = 16 * 1024**2
MAX_FILE = 12 * 1024**3


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def prepare(source: Path, destination: Path, *, chunk_bytes=CHUNK):
    """Create one new sidecar; compare each decoded frame with its input."""
    if type(chunk_bytes) is not int or not 4096 <= chunk_bytes <= CHUNK:
        raise ValueError("bounded compression chunks required")
    before = source.stat()
    offset, _ = header(source)
    payload = before.st_size - offset
    if not 0 < payload <= MAX_FILE or destination.exists() or destination.is_symlink():
        raise ValueError("one bounded source and a new destination required")
    destination.mkdir(mode=0o700)
    records = []
    compressor = zstandard.ZstdCompressor(level=3, threads=0, write_checksum=False)
    decoder = zstandard.ZstdDecompressor()
    with (
        source.open("rb", buffering=0) as original,
        (destination / "payload.zst").open("xb") as output,
    ):
        if identity(os.fstat(original.fileno())) != identity(before):
            raise ValueError("source changed before compression")
        original.seek(offset)
        remaining = payload
        while remaining:
            data = original.read(min(remaining, chunk_bytes))
            if not data:
                raise ValueError("source ended during compression")
            compressed = compressor.compress(data)
            if decoder.decompress(compressed, max_output_size=len(data)) != data:
                raise ValueError("compressed frame differs from its source")
            records.append([output.tell(), len(compressed), len(data)])
            output.write(compressed)
            remaining -= len(data)
        output.flush()
        os.fsync(output.fileno())
        if identity(os.fstat(original.fileno())) != identity(before) or identity(
            source.stat()
        ) != identity(before):
            raise ValueError("source changed during compression")
    manifest = dict(
        format=1,
        source_name=source.name,
        source_size=before.st_size,
        header_bytes=offset,
        payload_bytes=payload,
        chunk_bytes=chunk_bytes,
        compressed_bytes=(destination / "payload.zst").stat().st_size,
        frames=records,
        verified_roundtrip=True,
    )
    # Manifest-last: a failed preparation is never a consumable sidecar.
    with (destination / "index.json").open("x") as handle:
        json.dump(manifest, handle, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def read_into(directory: Path, target, *, source_name, source_size, header_bytes, workers=4):
    """Parallel independent frames fill disjoint slices of an owned CPU buffer."""
    if type(workers) is not int or workers not in (1, 2, 4, 8):
        raise ValueError("bounded compressed concurrency required")
    index = directory / "index.json"
    fd = os.open(index, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024**2:
            raise ValueError("bounded regular compression index required")
        raw = handle.read(1024**2 + 1)
        if (
            len(raw) != before.st_size
            or identity(os.fstat(handle.fileno())) != identity(before)
            or identity(index.stat()) != identity(before)
        ):
            raise ValueError("compression index changed during read")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("compression index must be an object")
    size = source_size - header_bytes
    if (
        data.get("format") != 1
        or data.get("verified_roundtrip") is not True
        or data.get("source_name") != source_name
        or Path(source_name).name != source_name
        or data.get("source_size") != source_size
        or data.get("header_bytes") != header_bytes
        or data.get("payload_bytes") != size
        or len(target) != size
        or not 0 < size <= MAX_FILE
    ):
        raise ValueError("compression source layout differs")
    chunk = data.get("chunk_bytes")
    if type(chunk) is not int or not 4096 <= chunk <= CHUNK:
        raise ValueError("invalid compression chunk bound")
    frames = data.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FILE // 4096:
        raise ValueError("invalid compression frames")
    plan = []
    compressed_end = output_end = 0
    for row in frames:
        if not isinstance(row, list) or len(row) != 3 or any(type(x) is not int for x in row):
            raise ValueError("invalid compression frame record")
        begin, packed, decoded = row
        if (
            begin != compressed_end
            or not 0 < packed <= chunk + 1024**2
            or not 0 < decoded <= chunk
            or output_end + decoded > size
        ):
            raise ValueError("compression frames overlap or exceed bounds")
        plan.append((begin, packed, output_end, decoded))
        compressed_end += packed
        output_end += decoded
    if output_end != size or compressed_end != data.get("compressed_bytes"):
        raise ValueError("compression frames do not cover the declared payload")
    path = directory / "payload.zst"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != compressed_end:
            raise ValueError("compressed payload size or type differs")

        def read_frame(row):
            begin, packed, output, decoded = row
            pieces = []
            remaining = packed
            while remaining:
                piece = os.pread(fd, remaining, begin)
                if not piece:
                    raise ValueError("compressed payload ended early")
                pieces.append(piece)
                begin += len(piece)
                remaining -= len(piece)
            frame = b"".join(pieces)
            params = zstandard.get_frame_parameters(frame)
            if params.content_size != decoded or params.window_size > CHUNK:
                raise ValueError("compressed frame content/window exceeds declaration")
            value = zstandard.ZstdDecompressor(max_window_size=CHUNK // 1024).decompress(
                frame, max_output_size=decoded, allow_extra_data=False
            )
            if len(value) != decoded:
                raise ValueError("decoded frame size differs")
            target[output : output + decoded] = value

        # executor.map may queue descriptors, but payload buffers exist only in
        # active workers. Each frame is capped; the output has one owner.
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="compressed-weight"
        ) as pool:
            list(pool.map(read_frame, plan))
        if identity(os.fstat(fd)) != identity(before) or identity(path.stat()) != identity(
            before
        ):
            raise ValueError("compressed payload changed during read")
    finally:
        os.close(fd)
    return dict(compressed_bytes=compressed_end, decoded_bytes=output_end, frames=len(frames))
