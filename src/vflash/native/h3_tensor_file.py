"""Validated tensor files, scoped memory maps, and atomic latent output."""

from __future__ import annotations

import json
import mmap
import os
import re
import stat
import struct
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vflash.native.errors import VflashNativeError


class H3TensorFileError(VflashNativeError):
    """A tensor file violates the runtime format or storage contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise RuntimeError("H3 tensor I/O requires PyTorch") from exc
    return torch


def _torch_dtypes() -> dict[str, Any]:
    torch = _torch()
    return {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
    }


def _safetensors_dtypes() -> dict[Any, str]:
    return {value: name for name, value in _torch_dtypes().items()}


def load_safetensor_tensor(path: Path, name: str) -> Any:
    """Load one tensor without materializing any other payload in the file."""

    torch = _torch()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise H3TensorFileError(f"H3 tensor file is unavailable: {path}") from exc
    header = inspect_safetensors_header(resolved)
    try:
        row = header[name]
    except KeyError as exc:
        raise H3TensorFileError(f"H3 tensor is missing: {name}") from exc
    dtype = _torch_dtypes().get(str(row["dtype"]))
    if dtype is None:
        raise H3TensorFileError(f"H3 tensor dtype is unsupported: {name}")
    try:
        with resolved.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise H3TensorFileError(f"safetensors prefix is truncated: {resolved.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            start, stop = row["data_offsets"]
            handle.seek(8 + header_size + start)
            payload = bytearray(handle.read(stop - start))
    except H3TensorFileError:
        raise
    except OSError as exc:
        raise H3TensorFileError(f"H3 tensor payload is unavailable: {name}") from exc
    if len(payload) != stop - start:
        raise H3TensorFileError(f"H3 tensor payload is truncated: {name}")
    try:
        tensor = torch.frombuffer(payload, dtype=dtype).reshape(row["shape"]).clone()
    except (RuntimeError, ValueError) as exc:
        raise H3TensorFileError(f"H3 tensor payload cannot be decoded: {name}") from exc
    return tensor.contiguous()


class H3MappedSafetensor:
    """Expose zero-copy CPU tensor views from one validated safetensors file."""

    def __init__(self, path: Path) -> None:
        try:
            self.path = path.resolve(strict=True)
        except OSError as exc:
            raise H3TensorFileError(f"H3 tensor file is unavailable: {path}") from exc
        self.header = inspect_safetensors_header(self.path)
        try:
            self._handle = self.path.open("rb")
            prefix = self._handle.read(8)
            if len(prefix) != 8:
                raise H3TensorFileError(f"safetensors prefix is truncated: {self.path.name}")
            self._data_offset = 8 + struct.unpack("<Q", prefix)[0]
            self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_COPY)
        except Exception:
            handle = getattr(self, "_handle", None)
            if handle is not None:
                handle.close()
            raise

    def load(self, name: str) -> Any:
        torch = _torch()
        mapping = getattr(self, "_mapping", None)
        if mapping is None:
            raise H3TensorFileError("H3 mapped tensor file is closed")
        try:
            row = self.header[name]
        except KeyError as exc:
            raise H3TensorFileError(f"H3 tensor is missing: {name}") from exc
        dtype = _torch_dtypes().get(str(row["dtype"]))
        if dtype is None:
            raise H3TensorFileError(f"H3 tensor dtype is unsupported: {name}")
        start, stop = row["data_offsets"]
        item_size = torch.empty((), dtype=dtype).element_size()
        size = stop - start
        if size % item_size:
            raise H3TensorFileError(f"H3 tensor payload is misaligned: {name}")
        try:
            return torch.frombuffer(
                mapping,
                dtype=dtype,
                count=size // item_size,
                offset=self._data_offset + start,
            ).reshape(row["shape"])
        except (RuntimeError, ValueError) as exc:
            raise H3TensorFileError(f"H3 tensor payload cannot be mapped: {name}") from exc

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is None:
            return
        try:
            mapping.close()
        except BufferError as exc:
            raise H3TensorFileError("H3 mapped tensor views outlived their load scope") from exc
        finally:
            self._mapping = None
            self._handle.close()

    def __enter__(self) -> H3MappedSafetensor:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def save_safetensors_atomic(
    path: Path,
    tensors: Mapping[str, Any],
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write a deterministic safetensors file and atomically publish it."""

    torch = _torch()
    if sys.byteorder != "little":
        raise H3TensorFileError("H3 safetensors output requires a little-endian host")
    if not tensors:
        raise H3TensorFileError("H3 safetensors output must contain tensors")
    dtype_names = _safetensors_dtypes()
    ordered: list[tuple[str, Any]] = []
    header: dict[str, Any] = {}
    offset = 0
    for name, tensor in sorted(tensors.items()):
        if (
            not isinstance(name, str)
            or not name
            or name == "__metadata__"
            or not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.dtype not in dtype_names
        ):
            raise H3TensorFileError(f"H3 cannot serialize tensor: {name}")
        contiguous = tensor.detach().contiguous()
        size = contiguous.numel() * contiguous.element_size()
        header[name] = {
            "dtype": dtype_names[contiguous.dtype],
            "shape": list(contiguous.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
        ordered.append((name, contiguous))
    if metadata is not None:
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise H3TensorFileError("H3 safetensors metadata must contain only strings")
        header["__metadata__"] = dict(sorted(metadata.items()))
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            for _name, tensor in ordered:
                handle.write(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class H3SingleTensorStore:
    """Read tensors from one verified, regular safetensors blob."""

    def __init__(self, path: Path) -> None:
        try:
            self.path = path.resolve(strict=True)
        except OSError as exc:
            raise H3TensorFileError(f"H3 tensor store is unavailable: {path}") from exc
        self.header = inspect_safetensors_header(self.path)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.header))

    def load(self, name: str) -> Any:
        if name not in self.header:
            raise H3TensorFileError(f"H3 tensor is missing: {name}")
        return load_safetensor_tensor(self.path, name)


_HEADER_LIMIT = 64 * 1024 * 1024
_TENSOR_DTYPE = re.compile(r"[A-Z][A-Z0-9_]{0,31}")


def _regular_file(path: Path) -> stat.struct_stat:
    try:
        value = path.lstat()
    except OSError as exc:
        raise H3TensorFileError(f"H3 tensor file is unavailable: {path.name}") from exc
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise H3TensorFileError(f"H3 tensor file must be a non-empty regular file: {path.name}")
    return value


def inspect_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Read and validate a safetensors header without loading tensor payloads."""
    file_stat = _regular_file(path)
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise H3TensorFileError(f"safetensors prefix is truncated: {path.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if (
                header_size < 2
                or header_size > _HEADER_LIMIT
                or header_size > file_stat.st_size - 8
            ):
                raise H3TensorFileError(f"safetensors header size is invalid: {path.name}")
            header = json.loads(handle.read(header_size).decode("utf-8"))
    except H3TensorFileError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise H3TensorFileError(f"safetensors header is invalid: {path.name}") from exc
    if not isinstance(header, dict):
        raise H3TensorFileError(f"safetensors header must be an object: {path.name}")

    data_size = file_stat.st_size - 8 - header_size
    tensors: dict[str, dict[str, Any]] = {}
    occupied: list[tuple[int, int, str]] = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, dict):
                raise H3TensorFileError(f"safetensors metadata is invalid: {path.name}")
            continue
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise H3TensorFileError(f"safetensors tensor entry is invalid: {path.name}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(dtype, str) or not _TENSOR_DTYPE.fullmatch(dtype):
            raise H3TensorFileError(f"safetensors dtype is invalid for {name}")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise H3TensorFileError(f"safetensors shape is invalid for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in offsets)
        ):
            raise H3TensorFileError(f"safetensors offsets are invalid for {name}")
        start, stop = offsets
        if start < 0 or stop < start or stop > data_size:
            raise H3TensorFileError(f"safetensors offsets are outside the payload for {name}")
        occupied.append((start, stop, name))
        tensors[name] = {"dtype": dtype, "shape": tuple(shape), "data_offsets": (start, stop)}
    if not tensors:
        raise H3TensorFileError(f"safetensors file has no tensors: {path.name}")
    previous_stop = 0
    for start, stop, name in sorted(occupied):
        if start < previous_stop:
            raise H3TensorFileError(f"safetensors tensors overlap at {name}")
        previous_stop = max(previous_stop, stop)
    return tensors
