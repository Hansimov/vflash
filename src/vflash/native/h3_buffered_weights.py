"""Explicit owned-buffer reads for immutable safetensors on slow storage.

The caller owns model identity, memory admission and policy. This opt-in helper
does not change the default mapped loader. Retained tensors keep the entire shard
buffer alive after the load scope; size host memory for that lifetime.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch

from vflash.native.h3_tensor_file import inspect_dense_safetensors as payload_header

DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def host_headroom():
    available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    try:
        maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if maximum.isdigit():
            available = min(available, max(0, int(maximum) - current))
    except OSError:
        pass
    return available


class SequentialWeights:
    def __init__(
        self,
        path,
        *,
        framework="pt",
        device="cpu",
        observation=None,
        read_workers=1,
        compressed_directory=None,
    ):
        if framework != "pt" or device != "cpu":
            raise ValueError("sequential model reads require CPU Torch storage")
        if type(read_workers) is not int or read_workers not in (1, 2, 4, 8):
            raise ValueError("weight read concurrency must be 1, 2, 4 or 8")
        if read_workers > 1 and not hasattr(os, "preadv"):
            raise ValueError("parallel weight reads require positioned vector I/O")
        self.read_workers = read_workers
        self.compressed_directory = compressed_directory
        self.path = Path(path)
        self.observation = observation
        self.tensors = {}

    def __enter__(self):
        start = time.monotonic()
        before = self.path.stat()
        offset, entries = payload_header(self.path)
        size = before.st_size - offset
        if not 0 < size <= 12 * 1024**3 or host_headroom() < size + 2 * 1024**3:
            raise ValueError("sequential weight read exceeds its bounded host memory budget")
        specs = []
        for name, row in entries.items():
            if name == "__metadata__":
                continue
            dtype = DTYPES.get(row["dtype"])
            shape = row["shape"]
            begin, end = row["data_offsets"]
            if (
                dtype is None
                or any(type(n) is not int or n < 0 for n in shape)
                or begin % dtype.itemsize
                or end - begin != math.prod(shape) * dtype.itemsize
            ):
                raise ValueError("unsupported or malformed sequential tensor layout")
            specs.append((name, dtype, shape, begin, end))
        buffer = torch.empty(size, dtype=torch.uint8, device="cpu")
        view = memoryview(buffer.numpy())
        try:
            with self.path.open("rb", buffering=0) as stream:
                if identity(os.fstat(stream.fileno())) != identity(before):
                    raise ValueError("weight source changed before sequential read")
                if self.compressed_directory is not None:
                    from vflash.native.h3_weight_codec import read_into

                    compressed = read_into(
                        Path(self.compressed_directory),
                        view,
                        source_name=self.path.name,
                        source_size=before.st_size,
                        header_bytes=offset,
                        workers=self.read_workers,
                    )
                    if self.observation is not None:
                        self.observation["compressed_bytes"] = (
                            self.observation.get("compressed_bytes", 0)
                            + compressed["compressed_bytes"]
                        )
                elif self.read_workers == 1:
                    stream.seek(offset)
                    count = 0
                    while count < size:
                        read = stream.readinto(view[count : min(size, count + 8 * 1024**2)])
                        if not read:
                            raise ValueError("weight source ended during sequential read")
                        count += read
                else:
                    # Disjoint ranges write into the same bounded tensor buffer; no
                    # per-worker payload copy and no shared seek position.
                    from concurrent.futures import ThreadPoolExecutor

                    span = math.ceil(size / self.read_workers)

                    def read_range(begin):
                        end = min(size, begin + span)
                        while begin < end:
                            read = os.preadv(
                                stream.fileno(),
                                [view[begin : min(end, begin + 8 * 1024**2)]],
                                offset + begin,
                            )
                            if not read:
                                raise ValueError("weight source ended during parallel read")
                            begin += read

                    with ThreadPoolExecutor(
                        max_workers=self.read_workers, thread_name_prefix="native-weight"
                    ) as pool:
                        list(pool.map(read_range, range(0, size, span)))
                if identity(os.fstat(stream.fileno())) != identity(before) or identity(
                    self.path.stat()
                ) != identity(before):
                    raise ValueError("weight source changed during sequential read")
            self.tensors = {
                name: buffer[begin:end].view(dtype).reshape(shape)
                for name, dtype, shape, begin, end in specs
            }
            if self.observation is not None:
                self.observation["files"] += 1
                self.observation["bytes"] += size
                self.observation["read_seconds"] += time.monotonic() - start
        except BaseException:
            self.tensors.clear()
            buffer = None
            raise
        finally:
            view.release()

        return self

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, name):
        return self.tensors[name]

    def __exit__(self, *args):
        # Tensors assigned into modules retain the buffer without a second copy.
        self.tensors.clear()
