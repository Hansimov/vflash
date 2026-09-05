"""Pack immutable host tensors into native Torch allocations of bounded size."""

from __future__ import annotations

from typing import Any


class PinnedHostArena:
    """Bump-allocate views across blocks without per-tensor power-of-two padding.

    Every returned tensor retains its allocation through Torch's own Storage.
    This preserves the host allocator context used by asynchronous CUDA copies;
    frombuffer/from_blob wrappers must not replace these views. The allocator
    owns only the current slab; prior slabs live through their returned tensors.
    Closing drops that final owner reference, never invalidating live views.
    """

    def __init__(self, chunk_bytes: int = 2 * 1024**3, *, pin_memory: bool = True) -> None:
        if chunk_bytes < 256 or chunk_bytes & (chunk_bytes - 1):
            raise ValueError("host arena size must be a power of two of at least 256 bytes")
        self.chunk_bytes = chunk_bytes
        self.pin_memory = pin_memory
        self.allocation_count = 0
        self.payload_bytes = 0
        self._buffer: Any = None
        self._offset = 0
        self._closed = False

    def copy(self, source: Any) -> Any:
        """Make a synchronous CPU copy into one contiguous, dtype-aligned view."""
        import torch

        if self._closed:
            raise RuntimeError("the host arena is closed")
        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "cpu"
            or source.layout != torch.strided
            or source.requires_grad
        ):
            raise ValueError("the host arena requires dense CPU tensors without gradients")
        size = source.numel() * source.element_size()
        if size <= 0 or size > self.chunk_bytes:
            raise ValueError("a host tensor must fit entirely inside one arena slab")
        try:
            offset = (self._offset + 255) // 256 * 256
            if self._buffer is None or offset + size > self.chunk_bytes:
                self._buffer = torch.empty(
                    self.chunk_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                self.allocation_count += 1
                offset = 0
            output = (
                self._buffer.narrow(0, offset, size).view(source.dtype).reshape(source.shape)
            )
            output.copy_(source)
            self._offset = offset + size
            self.payload_bytes += size
            return output
        except BaseException:
            self.close()
            raise
        finally:
            # The source may be an mmap export. A failed allocation/copy must
            # not leave it alive solely through this helper's traceback frame.
            source = None

    def close(self) -> None:
        self._buffer = None
        self._closed = True

    def __enter__(self) -> PinnedHostArena:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
