"""Reserved uint8 RGB arrays for explicit, lossless local media handoff."""

import errno
import math
import os
from pathlib import Path


def create_rgb_array(path, *, shape):
    import numpy as np

    if (
        len(shape) != 4
        or shape[-1] != 3
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError("RGB arrays require positive THWC dimensions and three channels")
    path = Path(path)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            np.lib.format.write_array_header_1_0(
                stream,
                {
                    "descr": np.dtype(np.uint8).str,
                    "fortran_order": False,
                    "shape": tuple(shape),
                },
            )
            stream.flush()
            offset = stream.tell()
        remaining = math.prod(shape)
        size = offset + remaining
        os.ftruncate(descriptor, size)
        try:
            os.posix_fallocate(descriptor, 0, size)
        except (AttributeError, OSError) as error:
            if isinstance(error, OSError) and error.errno not in {
                errno.ENOSYS,
                errno.EOPNOTSUPP,
                errno.EINVAL,
            }:
                raise
            os.lseek(descriptor, offset, os.SEEK_SET)
            zeros = bytes(min(1024**2, remaining))
            while remaining:
                written = os.write(descriptor, memoryview(zeros)[:remaining])
                if not written:
                    raise OSError(errno.ENOSPC, "RGB array allocation stopped") from error
                remaining -= written
            os.fsync(descriptor)
        return np.lib.format.open_memmap(path, mode="r+")
    except BaseException:
        path.unlink()
        raise
    finally:
        os.close(descriptor)
