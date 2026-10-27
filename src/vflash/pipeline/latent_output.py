"""Opt-in durable native AV output, independent of later media decoding."""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from pathlib import Path

from vflash.contracts import ContractError


def prepare_latent_output(path: Path | None, media: Path) -> Path | None:
    if path is None:
        return None
    if not isinstance(path, Path):
        raise ContractError("latent_output must be a pathlib.Path")
    target = path.parent.resolve() / path.name
    if target == media.parent.resolve() / media.name or target.exists() or target.is_symlink():
        raise ContractError("native latents require a distinct, unused output path")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def retain_native_latents(source: Path, target: Path) -> None:
    """Publish a complete native output without overwriting another owner's file."""
    try:
        os.link(source, target)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    # Cross-filesystem callers receive the same atomic, no-replace publication.
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".vflash-latents-", delete=False
        ) as handle:
            pending = Path(handle.name)
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, target)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
