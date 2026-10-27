import errno
from pathlib import Path

import pytest

from vflash.pipeline.latent_output import retain_native_latents


def test_cross_filesystem_copy_is_published_only_after_completion(tmp_path, monkeypatch):
    import vflash.pipeline.latent_output as storage

    source = tmp_path / "native.safetensors"
    source.write_bytes(b"complete native AV" * 100)
    target = tmp_path / "saved.safetensors"
    link = storage.os.link
    seen = []

    def cross_filesystem(first, second):
        if first == source:
            raise OSError(errno.EXDEV, "different filesystem")
        assert not target.exists()
        assert Path(first).read_bytes() == source.read_bytes()
        seen.append(first)
        return link(first, second)

    monkeypatch.setattr(storage.os, "link", cross_filesystem)
    retain_native_latents(source, target)
    assert target.read_bytes() == source.read_bytes() and len(seen) == 1
    assert not list(tmp_path.glob(".vflash-latents-*"))


def test_publication_race_preserves_the_other_owners_file(tmp_path):
    source, target = tmp_path / "native", tmp_path / "saved"
    source.write_bytes(b"this request")
    target.write_bytes(b"another request")
    with pytest.raises(FileExistsError):
        retain_native_latents(source, target)
    assert target.read_bytes() == b"another request"
