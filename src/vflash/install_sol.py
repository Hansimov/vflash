"""Install the pinned SM89 Sol dependency into the active Python environment.

Explicit setup only: package import never installs dependencies or initializes CUDA.
The upstream MIT implementation is fetched, not vendored into Vflash.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "d0c0a4685ab5dc2336d18b7213d85f13def92418"
DEPENDENCIES = ("nvidia-cutlass-dsl==4.5.0", "cuda-python==13.2.0", "apache-tvm-ffi==0.1.11")


def patch_stream_abi(path: Path) -> None:
    source = path.read_text()
    old = "                stream=stream,"
    if source.count(old) != 4:
        raise RuntimeError("pinned Sol stream ABI differs from the reviewed four call sites")
    path.write_text(source.replace(old, "                stream,"))


def fetch_package(destination: Path) -> None:
    """Use one immutable archive, retaining only the small Sol source package."""
    request = Request(
        f"https://codeload.github.com/NVlabs/Sana/tar.gz/{REVISION}",
        headers={"User-Agent": "vflash-installer"},
    )
    limit = 128 * 1024 * 1024
    with urlopen(request, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("upstream source archive exceeds installation budget")
    prefix = f"Sana-{REVISION}/techniques/sparse_backends/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            if not member.name.startswith(prefix) or member.isdir():
                continue
            relative = Path(member.name.removeprefix(prefix))
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("unsupported Sol source entry")
            if member.size > 4 * 1024 * 1024:
                raise RuntimeError("unexpectedly large Sol source file")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    def run(*args: str, cwd: Path | None = None, timeout: int = 900) -> None:
        subprocess.run(args, cwd=cwd, check=True, timeout=timeout)

    run(sys.executable, "-m", "pip", "install", *DEPENDENCIES)
    with tempfile.TemporaryDirectory(prefix="vflash-sol-") as folder:
        package = Path(folder)
        fetch_package(package)
        patch_stream_abi(package / "sol_attn/interface.py")
        run(sys.executable, "-m", "pip", "install", "--no-build-isolation", str(package))
    run(sys.executable, "-c", "import sol_attn, torch; assert not torch.cuda.is_initialized()")


if __name__ == "__main__":
    main()
