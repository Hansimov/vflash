"""Install the pinned standalone Veda core without ComfyUI or model downloads.

Upstream's flat-layout project does not build as a wheel. This explicit installer
retains its independent core and notices, adding only packaging and a revision
marker. Importing Vflash never installs code or initializes CUDA.
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

REVISION = "fd59c7277ccc37ebf1a8f6823474b8c2ef2e33a0"


def fetch_package(destination: Path) -> None:
    request = Request(
        f"https://codeload.github.com/veda-sparse/Veda-on-ComfyUI/tar.gz/{REVISION}",
        headers={"User-Agent": "vflash-installer"},
    )
    with urlopen(request, timeout=60) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise RuntimeError("Veda source archive exceeds the installation budget")
    prefix = f"Veda-on-ComfyUI-{REVISION}/"
    retained = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            if not member.name.startswith(prefix) or member.isdir():
                continue
            relative = Path(member.name.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise RuntimeError("unsupported Veda source entry")
            name = relative.as_posix()
            selected = name in {
                "LICENSE",
                "NOTICE.md",
                "veda_comfy/__init__.py",
                "veda_comfy/hardware.py",
            } or (
                name.endswith(".py")
                and name.startswith(
                    ("veda_comfy/core/", "veda_comfy/backends/", "veda_comfy/kernels/")
                )
            )
            if not selected:
                continue
            if member.size > 1024 * 1024 or name in retained:
                raise RuntimeError("unexpected Veda source file")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
            retained.add(name)
    if (
        not {
            "LICENSE",
            "NOTICE.md",
            "veda_comfy/core/engine.py",
            "veda_comfy/backends/triton_int8.py",
        }
        <= retained
    ):
        raise RuntimeError("incomplete pinned Veda source archive")
    (destination / "veda_comfy/_vflash_pin.py").write_text(f"REVISION = {REVISION!r}\n")
    (destination / "pyproject.toml").write_text("""[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"
[project]
name = "vflash-veda-core"
version = "0.2.0.post1"
requires-python = ">=3.11"
license = "MIT AND BSD-3-Clause"
license-files = ["LICENSE", "NOTICE.md", "SAGE-LICENSE"]
[tool.setuptools.packages.find]
include = ["veda_comfy*"]
""")
    # The preserved Veda NOTICE identifies SageAttention 1.0.6, not newer
    # releases that changed license. Keep that version's license verbatim.
    from importlib.resources import files

    (destination / "SAGE-LICENSE").write_text(
        files("vflash").joinpath("data/sageattention-1.0.6-LICENSE.txt").read_text()
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    with tempfile.TemporaryDirectory(prefix="vflash-veda-") as folder:
        package = Path(folder)
        fetch_package(package)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(package)],
            check=True,
            timeout=600,
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import veda_comfy.core.engine, torch; assert not torch.cuda.is_initialized()",
        ],
        check=True,
        timeout=60,
    )


if __name__ == "__main__":
    main()
