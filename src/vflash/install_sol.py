"""Install the pinned SM89 Sol dependency into the active Python environment.

Explicit setup only: package import never installs dependencies or initializes CUDA.
The upstream MIT implementation is fetched, not vendored into Vflash.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REVISION = "d0c0a4685ab5dc2336d18b7213d85f13def92418"
DEPENDENCIES = ("nvidia-cutlass-dsl==4.5.0", "cuda-python==13.2.0", "apache-tvm-ffi==0.1.11")


def patch_stream_abi(path: Path) -> None:
    source = path.read_text()
    old = "                stream=stream,"
    if source.count(old) != 4:
        raise RuntimeError("pinned Sol stream ABI differs from the reviewed four call sites")
    path.write_text(source.replace(old, "                stream,"))


def fetch_package(destination: Path) -> None:
    """Materialize only the pinned small source tree, with one total deadline."""
    if shutil.which("git") is None:
        raise RuntimeError("install Git before installing the optional Sol dependency")
    deadline = time.monotonic() + 120
    with tempfile.TemporaryDirectory(prefix="vflash-sol-source-") as folder:
        repository = Path(folder)

        def git(*args: str) -> str:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Sol source download deadline exceeded")
            return subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=remaining,
            ).stdout

        git("init", "--quiet")
        git("remote", "add", "origin", "https://github.com/NVlabs/Sana.git")
        git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", REVISION)
        if git("rev-parse", "FETCH_HEAD").strip() != REVISION:
            raise RuntimeError("Sol source revision differs from the pinned commit")
        prefix = "techniques/sparse_backends/"
        entries = git("ls-tree", "-r", "-z", "FETCH_HEAD", "--", prefix).split("\0")
        names = []
        for entry in filter(None, entries):
            metadata, name = entry.split("\t", 1)
            mode, kind, _ = metadata.split()
            relative = Path(name.removeprefix(prefix))
            if (
                mode != "100644"
                or kind != "blob"
                or not name.startswith(prefix)
                or relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
            ):
                raise RuntimeError("unsupported pinned Sol source entry")
            names.append(relative)
        if (
            not {Path("pyproject.toml"), Path("sol_attn/interface.py")} <= set(names)
            or len(names) > 128
            or len(set(names)) != len(names)
        ):
            raise RuntimeError("pinned Sol source inventory differs")
        # Partial clone fetches these blobs together; unrelated media/history
        # never enters the worktree. Git validates their pinned object identities.
        git("checkout", "FETCH_HEAD", "--", prefix)
        source = repository / prefix
        sizes = [(source / name).stat().st_size for name in names]
        if any(size > 256 * 1024 for size in sizes) or sum(sizes) > 2 * 1024 * 1024:
            raise RuntimeError("pinned Sol source exceeds installation budget")
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write((source / name).read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dependencies-only",
        action="store_true",
        help="Install pinned Python dependencies without fetching source",
    )
    mode.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="Reuse already installed pinned dependencies",
    )
    args = parser.parse_args()

    def run(*args: str, cwd: Path | None = None, timeout: int = 900) -> None:
        subprocess.run(args, cwd=cwd, check=True, timeout=timeout)

    if not args.skip_dependencies:
        run(sys.executable, "-m", "pip", "install", *DEPENDENCIES)
    if args.dependencies_only:
        return
    with tempfile.TemporaryDirectory(prefix="vflash-sol-") as folder:
        package = Path(folder)
        fetch_package(package)
        patch_stream_abi(package / "sol_attn/interface.py")
        run(sys.executable, "-m", "pip", "install", "--no-build-isolation", str(package))
    run(sys.executable, "-c", "import sol_attn, torch; assert not torch.cuda.is_initialized()")


if __name__ == "__main__":
    main()
