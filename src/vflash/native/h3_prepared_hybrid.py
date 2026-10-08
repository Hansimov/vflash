"""CPU-prepared hybrid modulation bound to one backbone, overlay and source inventory."""

import hashlib
import json
from pathlib import Path


def inspect(directory):
    path = directory / "hybrid-cache.json"
    if path.is_symlink() or path.stat().st_size > 128 * 1024:
        raise ValueError("invalid hybrid cache manifest")
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("preparation_device") != "cpu":
        raise ValueError("unsupported prepared hybrid recipe")
    tables = data.get("tables", [])
    if [row.get("index") for row in tables] != list(range(25, 50)):
        raise ValueError("prepared hybrid must cover exactly blocks 25..49")
    stamps = []
    for row in tables:
        expected = f"block-{row['index']:03d}.safetensors"
        if row.get("file") != expected or row.get("shape") != [4, 9, 6, 5376]:
            raise ValueError("prepared hybrid shape or file differs")
        table = directory / expected
        if table.is_symlink() or table.stat().st_size != row.get("size"):
            raise ValueError("prepared hybrid file differs")
        if table.stat().st_size > 4 * 1024**2:
            raise ValueError("prepared hybrid table exceeds its bound")
        # These are 2.3 MB derived tables, not multi-GB source weights.
        if hashlib.sha256(table.read_bytes()).hexdigest() != row.get("sha256"):
            raise ValueError("prepared hybrid table digest differs")
        st = table.stat()
        stamps.append((str(table), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns))
    st = path.stat()
    stamps.append((str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns))
    return data, tuple(stamps)


def load(directory: Path, artifact, overlay):
    from safetensors.torch import load_file

    from vflash.native.h3_hybrid import HybridModulationOverlay

    data, _ = inspect(directory)
    if (
        data["artifact_id"] != artifact.artifact_id
        or data["overlay_id"] != overlay.overlay_id
        or artifact.weight_profile != "lightx-turbo4-v0.1-544"
        or artifact.nfe != 4
    ):
        raise ValueError("prepared hybrid is bound to a different backbone or schedule")
    tables = {}
    for row in data["tables"]:
        values = load_file(str(directory / row["file"]), device="cpu")
        if set(values) != {"adaln.table"}:
            raise ValueError("unexpected prepared hybrid tensors")
        table = values["adaln.table"]
        if list(table.shape) != row["shape"] or str(table.dtype) != "torch.bfloat16":
            raise ValueError("prepared hybrid tensor shape differs")
        if not table.isfinite().all():
            raise ValueError("prepared hybrid contains nonfinite values")
        tables[row["index"]] = table
    return HybridModulationOverlay(overlay, artifact, tables, data["provenance"])


def prepare(directory: Path, artifact, overlay, checkpoint):
    """Compile the fixed Ref modulation on CPU before reserving a GPU.

    Source weights retain their pinned inventory checks. Only the small derived
    tables are hashed; the source model is never hardlinked or mutated.
    The destination must be new; its manifest is published after all tables.
    """
    import os
    import shutil
    import tempfile

    from safetensors.torch import save_file

    from vflash.native.h3_hybrid import compile_hybrid_modulation

    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise ValueError("prepared hybrid destination already exists")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".hybrid-", dir=directory.parent))
    try:
        compiled = compile_hybrid_modulation(artifact, overlay, checkpoint, device="cpu")
        rows = []
        for index in sorted(compiled.tables):
            table = compiled.tables[index]
            path = temporary / f"block-{index:03d}.safetensors"
            save_file({"adaln.table": table.contiguous()}, str(path))
            rows.append(
                {
                    "index": index,
                    "file": path.name,
                    "shape": list(table.shape),
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        (temporary / "hybrid-cache.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "preparation_device": "cpu",
                    "artifact_id": artifact.artifact_id,
                    "overlay_id": overlay.overlay_id,
                    "provenance": compiled.provenance,
                    "tables": rows,
                }
            )
        )
        load(temporary, artifact, overlay)
        # mkdir is exclusive even if another preparation raced this one.
        directory.mkdir(mode=0o700)
        for path in sorted(temporary.glob("*.safetensors")):
            os.replace(path, directory / path.name)
        os.replace(temporary / "hybrid-cache.json", directory / "hybrid-cache.json")
    finally:
        shutil.rmtree(temporary)
    return load(directory, artifact, overlay)
