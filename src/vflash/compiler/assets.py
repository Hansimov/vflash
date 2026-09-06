"""Once-only verification of official raw Ref4 weights before asset compilation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError
from vflash.model_assets import file_identity, ref4_weights_source, upstream_inventory
from vflash.native.h3_distilled_lora import LIGHTX_H3_REF_TURBO4_CONTRACT


def _required_files(transformer: Path, adapter: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    rows = {
        name: (transformer / name.removeprefix("transformer_ref/"), expected)
        for name, expected in upstream_inventory().items()
        if name.startswith("transformer_ref/")
    }
    contract = LIGHTX_H3_REF_TURBO4_CONTRACT
    rows["adapter"] = (adapter, {"size": contract.size_bytes, "sha256": contract.sha256})
    return rows


@dataclass(frozen=True)
class PreparedRef4Weights:
    transformer_directory: Path
    adapter_path: Path
    receipt: Path
    receipt_sha256: str
    inventory: tuple[dict[str, Any], ...]

    def check_unchanged(self) -> None:
        for row in self.inventory:
            if file_identity(Path(row["path"])) != row["identity"]:
                raise ContractError("a prepared compiler input changed; verify weights again")


def prepare_ref4_weights(
    transformer_directory: Path,
    adapter_path: Path,
    receipt: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> PreparedRef4Weights:
    """Hash the pinned transformer shards and LoRA without importing CUDA libraries."""
    if receipt.exists() or receipt.is_symlink():
        raise ContractError("the weights receipt already exists")
    transformer_directory = transformer_directory.resolve(strict=True)
    adapter_path = adapter_path.resolve(strict=True)
    required = _required_files(transformer_directory, adapter_path)
    rows = []
    for index, (role, (path, expected)) in enumerate(sorted(required.items()), 1):
        path = path.resolve(strict=True)
        before = file_identity(path)
        if before["size"] != expected["size"]:
            raise ContractError(f"raw weight size differs from its official source: {role}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if file_identity(path) != before or digest.hexdigest() != expected["sha256"]:
            raise ContractError(f"raw weight bytes differ from their official source: {role}")
        rows.append(
            {"role": role, "path": str(path), "sha256": digest.hexdigest(), "identity": before}
        )
        if progress is not None:
            progress(index, len(required))
    value = {
        "schema_version": 1,
        "kind": "h3-ref4-official-weights",
        "source": ref4_weights_source(),
        "transformer_directory": str(transformer_directory),
        "adapter_path": str(adapter_path),
        "files": rows,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.link(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return load_prepared_ref4_weights(receipt)


def load_prepared_ref4_weights(receipt: Path) -> PreparedRef4Weights:
    data = receipt.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("invalid raw-weights receipt") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "kind",
            "source",
            "transformer_directory",
            "adapter_path",
            "files",
        }
        or value["schema_version"] != 1
        or value["kind"] != "h3-ref4-official-weights"
        or value["source"] != ref4_weights_source()
        or not isinstance(value["files"], list)
    ):
        raise ContractError("raw-weights receipt differs from the fixed compiler contract")
    for name in ("transformer_directory", "adapter_path"):
        if not isinstance(value[name], str) or not Path(value[name]).is_absolute():
            raise ContractError("raw-weights receipt requires absolute local paths")
    transformer, adapter = Path(value["transformer_directory"]), Path(value["adapter_path"])
    required = _required_files(transformer, adapter)
    seen = set()
    for row in value["files"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"role", "path", "sha256", "identity"}
            or not isinstance(row["role"], str)
            or row["role"] not in required
            or row["role"] in seen
        ):
            raise ContractError("raw-weights receipt contains an invalid or duplicate file")
        path, expected = required[row["role"]]
        if (
            row["path"] != str(path.resolve(strict=True))
            or row["sha256"] != expected["sha256"]
            or not isinstance(row["identity"], dict)
            or set(row["identity"]) != {"size", "device", "inode", "mtime_ns", "ctime_ns"}
            or any(type(item) is not int for item in row["identity"].values())
            or row["identity"]["size"] != expected["size"]
        ):
            raise ContractError(
                "raw-weights receipt does not identify the required official files"
            )
        seen.add(row["role"])
    if seen != set(required):
        raise ContractError("raw-weights receipt is incomplete")
    result = PreparedRef4Weights(
        transformer, adapter, receipt, hashlib.sha256(data).hexdigest(), tuple(value["files"])
    )
    result.check_unchanged()
    return result
