"""Once-only verification of official raw H3 weights before asset compilation."""

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
from vflash.model_assets import (
    DEFAULT_MODEL_PROFILE,
    file_identity,
    model_profile,
    upstream_inventory,
    weights_source,
)


def _required_files(
    transformer: Path, adapter: Path | None, profile_id: str = DEFAULT_MODEL_PROFILE
) -> dict[str, tuple[Path, dict[str, Any]]]:
    profile = model_profile(profile_id)
    prefix = profile.transformer_component + "/"
    rows = {
        name: (transformer / name.removeprefix(prefix), expected)
        for name, expected in upstream_inventory().items()
        if name.startswith(prefix)
    }
    contract = profile.adapter
    if contract is not None:
        if adapter is None:
            raise ContractError("this compiler profile requires its pinned adapter")
        rows["adapter"] = (adapter, {"size": contract.size_bytes, "sha256": contract.sha256})
    elif adapter is not None:
        raise ContractError("the Base compiler profile does not accept an adapter")
    return rows


@dataclass(frozen=True)
class PreparedWeights:
    transformer_directory: Path
    adapter_path: Path | None
    receipt: Path
    receipt_sha256: str
    inventory: tuple[dict[str, Any], ...]
    profile_id: str = DEFAULT_MODEL_PROFILE

    def check_unchanged(self) -> None:
        for row in self.inventory:
            if file_identity(Path(row["path"])) != row["identity"]:
                raise ContractError("a prepared compiler input changed; verify weights again")


def prepare_weights(
    transformer_directory: Path,
    adapter_path: Path | None,
    receipt: Path,
    *,
    profile_id: str = DEFAULT_MODEL_PROFILE,
    progress: Callable[[int, int], None] | None = None,
) -> PreparedWeights:
    """Hash the pinned transformer shards and LoRA without importing CUDA libraries."""
    if receipt.exists() or receipt.is_symlink():
        raise ContractError("the weights receipt already exists")
    transformer_directory = transformer_directory.resolve(strict=True)
    adapter_path = adapter_path.resolve(strict=True) if adapter_path is not None else None
    required = _required_files(transformer_directory, adapter_path, profile_id)
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
        "kind": "h3-official-weights",
        "profile_id": profile_id,
        "source": weights_source(profile_id),
        "transformer_directory": str(transformer_directory),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "files": rows,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.link(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return load_prepared_weights(receipt)


def load_prepared_weights(receipt: Path) -> PreparedWeights:
    data = receipt.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("invalid raw-weights receipt") from exc
    if not isinstance(value, dict):
        raise ContractError("invalid raw-weights receipt")
    legacy = value.get("kind") == "h3-ref4-official-weights"
    fields = {
        "schema_version",
        "kind",
        "source",
        "transformer_directory",
        "adapter_path",
        "files",
    }
    profile_id = DEFAULT_MODEL_PROFILE if legacy else value.get("profile_id")
    model_profile(profile_id)
    if (
        set(value) != (fields if legacy else fields | {"profile_id"})
        or value["schema_version"] != 1
        or value["kind"] != ("h3-ref4-official-weights" if legacy else "h3-official-weights")
        or value["source"] != weights_source(profile_id)
        or not isinstance(value["files"], list)
    ):
        raise ContractError("raw-weights receipt differs from the fixed compiler contract")
    if not isinstance(value["transformer_directory"], str) or not Path(
        value["transformer_directory"]
    ).is_absolute():
        raise ContractError("raw-weights receipt requires an absolute transformer path")
    adapter_value = value["adapter_path"]
    if adapter_value is not None and (
        not isinstance(adapter_value, str) or not Path(adapter_value).is_absolute()
    ):
        raise ContractError("raw-weights receipt adapter path must be absolute or null")
    transformer = Path(value["transformer_directory"])
    adapter = Path(adapter_value) if adapter_value is not None else None
    required = _required_files(transformer, adapter, profile_id)
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
    result = PreparedWeights(
        transformer,
        adapter,
        receipt,
        hashlib.sha256(data).hexdigest(),
        tuple(value["files"]),
        profile_id,
    )
    result.check_unchanged()
    return result


# Keep the released single-profile Python entrypoints working. New callers use
# the profile-bound names above; a Ref4 loader never accepts a Base receipt.
PreparedRef4Weights = PreparedWeights


def prepare_ref4_weights(
    transformer_directory: Path,
    adapter_path: Path,
    receipt: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> PreparedWeights:
    return prepare_weights(transformer_directory, adapter_path, receipt, progress=progress)


def load_prepared_ref4_weights(receipt: Path) -> PreparedWeights:
    prepared = load_prepared_weights(receipt)
    if prepared.profile_id != DEFAULT_MODEL_PROFILE:
        raise ContractError("the Ref4 SM89 entrypoint requires its matching weights receipt")
    return prepared
