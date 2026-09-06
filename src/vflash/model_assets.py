"""Pinned official H3 file identities shared by ingestion and compilation."""

from __future__ import annotations

import hashlib
import json
import stat
from importlib.resources import files
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError
from vflash.native.h3_distilled_lora import LIGHTX_H3_REF_TURBO4_CONTRACT

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DIFFUSERS_REVISION = "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def upstream_inventory() -> dict[str, dict[str, Any]]:
    payload = json.loads(files("vflash").joinpath("data/h3-pipeline-assets.json").read_text())
    if payload["repository"] != "MiniMaxAI/MiniMax-H3" or payload["revision"] != MODEL_REVISION:
        raise ContractError("the bundled pipeline asset revision changed")
    return {row["path"]: row for row in payload["files"]}


def file_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise ContractError(f"pipeline asset must be a nonempty regular file: {path.name}")
    return {
        "size": value.st_size,
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _ref4_base_inventory() -> dict[str, Any]:
    inventory = upstream_inventory()
    transformer_files = [
        {key: row[key] for key in ("path", "size", "sha256")}
        for path, row in sorted(inventory.items())
        if path.startswith("transformer_ref/")
    ]
    return {
        "repository": "MiniMaxAI/MiniMax-H3",
        "revision": MODEL_REVISION,
        "transformer_files": transformer_files,
    }


def ref4_transformer_identity() -> dict[str, str]:
    # Existing conditioning bundles bind this combined identity. The compiler
    # also records a base-only digest, keeping LoRA ownership explicit.
    identity = {
        **_ref4_base_inventory(),
        "adapter_sha256": LIGHTX_H3_REF_TURBO4_CONTRACT.sha256,
    }
    return {
        "model_repository": "MiniMaxAI/MiniMax-H3",
        "model_revision": MODEL_REVISION,
        "transformer_sha256": canonical_sha256(identity),
        "oracle": "diffusers",
        "oracle_revision": DIFFUSERS_REVISION,
        "oracle_profile": "ref2va-adapter-bf16-torch-sdpa-sm89",
    }


def ref4_weights_source() -> dict[str, str]:
    """Weights provenance plus the required conditioning math; no request capture."""
    contract = LIGHTX_H3_REF_TURBO4_CONTRACT
    return {
        **ref4_transformer_identity(),
        "source_kind": "official-weights-v1",
        "compile_recipe": "ref4-bf16-runtime-residual-sm89-v1",
        "base_transformer_sha256": canonical_sha256(_ref4_base_inventory()),
        "adapter_repository": contract.repository,
        "adapter_revision": contract.revision,
        "adapter_sha256": contract.sha256,
        "adapter_rank": str(contract.rank),
        "adapter_alpha": format(contract.alpha, "g"),
        "adapter_strength": format(contract.strength, "g"),
    }
