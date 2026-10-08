"""Explicit operator-trusted lineage for exact consumed H3 tensor subsets.

No source digest is relabeled as a subset digest. Preparation binds the small
conversion receipt and immutable local file identity without rehashing weights.
The caller is responsible for trusted byte-preserving conversion and storage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vflash.contracts import ContractError


def apply_exact_subset(path: Path, planned):
    if not isinstance(path, Path) or not path.is_absolute():
        raise ContractError("an explicit absolute derivation path is required")
    if path.stat().st_size > 2 * 1024**2:
        raise ContractError("derived asset receipt exceeds bound")
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "exact-h3-consumed-subset"
        or value.get("source_revision") != "42ed227ee7df40d41602854ae760620d6eb651fe"
        or value.get("retained_text_layers") != 51
        or value.get("text_hidden_state") != 50
        or value.get("weights_rehashed") is not False
        or value.get("source_files_modified") is not False
    ):
        raise ContractError("unknown exact asset derivation")
    changes, omitted = value["changes"], value["omitted"]
    if (
        not isinstance(changes, dict)
        or not 1 <= len(changes) <= 32
        or not isinstance(omitted, list)
        or len(set(omitted)) != len(omitted)
        or any(
            not key.startswith("model/text_encoder/model-") or not key.endswith(".safetensors")
            for key in omitted
        )
    ):
        raise ContractError("invalid compact asset inventory")
    result, seen, removed = [], set(), set()
    for role, file, expected in planned:
        key = "adapter.safetensors" if role == "adapter" else "model/" + role
        if key in omitted:
            removed.add(key)
            continue
        if key in changes:
            change = changes[key]
            allowed = key == "adapter.safetensors" or key.startswith(
                ("model/text_encoder/", "model/transformer/", "model/vae/")
            )
            identity_matches = (
                change["source_identity"] == expected["sha256"]
                if expected.get("sha256")
                else bool(expected.get("git_blob_sha1"))
                and change.get("source_git_blob_sha1") == expected["git_blob_sha1"]
            )
            if (
                not allowed
                or change["source_size"] != expected.get("size")
                or not identity_matches
                or change["kind"]
                not in {"exact-tensor-subset", "text-config-51", "subset-index"}
                or type(change["size"]) is not int
                or not 0 < change["size"] <= change["source_size"]
            ):
                raise ContractError("derived file does not match its pinned source lineage")
            expected = {"size": change["size"], "sha256": None}
            seen.add(key)
        result.append((role, file, expected))
    if seen != set(changes) or removed != set(omitted):
        raise ContractError("derived receipt changes files outside the consumed asset set")
    result.append(
        (
            "derivation/exact-subset",
            path,
            {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        )
    )
    return result
