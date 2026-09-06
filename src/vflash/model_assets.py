"""Pinned official H3 file identities shared by ingestion and compilation."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError, HardwareTarget, Profile
from vflash.native.h3_distilled_lora import (
    H3DistilledLoraContract,
    h3_distilled_lora_contract_for_profile,
)

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


DEFAULT_MODEL_PROFILE = "ref2va-turbo4-exact-sm89"
COMPLETE_MODEL_PROFILES = (
    DEFAULT_MODEL_PROFILE,
    "ref2va-turbo4-exact-sm86",
    "t2va-turbo4-exact-sm89",
)


@dataclass(frozen=True)
class H3ModelProfile:
    """One explicit binding of base weights, adapter, schedule and compile hardware."""

    definition: Profile
    hardware: HardwareTarget
    adapter: H3DistilledLoraContract

    @property
    def transformer_component(self) -> str:
        return "transformer_ref" if self.definition.mode.value == "ref2va" else "transformer"

    @property
    def architecture(self) -> str:
        return "sm" + self.hardware.compute_capability.replace(".", "")

    @property
    def recipe(self) -> str:
        family = "ref4" if self.definition.mode.value == "ref2va" else "base4"
        return f"{family}-bf16-runtime-residual-{self.architecture}-v1"


def model_profile(profile_id: str = DEFAULT_MODEL_PROFILE) -> H3ModelProfile:
    if profile_id not in COMPLETE_MODEL_PROFILES:
        raise ContractError("unsupported complete-pipeline model profile")
    catalog = ProfileCatalog.bundled()
    definition = catalog.profile(profile_id)
    adapter_id = (
        "lightx-ref-turbo4-v0.1" if definition.mode.value == "ref2va" else "lightx-turbo4-v1.0"
    )
    adapter = h3_distilled_lora_contract_for_profile(adapter_id, workflow=definition.mode.value)
    if (definition.adapter, definition.adapter_revision, definition.nfe) != (
        adapter.repository,
        adapter.revision,
        adapter.nfe,
    ) or len(definition.target_ids) != 1:
        raise ContractError("the complete profile differs from its pinned adapter contract")
    return H3ModelProfile(definition, catalog.target(definition.target_ids[0]), adapter)


def _base_inventory(profile: H3ModelProfile) -> dict[str, Any]:
    prefix = profile.transformer_component + "/"
    transformer_files = [
        {key: row[key] for key in ("path", "size", "sha256")}
        for path, row in sorted(upstream_inventory().items())
        if path.startswith(prefix)
    ]
    if len(transformer_files) != 16:
        raise ContractError("the pinned H3 transformer inventory is incomplete")
    return {
        "repository": "MiniMaxAI/MiniMax-H3",
        "revision": MODEL_REVISION,
        "transformer_files": transformer_files,
    }


def transformer_identity(profile_id: str = DEFAULT_MODEL_PROFILE) -> dict[str, str]:
    profile = model_profile(profile_id)
    identity = {**_base_inventory(profile), "adapter_sha256": profile.adapter.sha256}
    return {
        "model_repository": "MiniMaxAI/MiniMax-H3",
        "model_revision": MODEL_REVISION,
        "transformer_sha256": canonical_sha256(identity),
        "oracle": "diffusers",
        "oracle_revision": DIFFUSERS_REVISION,
        "oracle_profile": (
            f"{profile.definition.mode.value}-adapter-bf16-torch-sdpa-{profile.architecture}"
        ),
    }


def weights_source(profile_id: str = DEFAULT_MODEL_PROFILE) -> dict[str, str]:
    """Request-independent provenance for one fixed compiler contract."""
    profile = model_profile(profile_id)
    contract = profile.adapter
    return {
        **transformer_identity(profile_id),
        "source_kind": "official-weights-v1",
        "compile_recipe": profile.recipe,
        "base_transformer_sha256": canonical_sha256(_base_inventory(profile)),
        "adapter_repository": contract.repository,
        "adapter_revision": contract.revision,
        "adapter_sha256": contract.sha256,
        "adapter_rank": str(contract.rank),
        "adapter_alpha": format(contract.alpha, "g"),
        "adapter_strength": format(contract.strength, "g"),
    }


def weights_source_profile(source: Any) -> H3ModelProfile:
    """Require the whole source object, including the hardware-specific recipe."""
    for profile_id in COMPLETE_MODEL_PROFILES:
        if source == weights_source(profile_id):
            return model_profile(profile_id)
    raise ContractError("weights-only source differs from the fixed Ref4 or Base4 contract")


def ref4_transformer_identity() -> dict[str, str]:
    """The original Ref4 SM89 identity remains byte-for-byte compatible."""
    return transformer_identity()


def ref4_weights_source() -> dict[str, str]:
    return weights_source()


def model_schedule(profile_id: str = DEFAULT_MODEL_PROFILE):
    """The pinned FP32 sigma grid; importing model identity alone stays CPU-light."""
    from vflash.native.h3_native_scheduler import H3NativeSchedule

    profile = model_profile(profile_id).definition
    return H3NativeSchedule.shifted_linear(
        profile.nfe, video_shift=profile.video_flow_shift, audio_shift=profile.audio_flow_shift
    )
