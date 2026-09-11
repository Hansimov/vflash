"""Verify model assets once, then reopen immutable local files without rehashing weights."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.adapters.checkpoints import read_weight_map
from vflash.contracts import ContractError
from vflash.model_assets import (
    canonical_sha256,
    conditioning_profile_for_request,
    model_profile,
    transformer_identity,
    upstream_inventory,
)
from vflash.model_assets import (
    file_identity as _stamp,
)
from vflash.native.h3_conditioning_bundle import (
    H3_FIRST_FRAME_POLICY,
    H3_FL2VA_KEYFRAME_POLICY,
    H3_VIDEO_REFERENCE_POLICY,
)
from vflash.native.h3_runtime_artifact import load_h3_runtime_artifact
from vflash.native.h3_schedule_overlay import load_h3_schedule_overlay
from vflash.pipeline.contracts import (
    PIPELINE_PROFILE,
    PipelineAssets,
)


def conditioning_source(
    *,
    profile_id: str = PIPELINE_PROFILE,
    request_mode: str | None = None,
    runtime_versions: dict[str, str] | None = None,
    reference_policy: str = "match",
) -> dict[str, str]:
    source_profile_id = (
        profile_id
        if request_mode is None
        else conditioning_profile_for_request(profile_id, request_mode)
    )
    profile = model_profile(source_profile_id)
    if reference_policy not in {"match", H3_VIDEO_REFERENCE_POLICY} or (
        reference_policy == H3_VIDEO_REFERENCE_POLICY
        and profile_id != "ref2va-turbo4-exact-sm89"
    ):
        raise ContractError("conditioning reference policy differs from the fixed profile")
    configuration = {
        "profile_id": source_profile_id,
        "reference_policy_revision": 3,
        "text_precision": "bf16",
        "text_execution": "full",
        "diffusers_workflow": profile.workflow,
        "video_flow_shift": profile.definition.video_flow_shift,
        "audio_flow_shift": profile.definition.audio_flow_shift,
        "scheduler": "training_euler",
        "nfe": profile.definition.nfe,
    }
    if profile.definition.mode.value == "i2va":
        if reference_policy != "match":
            raise ContractError("I2VA accepts one first frame, not a reference policy")
        configuration["first_frame_policy"] = H3_FIRST_FRAME_POLICY
        configuration["reference_policy_revision"] = 1
    elif profile.definition.mode.value == "fl2va":
        if reference_policy != "match":
            raise ContractError("FL2VA accepts first and last frames, not a reference policy")
        configuration["keyframe_policy"] = H3_FL2VA_KEYFRAME_POLICY
        configuration["reference_policy_revision"] = 1
    elif reference_policy == H3_VIDEO_REFERENCE_POLICY:
        # Preserve the released image/T2 identity, but never claim a video
        # capture used image-only match preprocessing.
        configuration["reference_video_policy"] = reference_policy
        configuration["reference_policy_revision"] = 1
    else:
        configuration["reference_image_policy"] = "match"
    return {
        **transformer_identity(source_profile_id),
        "oracle_config_sha256": canonical_sha256(configuration),
        "oracle_hardware": f"{profile.architecture}-single",
        "oracle_runtime_sha256": canonical_sha256(runtime_versions or {}),
    }


def _consumed_official_files(
    assets: PipelineAssets, profile_id: str = PIPELINE_PROFILE
) -> list[tuple[str, Path]]:
    inventory = upstream_inventory()
    model, decoder = assets.model_directory, assets.decoder_directory
    prefixes = ("text_encoder/", "processor/", "tokenizer/", "scheduler/", "audio_scheduler/")
    names = {name for name in inventory if name.startswith(prefixes)}
    names.update({"model_index.json", "audio_vae/config.json"})
    for component, prefixes in (
        (
            model_profile(profile_id).transformer_component,
            (
                "proj_in.",
                "audio_proj_in.",
                "context_embedder.",
                "time_embedder.",
                "token_refiner.",
            ),
        ),
        ("vae", ("encoder.", "quant_conv.")),
    ):
        index = "diffusion_pytorch_model.safetensors.index.json"
        weights = read_weight_map(model / component, index)
        selected = {shard for name, shard in weights.items() if name.startswith(prefixes)}
        if not selected:
            raise ContractError(f"the {component} index has no conditioning weights")
        names.update(f"{component}/{name}" for name in {index, "config.json", *selected})
    if (model / "modular_model_index.json").exists():
        names.add("modular_model_index.json")
    rows = [(name, model / name) for name in sorted(names)]
    rows.extend(
        (name, decoder / name.removeprefix("FL2VA/"))
        for name in sorted(inventory)
        if name.startswith(("FL2VA/video_vae/", "FL2VA/audio_vae/"))
    )
    if any(name not in inventory for name, _path in rows):
        raise ContractError("a component requests files outside the pinned upstream snapshot")
    return rows


@dataclass(frozen=True)
class PreparedPipelineAssets:
    assets: PipelineAssets
    receipt: Path
    receipt_sha256: str
    inventory: tuple[dict[str, Any], ...]
    profile_id: str = PIPELINE_PROFILE

    def check_unchanged(self) -> None:
        """Invalidate receipts after replacement, modification or relocation of an asset."""
        for row in self.inventory:
            try:
                current = _stamp(Path(row["path"]))
            except OSError as exc:
                raise ContractError(
                    "a prepared asset is missing; prepare the assets again"
                ) from exc
            if current != row["stamp"]:
                raise ContractError("a prepared asset changed; prepare the assets again")


def _planned_files(
    assets: PipelineAssets, profile_id: str = PIPELINE_PROFILE
) -> list[tuple[str, Path, dict[str, Any]]]:
    upstream = upstream_inventory()
    profile = model_profile(profile_id)
    planned = [
        (name, path, upstream[name])
        for name, path in _consumed_official_files(assets, profile_id)
    ]
    contract = profile.adapter
    if (contract is None) != (assets.adapter_path is None):
        raise ContractError("the adapter asset differs from the complete pipeline profile")
    if contract is not None:
        planned.append(
            (
                "adapter",
                assets.adapter_path,
                {
                    "size": contract.size_bytes,
                    "sha256": contract.sha256,
                },
            )
        )
    artifact = load_h3_runtime_artifact(assets.artifact, verify_content_hashes=False)
    overlay = load_h3_schedule_overlay(assets.schedule_overlay, artifact=artifact)
    source = conditioning_source(profile_id=profile_id)
    identity_keys = (
        "model_repository",
        "model_revision",
        "transformer_sha256",
        "oracle",
        "oracle_revision",
    )
    if (
        artifact.weight_profile != profile.weight_profile
        or artifact.nfe != profile.definition.nfe
        or artifact.adapter_execution != profile.adapter_execution
        or not artifact.is_complete_block_stack
        or profile.architecture != artifact.target.compute_capability
        or any(artifact.source.get(key) != source[key] for key in identity_keys)
        or (
            contract is not None
            and artifact.source.get("adapter_revision") != contract.revision
        )
        or artifact.source.get("oracle_profile", "").removesuffix("-sm89").removesuffix("-sm86")
        != source["oracle_profile"].removesuffix("-sm89").removesuffix("-sm86")
    ):
        raise ContractError("the native artifact differs from the complete pipeline profile")
    from vflash.native.h3_native_conditioning_runtime import (
        H3NativeConditioningRuntimeError,
        validate_declared_schedule,
    )

    try:
        validate_declared_schedule(
            overlay.schedule,
            expected_nfe=profile.definition.nfe,
            expected_scheduler="h3-training-euler",
            expected_video_flow_shift=profile.definition.video_flow_shift,
            expected_audio_flow_shift=profile.definition.audio_flow_shift,
        )
    except H3NativeConditioningRuntimeError as exc:
        raise ContractError(str(exc)) from exc
    planned.extend(
        (
            f"native/{row.path}",
            assets.artifact / row.path,
            {
                "size": row.size_bytes,
                "sha256": row.sha256,
            },
        )
        for row in artifact.blocks
    )
    # These small files are already structurally validated by their native
    # loaders; bind their local bytes and identity in the same ingestion pass.
    planned.append(("native/artifact.json", assets.artifact / "artifact.json", {}))
    planned.append(("native/auxiliary", assets.auxiliary_tensor, {}))
    planned.extend(
        (f"overlay/{path.relative_to(assets.schedule_overlay)}", path, {})
        for path in sorted(assets.schedule_overlay.rglob("*"))
        if path.is_file()
    )
    return planned


def prepare_pipeline_assets(
    assets: PipelineAssets, receipt: Path, *, profile_id: str = PIPELINE_PROFILE
) -> PreparedPipelineAssets:
    """Verify consumed official files and compiled blocks at an explicit ingestion boundary.

    The receipt belongs to this filesystem instance. Use a read-only asset snapshot
    before preparation. Startup compares file identity and timestamps, and never treats
    a local filename or safetensors metadata string as proof of upstream weights.
    """
    if receipt.exists() or receipt.is_symlink():
        raise ContractError("the asset receipt already exists")
    planned = _planned_files(assets, profile_id)
    inventory = []
    for role, path, expected in planned:
        before = _stamp(path)
        if expected.get("size", before["size"]) != before["size"]:
            raise ContractError(f"asset size differs from its pinned source: {role}")
        digest = hashlib.sha256()
        blob = hashlib.sha1(f"blob {before['size']}\0".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                if "git_blob_sha1" in expected:
                    blob.update(chunk)
        if (
            _stamp(path) != before
            or expected.get("sha256", digest.hexdigest()) != digest.hexdigest()
            or expected.get("git_blob_sha1", blob.hexdigest()) != blob.hexdigest()
        ):
            raise ContractError(f"asset bytes differ from their pinned source: {role}")
        inventory.append(
            {
                "role": role,
                "path": str(path.resolve(strict=True)),
                "sha256": digest.hexdigest(),
                "stamp": before,
            }
        )
    value = {
        "schema_version": 1,
        "profile_id": profile_id,
        "assets": assets.to_mapping(),
        "inventory": inventory,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.link(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return load_prepared_pipeline_assets(receipt)


def load_prepared_pipeline_assets(receipt: Path) -> PreparedPipelineAssets:
    data = receipt.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("the pipeline asset receipt is invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "profile_id",
            "assets",
            "inventory",
        }
        or value["schema_version"] != 1
        or not isinstance(value["profile_id"], str)
        or not isinstance(value["inventory"], list)
        or not value["inventory"]
    ):
        raise ContractError("the pipeline asset receipt has an unsupported schema")
    profile_id = value["profile_id"]
    model_profile(profile_id)
    assets = PipelineAssets.from_mapping(value["assets"])
    if any(path is not None and not path.is_absolute() for path in vars(assets).values()):
        raise ContractError("prepared asset paths must be absolute")
    planned = {
        role: (path.resolve(strict=True), expected)
        for role, path, expected in _planned_files(assets, profile_id)
    }
    seen: set[str] = set()
    for row in value["inventory"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"role", "path", "sha256", "stamp"}
            or not isinstance(row["role"], str)
            or row["role"] in seen
            or row["role"] not in planned
            or not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["sha256"])
            or not isinstance(row["stamp"], dict)
            or set(row["stamp"]) != {"size", "device", "inode", "mtime_ns", "ctime_ns"}
            or any(type(item) is not int for item in row["stamp"].values())
        ):
            raise ContractError("the pipeline asset receipt contains invalid file identities")
        path, expected = planned[row["role"]]
        if (
            Path(row["path"]) != path
            or row["stamp"]["size"] != expected.get("size", row["stamp"]["size"])
            or row["sha256"] != expected.get("sha256", row["sha256"])
        ):
            raise ContractError("a receipt file differs from the configured asset inventory")
        seen.add(row["role"])
    if seen != set(planned):
        raise ContractError("the receipt does not cover all required pipeline assets")
    result = PreparedPipelineAssets(
        assets,
        receipt,
        hashlib.sha256(data).hexdigest(),
        tuple(value["inventory"]),
        profile_id,
    )
    result.check_unchanged()
    return result
