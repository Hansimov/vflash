"""Explicit architecture views over already ingested original-v0.1 assets.

No weight download, GPU allocation, source hardlinks or payload rehashing occurs.
Only the device target and source lineage change; tensor payloads stay pinned.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from vflash.hardware_profiles import REMOTE_ARCHITECTURES, base_profile, target_id
from vflash.model_assets import model_profile, weights_source
from vflash.native.h3_artifact_contract import resolve_h3_artifact_target
from vflash.native.h3_runtime_artifact import load_h3_runtime_artifact
from vflash.native.h3_schedule_overlay import load_h3_schedule_overlay
from vflash.pipeline.assets import prepare_pipeline_assets
from vflash.pipeline.contracts import PipelineAssets


def prepare_portable_assets(assets: PipelineAssets, directory: Path, architecture: str):
    """Build a new local receipt for one explicit I2VA preview architecture.

    The caller owns the immutable source cache and keeps it mounted read-only.
    Existing destination state is never overwritten. Full-card/MIG capacity and
    complete-video qualification remain separate from this metadata operation.
    """
    if architecture not in REMOTE_ARCHITECTURES:
        raise ValueError("unsupported portable architecture")
    original = load_h3_runtime_artifact(assets.artifact, verify_content_hashes=False)
    load_h3_schedule_overlay(assets.schedule_overlay, artifact=original)
    profile = "i2va-turbo4-v01-544-exact-" + architecture
    if (
        original.weight_profile != "lightx-turbo4-v0.1-544"
        or original.nfe != 4
        or original.source != weights_source(base_profile(profile))
    ):
        raise ValueError("portable assets require the pinned original-v0.1 SM89 source")
    directory.mkdir(parents=True, exist_ok=False)
    artifact, schedule = directory / "artifact", directory / "schedule"
    artifact.mkdir()
    schedule.mkdir()
    # A hardlink would change the serving source's ctime and invalidate receipts.
    (artifact / "blocks").symlink_to(
        (assets.artifact / "blocks").resolve(), target_is_directory=True
    )
    source_manifest = assets.artifact / "artifact.json"
    metadata = json.loads(source_manifest.read_bytes())
    metadata.update(
        schema_version=7,
        artifact_id="h3-runtime-portable-"
        + architecture
        + "-"
        + hashlib.sha256(source_manifest.read_bytes()).hexdigest()[:16],
        target=asdict(resolve_h3_artifact_target(target_id(architecture))),
        source=weights_source(profile),
        portable_from={
            "artifact_id": original.artifact_id,
            "target_id": original.target.target_id,
            "source": original.source,
            "manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        },
    )
    (artifact / "artifact.json").write_text(json.dumps(metadata))
    overlay = json.loads((assets.schedule_overlay / "overlay.json").read_bytes())
    for row in [*overlay["blocks"], overlay["auxiliary"]]:
        source = assets.schedule_overlay / row["path"]
        destination = schedule / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    overlay["overlay_id"] = (
        "h3-schedule-portable-"
        + architecture
        + "-"
        + hashlib.sha256(json.dumps(overlay, sort_keys=True).encode()).hexdigest()[:16]
    )
    overlay["base_artifact"].update(
        artifact_id=metadata["artifact_id"], target_id=metadata["target"]["target_id"]
    )
    overlay["source"].update(
        source_artifact_id=metadata["artifact_id"], compile_recipe=model_profile(profile).recipe
    )
    (schedule / "overlay.json").write_text(json.dumps(overlay))
    fields = assets.to_mapping()
    fields.update(artifact=str(artifact), schedule_overlay=str(schedule))
    return prepare_pipeline_assets(
        PipelineAssets.from_mapping(fields),
        directory / "prepared.json",
        profile_id=profile,
        verify_content_hashes=False,
    )
