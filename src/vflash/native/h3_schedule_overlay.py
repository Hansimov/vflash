"""Read schedule-specific AdaLN tables for one pinned Ref2VA runtime artifact."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.native.h3_native_scheduler import (
    H3NativeSchedule,
)
from vflash.native.h3_runtime_artifact import H3RuntimeArtifact, load_h3_runtime_artifact
from vflash.native.h3_tensor_file import (
    inspect_safetensors_header,
    load_safetensor_tensor,
)

H3_SCHEDULE_OVERLAY_SCHEMA_VERSION = 1
H3_SCHEDULE_OVERLAY_LAYOUT = "h3-base-adaln-table-overlay-v1"
H3_SCHEDULE_OVERLAY_METHOD = "piecewise-linear-exact-source-table-v1"

_OVERLAY_ID = re.compile(r"h3-schedule-[a-z0-9][a-z0-9-]{0,79}")
_SHA256 = re.compile(r"[a-f0-9]{64}")


class H3ScheduleOverlayError(ValueError):
    """A schedule overlay is incomplete or differs from its base artifact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> stat.struct_stat:
    try:
        value = path.lstat()
    except OSError as exc:
        raise H3ScheduleOverlayError(
            f"H3 schedule-overlay file is unavailable: {path}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise H3ScheduleOverlayError(
            f"H3 schedule-overlay file must be a non-empty regular file: {path.name}"
        )
    return value


@dataclass(frozen=True)
class H3ScheduleOverlayBlock:
    index: int
    path: str
    size_bytes: int
    sha256: str
    adaln_rows: int


@dataclass(frozen=True)
class H3ScheduleOverlayAuxiliary:
    path: str
    size_bytes: int
    sha256: str
    timestep_rows: int


@dataclass(frozen=True)
class H3ScheduleOverlay:
    directory: Path
    overlay_id: str
    created_at: str
    status: str
    layout: str
    base_artifact_id: str
    target_id: str
    weight_profile: str
    adapter_execution: str
    schedule: H3NativeSchedule
    source: dict[str, str]
    auxiliary: H3ScheduleOverlayAuxiliary
    blocks: tuple[H3ScheduleOverlayBlock, ...]

    def load_block_table(self, block_index: int) -> Any:
        try:
            block = next(row for row in self.blocks if row.index == block_index)
        except StopIteration as exc:
            raise H3ScheduleOverlayError(
                f"H3 schedule overlay has no block {block_index}"
            ) from exc
        return load_safetensor_tensor(self.directory / block.path, "adaln.table")

    def load_auxiliary_tensors(self) -> dict[str, Any]:
        path = self.directory / self.auxiliary.path
        return {
            name: load_safetensor_tensor(path, name)
            for name in (
                "final_adaln_table",
                "time_embeddings",
                "timestep_counts",
                "timesteps",
            )
        }


def _safe_relative_file(directory: Path, value: Any, *, name: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise H3ScheduleOverlayError(f"H3 schedule-overlay {name} path is unsafe")
    path = directory / value
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise H3ScheduleOverlayError(f"H3 schedule-overlay {name} file is unavailable") from exc
    if not resolved.is_relative_to(directory):
        raise H3ScheduleOverlayError(f"H3 schedule-overlay {name} escapes its directory")
    return resolved


def _validated_file_record(
    directory: Path,
    value: Any,
    *,
    name: str,
) -> tuple[Path, int, str]:
    if not isinstance(value, dict):
        raise H3ScheduleOverlayError(f"H3 schedule-overlay {name} record is invalid")
    path_value = value.get("path")
    size_bytes = value.get("size_bytes")
    sha256 = value.get("sha256")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes <= 0
        or not isinstance(sha256, str)
        or _SHA256.fullmatch(sha256) is None
    ):
        raise H3ScheduleOverlayError(f"H3 schedule-overlay {name} metadata is invalid")
    path = _safe_relative_file(directory, path_value, name=name)
    actual = _regular_file(path)
    if actual.st_size != size_bytes or _sha256(path) != sha256:
        raise H3ScheduleOverlayError(f"H3 schedule-overlay {name} integrity check failed")
    return path, size_bytes, sha256


def load_h3_schedule_overlay(
    directory: Path,
    *,
    artifact: H3RuntimeArtifact | Path,
) -> H3ScheduleOverlay:
    """Load and fully validate an overlay against one complete runtime artifact."""

    base = (
        artifact
        if isinstance(artifact, H3RuntimeArtifact)
        else load_h3_runtime_artifact(artifact)
    )
    try:
        resolved = directory.resolve(strict=True)
    except OSError as exc:
        raise H3ScheduleOverlayError("H3 schedule-overlay directory is unavailable") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise H3ScheduleOverlayError("H3 schedule-overlay path must be a real directory")
    manifest_path = resolved / "overlay.json"
    _regular_file(manifest_path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H3ScheduleOverlayError("H3 schedule-overlay manifest is invalid") from exc
    expected = {
        "schema_version",
        "overlay_id",
        "created_at",
        "status",
        "layout",
        "base_artifact",
        "schedule",
        "source",
        "auxiliary",
        "blocks",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise H3ScheduleOverlayError("H3 schedule-overlay fields differ from schema v1")
    if value.get("schema_version") != H3_SCHEDULE_OVERLAY_SCHEMA_VERSION:
        raise H3ScheduleOverlayError("H3 schedule-overlay schema version is unsupported")
    overlay_id = value.get("overlay_id")
    if not isinstance(overlay_id, str) or _OVERLAY_ID.fullmatch(overlay_id) is None:
        raise H3ScheduleOverlayError("H3 schedule-overlay id is invalid")
    if value.get("layout") != H3_SCHEDULE_OVERLAY_LAYOUT:
        raise H3ScheduleOverlayError("H3 schedule-overlay layout is unsupported")
    if value.get("status") != "source-schedule-exact":
        raise H3ScheduleOverlayError("H3 schedule-overlay status is invalid")
    if not isinstance(value.get("created_at"), str) or not value["created_at"]:
        raise H3ScheduleOverlayError("H3 schedule-overlay timestamp is invalid")

    base_value = value.get("base_artifact")
    expected_base = {
        "artifact_id": base.artifact_id,
        "target_id": base.target.target_id,
        "weight_profile": base.weight_profile,
        "adapter_execution": base.adapter_execution,
    }
    if base_value != expected_base:
        raise H3ScheduleOverlayError("H3 schedule overlay differs from its base artifact")
    if not base.is_complete_block_stack:
        raise H3ScheduleOverlayError("schema-v1 schedule overlays require a complete trunk")
    schedule = H3NativeSchedule.from_mapping(value.get("schedule"))
    if schedule.nfe != base.nfe:
        raise H3ScheduleOverlayError(
            "adapted H3 artifacts accept only their exact source schedule"
        )
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "method",
            "source_artifact_id",
            "source_packed_input_sha256",
            "source_replay_case_id",
            "transformer_tensor_file",
            "transformer_tensor_sha256",
        }
        or source.get("method") != H3_SCHEDULE_OVERLAY_METHOD
        or source.get("source_artifact_id") != base.artifact_id
        or any(not isinstance(item, str) or not item for item in source.values())
        or _SHA256.fullmatch(source.get("source_packed_input_sha256", "")) is None
        or _SHA256.fullmatch(source.get("transformer_tensor_sha256", "")) is None
        or Path(source.get("transformer_tensor_file", "")).name
        != source.get("transformer_tensor_file")
    ):
        raise H3ScheduleOverlayError("H3 schedule-overlay source contract is invalid")

    auxiliary_value = value.get("auxiliary")
    if not isinstance(auxiliary_value, dict) or set(auxiliary_value) != {
        "path",
        "size_bytes",
        "sha256",
        "timestep_rows",
    }:
        raise H3ScheduleOverlayError("H3 schedule-overlay auxiliary record is invalid")
    auxiliary_path, auxiliary_size, auxiliary_sha = _validated_file_record(
        resolved,
        auxiliary_value,
        name="auxiliary",
    )
    timestep_rows = auxiliary_value.get("timestep_rows")
    if (
        not isinstance(timestep_rows, int)
        or isinstance(timestep_rows, bool)
        or timestep_rows <= 0
    ):
        raise H3ScheduleOverlayError("H3 schedule-overlay timestep row count is invalid")
    auxiliary_header = inspect_safetensors_header(auxiliary_path)
    expected_auxiliary = {
        "final_adaln_table": ("BF16", (schedule.nfe, timestep_rows, 2, base.spec.hidden_size)),
        "time_embeddings": ("F32", (schedule.nfe, timestep_rows, base.spec.time_embed_dim)),
        "timestep_counts": ("I64", (schedule.nfe,)),
        "timesteps": ("F32", (schedule.nfe, timestep_rows)),
    }
    if set(auxiliary_header) != set(expected_auxiliary):
        raise H3ScheduleOverlayError("H3 schedule-overlay auxiliary tensors are incomplete")
    for name, (dtype, shape) in expected_auxiliary.items():
        row = auxiliary_header[name]
        if row.get("dtype") != dtype or tuple(row.get("shape", ())) != shape:
            raise H3ScheduleOverlayError(
                f"H3 schedule-overlay auxiliary tensor differs: {name}"
            )

    blocks_value = value.get("blocks")
    if not isinstance(blocks_value, list) or len(blocks_value) not in {
        0,
        base.spec.num_layers,
    }:
        raise H3ScheduleOverlayError("H3 schedule-overlay block list is incomplete")
    if not blocks_value and (
        value.get("status") != "source-schedule-exact" or schedule.nfe != base.nfe
    ):
        raise H3ScheduleOverlayError(
            "only an exact source schedule may reuse artifact AdaLN tables"
        )
    base_blocks = {row.index: row for row in base.blocks}
    blocks: list[H3ScheduleOverlayBlock] = []
    for row in blocks_value:
        if not isinstance(row, dict) or set(row) != {
            "index",
            "path",
            "size_bytes",
            "sha256",
            "adaln_rows",
        }:
            raise H3ScheduleOverlayError("H3 schedule-overlay block record is invalid")
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in base_blocks:
            raise H3ScheduleOverlayError("H3 schedule-overlay block index is invalid")
        adaln_rows = row.get("adaln_rows")
        if adaln_rows != base_blocks[index].adaln_rows:
            raise H3ScheduleOverlayError("H3 schedule-overlay AdaLN row count drifted")
        path, size_bytes, sha256 = _validated_file_record(
            resolved,
            row,
            name=f"block {index}",
        )
        header = inspect_safetensors_header(path)
        expected_shape = (schedule.nfe, adaln_rows, 6, base.spec.hidden_size)
        if (
            set(header) != {"adaln.table"}
            or header["adaln.table"].get("dtype") != "BF16"
            or tuple(header["adaln.table"].get("shape", ())) != expected_shape
        ):
            raise H3ScheduleOverlayError(f"H3 schedule-overlay block {index} shape differs")
        blocks.append(
            H3ScheduleOverlayBlock(
                index=index,
                path=row["path"],
                size_bytes=size_bytes,
                sha256=sha256,
                adaln_rows=adaln_rows,
            )
        )
    blocks.sort(key=lambda row: row.index)
    if blocks and tuple(row.index for row in blocks) != tuple(range(base.spec.num_layers)):
        raise H3ScheduleOverlayError("H3 schedule-overlay block indices are not contiguous")
    return H3ScheduleOverlay(
        directory=resolved,
        overlay_id=overlay_id,
        created_at=value["created_at"],
        status=value["status"],
        layout=value["layout"],
        base_artifact_id=base.artifact_id,
        target_id=base.target.target_id,
        weight_profile=base.weight_profile,
        adapter_execution=base.adapter_execution,
        schedule=schedule,
        source=dict(source),
        auxiliary=H3ScheduleOverlayAuxiliary(
            path=auxiliary_value["path"],
            size_bytes=auxiliary_size,
            sha256=auxiliary_sha,
            timestep_rows=timestep_rows,
        ),
        blocks=tuple(blocks),
    )
