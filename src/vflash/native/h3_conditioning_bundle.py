"""Read content-bound Ref2VA conditioning bundles for native denoising."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vflash.native.errors import VflashNativeError
from vflash.native.h3_latent_layout import (
    h3_video_latent_frame_count,
)
from vflash.native.h3_native_scheduler import H3NativeSchedule
from vflash.native.h3_tensor_file import inspect_safetensors_header

H3_CONDITIONING_BUNDLE_SCHEMA_VERSION = 1

_BUNDLE_ID = re.compile(r"h3-conditioning-[a-z0-9][a-z0-9-]{0,95}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_FILES = {
    "conditioning": "conditioning.safetensors",
    "scheduler": "scheduler.json",
}
_TENSORS = frozenset(
    {
        "initial_video_latents",
        "initial_audio_latents",
        "encoder_hidden_states",
        "token_tags",
        "position_ids",
        "rotary_cos",
        "rotary_sin",
        "video_indices",
        "audio_indices",
        "text_indices",
        "first_timesteps",
        "first_timestep_indices",
        "first_time_embeddings",
        "first_packed_input",
    }
)


class H3ConditioningBundleError(VflashNativeError):
    """A conditioning bundle is incomplete, unsafe, or no longer reproducible."""


@dataclass(frozen=True)
class H3ConditioningProfile:
    task: str
    width: int
    height: int
    frames: int
    nfe: int
    video_flow_shift: float
    audio_flow_shift: float
    reference_token_budget: int
    num_condition_video_rows: int
    num_condition_audio_rows: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> H3ConditioningProfile:
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected:
            raise H3ConditioningBundleError(
                "H3 conditioning profile fields do not match schema v1"
            )
        task = value.get("task")
        if task != "ref2va":
            raise H3ConditioningBundleError("H3 conditioning task must be Ref2VA")
        integers = ("width", "height", "frames", "nfe", "reference_token_budget")
        if any(
            not isinstance(value.get(name), int) or isinstance(value.get(name), bool)
            for name in integers
        ):
            raise H3ConditioningBundleError(
                "H3 conditioning profile integer fields are invalid"
            )
        width = int(value["width"])
        height = int(value["height"])
        frames = int(value["frames"])
        nfe = int(value["nfe"])
        reference_token_budget = int(value["reference_token_budget"])
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise H3ConditioningBundleError(
                "H3 conditioning canvas must be positive and divisible by 32"
            )
        if frames < 5 or (frames - 5) % 17:
            raise H3ConditioningBundleError(
                "H3 conditioning frames must follow the 17k+5 contract"
            )
        if nfe <= 0 or reference_token_budget < 0:
            raise H3ConditioningBundleError("H3 conditioning NFE/token budget is invalid")
        if reference_token_budget == 0:
            raise H3ConditioningBundleError(
                "H3 Ref2VA conditioning requires condition-video rows"
            )
        shifts: list[float] = []
        for name in ("video_flow_shift", "audio_flow_shift"):
            raw = value.get(name)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise H3ConditioningBundleError("H3 conditioning flow shift is invalid")
            shift = float(raw)
            if not math.isfinite(shift) or shift <= 0:
                raise H3ConditioningBundleError("H3 conditioning flow shift is invalid")
            shifts.append(shift)
        counts = (value.get("num_condition_video_rows"), value.get("num_condition_audio_rows"))
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts
        ):
            raise H3ConditioningBundleError("H3 conditioning prefix counts are invalid")
        if counts[0] != reference_token_budget:
            raise H3ConditioningBundleError(
                "H3 conditioning video prefix differs from the reference budget"
            )
        return cls(
            task=str(task),
            width=width,
            height=height,
            frames=frames,
            nfe=nfe,
            video_flow_shift=shifts[0],
            audio_flow_shift=shifts[1],
            reference_token_budget=reference_token_budget,
            num_condition_video_rows=counts[0],
            num_condition_audio_rows=counts[1],
        )


@dataclass(frozen=True)
class H3ConditioningFile:
    role: str
    path: str
    size_bytes: int
    sha256: str
    tensors: tuple[str, ...] = ()


@dataclass(frozen=True)
class H3ConditioningBundle:
    directory: Path
    bundle_id: str
    created_at: str
    profile: H3ConditioningProfile
    request: Mapping[str, Any]
    source: Mapping[str, str]
    files: tuple[H3ConditioningFile, ...]

    @property
    def conditioning_sha256(self) -> str:
        return next(row.sha256 for row in self.files if row.role == "conditioning")

    @property
    def schedule(self) -> H3NativeSchedule:
        return H3NativeSchedule.from_json(
            self.directory / _FILES["scheduler"],
            expected_nfe=self.profile.nfe,
        )


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
        raise H3ConditioningBundleError(
            f"H3 conditioning file is unavailable: {path.name}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise H3ConditioningBundleError(
            f"H3 conditioning file must be a non-empty regular file: {path.name}"
        )
    return value


def _validate_source(value: Any) -> dict[str, str]:
    required = {
        "model_repository",
        "model_revision",
        "transformer_sha256",
        "oracle",
        "oracle_revision",
        "oracle_profile",
        "oracle_config_sha256",
        "oracle_hardware",
        "oracle_runtime_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise H3ConditioningBundleError("H3 conditioning source fields do not match schema v1")
    digests = {
        "transformer_sha256",
        "oracle_config_sha256",
        "oracle_runtime_sha256",
    }
    for name in required:
        item = value.get(name)
        pattern = _SHA256 if name in digests else _IDENTIFIER
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise H3ConditioningBundleError(f"H3 conditioning source field is invalid: {name}")
    return {name: str(value[name]) for name in sorted(required)}


def _validate_request(value: Any) -> dict[str, Any]:
    required = {
        "source_case_id",
        "prompt",
        "prompt_sha256",
        "seed",
        "reference_image_policy",
        "delivery_profiles",
        "references",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise H3ConditioningBundleError("H3 conditioning request fields do not match schema v1")
    prompt = value.get("prompt")
    prompt_sha = value.get("prompt_sha256")
    seed = value.get("seed")
    if (
        not isinstance(value.get("source_case_id"), str)
        or _IDENTIFIER.fullmatch(value["source_case_id"]) is None
        or not isinstance(prompt, str)
        or not prompt.strip()
        or not isinstance(prompt_sha, str)
        or _SHA256.fullmatch(prompt_sha) is None
        or hashlib.sha256(prompt.encode()).hexdigest() != prompt_sha
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or value.get("reference_image_policy") not in {"diffusers", "match"}
    ):
        raise H3ConditioningBundleError("H3 conditioning request identity is invalid")
    deliveries = value.get("delivery_profiles")
    if not isinstance(deliveries, list) or not deliveries:
        raise H3ConditioningBundleError(
            "H3 conditioning request must declare delivery profiles"
        )
    validated_deliveries = []
    seen_deliveries = set()
    for row in deliveries:
        if not isinstance(row, dict) or set(row) != {
            "temporal_profile",
            "frames",
            "fps",
        }:
            raise H3ConditioningBundleError("H3 conditioning delivery fields are invalid")
        temporal_profile = row.get("temporal_profile")
        frames = row.get("frames")
        fps = row.get("fps")
        if (
            not isinstance(temporal_profile, str)
            or _IDENTIFIER.fullmatch(temporal_profile) is None
            or temporal_profile in seen_deliveries
            or not isinstance(frames, int)
            or isinstance(frames, bool)
            or frames <= 0
            or not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or not math.isfinite(float(fps))
            or float(fps) <= 0
        ):
            raise H3ConditioningBundleError("H3 conditioning delivery identity is invalid")
        seen_deliveries.add(temporal_profile)
        validated_deliveries.append(
            {
                "temporal_profile": temporal_profile,
                "frames": frames,
                "fps": float(fps),
            }
        )
    references = value.get("references")
    if not isinstance(references, list) or not 1 <= len(references) <= 3:
        raise H3ConditioningBundleError(
            "H3 Ref2VA conditioning requires one to three references"
        )
    validated_references = []
    seen = set()
    for index, row in enumerate(references, start=1):
        if not isinstance(row, dict) or set(row) != {
            "picture_index",
            "role",
            "size_bytes",
            "sha256",
        }:
            raise H3ConditioningBundleError("H3 conditioning reference fields are invalid")
        digest = row.get("sha256")
        if (
            row.get("picture_index") != index
            or row.get("role") != "reference"
            or not isinstance(row.get("size_bytes"), int)
            or isinstance(row.get("size_bytes"), bool)
            or row["size_bytes"] <= 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or digest in seen
        ):
            raise H3ConditioningBundleError("H3 conditioning reference identity is invalid")
        seen.add(digest)
        validated_references.append(dict(row))
    return {
        **{
            name: value[name]
            for name in required
            if name not in {"references", "delivery_profiles"}
        },
        "delivery_profiles": validated_deliveries,
        "references": validated_references,
    }


def _validate_tensor_contract(
    path: Path,
    *,
    profile: H3ConditioningProfile,
) -> None:
    header = inspect_safetensors_header(path)
    if set(header) != _TENSORS:
        raise H3ConditioningBundleError("H3 conditioning tensor names differ from schema v1")

    def shape(name: str) -> tuple[int, ...]:
        return tuple(header[name]["shape"])

    video_shape = shape("initial_video_latents")
    audio_shape = shape("initial_audio_latents")
    text_shape = shape("encoder_hidden_states")
    index_shapes = {
        name: shape(name) for name in ("video_indices", "audio_indices", "text_indices")
    }
    if (
        header["initial_video_latents"]["dtype"] != "F32"
        or len(video_shape) != 3
        or video_shape[0] != 1
        or min(video_shape[1:]) <= 0
        or header["initial_audio_latents"]["dtype"] != "F32"
        or len(audio_shape) != 3
        or audio_shape[0] != 1
        or min(audio_shape[1:]) <= 0
        or len(text_shape) != 3
        or text_shape[0] != 1
        or min(text_shape[1:]) <= 0
        or any(len(value) != 1 for value in index_shapes.values())
        or index_shapes["video_indices"] != (video_shape[1],)
        or index_shapes["audio_indices"] != (audio_shape[1],)
        or index_shapes["text_indices"] != (text_shape[1],)
    ):
        raise H3ConditioningBundleError("H3 conditioning modality shapes are invalid")
    sequence = video_shape[1] + audio_shape[1] + text_shape[1]
    packed_shape = shape("first_packed_input")
    if (
        shape("token_tags") != (sequence,)
        or shape("position_ids") != (sequence, 3)
        or shape("rotary_cos") != shape("rotary_sin")
        or len(shape("rotary_cos")) != 2
        or shape("rotary_cos")[0] != sequence
        or shape("first_timestep_indices") != (sequence,)
        or len(shape("first_timesteps")) != 1
        or len(shape("first_time_embeddings")) != 2
        or shape("first_time_embeddings")[0] != shape("first_timesteps")[0]
        or len(packed_shape) != 3
        or packed_shape[:2] != (1, sequence)
    ):
        raise H3ConditioningBundleError("H3 conditioning packed shapes are invalid")
    target_rows = h3_target_video_tokens(
        width=profile.width,
        height=profile.height,
        frames=profile.frames,
    )
    if video_shape[1] - target_rows != profile.reference_token_budget:
        raise H3ConditioningBundleError("H3 conditioning video rows differ from the profile")


def _files(directory: Path, profile: H3ConditioningProfile) -> tuple[H3ConditioningFile, ...]:
    rows = []
    for role, filename in _FILES.items():
        path = directory / filename
        file_stat = _regular_file(path)
        tensors: tuple[str, ...] = ()
        if role == "conditioning":
            _validate_tensor_contract(path, profile=profile)
            tensors = tuple(sorted(inspect_safetensors_header(path)))
        rows.append(
            H3ConditioningFile(
                role=role,
                path=filename,
                size_bytes=file_stat.st_size,
                sha256=_sha256(path),
                tensors=tensors,
            )
        )
    return tuple(rows)


def load_h3_conditioning_bundle(directory: Path) -> H3ConditioningBundle:
    """Load a bundle and fail closed on paths, hashes, or tensor drift."""

    try:
        resolved = directory.resolve(strict=True)
    except OSError as exc:
        raise H3ConditioningBundleError("H3 conditioning directory is unavailable") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise H3ConditioningBundleError("H3 conditioning directory must be a real directory")
    manifest_path = resolved / "bundle.json"
    _regular_file(manifest_path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H3ConditioningBundleError("H3 conditioning manifest is invalid JSON") from exc
    expected = {
        "schema_version",
        "bundle_id",
        "created_at",
        "profile",
        "request",
        "source",
        "files",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema_version") != H3_CONDITIONING_BUNDLE_SCHEMA_VERSION
        or not isinstance(value.get("bundle_id"), str)
        or _BUNDLE_ID.fullmatch(value["bundle_id"]) is None
        or not isinstance(value.get("created_at"), str)
        or not value["created_at"]
        or not isinstance(value.get("profile"), dict)
    ):
        raise H3ConditioningBundleError("H3 conditioning manifest fields are invalid")
    profile = H3ConditioningProfile.from_mapping(value["profile"])
    request = _validate_request(value.get("request"))
    source = _validate_source(value.get("source"))
    manifest_files = value.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != len(_FILES):
        raise H3ConditioningBundleError("H3 conditioning file manifest is incomplete")
    expected_files = _files(resolved, profile)
    expected_by_role = {row.role: row for row in expected_files}
    seen = set()
    for row in manifest_files:
        if not isinstance(row, dict) or set(row) != {
            "role",
            "path",
            "size_bytes",
            "sha256",
            "tensors",
        }:
            raise H3ConditioningBundleError("H3 conditioning file record is invalid")
        role = row.get("role")
        if role not in expected_by_role or role in seen:
            raise H3ConditioningBundleError("H3 conditioning file role is invalid")
        expected_row = expected_by_role[role]
        if row != {
            **asdict(expected_row),
            "tensors": list(expected_row.tensors),
        }:
            raise H3ConditioningBundleError(
                f"H3 conditioning file integrity check failed: {role}"
            )
        seen.add(role)
    if seen != set(_FILES):
        raise H3ConditioningBundleError("H3 conditioning file roles are incomplete")
    H3NativeSchedule.from_json(resolved / _FILES["scheduler"], expected_nfe=profile.nfe)
    return H3ConditioningBundle(
        directory=resolved,
        bundle_id=value["bundle_id"],
        created_at=value["created_at"],
        profile=profile,
        request=request,
        source=source,
        files=expected_files,
    )


def h3_target_video_tokens(*, width: int, height: int, frames: int) -> int:
    """Return target-video rows after H3 VAE compression and 1x2x2 patching."""
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("H3 target canvas must be positive and divisible by 32")
    if frames < 5 or (frames - 5) % 17:
        raise ValueError("H3 target frames must follow the 17k+5 contract")
    latent_frames = h3_video_latent_frame_count(frames)
    return latent_frames * (height // 32) * (width // 32)
