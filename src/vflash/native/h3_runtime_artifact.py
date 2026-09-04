"""Read and validate target-specific BF16 Ref2VA runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vflash.native.h3_artifact_contract import (
    H3ArtifactTarget,
    H3Spec,
    resolve_h3_artifact_target,
)
from vflash.native.h3_distilled_lora import (
    LIGHTX_H3_REF_TURBO8_CONTRACT,
    H3DistilledLoraError,
    h3_distilled_lora_contract_for_profile,
)
from vflash.native.h3_tensor_file import (
    inspect_safetensors_header,
)

H3_RUNTIME_ARTIFACT_SCHEMA_VERSION = 4
H3_RUNTIME_ARTIFACT_LAYOUT = "row-major-reference-v1"


_ARTIFACT_ID = re.compile(r"h3-runtime-[a-z0-9][a-z0-9-]{0,79}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_DTYPE_BYTES = {"U8": 1, "I8": 1, "F16": 2, "BF16": 2}
_BLOCK_BASE_TENSOR_NAMES = frozenset(
    {
        "adaln.table",
        "attn.qkv.weight",
        "attn.qkv.scale",
        "attn.out.weight",
        "attn.out.scale",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
        "ffn.in.weight",
        "ffn.in.scale",
        "ffn.out.weight",
        "ffn.out.scale",
        "norm.attn.weight",
        "norm.ffn.weight",
    }
)
_BLOCK_RESIDUAL_TENSOR_NAMES = frozenset(
    {
        "adapter.attn.q.down",
        "adapter.attn.q.up",
        "adapter.attn.k.down",
        "adapter.attn.k.up",
        "adapter.attn.v.down",
        "adapter.attn.v.up",
        "adapter.attn.out.down",
        "adapter.attn.out.up",
        "adapter.ffn.in.down",
        "adapter.ffn.in.up",
        "adapter.ffn.out.down",
        "adapter.ffn.out.up",
    }
)


class H3RuntimeArtifactError(ValueError):
    """A compiled H3 artifact is incomplete, unsafe, or internally inconsistent."""


@dataclass(frozen=True)
class H3RuntimeArtifactBlock:
    index: int
    path: str
    size_bytes: int
    sha256: str
    adaln_rows: int
    tensors: tuple[str, ...]


@dataclass(frozen=True)
class H3RuntimeArtifact:
    directory: Path
    artifact_id: str
    created_at: str
    status: str
    layout: str
    target: H3ArtifactTarget
    spec: H3Spec
    nfe: int
    weight_profile: str
    source: Mapping[str, str]
    blocks: tuple[H3RuntimeArtifactBlock, ...]
    compile_environment: Mapping[str, Any] = field(default_factory=dict)
    adapter_execution: str = "none"

    @property
    def is_complete_block_stack(self) -> bool:
        return tuple(row.index for row in self.blocks) == tuple(range(self.spec.num_layers))


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
        raise H3RuntimeArtifactError(f"H3 RuntimeArtifact file is unavailable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise H3RuntimeArtifactError(
            f"H3 RuntimeArtifact file must be a non-empty regular file: {path.name}"
        )
    return value


def _specialized_spec(value: Any) -> H3Spec:
    if not isinstance(value, dict):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact spec is invalid")
    expected_fields = set(H3Spec.__dataclass_fields__)
    if set(value) != expected_fields or not isinstance(value.get("patch_size"), list):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact spec fields differ from schema")
    try:
        spec = H3Spec(**{**value, "patch_size": tuple(value["patch_size"])})
    except (TypeError, ValueError) as exc:
        raise H3RuntimeArtifactError("H3 RuntimeArtifact spec is invalid") from exc
    expected = H3Spec(
        num_layers=50,
        hidden_size=5376,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_dim=14336,
        time_embed_dim=2688,
        in_channels=24,
        audio_in_channels=32,
        patch_size=(1, 2, 2),
    )
    if spec != expected:
        raise H3RuntimeArtifactError(
            "H3 RuntimeArtifact architecture is not specialized MiniMax H3"
        )
    return spec


def _validate_source(
    value: Any,
    *,
    weight_profile: str,
    schema_version: int,
) -> dict[str, str]:
    base_fields = {
        "model_repository",
        "model_revision",
        "transformer_sha256",
        "oracle",
        "oracle_revision",
        "request_sha256",
        "replay_case_id",
        "replay_packed_input_sha256",
    }
    if schema_version >= 3:
        base_fields.add("replay_schema_version")
        replay_schema_version = (
            value.get("replay_schema_version") if isinstance(value, dict) else None
        )
        if replay_schema_version not in {"3", "4", "5"}:
            raise H3RuntimeArtifactError("H3 RuntimeArtifact replay schema identity is invalid")
        if replay_schema_version in {"4", "5"}:
            base_fields.update(
                {
                    "oracle_profile",
                    "oracle_config_sha256",
                    "oracle_hardware",
                    "oracle_runtime_sha256",
                }
            )
    adapter_fields = {"adapter_repository", "adapter_revision", "adapter_sha256"}
    has_adapter = weight_profile in {
        "lightx-turbo8-v1.0",
        "lightx-ref-turbo4-v0.1",
    }
    expected = base_fields | adapter_fields if has_adapter else base_fields
    if not isinstance(value, dict) or set(value) != expected:
        raise H3RuntimeArtifactError("H3 RuntimeArtifact source fields differ from schema")
    digests = {
        "transformer_sha256",
        "request_sha256",
        "adapter_sha256",
        "replay_packed_input_sha256",
        "oracle_config_sha256",
        "oracle_runtime_sha256",
    }
    for name, item in value.items():
        if (
            not isinstance(item, str)
            or not item
            or (name in digests and not _SHA256.fullmatch(item))
        ):
            raise H3RuntimeArtifactError(f"H3 RuntimeArtifact source field is invalid: {name}")
    if weight_profile in {"lightx-turbo8-v1.0", "lightx-ref-turbo4-v0.1"}:
        oracle_profile = value.get("oracle_profile", "")
        workflow, separator, _suffix = oracle_profile.partition("-adapter-")
        if not separator:
            raise H3RuntimeArtifactError("H3 distilled-LoRA artifact oracle identity drifted")
        try:
            expected_contract = h3_distilled_lora_contract_for_profile(
                weight_profile,
                workflow=workflow,
            )
        except H3DistilledLoraError as exc:
            raise H3RuntimeArtifactError(
                "H3 distilled-LoRA artifact oracle identity drifted"
            ) from exc
        actual_identity = (
            value["adapter_repository"],
            value["adapter_revision"],
            value["adapter_sha256"],
        )
        expected_identity = (
            expected_contract.repository,
            expected_contract.revision,
            expected_contract.sha256,
        )
        if actual_identity != expected_identity:
            raise H3RuntimeArtifactError("H3 distilled-LoRA artifact adapter identity drifted")
    return {name: value[name] for name in sorted(expected)}


def _validate_compile_environment(
    value: Any,
    *,
    target: H3ArtifactTarget,
) -> dict[str, str]:
    expected = {
        "device_type",
        "device_name",
        "compute_capability",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "adaln_math",
        "exact_attention_default",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or any(not isinstance(item, str) or not item for item in value.values())
        or value.get("device_type") != "cuda"
        or value.get("compute_capability") != target.compute_capability
        or value.get("adaln_math") != "per-evaluation-original-row-count-v1"
        or value.get("exact_attention_default") != "torch-flash"
    ):
        raise H3RuntimeArtifactError(
            "H3 RuntimeArtifact compile environment differs from its target math"
        )
    return {name: value[name] for name in sorted(expected)}


def _validate_block_shapes(
    header: Mapping[str, Mapping[str, Any]],
    *,
    target: H3ArtifactTarget,
    spec: H3Spec,
    nfe: int,
    adaln_rows: int,
    weight_profile: str,
    adapter_execution: str,
) -> None:
    tensor_names = _BLOCK_BASE_TENSOR_NAMES
    if adapter_execution == "runtime-residual":
        tensor_names = tensor_names | _BLOCK_RESIDUAL_TENSOR_NAMES
    if set(header) != tensor_names or adaln_rows <= 0:
        raise H3RuntimeArtifactError("H3 RuntimeArtifact block tensor names are incomplete")
    inner = spec.num_attention_heads * spec.attention_head_dim
    group = target.ffn_group_size
    ffn_in_width = (
        spec.hidden_size if target.ffn_weight_bits in {8, 16} else spec.hidden_size // 2
    )
    ffn_out_width = spec.ffn_dim if target.ffn_weight_bits in {8, 16} else spec.ffn_dim // 2
    ffn_in_scale = (2 * spec.ffn_dim,)
    ffn_out_scale = (spec.hidden_size,)
    if group is not None:
        ffn_in_scale = (2 * spec.ffn_dim, spec.hidden_size // group)
        ffn_out_scale = (spec.hidden_size, spec.ffn_dim // group)
    expected = {
        "adaln.table": ("BF16", (nfe, adaln_rows, 6, spec.hidden_size)),
        "attn.qkv.weight": (
            "BF16" if target.attention_weight_bits == 16 else "I8",
            (3 * inner, spec.hidden_size),
        ),
        "attn.qkv.scale": ("F16", (3 * inner,)),
        "attn.out.weight": (
            "BF16" if target.attention_weight_bits == 16 else "I8",
            (spec.hidden_size, inner),
        ),
        "attn.out.scale": ("F16", (spec.hidden_size,)),
        "attn.q_norm.weight": ("BF16", (spec.attention_head_dim,)),
        "attn.k_norm.weight": ("BF16", (spec.attention_head_dim,)),
        "ffn.in.weight": (
            "BF16"
            if target.ffn_weight_bits == 16
            else "I8"
            if target.ffn_weight_bits == 8
            else "U8",
            (2 * spec.ffn_dim, ffn_in_width),
        ),
        "ffn.in.scale": ("F16", ffn_in_scale),
        "ffn.out.weight": (
            "BF16"
            if target.ffn_weight_bits == 16
            else "I8"
            if target.ffn_weight_bits == 8
            else "U8",
            (spec.hidden_size, ffn_out_width),
        ),
        "ffn.out.scale": ("F16", ffn_out_scale),
        "norm.attn.weight": ("BF16", (spec.hidden_size,)),
        "norm.ffn.weight": ("BF16", (spec.hidden_size,)),
    }
    if adapter_execution == "runtime-residual":
        rank = LIGHTX_H3_REF_TURBO8_CONTRACT.rank
        expected.update(
            {
                "adapter.attn.q.down": ("BF16", (rank, spec.hidden_size)),
                "adapter.attn.q.up": ("BF16", (inner, rank)),
                "adapter.attn.k.down": ("BF16", (rank, spec.hidden_size)),
                "adapter.attn.k.up": ("BF16", (inner, rank)),
                "adapter.attn.v.down": ("BF16", (rank, spec.hidden_size)),
                "adapter.attn.v.up": ("BF16", (inner, rank)),
                "adapter.attn.out.down": ("BF16", (rank, inner)),
                "adapter.attn.out.up": ("BF16", (spec.hidden_size, rank)),
                "adapter.ffn.in.down": ("BF16", (rank, spec.hidden_size)),
                "adapter.ffn.in.up": ("BF16", (2 * spec.ffn_dim, rank)),
                "adapter.ffn.out.down": ("BF16", (rank, spec.ffn_dim)),
                "adapter.ffn.out.up": ("BF16", (spec.hidden_size, rank)),
            }
        )
    for name, (dtype, shape) in expected.items():
        row = header[name]
        start, stop = row["data_offsets"]
        expected_bytes = _DTYPE_BYTES[dtype]
        for dimension in shape:
            expected_bytes *= dimension
        if (
            row["dtype"] != dtype
            or tuple(row["shape"]) != shape
            or stop - start != expected_bytes
        ):
            raise H3RuntimeArtifactError(
                f"H3 RuntimeArtifact block tensor differs for {name}: "
                f"{row['dtype']} {tuple(row['shape'])} != {dtype} {shape}"
            )


def load_h3_runtime_artifact(
    directory: Path,
    *,
    verify_content_hashes: bool = True,
) -> H3RuntimeArtifact:
    """Load an artifact, optionally repeating its publication-time payload hashes."""

    if directory.is_symlink() or not directory.is_dir():
        raise H3RuntimeArtifactError("H3 RuntimeArtifact directory must be a real directory")
    manifest_path = directory / "artifact.json"
    _regular_file(manifest_path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H3RuntimeArtifactError("H3 RuntimeArtifact manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact manifest schema is invalid")
    common_fields = {
        "schema_version",
        "artifact_id",
        "created_at",
        "status",
        "layout",
        "target",
        "spec",
        "nfe",
        "source",
        "precision",
        "blocks",
    }
    schema_version = value.get("schema_version")
    expected_fields = common_fields | {
        "weight_profile",
        "compile_environment",
        "adapter_execution",
    }
    if schema_version != H3_RUNTIME_ARTIFACT_SCHEMA_VERSION or set(value) != expected_fields:
        raise H3RuntimeArtifactError("H3 RuntimeArtifact manifest schema is invalid")
    artifact_id = value.get("artifact_id")
    created_at = value.get("created_at")
    status = value.get("status")
    layout = value.get("layout")
    nfe = value.get("nfe")
    weight_profile = value.get("weight_profile")
    adapter_execution = value.get("adapter_execution")
    if (
        not isinstance(artifact_id, str)
        or not _ARTIFACT_ID.fullmatch(artifact_id)
        or not isinstance(created_at, str)
        or not created_at
        or status != "complete-block-stack-missing-auxiliary-runtime"
        or layout != H3_RUNTIME_ARTIFACT_LAYOUT
        or not isinstance(nfe, int)
        or isinstance(nfe, bool)
        or nfe not in {4, 8}
        or weight_profile not in {"lightx-turbo8-v1.0", "lightx-ref-turbo4-v0.1"}
        or adapter_execution != "runtime-residual"
    ):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact top-level contract is invalid")
    target_value = value.get("target")
    if not isinstance(target_value, dict) or not isinstance(target_value.get("target_id"), str):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact target is invalid")
    target = resolve_h3_artifact_target(target_value["target_id"])
    if target_value != asdict(target):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact target fields drifted")
    expected_precision = {
        "attention_weight_bits": target.attention_weight_bits,
        "attention_activation": target.attention_activation,
        "ffn_weight_bits": target.ffn_weight_bits,
        "ffn_group_size": target.ffn_group_size,
        "ffn_activation": target.ffn_activation,
        "sensitive": "bfloat16",
        "adaln_table": "bfloat16",
    }
    if value.get("precision") != expected_precision:
        raise H3RuntimeArtifactError("H3 RuntimeArtifact precision fields drifted")
    spec = _specialized_spec(value.get("spec"))
    source = _validate_source(
        value.get("source"),
        weight_profile=weight_profile,
        schema_version=schema_version,
    )
    if weight_profile in {"lightx-turbo8-v1.0", "lightx-ref-turbo4-v0.1"} and (
        source.get("oracle_profile") != "ref2va-adapter-bf16-torch-sdpa-sm89"
        or nfe
        != h3_distilled_lora_contract_for_profile(
            weight_profile,
            workflow=source["oracle_profile"].partition("-adapter-")[0],
        ).nfe
    ):
        raise H3RuntimeArtifactError(
            "H3 distilled-LoRA artifact lacks a pinned SM89 Diffusers oracle identity"
        )
    compile_environment = (
        _validate_compile_environment(value.get("compile_environment"), target=target)
        if schema_version >= 3
        else {}
    )
    rows = value.get("blocks")
    if not isinstance(rows, list) or not rows:
        raise H3RuntimeArtifactError("H3 RuntimeArtifact has no block files")
    blocks: list[H3RuntimeArtifactBlock] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(
            H3RuntimeArtifactBlock.__dataclass_fields__
        ):
            raise H3RuntimeArtifactError("H3 RuntimeArtifact block row is invalid")
        index = row.get("index")
        relative = row.get("path")
        size_bytes = row.get("size_bytes")
        digest = row.get("sha256")
        adaln_rows = row.get("adaln_rows")
        tensors = row.get("tensors")
        expected_path = (
            f"blocks/block-{index:03d}.safetensors" if isinstance(index, int) else ""
        )
        expected_tensor_names = _BLOCK_BASE_TENSOR_NAMES
        if adapter_execution == "runtime-residual":
            expected_tensor_names = expected_tensor_names | _BLOCK_RESIDUAL_TENSOR_NAMES
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < spec.num_layers
            or relative != expected_path
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or not isinstance(adaln_rows, int)
            or isinstance(adaln_rows, bool)
            or adaln_rows <= 0
            or not isinstance(tensors, list)
            or tuple(tensors) != tuple(sorted(expected_tensor_names))
        ):
            raise H3RuntimeArtifactError("H3 RuntimeArtifact block row is invalid")
        path = directory / relative
        file_stat = _regular_file(path)
        if file_stat.st_size != size_bytes or (
            verify_content_hashes and _sha256(path) != digest
        ):
            raise H3RuntimeArtifactError(f"H3 RuntimeArtifact block hash mismatch: {relative}")
        header = inspect_safetensors_header(path)
        _validate_block_shapes(
            header,
            target=target,
            spec=spec,
            nfe=nfe,
            adaln_rows=adaln_rows,
            weight_profile=weight_profile,
            adapter_execution=adapter_execution,
        )
        blocks.append(
            H3RuntimeArtifactBlock(
                index=index,
                path=relative,
                size_bytes=size_bytes,
                sha256=digest,
                adaln_rows=adaln_rows,
                tensors=tuple(tensors),
            )
        )
    blocks.sort(key=lambda row: row.index)
    if len({row.index for row in blocks}) != len(blocks):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact contains duplicate block indices")
    complete = tuple(row.index for row in blocks) == tuple(range(spec.num_layers))
    if complete != (status == "complete-block-stack-missing-auxiliary-runtime"):
        raise H3RuntimeArtifactError("H3 RuntimeArtifact completeness status is false")
    return H3RuntimeArtifact(
        directory=directory,
        artifact_id=artifact_id,
        created_at=created_at,
        status=status,
        layout=layout,
        target=target,
        spec=spec,
        nfe=nfe,
        weight_profile=weight_profile,
        source=source,
        blocks=tuple(blocks),
        compile_environment=compile_environment,
        adapter_execution=adapter_execution,
    )
