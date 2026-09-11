"""Compile the fixed BF16 H3 runtime packs directly from official weights.

Large unchanged linear weights stay on the CPU. Only the timestep MLP and one
AdaLN projection at a time enter CUDA. No request, replay or model-framework
module graph is an input to this compiler.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vflash.adapters.checkpoints import IndexedCheckpoint
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.model_assets import canonical_sha256, model_profile, model_schedule, weights_source
from vflash.native.h3_artifact_contract import H3Spec, resolve_h3_artifact_target
from vflash.native.h3_runtime_artifact import (
    H3_RUNTIME_ARTIFACT_LAYOUT,
    load_h3_runtime_artifact,
)
from vflash.native.h3_runtime_auxiliary import (
    H3_RUNTIME_AUXILIARY_LAYOUT,
    H3_RUNTIME_AUXILIARY_SPECS,
    load_h3_runtime_auxiliary,
)
from vflash.native.h3_schedule_overlay import (
    H3_SCHEDULE_OVERLAY_LAYOUT,
    H3_WEIGHTS_SCHEDULE_METHOD,
    load_h3_schedule_overlay,
)
from vflash.native.h3_tensor_file import (
    H3SingleTensorStore,
    inspect_safetensors_header,
    save_safetensors_atomic,
)

from .assets import PreparedWeights, load_prepared_weights
from .math import adaln_table, profile_timesteps, time_embeddings

SPEC = H3Spec(50, 5376, 56, 128, 14336, 2688, 24, 32, (1, 2, 2))


def compile_target(profile_id: str):
    profile = model_profile(profile_id)
    return resolve_h3_artifact_target(
        {
            "sm89": "rtx4090-48g-sm89-bf16-resident",
            "sm86": "rtx3080-20g-sm86-bf16-block-ring",
        }[profile.architecture]
    )


_LINEARS = {
    "attn.q": ("attn.to_q", 7168, 5376),
    "attn.k": ("attn.to_k", 7168, 5376),
    "attn.v": ("attn.to_v", 7168, 5376),
    "attn.out": ("attn.to_out.0", 5376, 7168),
    "ffn.in": ("ff.net.0.proj", 28672, 5376),
    "ffn.out": ("ff.net.2", 5376, 14336),
}
_NORMS = {
    "norm.attn.weight": ("norm1.weight", 5376),
    "norm.ffn.weight": ("norm2.weight", 5376),
    "attn.q_norm.weight": ("attn.norm_q.weight", 128),
    "attn.k_norm.weight": ("attn.norm_k.weight", 128),
}


@dataclass(frozen=True)
class CompiledAssets:
    directory: Path
    artifact: Path
    schedule_overlay: Path
    auxiliary_tensor: Path


def _input_specs(*, include_adapter: bool = True) -> tuple[
    dict[str, tuple[str, tuple[int, ...]]], dict[str, tuple[str, tuple[int, ...]]]
]:
    base = dict(H3_RUNTIME_AUXILIARY_SPECS)
    base.update(
        {
            "time_embedder.linear_1.weight": ("F32", (5376, 256)),
            "time_embedder.linear_1.bias": ("F32", (5376,)),
            "time_embedder.linear_2.weight": ("F32", (2688, 5376)),
            "time_embedder.linear_2.bias": ("F32", (2688,)),
            "norm_out.linear.weight": ("BF16", (10752, 2688)),
            "norm_out.linear.bias": ("BF16", (10752,)),
        }
    )
    adapter = {}
    for index in range(SPEC.num_layers):
        prefix = f"transformer_blocks.{index}."
        base[prefix + "adaln_proj.linear.weight"] = ("BF16", (96768, 2688))
        base[prefix + "adaln_proj.linear.bias"] = ("BF16", (96768,))
        for raw, size in _NORMS.values():
            base[prefix + raw] = ("BF16", (size,))
        for raw, rows, columns in _LINEARS.values():
            base[prefix + raw + ".weight"] = ("BF16", (rows, columns))
            if include_adapter:
                adapter[prefix + raw + ".lora_A.default.weight"] = ("BF16", (128, columns))
                adapter[prefix + raw + ".lora_B.default.weight"] = ("BF16", (rows, 128))
    return base, adapter


def validate_weight_headers(prepared: PreparedWeights) -> dict[str, int]:
    """Check every required tensor's name, dtype and shape without touching CUDA."""
    checked = load_prepared_weights(prepared.receipt)
    if checked != prepared:
        raise ContractError("the weights receipt changed after preparation")
    checkpoint = IndexedCheckpoint(prepared.transformer_directory)
    headers = {}
    profile = model_profile(prepared.profile_id)
    required_base, required_adapter = _input_specs(include_adapter=profile.adapter is not None)
    for name, (dtype, shape) in required_base.items():
        shard = checkpoint.weight_map.get(name)
        if shard is None:
            raise ContractError(f"official checkpoint tensor is missing: {name}")
        if shard not in headers:
            headers[shard] = inspect_safetensors_header(
                (prepared.transformer_directory / shard).resolve(strict=True)
            )
        row = headers[shard].get(name, {})
        if row.get("dtype") != dtype or tuple(row.get("shape", ())) != shape:
            raise ContractError(f"official checkpoint tensor shape or dtype differs: {name}")
    if required_adapter:
        if prepared.adapter_path is None:
            raise ContractError("the prepared compiler input is missing its adapter")
        header = inspect_safetensors_header(prepared.adapter_path)
        for name, (dtype, shape) in required_adapter.items():
            row = header.get(name, {})
            if row.get("dtype") != dtype or tuple(row.get("shape", ())) != shape:
                raise ContractError(f"official adapter tensor shape or dtype differs: {name}")
    return {
        "base_tensors": len(required_base),
        "adapter_tensors": len(required_adapter),
        "shards": len(headers),
    }


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _publish_directory(staging: Path, destination: Path) -> None:
    """Linux atomic no-replace rename: a concurrent creator keeps its directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise ContractError("the weights compiler requires Linux renameat2")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def _compile_block(
    index: int,
    checkpoint: IndexedCheckpoint,
    adapter: H3SingleTensorStore | None,
    timing: dict[str, Any],
    device: Any,
    counts: tuple[int, ...],
) -> dict[str, Any]:
    import torch

    prefix = f"transformer_blocks.{index}."
    weight = checkpoint.load(prefix + "adaln_proj.linear.weight").to(device)
    bias = checkpoint.load(prefix + "adaln_proj.linear.bias").to(device)
    table = adaln_table(
        timing["time_embeddings"], counts, weight, bias, modalities=3, modulations=6
    ).cpu()
    del weight, bias
    bases = checkpoint.load_many(
        prefix + raw + ".weight" for raw, _rows, _columns in _LINEARS.values()
    )
    output = {
        "adaln.table": table,
        "attn.qkv.weight": torch.cat(
            [
                bases.pop(prefix + _LINEARS[name][0] + ".weight")
                for name in ("attn.q", "attn.k", "attn.v")
            ],
            dim=0,
        ),
    }
    for name in ("attn.out", "ffn.in", "ffn.out"):
        output[name + ".weight"] = bases.pop(prefix + _LINEARS[name][0] + ".weight")
    for name in ("attn.qkv", "attn.out", "ffn.in", "ffn.out"):
        output[name + ".scale"] = torch.ones(
            output[name + ".weight"].shape[0], dtype=torch.float16
        )
    for name, (raw, _size) in _NORMS.items():
        output[name] = checkpoint.load(prefix + raw)
    if adapter is not None:
        for name, (raw, _rows, _columns) in _LINEARS.items():
            for output_suffix, raw_suffix in (("down", "A"), ("up", "B")):
                output[f"adapter.{name}.{output_suffix}"] = adapter.load(
                    prefix + raw + f".lora_{raw_suffix}.default.weight"
                )
    return output


def _write_assets(
    prepared: PreparedWeights,
    staging: Path,
    device: Any,
    progress: Callable[[str, int, int], None] | None,
) -> None:
    import torch

    profile = model_profile(prepared.profile_id)
    source = weights_source(prepared.profile_id)
    target = compile_target(prepared.profile_id)
    family = (
        "ref4"
        if profile.definition.mode.value == "ref2va"
        else "base4"
        if profile.definition.mode.value == "t2va"
        else "base16-i2va"
        if profile.definition.mode.value == "i2va"
        else "base16-fl2va"
    )
    artifact_id = (
        f"h3-runtime-{family}-bf16-{profile.architecture}-" + canonical_sha256(source)[:12]
    )
    created_at = datetime.now(UTC).isoformat()
    artifact_directory, overlay_directory = staging / "artifact", staging / "schedule"
    artifact_directory.mkdir()
    overlay_directory.mkdir()
    checkpoint = IndexedCheckpoint(prepared.transformer_directory)
    adapter = (
        H3SingleTensorStore(prepared.adapter_path)
        if prepared.adapter_path is not None
        else None
    )
    timing = time_embeddings(checkpoint.load, device, profile_id=prepared.profile_id)
    counts = tuple(row.numel() for row in profile_timesteps(prepared.profile_id))
    timestep_rows = max(counts)
    records = []
    for index in range(SPEC.num_layers):
        block = _compile_block(index, checkpoint, adapter, timing, device, counts)
        path = artifact_directory / "blocks" / f"block-{index:03d}.safetensors"
        save_safetensors_atomic(
            path,
            block,
            metadata={"layout": H3_RUNTIME_ARTIFACT_LAYOUT, "block_index": str(index)},
        )
        records.append(
            {
                "index": index,
                **_file_record(path, artifact_directory),
                "adaln_rows": 3 * timestep_rows,
                "tensors": sorted(block),
            }
        )
        del block
        if progress is not None:
            progress("blocks", index + 1, SPEC.num_layers)
    final = adaln_table(
        timing["time_embeddings"],
        counts,
        checkpoint.load("norm_out.linear.weight").to(device),
        checkpoint.load("norm_out.linear.bias").to(device),
        modalities=1,
        modulations=2,
    )
    save_safetensors_atomic(
        overlay_directory / "schedule.safetensors",
        {
            **{name: value.cpu() for name, value in timing.items()},
            "final_adaln_table": final.cpu(),
        },
    )
    save_safetensors_atomic(
        staging / "auxiliary.safetensors",
        checkpoint.load_many(H3_RUNTIME_AUXILIARY_SPECS),
        metadata={"layout": H3_RUNTIME_AUXILIARY_LAYOUT},
    )
    artifact = {
        "schema_version": 5,
        "artifact_id": artifact_id,
        "created_at": created_at,
        "status": "complete-block-stack",
        "layout": H3_RUNTIME_ARTIFACT_LAYOUT,
        "target": asdict(target),
        "spec": asdict(SPEC),
        "nfe": profile.definition.nfe,
        "source": source,
        "weight_profile": profile.weight_profile,
        "adapter_execution": profile.adapter_execution,
        "precision": {
            "attention_weight_bits": 16,
            "attention_activation": "bfloat16",
            "ffn_weight_bits": 16,
            "ffn_group_size": None,
            "ffn_activation": "bfloat16",
            "sensitive": "bfloat16",
            "adaln_table": "bfloat16",
        },
        "compile_environment": {
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": profile.architecture,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": str(torch.backends.cudnn.version()),
            "adaln_math": "per-evaluation-original-row-count-v1",
            "exact_attention_default": "torch-flash",
        },
        "blocks": records,
    }
    (artifact_directory / "artifact.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    overlay = {
        "schema_version": 2,
        "overlay_id": f"h3-schedule-{family}-" + canonical_sha256(source)[:12],
        "created_at": created_at,
        "status": "source-schedule-exact",
        "layout": H3_SCHEDULE_OVERLAY_LAYOUT,
        "base_artifact": {
            "artifact_id": artifact_id,
            "target_id": target.target_id,
            "weight_profile": profile.weight_profile,
            "adapter_execution": profile.adapter_execution,
        },
        "schedule": model_schedule(prepared.profile_id).to_mapping(),
        "source": {
            "method": H3_WEIGHTS_SCHEDULE_METHOD,
            "source_artifact_id": artifact_id,
            "transformer_sha256": source["transformer_sha256"],
            "compile_recipe": source["compile_recipe"],
        },
        "auxiliary": {
            **_file_record(overlay_directory / "schedule.safetensors", overlay_directory),
            "timestep_rows": timestep_rows,
        },
        "blocks": [],
    }
    (overlay_directory / "overlay.json").write_text(
        json.dumps(overlay, indent=2) + "\n", encoding="utf-8"
    )
    # The writer already hashed every block. Validate headers, schema and small
    # schedule payloads without reading another forty gigabytes here.
    loaded = load_h3_runtime_artifact(artifact_directory, verify_content_hashes=False)
    load_h3_schedule_overlay(overlay_directory, artifact=loaded)
    load_h3_runtime_auxiliary(staging / "auxiliary.safetensors")
    prepared.check_unchanged()
    receipt = {
        "schema_version": 1,
        "kind": "h3-compiled-assets",
        "profile_id": prepared.profile_id,
        "source": source,
        "artifact": "artifact",
        "schedule_overlay": "schedule",
        "auxiliary_tensor": _file_record(staging / "auxiliary.safetensors", staging),
        "input_receipt_sha256": prepared.receipt_sha256,
    }
    (staging / "compiled.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


def compile_assets(
    prepared: PreparedWeights,
    destination: Path,
    *,
    device: NvidiaDevice,
    progress: Callable[[str, int, int], None] | None = None,
) -> CompiledAssets:
    """Compile in a fresh process on an exclusively assigned matching GPU.

    Publication is atomic and never replaces an existing destination. A
    callback may raise to cancel; unpublished temporary output is removed.
    GPU leasing belongs to the caller or deployment system.
    """
    import torch

    validate_weight_headers(prepared)
    profile = model_profile(prepared.profile_id)
    if (
        not isinstance(device, NvidiaDevice)
        or device.compute_capability != profile.hardware.compute_capability
    ):
        raise ContractError(
            "the weights compiler requires an explicit device matching its profile"
        )
    if torch.cuda.is_initialized():
        raise ContractError("start the weights compiler before initializing CUDA")
    if torch.__version__.split("+")[0] != "2.11.0":
        raise ContractError("the fixed compiler arithmetic requires PyTorch 2.11.0")
    if torch.version.cuda != "13.0":
        raise ContractError("the fixed compiler arithmetic requires a CUDA 13.0 PyTorch build")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ContractError("the compiled destination already exists")
    if not destination.parent.is_dir():
        raise ContractError("create the destination parent before compiling")
    if shutil.disk_usage(destination.parent).free < 48 * 1024**3:
        raise ContractError("compiling H3 requires at least 48 GiB of free output space")
    if getattr(ctypes.CDLL(None), "renameat2", None) is None:
        raise ContractError("the weights compiler requires Linux renameat2")
    os.environ["CUDA_VISIBLE_DEVICES"] = device.uuid
    expected_capability = tuple(
        int(part) for part in profile.hardware.compute_capability.split(".")
    )
    if torch.cuda.get_device_capability(0) != expected_capability:
        raise ContractError("the visible CUDA device differs from the compile target")
    runtime_device = torch.device("cuda:0")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
    )
    try:
        with torch.inference_mode():
            _write_assets(prepared, staging, runtime_device, progress)
        torch.cuda.synchronize(runtime_device)
        _publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return CompiledAssets(
        destination,
        destination / "artifact",
        destination / "schedule",
        destination / "auxiliary.safetensors",
    )
