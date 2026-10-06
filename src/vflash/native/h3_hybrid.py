"""Explicit FL backbone + Ref block 25..49 modulation for original LightX v0.1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError


@dataclass(frozen=True)
class HybridModel:
    """Opt-in fixed FL-v0.1/Ref-modulation model, never a per-request weight swap.

    The directory is the pinned official ``transformer_ref`` component. Its
    bounded headers and inventory are checked before CUDA; no weight hash pass
    or second backbone is needed. Keyframes retain their FL conditioning graph.
    """

    reference_directory: Path

    def validate(self, *, profile_id, capability, strategy, attention_backend):
        if (
            profile_id
            not in {"i2va-turbo4-v01-544-exact-sm89", "fl2va-turbo4-v01-544-exact-sm89"}
            or capability != "8.9"
            or strategy != "single"
            or attention_backend not in {"torch-flash", "veda-sm89"}
            or not isinstance(self.reference_directory, Path)
        ):
            raise ContractError(
                "hybrid requires original FL v0.1 on one SM89 with dense or Veda attention"
            )
        from vflash.adapters.checkpoints import IndexedCheckpoint

        validate_hybrid_reference(IndexedCheckpoint(self.reference_directory))
        return self.stamps()

    def stamps(self):
        from vflash.model_assets import upstream_inventory

        rows = []
        for relative in upstream_inventory():
            if relative.startswith("transformer_ref/"):
                path = (
                    self.reference_directory / relative.removeprefix("transformer_ref/")
                ).resolve(strict=True)
                stat = path.stat()
                rows.append(
                    (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                )
        return tuple(rows)


@dataclass(frozen=True)
class HybridModulationOverlay:
    base: Any
    artifact: Any
    tables: dict[int, Any]
    provenance: dict[str, Any]

    @property
    def overlay_id(self):
        return "hybrid-ref-adaln-25-49-" + self.base.overlay_id

    @property
    def blocks(self):
        return self.artifact.blocks

    @property
    def schedule(self):
        return self.base.schedule

    def load_auxiliary_tensors(self):
        # In particular, never substitute the Ref timestep MLP or final layer.
        return self.base.load_auxiliary_tensors()

    def load_block_table(self, index):
        if type(index) is not int or not 0 <= index < 50:
            raise ValueError("hybrid block index must be within 0..49")
        if index >= 25:
            return self.tables[index]
        if self.base.blocks:
            return self.base.load_block_table(index)
        from vflash.native.h3_tensor_file import load_safetensor_tensor

        block = next(row for row in self.artifact.blocks if row.index == index)
        return load_safetensor_tensor(self.artifact.directory / block.path, "adaln.table")


def validate_hybrid_reference(checkpoint):
    """Resolve local weight views and inspect metadata only, before GPU borrowing."""
    from vflash.model_assets import upstream_inventory
    from vflash.native.h3_tensor_file import inspect_safetensors_header

    for path, record in upstream_inventory().items():
        if path.startswith("transformer_ref/"):
            local = checkpoint.directory / path.removeprefix("transformer_ref/")
            if local.stat().st_size != record["size"]:
                raise ValueError("hybrid reference inventory size differs")
    # Reject incompatible source headers before starting any tensor loads.
    headers = {}
    for index in range(25, 50):
        for suffix, expected in (("weight", (96768, 2688)), ("bias", (96768,))):
            name = f"transformer_blocks.{index}.adaln_proj.linear.{suffix}"
            shard = (checkpoint.directory / checkpoint.weight_map[name]).resolve(strict=True)
            if shard not in headers:
                headers[shard] = inspect_safetensors_header(shard)
            row = headers[shard][name]
            if tuple(row["shape"]) != expected or row["dtype"] != "BF16":
                raise ValueError("hybrid modulation source header differs")
    return {"selected_tensors": 50, "inspected_shards": len(headers)}


def compile_hybrid_modulation(artifact, overlay, checkpoint, *, device):
    """Compile only 25 small tables under an already selected, owned GPU."""
    import torch

    from vflash.compiler.math import adaln_table, profile_timesteps
    from vflash.model_assets import canonical_sha256, transformer_identity

    if (
        artifact.weight_profile != "lightx-turbo4-v0.1-544"
        or artifact.adapter_execution != "runtime-residual"
        or artifact.nfe != 4
        or artifact.spec.num_layers != 50
        or overlay.base_artifact_id != artifact.artifact_id
        or overlay.schedule.nfe != 4
    ):
        raise ValueError("native hybrid requires the original v0.1 FL artifact")
    validate_hybrid_reference(checkpoint)
    auxiliary = overlay.load_auxiliary_tensors()
    timing = auxiliary["time_embeddings"].to(device)
    counts = tuple(int(n) for n in auxiliary["timestep_counts"].tolist())
    expected_counts = tuple(
        row.numel() for row in profile_timesteps("i2va-turbo4-v01-544-exact-sm89")
    )
    if counts != expected_counts:
        raise ValueError("hybrid requires the four-step conditioned FL clock")
    tables = {}
    with torch.inference_mode():
        for index in range(25, 50):
            prefix = f"transformer_blocks.{index}.adaln_proj.linear."
            weight = checkpoint.load(prefix + "weight").to(device)
            bias = checkpoint.load(prefix + "bias").to(device)
            table = adaln_table(timing, counts, weight, bias, modalities=3, modulations=6)
            tables[index] = table.cpu()
            del weight, bias, table
    source = transformer_identity("ref2va-turbo4-exact-sm89")
    provenance = {
        "recipe": "fl-base-ref-block-adaln-25-49-bf16",
        "base_artifact_id": artifact.artifact_id,
        "reference_checkpoint_revision": source["model_revision"],
        "reference_component": "transformer_ref",
        "replaced_blocks": list(range(25, 50)),
        "compiled_table_bytes": sum(t.numel() * t.element_size() for t in tables.values()),
        "backbone_copied": False,
        "final_layer_changed": False,
        "base_artifact_modified": False,
        "adapter": artifact.weight_profile,
    }
    provenance["variant_id"] = canonical_sha256(provenance)
    return HybridModulationOverlay(overlay, artifact, tables, provenance)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hybrid_reference_source(base_source, *, reference_revision, runtime_versions=None):
    if not base_source["oracle_profile"].startswith(("i2va-adapter-", "fl2va-adapter-")):
        raise ValueError("hybrid Ref requires an adapter-bound FL source")
    recipe = dict(
        base_transformer=base_source["transformer_sha256"],
        reference_revision=reference_revision,
        reference_component="transformer_ref",
        modulation_blocks=list(range(25, 50)),
        prefix="FL",
        final_layer="FL",
    )
    return {
        **{
            k: base_source[k]
            for k in ("model_repository", "model_revision", "oracle", "oracle_revision")
        },
        "transformer_sha256": _digest(recipe),
        "oracle_profile": "ref2va-hybrid-fl-v01-bf16-sm89",
        "oracle_config_sha256": _digest(
            dict(
                workflow="ref2va",
                policy="match",
                nfe=4,
                video_shift=12,
                audio_shift=3,
                prefix="FL-v01",
                text_precision="bf16",
            )
        ),
        "oracle_hardware": "sm89-single",
        "oracle_runtime_sha256": _digest(runtime_versions or {}),
    }


def validate_hybrid_reference_bundle(bundle, *, source, schedule):
    if (
        bundle.profile.task != "ref2va"
        or bundle.schema_version != 1
        or bundle.profile.nfe != 4
        or bundle.profile.video_flow_shift != 12
        or bundle.profile.audio_flow_shift != 3
        or bundle.profile.num_condition_video_rows <= 0
        or bundle.profile.num_condition_audio_rows != 0
        or not 1 <= len(bundle.request.get("references", ())) <= 3
        or bundle.schedule.to_mapping() != schedule.to_mapping()
    ):
        raise ValueError("hybrid Ref image/clock/prefix contract differs")
    # Runtime dependency versions remain evidence, not a model identity alias.
    keys = set(source) - {"oracle_runtime_sha256"}
    if any(bundle.source.get(key) != source[key] for key in keys):
        raise ValueError("hybrid Ref source does not bind this FL trunk and Ref modulation")
