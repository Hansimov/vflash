"""A self-contained W8 H3 trunk, with explicit precision and source contracts.

The bundle contains its unmerged adapter, schedule tables and input/output
projections. Text/image encoders and video/audio VAEs remain separate stages.
Reading a bundle establishes its file contract, never its output quality.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from traceback import clear_frames
from typing import Any

from vflash.native.h3_artifact_contract import H3ArtifactTarget, H3Spec
from vflash.native.h3_native_denoiser import H3LowRankResidualWeights, H3NativeBlockWeights
from vflash.native.h3_native_scheduler import H3NativeSchedule
from vflash.native.h3_runtime_auxiliary import H3_RUNTIME_AUXILIARY_SPECS
from vflash.native.h3_schedule_overlay import H3ScheduleOverlay, H3ScheduleOverlayAuxiliary
from vflash.native.h3_tensor_file import (
    H3MappedSafetensor,
    inspect_safetensors_header,
    load_safetensor_tensors,
)
from vflash.native.h3_w8a8 import W8Weight, validate_weight

FORMAT = "vflash-h3-w8a8-native-v1"
SPEC = H3Spec(50, 5376, 56, 128, 14336, 2688, 24, 32, (1, 2, 2))
WEIGHT_PROFILE = "lightx-ref-turbo4-v0.1"
STEMS = {
    "qkv": "attn.qkv",
    "attention_out": "attn.out",
    "ffn_in": "ffn.in",
    "ffn_out": "ffn.out",
}
MATRIX_SHAPES = {
    "attn.qkv": (21504, 5376),
    "attn.out": (5376, 7168),
    "ffn.in": (28672, 5376),
    "ffn.out": (5376, 14336),
}
ADAPTER = {
    "execution": "six unmerged BF16 residual branches",
    "rank": 128,
    "alpha": 8,
    "internal_scale": 0.0625,
    "strength": 1.0,
}
SOURCE = {
    "model_repository": "MiniMaxAI/MiniMax-H3",
    "model_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
    "adapter_repository": "lightx2v/Minimax-h3-Turbo",
    "adapter_revision": "83b617309219e859c1c264520eba07492d22e958",
    "transformer_sha256": "c5c855614b46cb7954fdeace235d0daf4525dd5ba10599464f4ae9ed76422765",
    "adapter_sha256": "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
    "oracle": "diffusers",
    "oracle_revision": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
    "oracle_profile": "ref2va-adapter-bf16-torch-sdpa-sm89",
}
SCHEDULE_SPECS = {
    "final_adaln_table": ("BF16", (4, 3, 2, 5376)),
    "time_embeddings": ("F32", (4, 3, 2688)),
    "timestep_counts": ("I64", (4,)),
    "timesteps": ("F32", (4, 3)),
}
W8_TARGET = H3ArtifactTarget(
    "h3-sm89-w8a8-block-ring",
    "sm89",
    48 << 30,
    8 << 30,
    8,
    8,
    None,
    "dynamic-rowwise-int8",
    "dynamic-rowwise-int8",
    "event-driven-block-ring",
)


class NativeW8BundleError(ValueError):
    """The declared W8 bundle is incomplete, changed or unsupported."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def block_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    result = {"adaln.table": ("BF16", (4, 9, 6, 5376))}
    for name, shape in MATRIX_SHAPES.items():
        result[name + ".weight"] = ("I8", shape)
        result[name + ".scale"] = ("F32", (shape[0],))
    for name, width in (
        ("norm.attn.weight", 5376),
        ("norm.ffn.weight", 5376),
        ("attn.q_norm.weight", 128),
        ("attn.k_norm.weight", 128),
    ):
        result[name] = ("BF16", (width,))
    for stem, inputs, outputs in (
        ("attn.q", 5376, 7168),
        ("attn.k", 5376, 7168),
        ("attn.v", 5376, 7168),
        ("attn.out", 7168, 5376),
        ("ffn.in", 5376, 28672),
        ("ffn.out", 14336, 5376),
    ):
        result["adapter." + stem + ".down"] = ("BF16", (128, inputs))
        result["adapter." + stem + ".up"] = ("BF16", (outputs, 128))
    return result


def validate_header(path: Path, specs: dict) -> dict:
    header = inspect_safetensors_header(path)
    actual = {key: (row["dtype"], tuple(row["shape"])) for key, row in header.items()}
    if actual != specs:
        raise NativeW8BundleError(
            "W8 tensor set, dtype or dimensions differ from the model ABI"
        )
    return header


@dataclass(frozen=True)
class NativeW8BlockFile:
    index: int
    path: str


class NativeW8Bundle:
    """Validate all file headers once; read owned tensors or scoped block mappings."""

    target = W8_TARGET
    spec = SPEC
    weight_profile = WEIGHT_PROFILE
    adapter_execution = "runtime-residual"
    nfe = 4
    is_complete_block_stack = True

    def __init__(self, directory: Path, *, verify_payloads: bool = False):
        self.directory = Path(directory).resolve(strict=True)
        path = self.directory / "bundle.json"
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 2 << 20:
            raise NativeW8BundleError("W8 manifest must be a bounded regular file")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise NativeW8BundleError("W8 manifest must be an object")
        expected_spec = {**asdict(SPEC), "patch_size": list(SPEC.patch_size)}
        if (
            value.get("schema_version") != 1
            or value.get("format") != FORMAT
            or value.get("mode") != "ref2va"
            or value.get("spec") != expected_spec
            or value.get("weight_profile") != WEIGHT_PROFILE
            or value.get("adapter") != ADAPTER
            or not isinstance(value.get("source"), dict)
            or any(value["source"].get(key) != expected for key, expected in SOURCE.items())
        ):
            raise NativeW8BundleError(
                "W8 model, adapter or architecture differs from its contract"
            )
        quant = value.get("quantization", {})
        if not isinstance(quant, dict):
            raise NativeW8BundleError("W8 quantization contract must be an object")
        if (
            quant.get("weights") != "symmetric INT8 per output row"
            or quant.get("scales") != "FP32"
            or quant.get("activations") != "dynamic symmetric INT8 per input row"
            or quant.get("rotation") is not False
            or quant.get("main_matrices") != list(MATRIX_SHAPES)
            or quant.get("output_dtype") != "BF16"
            or quant.get("weight_formula")
            != "scale=max(max(abs(row)),1e-10)/127; q=clamp(round(row/scale),-127,127)"
        ):
            raise NativeW8BundleError("W8 quantization recipe is unsupported")
        schedule = value.get("schedule", {})
        if (
            not isinstance(schedule, dict)
            or schedule.get("schema_version") != 1
            or schedule.get("nfe") != 4
        ):
            raise NativeW8BundleError(
                "W8 bundle requires its compiled four-evaluation schedule"
            )
        self.schedule = H3NativeSchedule(
            tuple(schedule["video_sigmas"]),
            tuple(schedule["audio_sigmas"]),
            update_rule=schedule["update_rule"],
        )
        # This uses CPU float32 just as the original scheduler; no CUDA call occurs.
        if self.schedule != H3NativeSchedule.shifted_linear(4, video_shift=12, audio_shift=3):
            raise NativeW8BundleError("W8 sigma grid differs from its compiled AdaLN tables")
        expected = {f"blocks/block-{i:03d}.safetensors" for i in range(50)}
        expected |= {"auxiliary.safetensors", "schedule.safetensors"}
        inventory = value.get("files")
        if not isinstance(inventory, dict) or set(inventory) != expected:
            raise NativeW8BundleError(
                "W8 model requires exactly 50 blocks, auxiliary and schedule"
            )
        self.paths = {}
        for relative, row in inventory.items():
            if not isinstance(row, dict):
                raise NativeW8BundleError("W8 payload inventory entry must be an object")
            candidate = self.directory / relative
            payload = candidate.resolve(strict=True)
            if (
                any((self.directory / part).is_symlink() for part in (relative, "blocks"))
                or self.directory not in payload.parents
                or not payload.is_file()
                or type(row.get("size_bytes")) is not int
                or payload.stat().st_size != row["size_bytes"]
                or re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256"))) is None
            ):
                raise NativeW8BundleError(
                    "W8 payload escaped its directory or changed its size"
                )
            specs = (
                block_specs()
                if relative.startswith("blocks/")
                else H3_RUNTIME_AUXILIARY_SPECS
                if relative == "auxiliary.safetensors"
                else SCHEDULE_SPECS
            )
            header = validate_header(payload, specs)
            if relative.startswith("blocks/") and row.get("tensors") != {
                key: {"dtype": item["dtype"], "shape": list(item["shape"])}
                for key, item in header.items()
            }:
                raise NativeW8BundleError(
                    "W8 published block inventory differs from its header"
                )
            if verify_payloads and sha256_file(payload) != row["sha256"]:
                raise NativeW8BundleError("W8 payload digest changed")
            self.paths[relative] = payload
        if value.get("payload_bytes") != sum(row["size_bytes"] for row in inventory.values()):
            raise NativeW8BundleError("W8 declared total payload size differs")
        self.manifest = value
        self.source = dict(value["source"])
        self.manifest_sha256 = sha256_file(path)
        self.artifact_id = "h3-w8a8-" + self.manifest_sha256[:16]
        self.blocks = tuple(
            NativeW8BlockFile(i, f"blocks/block-{i:03d}.safetensors") for i in range(50)
        )

    def _block_weights(self, load: Callable[[str], Any]) -> H3NativeBlockWeights:
        import torch

        matrices = {}
        for name, stem in STEMS.items():
            values, scales = load(stem + ".weight"), load(stem + ".scale")
            weight = W8Weight(values, scales, values.shape[1], values.shape[0])
            validate_weight(weight)
            if not bool((scales.isfinite() & (scales > 0)).all()):
                raise NativeW8BundleError("W8 scales must remain positive and finite")
            matrices[name] = weight

        def residual(stem):
            down, up = load(stem + ".down"), load(stem + ".up")
            if down.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
                raise NativeW8BundleError("W8 adapter residuals must remain BF16")
            return H3LowRankResidualWeights(down, up, 0.0625)

        return H3NativeBlockWeights(
            adaln_table=load("adaln.table"),
            **matrices,
            attention_norm=load("norm.attn.weight"),
            ffn_norm=load("norm.ffn.weight"),
            query_norm=load("attn.q_norm.weight"),
            key_norm=load("attn.k_norm.weight"),
            qkv_residuals=tuple(residual(f"adapter.attn.{name}") for name in ("q", "k", "v")),
            attention_out_residual=residual("adapter.attn.out"),
            ffn_in_residual=residual("adapter.ffn.in"),
            ffn_out_residual=residual("adapter.ffn.out"),
        )

    def _block_path(self, index: int) -> Path:
        if type(index) is not int or not 0 <= index < self.spec.num_layers:
            raise NativeW8BundleError("W8 block index is outside the model")
        path = self.paths[f"blocks/block-{index:03d}.safetensors"]
        validate_header(path, block_specs())
        return path

    def block_weights(self, index: int) -> H3NativeBlockWeights:
        tensors = load_safetensor_tensors(self._block_path(index), block_specs())
        return self._block_weights(tensors.__getitem__)

    @contextmanager
    def mapped_block(self, index: int) -> Iterator[H3NativeBlockWeights]:
        """Borrow a block only until its copies finish; escaped views are rejected."""
        with H3MappedSafetensor(self._block_path(index)) as mapped:
            weights = None
            try:
                weights = self._block_weights(mapped.load)
                yield weights
            except BaseException as exc:
                clear_frames(exc.__traceback__)
                raise
            finally:
                weights = None

    def schedule_overlay(self) -> H3ScheduleOverlay:
        row = self.manifest["files"]["schedule.safetensors"]
        return H3ScheduleOverlay(
            directory=self.directory,
            overlay_id="h3-schedule-" + self.artifact_id,
            created_at="",
            status="explicit-w8-model",
            layout="h3-native-w8-schedule-v1",
            base_artifact_id=self.artifact_id,
            target_id=self.target.target_id,
            weight_profile=self.weight_profile,
            adapter_execution=self.adapter_execution,
            schedule=self.schedule,
            source=self.source,
            auxiliary=H3ScheduleOverlayAuxiliary(
                "schedule.safetensors", row["size_bytes"], row["sha256"], 3
            ),
            blocks=(),
        )
