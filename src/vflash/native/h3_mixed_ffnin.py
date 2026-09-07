"""Fixed mixed-precision FFN-in weights for the SM89 Ref4 block ring.

Only selected main FFN-in matrices and their activations use INT8. All other
weights, LoRA operands, attention and the two-slot event schedule stay BF16.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError

PROFILE_ID = "ref2va-turbo4-mixed-ffnin31-sm89"
BLOCKS = (2, *range(4, 27), 34, 35, 39, 40, 42, 45, 46)
_EXTENSION_SHA256 = "0aaafd8cb9485d2e6f76a7d2255464e61d14eaa350e5d2a62018256cc088e1de"
_SHA256 = re.compile(r"[a-f0-9]{64}")


def artifact_signature(artifact) -> str:
    """Bind to existing block digests; do not hash the base model again."""
    value = {
        "blocks": [(row.index, row.sha256) for row in artifact.blocks],
        "architecture": [
            artifact.spec.num_layers,
            artifact.spec.hidden_size,
            artifact.spec.ffn_dim,
        ],
        "target": artifact.target.compute_capability,
        "weight_profile": artifact.weight_profile,
        "adapter_execution": artifact.adapter_execution,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_artifact(artifact) -> None:
    if (
        not artifact.is_complete_block_stack
        or artifact.target.compute_capability != "sm89"
        or artifact.spec.num_layers != 50
        or artifact.weight_profile != "lightx-ref-turbo4-v0.1"
        or artifact.adapter_execution != "runtime-residual"
    ):
        raise ContractError("mixed FFN-in requires the complete SM89 Ref4 BF16 artifact")


class FFNInPayloads:
    """One immutable sidecar; only the currently loaded block is mapped.

    Run ``verify_content_hashes=True`` once when ingesting weights. Normal worker
    starts bind the manifest and check sizes, shapes and scales without hashing
    the same multi-gigabyte payload on each launch.
    """

    def __init__(self, directory: Path, artifact, *, verify_content_hashes: bool = False):
        _validate_artifact(artifact)
        self.directory = directory.resolve(strict=True)
        manifest_path = self.directory / "manifest.json"
        raw = manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        fields = {"schema_version", "profile_id", "base_signature", "alpha", "layers"}
        if (
            not isinstance(manifest, dict)
            or set(manifest) != fields
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["profile_id"] != PROFILE_ID
            or manifest["base_signature"] != artifact_signature(artifact)
            or manifest["alpha"] != 0.5
            or not isinstance(manifest["layers"], list)
            or len(manifest["layers"]) != len(BLOCKS)
        ):
            raise ContractError("mixed FFN-in manifest differs from its fixed profile/base")
        self.input_features = artifact.spec.hidden_size
        self.output_features = 2 * artifact.spec.ffn_dim
        self.entries = {}
        for index, row in zip(BLOCKS, manifest["layers"], strict=True):
            if (
                not isinstance(row, dict)
                or set(row) != {"block", "files"}
                or type(row["block"]) is not int
                or row["block"] != index
                or not isinstance(row["files"], dict)
                or set(row["files"]) != {"q", "scale", "d"}
            ):
                raise ContractError("mixed FFN-in layer selection changed")
            self.entries[index] = row["files"]
            for name in ("q", "scale", "d"):
                item = row["files"][name]
                if (
                    not isinstance(item, dict)
                    or set(item) != {"bytes", "sha256"}
                    or type(item["bytes"]) is not int
                    or item["bytes"] != self._bytes(name)
                    or not isinstance(item["sha256"], str)
                    or _SHA256.fullmatch(item["sha256"]) is None
                ):
                    raise ContractError("mixed FFN-in tensor identity is invalid")
                path = self._path(index, name)
                if verify_content_hashes:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != item["sha256"]:
                        raise ContractError("mixed FFN-in tensor digest changed")

    def _bytes(self, name):
        return {
            "q": self.output_features * self.input_features,
            "scale": self.output_features * 4,
            "d": self.input_features * 4,
        }[name]

    def _path(self, index, name):
        path = self.directory / f"block-{index:03d}" / f"{name}.bin"
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or path.resolve().parent != self.directory / f"block-{index:03d}"
            or info.st_size != self._bytes(name)
        ):
            raise ContractError(
                "mixed FFN-in payload must be a local regular file of fixed size"
            )
        return path

    def load(self, index: int, *, output_features: int, input_features: int):
        import torch

        if (output_features, input_features) != (self.output_features, self.input_features):
            raise ContractError("mixed FFN-in dimensions differ from the base artifact")
        if index not in self.entries:
            raise ContractError("the requested layer is not quantized by this profile")
        specifications = (
            ("q", torch.int8, (output_features, input_features)),
            ("scale", torch.float32, (output_features,)),
            ("d", torch.float32, (input_features,)),
        )
        tensors = []
        for name, dtype, shape in specifications:
            count = self._bytes(name) // torch.empty((), dtype=dtype).element_size()
            tensors.append(
                torch.from_file(str(self._path(index, name)), size=count, dtype=dtype).reshape(
                    shape
                )
            )
        weight = BalancedFFNIn(*tensors, block_index=index)
        validate_balanced(weight, payload=True)
        return weight


def _kitchen_linear():
    """Use the qualified native provider, never the package backend dispatcher."""
    import torch

    try:
        provider_version = version("comfy-kitchen")
    except PackageNotFoundError as exc:
        raise ContractError("install the w8 extra for the mixed FFN-in profile") from exc
    if (
        sys.version_info[:2] != (3, 11)
        or sys.platform != "linux"
        or torch.__version__ != "2.11.0+cu130"
        or provider_version != "0.2.31"
    ):
        raise ContractError(
            "mixed FFN-in requires Linux CPython 3.11, Torch 2.11/cu130 and the w8 extra"
        )
    import comfy_kitchen.backends.cuda as cuda

    if not cuda._EXT_AVAILABLE:
        raise ContractError("the mixed FFN-in native CUDA extension could not be loaded")
    if hashlib.sha256(Path(cuda._C.__file__).read_bytes()).hexdigest() != _EXTENSION_SHA256:
        raise ContractError("mixed FFN-in requires the qualified CPython 3.11 CUDA wheel")
    return cuda.int8_linear


def mixed_ffn_in_runtime(directory: Path):
    """Assemble one fixed session without patching modules or sharing model state."""
    import torch

    from vflash.native import h3_native_conditioning_runtime as conditioning
    from vflash.native import h3_native_denoiser as native

    provider = _kitchen_linear()
    payloads = None

    class MixedBlock(native.H3NativeBlockBF16Resident):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weights = replace(self.weights, ffn_in=FFNInSlot(self.weights.ffn_in))

        def _linear(self, states, weight):
            if not isinstance(weight, FFNInSlot):
                return super()._linear(states, weight)
            if states.dtype != torch.bfloat16 or states.device != self.device:
                raise ContractError("mixed FFN-in requires the original BF16 activations")
            if weight.block_index is None:
                return super()._linear(states, weight.bf16)
            output = provider(
                states.float() / weight.balance,
                weight.quantized,
                weight.scales,
                out_dtype=torch.bfloat16,
                convrot=False,
            )
            if (
                output.dtype != torch.bfloat16
                or output.device != states.device
                or output.shape != (*states.shape[:-1], weight.quantized.shape[0])
            ):
                raise ContractError("mixed FFN-in provider changed the BF16 output boundary")
            return output

    class MixedRing(native.H3NativeDenoiserBF16Ring):
        backend_id = "cuda-mixed-ffnin31-pinned-two-slot-v1"
        block_type = MixedBlock
        _copy_block = staticmethod(copy_block)
        _weight_bytes = staticmethod(block_bytes)

        @classmethod
        def load(cls, artifact, **kwargs):
            nonlocal payloads
            payloads = FFNInPayloads(directory, artifact)
            return super().load(artifact, **kwargs)

        @staticmethod
        def _load_block(artifact, index, **kwargs):
            if index in BLOCKS:
                kwargs["ffn_in_weight"] = payloads.load(
                    index,
                    output_features=2 * artifact.spec.ffn_dim,
                    input_features=artifact.spec.hidden_size,
                )
            return native.load_h3_native_block(artifact, index, **kwargs)

        @staticmethod
        def _pin_block(weights, *, pin):
            return native._pin_bf16_block(weights, pin=pin, ffn_in_pin=pin_ffn_in)

    class MixedRuntime(conditioning.H3NativeConditioningRuntime):
        backend_id = "vflash-native-mixed-ffnin31-ref4-v1"
        _block_ring_type = MixedRing

        def __init__(self, **kwargs):
            required = {
                "weight_residency": "block-ring",
                "parallel_strategy": "single",
                "expected_task": "ref2va",
                "expected_nfe": 4,
                "expected_weight_profile": "lightx-ref-turbo4-v0.1",
            }
            if any(kwargs.get(name, value) != value for name, value in required.items()):
                raise ContractError("mixed FFN-in supports only the single-SM89 Ref4 ring")
            super().__init__(**{**kwargs, **required})

        def metadata(self):
            return {
                **super().metadata(),
                "mixed_ffn_in": {
                    "profile_id": PROFILE_ID,
                    "manifest_sha256": payloads.manifest_sha256,
                    "blocks": list(BLOCKS),
                    "precision": "balanced-w8a8",
                    "remaining_weights": "bf16-runtime-residual",
                },
            }

    return MixedRuntime


@dataclass(frozen=True)
class BalancedFFNIn:
    values: Any
    scales: Any
    balance: Any
    block_index: int


def validate_balanced(weight: BalancedFFNIn, *, payload: bool = False) -> None:
    import torch

    q, scale, balance = weight.values, weight.scales, weight.balance
    if (
        q.dtype != torch.int8
        or q.ndim != 2
        or min(q.shape) <= 0
        or scale.dtype != torch.float32
        or tuple(scale.shape) != (q.shape[0],)
        or balance.dtype != torch.float32
        or tuple(balance.shape) != (q.shape[1],)
        or type(weight.block_index) is not int
        or weight.block_index not in BLOCKS
        or any(not x.is_contiguous() or x.requires_grad for x in (q, scale, balance))
        or any(x.device != q.device for x in (scale, balance))
    ):
        raise ValueError("balanced FFN-in requires its INT8 rows and FP32 scale/balance")
    if payload:
        if q.device.type != "cpu":
            raise ValueError("payload validation belongs to CPU loading")
        if not all(bool((x.isfinite() & (x > 0)).all()) for x in (scale, balance)):
            raise ValueError("balanced FFN-in scale/balance must be finite and positive")


def pin_ffn_in(weight, *, pin):
    if isinstance(weight, BalancedFFNIn):
        validate_balanced(weight)
        return replace(
            weight,
            values=pin(weight.values),
            scales=pin(weight.scales),
            balance=pin(weight.balance),
        )
    from vflash.native.h3_native_denoiser import _pin_bf16_weight

    return _pin_bf16_weight(weight, pin=pin)


class FFNInSlot:
    """Reuse the BF16 capacity required by retained layers for either precision.

    The INT8 view aliases the first half of this buffer. It is not an additional
    weight allocation; source precision determines which view is populated and
    consumed. Scale and balance are the only additional device allocations.
    """

    def __init__(self, bf16_weight):
        import torch

        values = bf16_weight.values
        if (
            values.dtype != torch.bfloat16
            or values.ndim != 2
            or min(values.shape) <= 0
            or not values.is_contiguous()
            or values.requires_grad
        ):
            raise ValueError("FFN-in staging requires its retained contiguous BF16 capacity")
        self.bf16 = bf16_weight
        self.quantized = (
            values.view(torch.int8).reshape(-1)[: values.numel()].view(values.shape)
        )
        self.scales = torch.empty(values.shape[0], dtype=torch.float32, device=values.device)
        self.balance = torch.empty(values.shape[1], dtype=torch.float32, device=values.device)
        self.block_index = None

    @property
    def values(self):
        return self.bf16.values if self.block_index is None else self.quantized

    @property
    def capacity_bytes(self):
        return self.bf16.values.nbytes + self.scales.nbytes + self.balance.nbytes

    def validate_source(self, source):
        if isinstance(source, BalancedFFNIn):
            validate_balanced(source)
        else:
            import torch

            from vflash.native.h3_native_denoiser import H3BF16Weight

            if (
                not isinstance(source, H3BF16Weight)
                or source.bits != 16
                or source.group_size is not None
                or source.values.dtype != torch.bfloat16
                or not source.values.is_contiguous()
                or source.values.requires_grad
            ):
                raise ValueError("retained FFN-in must remain original BF16")
        if tuple(source.values.shape) != tuple(self.bf16.values.shape):
            raise ValueError("FFN-in source and staging dimensions differ")

    def copy_from(self, source):
        self.validate_source(source)
        if isinstance(source, BalancedFFNIn):
            self.quantized.copy_(source.values, non_blocking=True)
            self.scales.copy_(source.scales, non_blocking=True)
            self.balance.copy_(source.balance, non_blocking=True)
            self.block_index = source.block_index
        else:
            self.bf16.values.copy_(source.values, non_blocking=True)
            self.block_index = None


def copy_block(destination, source):
    from vflash.native import h3_native_denoiser as native

    if not isinstance(destination.ffn_in, FFNInSlot):
        raise ValueError("owned FFN-in copy requires an explicit mixed slot")
    destination.ffn_in.validate_source(source.ffn_in)
    native._copy_bf16_block_(destination, source, copy_ffn_in=False)
    destination.ffn_in.copy_from(source.ffn_in)


def block_bytes(weights):
    from vflash.native.h3_native_denoiser import _block_tensor_bytes

    count = _block_tensor_bytes(weights)
    weight = weights.ffn_in
    if isinstance(weight, BalancedFFNIn):
        count += weight.scales.nbytes + weight.balance.nbytes
    elif isinstance(weight, FFNInSlot):
        count += weight.capacity_bytes - weight.values.nbytes
    return count
