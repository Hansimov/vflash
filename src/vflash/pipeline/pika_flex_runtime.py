"""Owned adapter to Pika's pinned native BF16 complete pipeline.

This reference path retains the upstream staged lifetime and full numerical
contract. It is distinct from accelerated/quantized runtimes and does not claim
a GPU qualification from source or CPU validation alone. Use a dedicated process
with the upstream pinned dependencies; the hybrid pipeline has a different stack.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from vflash.contracts import ContractError
from vflash.pipeline.pika_flex import (
    PIKA_ADAPTER_BYTES,
    PIKA_MODEL_REVISION,
    PIKA_RECIPE,
    PIKA_SOURCE_REVISION,
    PikaFlexRequest,
    adapter_partition,
    validate_adapter_manifest,
)


class PikaFlexRuntime:
    def __init__(
        self,
        *,
        source_root: Path,
        base_path: Path,
        adapters_root: Path,
        trust_local_code: bool = False,
        cpu_offload: bool = False,
    ):
        from vflash.adapters.pika_flex_source import load_pika_source, validate_pika_environment

        if type(cpu_offload) is not bool:
            raise ContractError("cpu_offload must be explicit boolean")
        validate_pika_environment()
        self.base = base_path.resolve(strict=True)
        self.adapters = adapters_root.resolve(strict=True)
        self.pipeline = load_pika_source(source_root, trust_local_code=trust_local_code)
        self.cpu_offload = cpu_offload
        self.closed = False
        self._owner = threading.Lock()
        self.manifests = {}
        for partition in ("fl2va", "ref2va"):
            root = self.adapters / partition
            if not root.exists():
                continue
            manifest = json.loads((root / "manifest.json").read_text())
            validate_adapter_manifest(manifest, partition)
            if (root / "adapter.safetensors").stat().st_size != PIKA_ADAPTER_BYTES:
                raise ContractError("Pika adapter size differs from its release")
            # The upstream pipeline verifies the full publisher hash before use.
            self.manifests[partition] = manifest
        if not self.manifests:
            raise ContractError("At least one original Pika partition is required")

    def generate(self, request: PikaFlexRequest, output: Path, *, on_progress=lambda *_: None):
        if not isinstance(request, PikaFlexRequest):
            raise ContractError("Pika requires an explicit PikaFlexRequest")
        if self.closed:
            raise ContractError("Pika runtime is closed")
        partition = adapter_partition(request.mode)
        if partition not in self.manifests:
            raise ContractError("The requested Pika partition is not loaded")
        if (
            output.suffix.lower() != ".mp4"
            or output.exists()
            or output.with_suffix(".json").exists()
        ):
            raise ContractError("Pika output and metadata must be new MP4/JSON paths")
        if not self._owner.acquire(blocking=False):
            raise ContractError("This Pika runtime already owns a request")
        start = time.monotonic()
        try:
            args = SimpleNamespace(
                task=request.mode,
                prompt=request.prompt,
                first_image=request.first_frame,
                last_image=request.last_frame,
                ref=[
                    f"{kind}:{path.resolve(strict=True)}" for kind, path in request.references
                ],
                resolution=request.resolution,
                aspect=request.aspect,
                duration=request.duration_seconds,
                seed=request.seed,
                cpu_offload=self.cpu_offload,
                output=output,
                model_repo="Pika-Labs/Pika-H3-Flex",
                revision=PIKA_MODEL_REVISION,
                base_path=self.base,
                adapter_path=self.adapters / partition,
                text_encoder_path=None,
            )
            on_progress("generation", 0, 1)
            self.pipeline.generate(args)
            metadata = json.loads(output.with_suffix(".json").read_text())
            manifest = self.manifests[partition]
            if (
                metadata.get("sampling") != manifest["sampling"]
                or metadata.get("adapter_sha256") != manifest["sha256"]
                or metadata.get("adapter_revision") != PIKA_MODEL_REVISION
                or metadata.get("base") != manifest["base"]
                or metadata.get("numerics_policy") != manifest["numerics_policy"]
                or metadata.get("task") != request.mode
                or metadata.get("prompt") != request.prompt
                or metadata.get("seed") != request.seed
            ):
                raise ContractError("Pika execution receipt differs from the request")
            on_progress("generation", 1, 1)
            return {
                "path": output,
                "recipe": PIKA_RECIPE,
                "source_revision": PIKA_SOURCE_REVISION,
                "seconds": time.monotonic() - start,
                "metadata": metadata,
                "weight_format": "original-bf16",
                "separate_audio_pass": False,
            }
        except BaseException:
            # Upstream stages need process teardown after a failed GPU operation.
            self.closed = True
            raise
        finally:
            self._owner.release()

    def close(self):
        if not self._owner.acquire(blocking=False):
            raise ContractError("Cannot close an active Pika request")
        try:
            self.closed = True
        finally:
            self._owner.release()
