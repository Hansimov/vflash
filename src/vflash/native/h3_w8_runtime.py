"""Explicit experimental W8 bundle consumer for a selected SM89 GPU.

Callers select their physical GPU before CUDA initialization and own any
device lease. This runtime consumes conditioning bundles and writes AV
latents; the original encoding and media stages remain separate.
"""

from __future__ import annotations

from pathlib import Path

from vflash.native.h3_native_conditioning_runtime import H3NativeConditioningRuntime
from vflash.native.h3_runtime_auxiliary import load_h3_runtime_auxiliary
from vflash.native.h3_w8_bundle import SOURCE, WEIGHT_PROFILE, NativeW8Bundle
from vflash.native.h3_w8a8 import NativeW8Ring


class NativeW8Runtime(H3NativeConditioningRuntime):
    """A self-contained W8 trunk with the fixed Ref2VA Turbo4 contract.

    This is an experimental, explicit API. It never selects W8 for an exact
    BF16 profile and does not declare the model's output production-qualified.
    """

    backend_id = "vflash-native-w8a8-conditioning-v1"

    def __init__(self, model: Path | NativeW8Bundle, *, device: str = "cuda:0"):
        self._bundle = model if isinstance(model, NativeW8Bundle) else NativeW8Bundle(model)
        super().__init__(
            artifact_path=self._bundle.directory,
            schedule_overlay_path=self._bundle.paths["schedule.safetensors"],
            auxiliary_tensor_path=self._bundle.paths["auxiliary.safetensors"],
            device=device,
            attention_backend="torch-flash",
            expected_weight_profile=WEIGHT_PROFILE,
            expected_model_repository=SOURCE["model_repository"],
            expected_model_revision=SOURCE["model_revision"],
            expected_adapter_repository=SOURCE["adapter_repository"],
            expected_adapter_revision=SOURCE["adapter_revision"],
            expected_nfe=4,
            expected_scheduler="h3-training-euler",
            expected_video_flow_shift=12,
            expected_audio_flow_shift=3,
            parallel_strategy="single",
            weight_residency="block-ring",
        )

    def _load_model_assets(self, artifact_path, schedule_overlay_path, auxiliary_tensor_path):
        if (
            artifact_path != self._bundle.directory
            or schedule_overlay_path != self._bundle.paths["schedule.safetensors"]
            or auxiliary_tensor_path != self._bundle.paths["auxiliary.safetensors"]
        ):
            raise ValueError("W8 components must come from the same validated model bundle")
        return (
            self._bundle,
            self._bundle.schedule_overlay(),
            load_h3_runtime_auxiliary(auxiliary_tensor_path),
        )

    def _load_denoiser(self):
        return NativeW8Ring.load_bundle(
            self._bundle,
            device=self.device,
            attention_backend="torch-flash",
            elementwise_backend="auto",
            adapter_fusion_backend="auto",
            rotary_backend="auto",
        )

    def metadata(self):
        return {
            **super().metadata(),
            "precision": "W8A8 main matrices; BF16 residual LoRA, norms and attention",
            "portable_bundle_sha256": self._bundle.manifest_sha256,
            "production_quality_qualified": False,
        }
