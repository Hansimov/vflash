"""A fixed-profile engine session shared by the CLI and resident service worker."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError, ExecutionPlan, GenerationMode

WEIGHT_PROFILES = {
    "ref2va-turbo4-exact-sm86": "lightx-ref-turbo4-v0.1",
    "ref2va-turbo4-exact-sm89": "lightx-ref-turbo4-v0.1",
    "ref2va-turbo8-exact-sm89": "lightx-turbo8-v1.0",
}


class NativeEngineSession:
    """Own one GPU/profile and its weights; request tensors never outlive a call.

    Construct before CUDA initialization, then call serially. The service gives this
    session a dedicated spawned process, whose exit releases its CUDA context.
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        *,
        artifact: Path,
        schedule_overlay: Path,
        auxiliary_tensor: Path,
    ) -> None:
        started = time.perf_counter()
        profile = plan.profile
        if (
            profile.mode is not GenerationMode.REF2VA
            or profile.id not in WEIGHT_PROFILES
            or profile.nfe not in {4, 8}
            or not profile.attention.exact
            or plan.target.id not in profile.target_ids
            or (plan.target.compute_capability, plan.target.weight_residency)
            not in {("8.6", "block-ring"), ("8.9", "resident")}
        ):
            raise ContractError(
                "the public denoiser supports exact Ref2VA Turbo4 on SM86 "
                "and Turbo4/Turbo8 on SM89"
            )
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise ContractError("select the Vflash GPU before initializing CUDA")
        os.environ["CUDA_VISIBLE_DEVICES"] = plan.gpu_uuid

        from vflash.native.h3_native_conditioning_runtime import H3NativeConditioningRuntime

        self.plan = plan
        self.runtime = H3NativeConditioningRuntime(
            artifact_path=artifact,
            schedule_overlay_path=schedule_overlay,
            auxiliary_tensor_path=auxiliary_tensor,
            device="cuda:0",
            attention_backend=profile.attention.backend,
            expected_weight_profile=WEIGHT_PROFILES[profile.id],
            expected_model_repository=profile.model,
            expected_model_revision=profile.model_revision,
            expected_adapter_repository=profile.adapter,
            expected_adapter_revision=profile.adapter_revision,
            expected_nfe=profile.nfe,
            expected_scheduler=profile.scheduler,
            expected_video_flow_shift=profile.video_flow_shift,
            expected_audio_flow_shift=profile.audio_flow_shift,
        )
        self.initialization_seconds = time.perf_counter() - started
        self.request_count = 0

    def generate(self, bundle: Path, output_latents: Path) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.runtime.generate_latents(bundle, output_latents)
        self.request_count += 1
        return {
            "schema_version": 1,
            "status": "complete",
            "profile_id": self.plan.profile.id,
            "gpu": {
                "index": self.plan.gpu_index,
                "name": self.plan.gpu_name,
                "memory_gib": self.plan.gpu_memory_gib,
            },
            "runtime": self.runtime.metadata(),
            "session": {
                "request_index": self.request_count,
                "initialization_seconds": self.initialization_seconds,
                "initialization_charged_seconds": (
                    self.initialization_seconds if self.request_count == 1 else 0.0
                ),
                "request_wall_seconds": time.perf_counter() - started,
            },
            "generation": {**asdict(result), "output_path": str(result.output_path)},
        }


def denoise_conditioning_bundle(
    plan: ExecutionPlan,
    *,
    bundle: Path,
    artifact: Path,
    schedule_overlay: Path,
    auxiliary_tensor: Path,
    output_latents: Path,
) -> dict[str, Any]:
    """One-shot CLI path; services retain a NativeEngineSession instead."""
    session = NativeEngineSession(
        plan,
        artifact=artifact,
        schedule_overlay=schedule_overlay,
        auxiliary_tensor=auxiliary_tensor,
    )
    return session.generate(bundle, output_latents)
