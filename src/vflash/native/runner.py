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
    """Own one GPU group/profile and its weights; request tensors live for one call.

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
        if (
            plan.parallel_strategy not in {"single", "tensor", "sequence-head"}
            or (plan.parallel_strategy == "single") != (plan.peer_device is None)
            or (
                plan.peer_device is not None
                and (
                    plan.peer_device.uuid == plan.gpu_uuid
                    or plan.target.compute_capability != "8.6"
                )
            )
        ):
            raise ContractError("invalid physical GPU group for native parallel execution")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(plan.gpu_uuids)

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
            parallel_strategy=plan.parallel_strategy,
        )
        self.initialization_seconds = time.perf_counter() - started
        self.request_count = 0
        self.closed = False

    def generate(self, bundle: Path, output_latents: Path) -> dict[str, Any]:
        if self.closed:
            raise ContractError("the native engine session is closed")
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
            "parallel": {
                "strategy": self.plan.parallel_strategy,
                "device_count": len(self.plan.gpu_uuids),
                "peer_device": (
                    {
                        key: value
                        for key, value in asdict(self.plan.peer_device).items()
                        if key != "uuid"
                    }
                    if self.plan.peer_device is not None
                    else None
                ),
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

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            close = getattr(self.runtime, "close", None)
            if close is not None:
                close()


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
    try:
        return session.generate(bundle, output_latents)
    finally:
        session.close()
