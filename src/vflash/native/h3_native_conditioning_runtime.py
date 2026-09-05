"""Native H3 denoising from a live request-conditioning bundle.

The official conditioner owns prompt/reference encoding and stops before
Transformer block zero.  This module owns the remaining request path without
calling LightX2V or rebuilding an upstream Diffusers module graph: input
projection, the resident or streamed 50-block native trunk, the selected schedule,
the output heads, and VAE-ready latent export.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vflash.native.h3_conditioning_bundle import load_h3_conditioning_bundle
from vflash.native.h3_latent_layout import (
    h3_video_latent_frame_count,
    unpack_h3_audio_rows,
    unpatchify_h3_video_rows,
)
from vflash.native.h3_native_denoiser import (
    H3NativeDenoiserBF16Resident,
    H3NativeDenoiserBF16Ring,
)
from vflash.native.h3_native_engine import H3NativeEngine
from vflash.native.h3_native_final_layer import (
    H3NativeFinalLayer,
    load_h3_native_final_layer_weights,
)
from vflash.native.h3_native_input_stage import (
    H3NativeInputPacker,
    load_h3_native_input_projection_weights,
)
from vflash.native.h3_native_scheduler import H3NativeLatentState, H3NativeSchedule
from vflash.native.h3_runtime_artifact import load_h3_runtime_artifact
from vflash.native.h3_runtime_auxiliary import load_h3_runtime_auxiliary
from vflash.native.h3_schedule_overlay import load_h3_schedule_overlay
from vflash.native.h3_tensor_file import (
    load_safetensor_tensor,
    save_safetensors_atomic,
)


class H3NativeConditioningRuntimeError(RuntimeError):
    """A live conditioning bundle cannot execute on the selected native runtime."""


def _attention_policy() -> dict[str, Any]:
    return {
        "attention_backend": "torch-flash",
        "strict_attention_backend": "torch-flash",
        "strict_prefix_evaluations": 0,
        "effective_backends": ["torch-flash"],
    }


def validate_declared_schedule(
    schedule: H3NativeSchedule,
    *,
    expected_nfe: int | None,
    expected_scheduler: str | None,
    expected_video_flow_shift: float | None,
    expected_audio_flow_shift: float | None,
) -> None:
    """Bind the declared profile to the schedule that will actually execute."""

    if expected_nfe is not None and schedule.nfe != expected_nfe:
        raise H3NativeConditioningRuntimeError("the schedule NFE differs from its profile")
    if expected_scheduler is not None and (
        expected_scheduler != "h3-training-euler" or schedule.update_rule != "training_euler"
    ):
        raise H3NativeConditioningRuntimeError("the scheduler differs from its profile")
    if expected_video_flow_shift is None and expected_audio_flow_shift is None:
        return
    if expected_video_flow_shift is None or expected_audio_flow_shift is None:
        raise H3NativeConditioningRuntimeError("both declared flow shifts are required")
    expected = H3NativeSchedule.shifted_linear(
        schedule.nfe,
        video_shift=expected_video_flow_shift,
        audio_shift=expected_audio_flow_shift,
    )
    if (
        schedule.video_sigmas != expected.video_sigmas
        or schedule.audio_sigmas != expected.audio_sigmas
    ):
        raise H3NativeConditioningRuntimeError(
            "the sigma grids differ from the declared flow shifts"
        )


@dataclass(frozen=True)
class H3NativeConditioningRuntimeResult:
    bundle_id: str
    output_path: Path
    conditioning_profile: dict[str, Any]
    execution_schedule: dict[str, Any]
    nfe: int
    elapsed_seconds: float
    peak_allocated_bytes: int
    attention_policy: dict[str, Any]
    stage_durations: dict[str, float]
    peak_allocated_bytes_by_device: tuple[int, ...]


class H3NativeConditioningRuntime:
    """Own one hardware-specialized H3 trunk and execute live Ref2VA bundles."""

    backend_id = "vflash-native-live-conditioning-v1"

    def __init__(
        self,
        *,
        artifact_path: Path,
        schedule_overlay_path: Path,
        auxiliary_tensor_path: Path,
        device: str = "cuda:0",
        attention_backend: str = "torch-flash",
        expected_weight_profile: str | None = None,
        expected_model_repository: str | None = None,
        expected_model_revision: str | None = None,
        expected_adapter_repository: str | None = None,
        expected_adapter_revision: str | None = None,
        expected_nfe: int | None = None,
        expected_scheduler: str | None = None,
        expected_video_flow_shift: float | None = None,
        expected_audio_flow_shift: float | None = None,
        parallel_strategy: str = "single",
    ) -> None:
        import torch

        if attention_backend != "torch-flash":
            raise H3NativeConditioningRuntimeError(
                "the Ref2VA runtime requires Torch Flash attention"
            )
        started = time.monotonic()
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or not torch.cuda.is_available():
            raise H3NativeConditioningRuntimeError("the live native runtime requires CUDA")
        if resolved_device.index is None:
            resolved_device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(resolved_device)
        if capability not in {(8, 6), (8, 9)}:
            raise H3NativeConditioningRuntimeError(
                "the live native runtime currently supports only SM86 and SM89"
            )
        if parallel_strategy not in {"single", "tensor", "sequence-head"}:
            raise H3NativeConditioningRuntimeError("unknown parallel execution strategy")
        devices = (resolved_device,)
        if parallel_strategy != "single":
            if (
                resolved_device.index != 0
                or torch.cuda.device_count() != 2
                or capability != (8, 6)
                or torch.cuda.get_device_capability(1) != (8, 6)
            ):
                raise H3NativeConditioningRuntimeError(
                    "parallel execution requires exactly two visible SM86 GPUs"
                )
            devices = (resolved_device, torch.device("cuda:1"))
            # Establish the complete device group before loading its weights.
            for selected in devices:
                with torch.cuda.device(selected):
                    torch.zeros((), device=selected).item()
            torch.cuda.set_device(resolved_device)
        torch.set_float32_matmul_precision("high")

        contract_started = time.monotonic()
        # Publication already binds every 800 MB block payload.  Re-hashing
        # roughly 40 GB on every worker start burns minutes while the GPU is
        # idle; runtime loading still checks manifest schema, file sizes, and
        # every safetensors header/shape.
        artifact = load_h3_runtime_artifact(
            artifact_path.resolve(strict=True),
            verify_content_hashes=False,
        )
        overlay = load_h3_schedule_overlay(
            schedule_overlay_path.resolve(strict=True),
            artifact=artifact,
        )
        validate_declared_schedule(
            overlay.schedule,
            expected_nfe=expected_nfe,
            expected_scheduler=expected_scheduler,
            expected_video_flow_shift=expected_video_flow_shift,
            expected_audio_flow_shift=expected_audio_flow_shift,
        )
        auxiliary_store = load_h3_runtime_auxiliary(auxiliary_tensor_path)
        expected_capability = f"sm{capability[0]}{capability[1]}"
        supported_weights = (artifact.weight_profile, artifact.adapter_execution) in {
            ("lightx-turbo8-v1.0", "runtime-residual"),
            ("lightx-ref-turbo4-v0.1", "runtime-residual"),
        }
        expected_source = {
            "model_repository": expected_model_repository,
            "model_revision": expected_model_revision,
            "adapter_repository": expected_adapter_repository,
            "adapter_revision": expected_adapter_revision,
        }
        if (
            not artifact.is_complete_block_stack
            or not supported_weights
            or (
                expected_weight_profile is not None
                and artifact.weight_profile != expected_weight_profile
            )
            or any(
                expected is not None and artifact.source.get(name) != expected
                for name, expected in expected_source.items()
            )
            or expected_capability not in artifact.target.compute_capability
            or overlay.target_id != artifact.target.target_id
            or overlay.base_artifact_id != artifact.artifact_id
        ):
            raise H3NativeConditioningRuntimeError(
                "the artifact, schedule, tensor file, and visible GPU do not form one runtime"
            )
        contract_seconds = time.monotonic() - contract_started

        store = auxiliary_store
        input_started = time.monotonic()
        projection_weights = load_h3_native_input_projection_weights(
            base_load=store.load,
            device=resolved_device,
        )
        input_packer = H3NativeInputPacker(projection_weights)
        input_seconds = time.monotonic() - input_started

        final_started = time.monotonic()
        auxiliary = overlay.load_auxiliary_tensors()
        final_weights = load_h3_native_final_layer_weights(
            base_load=store.load,
            time_embeddings=auxiliary["time_embeddings"],
            adaln_table_override=auxiliary["final_adaln_table"],
            device=resolved_device,
        )
        final_layer = H3NativeFinalLayer(final_weights)
        del auxiliary
        torch.cuda.empty_cache()
        final_seconds = time.monotonic() - final_started

        denoiser_started = time.monotonic()
        if parallel_strategy == "single":
            denoiser_type = (
                H3NativeDenoiserBF16Ring
                if capability == (8, 6)
                else H3NativeDenoiserBF16Resident
            )
            denoiser = denoiser_type.load(
                artifact,
                device=resolved_device,
                adaln_table_load=(overlay.load_block_table if overlay.blocks else None),
                attention_backend=attention_backend,
                elementwise_backend="auto",
                adapter_fusion_backend="auto",
                rotary_backend="auto",
            )
        else:
            from vflash.native.h3_parallel import H3NativeDenoiserParallel

            denoiser = H3NativeDenoiserParallel.load(
                artifact,
                devices=devices,
                strategy=parallel_strategy,
                adaln_table_load=(overlay.load_block_table if overlay.blocks else None),
            )
        for selected in devices:
            torch.cuda.synchronize(selected)
        denoiser_seconds = time.monotonic() - denoiser_started

        self._torch = torch
        self.device = resolved_device
        self.devices = devices
        self.parallel_strategy = parallel_strategy
        self.compute_capability = capability
        self.artifact = artifact
        self.overlay = overlay
        self.auxiliary_tensor_path = auxiliary_store.path
        self.input_packer = input_packer
        self.final_layer = final_layer
        self.denoiser = denoiser
        self.attention_backend = attention_backend
        self.initialization_seconds = time.monotonic() - started
        self.initialization_stages = {
            "contract": contract_seconds,
            "input_projection": input_seconds,
            "final_layer": final_seconds,
            "denoiser": denoiser_seconds,
        }

    def metadata(self) -> dict[str, Any]:
        torch = self._torch
        host_memory = torch.cuda.memory.host_memory_stats()
        return {
            "backend": self.backend_id,
            "artifact_id": self.artifact.artifact_id,
            "overlay_id": self.overlay.overlay_id,
            "weight_profile": self.artifact.weight_profile,
            "adapter_execution": self.artifact.adapter_execution,
            "nfe": self.overlay.schedule.nfe,
            "gpu": torch.cuda.get_device_name(self.device),
            "compute_capability": list(self.compute_capability),
            "parallel_strategy": self.parallel_strategy,
            "device_count": len(self.devices),
            "initialization_seconds": self.initialization_seconds,
            "initialization_stages": dict(self.initialization_stages),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "memory": {
                "weight_residency": (
                    "block-ring" if self.compute_capability == (8, 6) else "resident"
                ),
                "pinned_host_weight_payload_bytes": getattr(
                    self.denoiser, "host_weight_bytes", 0
                ),
                # The host allocator names live blocks "active" and all
                # retained CUDA host allocations "allocated", including cache.
                "pinned_host_allocated_bytes": host_memory.get("active_bytes.current", 0),
                "pinned_host_reserved_bytes": host_memory.get("allocated_bytes.current", 0),
                "device_allocated_bytes": torch.cuda.memory_allocated(self.device),
                "device_reserved_bytes": torch.cuda.memory_reserved(self.device),
                "devices": [
                    {
                        "index": device.index,
                        "allocated_bytes": torch.cuda.memory_allocated(device),
                        "reserved_bytes": torch.cuda.memory_reserved(device),
                    }
                    for device in self.devices
                ],
            },
            "attention_policy": _attention_policy(),
        }

    def _load_request_tensors(self, bundle_directory: Path) -> tuple[Any, dict[str, Any]]:
        torch = self._torch
        bundle = load_h3_conditioning_bundle(bundle_directory)
        if (
            bundle.profile.task != "ref2va"
            or bundle.source.get("model_repository")
            != self.artifact.source.get("model_repository")
            or bundle.source.get("model_revision") != self.artifact.source.get("model_revision")
            or bundle.source.get("transformer_sha256")
            != self.artifact.source.get("transformer_sha256")
            or bundle.source.get("oracle") != self.artifact.source.get("oracle")
            or bundle.source.get("oracle_revision")
            != self.artifact.source.get("oracle_revision")
            or bundle.source.get("oracle_profile") != self.artifact.source.get("oracle_profile")
            or bundle.source.get("oracle_hardware")
            != self.artifact.source.get("oracle_hardware")
        ):
            raise H3NativeConditioningRuntimeError(
                "the conditioning bundle and native Transformer are not the same H3 model"
            )
        tensor_path = bundle.directory / "conditioning.safetensors"
        video_indices = load_safetensor_tensor(tensor_path, "video_indices")
        audio_indices = load_safetensor_tensor(tensor_path, "audio_indices")
        text_indices = load_safetensor_tensor(tensor_path, "text_indices")
        first_packed = load_safetensor_tensor(tensor_path, "first_packed_input")
        refined_text = first_packed.index_select(1, text_indices.to(torch.int64)).contiguous()
        tensors = {
            "initial_video": load_safetensor_tensor(tensor_path, "initial_video_latents"),
            "initial_audio": load_safetensor_tensor(tensor_path, "initial_audio_latents"),
            "refined_text": refined_text,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "token_tags": load_safetensor_tensor(tensor_path, "token_tags"),
            "rotary_cos": load_safetensor_tensor(tensor_path, "rotary_cos"),
            "rotary_sin": load_safetensor_tensor(tensor_path, "rotary_sin"),
        }
        return bundle, tensors

    def generate_latents(
        self,
        bundle_directory: Path,
        output_path: Path,
        *,
        progress_callback: Callable[[float, float], None] | None = None,
    ) -> H3NativeConditioningRuntimeResult:
        """Run one live request and export target-only VAE-ready latents."""

        torch = self._torch
        if output_path.exists() or output_path.is_symlink():
            raise H3NativeConditioningRuntimeError("native latent output already exists")
        total_started = time.monotonic()
        prepare_started = time.monotonic()
        bundle, tensors = self._load_request_tensors(bundle_directory)
        device = self.device
        state = H3NativeLatentState(
            self.overlay.schedule,
            video_latents=tensors["initial_video"].to(device=device, dtype=torch.float32),
            audio_latents=tensors["initial_audio"].to(device=device, dtype=torch.float32),
            num_condition_video_rows=bundle.profile.num_condition_video_rows,
            num_condition_audio_rows=bundle.profile.num_condition_audio_rows,
        )
        reset = getattr(self.denoiser, "reset", None)
        if callable(reset):
            reset()
        engine = H3NativeEngine(
            schedule=self.overlay.schedule,
            latent_state=state,
            input_packer=self.input_packer,
            denoiser=self.denoiser,
            final_layer=self.final_layer,
            refined_text=tensors["refined_text"].to(device=device, dtype=torch.bfloat16),
            video_indices=tensors["video_indices"].to(device),
            audio_indices=tensors["audio_indices"].to(device),
            text_indices=tensors["text_indices"].to(device),
            token_tags=tensors["token_tags"].to(device),
            rotary_cos=tensors["rotary_cos"].to(device),
            rotary_sin=tensors["rotary_sin"].to(device),
        )
        del tensors
        torch.cuda.synchronize(device)
        prepare_seconds = time.monotonic() - prepare_started

        for selected in self.devices:
            torch.cuda.reset_peak_memory_stats(selected)
        denoise_started = time.monotonic()
        with torch.inference_mode():
            while not engine.complete:
                engine.step()
                if progress_callback is not None:
                    progress_callback(engine.evaluation_index, self.overlay.schedule.nfe)
        for selected in self.devices:
            torch.cuda.synchronize(selected)
        denoise_seconds = time.monotonic() - denoise_started
        if not bool(
            torch.isfinite(engine.latent_state.video_latents).all()
            and torch.isfinite(engine.latent_state.audio_latents).all()
        ):
            raise H3NativeConditioningRuntimeError(
                "the native trajectory produced non-finite latents"
            )
        device_peaks = tuple(
            int(torch.cuda.max_memory_allocated(selected)) for selected in self.devices
        )
        peak = max(device_peaks)

        export_started = time.monotonic()
        packed_video = engine.latent_state.video_latents.detach().to("cpu")
        packed_audio = engine.latent_state.audio_latents.detach().to("cpu")
        video_latents = unpatchify_h3_video_rows(
            packed_video[:, bundle.profile.num_condition_video_rows :],
            num_latent_frames=h3_video_latent_frame_count(bundle.profile.frames),
            latent_height=bundle.profile.height // 16,
            latent_width=bundle.profile.width // 16,
        )
        audio_latents = unpack_h3_audio_rows(
            packed_audio[:, bundle.profile.num_condition_audio_rows :]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors_atomic(
            output_path,
            {
                "audio_latents": audio_latents,
                "video_latents": video_latents,
            },
            metadata={
                "bundle_id": bundle.bundle_id,
                "attention_backend": self.attention_backend,
                "frames": str(bundle.profile.frames),
                "height": str(bundle.profile.height),
                "schedule_nfe": str(self.overlay.schedule.nfe),
                "width": str(bundle.profile.width),
                "parallel_strategy": self.parallel_strategy,
            },
        )
        export_seconds = time.monotonic() - export_started
        elapsed = time.monotonic() - total_started
        del engine, state, packed_video, packed_audio, video_latents, audio_latents
        torch.cuda.empty_cache()
        return H3NativeConditioningRuntimeResult(
            bundle_id=bundle.bundle_id,
            output_path=output_path.resolve(strict=True),
            conditioning_profile=asdict(bundle.profile),
            execution_schedule=self.overlay.schedule.to_mapping(),
            nfe=self.overlay.schedule.nfe,
            elapsed_seconds=elapsed,
            peak_allocated_bytes=peak,
            attention_policy=_attention_policy(),
            stage_durations={
                "request_prepare": prepare_seconds,
                "denoise": denoise_seconds,
                "latent_export": export_seconds,
            },
            peak_allocated_bytes_by_device=device_peaks,
        )

    def close(self) -> None:
        close = getattr(self.denoiser, "close", None)
        if close is not None:
            close()
