"""One owned prompt/reference-to-MP4 pipeline, without product service dependencies."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from traceback import clear_frames
from typing import Any

from vflash.adapters.diffusers_h3 import DiffusersConditioner, validate_adapter_dependencies
from vflash.adapters.references import DecodedReference, read_reference
from vflash.adapters.video_references import DecodedVideoReference, read_video_reference
from vflash.attention import resolve_attention_backend
from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.media.encoding import media_executables
from vflash.media.runtime import OfficialMediaDecoder
from vflash.model_assets import model_profile
from vflash.native.h3_conditioning_bundle import H3InMemoryConditioning
from vflash.native.h3_hybrid import HybridModel
from vflash.native.runner import NativeEngineSession
from vflash.pipeline.admission import validate_request
from vflash.pipeline.assets import PreparedPipelineAssets
from vflash.pipeline.attention_adapter import AttentionAdapter, open_attention_adapter
from vflash.pipeline.contracts import (
    ConditioningReuseScope,
    PipelineProgress,
    VideoRequest,
    VideoResult,
)
from vflash.planner import resolve_plan


class H3Pipeline:
    """Serial, persistent H3 encoding, native inference, decoding and MP4.

    Construct before CUDA initialization in a dedicated process. Construction
    validates the fixed profile on the CPU. ``prepare()`` explicitly warms its
    stages; otherwise the first valid request loads them after input decoding.
    Native weights
    use the block ring by default so encoding and VAE stages can take turns on
    the same device. An explicit H100 resident option retains the trunk on GPU
    within its measured request envelope; see pipeline.residency. The object owns
    its stages and temporary files;
    the caller owns scheduling, requests, final MP4s and any account/storage data.
    """

    def __init__(
        self,
        prepared: PreparedPipelineAssets,
        *,
        device: NvidiaDevice,
        trust_local_code: bool = False,
        peer_device: NvidiaDevice | None = None,
        strategy: str | None = None,
        attention_backend: str = "auto",
        veda_predictor: Path | None = None,
        veda_dense_backend: str = "torch-flash",
        attention_adapter: AttentionAdapter | None = None,
        hybrid_model: HybridModel | None = None,
        weight_residency: str = "block-ring",
        media_video_input: str = "file",
    ) -> None:
        if trust_local_code is not True:
            raise ContractError("the official decoder adapter requires trust_local_code=True")
        if not isinstance(prepared, PreparedPipelineAssets):
            raise ContractError("prepare the immutable pipeline assets before loading a model")
        prepared.check_unchanged()
        media_executables()
        validate_adapter_dependencies()
        plan = resolve_plan(
            ProfileCatalog.bundled(),
            profile_id=prepared.profile_id,
            device=device,
            peer_device=peer_device,
            strategy=strategy,
        )
        if (
            plan.target.compute_capability == "8.6"
            and plan.parallel_strategy != "sequence-head"
        ):
            raise ContractError(
                "the complete SM86 pipeline requires two GPUs with sequence-head execution"
            )
        if media_video_input not in {"file", "pipe"}:
            raise ContractError("media_video_input must be file or pipe")
        self.media_video_input = media_video_input
        self.prepared = prepared
        self.attention_backend = resolve_attention_backend(plan, attention_backend)
        from vflash.native.h3_veda_attention import validate_predictor

        validate_predictor(self.attention_backend, veda_predictor)
        self.veda_predictor = veda_predictor
        from vflash.native.h3_veda_dense import (
            require_dense_dependencies,
            validate_dense_backend,
        )

        validate_dense_backend(
            veda_dense_backend,
            attention_backend=self.attention_backend,
            capability=plan.target.compute_capability,
        )
        require_dense_dependencies(veda_dense_backend)
        self.veda_dense_backend = veda_dense_backend
        self.profile = model_profile(prepared.profile_id)
        if attention_adapter is not None and (
            not isinstance(attention_adapter, AttentionAdapter)
            or plan.target.compute_capability != "8.9"
            or plan.parallel_strategy != "single"
            or prepared.profile_id not in {"i2va-base16-bf16-sm89", "fl2va-base16-bf16-sm89"}
        ):
            raise ContractError(
                "attention adapter requires a single-SM89 Base16 keyframe pipeline"
            )
        self.attention_adapter = attention_adapter
        if hybrid_model is not None and not isinstance(hybrid_model, HybridModel):
            raise ContractError("hybrid_model requires a typed HybridModel")
        self.hybrid_model = hybrid_model
        from vflash.pipeline.residency import validate_residency

        self.weight_residency = validate_residency(
            plan, weight_residency, self.attention_backend, hybrid_model
        )
        self._hybrid_stamps = (
            hybrid_model.validate(
                profile_id=prepared.profile_id,
                capability=plan.target.compute_capability,
                strategy=plan.parallel_strategy,
                attention_backend=self.attention_backend,
            )
            if hybrid_model is not None
            else ()
        )
        self._reference_graph = None
        self._adapter_stack = None
        self._plan = plan
        self._lock = threading.Lock()
        self._active_thread_id: int | None = None
        self._closed = self._released = False
        self._core = self._conditioner = self._media = None
        self.request_count = 0
        self.initialization_stages: dict[str, float] = {}
        self.initialization_seconds = 0.0
        self._loaded = False

    def _ensure_prepared(self) -> float:
        """Called with the session lock held; return only this call's cold load."""
        if self._loaded:
            return 0.0
        started = time.monotonic()
        try:
            self._load_stages(self._plan)
        except BaseException as exc:
            try:
                self._close_owned()
            except BaseException:
                exc.add_note("Vflash cleanup could not complete; exit the pipeline process.")
            else:
                clear_frames(exc.__traceback__)
            raise
        self.initialization_seconds = time.monotonic() - started
        self._loaded = True
        return self.initialization_seconds

    def prepare(self) -> None:
        """Explicitly preload this fixed pipeline before accepting requests.

        Idempotent and serial; it does not require or decode a sample request.
        The caller chooses to initialize CUDA here. Without this call, generate
        validates and decodes its first input before touching model resources.
        """
        if not self._lock.acquire(blocking=False):
            raise ContractError("this pipeline is busy")
        try:
            if self._closed:
                raise ContractError("the pipeline is closed")
            self.prepared.check_unchanged()
            self._check_hybrid_unchanged()
            self._ensure_prepared()
        finally:
            self._lock.release()

    def _load_stages(self, plan: Any) -> None:
        import torch

        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        assets = self.prepared.assets
        started = time.monotonic()
        # This session selects the explicit physical device before the first
        # CUDA allocation. No other stage is allowed to choose or reset CUDA.
        self._core = NativeEngineSession(
            plan,
            artifact=assets.artifact,
            schedule_overlay=assets.schedule_overlay,
            auxiliary_tensor=assets.auxiliary_tensor,
            weight_residency=self.weight_residency,
            attention_backend=self.attention_backend,
            **(
                {"veda_dense_backend": self.veda_dense_backend}
                if self.veda_dense_backend != "torch-flash"
                else {}
            ),
            **(
                {"veda_predictor": self.veda_predictor}
                if self.veda_predictor is not None
                else {}
            ),
            **({"hybrid_model": self.hybrid_model} if self.hybrid_model is not None else {}),
        )
        self.initialization_stages["native"] = time.monotonic() - started
        if self.attention_adapter is not None:
            started = time.monotonic()
            self._adapter_stack = ExitStack()
            self._adapter_stack.enter_context(
                open_attention_adapter(self._core.runtime, self.attention_adapter)
            )
            self.initialization_stages["attention_adapter"] = time.monotonic() - started
        started = time.monotonic()
        self._conditioner = DiffusersConditioner(self.prepared)
        if self.hybrid_model is not None:
            from vflash.adapters.hybrid_reference import HybridReferenceGraph

            self._reference_graph = HybridReferenceGraph(
                self._conditioner, self._core.runtime.metadata()["model_variant"]
            )
        self.initialization_stages["conditioning"] = time.monotonic() - started
        started = time.monotonic()
        self._media = OfficialMediaDecoder(
            video_component=assets.decoder_directory / "video_vae",
            audio_component=assets.decoder_directory / "audio_vae",
            trust_local_code=True,
        )
        self.initialization_stages["media"] = time.monotonic() - started

    def generate(
        self,
        request: VideoRequest,
        output_path: Path,
        *,
        progress: Callable[[PipelineProgress], None] | None = None,
        profile_denoise: bool = False,
        conditioning_reuse_scope: ConditioningReuseScope | None = None,
    ) -> VideoResult:
        """Return a complete MP4; failed or concurrent requests never publish a partial file."""
        started = time.monotonic()
        if self._closed:
            raise ContractError("the pipeline is closed")
        if not isinstance(request, VideoRequest) or not isinstance(output_path, Path):
            raise ContractError("generate requires a VideoRequest and pathlib.Path output")
        if conditioning_reuse_scope is not None and not isinstance(
            conditioning_reuse_scope, ConditioningReuseScope
        ):
            raise ContractError("conditioning reuse requires a typed opaque scope")
        validate_request(
            request,
            profile_id=self.profile.definition.id,
            hybrid=self.hybrid_model is not None,
            parallel_strategy=self._plan.parallel_strategy,
            attention_backend=self.attention_backend,
            weight_residency=self.weight_residency,
        )
        if output_path.exists() or output_path.is_symlink():
            raise ContractError("the output path already exists")
        if not self._lock.acquire(blocking=False):
            raise ContractError(
                "this pipeline is busy; schedule the next request after completion"
            )
        references: list[DecodedReference | DecodedVideoReference] = []
        self._active_thread_id = threading.get_ident()
        try:
            if self._closed:
                raise ContractError("the pipeline is closed")
            # Input errors are checked before stage activation and do not retire
            # an otherwise healthy session. Read/hash exactly the decoded bytes.
            self.prepared.check_unchanged()
            self._check_hybrid_unchanged()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=output_path.parent):
                pass
            reference_started = time.monotonic()
            for path in request.ordered_references:
                references.append(read_reference(path))
            if request.first_frame is not None:
                references.append(read_reference(request.first_frame))
            if request.last_frame is not None:
                references.append(read_reference(request.last_frame))
            if request.reference_video is not None:
                references.append(read_video_reference(request.reference_video))
            reference_loading_seconds = time.monotonic() - reference_started
            input_preparation_seconds = time.monotonic() - started
            try:
                initialization_seconds = self._ensure_prepared()
                result = self._generate_one(
                    request,
                    tuple(references),
                    output_path,
                    progress=progress,
                    profile_denoise=profile_denoise,
                    conditioning_reuse_scope=conditioning_reuse_scope,
                )
            except BaseException as exc:
                try:
                    self._close_owned()
                except BaseException:
                    exc.add_note("CUDA cleanup could not complete; exit the pipeline process.")
                else:
                    clear_frames(exc.__traceback__)
                raise
        finally:
            try:
                # Attempt every reference release even if one close raises.
                from contextlib import ExitStack

                with ExitStack() as cleanup:
                    for reference in references:
                        cleanup.callback(reference.close)
            finally:
                self._active_thread_id = None
                self._lock.release()
        # Report the public call, including input checks, image decoding/hash,
        # temporary-file cleanup and reference.close(). A first cold load is
        # included, and separately identified. Reference loading is nested
        # inside preparation, not added again to an independent total.
        elapsed = time.monotonic() - started
        return replace(
            result,
            elapsed_seconds=elapsed,
            stages={
                **result.stages,
                "initialization_seconds": initialization_seconds,
                "request_elapsed_seconds": elapsed - initialization_seconds,
                "input_preparation": {
                    "elapsed_seconds": input_preparation_seconds,
                    "reference_loading_seconds": reference_loading_seconds,
                },
            },
        )

    def _generate_one(
        self,
        request: VideoRequest,
        references: tuple[DecodedReference | DecodedVideoReference, ...],
        output_path: Path,
        *,
        progress: Callable[[PipelineProgress], None] | None,
        profile_denoise: bool,
        conditioning_reuse_scope: ConditioningReuseScope | None,
    ) -> VideoResult:
        def report(stage: str, completed: int, total: int) -> None:
            if progress is not None:
                progress(PipelineProgress(stage, completed, total))

        started = time.monotonic()
        output_path = output_path.parent.resolve() / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stages: dict[str, Any] = {
            "session_initialization_seconds": self.initialization_seconds,
            "request_index": self.request_count + 1,
        }
        with tempfile.TemporaryDirectory(
            dir=output_path.parent, prefix=".vflash-request-"
        ) as tmp:
            directory = Path(tmp)
            stage_started = time.monotonic()
            report("encoding", 0, 1)
            weight_resume_seconds = self._conditioner.resume_cuda()
            call_started = time.monotonic()
            capture_options = (
                {"reuse_scope": conditioning_reuse_scope}
                if conditioning_reuse_scope is not None
                else {}
            )
            capture_owner = (
                self._reference_graph
                if self._reference_graph is not None and request.mode == "ref2va"
                else self._conditioner
            )
            # Reference prefixes are not the keyframe cache contract. Reuse is
            # request-scoped only on the existing keyframe conditioning path.
            if capture_owner is self._reference_graph:
                capture_options = {}
            bundle = capture_owner.capture(
                request,
                references,
                directory / "conditioning",
                **capture_options,
            )
            capture_call_seconds = time.monotonic() - call_started
            capture_diagnostics = getattr(capture_owner, "last_capture_diagnostics", {})
            in_memory = isinstance(bundle, H3InMemoryConditioning)
            suspend_seconds = self._conditioner.suspend_cuda()
            report("encoding", 1, 1)
            stages["encoding"] = {
                "elapsed_seconds": time.monotonic() - stage_started,
                "weight_resume_seconds": weight_resume_seconds,
                "capture_call_seconds": capture_call_seconds,
                "capture_diagnostics": dict(capture_diagnostics)
                if isinstance(capture_diagnostics, dict)
                else {},
                "suspend_seconds": suspend_seconds,
                "profile": asdict(bundle.profile),
                "source": dict(bundle.source),
                "bundle_id": bundle.bundle_id,
                "conditioning_transport": "in-memory" if in_memory else "persisted-file",
                "conditioning_tensor_bytes": next(
                    (
                        row.size_bytes
                        for row in getattr(bundle, "files", ())
                        if row.role == "conditioning"
                    ),
                    None,
                ),
            }
            report("denoising", 0, self.profile.definition.nfe)
            native_options: dict[str, Any] = {
                "progress_callback": lambda completed, total: report(
                    "denoising", completed, total
                )
            }
            if profile_denoise:
                native_options["profile_denoise"] = True
            native = self._core.generate(
                bundle if in_memory else bundle.directory,
                directory / "latents.safetensors",
                **native_options,
            )
            stages["denoising"] = {
                key: value
                for key, value in native["generation"].items()
                if key != "output_path"
            }
            if self.hybrid_model is not None:
                stages["model_variant"] = dict(self._core.runtime.metadata()["model_variant"])
            if self.attention_adapter is not None:
                stages["denoising"]["attention_adapter"] = self.attention_adapter.metadata()
            stage_started = time.monotonic()
            report("decoding", 0, 1)
            weight_resume_seconds = self._media.resume_cuda()
            call_started = time.monotonic()
            media_options = (
                {"audio_delivery_profile": request.audio_delivery_profile}
                if request.audio_delivery_profile != "unchanged"
                else {}
            )
            if self.media_video_input == "pipe":
                media_options["video_input"] = "pipe"
            if request.keyframe_delivery_profile != "decoded":
                media_options["keyframe_delivery_profile"] = request.keyframe_delivery_profile
                if request.mode in {"i2va", "fl2va"}:
                    media_options["first_frame"] = references[0].image
                if request.mode == "l2va":
                    media_options["last_frame"] = references[0].image
                elif request.mode == "fl2va":
                    media_options["last_frame"] = references[1].image
            media = self._media.generate_mp4(
                directory / "latents.safetensors",
                directory / "video.mp4",
                height=request.height,
                width=request.width,
                duration_seconds=request.duration_seconds,
                fps=24,
                **media_options,
            )
            decode_call_seconds = time.monotonic() - call_started
            suspend_seconds = self._media.suspend_cuda()
            report("decoding", 1, 1)
            stages["media"] = {
                "elapsed_seconds": time.monotonic() - stage_started,
                "weight_resume_seconds": weight_resume_seconds,
                "decode_call_seconds": decode_call_seconds,
                "suspend_seconds": suspend_seconds,
                "stage_durations": media.stage_durations,
                "peak_allocated_bytes": media.peak_allocated_bytes,
            }
            os.link(directory / "video.mp4", output_path)
        self.request_count += 1
        # A callback cannot turn a successfully published MP4 into a reported
        # generation failure. The final result is the completion notification.
        return VideoResult(
            output_path,
            self.prepared.profile_id,
            request.seed,
            time.monotonic() - started,
            stages,
            media.media,
            request_mode=request.mode,
        )

    def _check_hybrid_unchanged(self) -> None:
        if self.hybrid_model is not None and self.hybrid_model.stamps() != self._hybrid_stamps:
            raise ContractError(
                "the hybrid reference checkpoint changed; construct a new pipeline"
            )

    def _close_owned(self) -> None:
        if self._released:
            return
        self._closed = True
        failures: list[BaseException] = []
        for name in ("_reference_graph", "_adapter_stack", "_conditioner", "_media", "_core"):
            owner = getattr(self, name)
            if owner is not None:
                try:
                    owner.close()
                except BaseException as exc:
                    failures.append(exc)
                else:
                    setattr(self, name, None)
        if failures:
            for additional in failures[1:]:
                failures[0].add_note(
                    f"Another stage failed to close: {type(additional).__name__}"
                )
            raise failures[0]
        self._released = True
        self._loaded = False

    def close(self) -> None:
        if getattr(self, "_active_thread_id", None) == threading.get_ident():
            raise ContractError("raise from the progress callback to cancel; do not call close")
        with self._lock:
            self._close_owned()

    def __enter__(self) -> H3Pipeline:
        if self._closed:
            raise ContractError("the pipeline is closed")
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()
