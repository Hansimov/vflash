"""One owned prompt/reference-to-MP4 pipeline, without product service dependencies."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
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
from vflash.media.encoding import MediaError, media_executables, validate_video_threads
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
        veda_dense_edge_layers: int = 5,
        attention_adapter: AttentionAdapter | None = None,
        hybrid_model: HybridModel | None = None,
        weight_residency: str = "block-ring",
        media_video_input: str = "file",
        media_video_threads: int | None = None,
        conditioning_transport: str = "file",
        audio_decode_overlap: bool = False,
        text_encoder_loader: Callable[[Path], Any] | None = None,
        native_denoiser_loader: Callable[..., Any] | None = None,
    ) -> None:
        if trust_local_code is not True:
            raise ContractError("the official decoder adapter requires trust_local_code=True")
        if not isinstance(prepared, PreparedPipelineAssets):
            raise ContractError("prepare the immutable pipeline assets before loading a model")
        if text_encoder_loader is not None and not callable(text_encoder_loader):
            raise ContractError("text_encoder_loader must be a trusted Python callable")
        self._text_encoder_loader = text_encoder_loader
        if native_denoiser_loader is not None and (
            not callable(native_denoiser_loader)
            or strategy not in {None, "single"}
            or peer_device is not None
            or weight_residency != "resident"
        ):
            raise ContractError(
                "native_denoiser_loader requires a callable and single-device resident weights"
            )
        self._native_denoiser_loader = native_denoiser_loader
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
        if conditioning_transport not in {"file", "memory"}:
            raise ContractError("conditioning_transport must be file or memory")
        if conditioning_transport == "memory" and plan.parallel_strategy != "single":
            raise ContractError(
                "memory conditioning requires a same-process single-device pipeline"
            )
        self.conditioning_transport = conditioning_transport
        if type(audio_decode_overlap) is not bool:
            raise ContractError("audio_decode_overlap must be boolean")
        self.audio_decode_overlap = audio_decode_overlap
        if media_video_input not in {"file", "pipe"}:
            raise ContractError("media_video_input must be file or pipe")
        try:
            validate_video_threads(media_video_threads)
        except MediaError as exc:
            raise ContractError(str(exc)) from exc
        self.media_video_threads = media_video_threads
        self.media_video_input = media_video_input
        self.prepared = prepared
        self.attention_backend = resolve_attention_backend(plan, attention_backend)
        from vflash.native.h3_veda_attention import dense_layers_for_edges, validate_predictor

        validate_predictor(self.attention_backend, veda_predictor)
        self.veda_predictor = veda_predictor
        dense_layers_for_edges(veda_dense_edge_layers, attention_backend=self.attention_backend)
        self.veda_dense_edge_layers = veda_dense_edge_layers
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
        self._scene_conditioning = None
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
            **(
                {"native_denoiser_loader": self._native_denoiser_loader}
                if self._native_denoiser_loader is not None
                else {}
            ),
            schedule_overlay=assets.schedule_overlay,
            auxiliary_tensor=assets.auxiliary_tensor,
            weight_residency=self.weight_residency,
            attention_backend=self.attention_backend,
            **(
                {"veda_dense_edge_layers": self.veda_dense_edge_layers}
                if self.veda_dense_edge_layers != 5
                else {}
            ),
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
        self._conditioner = DiffusersConditioner(
            self.prepared,
            **(
                {"text_encoder_loader": self._text_encoder_loader}
                if self._text_encoder_loader is not None
                else {}
            ),
        )
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

    @contextmanager
    def source_scene_conditioning(self, request: VideoRequest):
        """Opt into non-equivalent source-scene context for related I2VA repaints.

        Generate the declared source in this scope, then enter owner.repaint()
        around derived video-only calls. The 128 MiB CPU feature owner is cleared
        on exit/failure/close and cannot be used by another thread or pipeline.
        Original-image Qwen context replaces the crop context; qualify quality
        for the selected workflow. Ordinary generation remains unchanged.
        """
        from vflash.adapters.scene_conditioning import SourceSceneConditioning

        if not self._lock.acquire(blocking=False):
            raise ContractError("scene conditioning requires an idle pipeline")
        try:
            if self._closed or getattr(self, "_scene_conditioning", None) is not None:
                raise ContractError("scene conditioning requires an open, unscoped pipeline")
            owner = SourceSceneConditioning(request)
            self._scene_conditioning = owner
        finally:
            self._lock.release()
        try:
            yield owner
        finally:
            with self._lock:
                owner.close()
                self._scene_conditioning = None

    def generate(
        self,
        request: VideoRequest,
        output_path: Path,
        *,
        progress: Callable[[PipelineProgress], None] | None = None,
        profile_denoise: bool = False,
        conditioning_reuse_scope: ConditioningReuseScope | None = None,
        audio_refinement: Any | None = None,
        latent_output: Path | None = None,
        native_av_prefix: Any | None = None,
    ) -> VideoResult:
        """Return a complete MP4; failed or concurrent requests never publish a partial file."""
        return self._generate(
            request,
            output_path,
            progress=progress,
            profile_denoise=profile_denoise,
            conditioning_reuse_scope=conditioning_reuse_scope,
            latent_output=latent_output,
            **({"native_av_prefix": native_av_prefix} if native_av_prefix is not None else {}),
            **({"audio_refinement": audio_refinement} if audio_refinement is not None else {}),
        )

    def generate_rgb(
        self,
        request: VideoRequest,
        output_path: Path,
        *,
        required_frames: tuple[int, ...] | None = None,
        progress: Callable[[PipelineProgress], None] | None = None,
        profile_denoise: bool = False,
        conditioning_reuse_scope: ConditioningReuseScope | None = None,
    ) -> VideoResult:
        """Return uint8 THWC NPY pixels without encoding an intermediate video.

        This explicit video-only path preserves the model's padded timeline.
        With required_frames, only those indices are valid; all others are zero.
        The normal generate() method continues to deliver complete MP4 media.
        """
        return self._generate(
            request,
            output_path,
            rgb=True,
            required_frames=required_frames,
            progress=progress,
            profile_denoise=profile_denoise,
            conditioning_reuse_scope=conditioning_reuse_scope,
        )

    def _generate(
        self,
        request: VideoRequest,
        output_path: Path,
        *,
        rgb: bool = False,
        required_frames: tuple[int, ...] | None = None,
        progress: Callable[[PipelineProgress], None] | None = None,
        profile_denoise: bool = False,
        conditioning_reuse_scope: ConditioningReuseScope | None = None,
        audio_refinement: Any | None = None,
        latent_output: Path | None = None,
        native_av_prefix: Any | None = None,
    ) -> VideoResult:
        """Publish one complete artifact; failures never publish partial output."""
        started = time.monotonic()
        if self._closed:
            raise ContractError("the pipeline is closed")
        if not isinstance(request, VideoRequest) or not isinstance(output_path, Path):
            raise ContractError("generate requires a VideoRequest and pathlib.Path output")
        from vflash.pipeline.latent_output import prepare_latent_output

        latent_output = prepare_latent_output(latent_output, output_path)
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
        if native_av_prefix is not None:
            from vflash.native.h3_av_continuation import NativeAVPrefix

            if (
                not isinstance(native_av_prefix, NativeAVPrefix)
                or request.mode != "i2va"
                or not request.generate_audio
                or rgb
                or audio_refinement is not None
                or profile_denoise
                or self._plan.parallel_strategy != "single"
                or (self.weight_residency, self.attention_backend)
                != ("block-ring", "veda-sm89")
                or tuple(native_av_prefix.video.shape[-2:])
                != (request.height // 16, request.width // 16)
            ):
                raise ContractError("native AV prefix requires original-audio Veda ring I2VA")
        if audio_refinement is not None:
            from vflash.media.audio_delivery import SILENT_AUDIO_PROFILE
            from vflash.native.h3_audio_refinement import AudioRefinementRequest

            if (
                not isinstance(audio_refinement, AudioRefinementRequest)
                or request.mode != "i2va"
                or not request.generate_audio
                or rgb
                or request.audio_delivery_profile == SILENT_AUDIO_PROFILE
                or profile_denoise
                or self.hybrid_model is None
                or (self.weight_residency, self.attention_backend)
                not in {("resident", "veda-triton"), ("block-ring", "veda-sm89")}
                or self._plan.parallel_strategy != "single"
                or self.veda_dense_edge_layers not in (3, 4)
            ):
                raise ContractError(
                    "audio refinement requires an unprofiled hybrid I2VA pipeline with Veda"
                )
        if rgb:
            from vflash.media.rgb_delivery import validate_rgb_selection

            if (
                request.generate_audio
                or request.audio_delivery_profile != "unchanged"
                or request.keyframe_delivery_profile != "decoded"
            ):
                raise ContractError("RGB delivery requires video-only decoded keyframes")
            validate_rgb_selection(request.model_frames, required_frames)
        elif required_frames is not None:
            raise ContractError("partial decoding is only supported by RGB delivery")
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
            scene = getattr(self, "_scene_conditioning", None)
            if scene is not None:
                scene.check_request(request, reuse_scope=conditioning_reuse_scope)
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
                    rgb=rgb,
                    required_frames=required_frames,
                    latent_output=latent_output,
                    **(
                        {"native_av_prefix": native_av_prefix}
                        if native_av_prefix is not None
                        else {}
                    ),
                    **(
                        {"audio_refinement": audio_refinement}
                        if audio_refinement is not None
                        else {}
                    ),
                )
                if scene is not None:
                    scene.request_complete()
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
        rgb: bool = False,
        required_frames: tuple[int, ...] | None = None,
        audio_refinement: Any | None = None,
        latent_output: Path | None = None,
        native_av_prefix: Any | None = None,
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
            scene = getattr(self, "_scene_conditioning", None)
            if scene is not None:
                capture_options["scene_conditioning"] = scene
            capture_owner = (
                self._reference_graph
                if self._reference_graph is not None and request.mode == "ref2va"
                else self._conditioner
            )
            # Reference prefixes are not the keyframe cache contract. Reuse is
            # request-scoped only on the existing keyframe conditioning path.
            if capture_owner is self._reference_graph:
                capture_options = {}
            capture = (
                capture_owner.capture_in_memory
                if capture_owner is self._conditioner
                and self.conditioning_transport == "memory"
                else capture_owner.capture
            )
            bundle = capture(
                request,
                references,
                directory / "conditioning",
                **capture_options,
            )
            capture_call_seconds = time.monotonic() - call_started
            capture_diagnostics = getattr(capture_owner, "last_capture_diagnostics", {})
            in_memory = isinstance(bundle, H3InMemoryConditioning)
            if (
                capture_owner is self._conditioner
                and self.conditioning_transport == "memory"
                and not in_memory
            ):
                raise ContractError("memory conditioning did not return its one-shot owner")
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
                "conditioning_tensor_bytes": bundle.tensor_bytes
                if in_memory
                else next(
                    (
                        row.size_bytes
                        for row in getattr(bundle, "files", ())
                        if row.role == "conditioning"
                    ),
                    None,
                ),
            }
            report(
                "denoising",
                0,
                self.profile.definition.nfe
                + (audio_refinement.steps if audio_refinement is not None else 0),
            )
            native_options: dict[str, Any] = {
                "progress_callback": lambda completed, total: report(
                    "denoising", completed, total
                )
            }
            if profile_denoise:
                native_options["profile_denoise"] = True
            if not request.generate_audio:
                native_options["generate_audio"] = False
            if audio_refinement is not None:
                native_options["audio_refinement"] = audio_refinement
            if native_av_prefix is not None:
                native_options["native_av_prefix"] = native_av_prefix
            predecoded_video = None
            overlap_enabled = (
                getattr(self, "audio_decode_overlap", False)
                and audio_refinement is not None
                and request.duration_seconds == 5
                and request.width * request.height <= 1024**2
            )
            with ExitStack() as decode_scope:
                overlap = None
                if overlap_enabled:
                    from vflash.media.overlap import AudioVideoOverlap

                    overlap = decode_scope.enter_context(
                        AudioVideoOverlap(
                            self._media, width=request.width, height=request.height
                        )
                    )
                    native_options["video_ready_callback"] = overlap.submit
                native = self._core.generate(
                    bundle if in_memory else bundle.directory,
                    directory / "latents.safetensors",
                    **native_options,
                )
                if latent_output is not None:
                    from vflash.pipeline.latent_output import retain_native_latents

                    retain_native_latents(directory / "latents.safetensors", latent_output)
                if overlap is not None:
                    predecoded_video = overlap.finish()
            if overlap is not None:
                stages["audio_video_overlap"] = dict(overlap.record)
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
            weight_resume_seconds = (
                self._media.resume_cuda()
                if request.generate_audio
                else self._media.resume_cuda(include_audio=False)
            )
            call_started = time.monotonic()
            media_options = (
                {"audio_delivery_profile": request.audio_delivery_profile}
                if request.audio_delivery_profile != "unchanged"
                else {}
            )
            if predecoded_video is not None:
                media_options["predecoded_video"] = predecoded_video
            if not request.generate_audio:
                media_options["generate_audio"] = False
            if self.media_video_threads is not None:
                media_options["video_threads"] = self.media_video_threads
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
            if rgb:
                from vflash.media.rgb_delivery import generate_rgb

                media_path = directory / "video.npy"
                media = generate_rgb(
                    self._media,
                    directory / "latents.safetensors",
                    media_path,
                    height=request.height,
                    width=request.width,
                    duration_seconds=request.duration_seconds,
                    model_frames=request.model_frames,
                    required_frames=required_frames,
                )
            else:
                media_path = directory / "video.mp4"
                headroom = {}
                with ExitStack() as audio_guard:
                    if audio_refinement is not None:
                        from vflash.media.audio_headroom import audio_decoder_headroom

                        audio_guard.enter_context(
                            audio_decoder_headroom(self._media.audio, headroom)
                        )
                    media = self._media.generate_mp4(
                        directory / "latents.safetensors",
                        media_path,
                        height=request.height,
                        width=request.width,
                        duration_seconds=request.duration_seconds,
                        fps=24,
                        **media_options,
                    )
                if headroom:
                    stages["audio_decoder_headroom"] = headroom
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
            os.link(media_path, output_path)
        self.request_count += 1
        # A callback cannot turn a successfully published artifact into a reported
        # generation failure. The final result is the completion notification.
        return VideoResult(
            output_path,
            self.prepared.profile_id,
            request.seed,
            time.monotonic() - started,
            stages,
            media.media,
            request_mode=request.mode,
            latent_output=latent_output,
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
        scene = getattr(self, "_scene_conditioning", None)
        if scene is not None:
            scene.close()
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
