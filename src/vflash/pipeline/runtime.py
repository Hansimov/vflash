"""One owned prompt/reference-to-MP4 pipeline, without product service dependencies."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from traceback import clear_frames
from typing import Any

from vflash.adapters.diffusers_h3 import DiffusersConditioner, validate_adapter_dependencies
from vflash.adapters.references import DecodedReference, read_reference
from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.media.encoding import media_executables
from vflash.media.runtime import OfficialMediaDecoder
from vflash.native.runner import NativeEngineSession
from vflash.pipeline.assets import PreparedPipelineAssets
from vflash.pipeline.contracts import (
    PIPELINE_PROFILE,
    PipelineProgress,
    VideoRequest,
    VideoResult,
)
from vflash.planner import resolve_plan


class H3Pipeline:
    """Serial, persistent Ref4 SM89 encoding, native inference, decoding and MP4.

    Construct before CUDA initialization in a dedicated process. Native weights
    use the tested block ring so the official encoding and VAE stages can take
    turns on the same device. The object owns its stages and temporary files;
    the caller owns scheduling, requests, final MP4s and any account/storage data.
    """

    def __init__(
        self,
        prepared: PreparedPipelineAssets,
        *,
        device: NvidiaDevice,
        trust_local_code: bool = False,
    ) -> None:
        if trust_local_code is not True:
            raise ContractError("the official decoder adapter requires trust_local_code=True")
        if not isinstance(prepared, PreparedPipelineAssets):
            raise ContractError("prepare the immutable pipeline assets before loading a model")
        prepared.check_unchanged()
        media_executables()
        validate_adapter_dependencies()
        plan = resolve_plan(
            ProfileCatalog.bundled(), profile_id=PIPELINE_PROFILE, device=device
        )
        self.prepared = prepared
        self._lock = threading.Lock()
        self._active_thread_id: int | None = None
        self._closed = self._released = False
        self._core = self._conditioner = self._media = None
        self.request_count = 0
        self.initialization_stages: dict[str, float] = {}
        started = time.monotonic()
        try:
            self._load_stages(plan)
        except BaseException as exc:
            try:
                self._close_owned()
            except BaseException:
                exc.add_note("Vflash cleanup could not complete; exit the pipeline process.")
            else:
                clear_frames(exc.__traceback__)
            raise
        self.initialization_seconds = time.monotonic() - started

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
            weight_residency="block-ring",
        )
        self.initialization_stages["native"] = time.monotonic() - started
        started = time.monotonic()
        self._conditioner = DiffusersConditioner(self.prepared)
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
    ) -> VideoResult:
        """Return a complete MP4; failed or concurrent requests never publish a partial file."""
        if self._closed:
            raise ContractError("the pipeline is closed")
        if not isinstance(request, VideoRequest) or not isinstance(output_path, Path):
            raise ContractError("generate requires a VideoRequest and pathlib.Path output")
        if output_path.exists() or output_path.is_symlink():
            raise ContractError("the output path already exists")
        if not self._lock.acquire(blocking=False):
            raise ContractError(
                "this pipeline is busy; schedule the next request after completion"
            )
        reference = None
        self._active_thread_id = threading.get_ident()
        try:
            if self._closed:
                raise ContractError("the pipeline is closed")
            # Input errors are checked before stage activation and do not retire
            # an otherwise healthy session. Read/hash exactly the decoded bytes.
            self.prepared.check_unchanged()
            reference = read_reference(request.reference)
            try:
                return self._generate_one(request, reference, output_path, progress=progress)
            except BaseException as exc:
                try:
                    self._close_owned()
                except BaseException:
                    exc.add_note("CUDA cleanup could not complete; exit the pipeline process.")
                else:
                    clear_frames(exc.__traceback__)
                raise
        finally:
            if reference is not None:
                reference.close()
            self._active_thread_id = None
            self._lock.release()

    def _generate_one(
        self,
        request: VideoRequest,
        reference: DecodedReference,
        output_path: Path,
        *,
        progress: Callable[[PipelineProgress], None] | None,
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
            report("encoding", 0, 1)
            stage_started = time.monotonic()
            self._conditioner.resume_cuda()
            bundle = self._conditioner.capture(request, reference, directory / "conditioning")
            self._conditioner.suspend_cuda()
            stages["encoding"] = {
                "elapsed_seconds": time.monotonic() - stage_started,
                "profile": asdict(bundle.profile),
                "source": dict(bundle.source),
                "bundle_id": bundle.bundle_id,
            }
            report("encoding", 1, 1)
            report("denoising", 0, 4)
            native = self._core.generate(
                bundle.directory,
                directory / "latents.safetensors",
                progress_callback=lambda completed, total: report(
                    "denoising", completed, total
                ),
            )
            stages["denoising"] = {
                key: value
                for key, value in native["generation"].items()
                if key != "output_path"
            }
            report("decoding", 0, 1)
            stage_started = time.monotonic()
            self._media.resume_cuda()
            media = self._media.generate_mp4(
                directory / "latents.safetensors",
                directory / "video.mp4",
                height=request.height,
                width=request.width,
                duration_seconds=5,
                fps=24,
            )
            self._media.suspend_cuda()
            stages["media"] = {
                "elapsed_seconds": time.monotonic() - stage_started,
                "stage_durations": media.stage_durations,
                "peak_allocated_bytes": media.peak_allocated_bytes,
            }
            report("decoding", 1, 1)
            os.link(directory / "video.mp4", output_path)
        self.request_count += 1
        # A callback cannot turn a successfully published MP4 into a reported
        # generation failure. The final result is the completion notification.
        return VideoResult(
            output_path,
            PIPELINE_PROFILE,
            request.seed,
            time.monotonic() - started,
            stages,
            media.media,
        )

    def _close_owned(self) -> None:
        if self._released:
            return
        self._closed = True
        for name in ("_conditioner", "_media", "_core"):
            owner = getattr(self, name)
            if owner is not None:
                owner.close()
                setattr(self, name, None)
        self._released = True

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
