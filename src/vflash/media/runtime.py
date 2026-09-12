"""An explicit CPU/CUDA lifetime for the official H3 decoder adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from traceback import clear_frames
from typing import Any

from vflash.adapters.official_vae import (
    decode_official_h3_audio_latents,
    decode_official_h3_video_latents,
    load_official_h3_vae_component,
    prepare_official_h3_audio_decoder,
    prepare_official_h3_video_decoder,
)
from vflash.media.encoding import MediaError, encode_mp4, media_executables
from vflash.native.h3_tensor_file import load_safetensor_tensors
from vflash.pipeline.residency import capture_cpu_master, restore_cpu_master


@dataclass(frozen=True)
class MediaResult:
    output_path: Path
    elapsed_seconds: float
    stage_durations: dict[str, float]
    peak_allocated_bytes: int
    media: dict[str, Any]


class OfficialMediaDecoder:
    """Keep validated CPU masters and upload VAE weights only for decoding.

    Calls are serial. ``close`` is idempotent after a successful CUDA fence;
    a failed fence keeps resources owned and the decoder unusable until close
    can be retried or its process exits. It never resets global CUDA state.
    """

    def __init__(
        self,
        *,
        video_component: Path,
        audio_component: Path,
        trust_local_code: bool = False,
        device: str = "cuda:0",
    ) -> None:
        media_executables()
        if trust_local_code is not True:
            raise MediaError("the official VAE adapter requires trust_local_code=True")
        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise MediaError("the official media decoder requires one CUDA device")
        self.video = self.audio = self._video_master = self._audio_master = None
        self._cuda_active = self._cuda_touched = self._closed = self._released = False
        started = time.monotonic()
        try:
            self._load_components(video_component, audio_component)
        except BaseException as exc:
            self.close()
            clear_frames(exc.__traceback__)
            raise
        self.initialization_seconds = time.monotonic() - started

    def _load_components(self, video_path: Path, audio_path: Path) -> None:
        torch = self._torch
        self.video = load_official_h3_vae_component(
            video_path,
            torch_module=torch,
            trust_local_code=True,
        )
        prepare_official_h3_video_decoder(
            self.video,
            device=torch.device("cpu"),
            torch_module=torch,
        )
        self._video_master = capture_cpu_master(self.video.module)
        self.audio = load_official_h3_vae_component(
            audio_path,
            torch_module=torch,
            trust_local_code=True,
        )
        prepare_official_h3_audio_decoder(
            self.audio,
            device=torch.device("cpu"),
            torch_module=torch,
        )
        self._audio_master = capture_cpu_master(self.audio.module)
        self.audio_sample_rate = int(self.audio.config["sample_rate"])
        if self.audio_sample_rate != 32000:
            raise MediaError("the H3 decoder requires its native 32000 Hz audio clock")

    def _require_open(self) -> None:
        if self._closed:
            raise MediaError("the media decoder is closed")

    def resume_cuda(self) -> float:
        self._require_open()
        if self._cuda_active:
            return 0.0
        started = time.monotonic()
        self._cuda_touched = True
        try:
            self.video.module.to(device=self.device)
            self.audio.module.to(device=self.device)
            self._torch.cuda.synchronize(self.device)
        except BaseException:
            self._closed = True
            raise
        self._cuda_active = True
        return time.monotonic() - started

    def suspend_cuda(self) -> float:
        self._require_open()
        if not self._cuda_touched:
            return 0.0
        started = time.monotonic()
        try:
            self._torch.cuda.synchronize(self.device)
            restore_cpu_master(self.video.module, self._video_master)
            restore_cpu_master(self.audio.module, self._audio_master)
        except BaseException:
            self._closed = True
            raise
        self._cuda_active = self._cuda_touched = False
        self._torch.cuda.empty_cache()
        return time.monotonic() - started

    def _decode_cpu(self, latent_path: Path, *, height: int, width: int) -> tuple[Any, ...]:
        torch = self._torch
        started = time.monotonic()
        tensors = load_safetensor_tensors(latent_path, ("video_latents", "audio_latents"))
        load_seconds = time.monotonic() - started
        if tensors["video_latents"].ndim != 5 or tensors["audio_latents"].ndim != 3:
            raise MediaError("native output is not VAE-ready")
        started = time.monotonic()
        video = decode_official_h3_video_latents(
            self.video,
            tensors["video_latents"],
            device=self.device,
            torch_module=torch,
        )
        if video.shape[-2] < height or video.shape[-1] < width:
            raise MediaError("decoded video is smaller than the requested canvas")
        video = video[..., :height, :width].contiguous().cpu()
        video_seconds = time.monotonic() - started
        started = time.monotonic()
        audio = decode_official_h3_audio_latents(
            self.audio,
            tensors["audio_latents"],
            device=self.device,
            torch_module=torch,
        ).cpu()
        return (
            video,
            audio,
            {
                "latent_load": load_seconds,
                "video_decode_and_copy": video_seconds,
                "audio_decode_and_copy": time.monotonic() - started,
            },
        )

    def generate_mp4(
        self,
        latent_path: Path,
        output_path: Path,
        *,
        height: int,
        width: int,
        duration_seconds: float,
        fps: int = 24,
    ) -> MediaResult:
        self._require_open()
        if not self._cuda_active:
            raise MediaError("resume the media decoder before generating video")
        if any(type(value) is not int or value <= 0 or value % 32 for value in (height, width)):
            raise MediaError("the H3 canvas must be positive and divisible by 32")
        if fps != 24 or type(fps) is not int:
            raise MediaError("this adapter preserves the native 24 fps model clock")
        if isinstance(duration_seconds, bool) or duration_seconds not in {5, 8, 10}:
            raise MediaError("this preview supports five-, eight-, and ten-second delivery")
        if output_path.exists() or output_path.is_symlink():
            raise MediaError("the output path already exists")
        started = time.monotonic()
        self._torch.cuda.reset_peak_memory_stats(self.device)
        try:
            video, audio, stages = self._decode_cpu(latent_path, height=height, width=width)
        except BaseException as exc:
            # Unwound decoder frames can own CUDA intermediates even when the
            # caller retains the exception. Clear them only after completion.
            try:
                self._torch.cuda.synchronize(self.device)
            except BaseException:
                self._closed = True
                exc.add_note("CUDA completion failed; retire the pipeline process.")
            else:
                clear_frames(exc.__traceback__)
            raise
        frames, samples = round(duration_seconds * fps), round(duration_seconds * 32000)
        if video.shape[2] < frames or audio.shape[2] < samples:
            raise MediaError("decoded media is shorter than the requested delivery")
        peak = int(self._torch.cuda.max_memory_allocated(self.device))
        encode_started = time.monotonic()
        media = encode_mp4(
            video[:, :, :frames],
            audio[:, :, :samples],
            output_path,
            fps=fps,
            audio_sample_rate=self.audio_sample_rate,
        )
        stages["media_encode"] = time.monotonic() - encode_started
        return MediaResult(output_path, time.monotonic() - started, stages, peak, media)

    def close(self) -> None:
        if self._released:
            return
        self._closed = True
        if self._cuda_touched:
            self._torch.cuda.synchronize(self.device)
        self.video = self.audio = self._video_master = self._audio_master = None
        self._cuda_active = self._cuda_touched = False
        self._released = True

    def __enter__(self) -> OfficialMediaDecoder:
        self._require_open()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()
