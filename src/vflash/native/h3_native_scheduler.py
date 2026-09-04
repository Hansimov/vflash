"""Native MiniMax-H3 row-timestep and latent scheduler.

This module owns the small, exact state transition around the expensive H3
denoiser.  It intentionally does not depend on Diffusers or LightX2V.  The
schedule is compiled once from an explicit sigma grid, the packed-row timestep
tables are built once per request, and the latent buffers keep stable addresses
while only their generated suffixes are updated.

The arithmetic follows the official MiniMax-H3 ``training_euler`` order.  In
particular, the denoised estimate recovers sigma from the rounded timestep while
the Euler ratio reads the explicit sigma grid.  Collapsing those two sources
into ``sample + (sigma - sigma_next) * velocity`` changes float32 rounding.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from vflash.native.errors import VflashNativeError

H3_NATIVE_SCHEDULER_SCHEMA_VERSION = 1
H3_NATIVE_SCHEDULER_UPDATE_RULE = "training_euler"
H3_KEYFRAME_NOISE_AUG = 0.999


class H3NativeSchedulerError(VflashNativeError):
    """The native H3 schedule or latent state violates its fixed contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3NativeSchedulerError("the H3 native scheduler requires PyTorch") from exc
    return torch


def _validate_sigmas(value: Any, *, name: str, nfe: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise H3NativeSchedulerError(f"H3 {name} must contain at least two values")
    if nfe is not None and len(value) != nfe + 1:
        raise H3NativeSchedulerError(f"H3 {name} must contain NFE+1 values")
    values: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise H3NativeSchedulerError(f"H3 {name} contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise H3NativeSchedulerError(f"H3 {name} contains a value outside [0,1]")
        values.append(number)
    if values[0] != 1.0 or values[-1] != 0.0:
        raise H3NativeSchedulerError(f"H3 {name} must span exactly 1 to 0")
    if any(right >= left for left, right in pairwise(values)):
        raise H3NativeSchedulerError(f"H3 {name} must be strictly decreasing")
    return tuple(values)


@dataclass(frozen=True)
class H3NativeSchedule:
    """Explicit dual-modality schedule used by one fixed H3 request profile."""

    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    update_rule: str = H3_NATIVE_SCHEDULER_UPDATE_RULE
    keyframe_noise_aug: float = H3_KEYFRAME_NOISE_AUG

    def __post_init__(self) -> None:
        video = _validate_sigmas(self.video_sigmas, name="video_sigmas")
        audio = _validate_sigmas(
            self.audio_sigmas,
            name="audio_sigmas",
            nfe=len(video) - 1,
        )
        if self.update_rule != H3_NATIVE_SCHEDULER_UPDATE_RULE:
            raise H3NativeSchedulerError(
                f"unsupported H3 scheduler update rule: {self.update_rule!r}"
            )
        if (
            not isinstance(self.keyframe_noise_aug, (int, float))
            or isinstance(self.keyframe_noise_aug, bool)
            or not math.isfinite(float(self.keyframe_noise_aug))
            or not 0.0 <= float(self.keyframe_noise_aug) <= 1.0
        ):
            raise H3NativeSchedulerError("H3 keyframe noise augmentation is invalid")
        object.__setattr__(self, "video_sigmas", video)
        object.__setattr__(self, "audio_sigmas", audio)
        object.__setattr__(self, "keyframe_noise_aug", float(self.keyframe_noise_aug))

    @property
    def nfe(self) -> int:
        return len(self.video_sigmas) - 1

    @classmethod
    def shifted_linear(
        cls,
        nfe: int,
        *,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
    ) -> H3NativeSchedule:
        """Build the official Base H3 shifted-linear inference schedule.

        ``torch.linspace`` and float32 arithmetic are deliberate.  They match
        the released scheduler's rounding and therefore reproduce captured
        Base40/Base50 grids instead of merely approximating their formula with
        Python float64 values.
        """

        torch = _torch()
        if not isinstance(nfe, int) or isinstance(nfe, bool) or nfe <= 0:
            raise H3NativeSchedulerError("H3 shifted-linear NFE must be positive")
        shifts = (video_shift, audio_shift)
        if any(
            not isinstance(shift, (int, float))
            or isinstance(shift, bool)
            or not math.isfinite(float(shift))
            or float(shift) <= 0.0
            for shift in shifts
        ):
            raise H3NativeSchedulerError("H3 shifted-linear flow shifts must be positive")
        grid = torch.linspace(1.0, 0.0, nfe + 1, dtype=torch.float32)

        def shifted(shift: float) -> tuple[float, ...]:
            value = float(shift) * grid / (1.0 + (float(shift) - 1.0) * grid)
            return tuple(float(item) for item in value.tolist())

        return cls(
            video_sigmas=shifted(float(video_shift)),
            audio_sigmas=shifted(float(audio_shift)),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the stable schema-v1 representation used by artifacts."""

        return {
            "schema_version": H3_NATIVE_SCHEDULER_SCHEMA_VERSION,
            "nfe": self.nfe,
            "update_rule": self.update_rule,
            "video_sigmas": list(self.video_sigmas),
            "audio_sigmas": list(self.audio_sigmas),
        }

    @classmethod
    def from_mapping(cls, value: Any, *, expected_nfe: int | None = None) -> H3NativeSchedule:
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "nfe",
            "update_rule",
            "video_sigmas",
            "audio_sigmas",
        }:
            raise H3NativeSchedulerError("H3 scheduler fields do not match schema v1")
        if value.get("schema_version") != H3_NATIVE_SCHEDULER_SCHEMA_VERSION:
            raise H3NativeSchedulerError("H3 scheduler schema version is unsupported")
        nfe = value.get("nfe")
        if not isinstance(nfe, int) or isinstance(nfe, bool) or nfe <= 0:
            raise H3NativeSchedulerError("H3 scheduler NFE is invalid")
        if expected_nfe is not None and nfe != expected_nfe:
            raise H3NativeSchedulerError(
                f"H3 scheduler NFE differs from its request profile: {nfe} != {expected_nfe}"
            )
        video = _validate_sigmas(value.get("video_sigmas"), name="video_sigmas", nfe=nfe)
        audio = _validate_sigmas(value.get("audio_sigmas"), name="audio_sigmas", nfe=nfe)
        return cls(
            video_sigmas=video,
            audio_sigmas=audio,
            update_rule=value.get("update_rule"),
        )

    @classmethod
    def from_json(cls, path: Path, *, expected_nfe: int | None = None) -> H3NativeSchedule:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise H3NativeSchedulerError("H3 scheduler JSON is unavailable or invalid") from exc
        return cls.from_mapping(value, expected_nfe=expected_nfe)

    def tensors(self, *, device: Any) -> H3NativeScheduleTensors:
        torch = _torch()
        runtime_device = torch.device(device)
        video_sigmas = torch.tensor(
            self.video_sigmas,
            dtype=torch.float32,
            device=runtime_device,
        )
        audio_sigmas = torch.tensor(
            self.audio_sigmas,
            dtype=torch.float32,
            device=runtime_device,
        )
        return H3NativeScheduleTensors(
            video_sigmas=video_sigmas,
            audio_sigmas=audio_sigmas,
            video_timesteps=1.0 - video_sigmas[:-1],
            audio_timesteps=1.0 - audio_sigmas[:-1],
        )


@dataclass(frozen=True)
class H3NativeScheduleTensors:
    video_sigmas: Any
    audio_sigmas: Any
    video_timesteps: Any
    audio_timesteps: Any


@dataclass(frozen=True)
class H3NativeRowTimestepStep:
    timesteps: Any
    timestep_indices: Any


def _index_tensor_cpu(value: Any, *, name: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.int64:
        raise H3NativeSchedulerError(f"H3 {name} must be a one-dimensional int64 Tensor")
    return value.detach().to(device="cpu", copy=True).contiguous()


def compile_h3_row_timestep_plan(
    schedule: H3NativeSchedule,
    *,
    video_indices: Any,
    audio_indices: Any,
    text_indices: Any,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    device: Any,
) -> tuple[H3NativeRowTimestepStep, ...]:
    """Build the official sorted unique timestep table for every packed row."""

    torch = _torch()
    if not isinstance(schedule, H3NativeSchedule):
        raise H3NativeSchedulerError("H3 row-timestep compilation requires a native schedule")
    video = _index_tensor_cpu(video_indices, name="video_indices")
    audio = _index_tensor_cpu(audio_indices, name="audio_indices")
    text = _index_tensor_cpu(text_indices, name="text_indices")
    counts = (num_condition_video_rows, num_condition_audio_rows)
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts):
        raise H3NativeSchedulerError("H3 conditioning row counts are invalid")
    if num_condition_video_rows > video.numel() or num_condition_audio_rows > audio.numel():
        raise H3NativeSchedulerError("H3 conditioning row counts exceed their modality")

    sequence_length = int(video.numel() + audio.numel() + text.numel())
    if sequence_length <= 0:
        raise H3NativeSchedulerError("H3 packed sequence is empty")
    packed = torch.cat((video, audio, text))
    if (
        int(packed.min()) < 0
        or int(packed.max()) >= sequence_length
        or int(torch.unique(packed).numel()) != sequence_length
    ):
        raise H3NativeSchedulerError(
            "H3 modality indices must partition the complete packed sequence"
        )

    tensors = schedule.tensors(device="cpu")
    target_device = torch.device(device)
    rows: list[H3NativeRowTimestepStep] = []
    for evaluation in range(schedule.nfe):
        video_timestep = float(tensors.video_timesteps[evaluation])
        audio_timestep = float(tensors.audio_timesteps[evaluation])
        row_timesteps = torch.full(
            (sequence_length,),
            video_timestep,
            dtype=torch.float32,
        )
        row_timesteps[video[:num_condition_video_rows]] = max(
            video_timestep,
            schedule.keyframe_noise_aug,
        )
        row_timesteps[audio[num_condition_audio_rows:]] = audio_timestep
        row_timesteps[audio[:num_condition_audio_rows]] = 1.0
        unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        rows.append(
            H3NativeRowTimestepStep(
                timesteps=unique.to(device=target_device),
                timestep_indices=inverse.to(device=target_device),
            )
        )
    return tuple(rows)


def _validate_latents(value: Any, *, name: str) -> None:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[0] != 1
        or value.shape[1] <= 0
        or value.shape[2] <= 0
        or value.dtype != torch.float32
        or not value.is_contiguous()
    ):
        raise H3NativeSchedulerError(
            f"H3 {name} must be contiguous FP32 with shape [1,rows,channels]"
        )


def h3_training_euler_step(
    sample: Any,
    model_output: Any,
    *,
    timestep: Any,
    sigma: Any,
    sigma_next: Any,
) -> Any:
    """Apply the official data-ward Euler operation without scheduler state."""

    torch = _torch()
    if not isinstance(sample, torch.Tensor) or not isinstance(model_output, torch.Tensor):
        raise H3NativeSchedulerError("H3 scheduler samples and velocities must be Tensors")
    if sample.shape != model_output.shape or sample.device != model_output.device:
        raise H3NativeSchedulerError("H3 scheduler sample and velocity contracts differ")
    if sample.dtype != torch.float32 or model_output.dtype != torch.float32:
        raise H3NativeSchedulerError("H3 scheduler requires FP32 samples and velocities")

    timestep = timestep.to(device=sample.device, dtype=sample.dtype)
    sigma_from_timestep = 1 - timestep
    while sigma_from_timestep.ndim < sample.ndim:
        sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
    denoised = sample + sigma_from_timestep * model_output
    sigma = sigma.to(device=sample.device, dtype=sample.dtype)
    sigma_next = sigma_next.to(device=sample.device, dtype=sample.dtype)
    ratio = sigma_next / sigma
    return ratio * sample + (1.0 - ratio) * denoised


class H3NativeLatentState:
    """Stable-address FP32 latent owner for a sequential H3 denoising loop."""

    def __init__(
        self,
        schedule: H3NativeSchedule,
        *,
        video_latents: Any,
        audio_latents: Any,
        num_condition_video_rows: int,
        num_condition_audio_rows: int,
    ) -> None:
        torch = _torch()
        if not isinstance(schedule, H3NativeSchedule):
            raise H3NativeSchedulerError("H3 latent state requires a native schedule")
        _validate_latents(video_latents, name="video_latents")
        _validate_latents(audio_latents, name="audio_latents")
        if video_latents.device != audio_latents.device:
            raise H3NativeSchedulerError("H3 video/audio latents must share one device")
        counts = (num_condition_video_rows, num_condition_audio_rows)
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts
        ):
            raise H3NativeSchedulerError("H3 conditioning row counts are invalid")
        if (
            num_condition_video_rows > video_latents.shape[1]
            or num_condition_audio_rows > audio_latents.shape[1]
        ):
            raise H3NativeSchedulerError("H3 conditioning row counts exceed their modality")
        self.schedule = schedule
        self.video_latents = video_latents
        self.audio_latents = audio_latents
        self.num_condition_video_rows = num_condition_video_rows
        self.num_condition_audio_rows = num_condition_audio_rows
        self.evaluation_index = 0
        self._tensors = schedule.tensors(device=video_latents.device)
        if self._tensors.video_sigmas.dtype != torch.float32:
            raise H3NativeSchedulerError("H3 schedule tensors must be FP32")

    @property
    def device(self) -> Any:
        return self.video_latents.device

    @property
    def complete(self) -> bool:
        return self.evaluation_index == self.schedule.nfe

    def step(
        self,
        *,
        video_velocity: Any,
        audio_velocity: Any,
        evaluation_index: int | None = None,
    ) -> None:
        """Update generated suffixes in place and advance exactly one evaluation."""

        torch = _torch()
        _validate_latents(video_velocity, name="video_velocity")
        _validate_latents(audio_velocity, name="audio_velocity")
        if self.complete:
            raise H3NativeSchedulerError("H3 latent schedule is already complete")
        expected = self.evaluation_index
        if evaluation_index is not None and evaluation_index != expected:
            raise H3NativeSchedulerError(
                f"H3 scheduler evaluation is out of order: {evaluation_index} != {expected}"
            )
        if (
            video_velocity.shape != self.video_latents.shape
            or audio_velocity.shape != self.audio_latents.shape
            or video_velocity.device != self.device
            or audio_velocity.device != self.device
        ):
            raise H3NativeSchedulerError("H3 latent velocities do not match their state")

        with torch.no_grad():
            video_start = self.num_condition_video_rows
            audio_start = self.num_condition_audio_rows
            video_sample = self.video_latents[0, video_start:]
            audio_sample = self.audio_latents[0, audio_start:]
            next_video = h3_training_euler_step(
                video_sample,
                video_velocity[0, video_start:],
                timestep=self._tensors.video_timesteps[expected],
                sigma=self._tensors.video_sigmas[expected],
                sigma_next=self._tensors.video_sigmas[expected + 1],
            )
            next_audio = h3_training_euler_step(
                audio_sample,
                audio_velocity[0, audio_start:],
                timestep=self._tensors.audio_timesteps[expected],
                sigma=self._tensors.audio_sigmas[expected],
                sigma_next=self._tensors.audio_sigmas[expected + 1],
            )
            video_sample.copy_(next_video)
            audio_sample.copy_(next_audio)
        self.evaluation_index += 1
