"""Continuous LightX2V-independent MiniMax-H3 denoising orchestration.

The expensive implementations remain behind the input-packer, denoiser and
final-layer interfaces.  This module owns the exact request-level ordering:
pack the current latent state, build the row-specific AdaLN indices, execute
the complete transformer stack, project video/audio velocities, and advance
the explicit dual-modality schedule.  Keeping that boundary small lets strict
and numerical kernel portfolios share one full-trajectory quality harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vflash.native.errors import VflashNativeError
from vflash.native.h3_native_scheduler import (
    H3NativeLatentState,
    H3NativeRowTimestepStep,
    H3NativeSchedule,
    compile_h3_row_timestep_plan,
)


class H3NativeEngineError(VflashNativeError):
    """The native H3 request graph violates its fixed execution contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3NativeEngineError("the H3 native engine requires PyTorch") from exc
    return torch


@dataclass(frozen=True)
class H3NativeEngineStepResult:
    """Optional validation payload from one completed denoiser evaluation."""

    evaluation_index: int
    packed_input: Any | None
    video_velocity: Any
    audio_velocity: Any


class H3NativeEngine:
    """Run a fixed H3 request from latent noise through every NFE.

    ``refined_text`` is already prompt-scoped and can outlive the released text
    conditioner.  Static indices and rotary tables are normalized once.  The
    latent buffers keep their addresses for the lifetime of the engine.
    """

    backend_id = "h3-native-continuous-denoising-v1"

    def __init__(
        self,
        *,
        schedule: H3NativeSchedule,
        latent_state: H3NativeLatentState,
        input_packer: Any,
        denoiser: Any,
        final_layer: Any,
        refined_text: Any,
        video_indices: Any,
        audio_indices: Any,
        text_indices: Any,
        token_tags: Any,
        rotary_cos: Any,
        rotary_sin: Any,
    ) -> None:
        torch = _torch()
        if not isinstance(schedule, H3NativeSchedule):
            raise H3NativeEngineError("the H3 native engine requires an explicit schedule")
        if not isinstance(latent_state, H3NativeLatentState):
            raise H3NativeEngineError("the H3 native engine requires native latent state")
        if latent_state.schedule != schedule:
            raise H3NativeEngineError("the H3 native engine schedule differs from latent state")
        device = latent_state.device
        if (
            not isinstance(refined_text, torch.Tensor)
            or refined_text.ndim != 3
            or int(refined_text.shape[0]) != 1
            or refined_text.dtype != torch.bfloat16
            or refined_text.device != device
        ):
            raise H3NativeEngineError("H3 refined text must be resident batch-one BF16")

        indices = []
        expected_rows = (
            int(latent_state.video_latents.shape[1]),
            int(latent_state.audio_latents.shape[1]),
            int(refined_text.shape[1]),
        )
        for value, rows, name in zip(
            (video_indices, audio_indices, text_indices),
            expected_rows,
            ("video", "audio", "text"),
            strict=True,
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or int(value.shape[0]) != rows
                or value.dtype not in {torch.int32, torch.int64}
            ):
                raise H3NativeEngineError(f"H3 native {name} indices are invalid")
            indices.append(value.to(device=device, dtype=torch.int64))
        sequence_length = sum(expected_rows)
        if (
            not isinstance(token_tags, torch.Tensor)
            or tuple(token_tags.shape) != (sequence_length,)
            or token_tags.dtype not in {torch.int32, torch.int64}
        ):
            raise H3NativeEngineError("H3 native token tags are invalid")
        normalized_tags = token_tags.to(device=device, dtype=torch.int64)
        if int(normalized_tags.min()) < 0 or int(normalized_tags.max()) > 2:
            raise H3NativeEngineError("H3 native token tags must be in [0,2]")
        if (
            not isinstance(rotary_cos, torch.Tensor)
            or not isinstance(rotary_sin, torch.Tensor)
            or rotary_cos.ndim != 2
            or tuple(rotary_cos.shape) != tuple(rotary_sin.shape)
            or int(rotary_cos.shape[0]) != sequence_length
        ):
            raise H3NativeEngineError("H3 native rotary tables are invalid")
        normalized_cos = rotary_cos.to(device=device)
        normalized_sin = rotary_sin.to(device=device)

        row_plan = compile_h3_row_timestep_plan(
            schedule,
            video_indices=indices[0],
            audio_indices=indices[1],
            text_indices=indices[2],
            num_condition_video_rows=latent_state.num_condition_video_rows,
            num_condition_audio_rows=latent_state.num_condition_audio_rows,
            device=device,
        )
        try:
            pack_invocation = input_packer.prepare_invocation(
                video_latents=latent_state.video_latents,
                audio_latents=latent_state.audio_latents,
                refined_text=refined_text,
                video_indices=indices[0],
                audio_indices=indices[1],
                text_indices=indices[2],
                sequence_length=sequence_length,
            )
        except (AttributeError, TypeError) as exc:
            raise H3NativeEngineError(
                "H3 input packer does not implement the native ABI"
            ) from exc
        final_nfe = getattr(getattr(final_layer, "weights", None), "nfe", schedule.nfe)
        if final_nfe != schedule.nfe:
            raise H3NativeEngineError("H3 final-layer NFE differs from the schedule")
        self.schedule = schedule
        self.latent_state = latent_state
        self.input_packer = input_packer
        self.denoiser = denoiser
        self.final_layer = final_layer
        self.refined_text = refined_text
        self.video_indices = indices[0]
        self.audio_indices = indices[1]
        self.text_indices = indices[2]
        self.token_tags = normalized_tags
        self.rotary_cos = normalized_cos
        self.rotary_sin = normalized_sin
        self.sequence_length = sequence_length
        self.row_timestep_plan: tuple[H3NativeRowTimestepStep, ...] = row_plan
        self.input_pack_invocation = pack_invocation

    @property
    def complete(self) -> bool:
        return self.latent_state.complete

    @property
    def evaluation_index(self) -> int:
        return self.latent_state.evaluation_index

    def _prepare_denoiser_invocation(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        adaln_indices: Any,
    ) -> Any:
        prepare = getattr(self.denoiser, "prepare_invocation", None)
        if callable(prepare):
            return prepare(
                hidden_states,
                evaluation_index=evaluation_index,
                adaln_indices=adaln_indices,
                rotary_cos=self.rotary_cos,
                rotary_sin=self.rotary_sin,
            )
        blocks = getattr(self.denoiser, "blocks", None)
        if not isinstance(blocks, tuple) or not blocks:
            raise H3NativeEngineError("H3 denoiser does not implement the native ABI")
        return blocks[0].prepare_invocation(
            hidden_states,
            evaluation_index=evaluation_index,
            adaln_indices=adaln_indices,
            rotary_cos=self.rotary_cos,
            rotary_sin=self.rotary_sin,
        )

    def _forward_denoiser(self, hidden_states: Any, invocation: Any) -> Any:
        forward = getattr(self.denoiser, "forward_prevalidated", None)
        if callable(forward):
            output = forward(hidden_states, invocation)
            if isinstance(output, tuple):
                if len(output) != 2 or not isinstance(output[1], dict):
                    raise H3NativeEngineError("H3 denoiser returned an invalid ring result")
                return output[0]
            return output
        blocks = getattr(self.denoiser, "blocks", None)
        if not isinstance(blocks, tuple) or not blocks:
            raise H3NativeEngineError("H3 denoiser does not implement the native ABI")
        for block in blocks:
            hidden_states = block.forward_prevalidated(hidden_states, invocation)
        return hidden_states

    def step(self, *, capture_packed_input: bool = False) -> H3NativeEngineStepResult:
        """Execute and commit exactly one evaluation in schedule order."""

        torch = _torch()
        if self.complete:
            raise H3NativeEngineError("the H3 native engine schedule is already complete")
        evaluation = self.evaluation_index
        row_step = self.row_timestep_plan[evaluation]
        adaln_indices = row_step.timestep_indices * 3 + self.token_tags
        with torch.inference_mode():
            packed = self.input_packer.forward_prevalidated(
                video_latents=self.latent_state.video_latents,
                audio_latents=self.latent_state.audio_latents,
                refined_text=self.refined_text,
                invocation=self.input_pack_invocation,
            )
            invocation = self._prepare_denoiser_invocation(
                packed,
                evaluation_index=evaluation,
                adaln_indices=adaln_indices,
            )
            final_invocation = self.final_layer.prepare_invocation(
                packed,
                evaluation_index=evaluation,
                timestep_indices=row_step.timestep_indices,
                video_indices=self.video_indices,
                audio_indices=self.audio_indices,
            )
            hidden_states = self._forward_denoiser(packed, invocation)
            video_velocity, audio_velocity = self.final_layer.forward_prevalidated(
                hidden_states,
                final_invocation,
            )
            self.latent_state.step(
                video_velocity=video_velocity,
                audio_velocity=audio_velocity,
                evaluation_index=evaluation,
            )
        return H3NativeEngineStepResult(
            evaluation_index=evaluation,
            packed_input=packed if capture_packed_input else None,
            video_velocity=video_velocity,
            audio_velocity=audio_velocity,
        )

    def run(
        self,
        *,
        capture_evaluations: frozenset[int] = frozenset(),
    ) -> tuple[H3NativeEngineStepResult, ...]:
        """Finish the remaining schedule and retain only requested probe rows."""

        if any(
            index < self.evaluation_index or index >= self.schedule.nfe
            for index in capture_evaluations
        ):
            raise H3NativeEngineError("H3 capture evaluation is outside the remaining schedule")
        captured: list[H3NativeEngineStepResult] = []
        while not self.complete:
            capture = self.evaluation_index in capture_evaluations
            result = self.step(capture_packed_input=capture)
            if capture:
                captured.append(result)
        return tuple(captured)
