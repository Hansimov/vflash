"""Native video/audio projections and packed-row input assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vflash.native.h3_kernel_plan import H3KernelPlanError, resolve_h3_kernel_plan


class H3NativeInputStageError(ValueError):
    """The input-stage source or invocation differs from the H3 contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise RuntimeError("the H3 native input stage requires PyTorch") from exc
    return torch


def _expect_tensor(value: Any, shape: tuple[int, ...], *, name: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise H3NativeInputStageError(
            f"H3 input-stage tensor shape differs for {name}: {actual} != {shape}"
        )
    return value


@dataclass(frozen=True)
class H3NativeInputProjectionWeights:
    video_weight: Any
    video_bias: Any
    audio_weight: Any
    audio_bias: Any
    hidden_size: int
    video_input_size: int
    audio_input_size: int


def load_h3_native_input_projection_weights(
    *,
    base_load: Callable[[str], Any],
    device: Any,
    hidden_size: int = 5_376,
    video_input_size: int = 96,
    audio_input_size: int = 32,
) -> H3NativeInputProjectionWeights:
    """Load only the per-NFE modality projections for a prepared prompt.

    ConditioningBundle quality runs already carry an attested refined-text
    tensor.  They must not load the roughly 1.6 GB prompt-only TokenRefiner
    merely to obtain the four small video/audio projection tensors needed by
    the resident denoising loop.
    """

    torch = _torch()
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda":
        raise H3NativeInputStageError("the resident H3 input stage requires CUDA")
    if min(hidden_size, video_input_size, audio_input_size) <= 0:
        raise H3NativeInputStageError("H3 input-projection dimensions must be positive")

    def fp32(name: str, shape: tuple[int, ...]) -> Any:
        return (
            _expect_tensor(base_load(name), shape, name=name)
            .to(device=runtime_device, dtype=torch.float32)
            .contiguous()
        )

    return H3NativeInputProjectionWeights(
        video_weight=fp32("proj_in.weight", (hidden_size, video_input_size)),
        video_bias=fp32("proj_in.bias", (hidden_size,)),
        audio_weight=fp32("audio_proj_in.weight", (hidden_size, audio_input_size)),
        audio_bias=fp32("audio_proj_in.bias", (hidden_size,)),
        hidden_size=hidden_size,
        video_input_size=video_input_size,
        audio_input_size=audio_input_size,
    )


@dataclass(frozen=True)
class H3NativeInputPackInvocation:
    batch_size: int
    video_rows: int
    audio_rows: int
    text_rows: int
    sequence_length: int
    device: Any
    video_indices: Any
    audio_indices: Any
    text_indices: Any


class _H3InputPackingOperations:
    """Project changing video/audio latents and scatter three packed modalities."""

    backend_id = "pytorch-input-packer-correctness-only"
    timing_eligible = False

    def __init__(self, weights: H3NativeInputProjectionWeights) -> None:
        torch = _torch()
        device = weights.video_weight.device
        shapes = (
            ((weights.hidden_size, weights.video_input_size), weights.video_weight),
            ((weights.hidden_size,), weights.video_bias),
            ((weights.hidden_size, weights.audio_input_size), weights.audio_weight),
            ((weights.hidden_size,), weights.audio_bias),
        )
        if min(
            weights.hidden_size, weights.video_input_size, weights.audio_input_size
        ) <= 0 or any(tuple(value.shape) != shape for shape, value in shapes):
            raise H3NativeInputStageError("H3 modality input-projection shape is invalid")
        tensors = tuple(value for _shape, value in shapes)
        if any(value.device != device for value in tensors) or any(
            value.dtype != torch.float32 for value in tensors
        ):
            raise H3NativeInputStageError(
                "H3 modality input projections must be co-resident FP32"
            )
        self.weights = weights

    def prepare_invocation(
        self,
        *,
        video_latents: Any,
        audio_latents: Any,
        refined_text: Any,
        video_indices: Any,
        audio_indices: Any,
        text_indices: Any,
        sequence_length: int,
    ) -> H3NativeInputPackInvocation:
        torch = _torch()
        weights = self.weights
        device = weights.video_weight.device
        if sequence_length <= 0:
            raise H3NativeInputStageError("H3 packed sequence length must be positive")
        inputs = (
            (video_latents, weights.video_input_size, torch.float32, "video"),
            (audio_latents, weights.audio_input_size, torch.float32, "audio"),
            (refined_text, weights.hidden_size, torch.bfloat16, "text"),
        )
        batch_sizes = set()
        row_counts = []
        for value, width, dtype, name in inputs:
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or int(value.shape[-1]) != width
                or value.device != device
                or value.dtype != dtype
            ):
                raise H3NativeInputStageError(f"H3 packed {name} input is invalid")
            batch_sizes.add(int(value.shape[0]))
            row_counts.append(int(value.shape[1]))
        if len(batch_sizes) != 1:
            raise H3NativeInputStageError("H3 packed modality batch sizes differ")

        normalized_indices = []
        for indices, rows, name in zip(
            (video_indices, audio_indices, text_indices),
            row_counts,
            ("video", "audio", "text"),
            strict=True,
        ):
            if (
                not isinstance(indices, torch.Tensor)
                or indices.ndim != 1
                or int(indices.shape[0]) != rows
                or indices.dtype not in {torch.int32, torch.int64}
            ):
                raise H3NativeInputStageError(f"H3 packed {name} indices are invalid")
            normalized_indices.append(indices.to(device=device, dtype=torch.int64))
        combined = torch.cat(tuple(normalized_indices))
        if (
            int(combined.min()) < 0
            or int(combined.max()) >= sequence_length
            or combined.numel() != sequence_length
            or int(torch.unique(combined).numel()) != sequence_length
        ):
            raise H3NativeInputStageError("H3 packed indices do not form one complete sequence")
        return H3NativeInputPackInvocation(
            batch_size=batch_sizes.pop(),
            video_rows=row_counts[0],
            audio_rows=row_counts[1],
            text_rows=row_counts[2],
            sequence_length=sequence_length,
            device=device,
            video_indices=normalized_indices[0],
            audio_indices=normalized_indices[1],
            text_indices=normalized_indices[2],
        )

    def forward_prevalidated(
        self,
        *,
        video_latents: Any,
        audio_latents: Any,
        refined_text: Any,
        invocation: H3NativeInputPackInvocation,
    ) -> Any:
        torch = _torch()
        import torch.nn.functional as functional

        weights = self.weights
        expected = (
            (video_latents, invocation.video_rows, weights.video_input_size, torch.float32),
            (audio_latents, invocation.audio_rows, weights.audio_input_size, torch.float32),
            (refined_text, invocation.text_rows, weights.hidden_size, torch.bfloat16),
        )
        if any(
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (invocation.batch_size, rows, width)
            or value.device != invocation.device
            or value.dtype != dtype
            for value, rows, width, dtype in expected
        ):
            raise H3NativeInputStageError(
                "H3 prevalidated input-pack invocation does not match its tensors"
            )

        video_embeds = functional.linear(
            video_latents,
            weights.video_weight,
            weights.video_bias,
        )
        audio_embeds = functional.linear(
            audio_latents,
            weights.audio_weight,
            weights.audio_bias,
        )
        hidden_states = refined_text.new_zeros(
            (invocation.batch_size, invocation.sequence_length, weights.hidden_size)
        )
        hidden_states.index_copy_(1, invocation.text_indices, refined_text)
        hidden_states.index_copy_(
            1,
            invocation.video_indices,
            video_embeds.to(refined_text.dtype),
        )
        hidden_states.index_copy_(
            1,
            invocation.audio_indices,
            audio_embeds.to(refined_text.dtype),
        )
        return hidden_states

    def __call__(
        self,
        *,
        video_latents: Any,
        audio_latents: Any,
        refined_text: Any,
        video_indices: Any,
        audio_indices: Any,
        text_indices: Any,
        sequence_length: int,
    ) -> Any:
        invocation = self.prepare_invocation(
            video_latents=video_latents,
            audio_latents=audio_latents,
            refined_text=refined_text,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            sequence_length=sequence_length,
        )
        return self.forward_prevalidated(
            video_latents=video_latents,
            audio_latents=audio_latents,
            refined_text=refined_text,
            invocation=invocation,
        )


class H3NativeInputPacker(_H3InputPackingOperations):
    """Resident CUDA modality projections using the oracle's TF32 contract."""

    backend_id = "cuda-fp32-tf32-input-projections-bf16-packer-v1"
    timing_eligible = True

    def __init__(self, weights: H3NativeInputProjectionWeights) -> None:
        super().__init__(weights)
        torch = _torch()
        device = weights.video_weight.device
        if device.type != "cuda":
            raise H3NativeInputStageError("the resident H3 input packer requires CUDA")
        capability = torch.cuda.get_device_capability(device)
        try:
            self.kernel_plan = resolve_h3_kernel_plan(f"sm{capability[0]}{capability[1]}")
        except H3KernelPlanError as exc:
            raise H3NativeInputStageError(
                "the resident H3 input packer has no qualified GPU kernel plan"
            ) from exc
        if torch.get_float32_matmul_precision() != self.kernel_plan.float32_matmul_precision:
            raise H3NativeInputStageError(
                "the resident H3 input packer requires the oracle's high/TF32 precision"
            )
