"""Native BF16 final normalization and FP32 video/audio output heads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vflash.native.h3_kernel_plan import H3KernelPlanError, resolve_h3_kernel_plan


class H3NativeFinalLayerError(ValueError):
    """The final-layer source or invocation differs from the H3 contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise RuntimeError("the H3 native final layer requires PyTorch") from exc
    return torch


def _expect_tensor(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise H3NativeFinalLayerError(
            f"H3 final-layer tensor shape differs for {name}: {actual} != {shape}"
        )
    return value


@dataclass(frozen=True)
class H3NativeFinalLayerWeights:
    """Resident exact-math weights for one fixed H3 denoising schedule."""

    adaln_table: Any
    norm_weight: Any
    video_head_bank: Any
    video_bias_bank: Any
    audio_head_bank: Any
    audio_bias_bank: Any
    video_head_plans: Any
    audio_head_plans: Any
    weight_profile: str

    @property
    def nfe(self) -> int:
        return int(self.adaln_table.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.norm_weight.shape[0])


def load_h3_native_final_layer_weights(
    *,
    base_load: Callable[[str], Any],
    time_embeddings: Any,
    device: Any,
    adaln_table_override: Any,
    hidden_size: int = 5_376,
    time_embedding_size: int = 2_688,
    video_output_size: int = 96,
    audio_output_size: int = 32,
) -> H3NativeFinalLayerWeights:
    """Load the fixed output heads and schedule-provided AdaLN table.

    The single-entry head representation preserves the validated FP32 output
    projection order used by existing artifacts.
    """

    torch = _torch()
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda":
        raise H3NativeFinalLayerError("the H3 native final layer requires CUDA")
    if (
        min(
            hidden_size,
            time_embedding_size,
            video_output_size,
            audio_output_size,
        )
        <= 0
    ):
        raise H3NativeFinalLayerError("H3 final-layer dimensions must be positive")
    if (
        not isinstance(time_embeddings, torch.Tensor)
        or time_embeddings.ndim != 3
        or int(time_embeddings.shape[-1]) != time_embedding_size
    ):
        raise H3NativeFinalLayerError("H3 final-layer time embeddings are invalid")
    nfe = int(time_embeddings.shape[0])
    if nfe <= 0:
        raise H3NativeFinalLayerError("H3 final-layer schedule must be non-empty")

    if (
        not isinstance(adaln_table_override, torch.Tensor)
        or tuple(adaln_table_override.shape)
        != (nfe, int(time_embeddings.shape[1]), 2, hidden_size)
        or adaln_table_override.dtype != torch.bfloat16
    ):
        raise H3NativeFinalLayerError("H3 final AdaLN override is invalid")
    adaln_table = adaln_table_override.to(runtime_device).contiguous()
    norm_weight = (
        _expect_tensor(
            base_load("norm_out.norm.weight"),
            (hidden_size,),
            name="norm_out.norm.weight",
        )
        .to(device=runtime_device, dtype=torch.bfloat16)
        .contiguous()
    )

    video_head_bank = (
        _expect_tensor(
            base_load("proj_out.weight"),
            (video_output_size, hidden_size),
            name="proj_out.weight",
        )
        .to(device=runtime_device, dtype=torch.float32)
        .unsqueeze(0)
        .contiguous()
    )
    video_bias_bank = (
        _expect_tensor(
            base_load("proj_out.bias"),
            (video_output_size,),
            name="proj_out.bias",
        )
        .to(device=runtime_device, dtype=torch.float32)
        .unsqueeze(0)
        .contiguous()
    )
    audio_head_bank = (
        _expect_tensor(
            base_load("audio_proj_out.weight"),
            (audio_output_size, hidden_size),
            name="audio_proj_out.weight",
        )
        .to(device=runtime_device, dtype=torch.float32)
        .unsqueeze(0)
        .contiguous()
    )
    audio_bias_bank = (
        _expect_tensor(
            base_load("audio_proj_out.bias"),
            (audio_output_size,),
            name="audio_proj_out.bias",
        )
        .to(device=runtime_device, dtype=torch.float32)
        .unsqueeze(0)
        .contiguous()
    )
    video_head_plans = torch.ones((nfe, 1, 1), device=runtime_device, dtype=torch.float32)
    audio_head_plans = video_head_plans
    weight_profile = "base"

    return H3NativeFinalLayerWeights(
        adaln_table=adaln_table.contiguous(),
        norm_weight=norm_weight,
        video_head_bank=video_head_bank,
        video_bias_bank=video_bias_bank,
        audio_head_bank=audio_head_bank,
        audio_bias_bank=audio_bias_bank,
        video_head_plans=video_head_plans,
        audio_head_plans=audio_head_plans,
        weight_profile=weight_profile,
    )


@dataclass(frozen=True)
class H3NativeFinalLayerInvocation:
    evaluation_index: int
    batch_size: int
    sequence_length: int
    hidden_size: int
    device: Any
    dtype: Any
    timestep_indices: Any
    video_indices: Any
    audio_indices: Any


class _H3FinalLayerOperations:
    """Shared final-layer validation and numerical operations."""

    backend_id = "pytorch-final-layer-correctness-only"
    timing_eligible = False

    def __init__(
        self,
        weights: H3NativeFinalLayerWeights,
        *,
        final_norm_eps: float = 1e-5,
    ) -> None:
        torch = _torch()
        if final_norm_eps <= 0:
            raise H3NativeFinalLayerError("H3 final norm epsilon must be positive")
        hidden_size = weights.hidden_size
        nfe = weights.nfe
        device = weights.norm_weight.device
        if weights.norm_weight.dtype != torch.bfloat16:
            raise H3NativeFinalLayerError("H3 final norm must be BF16")
        if tuple(weights.adaln_table.shape[:1]) != (nfe,) or tuple(
            weights.adaln_table.shape[2:]
        ) != (2, hidden_size):
            raise H3NativeFinalLayerError("H3 final AdaLN table shape is invalid")
        if weights.adaln_table.dtype != torch.bfloat16:
            raise H3NativeFinalLayerError("H3 final AdaLN table must be BF16")
        if weights.video_head_bank.ndim != 3 or weights.audio_head_bank.ndim != 3:
            raise H3NativeFinalLayerError("H3 final output-head bank is invalid")
        interval_count = int(weights.video_head_bank.shape[0])
        video_output_size = int(weights.video_head_bank.shape[1])
        audio_output_size = int(weights.audio_head_bank.shape[1])
        if (
            interval_count != 1
            or video_output_size <= 0
            or audio_output_size <= 0
            or tuple(weights.video_head_bank.shape)
            != (
                interval_count,
                video_output_size,
                hidden_size,
            )
            or tuple(weights.audio_head_bank.shape)
            != (
                interval_count,
                audio_output_size,
                hidden_size,
            )
            or tuple(weights.video_bias_bank.shape)
            != (
                interval_count,
                video_output_size,
            )
            or tuple(weights.audio_bias_bank.shape)
            != (
                interval_count,
                audio_output_size,
            )
        ):
            raise H3NativeFinalLayerError("H3 final output-head bank is invalid")
        if tuple(weights.video_head_plans.shape) != (nfe, 1, interval_count) or tuple(
            weights.audio_head_plans.shape
        ) != (nfe, 1, interval_count):
            raise H3NativeFinalLayerError("H3 final output-head plans are invalid")
        tensors = (
            weights.adaln_table,
            weights.video_head_bank,
            weights.video_bias_bank,
            weights.audio_head_bank,
            weights.audio_bias_bank,
            weights.video_head_plans,
            weights.audio_head_plans,
        )
        if any(value.device != device for value in tensors):
            raise H3NativeFinalLayerError("H3 final-layer tensors are not co-resident")
        if any(
            value.dtype != torch.float32
            for value in (
                weights.video_head_bank,
                weights.video_bias_bank,
                weights.audio_head_bank,
                weights.audio_bias_bank,
                weights.video_head_plans,
                weights.audio_head_plans,
            )
        ):
            raise H3NativeFinalLayerError("H3 final output heads must be FP32")
        self.weights = weights
        self.final_norm_eps = final_norm_eps

    def prepare_invocation(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        timestep_indices: Any,
        video_indices: Any,
        audio_indices: Any,
    ) -> H3NativeFinalLayerInvocation:
        torch = _torch()
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or int(hidden_states.shape[-1]) != self.weights.hidden_size
            or hidden_states.device != self.weights.norm_weight.device
            or hidden_states.dtype != torch.bfloat16
        ):
            raise H3NativeFinalLayerError("H3 final hidden_states contract is invalid")
        sequence_length = int(hidden_states.shape[1])
        if not 0 <= evaluation_index < self.weights.nfe:
            raise H3NativeFinalLayerError("H3 final evaluation index is outside the schedule")

        def normalize_indices(value: Any, *, name: str, expected_rows: int | None) -> Any:
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.dtype not in {torch.int32, torch.int64}
                or (expected_rows is not None and int(value.shape[0]) != expected_rows)
                or not value.numel()
            ):
                raise H3NativeFinalLayerError(f"H3 final {name} are invalid")
            normalized = value.to(device=hidden_states.device, dtype=torch.int64)
            if int(normalized.min()) < 0:
                raise H3NativeFinalLayerError(f"H3 final {name} contain a negative row")
            return normalized

        timesteps = normalize_indices(
            timestep_indices,
            name="timestep indices",
            expected_rows=sequence_length,
        )
        if int(timesteps.max()) >= int(self.weights.adaln_table.shape[1]):
            raise H3NativeFinalLayerError("H3 final timestep index is outside the AdaLN table")
        videos = normalize_indices(video_indices, name="video indices", expected_rows=None)
        audios = normalize_indices(audio_indices, name="audio indices", expected_rows=None)
        if int(videos.max()) >= sequence_length or int(audios.max()) >= sequence_length:
            raise H3NativeFinalLayerError(
                "H3 final modality index is outside the packed sequence"
            )
        return H3NativeFinalLayerInvocation(
            evaluation_index=evaluation_index,
            batch_size=int(hidden_states.shape[0]),
            sequence_length=sequence_length,
            hidden_size=self.weights.hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            timestep_indices=timesteps,
            video_indices=videos,
            audio_indices=audios,
        )

    @staticmethod
    def _blended_head(bank: Any, bias_bank: Any, plan: Any) -> tuple[Any, Any]:
        torch = _torch()
        weight = torch.einsum("bi,ioh->boh", plan, bank).flatten(0, 1)
        bias = torch.einsum("bi,io->bo", plan, bias_bank).flatten()
        return weight, bias

    def forward_prevalidated(
        self,
        hidden_states: Any,
        invocation: H3NativeFinalLayerInvocation,
    ) -> tuple[Any, Any]:
        torch = _torch()
        import torch.nn.functional as functional

        if (
            not isinstance(hidden_states, torch.Tensor)
            or tuple(hidden_states.shape)
            != (
                invocation.batch_size,
                invocation.sequence_length,
                invocation.hidden_size,
            )
            or hidden_states.device != invocation.device
            or hidden_states.dtype != invocation.dtype
            or hidden_states.device != self.weights.norm_weight.device
        ):
            raise H3NativeFinalLayerError(
                "H3 prevalidated final invocation does not match the activation"
            )
        modulation = self.weights.adaln_table[invocation.evaluation_index].index_select(
            0, invocation.timestep_indices
        )
        shift, scale = modulation.unbind(1)
        normalized = functional.rms_norm(
            hidden_states,
            (invocation.hidden_size,),
            self.weights.norm_weight,
            self.final_norm_eps,
        )
        projected_input = (normalized * (1.0 + scale) + shift).to(torch.float32)
        video_weight, video_bias = self._blended_head(
            self.weights.video_head_bank,
            self.weights.video_bias_bank,
            self.weights.video_head_plans[invocation.evaluation_index],
        )
        audio_weight, audio_bias = self._blended_head(
            self.weights.audio_head_bank,
            self.weights.audio_bias_bank,
            self.weights.audio_head_plans[invocation.evaluation_index],
        )
        video = functional.linear(projected_input, video_weight, video_bias).index_select(
            1, invocation.video_indices
        )
        audio = functional.linear(projected_input, audio_weight, audio_bias).index_select(
            1, invocation.audio_indices
        )
        return video, audio

    def __call__(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        timestep_indices: Any,
        video_indices: Any,
        audio_indices: Any,
    ) -> tuple[Any, Any]:
        invocation = self.prepare_invocation(
            hidden_states,
            evaluation_index=evaluation_index,
            timestep_indices=timestep_indices,
            video_indices=video_indices,
            audio_indices=audio_indices,
        )
        return self.forward_prevalidated(hidden_states, invocation)


class H3NativeFinalLayer(_H3FinalLayerOperations):
    """Resident CUDA implementation of the exact H3 final layer."""

    backend_id = "cuda-bf16-final-adaln-fp32-output-heads-v1"
    timing_eligible = True
    float32_matmul_precision = "high"

    def __init__(
        self,
        weights: H3NativeFinalLayerWeights,
        *,
        final_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__(weights, final_norm_eps=final_norm_eps)
        torch = _torch()
        if weights.norm_weight.device.type != "cuda":
            raise H3NativeFinalLayerError("the resident H3 final layer requires CUDA")
        capability = torch.cuda.get_device_capability(weights.norm_weight.device)
        try:
            plan = resolve_h3_kernel_plan(f"sm{capability[0]}{capability[1]}")
        except H3KernelPlanError as exc:
            raise H3NativeFinalLayerError(
                "the resident H3 final layer has no qualified GPU kernel plan"
            ) from exc
        if (
            plan.final_output_head_backend != "cublas-tf32-fp32-accumulate"
            or plan.float32_matmul_precision != self.float32_matmul_precision
            or torch.get_float32_matmul_precision() != plan.float32_matmul_precision
        ):
            raise H3NativeFinalLayerError(
                "the resident H3 final layer requires the oracle's high/TF32 "
                "float32 matmul precision"
            )
        self.kernel_plan = plan
