"""Native BF16 H3 blocks with device residency or event-driven weight streaming."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.native.h3_distilled_lora import LIGHTX_H3_REF_TURBO8_CONTRACT
from vflash.native.h3_runtime_artifact import H3RuntimeArtifact, load_h3_runtime_artifact
from vflash.native.h3_tensor_file import H3MappedSafetensor, load_safetensor_tensor


class H3NativeDenoiserError(ValueError):
    """A native H3 block invocation differs from the compiled artifact contract."""


@dataclass(frozen=True)
class H3BF16Weight:
    """BF16 linear weight plus identity metadata from the artifact format."""

    values: Any
    scales: Any
    bits: int
    group_size: int | None
    input_features: int
    output_features: int


_ATTENTION_BACKENDS = frozenset({"torch-flash"})


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise RuntimeError("the H3 native denoiser requires PyTorch") from exc
    return torch


@dataclass(frozen=True)
class H3LowRankResidualWeights:
    down: Any
    up: Any
    scaling: float


@dataclass(frozen=True)
class H3NativeBlockWeights:
    adaln_table: Any
    qkv: H3BF16Weight
    attention_out: H3BF16Weight
    ffn_in: H3BF16Weight
    ffn_out: H3BF16Weight
    attention_norm: Any
    ffn_norm: Any
    query_norm: Any
    key_norm: Any
    qkv_residuals: tuple[H3LowRankResidualWeights, ...] = ()
    attention_out_residual: H3LowRankResidualWeights | None = None
    ffn_in_residual: H3LowRankResidualWeights | None = None
    ffn_out_residual: H3LowRankResidualWeights | None = None


@dataclass(frozen=True)
class H3NativeBlockInvocation:
    """One fail-closed, device-normalized invocation shared by every H3 block.

    Tensor content checks such as ``min``/``max`` synchronize a CUDA stream.
    They belong at the NFE boundary, not inside each of the 50 blocks.  The
    artifact id and fixed tensor contract keep the synchronization-free block
    entry point from becoming an unchecked public fast path.
    """

    artifact_id: str
    evaluation_index: int
    batch_size: int
    sequence_length: int
    hidden_size: int
    device: Any
    dtype: Any
    adaln_indices: Any
    rotary_cos: Any
    rotary_sin: Any


def _bf16_weight(
    values: Any,
    scales: Any,
    *,
    bits: int,
    group_size: int | None,
    input_features: int,
    output_features: int,
) -> H3BF16Weight:
    return H3BF16Weight(
        values=values,
        scales=scales,
        bits=bits,
        group_size=group_size,
        input_features=input_features,
        output_features=output_features,
    )


def load_h3_native_block(
    artifact_or_directory: H3RuntimeArtifact | Path,
    block_index: int,
    *,
    adaln_table: Any | None = None,
    tensor_load: Callable[[str], Any] | None = None,
) -> tuple[H3RuntimeArtifact, H3NativeBlockWeights]:
    """Load one validated block without constructing a Diffusers model graph."""

    artifact = (
        artifact_or_directory
        if isinstance(artifact_or_directory, H3RuntimeArtifact)
        else load_h3_runtime_artifact(artifact_or_directory)
    )
    try:
        block = next(row for row in artifact.blocks if row.index == block_index)
    except StopIteration as exc:
        raise H3NativeDenoiserError(f"H3 RuntimeArtifact has no block {block_index}") from exc
    path = artifact.directory / block.path
    load = tensor_load or (lambda name: load_safetensor_tensor(path, name))
    spec = artifact.spec
    inner = spec.num_attention_heads * spec.attention_head_dim
    target = artifact.target
    load_residual = None
    qkv_residuals: tuple[H3LowRankResidualWeights, ...] = ()
    if artifact.adapter_execution == "runtime-residual":
        scaling = LIGHTX_H3_REF_TURBO8_CONTRACT.scaling

        def _load_residual(stem: str) -> H3LowRankResidualWeights:
            return H3LowRankResidualWeights(
                down=load(f"{stem}.down"),
                up=load(f"{stem}.up"),
                scaling=scaling,
            )

        load_residual = _load_residual
        qkv_residuals = tuple(
            load_residual(f"adapter.attn.{suffix}") for suffix in ("q", "k", "v")
        )
    if adaln_table is None:
        resolved_adaln_table = load("adaln.table")
    else:
        torch = _torch()
        expected_shape = (
            int(adaln_table.shape[0]) if isinstance(adaln_table, torch.Tensor) else 0,
            block.adaln_rows,
            6,
            spec.hidden_size,
        )
        if (
            not isinstance(adaln_table, torch.Tensor)
            or adaln_table.device.type != "cpu"
            or adaln_table.dtype != torch.bfloat16
            or adaln_table.ndim != 4
            or int(adaln_table.shape[0]) <= 0
            or tuple(adaln_table.shape) != expected_shape
            or not adaln_table.is_contiguous()
        ):
            raise H3NativeDenoiserError(
                "H3 schedule-overlay AdaLN table differs from the block contract"
            )
        resolved_adaln_table = adaln_table
    return artifact, H3NativeBlockWeights(
        adaln_table=resolved_adaln_table,
        qkv=_bf16_weight(
            load("attn.qkv.weight"),
            load("attn.qkv.scale"),
            bits=target.attention_weight_bits,
            group_size=None,
            input_features=spec.hidden_size,
            output_features=3 * inner,
        ),
        attention_out=_bf16_weight(
            load("attn.out.weight"),
            load("attn.out.scale"),
            bits=target.attention_weight_bits,
            group_size=None,
            input_features=inner,
            output_features=spec.hidden_size,
        ),
        ffn_in=_bf16_weight(
            load("ffn.in.weight"),
            load("ffn.in.scale"),
            bits=target.ffn_weight_bits,
            group_size=target.ffn_group_size,
            input_features=spec.hidden_size,
            output_features=2 * spec.ffn_dim,
        ),
        ffn_out=_bf16_weight(
            load("ffn.out.weight"),
            load("ffn.out.scale"),
            bits=target.ffn_weight_bits,
            group_size=target.ffn_group_size,
            input_features=spec.ffn_dim,
            output_features=spec.hidden_size,
        ),
        attention_norm=load("norm.attn.weight"),
        ffn_norm=load("norm.ffn.weight"),
        query_norm=load("attn.q_norm.weight"),
        key_norm=load("attn.k_norm.weight"),
        qkv_residuals=qkv_residuals,
        attention_out_residual=(
            load_residual("adapter.attn.out") if load_residual is not None else None
        ),
        ffn_in_residual=(
            load_residual("adapter.ffn.in") if load_residual is not None else None
        ),
        ffn_out_residual=(
            load_residual("adapter.ffn.out") if load_residual is not None else None
        ),
    )


def _rms_norm(hidden_states: Any, weight: Any, *, eps: float) -> Any:
    import torch.nn.functional as functional

    # Match the official Diffusers MiniMax-H3 ``nn.RMSNorm`` path exactly.
    # A hand-written float32 variance decomposition is mathematically similar,
    # but it selects different CUDA reductions and accumulated a measurable
    # BF16 block error while also running slower on SM86/SM89.
    return functional.rms_norm(
        hidden_states,
        (int(hidden_states.shape[-1]),),
        weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
        eps,
    )


def _apply_rotary(hidden_states: Any, cos: Any, sin: Any) -> Any:
    rotary_dim = int(cos.shape[-1])
    if rotary_dim <= 0 or rotary_dim > int(hidden_states.shape[-1]) or rotary_dim % 2:
        raise H3NativeDenoiserError("H3 rotary dimension is invalid")
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]
    cos = cos.to(device=hidden_states.device, dtype=hidden_states.dtype)[None, :, None, :]
    sin = sin.to(device=hidden_states.device, dtype=hidden_states.dtype)[None, :, None, :]
    first, second = rotary.chunk(2, dim=-1)
    rotated = _torch().cat((-second, first), dim=-1)
    return _torch().cat((rotary * cos + rotated * sin, passthrough), dim=-1).contiguous()


class _H3BlockOperations:
    """Shared input validation and BF16 block arithmetic."""

    backend_id = "bf16-block-operations"
    timing_eligible = False

    def __init__(
        self,
        artifact: H3RuntimeArtifact,
        weights: H3NativeBlockWeights,
        *,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
    ) -> None:
        if norm_eps <= 0 or qk_norm_eps <= 0:
            raise H3NativeDenoiserError("H3 native norm epsilons must be positive")
        self.artifact = artifact
        self.weights = weights
        self.norm_eps = norm_eps
        self.qk_norm_eps = qk_norm_eps

    def _residual_linear(
        self,
        states: Any,
        residual: H3LowRankResidualWeights,
    ) -> Any:
        value = self._residual_linear_unscaled(states, residual)
        return value * residual.scaling

    def _residual_linear_unscaled(
        self,
        states: Any,
        residual: H3LowRankResidualWeights,
    ) -> Any:
        """Keep the two LoRA GEMMs separate from replaceable output fusion."""

        import torch.nn.functional as functional

        down = residual.down.to(device=states.device, dtype=states.dtype)
        up = residual.up.to(device=states.device, dtype=states.dtype)
        return functional.linear(functional.linear(states, down), up)

    def _qkv_linear(self, states: Any) -> Any:
        torch = _torch()
        base = self._linear(states, self.weights.qkv)
        residuals = self.weights.qkv_residuals
        if not residuals:
            return base
        if len(residuals) != 3:
            raise H3NativeDenoiserError("H3 QKV residual pack must contain three branches")
        chunks = base.chunk(3, dim=-1)
        return torch.cat(
            tuple(
                value + self._residual_linear(states, residual)
                for value, residual in zip(chunks, residuals, strict=True)
            ),
            dim=-1,
        )

    def _adapted_linear(
        self,
        states: Any,
        weight: H3BF16Weight,
        residual: H3LowRankResidualWeights | None,
    ) -> Any:
        base = self._linear(states, weight)
        return base if residual is None else base + self._residual_linear(states, residual)

    def _adapted_gate_residual(
        self,
        states: Any,
        weight: H3BF16Weight,
        adapter: H3LowRankResidualWeights | None,
        residual: Any,
        gate: Any,
    ) -> Any:
        """Apply the BF16 adapter residual before the gated block residual."""

        projected = self._adapted_linear(states, weight, adapter)
        return self._gate_residual(residual, gate, projected)

    def _modulate(self, normalized: Any, scale: Any, shift: Any) -> Any:
        return normalized * (1.0 + scale) + shift

    def _gate_residual(self, residual: Any, gate: Any, projected: Any) -> Any:
        return residual + gate * projected

    def _silu_mul(self, packed: Any) -> Any:
        import torch.nn.functional as functional

        value, gate = packed.chunk(2, dim=-1)
        return value * functional.silu(gate)

    def _rotary(self, hidden_states: Any, cos: Any, sin: Any) -> Any:
        return _apply_rotary(hidden_states, cos, sin)

    def _qk_norm_rotary(
        self,
        query: Any,
        key: Any,
        invocation: H3NativeBlockInvocation,
    ) -> tuple[Any, Any]:
        query = _rms_norm(query, self.weights.query_norm, eps=self.qk_norm_eps)
        key = _rms_norm(key, self.weights.key_norm, eps=self.qk_norm_eps)
        query = self._rotary(query, invocation.rotary_cos, invocation.rotary_sin)
        key = self._rotary(key, invocation.rotary_cos, invocation.rotary_sin)
        return query, key

    def __call__(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        adaln_indices: Any,
        rotary_cos: Any,
        rotary_sin: Any,
    ) -> Any:
        invocation = self.prepare_invocation(
            hidden_states,
            evaluation_index=evaluation_index,
            adaln_indices=adaln_indices,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        return self.forward_prevalidated(hidden_states, invocation)

    def prepare_invocation(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        adaln_indices: Any,
        rotary_cos: Any,
        rotary_sin: Any,
    ) -> H3NativeBlockInvocation:
        """Validate dynamic inputs once before entering a block stack hot path."""

        torch = _torch()
        spec = self.artifact.spec
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or int(hidden_states.shape[-1]) != spec.hidden_size
        ):
            raise H3NativeDenoiserError("H3 native hidden_states shape is invalid")
        sequence_length = int(hidden_states.shape[1])
        if (
            not isinstance(adaln_indices, torch.Tensor)
            or adaln_indices.ndim != 1
            or int(adaln_indices.shape[0]) != sequence_length
            or adaln_indices.dtype not in {torch.int32, torch.int64}
        ):
            raise H3NativeDenoiserError("H3 native AdaLN indices are invalid")
        table = self.weights.adaln_table
        if not 0 <= evaluation_index < int(table.shape[0]):
            raise H3NativeDenoiserError("H3 native evaluation index is outside the AdaLN table")
        if int(adaln_indices.min()) < 0 or int(adaln_indices.max()) >= int(table.shape[1]):
            raise H3NativeDenoiserError("H3 native AdaLN row index is outside the table")
        if (
            tuple(rotary_cos.shape) != tuple(rotary_sin.shape)
            or int(rotary_cos.shape[0]) != sequence_length
        ):
            raise H3NativeDenoiserError("H3 native rotary tables are invalid")

        device = hidden_states.device
        dtype = hidden_states.dtype
        return H3NativeBlockInvocation(
            artifact_id=self.artifact.artifact_id,
            evaluation_index=evaluation_index,
            batch_size=int(hidden_states.shape[0]),
            sequence_length=sequence_length,
            hidden_size=spec.hidden_size,
            device=device,
            dtype=dtype,
            adaln_indices=adaln_indices.to(device=device, dtype=torch.int64),
            rotary_cos=rotary_cos.to(device=device, dtype=dtype),
            rotary_sin=rotary_sin.to(device=device, dtype=dtype),
        )

    def forward_prevalidated(
        self,
        hidden_states: Any,
        invocation: H3NativeBlockInvocation,
    ) -> Any:
        """Execute one block after the NFE boundary validated its invocation."""

        torch = _torch()

        spec = self.artifact.spec
        if (
            invocation.artifact_id != self.artifact.artifact_id
            or not isinstance(hidden_states, torch.Tensor)
            or tuple(hidden_states.shape)
            != (
                invocation.batch_size,
                invocation.sequence_length,
                invocation.hidden_size,
            )
            or hidden_states.device != invocation.device
            or hidden_states.dtype != invocation.dtype
        ):
            raise H3NativeDenoiserError(
                "H3 prevalidated invocation does not match this block activation"
            )
        table = self.weights.adaln_table
        modulation = (
            table[invocation.evaluation_index]
            .to(
                device=invocation.device,
                dtype=invocation.dtype,
            )
            .index_select(0, invocation.adaln_indices)
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.unbind(1)

        residual = hidden_states
        normalized = _rms_norm(
            hidden_states,
            self.weights.attention_norm,
            eps=self.norm_eps,
        )
        normalized = self._modulate(normalized, scale_msa, shift_msa)
        query, key, value = self._qkv_linear(normalized).chunk(3, dim=-1)
        heads = spec.num_attention_heads
        head_dim = spec.attention_head_dim
        query = query.unflatten(-1, (heads, head_dim))
        key = key.unflatten(-1, (heads, head_dim))
        value = value.unflatten(-1, (heads, head_dim))
        query, key = self._qk_norm_rotary(query, key, invocation)
        attention = self._attention(query, key, value)
        hidden_states = self._adapted_gate_residual(
            attention.flatten(2, 3),
            self.weights.attention_out,
            self.weights.attention_out_residual,
            residual,
            gate_msa,
        )

        residual = hidden_states
        normalized = _rms_norm(hidden_states, self.weights.ffn_norm, eps=self.norm_eps)
        normalized = self._modulate(normalized, scale_mlp, shift_mlp)
        ffn = self._silu_mul(
            self._adapted_linear(
                normalized,
                self.weights.ffn_in,
                self.weights.ffn_in_residual,
            )
        )
        return self._adapted_gate_residual(
            ffn,
            self.weights.ffn_out,
            self.weights.ffn_out_residual,
            residual,
            gate_mlp,
        )


def _resident_bf16_weight(weight: H3BF16Weight, device: Any) -> H3BF16Weight:
    """Move one exact BF16 matrix to its final resident device."""

    torch = _torch()
    if weight.bits != 16 or weight.group_size is not None:
        raise H3NativeDenoiserError("the BF16 backend accepts only unquantized weights")
    return H3BF16Weight(
        values=weight.values.to(device=device, dtype=torch.bfloat16).contiguous(),
        # Identity scales are an artifact ABI field and are not read by BF16 GEMM.
        scales=weight.scales,
        bits=16,
        group_size=None,
        input_features=weight.input_features,
        output_features=weight.output_features,
    )


def _resident_low_rank(
    residual: H3LowRankResidualWeights | None,
    device: Any,
) -> H3LowRankResidualWeights | None:
    if residual is None:
        return None
    torch = _torch()
    return H3LowRankResidualWeights(
        down=residual.down.to(device=device, dtype=torch.bfloat16).contiguous(),
        up=residual.up.to(device=device, dtype=torch.bfloat16).contiguous(),
        scaling=residual.scaling,
    )


def _pin_tensor(tensor: Any) -> Any:
    """Copy one CPU tensor into page-locked memory for asynchronous H2D."""

    torch = _torch()
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise H3NativeDenoiserError("the BF16 block ring requires CPU source tensors")
    return tensor if tensor.is_pinned() else tensor.pin_memory()


def _pin_bf16_weight(weight: H3BF16Weight) -> H3BF16Weight:
    if weight.bits != 16 or weight.group_size is not None:
        raise H3NativeDenoiserError("the BF16 block ring accepts only unquantized weights")
    return H3BF16Weight(
        values=_pin_tensor(weight.values),
        # Keep the tiny identity metadata pageable, but detach it from the
        # mapped file whose lifetime ends after the pinned copy is complete.
        scales=weight.scales.clone(),
        bits=16,
        group_size=None,
        input_features=weight.input_features,
        output_features=weight.output_features,
    )


def _pin_low_rank(
    residual: H3LowRankResidualWeights | None,
) -> H3LowRankResidualWeights | None:
    if residual is None:
        return None
    return H3LowRankResidualWeights(
        down=_pin_tensor(residual.down),
        up=_pin_tensor(residual.up),
        scaling=residual.scaling,
    )


def _pin_bf16_block(weights: H3NativeBlockWeights) -> H3NativeBlockWeights:
    """Create the immutable host half of the SM86 two-slot weight ring."""

    return H3NativeBlockWeights(
        adaln_table=_pin_tensor(weights.adaln_table),
        qkv=_pin_bf16_weight(weights.qkv),
        attention_out=_pin_bf16_weight(weights.attention_out),
        ffn_in=_pin_bf16_weight(weights.ffn_in),
        ffn_out=_pin_bf16_weight(weights.ffn_out),
        attention_norm=_pin_tensor(weights.attention_norm),
        ffn_norm=_pin_tensor(weights.ffn_norm),
        query_norm=_pin_tensor(weights.query_norm),
        key_norm=_pin_tensor(weights.key_norm),
        qkv_residuals=tuple(_pin_low_rank(row) for row in weights.qkv_residuals),
        attention_out_residual=_pin_low_rank(weights.attention_out_residual),
        ffn_in_residual=_pin_low_rank(weights.ffn_in_residual),
        ffn_out_residual=_pin_low_rank(weights.ffn_out_residual),
    )


def _empty_bf16_weight_like(weight: H3BF16Weight, device: Any) -> H3BF16Weight:
    torch = _torch()
    return H3BF16Weight(
        values=torch.empty_like(weight.values, device=device),
        scales=weight.scales,
        bits=16,
        group_size=None,
        input_features=weight.input_features,
        output_features=weight.output_features,
    )


def _empty_low_rank_like(
    residual: H3LowRankResidualWeights | None,
    device: Any,
) -> H3LowRankResidualWeights | None:
    if residual is None:
        return None
    torch = _torch()
    return H3LowRankResidualWeights(
        down=torch.empty_like(residual.down, device=device),
        up=torch.empty_like(residual.up, device=device),
        scaling=residual.scaling,
    )


def _empty_bf16_block_like(
    weights: H3NativeBlockWeights,
    device: Any,
) -> H3NativeBlockWeights:
    torch = _torch()
    return H3NativeBlockWeights(
        adaln_table=torch.empty_like(weights.adaln_table, device=device),
        qkv=_empty_bf16_weight_like(weights.qkv, device),
        attention_out=_empty_bf16_weight_like(weights.attention_out, device),
        ffn_in=_empty_bf16_weight_like(weights.ffn_in, device),
        ffn_out=_empty_bf16_weight_like(weights.ffn_out, device),
        attention_norm=torch.empty_like(weights.attention_norm, device=device),
        ffn_norm=torch.empty_like(weights.ffn_norm, device=device),
        query_norm=torch.empty_like(weights.query_norm, device=device),
        key_norm=torch.empty_like(weights.key_norm, device=device),
        qkv_residuals=tuple(_empty_low_rank_like(row, device) for row in weights.qkv_residuals),
        attention_out_residual=_empty_low_rank_like(weights.attention_out_residual, device),
        ffn_in_residual=_empty_low_rank_like(weights.ffn_in_residual, device),
        ffn_out_residual=_empty_low_rank_like(weights.ffn_out_residual, device),
    )


def _copy_bf16_block_(
    destination: H3NativeBlockWeights,
    source: H3NativeBlockWeights,
) -> None:
    """Queue one fixed-shape pinned-host block into an existing CUDA slot."""

    pairs = [
        (destination.adaln_table, source.adaln_table),
        (destination.qkv.values, source.qkv.values),
        (destination.attention_out.values, source.attention_out.values),
        (destination.ffn_in.values, source.ffn_in.values),
        (destination.ffn_out.values, source.ffn_out.values),
        (destination.attention_norm, source.attention_norm),
        (destination.ffn_norm, source.ffn_norm),
        (destination.query_norm, source.query_norm),
        (destination.key_norm, source.key_norm),
    ]
    destination_residuals = (
        *destination.qkv_residuals,
        destination.attention_out_residual,
        destination.ffn_in_residual,
        destination.ffn_out_residual,
    )
    source_residuals = (
        *source.qkv_residuals,
        source.attention_out_residual,
        source.ffn_in_residual,
        source.ffn_out_residual,
    )
    if len(destination_residuals) != len(source_residuals):
        raise H3NativeDenoiserError("BF16 ring residual packs have inconsistent layouts")
    for target, value in zip(destination_residuals, source_residuals, strict=True):
        if (target is None) != (value is None):
            raise H3NativeDenoiserError("BF16 ring residual packs have inconsistent layouts")
        if target is not None and value is not None:
            if target.scaling != value.scaling:
                raise H3NativeDenoiserError("BF16 ring residual scaling differs")
            pairs.extend(((target.down, value.down), (target.up, value.up)))
    for target, value in pairs:
        if tuple(target.shape) != tuple(value.shape) or target.dtype != value.dtype:
            raise H3NativeDenoiserError("BF16 ring block tensors have inconsistent layouts")
        target.copy_(value, non_blocking=True)


def _block_tensor_bytes(weights: H3NativeBlockWeights) -> int:
    tensors = [
        weights.adaln_table,
        weights.qkv.values,
        weights.attention_out.values,
        weights.ffn_in.values,
        weights.ffn_out.values,
        weights.attention_norm,
        weights.ffn_norm,
        weights.query_norm,
        weights.key_norm,
    ]
    for residual in (
        *weights.qkv_residuals,
        weights.attention_out_residual,
        weights.ffn_in_residual,
        weights.ffn_out_residual,
    ):
        if residual is not None:
            tensors.extend((residual.down, residual.up))
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


class H3NativeBlockBF16Resident(_H3BlockOperations):
    """BF16 block with fixed Torch Flash attention and strict fusion kernels."""

    backend_id = "cuda-bf16-resident-exact-sdpa-block-v1"
    timing_eligible = True

    def __init__(
        self,
        artifact: H3RuntimeArtifact,
        weights: H3NativeBlockWeights,
        *,
        device: Any,
        attention_backend: str = "torch-flash",
        elementwise_backend: str = "auto",
        elementwise_block_size: int = 0,
        adapter_fusion_backend: str = "auto",
        rotary_backend: str = "auto",
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
    ) -> None:
        torch = _torch()
        runtime_device = torch.device(device)
        if attention_backend not in _ATTENTION_BACKENDS:
            raise H3NativeDenoiserError(f"unsupported attention backend: {attention_backend}")
        if any(
            backend not in {"auto", "torch-eager", "triton-strict"}
            for backend in (elementwise_backend, adapter_fusion_backend, rotary_backend)
        ):
            raise H3NativeDenoiserError("unsupported BF16 fusion backend")
        if elementwise_block_size not in {0, 128, 256, 512, 1024}:
            raise H3NativeDenoiserError("unsupported fused H3 block size")
        if runtime_device.type != "cuda":
            raise H3NativeDenoiserError("the resident BF16 backend requires a CUDA device")
        if artifact.target.attention_weight_bits != 16 or artifact.target.ffn_weight_bits != 16:
            raise H3NativeDenoiserError("the resident BF16 backend requires BF16 weights")
        capability = torch.cuda.get_device_capability(runtime_device)
        actual_target = f"sm{capability[0]}{capability[1]}"
        if actual_target != artifact.target.compute_capability:
            raise H3NativeDenoiserError("the BF16 device differs from the artifact target")
        if (
            elementwise_backend == "auto"
            or elementwise_block_size == 0
            or adapter_fusion_backend == "auto"
            or rotary_backend == "auto"
        ):
            from vflash.native.h3_kernel_plan import resolve_h3_kernel_plan

            kernel_plan = resolve_h3_kernel_plan(actual_target)
            if elementwise_backend == "auto":
                elementwise_backend = kernel_plan.elementwise_backend
            if elementwise_block_size == 0:
                elementwise_block_size = kernel_plan.elementwise_block_size
            if adapter_fusion_backend == "auto":
                adapter_fusion_backend = kernel_plan.adapter_fusion_backend
            if rotary_backend == "auto":
                rotary_backend = kernel_plan.rotary_backend
        resident = H3NativeBlockWeights(
            adaln_table=weights.adaln_table.to(runtime_device),
            qkv=_resident_bf16_weight(weights.qkv, runtime_device),
            attention_out=_resident_bf16_weight(weights.attention_out, runtime_device),
            ffn_in=_resident_bf16_weight(weights.ffn_in, runtime_device),
            ffn_out=_resident_bf16_weight(weights.ffn_out, runtime_device),
            attention_norm=weights.attention_norm.to(runtime_device),
            ffn_norm=weights.ffn_norm.to(runtime_device),
            query_norm=weights.query_norm.to(runtime_device),
            key_norm=weights.key_norm.to(runtime_device),
            qkv_residuals=tuple(
                _resident_low_rank(row, runtime_device) for row in weights.qkv_residuals
            ),
            attention_out_residual=_resident_low_rank(
                weights.attention_out_residual, runtime_device
            ),
            ffn_in_residual=_resident_low_rank(weights.ffn_in_residual, runtime_device),
            ffn_out_residual=_resident_low_rank(weights.ffn_out_residual, runtime_device),
        )
        super().__init__(artifact, resident, norm_eps=norm_eps, qk_norm_eps=qk_norm_eps)
        self.device = runtime_device
        self.attention_backend = attention_backend
        self.elementwise_backend = elementwise_backend
        self.elementwise_block_size = elementwise_block_size
        self.adapter_fusion_backend = adapter_fusion_backend
        self.rotary_backend = rotary_backend

    @classmethod
    def load(
        cls,
        artifact_or_directory: H3RuntimeArtifact | Path,
        block_index: int,
        *,
        device: Any = "cuda:0",
        adaln_table: Any | None = None,
        attention_backend: str = "torch-flash",
        elementwise_backend: str = "auto",
        elementwise_block_size: int = 0,
        adapter_fusion_backend: str = "auto",
        rotary_backend: str = "auto",
    ) -> H3NativeBlockBF16Resident:
        artifact = (
            artifact_or_directory
            if isinstance(artifact_or_directory, H3RuntimeArtifact)
            else load_h3_runtime_artifact(artifact_or_directory)
        )
        try:
            row = next(item for item in artifact.blocks if item.index == block_index)
        except StopIteration as exc:
            raise H3NativeDenoiserError(
                f"H3 RuntimeArtifact has no block {block_index}"
            ) from exc
        with H3MappedSafetensor(artifact.directory / row.path) as mapped:
            loaded_artifact, weights = load_h3_native_block(
                artifact,
                block_index,
                adaln_table=adaln_table,
                tensor_load=mapped.load,
            )
            resident = cls(
                loaded_artifact,
                weights,
                device=device,
                attention_backend=attention_backend,
                elementwise_backend=elementwise_backend,
                elementwise_block_size=elementwise_block_size,
                adapter_fusion_backend=adapter_fusion_backend,
                rotary_backend=rotary_backend,
            )
            del weights
        return resident

    def _linear(self, states: Any, weight: H3BF16Weight) -> Any:
        torch = _torch()
        import torch.nn.functional as functional

        if states.device != self.device or states.dtype != torch.bfloat16:
            raise H3NativeDenoiserError("the BF16 backend requires resident BF16 activations")
        return functional.linear(states, weight.values)

    def _silu_mul(self, packed: Any) -> Any:
        if self.elementwise_backend == "torch-eager":
            return super()._silu_mul(packed)
        from vflash.native.h3_fused_ops import triton_strict_bf16_silu_mul

        return triton_strict_bf16_silu_mul(
            packed,
            block_size=self.elementwise_block_size,
        )

    def _qkv_linear(self, states: Any) -> Any:
        if self.adapter_fusion_backend == "torch-eager" or not self.weights.qkv_residuals:
            return super()._qkv_linear(states)
        residuals = self.weights.qkv_residuals
        if len(residuals) != 3:
            raise H3NativeDenoiserError("H3 QKV residual pack must contain three branches")
        torch = _torch()
        from vflash.native.h3_fused_ops import triton_strict_bf16_qkv_adapter_merge

        base = self._linear(states, self.weights.qkv)
        packed_adapter = torch.cat(
            tuple(self._residual_linear_unscaled(states, row) for row in residuals),
            dim=-1,
        )
        return triton_strict_bf16_qkv_adapter_merge(
            base,
            packed_adapter,
            scalings=tuple(row.scaling for row in residuals),
            block_size=self.elementwise_block_size,
        )

    def _adapted_linear(
        self,
        states: Any,
        weight: H3BF16Weight,
        residual: H3LowRankResidualWeights | None,
    ) -> Any:
        if self.adapter_fusion_backend == "torch-eager" or residual is None:
            return super()._adapted_linear(states, weight, residual)
        from vflash.native.h3_fused_ops import triton_strict_bf16_adapter_merge

        return triton_strict_bf16_adapter_merge(
            self._linear(states, weight),
            self._residual_linear_unscaled(states, residual),
            scaling=residual.scaling,
            block_size=self.elementwise_block_size,
        )

    def _attention(self, query: Any, key: Any, value: Any) -> Any:
        import torch.nn.functional as functional
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION], set_priority=True):
            return functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

    def _modulate(self, normalized: Any, scale: Any, shift: Any) -> Any:
        if self.elementwise_backend == "torch-eager":
            return super()._modulate(normalized, scale, shift)
        from vflash.native.h3_fused_ops import triton_strict_bf16_modulate

        return triton_strict_bf16_modulate(
            normalized,
            scale,
            shift,
            block_size=self.elementwise_block_size,
        )

    def _gate_residual(self, residual: Any, gate: Any, projected: Any) -> Any:
        if self.elementwise_backend == "torch-eager":
            return super()._gate_residual(residual, gate, projected)
        from vflash.native.h3_fused_ops import triton_strict_bf16_gate_residual

        return triton_strict_bf16_gate_residual(
            residual,
            gate,
            projected,
            block_size=self.elementwise_block_size,
        )

    def _rotary(self, hidden_states: Any, cos: Any, sin: Any) -> Any:
        if self.rotary_backend == "torch-eager":
            return super()._rotary(hidden_states, cos, sin)
        from vflash.native.h3_fused_ops import triton_strict_bf16_rotary

        return triton_strict_bf16_rotary(
            hidden_states,
            cos,
            sin,
            block_size=self.elementwise_block_size,
        )

    def _adapted_gate_residual(
        self,
        states: Any,
        weight: H3BF16Weight,
        adapter: H3LowRankResidualWeights | None,
        residual: Any,
        gate: Any,
    ) -> Any:
        if self.adapter_fusion_backend != "torch-eager" and adapter is not None:
            from vflash.native.h3_fused_ops import triton_strict_bf16_adapter_gate_residual

            return triton_strict_bf16_adapter_gate_residual(
                self._linear(states, weight),
                self._residual_linear_unscaled(states, adapter),
                residual,
                gate,
                scaling=adapter.scaling,
                block_size=self.elementwise_block_size,
            )
        return super()._adapted_gate_residual(states, weight, adapter, residual, gate)


class _H3BlockStack:
    """Execute a complete compiled 50-block stack on one packed activation."""

    backend_id = "bf16-block-stack"
    timing_eligible = False

    def __init__(self, blocks: tuple[_H3BlockOperations, ...]) -> None:
        if not blocks:
            raise H3NativeDenoiserError("H3 native denoiser requires compiled blocks")
        artifact = blocks[0].artifact
        if any(block.artifact != artifact for block in blocks):
            raise H3NativeDenoiserError(
                "H3 native denoiser blocks come from different artifacts"
            )
        if len(blocks) != artifact.spec.num_layers:
            raise H3NativeDenoiserError(
                "H3 native denoiser requires the complete transformer block stack"
            )
        self.artifact = artifact
        self.blocks = blocks

    def __call__(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        adaln_indices: Any,
        rotary_cos: Any,
        rotary_sin: Any,
    ) -> Any:
        invocation = self.blocks[0].prepare_invocation(
            hidden_states,
            evaluation_index=evaluation_index,
            adaln_indices=adaln_indices,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        for block in self.blocks:
            hidden_states = block.forward_prevalidated(hidden_states, invocation)
        return hidden_states


class H3NativeDenoiserBF16Resident(_H3BlockStack):
    """Resident 50-block same-weight stack for a 48 GiB SM89 target."""

    backend_id = "cuda-bf16-resident-exact-sdpa-50-block-v1"
    timing_eligible = True

    def __init__(self, blocks: tuple[H3NativeBlockBF16Resident, ...]) -> None:
        super().__init__(blocks)
        for name in (
            "attention_backend",
            "elementwise_backend",
            "elementwise_block_size",
            "adapter_fusion_backend",
            "rotary_backend",
        ):
            values = {getattr(block, name) for block in blocks}
            if len(values) != 1:
                raise H3NativeDenoiserError(f"the BF16 stack contains mixed {name}")
            setattr(self, name, values.pop())

    @classmethod
    def load(
        cls,
        artifact_or_directory: H3RuntimeArtifact | Path,
        *,
        device: Any = "cuda:0",
        adaln_table_load: Callable[[int], Any] | None = None,
        attention_backend: str = "torch-flash",
        elementwise_backend: str = "auto",
        elementwise_block_size: int = 0,
        adapter_fusion_backend: str = "auto",
        rotary_backend: str = "auto",
    ) -> H3NativeDenoiserBF16Resident:
        artifact = (
            artifact_or_directory
            if isinstance(artifact_or_directory, H3RuntimeArtifact)
            else load_h3_runtime_artifact(artifact_or_directory)
        )
        if not artifact.is_complete_block_stack:
            raise H3NativeDenoiserError("H3 RuntimeArtifact has no complete block stack")
        return cls(
            tuple(
                H3NativeBlockBF16Resident.load(
                    artifact,
                    block.index,
                    device=device,
                    adaln_table=(
                        adaln_table_load(block.index) if adaln_table_load is not None else None
                    ),
                    attention_backend=attention_backend,
                    elementwise_backend=elementwise_backend,
                    elementwise_block_size=elementwise_block_size,
                    adapter_fusion_backend=adapter_fusion_backend,
                    rotary_backend=rotary_backend,
                )
                for block in artifact.blocks
            )
        )


class H3NativeDenoiserBF16Ring:
    """Exact BF16 50-block executor with bounded SM86/SM89 device storage.

    The complete block stack remains in page-locked host memory.  Two fixed
    device slots alternate between compute and asynchronous H2D on a dedicated
    CUDA stream.  Events protect slot reuse, so block ``i + 2`` can replace
    block ``i`` only after the compute stream has consumed it.  This removes
    per-block CUDA allocation and makes PCIe transfer overlap an explicit
    runtime invariant rather than an offload-framework side effect.
    """

    backend_id = "cuda-bf16-pinned-host-two-slot-event-ring-torch-flash-v1"
    timing_eligible = True
    block_type = H3NativeBlockBF16Resident

    def __init__(
        self,
        artifact: H3RuntimeArtifact,
        host_blocks: tuple[H3NativeBlockWeights, ...],
        *,
        device: Any,
        attention_backend: str = "torch-flash",
        elementwise_backend: str = "auto",
        elementwise_block_size: int = 0,
        adapter_fusion_backend: str = "auto",
        rotary_backend: str = "auto",
    ) -> None:
        torch = _torch()
        runtime_device = torch.device(device)
        if runtime_device.type != "cuda":
            raise H3NativeDenoiserError("the BF16 block ring requires a CUDA device")
        if not artifact.is_complete_block_stack or len(host_blocks) != artifact.spec.num_layers:
            raise H3NativeDenoiserError("the BF16 block ring requires a complete block stack")
        if artifact.target.compute_capability not in {"sm86", "sm89"}:
            raise H3NativeDenoiserError("the BF16 block ring requires an SM86 or SM89 artifact")
        capability = torch.cuda.get_device_capability(runtime_device)
        if artifact.target.compute_capability != f"sm{capability[0]}{capability[1]}":
            raise H3NativeDenoiserError("the BF16 block ring device differs from its artifact")
        if artifact.target.attention_weight_bits != 16 or artifact.target.ffn_weight_bits != 16:
            raise H3NativeDenoiserError("the BF16 block ring requires exact BF16 weights")
        if len(host_blocks) < 2:
            raise H3NativeDenoiserError("the BF16 block ring requires at least two blocks")

        self.artifact = artifact
        self.host_blocks = host_blocks
        self.device = runtime_device
        self.attention_backend = attention_backend
        self.host_weight_bytes = sum(_block_tensor_bytes(row) for row in host_blocks)

        with torch.cuda.device(runtime_device):
            slot_weights = tuple(
                _empty_bf16_block_like(host_blocks[index], runtime_device) for index in range(2)
            )
            self.slots = tuple(
                self.block_type(
                    artifact,
                    weights,
                    device=runtime_device,
                    attention_backend=attention_backend,
                    elementwise_backend=elementwise_backend,
                    elementwise_block_size=elementwise_block_size,
                    adapter_fusion_backend=adapter_fusion_backend,
                    rotary_backend=rotary_backend,
                )
                for weights in slot_weights
            )
            self.copy_stream = torch.cuda.Stream(device=runtime_device)
            self.ready_events = tuple(
                torch.cuda.Event(enable_timing=False, blocking=False) for _ in range(2)
            )
            self.compute_done_events = tuple(
                torch.cuda.Event(enable_timing=False, blocking=False) for _ in range(2)
            )
        resolved_elementwise_backends = {slot.elementwise_backend for slot in self.slots}
        resolved_block_sizes = {slot.elementwise_block_size for slot in self.slots}
        if len(resolved_elementwise_backends) != 1 or len(resolved_block_sizes) != 1:
            raise H3NativeDenoiserError("the BF16 ring slots resolved different kernel plans")
        self.elementwise_backend = resolved_elementwise_backends.pop()
        self.elementwise_block_size = resolved_block_sizes.pop()
        resolved_adapter_fusion_backends = {slot.adapter_fusion_backend for slot in self.slots}
        if len(resolved_adapter_fusion_backends) != 1:
            raise H3NativeDenoiserError(
                "the BF16 ring slots resolved different adapter fusion backends"
            )
        self.adapter_fusion_backend = resolved_adapter_fusion_backends.pop()
        resolved_rotary_backends = {slot.rotary_backend for slot in self.slots}
        if len(resolved_rotary_backends) != 1:
            raise H3NativeDenoiserError(
                "the BF16 ring slots resolved different rotary backends"
            )
        self.rotary_backend = resolved_rotary_backends.pop()
        self.device_slot_weight_bytes = sum(
            _block_tensor_bytes(slot.weights) for slot in self.slots
        )

    @classmethod
    def load(
        cls,
        artifact_or_directory: H3RuntimeArtifact | Path,
        *,
        device: Any = "cuda:0",
        adaln_table_load: Callable[[int], Any] | None = None,
        attention_backend: str = "torch-flash",
        elementwise_backend: str = "auto",
        elementwise_block_size: int = 0,
        adapter_fusion_backend: str = "auto",
        rotary_backend: str = "auto",
    ) -> H3NativeDenoiserBF16Ring:
        """Load and pin the complete artifact before any timing begins."""

        artifact = (
            artifact_or_directory
            if isinstance(artifact_or_directory, H3RuntimeArtifact)
            else load_h3_runtime_artifact(artifact_or_directory)
        )
        if not artifact.is_complete_block_stack:
            raise H3NativeDenoiserError("H3 RuntimeArtifact has no complete block stack")
        host_blocks: list[H3NativeBlockWeights] = []
        for block in artifact.blocks:
            with H3MappedSafetensor(artifact.directory / block.path) as mapped:
                loaded_artifact, weights = load_h3_native_block(
                    artifact,
                    block.index,
                    adaln_table=(
                        adaln_table_load(block.index) if adaln_table_load is not None else None
                    ),
                    tensor_load=mapped.load,
                )
                if loaded_artifact != artifact:
                    raise H3NativeDenoiserError(
                        "BF16 ring block artifact changed while loading"
                    )
                host_blocks.append(_pin_bf16_block(weights))
                del weights
        return cls(
            artifact,
            tuple(host_blocks),
            device=device,
            attention_backend=attention_backend,
            elementwise_backend=elementwise_backend,
            elementwise_block_size=elementwise_block_size,
            adapter_fusion_backend=adapter_fusion_backend,
            rotary_backend=rotary_backend,
        )

    def prepare_invocation(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        adaln_indices: Any,
        rotary_cos: Any,
        rotary_sin: Any,
    ) -> H3NativeBlockInvocation:
        return self.slots[0].prepare_invocation(
            hidden_states,
            evaluation_index=evaluation_index,
            adaln_indices=adaln_indices,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )

    def _queue_initial_slots(self) -> None:
        torch = _torch()
        with torch.cuda.stream(self.copy_stream):
            for index in range(2):
                # The prior invocation may still be consuming the last two
                # blocks. An unrecorded event on the first invocation is a no-op.
                self.copy_stream.wait_event(self.compute_done_events[index])
                _copy_bf16_block_(self.slots[index].weights, self.host_blocks[index])
                self.ready_events[index].record(self.copy_stream)

    def forward_prevalidated(
        self,
        hidden_states: Any,
        invocation: H3NativeBlockInvocation,
        *,
        checkpoint_blocks: frozenset[int] = frozenset(),
    ) -> tuple[Any, dict[int, Any]]:
        """Run one NFE and optionally copy selected block outputs to host.

        Checkpoint copies are a correctness-only path and intentionally
        synchronize; production timing calls this with an empty set.
        """

        torch = _torch()
        if any(index < 0 or index >= len(self.host_blocks) for index in checkpoint_blocks):
            raise H3NativeDenoiserError("BF16 ring checkpoint index is outside the stack")
        compute_stream = torch.cuda.current_stream(self.device)
        self._queue_initial_slots()
        checkpoints: dict[int, Any] = {}
        for index in range(len(self.host_blocks)):
            slot_index = index % 2
            slot = self.slots[slot_index]
            compute_stream.wait_event(self.ready_events[slot_index])
            hidden_states = slot.forward_prevalidated(hidden_states, invocation)
            self.compute_done_events[slot_index].record(compute_stream)

            if index in checkpoint_blocks:
                # Keep GPU timing and oracle I/O separate at the caller.  This
                # branch exists only for the untimed correctness pass.
                checkpoints[index] = hidden_states.to(device="cpu")

            next_index = index + 2
            if next_index < len(self.host_blocks):
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_event(self.compute_done_events[slot_index])
                    _copy_bf16_block_(
                        slot.weights,
                        self.host_blocks[next_index],
                    )
                    self.ready_events[slot_index].record(self.copy_stream)
        return hidden_states, checkpoints

    def forward_prevalidated_serial(
        self,
        hidden_states: Any,
        invocation: H3NativeBlockInvocation,
        *,
        checkpoint_blocks: frozenset[int] = frozenset(),
    ) -> tuple[Any, dict[int, Any]]:
        """Correctness/control path with H2D serialized before every block."""

        torch = _torch()
        if any(index < 0 or index >= len(self.host_blocks) for index in checkpoint_blocks):
            raise H3NativeDenoiserError("BF16 ring checkpoint index is outside the stack")
        compute_stream = torch.cuda.current_stream(self.device)
        checkpoints: dict[int, Any] = {}
        slot = self.slots[0]
        for index, host_block in enumerate(self.host_blocks):
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(self.compute_done_events[0])
                _copy_bf16_block_(slot.weights, host_block)
                self.ready_events[0].record(self.copy_stream)
            compute_stream.wait_event(self.ready_events[0])
            hidden_states = slot.forward_prevalidated(hidden_states, invocation)
            self.compute_done_events[0].record(compute_stream)
            if index in checkpoint_blocks:
                checkpoints[index] = hidden_states.to(device="cpu")
        return hidden_states, checkpoints

    def __call__(
        self,
        hidden_states: Any,
        *,
        evaluation_index: int,
        adaln_indices: Any,
        rotary_cos: Any,
        rotary_sin: Any,
    ) -> Any:
        invocation = self.prepare_invocation(
            hidden_states,
            evaluation_index=evaluation_index,
            adaln_indices=adaln_indices,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        output, _checkpoints = self.forward_prevalidated(hidden_states, invocation)
        return output
