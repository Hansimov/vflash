"""Narrow, replaceable fused elementwise kernels for the H3 hot path.

The first backend preserves the eager BF16 rounding boundaries instead of
silently changing the model to a mathematically similar fused expression.
These kernels are intentionally separate from GEMM and attention selection so
the runtime can compose them with resident weights, streamed weights, and
runtime low-rank adapters.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any


class H3FusedOpsError(ValueError):
    """A fused H3 invocation violates the fixed CUDA/BF16 contract."""


_BLOCK_SIZES = frozenset({128, 256, 512, 1024})


@cache
def _strict_bf16_kernels() -> tuple[Any, Any, Any]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the strict fused BF16 backend requires Triton") from exc

    @triton.jit
    def strict_modulate_kernel(
        output_ptr,
        input_ptr,
        scale_ptr,
        shift_ptr,
        elements,
        tokens,
        hidden_size,
        scale_token_stride,
        scale_hidden_stride,
        shift_token_stride,
        shift_hidden_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        hidden_offsets = offsets % hidden_size
        token_offsets = (offsets // hidden_size) % tokens
        input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale_value = tl.load(
            scale_ptr
            + token_offsets * scale_token_stride
            + hidden_offsets * scale_hidden_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        shift_value = tl.load(
            shift_ptr
            + token_offsets * shift_token_stride
            + hidden_offsets * shift_hidden_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Eager H3 executes three BF16 TensorIterator kernels.  Retain each
        # intermediate BF16 rounding boundary while eliminating their global
        # memory round trips and two launches.
        one_plus_scale = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_scale;
            cvt.rn.bf16.f32 rounded_scale, $1;
            cvt.f32.bf16 $0, rounded_scale;
            }
            """,
            constraints="=f,f",
            args=[1.0 + scale_value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        scaled = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[input_value * one_plus_scale],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        output_value = scaled + shift_value
        tl.store(output_ptr + offsets, output_value, mask=mask)

    @triton.jit
    def strict_gate_residual_kernel(
        output_ptr,
        residual_ptr,
        gate_ptr,
        projected_ptr,
        elements,
        tokens,
        hidden_size,
        gate_token_stride,
        gate_hidden_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        hidden_offsets = offsets % hidden_size
        token_offsets = (offsets // hidden_size) % tokens
        residual_value = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        projected_value = tl.load(projected_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_value = tl.load(
            gate_ptr + token_offsets * gate_token_stride + hidden_offsets * gate_hidden_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # The explicit BF16 conversion prevents contraction into one FP32 FMA
        # and matches eager ``gate * projected`` followed by residual addition.
        gated = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_gate;
            cvt.rn.bf16.f32 rounded_gate, $1;
            cvt.f32.bf16 $0, rounded_gate;
            }
            """,
            constraints="=f,f",
            args=[gate_value * projected_value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        output_value = residual_value + gated
        tl.store(output_ptr + offsets, output_value, mask=mask)

    return triton, strict_modulate_kernel, strict_gate_residual_kernel


@cache
def _strict_bf16_silu_mul_kernel() -> tuple[Any, Any]:
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the strict fused BF16 backend requires Triton") from exc

    @triton.jit
    def strict_silu_mul_kernel(
        output_ptr,
        packed_ptr,
        elements,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        hidden_offsets = offsets % hidden_size
        row_offsets = offsets // hidden_size
        input_offsets = row_offsets * hidden_size * 2 + hidden_offsets
        value = tl.load(packed_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            packed_ptr + input_offsets + hidden_size,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        activated = gate / (1.0 + libdevice.exp(-gate))
        # Eager BF16 SiLU stores its result before the following multiply.
        # Keep that rounding boundary inside this one-pass kernel.
        rounded_activation = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[activated],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(output_ptr + offsets, value * rounded_activation, mask=mask)

    return triton, strict_silu_mul_kernel


@cache
def _strict_bf16_ffn_adapter_silu_kernel() -> Any:
    """Build the JIT definition without requesting a device or compiling it."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def round_bf16(value):
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded;
            cvt.rn.bf16.f32 rounded, $1;
            cvt.f32.bf16 $0, rounded;
            }
            """,
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def strict_ffn_adapter_silu_kernel(
        output_ptr,
        base_ptr,
        adapter_ptr,
        elements,
        hidden_size,
        scaling,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        hidden_offsets = offsets % hidden_size
        rows = offsets // hidden_size
        value_offsets = rows * (2 * hidden_size) + hidden_offsets
        gate_offsets = value_offsets + hidden_size
        base_value = tl.load(base_ptr + value_offsets, mask=mask, other=0).to(tl.float32)
        adapter_value = tl.load(adapter_ptr + value_offsets, mask=mask, other=0).to(tl.float32)
        base_gate = tl.load(base_ptr + gate_offsets, mask=mask, other=0).to(tl.float32)
        adapter_gate = tl.load(adapter_ptr + gate_offsets, mask=mask, other=0).to(tl.float32)

        # Retain the separate adapter merge and SiLU BF16 round boundaries.
        value = round_bf16(base_value + round_bf16(adapter_value * scaling))
        gate = round_bf16(base_gate + round_bf16(adapter_gate * scaling))
        activated = gate / (1.0 + libdevice.exp(-gate))
        output = value * round_bf16(activated)
        tl.store(output_ptr + offsets, output, mask=mask)

    return strict_ffn_adapter_silu_kernel


@cache
def _strict_bf16_adapter_kernels() -> tuple[Any, Any, Any]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the strict fused BF16 backend requires Triton") from exc

    @triton.jit
    def strict_adapter_merge_kernel(
        output_ptr,
        base_ptr,
        adapter_ptr,
        elements,
        hidden_size,
        branch_size,
        scaling_0,
        scaling_1,
        scaling_2,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        hidden_offsets = offsets % hidden_size
        branches = hidden_offsets // branch_size
        scaling = tl.where(
            branches == 0,
            scaling_0,
            tl.where(branches == 1, scaling_1, scaling_2),
        )
        base = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        adapter = tl.load(adapter_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scaled = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[adapter * scaling],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(output_ptr + offsets, base + scaled, mask=mask)

    @triton.jit
    def strict_adapter_gate_residual_kernel(
        output_ptr,
        base_ptr,
        adapter_ptr,
        residual_ptr,
        gate_ptr,
        elements,
        tokens,
        hidden_size,
        gate_token_stride,
        gate_hidden_stride,
        scaling,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        hidden_offsets = offsets % hidden_size
        token_offsets = (offsets // hidden_size) % tokens
        base = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        adapter = tl.load(adapter_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            gate_ptr + token_offsets * gate_token_stride + hidden_offsets * gate_hidden_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        scaled = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[adapter * scaling],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        adapted = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[base + scaled],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        gated = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[gate * adapted],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(output_ptr + offsets, residual + gated, mask=mask)

    return triton, strict_adapter_merge_kernel, strict_adapter_gate_residual_kernel


@cache
def _strict_bf16_rotary_kernel() -> tuple[Any, Any]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the strict fused BF16 backend requires Triton") from exc

    @triton.jit
    def strict_rotary_kernel(
        output_ptr,
        input_ptr,
        cos_ptr,
        sin_ptr,
        elements,
        tokens,
        heads,
        head_dim,
        rotary_dim,
        cos_token_stride,
        cos_hidden_stride,
        sin_token_stride,
        sin_hidden_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        hidden_offsets = offsets % head_dim
        token_offsets = (offsets // (heads * head_dim)) % tokens
        rotary_mask = mask & (hidden_offsets < rotary_dim)
        half = rotary_dim // 2
        partner_hidden = tl.where(
            hidden_offsets < half,
            hidden_offsets + half,
            hidden_offsets - half,
        )
        partner_offsets = offsets - hidden_offsets + partner_hidden
        value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        partner = tl.load(
            input_ptr + partner_offsets,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        partner = tl.where(hidden_offsets < half, -partner, partner)
        cosine = tl.load(
            cos_ptr + token_offsets * cos_token_stride + hidden_offsets * cos_hidden_stride,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        sine = tl.load(
            sin_ptr + token_offsets * sin_token_stride + hidden_offsets * sin_hidden_stride,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        cosine_product = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[value * cosine],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        sine_product = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded_value;
            cvt.rn.bf16.f32 rounded_value, $1;
            cvt.f32.bf16 $0, rounded_value;
            }
            """,
            constraints="=f,f",
            args=[partner * sine],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        output = tl.where(rotary_mask, cosine_product + sine_product, value)
        tl.store(output_ptr + offsets, output, mask=mask)

    return triton, strict_rotary_kernel


def _validate_common(
    value: Any,
    modifier: Any,
    *,
    modifier_name: str,
    block_size: int,
) -> tuple[int, int, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the fused H3 backend requires PyTorch") from exc
    if block_size not in _BLOCK_SIZES:
        raise H3FusedOpsError(f"unsupported fused H3 block size: {block_size}")
    if not isinstance(value, torch.Tensor) or not isinstance(modifier, torch.Tensor):
        raise H3FusedOpsError("fused H3 inputs must be tensors")
    if value.device.type != "cuda" or modifier.device != value.device:
        raise H3FusedOpsError("fused H3 inputs must share one CUDA device")
    if value.dtype != torch.bfloat16 or modifier.dtype != torch.bfloat16:
        raise H3FusedOpsError("fused H3 inputs must use BF16")
    if value.ndim != 3 or modifier.ndim != 2:
        raise H3FusedOpsError(
            f"fused H3 value must be [B,S,H] and {modifier_name} must be [S,H]"
        )
    batch, tokens, hidden_size = (int(item) for item in value.shape)
    if tuple(modifier.shape) != (tokens, hidden_size):
        raise H3FusedOpsError(f"fused H3 {modifier_name} shape differs from the activation")
    if not value.is_contiguous() or modifier.stride(-1) != 1:
        raise H3FusedOpsError("fused H3 activations require a contiguous hidden dimension")
    return batch, tokens, hidden_size


def _validate_matching_activation(
    value: Any,
    candidate: Any,
    *,
    candidate_name: str,
    block_size: int,
) -> tuple[int, int, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the fused H3 backend requires PyTorch") from exc
    if block_size not in _BLOCK_SIZES:
        raise H3FusedOpsError(f"unsupported fused H3 block size: {block_size}")
    if not isinstance(value, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise H3FusedOpsError("fused H3 inputs must be tensors")
    if value.device.type != "cuda" or candidate.device != value.device:
        raise H3FusedOpsError("fused H3 inputs must share one CUDA device")
    if value.dtype != torch.bfloat16 or candidate.dtype != torch.bfloat16:
        raise H3FusedOpsError("fused H3 inputs must use BF16")
    if value.ndim != 3 or tuple(candidate.shape) != tuple(value.shape):
        raise H3FusedOpsError(f"fused H3 {candidate_name} must match the [B,S,H] activation")
    if not value.is_contiguous() or not candidate.is_contiguous():
        raise H3FusedOpsError("fused H3 activations must be contiguous")
    return tuple(int(item) for item in value.shape)


def _validate_scalings(scalings: tuple[float, ...], expected: int) -> tuple[float, ...]:
    if len(scalings) != expected or any(not math.isfinite(value) for value in scalings):
        raise H3FusedOpsError(f"fused H3 adapter requires {expected} finite scalings")
    return tuple(float(value) for value in scalings)


def triton_strict_bf16_modulate(
    value: Any,
    scale: Any,
    shift: Any,
    *,
    block_size: int = 256,
) -> Any:
    """Compute ``value * (1 + scale) + shift`` with eager BF16 rounding."""

    _batch, tokens, hidden_size = _validate_common(
        value,
        scale,
        modifier_name="scale",
        block_size=block_size,
    )
    _validate_common(value, shift, modifier_name="shift", block_size=block_size)
    triton, kernel, _gate_kernel = _strict_bf16_kernels()
    import torch

    output = torch.empty_like(value)
    kernel[(triton.cdiv(value.numel(), block_size),)](
        output,
        value,
        scale,
        shift,
        value.numel(),
        tokens,
        hidden_size,
        scale.stride(0),
        scale.stride(1),
        shift.stride(0),
        shift.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=min(8, block_size // 32),
    )
    return output


def triton_strict_bf16_gate_residual(
    residual: Any,
    gate: Any,
    projected: Any,
    *,
    block_size: int = 256,
) -> Any:
    """Compute ``residual + gate * projected`` with eager BF16 rounding."""

    _batch, tokens, hidden_size = _validate_common(
        residual,
        gate,
        modifier_name="gate",
        block_size=block_size,
    )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the fused H3 backend requires PyTorch") from exc
    if (
        not isinstance(projected, torch.Tensor)
        or projected.device != residual.device
        or projected.dtype != torch.bfloat16
        or tuple(projected.shape) != tuple(residual.shape)
        or not projected.is_contiguous()
    ):
        raise H3FusedOpsError("fused H3 projected activation differs from the residual")
    triton, _modulate_kernel, kernel = _strict_bf16_kernels()
    output = torch.empty_like(residual)
    kernel[(triton.cdiv(residual.numel(), block_size),)](
        output,
        residual,
        gate,
        projected,
        residual.numel(),
        tokens,
        hidden_size,
        gate.stride(0),
        gate.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=min(8, block_size // 32),
    )
    return output


def triton_strict_bf16_silu_mul(
    packed: Any,
    *,
    block_size: int = 256,
) -> Any:
    """Fuse H3 ``value * silu(gate)`` while retaining the eager BF16 round."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the fused H3 backend requires PyTorch") from exc
    if block_size not in _BLOCK_SIZES:
        raise H3FusedOpsError(f"unsupported fused H3 block size: {block_size}")
    if (
        not isinstance(packed, torch.Tensor)
        or packed.device.type != "cuda"
        or packed.dtype != torch.bfloat16
        or packed.ndim != 3
        or int(packed.shape[-1]) % 2
        or not packed.is_contiguous()
    ):
        raise H3FusedOpsError("fused H3 SiLU/mul requires contiguous CUDA BF16 [B,S,2H] input")
    hidden_size = int(packed.shape[-1]) // 2
    if hidden_size <= 0:
        raise H3FusedOpsError("fused H3 SiLU/mul hidden size must be positive")
    output = torch.empty(
        (*packed.shape[:-1], hidden_size),
        device=packed.device,
        dtype=packed.dtype,
    )
    triton, kernel = _strict_bf16_silu_mul_kernel()
    kernel[(triton.cdiv(output.numel(), block_size),)](
        output,
        packed,
        output.numel(),
        hidden_size,
        BLOCK_SIZE=block_size,
        num_warps=min(8, block_size // 32),
    )
    return output


def triton_strict_bf16_ffn_adapter_silu(
    base: Any, adapter: Any, *, scaling: float, block_size: int
) -> Any:
    """Merge the FFN adapter and SiLU without a merged [B,S,2F] temporary.

    The separate BF16 LoRA product, addition and SiLU round boundaries remain
    explicit. This entry is dispatched only for the qualified SM89 Ref4 profile.
    """

    import torch
    import triton

    if (
        not isinstance(base, torch.Tensor)
        or not isinstance(adapter, torch.Tensor)
        or base.device.type != "cuda"
        or adapter.device != base.device
        or base.dtype != torch.bfloat16
        or adapter.dtype != base.dtype
        or base.ndim != 3
        or tuple(adapter.shape) != tuple(base.shape)
        or not base.is_contiguous()
        or not adapter.is_contiguous()
        or any(size <= 0 for size in base.shape)
        or base.shape[-1] % 2
        or base.numel() >= 2**31
    ):
        raise H3FusedOpsError("FFN fusion requires matching contiguous CUDA BF16 [B,S,2F]")
    if (
        not isinstance(scaling, (float, int))
        or isinstance(scaling, bool)
        or not math.isfinite(scaling)
        or block_size != 1024
    ):
        raise H3FusedOpsError("FFN fusion requires finite scaling and block size 1024")
    width = base.shape[-1] // 2
    output = torch.empty((*base.shape[:-1], width), device=base.device, dtype=base.dtype)
    _strict_bf16_ffn_adapter_silu_kernel()[(triton.cdiv(output.numel(), block_size),)](
        output,
        base,
        adapter,
        output.numel(),
        width,
        float(scaling),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def triton_strict_bf16_adapter_merge(
    base: Any,
    adapter: Any,
    *,
    scaling: float,
    block_size: int = 256,
) -> Any:
    """Fuse LoRA scaling and base addition with official BF16 boundaries."""

    _batch, _tokens, hidden_size = _validate_matching_activation(
        base,
        adapter,
        candidate_name="adapter",
        block_size=block_size,
    )
    (scaling,) = _validate_scalings((scaling,), 1)
    triton, kernel, _gate_kernel = _strict_bf16_adapter_kernels()
    import torch

    output = torch.empty_like(base)
    kernel[(triton.cdiv(base.numel(), block_size),)](
        output,
        base,
        adapter,
        base.numel(),
        hidden_size,
        hidden_size,
        scaling,
        scaling,
        scaling,
        BLOCK_SIZE=block_size,
        num_warps=min(8, block_size // 32),
    )
    return output


def triton_strict_bf16_qkv_adapter_merge(
    base: Any,
    concatenated_adapter: Any,
    *,
    scalings: tuple[float, float, float],
    block_size: int = 256,
) -> Any:
    """Fuse three Q/K/V LoRA scaling/add branches after one contiguous pack."""

    _batch, _tokens, hidden_size = _validate_matching_activation(
        base,
        concatenated_adapter,
        candidate_name="concatenated QKV adapter",
        block_size=block_size,
    )
    if hidden_size % 3:
        raise H3FusedOpsError("fused H3 QKV hidden size must be divisible by three")
    scaling_0, scaling_1, scaling_2 = _validate_scalings(scalings, 3)
    triton, kernel, _gate_kernel = _strict_bf16_adapter_kernels()
    import torch

    output = torch.empty_like(base)
    kernel[(triton.cdiv(base.numel(), block_size),)](
        output,
        base,
        concatenated_adapter,
        base.numel(),
        hidden_size,
        hidden_size // 3,
        scaling_0,
        scaling_1,
        scaling_2,
        BLOCK_SIZE=block_size,
        num_warps=min(8, block_size // 32),
    )
    return output


def _shares_storage(left: Any, right: Any) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _validate_qkv_direct_merge(
    base: Any,
    adapters: tuple[Any, Any, Any],
    scalings: tuple[float, float, float],
    *,
    block_size: int,
) -> tuple[int, int, tuple[float, float, float]]:
    """Validate separate BF16 branches and exclusive ownership of the base storage."""
    import torch

    if (
        not isinstance(base, torch.Tensor)
        or base.dtype != torch.bfloat16
        or base.ndim != 3
        or any(size <= 0 for size in base.shape)
        or base.shape[-1] % 3
        or not base.is_contiguous()
        or base.numel() >= 2**31
        or base.requires_grad
        or not isinstance(adapters, tuple)
        or len(adapters) != 3
    ):
        raise H3FusedOpsError("base must be a contiguous inference BF16 [B,S,3H]")
    expected = (*base.shape[:-1], base.shape[-1] // 3)
    for adapter in adapters:
        if (
            not isinstance(adapter, torch.Tensor)
            or adapter.device != base.device
            or adapter.dtype != base.dtype
            or tuple(adapter.shape) != expected
            or not adapter.is_contiguous()
            or adapter.requires_grad
            or _shares_storage(base, adapter)
        ):
            raise H3FusedOpsError("adapters must be separate contiguous BF16 [B,S,H]")
    if (
        not isinstance(scalings, tuple)
        or len(scalings) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in scalings
        )
        or block_size != 1024
    ):
        raise H3FusedOpsError("require three finite scalings and SM89 block size 1024")
    return base.numel() // 3, expected[-1], tuple(float(value) for value in scalings)


@cache
def _strict_bf16_qkv_direct_merge_kernel() -> Any:
    """Build a Triton JIT definition without selecting a device or compiling."""
    import triton
    import triton.language as tl

    @triton.jit
    def strict_qkv_direct_merge_kernel(
        base_ptr,
        query_adapter_ptr,
        key_adapter_ptr,
        value_adapter_ptr,
        branch_elements,
        branch_width,
        scaling_0,
        scaling_1,
        scaling_2,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        branch = tl.program_id(1)
        mask = offsets < branch_elements
        rows = offsets // branch_width
        columns = offsets % branch_width
        packed_offsets = rows * (3 * branch_width) + branch * branch_width + columns
        adapter_ptr = tl.where(
            branch == 0,
            query_adapter_ptr,
            tl.where(branch == 1, key_adapter_ptr, value_adapter_ptr),
        )
        scaling = tl.where(branch == 0, scaling_0, tl.where(branch == 1, scaling_1, scaling_2))
        base = tl.load(base_ptr + packed_offsets, mask=mask, other=0).to(tl.float32)
        adapter = tl.load(adapter_ptr + offsets, mask=mask, other=0).to(tl.float32)
        scaled = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 rounded;
            cvt.rn.bf16.f32 rounded, $1;
            cvt.f32.bf16 $0, rounded;
            }
            """,
            constraints="=f,f",
            args=[adapter * scaling],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        # Store the second BF16 round, exactly as the original merged output.
        merged = (base + scaled).to(tl.bfloat16)
        tl.store(base_ptr + packed_offsets, merged, mask=mask)

    return strict_qkv_direct_merge_kernel


def triton_strict_bf16_qkv_direct_merge(
    base: Any,
    adapters: tuple[Any, Any, Any],
    *,
    scalings: tuple[float, float, float],
    block_size: int = 1024,
) -> Any:
    """Merge three LoRA branches into a fresh base; the caller relinquishes its old values."""
    elements, width, scales = _validate_qkv_direct_merge(
        base, adapters, scalings, block_size=block_size
    )
    if base.device.type != "cuda":
        raise H3FusedOpsError("the direct QKV merge requires CUDA BF16 tensors")
    import triton

    _strict_bf16_qkv_direct_merge_kernel()[(triton.cdiv(elements, block_size), 3)](
        base,
        *adapters,
        elements,
        width,
        *scales,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return base


def triton_strict_bf16_adapter_gate_residual(
    base: Any,
    adapter: Any,
    residual: Any,
    gate: Any,
    *,
    scaling: float,
    block_size: int = 256,
) -> Any:
    """Fuse LoRA scale/add and output gate/residual with strict BF16 rounding."""

    _validate_matching_activation(
        base,
        adapter,
        candidate_name="adapter",
        block_size=block_size,
    )
    _batch, tokens, hidden_size = _validate_common(
        residual,
        gate,
        modifier_name="gate",
        block_size=block_size,
    )
    if tuple(base.shape) != tuple(residual.shape):
        raise H3FusedOpsError("fused H3 projection must match the residual")
    (scaling,) = _validate_scalings((scaling,), 1)
    triton, _merge_kernel, kernel = _strict_bf16_adapter_kernels()
    import torch

    output = torch.empty_like(residual)
    kernel[(triton.cdiv(residual.numel(), block_size),)](
        output,
        base,
        adapter,
        residual,
        gate,
        residual.numel(),
        tokens,
        hidden_size,
        gate.stride(0),
        gate.stride(1),
        scaling,
        BLOCK_SIZE=block_size,
        num_warps=min(8, block_size // 32),
    )
    return output


def triton_strict_bf16_rotary(
    value: Any,
    cos: Any,
    sin: Any,
    *,
    block_size: int = 256,
) -> Any:
    """Apply H3 split-half RoPE while preserving eager BF16 products."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3FusedOpsError("the fused H3 backend requires PyTorch") from exc
    if block_size not in _BLOCK_SIZES:
        raise H3FusedOpsError(f"unsupported fused H3 block size: {block_size}")
    if not all(isinstance(item, torch.Tensor) for item in (value, cos, sin)):
        raise H3FusedOpsError("fused H3 rotary inputs must be tensors")
    if value.device.type != "cuda" or cos.device != value.device or sin.device != value.device:
        raise H3FusedOpsError("fused H3 rotary inputs must share one CUDA device")
    if value.dtype != torch.bfloat16 or cos.dtype != value.dtype or sin.dtype != value.dtype:
        raise H3FusedOpsError("fused H3 rotary inputs must use BF16")
    if value.ndim != 4 or cos.ndim != 2 or tuple(sin.shape) != tuple(cos.shape):
        raise H3FusedOpsError("fused H3 rotary shapes are invalid")
    _batch, tokens, heads, head_dim = (int(item) for item in value.shape)
    rotary_dim = int(cos.shape[1])
    if (
        int(cos.shape[0]) != tokens
        or rotary_dim <= 0
        or rotary_dim > head_dim
        or rotary_dim % 2
    ):
        raise H3FusedOpsError("fused H3 rotary dimension is invalid")
    if not value.is_contiguous() or cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise H3FusedOpsError("fused H3 rotary inputs require contiguous hidden dimensions")
    triton, kernel = _strict_bf16_rotary_kernel()
    output = torch.empty_like(value)
    kernel[(triton.cdiv(value.numel(), block_size),)](
        output,
        value,
        cos,
        sin,
        value.numel(),
        tokens,
        heads,
        head_dim,
        rotary_dim,
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=min(8, block_size // 32),
    )
    return output
