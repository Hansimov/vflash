"""Explicit approximate row-scaled E4M3 linear execution.

This is a numerical primitive, not a pipeline precision selector or an asset
loader. The caller owns trusted prepared codes/scales and model admission.
Torch and Triton activation quantization are explicit alternatives; W8A8 never
falls back to restored-weight arithmetic on an unsupported device.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


def _quantize_activation_torch(matrix):
    """Portable reference for the fused kernel, with GEMM row padding."""
    rows = matrix.shape[0]
    if rows % 16:
        matrix = torch.nn.functional.pad(matrix, (0, 0, 0, 16 - rows % 16))
    floats = matrix.float()
    maxima = floats.abs().amax(dim=1, keepdim=True)
    act_scale = torch.where(
        maxima > 0,
        (maxima / 448.0).clamp_min(torch.finfo(torch.float32).tiny),
        torch.ones_like(maxima),
    )
    codes = (floats / act_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return codes, act_scale


class RowFP8Execution:
    """Own precision choice and counters without changing any process globals.

    Weight-only restores approximate weights for the ordinary GEMM. W8A8 uses
    BF16 input, E4M3 activations/weights and FP32 per-row scales on SM90/SM120.
    ``measure`` may supply asynchronous profiling scopes; this class inserts no
    synchronization, model reads, environment changes or device selection.
    """

    def __init__(
        self,
        mode: str = "weight-only",
        *,
        activation_quantizer: str = "torch",
        fast_accum: bool = False,
        measure: Callable[[str], AbstractContextManager[Any]] | None = None,
    ) -> None:
        if activation_quantizer not in ("torch", "triton"):
            raise ValueError("unknown FP8 activation quantizer")
        if type(fast_accum) is not bool:
            raise ValueError("fast_accum must be an explicit boolean")
        self.activation_quantizer = activation_quantizer
        self.fast_accum = fast_accum
        self.measure = measure if measure is not None else lambda _name: nullcontext()
        self.observer: Callable[[], None] | None = None
        self.calls = {"weight_only": 0, "w8a8": 0}
        self.select(mode)

    def select(self, mode: str) -> None:
        if mode not in ("weight-only", "w8a8"):
            raise ValueError("unknown FP8 execution mode")
        self.mode = mode

    def linear(self, value, weight, scale, bias=None):
        if self.observer is not None and sum(self.calls.values()) % 200 == 0:
            self.observer()
        if (
            weight.ndim != 2
            or value.ndim < 1
            or min(weight.shape) <= 0
            or value.shape[-1] != weight.shape[1]
            or weight.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float32
            or scale.shape != (weight.shape[0], 1)
            or weight.device != value.device
            or scale.device != value.device
        ):
            raise ValueError("FP8 codes/scales differ from the declared layout")
        if bias is not None and (
            bias.shape != (weight.shape[0],)
            or bias.device != value.device
            or bias.dtype != value.dtype
        ):
            raise ValueError("FP8 bias differs from the declared output")
        if self.mode == "weight-only":
            self.calls["weight_only"] += 1
            restored = (weight.float() * scale).to(value.dtype)
            return torch.nn.functional.linear(value, restored, bias)
        if (
            value.dtype != torch.bfloat16
            or value.device.type != "cuda"
            or torch.cuda.get_device_capability(value.device) not in ((9, 0), (12, 0))
        ):
            raise ValueError("W8A8 requires Hopper/RTX Blackwell BF16 inputs")
        self.calls["w8a8"] += 1
        shape = value.shape
        matrix = value.reshape(-1, shape[-1])
        rows = matrix.shape[0]
        if self.activation_quantizer == "triton":
            from vflash.native.h3_fp8_activation import quantize_activation

            with self.measure("linear.activation_quantize"):
                codes, act_scale = quantize_activation(matrix)
        else:
            codes, act_scale = _quantize_activation_torch(matrix)
        with self.measure("linear.fp8_gemm"):
            output = torch._scaled_mm(
                codes,
                weight.t(),
                scale_a=act_scale,
                scale_b=scale.t(),
                out_dtype=torch.bfloat16,
                use_fast_accum=self.fast_accum,
            )
        if isinstance(output, tuple):
            output = output[0]
        output = output[:rows].reshape(*shape[:-1], weight.shape[0])
        return output if bias is None else output + bias


class RowFP8Linear(torch.nn.Module):
    """Meta-initialized Linear replacement; assign trusted codes and row_scale.

    The original bias is preserved. A loader must assign all meta tensors before
    execution, and retain the intended host/device placement of each module.
    """

    def __init__(self, original: torch.nn.Linear, execution: RowFP8Execution):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.out_features, self.in_features), device="meta", dtype=torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "row_scale",
            torch.empty((self.out_features, 1), device="meta", dtype=torch.float32),
        )
        self.bias = original.bias
        self.execution = execution

    def forward(self, value):
        return self.execution.linear(value, self.weight, self.row_scale, self.bias)
