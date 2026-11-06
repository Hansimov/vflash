"""Explicit NVFP4 arithmetic for the original packed H3 QAD linear contract.

The caller selects already merged weights and owns all model/stage lifetimes.
No adapter merge, model lookup, attention selection or fallback occurs here.
The packing/scaling contract follows FastVideo's Apache-2.0 NVFP4 implementation:
https://github.com/hao-ai-lab/FastVideo/blob/764f50c711127985d7d4995cac307eaca7fff14a/fastvideo/layers/quantization/nvfp4_config.py
"""

from dataclasses import dataclass
from functools import cache

import torch


@dataclass(frozen=True, eq=False)
class NVFP4Weight:
    packed: torch.Tensor
    scales: torch.Tensor
    alpha: torch.Tensor
    input_scale: torch.Tensor

    def __post_init__(self):
        if (
            self.packed.dtype != torch.uint8
            or self.packed.ndim != 2
            or min(self.packed.shape) < 1
            or self.packed.shape[0] % 128
            or self.packed.shape[1] % 32
            or self.scales.dtype != torch.uint8
            or self.scales.shape != (self.packed.shape[0], self.packed.shape[1] // 8)
            or not self.packed.is_contiguous()
            or not self.scales.is_contiguous()
        ):
            raise ValueError("NVFP4 requires declared packed codes and 128x4 block scales")
        for scalar in (self.alpha, self.input_scale):
            if scalar.dtype != torch.float32 or scalar.shape != torch.Size([]):
                raise ValueError("NVFP4 calibration and output alpha must be FP32 scalars")
            if scalar.device.type != "meta" and not bool(torch.isfinite(scalar) & (scalar > 0)):
                raise ValueError("NVFP4 scales must be positive and finite")
        if (
            len({v.device for v in (self.packed, self.scales, self.alpha, self.input_scale)})
            != 1
        ):
            raise ValueError("NVFP4 tensors must share one explicit device")

    @property
    def in_features(self):
        return self.packed.shape[1] * 2

    @property
    def out_features(self):
        return self.packed.shape[0]


@cache
def _require_device(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("This NVFP4 execution requires an explicit SM120 CUDA device")


def _operators():
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    return SfLayout.layout_128x4, nvfp4_quantize, mm_fp4


class NVFP4Execution:
    """Frozen CUTLASS W4A4 execution; host clocks/counters remain diagnostic."""

    contract = "nvfp4-e2m1-block16-128x4-cutlass-v1"

    def __init__(self):
        self.calls = 0
        self.rows = 0
        self._ops = None

    def linear(self, values, weight: NVFP4Weight, bias=None):
        if (
            values.ndim < 1
            or values.shape[-1] != weight.in_features
            or values.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or values.device != weight.packed.device
        ):
            raise ValueError("NVFP4 input differs from its frozen matrix")
        if torch.is_grad_enabled() and values.requires_grad:
            raise ValueError("NVFP4 execution is inference-only")
        _require_device(values.device)
        if bias is not None and (
            bias.shape != (weight.out_features,)
            or bias.device != values.device
            or bias.dtype != torch.bfloat16
        ):
            raise ValueError("NVFP4 bias must match its BF16 output")
        if self._ops is None:
            self._ops = _operators()
        layout, quantize, matmul = self._ops
        shape = values.shape
        matrix = values.to(torch.bfloat16) if values.dtype == torch.float32 else values
        matrix = matrix.reshape(-1, weight.in_features)
        if matrix.shape[0] == 0:
            raise ValueError("NVFP4 input rows must be nonempty")
        codes, scales = quantize(matrix, weight.input_scale, sfLayout=layout, do_shuffle=False)
        padded_rows = matrix.shape[0] + (-matrix.shape[0]) % 128
        padded_columns = (weight.in_features // 16 + 3) // 4 * 4
        if (
            scales.dtype != torch.uint8
            or scales.device != matrix.device
            or scales.numel() != padded_rows * padded_columns
        ):
            raise ValueError("The quantizer omitted the required 128x4 scale padding")
        if codes.dtype == torch.float4_e2m1fn_x2:
            codes = codes.contiguous().view(torch.uint8)
        if codes.dtype != torch.uint8 or codes.shape != (
            matrix.shape[0],
            weight.in_features // 2,
        ):
            raise ValueError("The quantizer returned a different packed layout")
        output = matmul(
            codes,
            weight.packed.T,
            scales,
            weight.scales.T,
            weight.alpha / weight.input_scale,
            torch.bfloat16,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )
        if output.dtype != torch.bfloat16 or output.shape != (
            matrix.shape[0],
            weight.out_features,
        ):
            raise ValueError("The NVFP4 kernel returned a different output layout")
        self.calls += 1
        self.rows += matrix.shape[0]
        output = output.reshape(*shape[:-1], weight.out_features)
        return output if bias is None else output + bias
