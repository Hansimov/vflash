"""Explicit bounded row FP8 activation kernel for SM90/SM120.

One row reduction + cast replaces intermediate FP32 matrices. The portable Torch
expression remains the control. This module does not select model precision.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rows_fp8(
    X, Q, S, M: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    # SM120's owned FFN buffer returns the value half of [M, 2*K].
    # Its row stride can cross 2**31 elements on a long video even when
    # M*K fits int32. Widen BEFORE multiplication; pointer addition is too late.
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK)
    value = tl.load(X + row * STRIDE + column, (row < M) & (column < K), 0).to(tl.float32)
    maximum = tl.max(tl.abs(value), axis=0)
    scale = tl.where(
        maximum > 0, tl.maximum(tl.div_rn(maximum, 448.0), 1.1754943508222875e-38), 1.0
    )
    normalized = tl.minimum(tl.maximum(tl.div_rn(value, scale), -448.0), 448.0)
    tl.store(Q + row * K + column, normalized.to(tl.float8e4nv), column < K)
    tl.store(S + row, scale)


def quantize_activation(matrix):
    if (
        matrix.device.type != "cuda"
        or matrix.dtype != torch.bfloat16
        or matrix.ndim != 2
        or matrix.stride(1) != 1
        or not 16 <= matrix.shape[1] <= 32768
        or torch.cuda.get_device_capability(matrix.device) not in ((9, 0), (12, 0))
    ):
        raise ValueError("row FP8 activation requires a bounded SM90/SM120 BF16 matrix")
    m, k = matrix.shape
    padded = triton.cdiv(m, 16) * 16
    codes = torch.empty((padded, k), device=matrix.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((padded, 1), device=matrix.device, dtype=torch.float32)
    _rows_fp8[(padded,)](
        matrix,
        codes,
        scales,
        m,
        k,
        matrix.stride(0),
        triton.next_power_of_2(k),
        num_warps=4 if k <= 8192 else 8,
    )
    return codes, scales
