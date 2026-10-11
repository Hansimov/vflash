"""Explicit SM120 BF16 full-context dense attention through NVIDIA FROST.

NVIDIA cuDNN frontend 1.31.0 FROST. Plans retain metadata, not request tensors.
Requires the architecture-specific sm_120a compiler target, including CPU builds.
"""

import os
from functools import lru_cache

from vflash.contracts import ContractError


def require_compile_target():
    # Keep compiler selection explicit; changing a process-wide variable during
    # compilation could race another device or an unrelated CuTe caller.
    if os.environ.get("CUTE_DSL_ARCH") != "sm_120a":
        raise ContractError("FROST requires CUTE_DSL_ARCH=sm_120a before process startup")


@lru_cache(maxsize=16)
def _prepare(shape, strides, device, tile):
    require_compile_target()
    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120, TensorDesc

    descriptors = [
        TensorDesc(
            torch.bfloat16,
            shape,
            stride,
            tuple(sorted(range(4), key=lambda i: (stride[i], shape[i]))),
            device,
        )
        for stride in strides
    ]
    plan = SdpaFwdDslSm120(*descriptors, tile_m=tile[0], tile_n=tile[1], split_kv=1)
    plan.check_support()
    plan.compile()
    if plan.scratch_workspace_bytes() != 0:
        raise ContractError("FROST requires a native layout without scratch copies")
    return plan


def frost_dense(query, key, value):
    tile_mn = (128, 128)
    from importlib.metadata import version

    import torch

    if (
        query.device.type != "cuda"
        or query.dtype != torch.bfloat16
        or torch.cuda.get_device_capability(query.device) != (12, 0)
        or query.ndim != 4
        or query.shape[0] != 1
        or not 1 <= query.shape[1] <= 524288
        or tuple(query.shape[2:]) != (56, 128)
        or any(
            t.shape != query.shape
            or t.dtype != query.dtype
            or t.device != query.device
            or t.stride(-1) != 1
            or t.stride(-2) != 128
            or t.stride(1) not in (56 * 128, 3 * 56 * 128)
            for t in (query, key, value)
        )
    ):
        raise ContractError("FROST requires native SM120 BF16 H3 BSHD inputs")
    if version("nvidia-cudnn-frontend") != "1.31.0":
        raise ContractError("FROST differs from its pinned implementation")
    from cuda.bindings import driver as cuda

    output = torch.empty_like(query)
    views = [t.transpose(1, 2) for t in (query, key, value, output)]
    plan = _prepare(
        tuple(views[0].shape), tuple(tuple(t.stride()) for t in views), query.device, tile_mn
    )
    stream = cuda.CUstream(torch.cuda.current_stream(query.device).cuda_stream)
    plan.execute(*views, current_stream=stream)
    return output
