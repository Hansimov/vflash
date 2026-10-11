"""Explicit official FA3 BF16 full-context dense attention for SM90.

Build the optional upstream dependency from the revision in the Veda guide.
No quantized attention, shortened context, input copies or silent fallback.
"""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from vflash.contracts import ContractError


@lru_cache(maxsize=1)
def implementation():
    try:
        if version("flash-attn-3") != "3.0.0":
            raise ContractError("FA3 requires the qualified flash-attn-3 3.0.0 build")
        from flash_attn_interface import flash_attn_func
    except (PackageNotFoundError, ImportError, OSError) as exc:
        raise ContractError("FA3 requires an explicitly installed SM90 CUDA build") from exc
    if not callable(flash_attn_func):
        raise ContractError("FA3 forward interface is unavailable")
    return flash_attn_func


def fa3_dense(query, key, value):
    import torch

    if (
        query.device.type != "cuda"
        or query.dtype != torch.bfloat16
        or torch.cuda.get_device_capability(query.device) != (9, 0)
        or query.ndim != 4
        or query.shape[0] != 1
        or not 1 <= query.shape[1] <= 524288
        or tuple(query.shape[2:]) != (56, 128)
        or any(
            t.shape != query.shape
            or t.dtype != query.dtype
            or t.device != query.device
            or t.stride(-1) != 1
            for t in (query, key, value)
        )
    ):
        raise ContractError("FA3 requires single SM90 BF16 H3 BSHD inputs")
    # Pass the original fused-QKV V view; the pinned upstream accepts its stride.
    output = implementation()(
        query, key, value, causal=False, num_splits=1, pack_gqa=False, return_attn_probs=False
    )
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ContractError("FA3 returned an incompatible attention result")
    return output
