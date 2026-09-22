"""Explicit restoration-only attention; never patches H3 or torch globally."""

from __future__ import annotations

from typing import Any

ATTENTION_BACKENDS = ("torch", "sage-int8-fp16")


def configure_attention(modules: list[Any], backend: str) -> None:
    if backend not in ATTENTION_BACKENDS:
        raise ValueError("restoration attention must be torch or sage-int8-fp16")
    if backend == "torch":
        return
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton
    except ImportError as error:
        raise RuntimeError(
            "sage-int8-fp16 requires an ABI-compatible SageAttention 2 installation"
        ) from error

    def attention(q: Any, k: Any, v: Any, num_heads: int, compatibility_mode=False):
        import torch

        if compatibility_mode or not q.is_cuda:
            raise ValueError("sage-int8-fp16 requires the CUDA restoration path")
        if torch.cuda.get_device_capability(q.device) != (8, 9):
            raise ValueError("this restoration Sage profile is restricted to SM89")
        # BF16 has a wider exponent range than the chosen FP16 value kernel.
        maximum = v.abs().amax()
        if not torch.isfinite(maximum) or maximum >= 65504:
            raise ValueError("restoration values exceed the FP16 attention range")
        q = q.unflatten(-1, (num_heads, -1))
        k = k.unflatten(-1, (num_heads, -1))
        v = v.unflatten(-1, (num_heads, -1))
        result = sageattn_qk_int8_pv_fp16_triton(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=False,
            smooth_k=True,
            quantization_backend="triton",
        )
        return result.flatten(-2)

    for module in modules:
        module.flash_attention = attention
