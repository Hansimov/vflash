"""Explicit optional dense blocks within Veda's mixed attention policy.

SageAttention changes precision and outputs; it is never a silent Flash fallback. Selecting it
is the whole INT8 Q·K / FP8 P·V policy: Veda's sparse layers then also use FP8 P·V.
CuTe/FA3 (SM90) and FROST (SM120) retain BF16 full-context dense attention.
Only the selected dense blocks change (ten by default, six when explicitly selected).
Sparse/protected Veda connections stay intact.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vflash.contracts import ContractError


def validate_dense_backend(value: str, *, attention_backend: str, capability: str) -> None:
    if value not in {"torch-flash", "sageattention2", "cute-flash", "frost-flash", "fa3-flash"}:
        raise ContractError("unknown Veda dense backend")
    if value == "sageattention2" and (
        attention_backend != "veda-triton" or capability != "12.0"
    ):
        raise ContractError("SageAttention2 dense blocks require explicit single-SM120 Veda")

    if value == "cute-flash" and (attention_backend != "veda-triton" or capability != "9.0"):
        raise ContractError("CuTe Flash dense blocks require explicit single-SM90 Veda")

    if value == "fa3-flash" and (attention_backend != "veda-triton" or capability != "9.0"):
        raise ContractError("FA3 dense blocks require explicit single-SM90 Veda")

    if value == "frost-flash" and (attention_backend != "veda-triton" or capability != "12.0"):
        raise ContractError("FROST dense blocks require explicit single-SM120 Veda")


def require_dense_dependencies(value: str) -> None:
    if value == "torch-flash":
        return
    if value == "fa3-flash":
        from vflash.native.h3_fa3_dense import implementation

        implementation()
        return
    if value == "frost-flash":
        try:
            if version("nvidia-cudnn-frontend") != "1.31.0":
                raise ContractError("FROST requires the qualified nvidia-cudnn-frontend 1.31.0")
            from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120, TensorDesc
        except (PackageNotFoundError, ImportError, OSError) as exc:
            raise ContractError(
                "FROST requires an explicitly installed 1.31.0 CUDA build"
            ) from exc
        if not callable(SdpaFwdDslSm120) or not callable(TensorDesc):
            raise ContractError("FROST SM120 interface is unavailable")
        from vflash.native.h3_frost_dense import require_compile_target

        require_compile_target()
        return
    if value == "cute-flash":
        try:
            installed = version("flash-attn-4")
            if installed != "4.0.0b34":
                raise ContractError("CuTe Flash requires the qualified flash-attn-4 4.0.0b34")
            from flash_attn.cute import flash_attn_func
        except (PackageNotFoundError, ImportError, OSError) as exc:
            raise ContractError(
                "CuTe Flash requires an explicitly installed 4.0.0b34 CUDA build"
            ) from exc
        if not callable(flash_attn_func):
            raise ContractError("CuTe Flash kernel is unavailable")
        return
    if value != "sageattention2":
        raise ContractError("unknown Veda dense backend")
    try:
        installed = version("sageattention")
    except PackageNotFoundError as exc:
        raise ContractError(
            "SageAttention2 requires an explicitly installed 2.2.0 CUDA build"
        ) from exc
    if installed != "2.2.0":
        raise ContractError("SageAttention2 requires the qualified 2.2.0 CUDA build")
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
    except (ImportError, OSError) as exc:
        raise ContractError("SageAttention2 CUDA extension could not be loaded") from exc
    if not callable(sageattn_qk_int8_pv_fp8_cuda):
        raise ContractError("SageAttention2 CUDA kernel is unavailable")


def sage_dense(query: Any, key: Any, value: Any) -> Any:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    return sageattn_qk_int8_pv_fp8_cuda(
        query,
        key,
        value,
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran="per_warp",
        pv_accum_dtype="fp32+fp16",
        smooth_k=True,
        smooth_v=False,
    )


def cute_dense(query: Any, key: Any, value: Any) -> Any:
    """Pinned BF16 full-context H3 Flash on SM90; no approximate fallback."""
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
        raise ContractError("CuTe Flash requires CUDA SM90 BF16 H3 BSHD inputs")
    from flash_attn.cute import flash_attn_func

    result = flash_attn_func(query, key, value, causal=False, num_splits=1, return_lse=False)
    # The pinned public API returns (output, lse), including with LSE disabled.
    if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None:
        raise ContractError("CuTe Flash returned an incompatible public result tuple")
    output = result[0]
    if (
        output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ContractError("CuTe Flash returned an incompatible attention result")
    return output
