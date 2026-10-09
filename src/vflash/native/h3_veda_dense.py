"""Explicit optional dense blocks within Veda's mixed attention policy.

SageAttention changes precision and outputs; it is never a silent Flash fallback.
Only the ten dense blocks change. Sparse/protected Veda connections stay intact.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vflash.contracts import ContractError


def validate_dense_backend(value: str, *, attention_backend: str, capability: str) -> None:
    if value not in {"torch-flash", "sageattention2"}:
        raise ContractError("unknown Veda dense backend")
    if value == "sageattention2" and (
        attention_backend != "veda-triton" or capability != "12.0"
    ):
        raise ContractError("SageAttention2 dense blocks require explicit single-SM120 Veda")


def require_dense_dependencies(value: str) -> None:
    if value == "torch-flash":
        return
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
