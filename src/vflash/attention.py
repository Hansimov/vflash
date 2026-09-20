"""CPU-only attention selection shared by every public execution entry point."""

from __future__ import annotations

from importlib.util import find_spec

from vflash.contracts import ContractError, ExecutionPlan

ATTENTION_BACKENDS = ("auto", "torch-flash", "sol-sm89")


def resolve_attention_backend(plan: ExecutionPlan, requested: str = "auto") -> str:
    if requested not in ATTENTION_BACKENDS:
        raise ContractError(f"unknown attention backend: {requested}")
    supported = (
        plan.target.compute_capability == "8.9"
        and plan.parallel_strategy == "single"
        and plan.peer_device is None
        and plan.profile.id in {"i2va-base16-bf16-sm89", "fl2va-base16-bf16-sm89"}
        and plan.profile.nfe == 16
        and plan.profile.adapter is None
    )
    if requested == "sol-sm89" and not supported:
        raise ContractError("Sol requires single-SM89 official Base16; use torch-flash here")
    return ("sol-sm89" if supported else "torch-flash") if requested == "auto" else requested


def require_attention_dependencies(backend: str) -> None:
    if backend == "sol-sm89" and find_spec("sol_attn") is None:
        raise ContractError(
            "Sol attention is selected but sol_attn is not installed; run "
            "python -m vflash.install_sol or explicitly select torch-flash. "
            "No dense fallback is performed."
        )
