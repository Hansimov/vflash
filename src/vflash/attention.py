"""CPU-only attention selection shared by every public execution entry point."""

from __future__ import annotations

from importlib.util import find_spec

from vflash.contracts import ContractError, ExecutionPlan

ATTENTION_BACKENDS = ("auto", "torch-flash", "sol-sm89", "veda-sm89")


def resolve_attention_backend(plan: ExecutionPlan, requested: str = "auto") -> str:
    if requested not in ATTENTION_BACKENDS:
        raise ContractError(f"unknown attention backend: {requested}")
    single_sm89 = (
        plan.target.compute_capability == "8.9"
        and plan.parallel_strategy == "single"
        and plan.peer_device is None
    )
    base16 = (
        plan.profile.id in {"i2va-base16-bf16-sm89", "fl2va-base16-bf16-sm89"}
        and plan.profile.nfe == 16
        and plan.profile.adapter is None
    )
    original_v01 = (
        plan.profile.id in {"i2va-turbo4-v01-544-exact-sm89", "fl2va-turbo4-v01-544-exact-sm89"}
        and plan.profile.nfe == 4
        and plan.profile.adapter == "lightx2v/Minimax-h3-Turbo"
    )
    if requested == "sol-sm89" and not (single_sm89 and (base16 or original_v01)):
        raise ContractError(
            "Sol requires single-SM89 official Base16 or original LightX v0.1 keyframes; "
            "use torch-flash here"
        )
    if requested == "veda-sm89" and not (single_sm89 and original_v01):
        raise ContractError("Veda requires original LightX v0.1 on one SM89")
    # Few-step approximation is opt-in, independently of the Base16 default.
    return (
        ("sol-sm89" if single_sm89 and base16 else "torch-flash")
        if requested == "auto"
        else requested
    )


def require_attention_dependencies(backend: str) -> None:
    if backend == "sol-sm89" and find_spec("sol_attn") is None:
        raise ContractError(
            "Sol attention is selected but sol_attn is not installed; run "
            "python -m vflash.install_sol or explicitly select torch-flash. "
            "No dense fallback is performed."
        )

    if backend == "veda-sm89":
        try:
            from veda_comfy._vflash_pin import REVISION

            from vflash.install_veda import REVISION as expected
        except ImportError as exc:
            raise ContractError(
                "Install the pinned Veda core: python -m vflash.install_veda"
            ) from exc
        if expected != REVISION or find_spec("triton") is None:
            raise ContractError(
                "Veda requires the pinned core and Triton; no fallback is performed"
            )
