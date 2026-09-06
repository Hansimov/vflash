"""Compatibility entrypoints for the original Ref4 SM89 weights compiler."""

from vflash.contracts import ContractError
from vflash.model_assets import DEFAULT_MODEL_PROFILE

from .h3 import CompiledAssets as CompiledRef4Assets
from .h3 import compile_assets, validate_weight_headers

__all__ = ["CompiledRef4Assets", "compile_ref4_assets", "validate_ref4_weight_headers"]


def _require_ref4(prepared):
    if prepared.profile_id != DEFAULT_MODEL_PROFILE:
        raise ContractError("the Ref4 SM89 entrypoint requires its matching weights receipt")


def validate_ref4_weight_headers(prepared):
    _require_ref4(prepared)
    return validate_weight_headers(prepared)


def compile_ref4_assets(prepared, destination, *, device, progress=None):
    _require_ref4(prepared)
    return compile_assets(prepared, destination, device=device, progress=progress)
