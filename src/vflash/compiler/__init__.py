"""Explicit preparation and compilation of fixed official H3 model assets."""

from .assets import PreparedRef4Weights, load_prepared_ref4_weights, prepare_ref4_weights

__all__ = [
    "PreparedRef4Weights",
    "compile_ref4_assets",
    "load_prepared_ref4_weights",
    "prepare_ref4_weights",
]


def __getattr__(name):
    if name == "compile_ref4_assets":
        from .ref4 import compile_ref4_assets

        return compile_ref4_assets
    raise AttributeError(name)
