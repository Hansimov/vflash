"""The explicit complete H3 pipeline and its local asset contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from vflash.pipeline.contracts import (
    PipelineAssets,
    PipelineProgress,
    VideoRequest,
    VideoResult,
)

__all__ = [
    "H3Pipeline",
    "PipelineAssets",
    "PipelineProgress",
    "PreparedPipelineAssets",
    "VideoRequest",
    "VideoResult",
    "load_prepared_pipeline_assets",
    "prepare_pipeline_assets",
]


def __getattr__(name: str) -> Any:
    # Asset/CPU helpers stay importable without importing model adapters or
    # establishing a dependency cycle through their residency primitives.
    module = {
        "H3Pipeline": "runtime",
        "PreparedPipelineAssets": "assets",
        "load_prepared_pipeline_assets": "assets",
        "prepare_pipeline_assets": "assets",
    }.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f"vflash.pipeline.{module}"), name)
    globals()[name] = value
    return value
