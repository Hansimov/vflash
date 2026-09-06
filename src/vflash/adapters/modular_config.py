"""Rebind the pinned official component index to explicit local assets."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def local_modular_config(
    model_path: Path,
    *,
    transformer_component: str = "transformer_ref",
) -> dict[str, Any]:
    """Load the official index while rebinding its Hub component specs locally.

    MiniMax-H3's public ``model_index.json`` already uses the three-element
    Modular Diffusers component format. Older local snapshots may not carry the
    duplicate ``modular_model_index.json`` filename added on the Hub, so the
    same official object is passed in memory and only its repository locations
    are rebound to the verified read-only model view.
    """
    index_path = model_path / "modular_model_index.json"
    if not index_path.is_file():
        index_path = model_path / "model_index.json"
    try:
        config = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"MiniMax-H3 modular model index is missing or invalid: {index_path}"
        ) from exc
    if not isinstance(config, dict):
        raise RuntimeError("MiniMax-H3 modular model index must be a JSON object")
    if (
        config.get("_class_name") != "MiniMaxH3ModularPipeline"
        or config.get("_blocks_class_name") != "MiniMaxH3Blocks"
    ):
        raise RuntimeError("MiniMax-H3 modular model index has an unexpected pipeline contract")
    if transformer_component not in {"transformer", "transformer_ref"}:
        raise ValueError("unsupported H3 transformer component")
    rebound = copy.deepcopy(config)
    expected_components = {
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
        transformer_component,
        "scheduler",
        "audio_scheduler",
    }
    missing = expected_components - set(rebound)
    if missing:
        raise RuntimeError(
            f"MiniMax-H3 modular model index lacks components: {sorted(missing)}"
        )
    for name in expected_components:
        value = rebound[name]
        if not isinstance(value, list) or len(value) != 3 or not isinstance(value[2], dict):
            raise RuntimeError(f"MiniMax-H3 component spec is invalid: {name}")
        value[2]["pretrained_model_name_or_path"] = str(model_path)
        value[2]["subfolder"] = name
        value[2]["revision"] = None
    return rebound
