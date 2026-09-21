"""Retain the exact intermediate text state consumed by pinned H3 encoding."""

from __future__ import annotations

from typing import Any

from vflash.contracts import ContractError


def retain_h3_text_encoder_prefix(encoder: Any) -> None:
    """Drop unused decoder layers before installing any offloading hooks.

    H3 consumes ``hidden_states[50]``. Keep 51 decoder layers, not 50: the
    Transformers final entry is normalized, whereas H3 needs the raw state
    entering layer 50. This does not change weights, precision, vision encoding,
    or the first 50 layers. Original checkpoint loading remains unchanged.
    """
    language = encoder.model.language_model
    count = len(language.layers)
    if (
        count < 51
        or encoder.config.text_config.num_hidden_layers != count
        or language.config.num_hidden_layers != count
    ):
        raise ContractError("H3 requires a coherent encoder with at least 51 layers")
    language.layers = language.layers[:51]
    encoder.config.text_config.num_hidden_layers = 51
    language.config.num_hidden_layers = 51
