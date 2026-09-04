"""Fixed architecture and BF16 artifact targets accepted by the native runtime."""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3


class H3ArtifactContractError(ValueError):
    """An artifact target is outside the supported BF16 runtime contract."""


@dataclass(frozen=True)
class H3Spec:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    attention_head_dim: int
    ffn_dim: int
    time_embed_dim: int
    in_channels: int
    audio_in_channels: int
    patch_size: tuple[int, int, int]


@dataclass(frozen=True)
class H3ArtifactTarget:
    target_id: str
    compute_capability: str
    nominal_device_bytes: int
    arena_reserve_bytes: int
    attention_weight_bits: int
    ffn_weight_bits: int
    ffn_group_size: int | None
    attention_activation: str
    ffn_activation: str
    fallback: str


H3_ARTIFACT_TARGETS = {
    "rtx4090-48g-sm89-bf16-resident": H3ArtifactTarget(
        target_id="rtx4090-48g-sm89-bf16-resident",
        compute_capability="sm89",
        nominal_device_bytes=48 * GIB,
        arena_reserve_bytes=8 * GIB,
        attention_weight_bits=16,
        ffn_weight_bits=16,
        ffn_group_size=None,
        attention_activation="bfloat16",
        ffn_activation="bfloat16",
        fallback="fail-closed-no-streaming",
    ),
    "rtx3080-20g-sm86-bf16-block-ring": H3ArtifactTarget(
        target_id="rtx3080-20g-sm86-bf16-block-ring",
        compute_capability="sm86",
        nominal_device_bytes=20 * GIB,
        arena_reserve_bytes=7 * GIB,
        attention_weight_bits=16,
        ffn_weight_bits=16,
        ffn_group_size=None,
        attention_activation="bfloat16",
        ffn_activation="bfloat16",
        fallback="event-driven-block-ring",
    ),
}


def resolve_h3_artifact_target(target_id: str) -> H3ArtifactTarget:
    try:
        return H3_ARTIFACT_TARGETS[target_id]
    except KeyError as exc:
        raise H3ArtifactContractError(f"unknown H3 artifact target: {target_id}") from exc
