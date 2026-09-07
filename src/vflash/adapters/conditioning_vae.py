"""Minimal official VAE components for visual-only H3 conditioning."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class H3ConditioningVAEError(RuntimeError):
    """The encoder-only conditioning checkpoint contract is invalid."""


@dataclass(frozen=True)
class H3ConditioningVAEComponents:
    video_vae: Any
    audio_vae: Any
    checkpoint_bytes: int
    checkpoint_tensor_count: int
    load_seconds: float


def _read_weight_map(component_path: Path) -> dict[str, str]:
    index_path = component_path / "diffusion_pytorch_model.safetensors.index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H3ConditioningVAEError("H3 video VAE checkpoint index is invalid") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str)
        and name
        and isinstance(shard, str)
        and Path(shard).name == shard
        and shard.endswith(".safetensors")
        for name, shard in weight_map.items()
    ):
        raise H3ConditioningVAEError("H3 video VAE checkpoint weight map is invalid")
    return weight_map


def _open_safetensors(path: Path) -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - pinned inference-image contract
        raise H3ConditioningVAEError("H3 conditioning VAE requires safetensors") from exc
    return safe_open(path, framework="pt", device="cpu")


def load_h3_image_conditioning_vae_components(
    *,
    video_component_path: Path,
    audio_component_path: Path,
    video_class: Any,
    audio_class: Any,
    device: Any,
    torch_module: Any,
    init_empty_weights: Callable[[], AbstractContextManager[Any]],
) -> H3ConditioningVAEComponents:
    """Load only the modules reached by image or silent-video reference encoding.

    Visual-only Ref2VA calls ``video_vae.encode`` and reads normalization values
    from ``audio_vae.config``. The multi-gigabyte video decoder and every audio
    VAE parameter are downstream of the block-zero capture boundary.
    """

    started = time.monotonic()
    video_path = video_component_path.resolve(strict=True)
    audio_path = audio_component_path.resolve(strict=True)
    with init_empty_weights():
        video_vae = video_class.from_config(video_class.load_config(video_path))
        audio_vae = audio_class.from_config(audio_class.load_config(audio_path))

    weight_map = _read_weight_map(video_path)
    required_names = {
        name
        for name in video_vae.state_dict()
        if name.startswith("encoder.") or name.startswith("quant_conv.")
    }
    if not required_names or not required_names.issubset(weight_map):
        raise H3ConditioningVAEError("H3 video VAE encoder checkpoint is incomplete")

    names_by_shard: dict[str, list[str]] = {}
    for name in sorted(required_names):
        names_by_shard.setdefault(weight_map[name], []).append(name)
    tensors: dict[str, Any] = {}
    checkpoint_bytes = 0
    for shard, names in sorted(names_by_shard.items()):
        with _open_safetensors(video_path / shard) as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                tensors[name] = tensor
                checkpoint_bytes += tensor.numel() * tensor.element_size()

    incompatible = video_vae.load_state_dict(tensors, strict=False, assign=True)
    if incompatible.unexpected_keys or any(
        not (name.startswith("decoder.") or name.startswith("post_quant_conv."))
        for name in incompatible.missing_keys
    ):
        raise H3ConditioningVAEError(
            "H3 encoder-only VAE assignment changed its module boundary"
        )
    remaining_meta = [
        name
        for name, parameter in video_vae.named_parameters()
        if (name.startswith("encoder.") or name.startswith("quant_conv.")) and parameter.is_meta
    ]
    if remaining_meta:
        raise H3ConditioningVAEError("H3 video VAE encoder retained meta parameters")

    video_vae.encoder.eval().requires_grad_(False).to(device=device)
    video_vae.quant_conv.eval().requires_grad_(False).to(device=device)
    audio_vae.eval().requires_grad_(False)
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)
    return H3ConditioningVAEComponents(
        video_vae=video_vae,
        audio_vae=audio_vae,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_tensor_count=len(tensors),
        load_seconds=time.monotonic() - started,
    )


__all__ = [
    "H3ConditioningVAEComponents",
    "H3ConditioningVAEError",
    "load_h3_image_conditioning_vae_components",
]
