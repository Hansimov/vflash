"""Original adapter layouts, independent of activation-hook lifetime and device.

The Akatz layout describes its complete rank-16 V1.5 Ref adapter, not a
qualified character-replacement pipeline. Sampling and masking are separate.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdapterLayout:
    source_pairs: int
    target_pairs: int
    default_suffix: bool = False
    rank: int | None = None

    @property
    def suffixes(self):
        middle = ".default" if self.default_suffix else ""
        return tuple(f".lora_{part}{middle}.weight" for part in ("A", "B"))


LAYOUTS = {
    "lightx": AdapterLayout(312, 312, True),
    "larry": AdapterLayout(259, 363),
    "pika": AdapterLayout(312, 312, True, 128),
    "akatz-v1.5": AdapterLayout(208, 312, rank=16),
}


def adapter_names(keys, kind):
    layout = LAYOUTS[kind]
    suffix, other = layout.suffixes
    names = sorted(key[: -len(suffix)] for key in keys if key.endswith(suffix))
    if keys != {name + part for name in names for part in (suffix, other)}:
        raise ValueError("Adapter must contain complete A/B pairs without extra tensors")
    if len(names) != layout.source_pairs:
        raise ValueError("Adapter tensor coverage differs from its declared family")
    if kind == "akatz-v1.5":
        blocks = [f"blocks.{i}" for i in range(50)] + [
            f"token_refiner.blocks.{i}" for i in range(2)
        ]
        expected = {
            f"diffusion_model.{block}.{projection}"
            for block in blocks
            for projection in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
        }
        if set(names) != expected:
            raise ValueError("Akatz V1.5 requires its original 52-block projection coverage")
    return names


def adapter_scale(kind, metadata, rank, manifest=None):
    if kind == "lightx":
        scale = float((metadata or {}).get("alpha", 8.0)) / rank
    elif kind == "pika":
        scale = manifest["alpha"] / manifest["rank"]
    else:
        # Original Larry and Akatz files have no per-projection alpha tensors;
        # ComfyUI therefore applies their A/B update at unit scale.
        scale = 1.0
    if not math.isfinite(scale):
        raise ValueError("Adapter scale must be finite")
    return scale


def converted(name, down, up, kind):
    if kind not in LAYOUTS:
        raise ValueError("An explicit supported adapter family is required")
    if down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or up.shape[1] != down.shape[0]:
        raise ValueError("Adapter projection requires compatible nonempty A/B matrices")
    rank = LAYOUTS[kind].rank
    if rank is not None and down.shape[0] != rank:
        raise ValueError(f"{kind} requires every original rank-{rank} tensor pair")
    if kind in {"lightx", "pika"}:
        return [(name, down, up)]
    if kind == "akatz-v1.5":
        if not name.startswith("diffusion_model."):
            raise ValueError("Akatz V1.5 requires its original diffusion_model prefix")
        name = name.removeprefix("diffusion_model.")
    if name.startswith("blocks."):
        name = "transformer_blocks." + name.removeprefix("blocks.")
    else:
        name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
    name = name.replace("final_layer.adaln_proj.linear", "norm_out.linear")
    if name.endswith(".attn.qkv_proj"):
        if up.shape[0] % 3:
            raise ValueError("Fused QKV adapter must have three equal output sections")
        return [
            (name.replace(".attn.qkv_proj", ".attn.to_" + k), down, b)
            for k, b in zip(("q", "k", "v"), up.chunk(3, dim=0), strict=True)
        ]
    if name.endswith(".mlp.fc1"):
        if up.shape[0] % 2:
            raise ValueError("Fused SwiGLU adapter must have equal gate/value sections")
        gate, value = up.chunk(2, dim=0)
        return [
            (name.replace(".mlp.fc1", ".ff.net.0.proj"), down, torch.cat((value, gate), dim=0))
        ]
    return [
        (
            name.replace(".attn.out_proj", ".attn.to_out.0").replace(".mlp.fc2", ".ff.net.2"),
            down,
            up,
        )
    ]
