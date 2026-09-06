"""Local, owned reference images and the verified H3 matching geometry."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError


@dataclass(frozen=True)
class DecodedReference:
    image: Any
    size_bytes: int
    sha256: str

    def close(self) -> None:
        self.image.close()


def read_reference(path: Path) -> DecodedReference:
    from PIL import Image, ImageOps, UnidentifiedImageError

    with path.open("rb") as handle:
        data = handle.read(32 * 1024 * 1024 + 1)
    if not data or len(data) > 32 * 1024 * 1024:
        raise ContractError("a reference image must contain at most 32 MiB")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ContractError("reference images must be PNG, JPEG or WebP")
            if source.width * source.height > 16_777_216:
                raise ContractError("the reference image exceeds 16 megapixels")
            if not 0.25 <= source.width / source.height <= 4:
                raise ContractError("the reference aspect ratio must be between 1:4 and 4:1")
            # Match Diffusers load_image orientation and color normalization,
            # while binding the exact input bytes only once before CUDA work.
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ContractError("the reference image cannot be decoded") from exc
    return DecodedReference(image, len(data), hashlib.sha256(data).hexdigest())


def resolve_match_reference_image_size(
    width: int,
    height: int,
    *,
    target_width: int,
    target_height: int,
    dimension_multiple: int = 32,
) -> tuple[int, int]:
    """Resolve H3's target-area, downscale-only reference geometry.

    The returned tuple is ``(height, width)``. This mirrors the public
    LightX2V ``match`` contract introduced by upstream commit ``5169278f``:
    preserve aspect ratio, never upscale, cap each reference by the target
    canvas area, and round both dimensions to the H3/VAE multiple.
    """
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError(
            f"an H3 reference image must be positive and within 1:4..4:1, got {width}x{height}"
        )
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"an H3 target canvas must be positive, got {target_width}x{target_height}"
        )
    if dimension_multiple <= 0:
        raise ValueError("H3 reference image dimension multiple must be positive")
    scale = min(1.0, math.sqrt((target_width * target_height) / (width * height)))
    return (
        max(
            dimension_multiple,
            round(height * scale / dimension_multiple) * dimension_multiple,
        ),
        max(
            dimension_multiple,
            round(width * scale / dimension_multiple) * dimension_multiple,
        ),
    )


def install_match_reference_setup_block(pipe: Any) -> Any:
    """Use the verified target-area reference policy without patching Diffusers."""
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    from diffusers.modular_pipelines.minimax_h3.before_encoder import (
        MiniMaxH3Ref2VASetupStep,
    )
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        align_num_frames,
    )
    from PIL import Image

    class MatchReferenceSetupStep(MiniMaxH3Ref2VASetupStep):
        resize_mode = "match"
        last_metadata: dict[str, Any] | None = None

        def __call__(self, components: Any, state: Any) -> Any:
            block_state = self.get_block_state(state)
            if block_state.height is None or block_state.width is None:
                raise ValueError("match reference sizing requires an explicit target canvas")
            multiple = int(components.canvas_multiple)
            if block_state.height % multiple or block_state.width % multiple:
                raise ValueError(
                    f"height and width must be multiples of {multiple}, got "
                    f"{block_state.height}x{block_state.width}"
                )
            if not block_state.references:
                raise ValueError("Ref2VA match sizing requires at least one reference")
            if len(block_state.references) > self.max_images:
                raise ValueError(
                    f"MiniMax-H3 accepts at most {self.max_images} image references, "
                    f"got {len(block_state.references)}"
                )

            aligned_num_frames = align_num_frames(
                block_state.num_frames,
                components.vae_frames_per_chunk,
                components.vae_latents_per_chunk,
            )
            duration = aligned_num_frames / components.fps
            if not components.min_duration <= duration <= components.max_duration:
                raise ValueError(
                    f"MiniMax-H3 generates between {components.min_duration} and "
                    f"{components.max_duration} seconds, got {duration:.6f}"
                )
            block_state.num_frames = aligned_num_frames

            normalized = []
            resolved_sizes: list[dict[str, int]] = []
            for entry in block_state.references:
                if entry.kind != "image" or not isinstance(entry.image, Image.Image):
                    raise ValueError(
                        "native Diffusers match sizing accepts decoded image references only"
                    )
                source_width, source_height = entry.image.size
                target_height, target_width = resolve_match_reference_image_size(
                    source_width,
                    source_height,
                    target_width=block_state.width,
                    target_height=block_state.height,
                    dimension_multiple=multiple,
                )
                image = entry.image
                if image.size != (target_width, target_height):
                    image = image.resize(
                        (target_width, target_height),
                        Image.Resampling.LANCZOS,
                    )
                normalized.append(MiniMaxH3ImageReference(image=image))
                resolved_sizes.append(
                    {
                        "source_height": source_height,
                        "source_width": source_width,
                        "height": target_height,
                        "width": target_width,
                    }
                )
            block_state.normalized_references = normalized
            self.last_metadata = {
                "policy": "match",
                "contract": "target-area-downscale-only-round-to-32-v3",
                "resolved_sizes": resolved_sizes,
            }
            self.set_block_state(state, block_state)
            return components, state

    block = MatchReferenceSetupStep()
    before_encode = pipe._blocks.sub_blocks.get("before_encode")
    if before_encode is None or before_encode.__class__.__name__ != (
        "MiniMaxH3Ref2VASetupStep"
    ):
        raise RuntimeError("pinned Diffusers Ref2VA workflow block layout changed")
    pipe._blocks.sub_blocks["before_encode"] = block
    return block
