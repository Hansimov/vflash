"""CPU geometry of the pinned official visual-only H3 video setup.

The source RGB remains at its display size. Diffusers d035dcd7 performs
the actual Lanczos resize once, after this budget calculation.
"""

from __future__ import annotations

from vflash.contracts import ContractError


def reference_video_geometry(width: int, height: int, frames: int) -> dict[str, int]:
    """Return the normalized canvas and complete temporal-VAE prefix budget."""
    if (
        any(type(value) is not int for value in (width, height, frames))
        or min(width, height) <= 0
        or width * height > 928 * 512
        or not 0.25 <= width / height <= 4
        or not 48 <= frames <= 120
    ):
        raise ContractError(
            "a video reference requires 48-120 CFR frames, at most 928x512 source pixels, "
            "and an aspect ratio between 1:4 and 4:1"
        )
    ratio = width / height
    w, h = (768 * ratio, 768.0) if ratio >= 1 else (768.0, 768 / ratio)
    scale = min(1.0, (768 * 1344 / (w * h)) ** 0.5)
    w, h = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
    chunks = (frames - 5) // 17
    latent_frames = chunks * 5 + 2
    rows = latent_frames * (w // 32) * (h // 32)
    if max(w, h) > 1376 or w * h > 1376 * 768 or rows > 33024:
        raise ContractError("the official video canvas or temporal prefix exceeds its budget")
    return {
        "width": w,
        "height": h,
        "vae_input_frames": chunks * 17 + 5,
        "vae_latent_frames": latent_frames,
        "condition_video_rows": rows,
    }
