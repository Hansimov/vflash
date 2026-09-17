"""Explicit delivery-time restoration of authoritative H3 keyframes."""

from __future__ import annotations

from typing import Any

from vflash.media.encoding import MediaError

DECODED_KEYFRAME_DELIVERY_PROFILE = "decoded"
EXACT_KEYFRAME_DELIVERY_PROFILE = "exact-v1"
EXACT_KEYFRAME_TRANSITION_FRAMES = 4
KEYFRAME_DELIVERY_PROFILES = frozenset(
    {DECODED_KEYFRAME_DELIVERY_PROFILE, EXACT_KEYFRAME_DELIVERY_PROFILE}
)


def apply_exact_keyframe_delivery(
    video: Any,
    *,
    width: int,
    height: int,
    delivery_frames: int,
    first_frame: Any | None = None,
    last_frame: Any | None = None,
) -> dict[str, Any]:
    """Replace delivered endpoints with owned RGB inputs using the H3 stretch policy.

    H3's official keyframe conditioner stretches each temporal anchor to the
    requested canvas. This delivery step mirrors that geometry with high-quality
    LANCZOS resampling after VAE decode. The endpoint is exact and the next
    three frames feather the authoritative image into the decoded motion. This
    bounded 125 ms transition avoids a one-frame sharpness flash at 24 fps.
    """

    import numpy as np
    from PIL import Image

    if getattr(video, "ndim", None) != 5 or tuple(video.shape[:2]) != (1, 3):
        raise MediaError("keyframe delivery requires video shaped [1,3,frames,height,width]")
    if video.device.type != "cpu":
        raise MediaError("keyframe delivery requires an owned CPU video tensor")
    if tuple(video.shape[-2:]) != (height, width):
        raise MediaError("keyframe delivery canvas differs from the decoded video")
    if not 1 <= delivery_frames <= int(video.shape[2]):
        raise MediaError("keyframe delivery range differs from the decoded video")
    if first_frame is None and last_frame is None:
        raise MediaError("exact keyframe delivery requires a first or last frame")

    def normalized(image: Any, role: str) -> Any:
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise MediaError(f"the {role} must be an owned RGB PIL image")
        resized = image
        if image.size != (width, height):
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
        try:
            pixels = np.asarray(resized, dtype=np.uint8).copy()
            return video.new_tensor(pixels).permute(2, 0, 1).div_(255)
        finally:
            if resized is not image:
                resized.close()

    exact: list[int] = []
    transitions: list[int] = []

    def feather(anchor: Any, indices: list[int]) -> None:
        for offset, index in enumerate(indices):
            strength = (EXACT_KEYFRAME_TRANSITION_FRAMES - offset) / (
                EXACT_KEYFRAME_TRANSITION_FRAMES
            )
            if strength == 1:
                video[0, :, index].copy_(anchor)
                exact.append(index)
            else:
                video[0, :, index].mul_(1 - strength).add_(anchor, alpha=strength)
                transitions.append(index)

    if first_frame is not None:
        count = min(EXACT_KEYFRAME_TRANSITION_FRAMES, delivery_frames)
        feather(normalized(first_frame, "first frame"), list(range(count)))
    if last_frame is not None:
        count = min(EXACT_KEYFRAME_TRANSITION_FRAMES, delivery_frames)
        feather(
            normalized(last_frame, "last frame"),
            list(range(delivery_frames - 1, delivery_frames - count - 1, -1)),
        )
    return {
        "profile": EXACT_KEYFRAME_DELIVERY_PROFILE,
        "geometry": "official-stretch-lanczos-rgb-v1",
        "transition": "four-frame-linear-feather-v1",
        "exact_frame_indices": sorted(set(exact)),
        "transition_frame_indices": sorted(set(transitions)),
    }
