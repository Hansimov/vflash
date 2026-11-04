"""Crop the decoded RGB canvas before its only video encode."""

from __future__ import annotations

from vflash.media.encoding import MediaError


def crop_geometry(width: int, height: int, crop):
    if crop is None:
        return None
    if (
        not isinstance(crop, tuple)
        or len(crop) != 4
        or any(type(n) is not int or n < 0 or n % 2 for n in crop)
    ):
        raise MediaError("video_crop requires four nonnegative even pixel offsets")
    left, top, right, bottom = crop
    if width - left - right < 2 or height - top - bottom < 2:
        raise MediaError("video_crop must retain a nonempty even canvas")
    return dict(
        source_width=width,
        source_height=height,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        width=width - left - right,
        height=height - top - bottom,
    )


def crop_rgb(video, geometry):
    if geometry is None:
        return video
    if tuple(video.shape[-2:]) != (geometry["source_height"], geometry["source_width"]):
        raise MediaError("decoded RGB dimensions differ from the requested crop")
    return video[
        ...,
        geometry["top"] : geometry["top"] + geometry["height"],
        geometry["left"] : geometry["left"] + geometry["width"],
    ]
