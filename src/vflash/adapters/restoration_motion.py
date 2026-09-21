"""Complete, bounded motion segmentation for restoration conditioning (not shot detection)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def restoration_segments(frames: Sequence[Any], *, maximum_segment_frames: int = 9):
    """Use local affine motion; uncertain matches start a new conditioning segment.

    Never truncate coverage when a blank or badly damaged frame has no features.
    The thresholds match the author's tiny restoration recipe at source resolution.
    Boundaries select conditioning frames; they are not semantic camera cuts.
    """
    import cv2
    import numpy as np
    from PIL import Image

    if type(maximum_segment_frames) is not int or maximum_segment_frames < 1:
        raise ValueError("maximum_segment_frames must be a positive integer")
    if not frames or any(
        not isinstance(frame, Image.Image)
        or frame.mode != "RGB"
        or frame.size != frames[0].size
        for frame in frames
    ):
        raise ValueError("motion segmentation requires equal-size RGB frames")
    segments = []
    start = 0
    previous = None
    for index, frame in enumerate(frames):
        current = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY)
        boundary = index - start >= maximum_segment_frames
        if previous is not None:
            points = cv2.goodFeaturesToTrack(
                previous, maxCorners=100, qualityLevel=0.3, minDistance=7
            )
            transform = None
            if points is not None and len(points) >= 3:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous,
                    current,
                    points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if tracked is not None and status is not None:
                    valid = status.reshape(-1).astype(bool)
                    if valid.sum() >= 3:
                        transform, _ = cv2.estimateAffine2D(points[valid], tracked[valid])
            if transform is None or not np.isfinite(transform).all():
                boundary = True
            else:
                angle = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
                scale = (
                    np.linalg.norm(transform[:2, 0]) + np.linalg.norm(transform[:2, 1])
                ) / 2
                boundary |= (
                    abs(transform[0, 2]) > 5
                    or abs(transform[1, 2]) > 5
                    or abs(angle) > 0.5
                    or abs(scale - 1) > 0.01
                )
        if boundary and index > start:
            segments.append((start, index))
            start = index
        previous = current
    segments.append((start, len(frames)))
    return tuple(segments)
