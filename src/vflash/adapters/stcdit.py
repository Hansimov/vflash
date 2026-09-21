"""Opt-in, same-size STCDiT restoration over a caller-owned compatible pipeline.

The external model/runtime is not bundled or downloaded. H3 generation, audio,
source assets, scheduling and billing are deliberately outside this adapter.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

STCDIT_SOURCE_REVISION = "7c4be6e2774b1bdf51658d1e495a0a3a3ace3772"
STCDIT_WEIGHT_REVISION = "3bc4dfb4e720bcc39d7313e62fd77dca4cb17a25"


@dataclass(frozen=True)
class RestorationResult:
    frames: tuple[Any, ...]
    elapsed_seconds: float
    frame_count: int
    width: int
    height: int
    segments: tuple[tuple[int, int], ...]
    seed: int
    fps: int = 24
    profile: str = "stcdit-tiny-same-size-v1"


def validate_segments(
    segments: Sequence[tuple[int, int]], frame_count: int
) -> tuple[tuple[int, int], ...]:
    """Segments are ordered, half-open, disjoint, and cover every input frame."""
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    result = []
    cursor = 0
    for segment in segments:
        if not isinstance(segment, (tuple, list)) or len(segment) != 2:
            raise ValueError("each segment must contain start and end")
        start, end = segment
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or not start < end <= frame_count
        ):
            raise ValueError("segments must cover the input without gaps or overlaps")
        result.append((start, end))
        cursor = end
    if cursor != frame_count:
        raise ValueError("segments must cover the complete input")
    return tuple(result)


class StcditTinyRestorer:
    """Serial adapter; the caller owns model loading, CUDA lifetime and inference mode.

    Supply a BF16 STCDiT-tiny pipeline exposing ``test_tlc_seg``. The qualified
    exploratory recipe uses ten evaluations, CFG=1 and shift=5 with no rescaling.
    Passing this contract is not a quality approval. Always retain the source.
    """

    def __init__(self, pipeline: Any) -> None:
        if not callable(getattr(pipeline, "test_tlc_seg", None)):
            raise TypeError("a compatible STCDiT-tiny pipeline is required")
        self.pipeline = pipeline

    def restore(
        self,
        frames: Sequence[Any],
        *,
        caption: str,
        segments: Sequence[tuple[int, int]],
        seed: int = 42,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> RestorationResult:
        from PIL import Image

        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("an observation caption is required")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if not 1 <= len(frames) <= 240:
            raise ValueError("restore between one and 240 source-clock frames per request")
        spans = validate_segments(segments, len(frames))
        if any(not isinstance(frame, Image.Image) or frame.mode != "RGB" for frame in frames):
            raise ValueError("input frames must be RGB PIL images")
        width, height = frames[0].size
        if (
            width < 32
            or height < 32
            or width % 32
            or height % 32
            or width * height > 1024**2
            or any(frame.size != (width, height) for frame in frames)
        ):
            raise ValueError("frames must share a 32-aligned canvas of at most one mebipixel")
        owned = [frame.copy() for frame in frames]
        result = []
        started = time.monotonic()
        try:
            options = {}
            if on_progress is not None:
                on_progress("encoding", 0, 10)

                def steps(timesteps: Any) -> Any:
                    for index, timestep in enumerate(timesteps):
                        on_progress("denoising", index, 10)
                        yield timestep
                    on_progress("decoding", 10, 10)

                options["progress_bar_cmd"] = steps
            generated = self.pipeline.test_tlc_seg(
                prompt=caption,
                input_lq=owned,
                segments=list(spans),
                height=height,
                width=width,
                num_frames=len(owned),
                num_inference_steps=10,
                cfg_scale=1.0,
                sigma_shift=5.0,
                tiled=True,
                tile_kernel=(21, min(128, height // 8), min(128, width // 8)),
                seed=seed,
                **options,
            )
            if len(generated) != len(frames):
                raise ValueError("restoration changed the source frame count")
            for frame in generated:
                if (
                    not isinstance(frame, Image.Image)
                    or frame.mode != "RGB"
                    or frame.size != (width, height)
                ):
                    raise ValueError("restoration changed the RGB canvas")
                result.append(frame.copy())
        except BaseException:
            for frame in result:
                frame.close()
            raise
        finally:
            for frame in owned:
                frame.close()
        return RestorationResult(
            tuple(result), time.monotonic() - started, len(frames), width, height, spans, seed
        )
