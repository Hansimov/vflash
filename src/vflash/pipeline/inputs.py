"""Ordered local conditioning, independent of a recipe's resource envelope.

These are model inputs, not an application's analysis attachments, identities,
uploads or workflow graph. Accepting a kind here does not qualify a runtime to
encode it. Each request boundary applies its own explicit reference limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vflash.contracts import ContractError


@dataclass(frozen=True)
class MediaReference:
    kind: Literal["image", "video", "audio"]
    path: Path

    def __post_init__(self):
        if self.kind not in ("image", "video", "audio") or not isinstance(self.path, Path):
            raise ContractError(
                "Reference media requires a known kind and a local pathlib.Path"
            )


@dataclass(frozen=True)
class ReferenceLimits:
    images: int
    videos: int
    audio: int
    total: int
    images_with_video: int | None = None

    def __post_init__(self):
        values = (self.images, self.videos, self.audio, self.total)
        if any(type(value) is not int or value < 0 for value in values):
            raise ContractError("Reference limits must be nonnegative integers")
        if self.images_with_video is not None and (
            type(self.images_with_video) is not int
            or not 0 <= self.images_with_video <= self.images
        ):
            raise ContractError("Mixed image limit must fit the image allocation")


@dataclass(frozen=True)
class ConditioningInputs:
    first_frame: Path | None = None
    last_frame: Path | None = None
    references: tuple[MediaReference, ...] = ()

    def __post_init__(self):
        if any(
            p is not None and not isinstance(p, Path)
            for p in (self.first_frame, self.last_frame)
        ):
            raise ContractError("Temporal anchors must be local pathlib.Paths")
        if not isinstance(self.references, tuple) or any(
            not isinstance(item, MediaReference) for item in self.references
        ):
            raise ContractError(
                "Content references must be an ordered tuple of MediaReferences"
            )
        if self.references and (self.first_frame is not None or self.last_frame is not None):
            raise ContractError("Provide temporal keyframes or content references, not both")

    @property
    def mode(self) -> str:
        if self.first_frame is not None and self.last_frame is not None:
            return "fl2va"
        if self.first_frame is not None:
            return "i2va"
        if self.last_frame is not None:
            return "l2va"
        return "ref2va" if self.references else "t2va"

    @property
    def ordered_media(self) -> tuple[MediaReference, ...]:
        return self.references or tuple(
            MediaReference("image", path)
            for path in (self.first_frame, self.last_frame)
            if path is not None
        )

    def validate_references(self, limits: ReferenceLimits) -> None:
        counts = {
            kind: sum(item.kind == kind for item in self.references)
            for kind in ("image", "video", "audio")
        }
        if (
            counts["image"] > limits.images
            or counts["video"] > limits.videos
            or counts["audio"] > limits.audio
            or len(self.references) > limits.total
            or (
                counts["video"]
                and limits.images_with_video is not None
                and counts["image"] > limits.images_with_video
            )
        ):
            raise ContractError("References exceed this pipeline's explicit media allocation")
        if self.references and counts["image"] + counts["video"] == 0:
            raise ContractError("H3 audio references require at least one visual reference")

    @property
    def labeled_media(self) -> tuple[tuple[str, MediaReference], ...]:
        counts = dict.fromkeys(("image", "video", "audio"), 0)
        names = {"image": "Picture", "video": "Video", "audio": "Audio"}
        result = []
        for item in self.ordered_media:
            counts[item.kind] += 1
            result.append((f"<{names[item.kind]} {counts[item.kind]}>", item))
        return tuple(result)
