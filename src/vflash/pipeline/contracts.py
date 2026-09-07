"""Small explicit contracts for the complete H3 pipelines."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError
from vflash.model_assets import DEFAULT_MODEL_PROFILE
from vflash.model_assets import DIFFUSERS_REVISION as DIFFUSERS_REVISION
from vflash.model_assets import MODEL_REVISION as MODEL_REVISION

PIPELINE_PROFILE = DEFAULT_MODEL_PROFILE


@dataclass(frozen=True)
class PipelineAssets:
    """Local immutable assets; no component path is inferred from environment variables."""

    model_directory: Path
    adapter_path: Path
    decoder_directory: Path
    artifact: Path
    schedule_overlay: Path
    auxiliary_tensor: Path

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not isinstance(getattr(self, name), Path):
                raise ContractError(f"{name} must be a pathlib.Path")

    def to_mapping(self) -> dict[str, str]:
        return {name: str(value.resolve(strict=True)) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value: Any) -> PipelineAssets:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ContractError("pipeline asset fields do not match the supported schema")
        if any(not isinstance(path, str) or not path for path in value.values()):
            raise ContractError("pipeline asset paths must be nonempty strings")
        return cls(**{name: Path(path) for name, path in value.items()})

    @classmethod
    def from_json(cls, path: Path) -> PipelineAssets:
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class VideoRequest:
    """A text-, image- or video-guided five-second video at the native 24 fps clock.

    The prompt is used verbatim; an application may format or polish it before
    this boundary. Images are ordered and labeled ``<Picture 1>`` through
    ``<Picture 3>``. ``reference`` preserves the original single-image API;
    use ``references`` for an ordered tuple instead. ``reference_video`` accepts
    one visual-only local clip labeled ``<Video 1>``, without images. No references
    means T2VA and requires a T2VA pipeline. A session never swaps models.
    """

    prompt: str
    reference: Path | None = None
    width: int = 928
    height: int = 512
    seed: int = 0
    references: tuple[Path, ...] = field(default=(), kw_only=True)
    reference_video: Path | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prompt, str)
            or not self.prompt.strip()
            or len(self.prompt) > 65536
        ):
            raise ContractError("prompt must contain 1-65536 characters")
        if self.reference is not None and not isinstance(self.reference, Path):
            raise ContractError("reference must be a local pathlib.Path")
        if not isinstance(self.references, tuple) or any(
            not isinstance(path, Path) for path in self.references
        ):
            raise ContractError("references must be an ordered tuple of local pathlib.Paths")
        if self.reference is not None and self.references:
            raise ContractError("provide reference or references, not both")
        if self.reference_video is not None:
            if not isinstance(self.reference_video, Path):
                raise ContractError("reference_video must be a local pathlib.Path")
            if self.ordered_references:
                raise ContractError("provide one reference video or image references, not both")
        if len(self.ordered_references) > 3:
            raise ContractError("Ref2VA supports at most three reference images")
        if any(
            index not in {str(value) for value in range(1, len(self.ordered_references) + 1)}
            for index in re.findall(r"<Picture (\d+)>", self.prompt)
        ):
            raise ContractError("a prompt picture label has no corresponding reference image")
        if any(
            label != "<Video 1>" or self.reference_video is None
            for label in re.findall(r"<Video\s*\d+>", self.prompt)
        ):
            raise ContractError("a prompt video label has no corresponding reference video")
        if any(
            type(value) is not int or value < 32 or value % 32
            for value in (self.width, self.height)
        ):
            raise ContractError("the canvas must use positive multiples of 32")
        if self.width * self.height > 928 * 512 or not 0.25 <= self.width / self.height <= 4:
            raise ContractError(
                "this pipeline supports a canvas up to 928x512 pixels at 1:4-4:1"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ContractError("seed must be an integer between 0 and 2^63-1")

    @property
    def ordered_references(self) -> tuple[Path, ...]:
        return self.references or ((self.reference,) if self.reference is not None else ())

    @property
    def mode(self) -> str:
        return (
            "ref2va" if self.ordered_references or self.reference_video is not None else "t2va"
        )

    @property
    def model_frames(self) -> int:
        return 124

    @property
    def delivery_frames(self) -> int:
        return 120


@dataclass(frozen=True)
class PipelineProgress:
    stage: str
    completed: int
    total: int


@dataclass(frozen=True)
class VideoResult:
    output_path: Path
    profile_id: str
    seed: int
    elapsed_seconds: float
    stages: dict[str, Any]
    media: dict[str, Any]
