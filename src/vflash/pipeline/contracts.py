"""Small explicit contracts for the first complete Ref2VA pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError

PIPELINE_PROFILE = "ref2va-turbo4-exact-sm89"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DIFFUSERS_REVISION = "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"


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
    """One image-guided five-second video at the native 24 fps clock.

    The prompt is used verbatim; an application may format or polish it before
    this boundary. Its image label is ``<Picture 1>``. Other modes and clocks
    remain separate profiles until their complete pipeline has been qualified.
    """

    prompt: str
    reference: Path
    width: int = 928
    height: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prompt, str)
            or not self.prompt.strip()
            or len(self.prompt) > 65536
        ):
            raise ContractError("prompt must contain 1-65536 characters")
        if not isinstance(self.reference, Path):
            raise ContractError("reference must be a local pathlib.Path")
        if any(
            type(value) is not int or value < 32 or value % 32
            for value in (self.width, self.height)
        ):
            raise ContractError("the canvas must use positive multiples of 32")
        if self.width * self.height > 928 * 512 or not 0.25 <= self.width / self.height <= 4:
            raise ContractError(
                "this preview supports a canvas up to 928x512 pixels at 1:4-4:1"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ContractError("seed must be an integer between 0 and 2^63-1")

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
