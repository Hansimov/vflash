"""Pinned original Pika H3 Flex release and explicit six-evaluation contract.

This optional recipe does not change the existing hybrid or segmented defaults.
Its adapter weights remain subject to the MiniMax H3 Community License.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vflash.contracts import ContractError
from vflash.pipeline.inputs import ConditioningInputs, MediaReference, ReferenceLimits

PIKA_SOURCE_REVISION = "4bf8ea508891a432106e70f675d287c27f02306d"
PIKA_MODEL_REVISION = "0ddf55ad7067ca1e76812253bd7b4600df87f759"
PIKA_BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
PIKA_RECIPE = "pika-h3-flex-6-v1"
PIKA_KNOTS = (1000, 750, 500, 250, 100, 40, 0)
PIKA_ADAPTER_SHA256 = {
    "fl2va": "8ed6115d647ab11c1b38d04e63c4c6e1594128ae312f7b5635cd50563b7b9327",
    "ref2va": "eef684ba1258f0cdbbbda6128188ad2688417f2eb17ff875375e74a32b6d61b5",
}
PIKA_ADAPTER_BYTES = 2767274120
PIKA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")


def adapter_partition(mode: str) -> str:
    if mode not in {"t2va", "i2va", "l2va", "fl2va", "ref2va"}:
        raise ContractError("Unsupported Pika H3 Flex input mode")
    return "ref2va" if mode == "ref2va" else "fl2va"


def shifted_sigmas(shift: float) -> tuple[float, ...]:
    """Published knots, in float64 here; the sampler materializes them as FP32."""
    if shift not in (12.0, 3.0):
        raise ContractError("Pika H3 Flex fixes video/audio shifts to 12/3")
    return tuple(shift * (k / 1000) / (1 + (shift - 1) * (k / 1000)) for k in PIKA_KNOTS)


def validate_adapter_manifest(manifest: dict, mode: str) -> None:
    partition = adapter_partition(mode)
    expected = {
        "format_version": 1,
        "model": "pika-flex",
        "task": partition,
        "release_id": f"pika-flex-{partition}-v1",
        "rank": 128,
        "alpha": 128,
        "trunk": "transformer_ref" if partition == "ref2va" else "transformer",
        "numerics_policy": (
            "fp32_conditioning_v1" if partition == "ref2va" else "bf16_conditioning_v1"
        ),
        "sha256": PIKA_ADAPTER_SHA256[partition],
        "base": {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": PIKA_BASE_REVISION},
        "sampling": {
            "knots": list(PIKA_KNOTS),
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "method": "renoise",
        },
        "target_modules": list(PIKA_TARGET_MODULES),
    }
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ContractError("Pika adapter differs from the pinned original release")


@dataclass(frozen=True)
class PikaFlexRequest:
    """Original local-release request; output uses the model's aligned frame clock.

    These are native release buckets, not a promise of arbitrary-canvas or 4K
    generation. Application delivery cropping/trimming must be recorded separately.
    The original recipe always generates both audio and video jointly.
    """

    prompt: str
    mode: str = "t2va"
    resolution: str = "480p"
    aspect: str = "16:9"
    duration_seconds: int = 5
    seed: int = 0
    first_frame: Path | None = None
    last_frame: Path | None = None
    references: tuple[tuple[str, Path], ...] = ()

    def __post_init__(self):
        adapter_partition(self.mode)
        if (
            not isinstance(self.prompt, str)
            or not self.prompt.strip()
            or len(self.prompt) > 8000
        ):
            raise ContractError("Pika prompt must contain 1-8000 characters")
        if self.resolution not in {"480p", "768p"}:
            raise ContractError("The original local release supports 480p and 768p buckets")
        if self.aspect not in {
            "1:1",
            "3:4",
            "4:3",
            "2:3",
            "3:2",
            "9:16",
            "16:9",
            "9:21",
            "21:9",
            "5:4",
            "4:5",
        }:
            raise ContractError("Unsupported original Pika aspect bucket")
        if type(self.duration_seconds) is not int or not 5 <= self.duration_seconds <= 15:
            raise ContractError("Pika duration must be an integer from 5 to 15 seconds")
        if type(self.seed) is not int or not 0 <= self.seed <= 2147483647:
            raise ContractError("Pika seed must be an integer in the model range")
        if not isinstance(self.references, tuple) or any(
            not isinstance(r, tuple)
            or len(r) != 2
            or r[0] not in {"image", "video", "audio"}
            or not isinstance(r[1], Path)
            for r in self.references
        ):
            raise ContractError("References must be ordered (kind, local Path) tuples")
        inputs = self.conditioning
        if inputs.mode != self.mode:
            raise ContractError("Media roles disagree with the Pika input mode")
        inputs.validate_references(ReferenceLimits(9, 3, 3, 12))

    @property
    def conditioning(self) -> ConditioningInputs:
        return ConditioningInputs(
            self.first_frame,
            self.last_frame,
            tuple(MediaReference(kind, path) for kind, path in self.references),
        )
