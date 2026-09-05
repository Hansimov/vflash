"""Small, immutable contracts shared by the CLI, service, and native runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Any


class ContractError(ValueError):
    """A user-visible execution contract is internally inconsistent."""


class GenerationMode(StrEnum):
    REF2VA = "ref2va"


class Availability(StrEnum):
    PREVIEW = "preview"
    STABLE = "stable"


@dataclass(frozen=True, slots=True)
class AttentionPolicy:
    backend: str
    exact: bool
    backend_by_evaluation: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, nfe: int) -> AttentionPolicy:
        backend = _required_text(value, "backend")
        exact = value.get("exact")
        if not isinstance(exact, bool):
            raise ContractError("attention.exact must be a boolean")
        raw_schedule = value.get("backend_by_evaluation", [])
        if not isinstance(raw_schedule, list) or not all(
            isinstance(item, str) and item for item in raw_schedule
        ):
            raise ContractError("attention.backend_by_evaluation must contain backend names")
        schedule = tuple(raw_schedule)
        if schedule and len(schedule) != nfe:
            raise ContractError("attention schedule length must equal NFE")
        if exact and schedule and any(item != backend for item in schedule):
            raise ContractError("exact attention cannot switch to another backend")
        return cls(backend=backend, exact=exact, backend_by_evaluation=schedule)


@dataclass(frozen=True, slots=True)
class HardwareTarget:
    id: str
    compute_capability: str
    minimum_memory_gib: float
    weight_residency: str
    block_prefetch_slots: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HardwareTarget:
        target = cls(
            id=_required_text(value, "id"),
            compute_capability=_required_text(value, "compute_capability"),
            minimum_memory_gib=_required_number(value, "minimum_memory_gib"),
            weight_residency=_required_text(value, "weight_residency"),
            block_prefetch_slots=_required_int(value, "block_prefetch_slots"),
        )
        if target.compute_capability not in {"8.6", "8.9"}:
            raise ContractError("Vflash currently targets compute capability 8.6 or 8.9")
        if target.minimum_memory_gib <= 0 or target.block_prefetch_slots < 1:
            raise ContractError("hardware memory and prefetch slots must be positive")
        return target


@dataclass(frozen=True, slots=True)
class Profile:
    id: str
    mode: GenerationMode
    availability: Availability
    model: str
    model_revision: str
    adapter: str | None
    adapter_revision: str | None
    nfe: int
    scheduler: str
    video_flow_shift: float
    audio_flow_shift: float
    precision: str
    attention: AttentionPolicy
    target_ids: tuple[str, ...]
    quality_statement: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Profile:
        nfe = _required_int(value, "nfe")
        if nfe not in {4, 8}:
            raise ContractError("released Ref2VA profiles use 4 or 8 evaluations")
        adapter = value.get("adapter")
        adapter_revision = value.get("adapter_revision")
        if adapter is not None and (not isinstance(adapter, str) or not adapter):
            raise ContractError("profile adapter must be null or a non-empty string")
        if (adapter is None) != (adapter_revision is None):
            raise ContractError("adapter and adapter_revision must be declared together")
        if adapter_revision is not None and (
            not isinstance(adapter_revision, str) or not adapter_revision
        ):
            raise ContractError("adapter_revision must be null or a non-empty string")
        target_ids = _text_tuple(value, "target_ids")
        if not target_ids:
            raise ContractError("profile must support at least one hardware target")
        return cls(
            id=_required_text(value, "id"),
            mode=GenerationMode(_required_text(value, "mode")),
            availability=Availability(_required_text(value, "availability")),
            model=_required_text(value, "model"),
            model_revision=_required_text(value, "model_revision"),
            adapter=adapter,
            adapter_revision=adapter_revision,
            nfe=nfe,
            scheduler=_required_text(value, "scheduler"),
            video_flow_shift=_required_number(value, "video_flow_shift"),
            audio_flow_shift=_required_number(value, "audio_flow_shift"),
            precision=_required_text(value, "precision"),
            attention=AttentionPolicy.from_dict(_required_dict(value, "attention"), nfe=nfe),
            target_ids=target_ids,
            quality_statement=_required_text(value, "quality_statement"),
        )

    @property
    def selectable(self) -> bool:
        return self.availability in {Availability.PREVIEW, Availability.STABLE}


@dataclass(frozen=True, slots=True)
class PeerDevice:
    index: int
    uuid: str
    name: str
    memory_gib: float


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    profile: Profile
    target: HardwareTarget
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    gpu_memory_gib: float
    parallel_strategy: str = "single"
    peer_device: PeerDevice | None = None

    @property
    def gpu_uuids(self) -> tuple[str, ...]:
        return (self.gpu_uuid,) + (
            (self.peer_device.uuid,) if self.peer_device is not None else ()
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profile"]["mode"] = self.profile.mode.value
        payload["profile"]["availability"] = self.profile.availability.value
        return payload


def _required_dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ContractError(f"{key} must be an object")
    return item


def _required_text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ContractError(f"{key} must be a non-empty string")
    return item


def _required_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ContractError(f"{key} must be an integer")
    return item


def _required_number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not isfinite(item):
        raise ContractError(f"{key} must be a finite number")
    return float(item)


def _text_tuple(
    value: dict[str, Any], key: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list) or not all(isinstance(part, str) and part for part in item):
        raise ContractError(f"{key} must be a list of non-empty strings")
    if not item and not allow_empty:
        raise ContractError(f"{key} must not be empty")
    return tuple(item)
