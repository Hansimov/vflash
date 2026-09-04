"""The release's model revisions, execution settings and supported hardware."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError, HardwareTarget, Profile


@dataclass(frozen=True, slots=True)
class ProfileCatalog:
    catalog_id: str
    targets: tuple[HardwareTarget, ...]
    profiles: tuple[Profile, ...]

    @classmethod
    def bundled(cls) -> ProfileCatalog:
        resource = files("vflash").joinpath("data/h3-profiles.json")
        return cls.from_dict(json.loads(resource.read_text(encoding="utf-8")))

    @classmethod
    def load(cls, path: Path) -> ProfileCatalog:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProfileCatalog:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ContractError("unsupported profile catalog schema")
        catalog_id = payload.get("catalog_id")
        if not isinstance(catalog_id, str) or not catalog_id:
            raise ContractError("catalog_id must be a non-empty string")
        targets = payload.get("targets")
        profiles = payload.get("profiles")
        if not all(
            isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows)
            for rows in (targets, profiles)
        ):
            raise ContractError("catalog targets and profiles must be non-empty object arrays")
        try:
            parsed_targets = tuple(HardwareTarget.from_dict(item) for item in targets)
            parsed_profiles = tuple(Profile.from_dict(item) for item in profiles)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid release profile: {exc}") from exc
        for label, rows in (("hardware target", parsed_targets), ("profile", parsed_profiles)):
            if len({item.id for item in rows}) != len(rows):
                raise ContractError(f"duplicate {label} id")
        known_targets = {item.id for item in parsed_targets}
        for profile in parsed_profiles:
            if unknown := set(profile.target_ids) - known_targets:
                raise ContractError(
                    f"profile {profile.id} has unknown targets: {sorted(unknown)}"
                )
        return cls(catalog_id, parsed_targets, parsed_profiles)

    def profile(self, profile_id: str) -> Profile:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        raise ContractError(f"unknown profile: {profile_id}")

    def target(self, target_id: str) -> HardwareTarget:
        for target in self.targets:
            if target.id == target_id:
                return target
        raise ContractError(f"unknown hardware target: {target_id}")
