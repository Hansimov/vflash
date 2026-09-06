"""Selective owned reads from a prepared official checkpoint index."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError
from vflash.native.h3_tensor_file import H3SingleTensorStore


def read_weight_map(directory: Path, filename: str) -> dict[str, str]:
    try:
        value = json.loads((directory / filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("the component checkpoint index is unavailable") from exc
    weights = value.get("weight_map") if isinstance(value, dict) else None
    if (
        not isinstance(weights, dict)
        or not weights
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(shard, str)
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
            for name, shard in weights.items()
        )
    ):
        raise ContractError("the component checkpoint index contains invalid paths")
    return weights


class IndexedCheckpoint:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.weight_map = read_weight_map(
            directory,
            "diffusion_pytorch_model.safetensors.index.json",
        )
        self._stores: dict[str, H3SingleTensorStore] = {}

    def load_many(self, names: Iterable[str]) -> dict[str, Any]:
        grouped: dict[str, list[str]] = {}
        for name in names:
            try:
                shard = self.weight_map[name]
            except KeyError as exc:
                raise ContractError(f"checkpoint tensor is missing: {name}") from exc
            grouped.setdefault(shard, []).append(name)
        result = {}
        for shard, selected in grouped.items():
            if shard not in self._stores:
                self._stores[shard] = H3SingleTensorStore(self.directory / shard)
            result.update(self._stores[shard].load_many(selected))
        return result

    def load(self, name: str) -> Any:
        return self.load_many((name,))[name]
