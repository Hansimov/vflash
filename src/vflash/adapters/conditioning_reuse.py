"""Exact, bounded reuse of seed-independent H3 keyframe encodings.

The cache belongs to one serial conditioner instance. It never crosses a
caller-provided scope, stores only ordinary CPU tensors, and excludes request
noise, packed layout, transformer activations, scheduler state, and media.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from vflash.contracts import ContractError
from vflash.pipeline.contracts import ConditioningReuseScope, VideoRequest

_SHA256 = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class _EncodingKey:
    scope: ConditioningReuseScope
    identity: str
    reference_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scope, ConditioningReuseScope)
            or _SHA256.fullmatch(self.identity) is None
            or self.reference_count not in {1, 2}
        ):
            raise ContractError("conditioning reuse key is incomplete")


@dataclass(frozen=True)
class _CleanEncoding:
    prompt_embeds: Any
    text_token_tags: Any
    condition_latents: tuple[Any, ...]

    def _tensors(self) -> tuple[Any, ...]:
        return self.prompt_embeds, self.text_token_tags, *self.condition_latents

    def validate(self, key: _EncodingKey) -> int:
        import torch

        tensors = self._tensors()
        if (
            len(self.condition_latents) != key.reference_count
            or any(type(value) is not torch.Tensor for value in tensors)
            or any(value.device.type != "cpu" for value in tensors)
        ):
            raise ContractError("conditioning cache accepts owned CPU tensors only")
        if (
            self.prompt_embeds.ndim != 3
            or self.prompt_embeds.shape[0] != 1
            or self.prompt_embeds.shape[1] <= 0
            or self.prompt_embeds.shape[2] != 5120
            or self.prompt_embeds.dtype not in {torch.bfloat16, torch.float32}
            or self.text_token_tags.shape != (self.prompt_embeds.shape[1],)
            or self.text_token_tags.dtype != torch.int64
        ):
            raise ContractError("clean H3 text encoding has an unexpected contract")
        if any(
            value.ndim != 5
            or value.shape[0] != 1
            or any(size <= 0 for size in value.shape)
            or value.dtype != torch.float32
            for value in self.condition_latents
        ):
            raise ContractError("clean H3 keyframe encoding has an unexpected contract")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ContractError("non-finite values cannot enter the conditioning cache")
        return sum(value.numel() * value.element_size() for value in tensors)

    def independent_copy(self) -> _CleanEncoding:
        return _CleanEncoding(
            self.prompt_embeds.detach().clone(),
            self.text_token_tags.detach().clone(),
            tuple(value.detach().clone() for value in self.condition_latents),
        )


class _RequestEncodingCache:
    """One copied entry with explicit capacity and lifetime."""

    def __init__(
        self,
        *,
        maximum_bytes: int = 128 * 1024 * 1024,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(maximum_bytes) is not int or not 0 < maximum_bytes <= 256 * 1024 * 1024:
            raise ContractError("conditioning cache capacity must be at most 256 MiB")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int | float)
            or not math.isfinite(ttl_seconds)
            or not 0 < ttl_seconds <= 600
        ):
            raise ContractError("conditioning cache lifetime must be at most ten minutes")
        self.maximum_bytes = maximum_bytes
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self.hits = self.misses = 0
        self.clear()

    def clear(self) -> None:
        self._key: _EncodingKey | None = None
        self._value: _CleanEncoding | None = None
        self._bytes = 0
        self._expires_at = 0.0

    @property
    def cached_bytes(self) -> int:
        if self._clock() >= self._expires_at:
            self.clear()
        return self._bytes

    def get(self, key: _EncodingKey) -> _CleanEncoding | None:
        if not isinstance(key, _EncodingKey):
            raise ContractError("conditioning cache requires a bound identity")
        if self._clock() >= self._expires_at or self._key != key:
            self.clear()
        if self._value is None:
            self.misses += 1
            return None
        self.hits += 1
        return self._value.independent_copy()

    def put(self, key: _EncodingKey, value: _CleanEncoding) -> bool:
        if not isinstance(key, _EncodingKey) or not isinstance(value, _CleanEncoding):
            raise ContractError("conditioning cache requires typed clean encodings")
        size = value.validate(key)
        self.clear()
        if size > self.maximum_bytes:
            return False
        self._key = key
        self._value = value.independent_copy()
        self._bytes = size
        self._expires_at = self._clock() + self.ttl_seconds
        return True


class _EncoderCall:
    def __init__(self, block: Any, call: Any) -> None:
        self.block = block
        self.call = call

    def __getattr__(self, name: str) -> Any:
        return getattr(self.block, name)

    def __call__(self, components: Any, state: Any) -> Any:
        return self.call(self.block, components, state)


class H3CleanConditioningReuse:
    """Wrap the pinned text and keyframe VAE blocks for one capture."""

    def __init__(
        self,
        conditioner: Any,
        *,
        cache: _RequestEncodingCache | None = None,
    ) -> None:
        self.conditioner = conditioner
        self.cache = cache if cache is not None else _RequestEncodingCache()
        self._scope: ConditioningReuseScope | None = None
        self._active = False
        self._blocks()

    def _blocks(self) -> dict[str, Any]:
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3FL2VATextEncoderStep,
            MiniMaxH3KeyframeVaeEncoderStep,
        )

        blocks = self.conditioner.pipe._blocks.sub_blocks
        expected = {
            "text_encoder": MiniMaxH3FL2VATextEncoderStep,
            "vae_encoder": MiniMaxH3KeyframeVaeEncoderStep,
        }
        if any(type(blocks.get(name)) is not kind for name, kind in expected.items()):
            raise ContractError("pinned H3 keyframe encoder block layout changed")
        return {name: blocks[name] for name in expected}

    def clear(self) -> None:
        self.cache.clear()
        self._scope = None

    def observe_scope(self, scope: ConditioningReuseScope | None) -> None:
        if scope is not None and not isinstance(scope, ConditioningReuseScope):
            self.clear()
            raise ContractError("conditioning reuse requires a typed caller scope")
        if scope is None or scope != self._scope:
            self.cache.clear()
        self._scope = scope

    def _key(
        self,
        scope: ConditioningReuseScope,
        request: VideoRequest,
        references: tuple[Any, ...],
    ) -> _EncodingKey:
        if request.mode not in {"i2va", "l2va", "fl2va"}:
            raise ContractError("clean conditioning reuse supports keyframe requests only")
        expected_count = 2 if request.mode == "fl2va" else 1
        reference_identity = []
        for reference in references:
            digest = getattr(reference, "sha256", None)
            image = getattr(reference, "image", None)
            size = getattr(image, "size", None)
            if _SHA256.fullmatch(digest or "") is None or (
                not isinstance(size, tuple)
                or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
            ):
                raise ContractError("conditioning reuse requires decoded image identities")
            reference_identity.append({"sha256": digest, "width": size[0], "height": size[1]})
        if len(reference_identity) != expected_count:
            raise ContractError("conditioning reuse reference count differs from the request")

        pipe = self.conditioner.pipe
        blocks = self._blocks()
        identity = {
            "schema": 1,
            "profile_id": self.conditioner.prepared.profile_id,
            "runtime_versions": self.conditioner.versions,
            "request": {
                "mode": request.mode,
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "duration_seconds": request.duration_seconds,
                "model_frames": request.model_frames,
                "delivery_frames": request.delivery_frames,
                "references": reference_identity,
            },
            "encoder": {
                "keyframe_encode_seed": pipe.keyframe_encode_seed,
                "text_encoder_layer": pipe.text_encoder_layer,
                "pixel_mean": pipe.pixel_mean,
                "pixel_std": pipe.pixel_std,
                "canvas_short_edge": pipe.canvas_short_edge,
                "canvas_max_pixels": pipe.canvas_max_pixels,
                "canvas_multiple": pipe.canvas_multiple,
                "components": {
                    name: id(getattr(pipe, name))
                    for name in ("text_encoder", "tokenizer", "processor", "vae")
                },
                "blocks": {name: type(block).__qualname__ for name, block in blocks.items()},
            },
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        return _EncodingKey(scope, digest, expected_count)

    @contextmanager
    def capture(
        self,
        scope: ConditioningReuseScope,
        request: VideoRequest,
        references: tuple[Any, ...],
    ) -> Iterator[dict[str, Any]]:
        """Reuse only encoder outputs; publish a miss after capture succeeds."""

        if self._active:
            raise ContractError("conditioning reuse requires serial, non-nested capture")
        self.observe_scope(scope)
        self._active = True
        originals: dict[str, Any] = {}
        report: dict[str, Any] = {
            "mode": "clean-conditioning-v1",
            "status": "miss",
            "cached_bytes": 0,
            "lookup_seconds": 0.0,
            "capture_copy_seconds": 0.0,
            "restore_seconds": 0.0,
            "store_seconds": 0.0,
            "encoder_calls": {"text_encoder": 0, "vae_encoder": 0},
        }
        try:
            originals = self._blocks()
            started = time.monotonic()
            key = self._key(scope, request, references)
            cached = self.cache.get(key)
            report["lookup_seconds"] = time.monotonic() - started
            report["status"] = "hit" if cached is not None else "miss"
            captured: dict[str, Any] = {}
            calls: set[str] = set()

            def cpu_copy(value: Any) -> Any:
                started = time.monotonic()
                result = value.detach().to(device="cpu").clone()
                report["capture_copy_seconds"] += time.monotonic() - started
                return result

            def encode(name: str, block: Any, components: Any, state: Any) -> Any:
                if name in calls:
                    raise ContractError("an H3 encoder executed twice in one capture")
                calls.add(name)
                if cached is None:
                    components, state = block(components, state)
                    report["encoder_calls"][name] = 1
                    if name == "text_encoder":
                        captured["prompt_embeds"] = cpu_copy(state.get("prompt_embeds"))
                        captured["text_token_tags"] = cpu_copy(state.get("text_token_tags"))
                    else:
                        captured["condition_latents"] = tuple(
                            cpu_copy(value) for value in state.get("condition_latents")
                        )
                else:
                    started = time.monotonic()
                    if name == "text_encoder":
                        state.set(
                            "prompt_embeds",
                            cached.prompt_embeds.to(
                                device=components._execution_device,
                                dtype=components.text_encoder.dtype,
                            ),
                        )
                        state.set("text_token_tags", cached.text_token_tags)
                    else:
                        state.set("condition_latents", list(cached.condition_latents))
                    report["restore_seconds"] += time.monotonic() - started
                return components, state

            blocks = self.conditioner.pipe._blocks.sub_blocks
            for name, block in originals.items():
                blocks[name] = _EncoderCall(
                    block,
                    lambda original, components, state, name=name: encode(
                        name, original, components, state
                    ),
                )
            yield report
            if calls != set(originals):
                raise ContractError("capture did not execute both clean encoder boundaries")
            if cached is None:
                started = time.monotonic()
                stored = self.cache.put(key, _CleanEncoding(**captured))
                report["store_seconds"] = time.monotonic() - started
                if not stored:
                    report["status"] = "capacity-bypass"
            report["cached_bytes"] = self.cache.cached_bytes
        except BaseException:
            self.clear()
            raise
        finally:
            if originals:
                self.conditioner.pipe._blocks.sub_blocks.update(originals)
            self._active = False
