"""Seed-independent VDN conditioning reuse inside one explicit caller scope.

Only the pinned encoder's raw text features and clean keyframe latents are reused.
DiT, TokenRefiner, sampling noise and denoising states are never cached. Compare
canonical RGB bytes directly: no model or media content hashing is performed.
The owner must keep its model assets immutable for the lifetime of this cache.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

from PIL import Image, ImageOps

from vflash.contracts import ContractError
from vflash.pipeline.contracts import ConditioningReuseScope


class VDNCleanConditioningCache:
    """A serial, bounded, memory-only cache owned by one pipeline process."""

    def __init__(
        self,
        *,
        max_entries: int = 2,
        max_bytes: int = 256 * 1024**2,
        ttl_seconds: float = 3600,
        clock=time.monotonic,
    ):
        if type(max_entries) is not int or not 1 <= max_entries <= 4:
            raise ValueError("one to four conditioning entries required")
        if type(max_bytes) is not int or not 0 < max_bytes <= 256 * 1024**2:
            raise ValueError("positive conditioning byte budget required")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        if type(ttl_seconds) not in (int, float) or not 0 < ttl_seconds <= 3600:
            raise ValueError("conditioning lifetime must be at most one hour")
        self.ttl_seconds, self._clock = float(ttl_seconds), clock
        self._entries = deque()
        self._scope = None
        self._expires_at = 0.0

    @property
    def retained_bytes(self):
        if self._clock() >= self._expires_at:
            self.clear()
        return sum(entry[3] for entry in self._entries)

    def clear(self):
        self._entries.clear()
        self._scope = None
        self._expires_at = 0.0

    def encode(self, model: Path, request, plan: dict, directory: Path, encoder, *, scope=None):
        if scope is not None and not isinstance(scope, ConditioningReuseScope):
            raise ContractError("VDN reuse requires an explicit caller-owned scope")
        if scope is None:
            self.clear()
            return {**encoder(model, request, plan, directory), "conditioning_cache_hit": False}
        if scope != self._scope or self._clock() >= self._expires_at:
            self.clear()
        self._scope = scope
        self._expires_at = self._clock() + self.ttl_seconds
        started = time.monotonic()
        if request.mode not in {"i2va", "fl2va"}:
            raise ValueError("clean keyframe conditioning only")
        names = ["first"] if request.mode == "i2va" else ["first", "last"]
        paths = [request.first_frame]
        if request.mode == "fl2va":
            paths.append(request.last_frame)
        pixels = []
        for path in paths:
            with Image.open(path) as image:
                canonical = ImageOps.fit(
                    image.convert("RGB"),
                    (request.width, request.height),
                    method=Image.Resampling.LANCZOS,
                )
                pixels.append(canonical.tobytes())
        # The original encoder uses a fixed, local VAE generator (42), not the
        # candidate seed. All plan fields are included to avoid accidental
        # reuse across future geometry/conditioning contract extensions.
        metadata = (
            str(model.resolve()),
            request.mode,
            request.prompt,
            request.width,
            request.height,
            json.dumps(plan, sort_keys=True),
        )
        key = (metadata, tuple(pixels))
        for index, entry in enumerate(self._entries):
            if entry[0] != key:
                continue
            del self._entries[index]
            self._entries.append(entry)
            _, files, report, _ = entry
            for name, data in files.items():
                with (directory / name).open("xb") as stream:
                    stream.write(data)
            return {
                **report,
                "elapsed_seconds": time.monotonic() - started,
                "text_seconds": 0.0,
                "text_encoder_reused": False,
                "text_embedding_reused": True,
                "conditioning_cache_hit": True,
            }
        report = encoder(model, request, plan, directory)
        files = ["target.pt", *(f"{name}-reference.png" for name in names)]
        if plan["enabled"]:
            files.append("first.pt")
        # Metadata and reference bytes count against the budget too. Do not
        # read a huge tensor output into RAM just to discover it is oversized.
        key_bytes = len(repr(metadata).encode()) + sum(map(len, pixels))
        size = key_bytes + sum((directory / name).stat().st_size for name in files)
        if size <= self.max_bytes:
            values = {name: (directory / name).read_bytes() for name in files}
            while self._entries and (
                len(self._entries) >= self.max_entries
                or sum(entry[3] for entry in self._entries) + size > self.max_bytes
            ):
                self._entries.popleft()
            self._entries.append((key, values, dict(report), size))
            self._expires_at = self._clock() + self.ttl_seconds
        return {
            **report,
            "elapsed_seconds": time.monotonic() - started,
            "text_embedding_reused": False,
            "conditioning_cache_hit": False,
        }
