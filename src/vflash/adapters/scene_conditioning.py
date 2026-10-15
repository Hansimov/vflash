"""Bounded source-scene Qwen features for explicitly related I2VA repaints.

This changes conditioning: the original scene replaces the crop's Qwen view.
Keyframe/video VAE inputs, noise, denoising and delivery retain their own owners.
No feature is serializable or reusable outside this one pipeline/thread scope.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from vflash.contracts import ContractError
from vflash.pipeline.contracts import VideoRequest


class _TextCall:
    def __init__(self, block, owner):
        self.block, self.owner = block, owner

    def __getattr__(self, name):
        return getattr(self.block, name)

    def __call__(self, components, state):
        return self.owner._call(self.block, components, state)


class SourceSceneConditioning:
    """Created by H3Pipeline.source_scene_conditioning(), never shared or persisted."""

    def __init__(self, request: VideoRequest):
        if not isinstance(request, VideoRequest) or request.mode != "i2va":
            raise ContractError("source-scene conditioning requires an I2VA source")
        self.source = request
        self._thread = threading.get_ident()
        self._expires = time.monotonic() + 3600
        self._phase = "source"
        self._features = None
        self._active = False
        self._report = {
            "mode": "source-scene-qwen",
            "equivalent": False,
            "source_captures": 0,
            "repaint_encoder_calls_elided": 0,
            "cached_bytes": 0,
        }

    @property
    def report(self):
        return dict(self._report)

    def _check(self):
        if self._thread != threading.get_ident():
            raise ContractError("source-scene conditioning belongs to another thread")
        if time.monotonic() >= self._expires:
            self.close()
        if self._phase == "closed":
            raise ContractError("source-scene conditioning has ended")

    def check_request(self, request, *, reuse_scope=None):
        self._check()
        if self._phase == "source":
            if request != self.source or self._features is not None:
                raise ContractError("capture exactly the declared source first")
        elif self._phase == "repaint":
            if reuse_scope is not None:
                raise ContractError("non-equivalent repaint cannot populate the exact cache")
            if (
                request.mode != "i2va"
                or request.prompt != self.source.prompt
                or request.seed != self.source.seed
                or request.generate_audio
                or request.duration_seconds > self.source.duration_seconds
                or request.audio_delivery_profile != "unchanged"
                or request.keyframe_delivery_profile != "decoded"
            ):
                raise ContractError("scene reuse requires a related video-only I2VA repaint")
        else:
            raise ContractError("enter repaint() after the source has completed")

    def request_complete(self):
        self._check()
        if self._phase == "source":
            if self._features is None:
                raise ContractError("source did not capture Qwen features")
            self._phase = "ready"

    @contextmanager
    def repaint(self):
        """Use the completed source's scene context for serial video-only repaints."""
        self._check()
        if self._phase != "ready" or self._features is None:
            raise ContractError("repaint requires this scope's completed source")
        self._phase = "repaint"
        try:
            yield self
        finally:
            if self._phase != "closed":
                self._phase = "ready"

    @contextmanager
    def capture(self, conditioner):
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3FL2VATextEncoderStep,
        )

        from vflash.adapters.conditioning_reuse import _EncoderCall

        self._check()
        if self._active or self._phase not in {"source", "repaint"}:
            raise ContractError("source-scene capture requires a serial active request")
        blocks = conditioner.pipe._blocks.sub_blocks
        original = blocks["text_encoder"]
        # The exact sibling cache may already wrap this block. Capture its output
        # on both a hit and a miss; neither cache can retain the other's wrapper.
        underlying = original.block if type(original) is _EncoderCall else original
        if type(underlying) is not MiniMaxH3FL2VATextEncoderStep:
            raise ContractError("pinned I2VA Qwen block required")
        self._active = True
        blocks["text_encoder"] = _TextCall(original, self)
        try:
            yield
        finally:
            blocks["text_encoder"] = original
            self._active = False

    def _call(self, block, components, state):
        import torch

        self._check()
        if self._phase == "source":
            if self._features is not None:
                raise ContractError("only one source text capture is permitted")
            components, state = block(components, state)
            embeds, tags = (state.get(name) for name in ("prompt_embeds", "text_token_tags"))
            if (
                type(embeds) is not torch.Tensor
                or type(tags) is not torch.Tensor
                or embeds.ndim != 3
                or embeds.shape[0] != 1
                or embeds.shape[1] <= 0
                or embeds.shape[2] != 5120
                or embeds.dtype not in {torch.bfloat16, torch.float32}
                or tags.shape != (embeds.shape[1],)
                or tags.dtype != torch.int64
            ):
                raise ContractError("unexpected source Qwen features")
            size = sum(v.numel() * v.element_size() for v in (embeds, tags))
            if not 0 < size <= 128 * 1024**2:
                raise ContractError("source Qwen features exceed the 128 MiB bound")
            values = tuple(v.detach().to("cpu").clone() for v in (embeds, tags))
            if not all(bool(torch.isfinite(v).all()) for v in values):
                raise ContractError("source Qwen features must be finite")
            self._features = values
            self._report.update(source_captures=1, cached_bytes=size)
            return components, state
        if self._phase != "repaint" or self._features is None:
            raise ContractError("completed source features required")
        embeds, tags = self._features
        # copy=True also prevents aliasing on a CPU fixture or same-device call.
        state.set(
            "prompt_embeds",
            embeds.to(
                device=components._execution_device,
                dtype=components.text_encoder.dtype,
                copy=True,
            ),
        )
        state.set("text_token_tags", tags.clone())
        self._report["repaint_encoder_calls_elided"] += 1
        return components, state

    def close(self):
        self._features = None
        self._phase = "closed"
