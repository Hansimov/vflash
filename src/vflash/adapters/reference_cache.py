"""Bounded CPU reference latents owned by one fixed conditioner instance."""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager


class ReferenceLatentCache:
    """Reuse exact normalized VAE output; text encoding remains request-specific.

    The owning graph fixes the VAE, normalization and posterior seed. Keys use
    the already decoded input identity. Entries stay in process memory and
    disappear with the conditioner; tensors are copied at the ownership boundary.
    """

    def __init__(self, maximum_bytes=128 * 1024**2):
        if type(maximum_bytes) is not int or not 0 <= maximum_bytes <= 512 * 1024**2:
            raise ValueError("reference cache capacity must be between zero and 512 MiB")
        self.maximum_bytes = maximum_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.key = None
        self.diagnostics = {}
        self._bound = False

    @contextmanager
    def bind(self, reference, *, spatial_divisor=1, context_key=()):
        if self._bound:
            raise ValueError("reference cache requires serial request ownership")
        if type(spatial_divisor) is not int or spatial_divisor not in {1, 2}:
            raise ValueError("reference spatial divisor must be one or two")
        if not isinstance(context_key, tuple) or any(
            type(value) not in (str, int) for value in context_key
        ):
            raise ValueError("reference context identity requires strings and integers")
        self.spatial_divisor = spatial_divisor
        self._bound = True
        self.key = (
            None
            if reference is None
            else (
                reference.metadata["sha256"],
                reference.metadata["decoded_rgb_sha256"],
                reference.metadata["frames"],
                reference.metadata["width"],
                reference.metadata["height"],
                reference.metadata["vae_input_frames"],
            )
        )
        if self.key is not None and context_key:
            self.key += (context_key,)
        self.diagnostics = {
            "enabled": self.key is not None and self.maximum_bytes > 0,
            "hit": False,
            "stored_bytes": self.bytes,
        }
        try:
            yield
        finally:
            self.key = None
            self._bound = False

    def get(self):
        values = self.entries.get(self.key)
        if values is None:
            return None
        self.entries.move_to_end(self.key)
        self.diagnostics["hit"] = True
        return [value.clone() for value in values]

    def put(self, values):
        if self.key is None or not self.maximum_bytes:
            return
        if not values or any(v.device.type != "cpu" or v.ndim != 5 for v in values):
            raise ValueError("reference cache accepts CPU VAE tensors only")
        size = sum(v.numel() * v.element_size() for v in values)
        if size > self.maximum_bytes:
            return
        previous = self.entries.pop(self.key, ())
        self.bytes -= sum(v.numel() * v.element_size() for v in previous)
        while self.entries and self.bytes + size > self.maximum_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= sum(v.numel() * v.element_size() for v in removed)
        self.entries[self.key] = tuple(v.detach().clone() for v in values)
        self.bytes += size
        self.diagnostics["stored_bytes"] = self.bytes

    def clear(self):
        self.entries.clear()
        self.bytes = 0


def install_reference_latent_cache(pipe):
    """Replace this graph's encoder block, leaving all other graphs untouched."""
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VAReferenceEncoderStep,
    )

    cache = ReferenceLatentCache()

    class CachedReferenceEncoder(MiniMaxH3Ref2VAReferenceEncoderStep):
        def __call__(self, components, state):
            started = time.monotonic()
            values = cache.get()
            if values is None:
                components, state = super().__call__(components, state)
                if cache.key is not None:
                    if state.get("audio_condition_latents"):
                        raise ValueError(
                            "visual reference caching requires an empty audio prefix"
                        )
                    cache.put(state.get("condition_latents"))
            else:
                block = self.get_block_state(state)
                block.condition_latents = values
                block.audio_condition_latents = []
                self.set_block_state(state, block)
            if cache.key is not None and cache.spatial_divisor == 2:
                full = state.get("condition_latents")
                pooled = spatial_half_video_latents(full)
                block = self.get_block_state(state)
                block.condition_latents = pooled
                block.audio_condition_latents = []
                self.set_block_state(state, block)
                cache.diagnostics["spatial_compression"] = {
                    "method": "average-pool-spatial-half",
                    "source_shapes": [list(v.shape) for v in full],
                    "encoded_shapes": [list(v.shape) for v in pooled],
                    "all_temporal_positions_retained": True,
                    "pooled_reference_index": len(full) - 1,
                    "appearance_images_preserved": len(full) - 1,
                }
            cache.diagnostics["reference_vae_seconds"] = time.monotonic() - started
            return components, state

    def replace(block):
        for key, child in block.sub_blocks.items():
            if type(child).__name__ == "MiniMaxH3Ref2VAReferenceEncoderStep":
                block.sub_blocks[key] = CachedReferenceEncoder()
                return 1
            found = replace(child) if hasattr(child, "sub_blocks") else 0
            if found:
                return found
        return 0

    if replace(pipe._blocks) != 1:
        raise ValueError("pinned reference encoder block layout changed")
    return cache


def spatial_half_video_latents(values):
    """Pool the final video reference while preserving preceding appearance images."""
    if not values:
        raise ValueError("video pooling requires an ordered visual reference group")
    return [*values[:-1], *spatial_half_latents(values[-1:])]


def spatial_half_latents(values):
    """Keep every temporal position while pooling even spatial patch grids.

    This optional approximation follows the RefMod distinction between exact
    pre-encoding and compressed reference latents. It runs after cache lookup,
    retaining the full CPU entry for later requests.
    """
    import torch.nn.functional as functional

    return [
        functional.adaptive_avg_pool3d(
            value,
            (value.shape[2], max(2, value.shape[3] // 4 * 2), max(2, value.shape[4] // 4 * 2)),
        )
        for value in values
    ]
