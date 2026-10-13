"""Explicit pre-clamp protection for the pinned H3 stereo waveform decoder."""

import math
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any


@contextmanager
def audio_decoder_headroom(
    component: Any,
    observation: MutableMapping[str, Any],
    *,
    on_unclamped: Callable[[Any], None] | None = None,
) -> Iterator[None]:
    """Apply one attenuation-only gain before BigVGAN's hard output clamp.

    The caller exclusively owns this component for one complete stereo decode.
    Tracks whose sample peak is at most 0.98 pass through unchanged. A larger
    peak receives the same gain on both channels, preserving their relationship.
    This prevents sample clipping; it does not repair synthesis, synchronize
    audio, normalize loudness, or establish an encoded true-peak limit.
    The optional diagnostic callback receives the unmodified [2, 1, T] tensor.
    """
    if observation:
        raise ValueError("audio headroom requires an empty observation")
    decoder = component.module.model.decoder
    if type(decoder).__name__ != "BigVGAN" or decoder.use_tanh_at_final is not False:
        raise ValueError("headroom requires the original H3 hard-clamp audio decoder")

    def protect(module: Any, inputs: Any, output: Any) -> Any:
        if observation or output.ndim != 3 or output.shape[:2] != (2, 1):
            raise ValueError("one complete stereo waveform must own audio headroom")
        peak = float(output.abs().amax())
        if not math.isfinite(peak) or not output.isfinite().all():
            raise ValueError("audio decoder produced a nonfinite waveform")
        gain = min(1.0, 0.98 / peak) if peak else 1.0
        observation.update(
            raw_peak=peak,
            gain=gain,
            linked_stereo=True,
            raw_samples_outside_range=int((output.abs() > 1).sum()),
            method="uniform-pre-clamp-headroom-v1",
        )
        if on_unclamped is not None:
            on_unclamped(output)
        return output if gain == 1 else output * gain

    handle = decoder.conv_post.register_forward_hook(protect)
    try:
        yield
        if not observation:
            raise ValueError("audio headroom did not reach the official output convolution")
    finally:
        handle.remove()
