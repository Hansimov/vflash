"""Adapt the pinned official layout to padded model frames and exact delivery."""

from __future__ import annotations


def install_delivery_frame_layout(pipe):
    """Permit only VAE padding at the official fifteen-second endpoint.

    VideoRequest still bounds delivered seconds to 15, and media delivery trims
    the extra two model frames. Other components and pipeline limits are intact.
    """
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        align_num_frames,
    )

    class PaddedComponents:
        def __init__(self, original):
            self.original = original
            self.max_duration = (
                align_num_frames(
                    round(original.max_duration * original.fps),
                    original.vae_frames_per_chunk,
                    original.vae_latents_per_chunk,
                )
                / original.fps
            )

        def __getattr__(self, name):
            return getattr(self.original, name)

    class DeliveryFrameLayout(MiniMaxH3PrepareLayoutStep):
        def __call__(self, components, state):
            _, result = super().__call__(PaddedComponents(components), state)
            return components, result

    replaced = 0

    def replace(parent):
        nonlocal replaced
        for name, block in list(getattr(parent, "sub_blocks", {}).items()):
            if type(block) is MiniMaxH3PrepareLayoutStep:
                parent.sub_blocks[name] = DeliveryFrameLayout()
                replaced += 1
            else:
                replace(block)

    replace(pipe._blocks)
    if replaced != 1:
        raise RuntimeError("pinned Diffusers keyframe layout block changed")
