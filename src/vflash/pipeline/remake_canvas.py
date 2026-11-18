"""Larger explicit canvases through the pinned original Remake conditioning.

The original node owns canvas transforms, noise, pins, text and audio. An image
that it encodes as both a reference and an exact-size pin can share its latent.
The caller owns hardware and complete-output qualification for each geometry.
"""

from copy import deepcopy


def original_conditioning():
    from nodes import NODE_CLASS_MAPPINGS

    return NODE_CLASS_MAPPINGS["RemakeConditioning"]


class _PinnedFrameEncoder:
    """Reuse an identical image tensor only during this conditioning call."""

    def __init__(self, vae):
        self.vae = vae
        self.frames = {}

    def encode(self, pixels):
        if pixels.shape[0] != 1:
            return self.vae.encode(pixels)
        key = id(pixels)
        if key in self.frames:
            # Give each native role independent storage, as with two encodes.
            return self.frames[key][1].clone()
        latent = self.vae.encode(pixels)
        # Retaining the tensor prevents ID reuse for a later resized view.
        self.frames[key] = (pixels, latent)
        return latent


class VFlashRemakeCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(original_conditioning().INPUT_TYPES())
        inputs["required"]["short_edge"][1]["max"] = 1440
        inputs["required"]["short_edge"][1]["tooltip"] = (
            "Explicit prepared 32-pixel-grid canvas; "
            "qualify memory and quality on the selected runtime."
        )
        return inputs

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "run"
    CATEGORY = "Vflash/Remake"

    def run(self, **inputs):
        if inputs.get("pins", "").strip():
            inputs["vae"] = _PinnedFrameEncoder(inputs["vae"])
        return original_conditioning()().run(**inputs)
