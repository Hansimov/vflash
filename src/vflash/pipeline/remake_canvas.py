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
        # The native reference role may modify its latent before encoding the pin.
        # Keep the cache independent from the tensor handed to that role.
        self.frames[key] = (pixels, latent.clone())
        return latent


class _HalfSizeSourceEncoder:
    def __init__(self, vae):
        self.vae = vae

    def encode(self, pixels):
        if pixels.shape[0] == 1:
            return self.vae.encode(pixels)
        import torch.nn.functional as F

        frames, height, width, channels = pixels.shape
        if height % 64 or width % 64:
            raise ValueError("The source reference must preserve its aspect and 32-pixel grid")
        smaller = pixels.new_empty((frames, height // 2, width // 2, channels))
        for start in range(0, frames, 4):
            chunk = pixels[start : start + 4].movedim(-1, 1)
            smaller[start : start + 4] = F.avg_pool2d(chunk, 2).movedim(1, -1)
        return self.vae.encode(smaller)


class VFlashRemakeCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(original_conditioning().INPUT_TYPES())
        inputs["required"]["short_edge"][1]["max"] = 1440
        inputs["required"]["short_edge"][1]["tooltip"] = (
            "Explicit prepared 32-pixel-grid canvas; "
            "qualify memory and quality on the selected runtime."
        )
        inputs.setdefault("optional", {})["half_size_source_reference"] = (
            "BOOLEAN",
            {"default": False},
        )
        return inputs

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "run"
    CATEGORY = "Vflash/Remake"

    def run(self, half_size_source_reference=False, **inputs):
        if half_size_source_reference:
            inputs["vae"] = _HalfSizeSourceEncoder(inputs["vae"])
        if inputs.get("pins", "").strip():
            inputs["vae"] = _PinnedFrameEncoder(inputs["vae"])
        result = original_conditioning()().run(**inputs)
        if half_size_source_reference:
            positive, _ = result
            for _, metadata in positive:
                source = metadata["minimax_refs"][0]
                if source["kind"] != "video":
                    raise ValueError("Expected the original source-video reference first")
                source["latent_h"], source["latent_w"] = source["latent"].shape[-2:]
        return result
