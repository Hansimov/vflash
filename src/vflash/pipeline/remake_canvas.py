"""Larger explicit canvases through the pinned original Remake conditioning.

Only the original node's UI admission range changes. Its source/reference VAE,
noise, pins, text and audio implementation remains the original node's run method.
The caller owns hardware and complete-output qualification for each geometry.
"""

from copy import deepcopy


def original_conditioning():
    from nodes import NODE_CLASS_MAPPINGS

    return NODE_CLASS_MAPPINGS["RemakeConditioning"]


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
        return original_conditioning()().run(**inputs)
