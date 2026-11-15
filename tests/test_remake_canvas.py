from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_remake_inputs import remake_request as remake_request

from vflash.pipeline.remake_canvas import VFlashRemakeCanvas


def test_extended_canvas_reuses_original_conditioning_without_mutating_its_schema(monkeypatch):
    original = {
        "required": {"short_edge": ("INT", {"min": 480, "max": 736}), "ref_1": ("IMAGE",)}
    }
    before = deepcopy(original)
    calls = []

    class Original:
        @classmethod
        def INPUT_TYPES(cls):
            return original

        def run(self, **inputs):
            calls.append(inputs)
            return ("original conditioning", "original latent")

    monkeypatch.setitem(
        __import__("sys").modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"RemakeConditioning": Original}),
    )
    assert VFlashRemakeCanvas.INPUT_TYPES()["required"]["short_edge"][1]["max"] == 1440
    assert original == before
    source, ref = object(), object()
    inputs = dict(source=source, ref_1=ref, short_edge=1088, frames=226, pins="0")
    assert VFlashRemakeCanvas().run(**inputs) == ("original conditioning", "original latent")
    assert calls == [inputs]


@pytest.mark.parametrize("width,height", [(1920, 1088), (1088, 1920), (1440, 1440)])
def test_two_megapixel_request_keeps_grid_and_reference_contract(remake_request, width, height):
    request = replace(remake_request, width=width, height=height)
    assert (request.width, request.height) == (width, height)
    with pytest.raises(ValueError):
        replace(request, width=width + 1)
    with pytest.raises(ValueError):
        replace(request, references=())
