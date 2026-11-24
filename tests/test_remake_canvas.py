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
    inputs = dict(source=source, ref_1=ref, short_edge=1088, frames=226, pins="")
    assert VFlashRemakeCanvas().run(**inputs) == ("original conditioning", "original latent")
    assert calls == [inputs]


def test_pinned_reference_shares_only_the_same_image_within_one_request(monkeypatch):
    class Latent(list):
        def clone(self):
            return Latent(self)

    class Encoder:
        def __init__(self):
            self.inputs = []

        def encode(self, pixels):
            self.inputs.append(pixels)
            return Latent([id(pixels)])

    class Original:
        def run(self, vae, source, ref_1, ref_2, pins):
            video = vae.encode(source)
            ref = vae.encode(ref_1)
            other = vae.encode(ref_2)
            return video, ref, other, vae.encode(ref_1), vae.encode(ref_2)

    monkeypatch.setitem(
        __import__("sys").modules,
        "nodes",
        SimpleNamespace(NODE_CLASS_MAPPINGS={"RemakeConditioning": Original}),
    )
    vae = Encoder()
    video = SimpleNamespace(shape=(90, 1088, 1920, 3))
    first = SimpleNamespace(shape=(1, 1088, 1920, 3))
    second = SimpleNamespace(shape=first.shape)
    inputs = dict(vae=vae, source=video, ref_1=first, ref_2=second, pins="0, 89")
    for _ in range(2):
        _, a, b, pinned_a, pinned_b = VFlashRemakeCanvas().run(**inputs)
        assert a == pinned_a and b == pinned_b and a != b
        assert a is not pinned_a and b is not pinned_b
    assert list(map(id, vae.inputs)) == list(map(id, [video, first, second] * 2))


@pytest.mark.parametrize("width,height", [(1920, 1088), (1088, 1920), (1440, 1440)])
def test_two_megapixel_request_keeps_grid_and_reference_contract(remake_request, width, height):
    request = replace(remake_request, width=width, height=height)
    assert (request.width, request.height) == (width, height)
    with pytest.raises(ValueError):
        replace(request, width=width + 1)
    with pytest.raises(ValueError):
        replace(request, references=())


def test_half_size_video_keeps_time_and_full_size_image_pins(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    source = torch.arange(5 * 64 * 128 * 3).reshape(5, 64, 128, 3).float()
    picture = source[:1]

    class Encoder:
        def encode(self, pixels):
            calls.append(pixels)
            return torch.zeros(
                1, 24, pixels.shape[0], pixels.shape[1] // 16, pixels.shape[2] // 16
            )

    class Original:
        def run(self, vae, source, ref_1, pins):
            z = vae.encode(source)
            image = vae.encode(ref_1)
            image.add_(2)
            pin = vae.encode(ref_1)
            assert pin.count_nonzero() == 0
            meta = {
                "minimax_refs": [
                    {"kind": "video", "latent": z, "latent_h": 4, "latent_w": 8},
                    {"kind": "image", "latent": image},
                ],
                "minimax_keyframes": [pin],
            }
            return [[None, meta]], "original-target"

    monkeypatch.setattr("vflash.pipeline.remake_canvas.original_conditioning", lambda: Original)
    positive, latent = VFlashRemakeCanvas().run(
        half_size_source_reference=True,
        vae=Encoder(),
        source=source,
        ref_1=picture,
        pins="0",
    )
    assert latent == "original-target" and len(calls) == 2
    pooled = source.reshape(5, 32, 2, 64, 2, 3).mean((2, 4))
    assert torch.equal(calls[0], pooled) and calls[1] is picture
    video, ref = positive[0][1]["minimax_refs"]
    assert (video["latent_h"], video["latent_w"]) == (2, 4)
    assert (
        ref["latent"].shape[-2:] == positive[0][1]["minimax_keyframes"][0].shape[-2:] == (4, 8)
    )


def test_half_size_source_rejects_incompatible_canvas_or_mask(remake_request, tmp_path):
    with pytest.raises(ValueError, match="aligned large edit canvas"):
        replace(remake_request, half_size_source_reference=True)
    request = replace(remake_request, width=1920, height=1088, half_size_source_reference=True)
    with pytest.raises(ValueError, match="aligned large edit canvas"):
        replace(request, source_mask=tmp_path / "mask.png")
