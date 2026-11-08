from dataclasses import replace
from pathlib import Path

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.contracts import VideoRequest
from vflash.pipeline.inputs import ConditioningInputs, MediaReference, ReferenceLimits
from vflash.pipeline.pika_flex import PikaFlexRequest


@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("t2va", {}),
        ("i2va", {"first_frame": Path("first.png")}),
        ("l2va", {"last_frame": Path("last.png")}),
        ("fl2va", {"first_frame": Path("first.png"), "last_frame": Path("last.png")}),
    ],
)
def test_both_request_boundaries_have_identical_anchor_semantics(mode, kwargs):
    complete = VideoRequest("A person turns.", **kwargs)
    original = PikaFlexRequest("A person turns.", mode=mode, **kwargs)
    assert complete.conditioning == original.conditioning
    assert complete.mode == original.conditioning.mode == mode
    changed = replace(complete, seed=10)
    assert changed.conditioning == complete.conditioning


def test_mixed_order_and_per_kind_numbering_survive_a_round_trip():
    request = PikaFlexRequest(
        "A performance.",
        mode="ref2va",
        references=(
            ("audio", Path("voice.wav")),
            ("image", Path("face.png")),
            ("video", Path("motion.mp4")),
            ("image", Path("outfit.png")),
            ("video", Path("camera.mp4")),
            ("audio", Path("environment.wav")),
        ),
    )
    inputs = request.conditioning
    assert [label for label, _ in inputs.labeled_media] == [
        "<Audio 1>",
        "<Picture 1>",
        "<Video 1>",
        "<Picture 2>",
        "<Video 2>",
        "<Audio 2>",
    ]
    assert tuple((r.kind, r.path) for r in inputs.ordered_media) == request.references
    with pytest.raises(ContractError, match="allocation"):
        inputs.validate_references(ReferenceLimits(9, 1, 0, 10, images_with_video=3))


def test_reference_images_do_not_become_temporal_anchors():
    path = Path("person.png")
    content = VideoRequest("A person walks.", references=(path,))
    first = VideoRequest("A person walks.", first_frame=path)
    assert content.conditioning.mode == "ref2va"
    assert first.conditioning.mode == "i2va"
    assert content.conditioning.ordered_media == first.conditioning.ordered_media
    assert content.conditioning != first.conditioning
    with pytest.raises(ContractError, match="not both"):
        ConditioningInputs(first_frame=path, references=(MediaReference("image", path),))


@pytest.mark.parametrize(
    "references",
    [
        [("image", Path("face.png"))],
        (("image", "face.png"),),
        (("mask", Path("mask.png")),),
        (("audio", Path("sound.wav")),),
    ],
)
def test_invalid_inputs_cannot_silently_disappear(references):
    with pytest.raises(ContractError):
        PikaFlexRequest("A scene.", mode="ref2va", references=references)


def test_complete_contract_keeps_its_narrower_video_budget():
    images = tuple(Path(f"image-{n}.png") for n in range(4))
    with pytest.raises(ContractError, match="allocation"):
        VideoRequest("A scene.", references=images, reference_video=Path("motion.mp4"))
    request = VideoRequest(
        "A scene.", references=images[:3], reference_video=Path("motion.mp4")
    )
    assert [kind for kind, _ in request.conditioning.labeled_media] == [
        "<Picture 1>",
        "<Picture 2>",
        "<Picture 3>",
        "<Video 1>",
    ]
