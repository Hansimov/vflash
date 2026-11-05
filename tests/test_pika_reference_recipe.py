from dataclasses import FrozenInstanceError

import pytest

from vflash.pipeline.pika_av import PikaAVRecipe
from vflash.pipeline.pika_reference_recipe import PikaReferenceRecipe


@pytest.mark.parametrize("edge", [None, 1024, 2048])
@pytest.mark.parametrize("steps", [0, 10, 12])
def test_independent_picture_partition_and_budget_have_distinct_frozen_identity(edge, steps):
    recipe = PikaReferenceRecipe(audio_steps=steps, reference_short_edge=edge)
    assert recipe.identity != PikaAVRecipe(audio_steps=steps).identity
    assert recipe.identity.endswith(f"-r{edge if edge is not None else 'tiered'}-v1")
    assert recipe.reference_encoding_policy["short_edge"] == edge
    with pytest.raises(FrozenInstanceError):
        recipe.reference_short_edge = 2048


@pytest.mark.parametrize(
    "options",
    [
        {"reference_short_edge": True},
        {"reference_short_edge": 1024.0},
        {"reference_short_edge": 512},
        {"overlap_decode": True},
        {"decode_tile_batch_size": 4},
    ],
)
def test_unqualified_execution_combination_is_rejected(options):
    with pytest.raises(ValueError):
        PikaReferenceRecipe(**options)


def test_initialized_reference_budget_cannot_be_changed_by_a_request():
    from pathlib import Path
    from types import SimpleNamespace

    from vflash.pipeline import VideoRequest
    from vflash.pipeline.pika_reference_av import PikaReferenceAVSession

    runtime = SimpleNamespace(closed=False, reference_short_edge=1024)
    session = PikaReferenceAVSession(runtime, predictor=None)
    request = VideoRequest("Use the reference subject.", reference=Path("subject.png"))
    for recipe in [PikaAVRecipe(), PikaReferenceRecipe(reference_short_edge=2048)]:
        with pytest.raises(ValueError, match="candidate"):
            session.generate(request, Path("unused"), recipe=recipe)
