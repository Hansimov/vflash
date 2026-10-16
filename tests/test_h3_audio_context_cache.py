"""Current public six-step scope, hook ownership and failure rollback on CPU."""

from contextlib import contextmanager

import pytest
from test_h3_refinement import engine_fixture

from vflash.native import h3_audio_hidden_cache as candidate
from vflash.native import h3_audio_refinement as audio
from vflash.native.h3_refinement_schedule import refinement_schedule

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("failure", [None, "step", "receipt"])
def test_current_public_context_and_schedule_are_preserved(monkeypatch, failure):
    engine = engine_fixture()
    state = engine.latent_state
    before = state.audio_latents.clone()
    video = state.video_latents.clone()
    original = audio._refine_full_audio
    owned = []
    cache = {}

    @contextmanager
    def cached(owner, **kwargs):
        assert (
            owner.latent_state.num_condition_video_rows
            == owner.latent_state.video_latents.shape[1]
        )
        assert owner.schedule.nfe == 7
        cache.update(
            sequence_length=7,
            build_block_calls=0,
            reuse_block_calls=0,
            rebuilt_projection_rows=0,
            approximate=True,
            execution_layout="two-slot-ring",
        )
        owned.append("enter")
        try:
            yield cache
        finally:
            owned.append("exit")

    monkeypatch.setattr(candidate, "frozen_video_hidden", cached)
    request = audio.AudioRefinementRequest(
        refinement_schedule(steps=6, denoise=0.5),
        tuple([torch.zeros(7, 9, 6, 2) for _ in range(50)] + [torch.zeros(7, 3, 2, 2)]),
        context_pixels=None,
        reuse_hidden_context=True,
    )

    def advance(owner, **kwargs):
        index = owner.latent_state.evaluation_index
        if index == 1:
            cache["build_block_calls"] = 50
        else:
            cache["reuse_block_calls"] += 50
            cache["rebuilt_projection_rows"] += 350
        owner.latent_state.step(
            video_velocity=torch.ones_like(video), audio_velocity=torch.ones_like(before)
        )
        if failure == "step":
            raise RuntimeError("injected")
        if failure == "receipt" and index == 6:
            cache["reuse_block_calls"] -= 1

    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            audio.refine_audio(engine, request, noise=torch.zeros_like(before), step=advance)
        assert torch.equal(state.audio_latents, before)
    else:
        baseline, report = audio.refine_audio(
            engine, request, noise=torch.zeros_like(before), step=advance
        )
        assert torch.equal(baseline, before) and report["video_cache"]["approximate"]
        assert report["extra_audio_nfe"] == 6 and report["video_latents_identical"]
    assert owned == ["enter", "exit"] and audio._refine_full_audio is original
    assert (
        engine.latent_state is state
        and state.evaluation_index == 4
        and torch.equal(video, state.video_latents)
    )
