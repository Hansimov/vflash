from copy import copy
from types import SimpleNamespace as N

import pytest
from test_h3_refinement import engine_fixture

from vflash.native.h3_audio_refinement import (
    AudioRefinementRequest,
    reduced_audio_context,
    refine_audio,
    select_context,
)
from vflash.native.h3_native_scheduler import H3NativeLatentState
from vflash.native.h3_refinement_schedule import refinement_schedule

torch = pytest.importorskip("torch")


def context_engine(height=32, width=32):
    engine = engine_fixture()
    spatial, frames, text_rows, audio_rows = height * width, 2, 5, 4
    state = H3NativeLatentState(
        engine.schedule,
        video_latents=torch.randn(1, spatial * (frames + 1), 96),
        audio_latents=torch.randn(1, audio_rows, 32),
        num_condition_video_rows=spatial,
        num_condition_audio_rows=0,
    )
    state.evaluation_index = 4
    engine.latent_state = state
    prefix = spatial + text_rows + audio_rows
    length = prefix + frames * spatial
    engine.video_indices = torch.cat(
        (torch.arange(text_rows, text_rows + spatial), torch.arange(prefix, length))
    )
    engine.audio_indices = torch.arange(text_rows + spatial, prefix)
    engine.text_indices = torch.arange(text_rows)
    engine.sequence_length = length
    engine.refined_text = torch.randn(1, text_rows, 8, dtype=torch.bfloat16)
    engine.token_tags = torch.zeros(length, dtype=torch.long)
    engine.token_tags[engine.audio_indices] = 1
    engine.token_tags[engine.text_indices] = 2
    engine.rotary_cos = torch.arange(length * 4).reshape(length, 4).float()
    engine.rotary_sin = -engine.rotary_cos
    engine.input_packer = N(prepare_invocation=lambda **kwargs: N(**kwargs))
    attention = engine.denoiser.blocks[0]._veda_attention
    attention.layout = N(target=N(grid=(frames, height, width)))
    attention.dense_layers = frozenset([0, 1, 2, 47, 48, 49])
    return engine


def fake_attention(original, selection):
    result = copy(original)
    result.layout = N(target=N(grid=selection.grid))
    result.sparse_calls = result.dense_calls = 0
    return result


def test_context_keeps_every_time_step_audio_text_and_original_rope(monkeypatch):
    import vflash.native.h3_audio_refinement as module

    monkeypatch.setattr(module, "_attention_view", fake_attention)
    engine = context_engine()
    selection = select_context(engine)
    assert selection.source_grid == (2, 32, 32) and selection.grid == (2, 22, 22)
    assert selection.condition_rows == 484
    assert selection.audio_indices.numel() == 4 and selection.text_indices.numel() == 5
    original_video = engine.latent_state.video_latents.clone()
    original_audio = engine.latent_state.audio_latents.clone()
    attention = engine.denoiser.blocks[0]._veda_attention
    with reduced_audio_context(engine) as (view, report):
        assert report["context_pixels"] == 704 * 704
        assert view.denoiser is engine.denoiser and view.input_packer is engine.input_packer
        assert view.refined_text is engine.refined_text
        assert view.complete and view.latent_state.evaluation_index == 4
        assert torch.equal(view.rotary_cos, engine.rotary_cos[selection.packed_rows])
        assert torch.equal(view.rotary_sin, engine.rotary_sin[selection.packed_rows])
        assert torch.equal(
            view.latent_state.video_latents, original_video[:, selection.video_rows]
        )
        view.latent_state.video_latents.zero_()
        view.latent_state.audio_latents.zero_()
        assert torch.equal(engine.latent_state.video_latents, original_video)
        assert torch.equal(engine.latent_state.audio_latents, original_audio)
        assert engine.denoiser.blocks[0]._veda_attention is not attention
    assert all(b._veda_attention is attention for b in engine.denoiser.blocks)


@pytest.mark.parametrize("fail", [False, True])
def test_six_step_result_commits_only_audio_and_failure_preserves_source(monkeypatch, fail):
    import vflash.native.h3_audio_refinement as module

    monkeypatch.setattr(module, "_attention_view", fake_attention)
    engine = context_engine()
    state = engine.latent_state
    video, audio = state.video_latents.clone(), state.audio_latents.clone()
    attention = engine.denoiser.blocks[0]._veda_attention
    weights = [b.weights for b in engine.denoiser.blocks]
    final = engine.final_layer.weights
    schedule = refinement_schedule(steps=6, denoise=0.5)
    tables = [torch.zeros(7, 9, 6, 2) for _ in range(50)] + [torch.zeros(7, 3, 2, 2)]
    evaluations = []

    def step(view, **kwargs):
        s = view.latent_state
        assert s.video_latents.shape[1] == 1452 and s.num_condition_video_rows == 1452
        assert view.refined_text is engine.refined_text
        assert all(not b.weights.qkv_residuals for b in view.denoiser.blocks)
        evaluations.append(s.evaluation_index)
        s.step(
            video_velocity=torch.ones_like(s.video_latents),
            audio_velocity=torch.ones_like(s.audio_latents),
        )
        if fail:
            raise RuntimeError("injected audio step")

    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            refine_audio(
                engine,
                AudioRefinementRequest(schedule, tuple(tables)),
                step=step,
                noise=torch.zeros_like(audio),
            )
        assert torch.equal(state.audio_latents, audio)
    else:
        baseline, report = refine_audio(
            engine,
            AudioRefinementRequest(schedule, tuple(tables)),
            step=step,
            noise=torch.zeros_like(audio),
        )
        assert evaluations == [1, 2, 3, 4, 5, 6]
        assert torch.equal(baseline, audio) and not torch.equal(state.audio_latents, audio)
        assert report["extra_audio_nfe"] == 6 and report["context_reduction"]["approximate"]
    assert torch.equal(state.video_latents, video)
    assert engine.latent_state is state and state.evaluation_index == 4
    assert all(
        b._veda_attention is attention and b.weights is w
        for b, w in zip(engine.denoiser.blocks, weights, strict=True)
    )
    assert engine.final_layer.weights is final
    assert (attention.sparse_calls, attention.dense_calls) == (160, 40)


def test_small_context_stays_original_and_undeclared_layouts_fail():
    engine = context_engine(16, 24)
    assert select_context(engine) is None
    with reduced_audio_context(engine) as (view, report):
        assert view is engine and not report["applied"]
    engine = context_engine()
    engine.latent_state.num_condition_video_rows += 1
    with pytest.raises(ValueError, match="same-canvas"):
        select_context(engine)
    with pytest.raises(ValueError, match=r"0\.5MP"):
        select_context(context_engine(), max_pixels=250_000)


def test_request_rejects_unmatched_schedule_and_unqualified_budget():
    schedule = refinement_schedule(steps=6, denoise=0.5)
    tables = tuple([torch.zeros(7, 9, 6, 2) for _ in range(50)] + [torch.zeros(7, 3, 2, 2)])
    request = AudioRefinementRequest(schedule, tables)
    assert request.steps == 6 and request.context_pixels == 500000
    for bad in [True, 250000, 0]:
        with pytest.raises(ValueError, match=r"0\.5MP"):
            AudioRefinementRequest(schedule, tables, context_pixels=bad)
    with pytest.raises(ValueError, match="matched six-step"):
        AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.25), tables)
    with pytest.raises(ValueError, match="matched six-step"):
        AudioRefinementRequest(schedule, tables[:-1])


def test_full_context_reports_each_added_evaluation_and_restores_video():
    engine = context_engine()
    state = engine.latent_state
    video = state.video_latents.clone()
    schedule = refinement_schedule(steps=6, denoise=0.5)
    tables = tuple([torch.zeros(7, 9, 6, 2) for _ in range(50)] + [torch.zeros(7, 3, 2, 2)])
    progress = []

    def step(owner, **kwargs):
        assert owner is engine
        owner.latent_state.step(
            video_velocity=torch.ones_like(video),
            audio_velocity=torch.ones_like(state.audio_latents),
        )

    _, result = refine_audio(
        engine,
        AudioRefinementRequest(schedule, tables, context_pixels=None),
        noise=torch.zeros_like(state.audio_latents),
        step=step,
        progress_callback=lambda n, total: progress.append((n, total)),
    )
    assert progress == [(i, 6) for i in range(1, 7)]
    assert "context_reduction" not in result and result["extra_audio_nfe"] == 6
    assert engine.latent_state is state and state.evaluation_index == 4
    assert torch.equal(video, state.video_latents)
