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


def context_engine(height=32, width=32, edges=3):
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
    attention.dense_layers = frozenset((*range(edges), *range(50 - edges, 50)))
    return engine


def fake_attention(original, selection):
    result = copy(original)
    result.layout = N(target=N(grid=selection.grid))
    result.sparse_calls = result.dense_calls = 0
    return result


@pytest.mark.parametrize("edges", [3, 4])
def test_context_keeps_every_time_step_audio_text_and_original_rope(monkeypatch, edges):
    import vflash.native.h3_audio_refinement as module

    monkeypatch.setattr(module, "_attention_view", fake_attention)
    engine = context_engine(edges=edges)
    selection = select_context(engine)
    assert selection.source_grid == (2, 32, 32) and selection.grid == (2, 22, 22)
    assert selection.condition_rows == 484
    assert selection.audio_indices.numel() == 4 and selection.text_indices.numel() == 5
    original_video = engine.latent_state.video_latents.clone()
    original_audio = engine.latent_state.audio_latents.clone()
    attention = engine.denoiser.blocks[0]._veda_attention
    with reduced_audio_context(engine) as (view, report):
        assert report["dense_layers"] == sorted(attention.dense_layers)
        assert len(view.denoiser.blocks[0]._veda_attention.dense_layers) == 2 * edges
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
    with pytest.raises(ValueError, match=r"0\.25MP or 0\.5MP"):
        select_context(context_engine(), max_pixels=300_000)


def test_quarter_megapixel_context_keeps_every_time_step(monkeypatch):
    import vflash.native.h3_audio_refinement as module

    monkeypatch.setattr(module, "_attention_view", fake_attention)
    engine = context_engine()
    selection = select_context(engine, max_pixels=250_000)
    # sqrt(250000 / (32*32*1024)) of each 32-row side, rounded down.
    assert selection.source_grid == (2, 32, 32) and selection.grid == (2, 15, 15)
    assert selection.condition_rows == 225
    assert selection.audio_indices.numel() == 4 and selection.text_indices.numel() == 5
    with reduced_audio_context(engine, max_pixels=250_000) as (view, report):
        assert report["context_pixels"] == 480 * 480
        assert torch.equal(view.rotary_cos, engine.rotary_cos[selection.packed_rows])


def test_request_rejects_unmatched_schedule_and_unqualified_budget():
    schedule = refinement_schedule(steps=6, denoise=0.5)
    tables = tuple([torch.zeros(7, 9, 6, 2) for _ in range(50)] + [torch.zeros(7, 3, 2, 2)])
    request = AudioRefinementRequest(schedule, tables)
    assert request.steps == 6 and request.context_pixels == 500000
    assert request.reuse_hidden_context is False
    assert request.reuse_kv_context is False
    with pytest.raises(ValueError, match="at most one"):
        AudioRefinementRequest(
            schedule, tables, reuse_hidden_context=True, reuse_kv_context=True
        )
    for bad in (1, "true", None):
        with pytest.raises(ValueError, match="explicit boolean"):
            AudioRefinementRequest(schedule, tables, reuse_hidden_context=bad)
    assert (
        AudioRefinementRequest(schedule, tables, context_pixels=250000).context_pixels == 250000
    )
    for bad in [True, 300000, 0]:
        with pytest.raises(ValueError, match=r"0\.25MP or 0\.5MP"):
            AudioRefinementRequest(schedule, tables, context_pixels=bad)
    with pytest.raises(ValueError, match="matched audio schedule"):
        AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.25), tables)
    with pytest.raises(ValueError, match="matched audio schedule"):
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


def test_encoded_audio_repair_seeds_targets_keeps_keyframe_and_owns_storage():
    from vflash.native.h3_audio_refinement import (
        EncodedAudioSource,
        install_encoded_audio_source,
    )

    engine = context_engine()
    state = engine.latent_state
    state.evaluation_index = 0
    prefix = state.video_latents[:, : state.num_condition_video_rows].clone()
    video = torch.full_like(state.video_latents[:, state.num_condition_video_rows :], 0.25)
    audio = torch.full_like(state.audio_latents, 0.125)
    record = install_encoded_audio_source(engine, EncodedAudioSource(video, audio))
    assert engine.complete and record["video_nfe"] == 0
    assert torch.equal(state.video_latents[:, : state.num_condition_video_rows], prefix)
    assert torch.equal(state.video_latents[:, state.num_condition_video_rows :], video)
    assert torch.equal(state.audio_latents, audio)
    video.zero_()
    audio.zero_()
    assert state.audio_latents.eq(0.125).all()
    assert state.video_latents[:, state.num_condition_video_rows :].eq(0.25).all()


@pytest.mark.parametrize("fault", ["video_shape", "audio_shape", "nan", "started"])
def test_invalid_encoded_source_leaves_owned_request_unchanged(fault):
    from vflash.native.h3_audio_refinement import (
        EncodedAudioSource,
        install_encoded_audio_source,
    )

    engine = context_engine()
    state = engine.latent_state
    state.evaluation_index = 1 if fault == "started" else 0
    original_video, original_audio = state.video_latents.clone(), state.audio_latents.clone()
    video = torch.ones_like(state.video_latents[:, state.num_condition_video_rows :])
    audio = torch.ones_like(state.audio_latents)
    if fault == "video_shape":
        video = video[:, :-1]
    if fault == "audio_shape":
        audio = audio[:, :-1]
    if fault == "nan":
        audio[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        install_encoded_audio_source(engine, EncodedAudioSource(video, audio))
    assert torch.equal(state.video_latents, original_video)
    assert torch.equal(state.audio_latents, original_audio)
    assert state.evaluation_index == (1 if fault == "started" else 0)


def test_twenty_step_restoration_keeps_picture_and_uses_declared_schedule():
    engine = context_engine(16, 24)
    state = engine.latent_state
    picture = state.video_latents.clone()
    original_audio = state.audio_latents.clone()
    schedule = refinement_schedule(steps=20, denoise=0.95)
    tables = tuple([torch.zeros(21, 9, 6, 2) for _ in range(50)] + [torch.zeros(21, 3, 2, 2)])
    request = AudioRefinementRequest(
        schedule, tables, context_pixels=None, evaluations=20, denoise=0.95
    )
    evaluated = []

    def step(view, **kwargs):
        evaluated.append(view.latent_state.evaluation_index)
        view.latent_state.step(
            video_velocity=torch.zeros_like(view.latent_state.video_latents),
            audio_velocity=torch.ones_like(view.latent_state.audio_latents),
        )

    _, report = refine_audio(engine, request, step=step, noise=torch.zeros_like(original_audio))
    assert evaluated == list(range(1, 21))
    assert report["extra_audio_nfe"] == 20
    assert torch.equal(state.video_latents, picture)
    assert not torch.equal(state.audio_latents, original_audio)
    with pytest.raises(ValueError, match="matched audio schedule"):
        AudioRefinementRequest(schedule, tables)


@pytest.mark.parametrize(
    "options",
    [
        {"context_pixels": 250000},
        {"reuse_hidden_context": True},
        {"reuse_kv_context": True},
    ],
)
def test_twenty_step_restoration_requires_qualified_full_context(options):
    schedule = refinement_schedule(steps=20, denoise=0.95)
    tables = tuple([torch.zeros(21, 9, 6, 2) for _ in range(50)] + [torch.zeros(21, 3, 2, 2)])
    with pytest.raises(ValueError, match="full uncached context"):
        AudioRefinementRequest(
            schedule,
            tables,
            **{"context_pixels": None, **options},
            evaluations=20,
            denoise=0.95,
        )
