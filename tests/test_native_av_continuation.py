from types import SimpleNamespace

import pytest

from vflash.native.h3_av_continuation import NativeAVPrefix, NativeAVPreservation
from vflash.native.h3_native_scheduler import (
    H3NativeLatentState,
    H3NativeSchedule,
    compile_h3_row_timestep_plan,
)

torch = pytest.importorskip("torch")


def fixture(context=39, keyframes=1):
    torch.manual_seed(771)
    source = {
        "video_latents": torch.randn(1, 24, 42, 8, 8),
        "audio_latents": torch.randn(2, 32, 235),
    }
    block = torch.zeros(1, 3, 6, 5376)
    tables = (block,) * 50 + (torch.zeros(1, 1, 2, 5376),)
    prefix = NativeAVPrefix.from_latents(
        source, source_frames=141, context_frames=context, clean_timestep_tables=tables
    )
    schedule = H3NativeSchedule.shifted_linear(4)
    video, audio = torch.randn(1, 16 * keyframes + 37 * 16, 96), torch.randn(1, 414, 32)
    state = H3NativeLatentState(
        schedule,
        video_latents=video,
        audio_latents=audio,
        num_condition_video_rows=16 * keyframes,
        num_condition_audio_rows=0,
    )
    vi, ai, ti = (
        torch.arange(video.shape[1]),
        torch.arange(414) + video.shape[1],
        torch.arange(3) + video.shape[1] + 414,
    )
    plan = compile_h3_row_timestep_plan(
        schedule,
        video_indices=vi,
        audio_indices=ai,
        text_indices=ti,
        num_condition_video_rows=16 * keyframes,
        num_condition_audio_rows=0,
        device="cpu",
    )
    engine = SimpleNamespace(
        schedule=schedule,
        latent_state=state,
        video_indices=vi,
        audio_indices=ai,
        row_timestep_plan=plan,
        _denoise_profile_records=None,
        step=lambda: state.step(
            video_velocity=torch.ones_like(video), audio_velocity=torch.ones_like(audio)
        ),
    )
    return source, prefix, engine


@pytest.mark.parametrize("context", [39, 90])
def test_context_survives_every_evaluation_and_labels_both_modalities(context):
    source, prefix, engine = fixture(context)
    state, original_plan = engine.latent_state, engine.row_timestep_plan
    original_image = state.video_latents[:, :16].clone()
    original_tail = state.video_latents[:, -16:].clone()
    scope = NativeAVPreservation(engine, prefix)
    with scope.preserve_targets():
        for row in engine.row_timestep_plan:
            assert (
                row.timesteps[row.timestep_indices[engine.video_indices[16 : 16 + scope.rows]]]
                .eq(0.999)
                .all()
            )
            indices = engine.audio_indices.reshape(2, -1)[:, : scope.ticks]
            assert row.timesteps[row.timestep_indices[indices]].eq(1).all()
        for _ in range(4):
            scope.advance()
            assert torch.equal(
                state.audio_latents.reshape(2, -1, 32)[:, : scope.ticks], scope.audio
            )
            expected = scope.video if state.complete else scope.kept
            assert torch.equal(state.video_latents[:, 16 : 16 + scope.rows], expected)
    assert scope.record()["evaluations"] == 4
    assert engine.row_timestep_plan is original_plan
    assert torch.equal(state.video_latents[:, :16], original_image)
    assert not torch.equal(state.video_latents[:, -16:], original_tail)
    # The request's retained context is independent of later source mutations.
    source["video_latents"].zero_()
    assert prefix.video.abs().sum() > 0
    with pytest.raises(ValueError, match="reused"), scope.preserve_targets():
        pass


def test_cancelled_scope_restores_only_its_own_plan_and_never_changes_global_step():
    _, prefix, engine = fixture()
    original_plan, original_step = engine.row_timestep_plan, H3NativeLatentState.step
    scope = NativeAVPreservation(engine, prefix)
    with pytest.raises(RuntimeError, match="cancelled"), scope.preserve_targets():
        scope.advance()
        raise RuntimeError("cancelled")
    assert engine.row_timestep_plan is original_plan
    assert H3NativeLatentState.step is original_step
    with pytest.raises(ValueError, match="incomplete"):
        scope.record()
    with pytest.raises(ValueError, match="not active"):
        scope.advance()


@pytest.mark.parametrize(
    "source_frames,context_frames", [(140, 39), (141, 22), (90, 141), (90, True)]
)
def test_rejects_different_video_audio_clock_boundaries(source_frames, context_frames):
    source, prefix, _ = fixture()
    with pytest.raises(ValueError, match="boundaries"):
        NativeAVPrefix.from_latents(
            source,
            source_frames=source_frames,
            context_frames=context_frames,
            clean_timestep_tables=prefix.clean_timestep_tables,
        )


@pytest.mark.parametrize("frames", [90, 107, 124, 141])
def test_join_uses_global_clock_over_twenty_four_continuations(frames):
    from vflash.native.h3_latent_layout import h3_video_latent_frame_count

    _, old, _ = fixture()
    source = {
        "video_latents": torch.randn(1, 24, h3_video_latent_frame_count(frames), 2, 2),
        "audio_latents": torch.randn(2, 32, (frames * 5 + 1) // 3),
    }
    for _ in range(24):
        prefix = NativeAVPrefix.from_latents(
            source,
            source_frames=frames,
            context_frames=39,
            clean_timestep_tables=old.clean_timestep_tables,
        )
        original_audio = source["audio_latents"].clone()
        generated = {
            "video_latents": torch.randn(1, 24, 37, 2, 2),
            "audio_latents": torch.randn(2, 32, 207),
        }
        generated["video_latents"][:, :, :12] = prefix.video
        generated["audio_latents"][:, :, :65] = prefix.audio
        source = prefix.join(source, generated, generated_frames=124)
        frames += 85
        assert source["audio_latents"].shape[-1] == (frames * 5 + 1) // 3
        assert abs(source["audio_latents"].shape[-1] / 40 - frames / 24) <= 1 / 120 + 1e-10
        assert torch.equal(
            source["audio_latents"][:, :, : original_audio.shape[-1]], original_audio
        )
        assert source["video_latents"].shape[2] == h3_video_latent_frame_count(frames)


def test_join_rejects_changed_source_prefix_and_missing_audio():
    source, prefix, _ = fixture()
    generated = {
        "video_latents": torch.randn(1, 24, 37, 8, 8),
        "audio_latents": torch.randn(2, 32, 207),
    }
    generated["video_latents"][:, :, :12] = prefix.video
    generated["audio_latents"][:, :, :65] = prefix.audio
    generated["video_latents"][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="exactly preserved"):
        prefix.join(source, generated, generated_frames=124)
    generated["video_latents"][:, :, :12] = prefix.video
    generated["audio_latents"] = generated["audio_latents"][:, :, :-2]
    with pytest.raises(ValueError, match="clock is incomplete"):
        prefix.join(source, generated, generated_frames=124)


def test_audio_only_history_keeps_sound_while_all_target_video_is_generated():
    from dataclasses import replace

    source, prefix, engine = fixture()
    prefix = replace(prefix, preserve_video=False)
    state = engine.latent_state
    before = state.video_latents.clone()
    scope = NativeAVPreservation(engine, prefix)
    assert scope.rows == 0
    with scope.preserve_targets():
        for _ in range(4):
            scope.advance()
            assert torch.equal(
                state.audio_latents.reshape(2, -1, 32)[:, : scope.ticks], scope.audio
            )
    assert torch.equal(state.video_latents[:, :16], before[:, :16])
    assert not torch.equal(state.video_latents[:, 16 : 16 + 192], before[:, 16 : 16 + 192])
    assert scope.record()["video_prefix_exact"] is None
    generated = {"audio_latents": torch.randn(2, 32, 207)}
    generated["audio_latents"][:, :, :65] = prefix.audio
    joined = prefix.join_audio(source, generated, generated_frames=124)
    assert joined.shape == (2, 32, 377)
    assert torch.equal(joined[:, :, :235], source["audio_latents"])
    with pytest.raises(ValueError, match="independently cut pictures"):
        prefix.join(source, generated, generated_frames=124)


def test_explicit_delivery_can_discard_an_unavailable_model_padding_tick():
    from vflash.native.h3_latent_layout import h3_video_latent_frame_count

    _, old, _ = fixture()
    source = {
        "video_latents": torch.randn(1, 24, h3_video_latent_frame_count(158), 2, 2),
        "audio_latents": torch.randn(2, 32, 263),
    }
    prefix = NativeAVPrefix.from_latents(
        source,
        source_frames=158,
        context_frames=39,
        clean_timestep_tables=old.clean_timestep_tables,
        preserve_video=False,
    )
    generated = {"audio_latents": torch.randn(2, 32, 263)}
    generated["audio_latents"][:, :, :65] = prefix.audio
    with pytest.raises(ValueError, match="clock is incomplete"):
        prefix.join_audio(source, generated, generated_frames=158)
    joined = prefix.join_audio(source, generated, generated_frames=158, delivery_frames=240)
    assert joined.shape[-1] == 461  # 10s delivery needs400; retain the real decoder padding.
    assert torch.equal(joined[:, :, :263], source["audio_latents"])
    for invalid in [277, 278, True, 0]:
        with pytest.raises(ValueError):
            prefix.join_audio(source, generated, generated_frames=158, delivery_frames=invalid)
    generated["audio_latents"] = generated["audio_latents"][:, :, :-2]
    with pytest.raises(ValueError, match="clock is incomplete"):
        prefix.join_audio(source, generated, generated_frames=158, delivery_frames=240)


def test_two_condition_keyframes_remain_untouched_by_target_history():
    _, prefix, engine = fixture(keyframes=2)
    state = engine.latent_state
    conditions = state.video_latents[:, :32].clone()
    scope = NativeAVPreservation(engine, prefix)
    assert scope.offset == 32
    with scope.preserve_targets():
        for _ in range(4):
            scope.advance()
            assert torch.equal(state.video_latents[:, :32], conditions)
    assert scope.record()["video_prefix_exact"]
    assert scope.record()["audio_prefix_exact"]


def test_explicit_delivery_retains_the_real_audio_tick_covering_its_last_video_frame():
    _, old, _ = fixture()
    source = {
        "video_latents": torch.randn(1, 24, 37, 2, 2),
        "audio_latents": torch.randn(2, 32, 207),
    }
    prefix = NativeAVPrefix.from_latents(
        source,
        source_frames=124,
        context_frames=39,
        clean_timestep_tables=old.clean_timestep_tables,
    )
    generated = {
        "video_latents": torch.randn(1, 24, 37, 2, 2),
        "audio_latents": torch.randn(2, 32, 207),
    }
    generated["video_latents"][:, :, :12] = prefix.video
    generated["audio_latents"][:, :, :65] = prefix.audio
    legacy = prefix.join(source, generated, generated_frames=124)
    delivered = prefix.join(source, generated, generated_frames=124, delivery_frames=209)
    assert legacy["audio_latents"].shape[-1] == 348
    assert delivered["audio_latents"].shape[-1] == 349
    assert torch.equal(
        delivered["audio_latents"][:, :, -1], generated["audio_latents"][:, :, -1]
    )
    assert delivered["audio_latents"].shape[-1] / 40 >= 209 / 24
    assert torch.equal(delivered["video_latents"], legacy["video_latents"])
    # A genuinely absent delivery tick remains an error; never pad it.
    generated["audio_latents"] = generated["audio_latents"][:, :, :-1]
    with pytest.raises(ValueError, match="clock is incomplete"):
        prefix.join(source, generated, generated_frames=124, delivery_frames=209)
