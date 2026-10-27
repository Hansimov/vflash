from types import SimpleNamespace

import pytest

from vflash.native.h3_av_continuation import NativeAVPrefix, NativeAVPreservation
from vflash.native.h3_native_scheduler import (
    H3NativeLatentState,
    H3NativeSchedule,
    compile_h3_row_timestep_plan,
)

torch = pytest.importorskip("torch")


def fixture(context=39):
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
    video, audio = torch.randn(1, 16 + 37 * 16, 96), torch.randn(1, 414, 32)
    state = H3NativeLatentState(
        schedule,
        video_latents=video,
        audio_latents=audio,
        num_condition_video_rows=16,
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
        num_condition_video_rows=16,
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
