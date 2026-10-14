from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vflash.native.h3_native_scheduler import H3NativeLatentState, H3NativeSchedule
from vflash.native.h3_refinement import VideoRepaint, refinement_plan
from vflash.native.h3_refinement_schedule import refinement_schedule

torch = pytest.importorskip("torch")


@dataclass
class BlockWeights:
    adaln_table: object
    qkv_residuals: tuple = (1, 2, 3)
    attention_out_residual: object = 1
    ffn_in_residual: object = 1
    ffn_out_residual: object = 1


@dataclass
class FinalWeights:
    adaln_table: object
    video_head_plans: object
    audio_head_plans: object


def engine_fixture():
    schedule = H3NativeSchedule.shifted_linear(4)
    state = H3NativeLatentState(
        schedule,
        video_latents=torch.rand(1, 4, 3),
        audio_latents=torch.rand(1, 2, 32),
        num_condition_video_rows=1,
        num_condition_audio_rows=0,
    )
    state.evaluation_index = 4
    attention = SimpleNamespace(sparse_calls=160, dense_calls=40)
    blocks = tuple(
        SimpleNamespace(
            weights=BlockWeights(torch.zeros(4, 9, 6, 2)), _veda_attention=attention
        )
        for _ in range(50)
    )
    return SimpleNamespace(
        schedule=schedule,
        latent_state=state,
        complete=True,
        video_indices=torch.arange(4),
        audio_indices=torch.arange(4, 6),
        text_indices=torch.tensor([6]),
        denoiser=SimpleNamespace(blocks=blocks),
        _denoise_profile_records=None,
        row_timestep_plan=(),
        final_layer=SimpleNamespace(
            weights=FinalWeights(
                torch.zeros(4, 3, 2, 2), torch.ones(4, 1, 1), torch.ones(4, 1, 1)
            )
        ),
    )


def test_low_noise_repaint_is_not_a_high_sigma_turbo_tail():
    face = refinement_schedule(steps=4, denoise=0.035)
    assert face.nfe == 5 and 0.29 < face.video_sigmas[1] < 0.31
    assert face.video_sigmas[-1] == 0 and len(face.video_sigmas[1:-1]) == 4
    audio = refinement_schedule(steps=6, denoise=0.5)
    assert audio.nfe == 7 and 0.74 < audio.audio_sigmas[1] < 0.76
    for value in (0, 1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            refinement_schedule(steps=6, denoise=value)


@pytest.mark.parametrize("conditions", [0, 1, 4])
@pytest.mark.parametrize("mode", ["audio", "face"])
def test_new_schedule_row_indices_match_compiled_modulation(conditions, mode):
    engine = engine_fixture()
    engine.latent_state.num_condition_video_rows = conditions
    schedule = refinement_schedule(steps=6, denoise=0.5)
    plans = refinement_plan(engine, schedule, mode=mode)
    for index, plan in enumerate(plans):
        times = plan.timesteps[plan.timestep_indices]
        assert torch.all(
            times[engine.video_indices if mode == "audio" else engine.audio_indices] == 0.999
        )
        if mode == "audio":
            assert torch.all(
                times[engine.audio_indices] == torch.tensor(1 - schedule.audio_sigmas[index])
            )
        elif conditions < 4:
            assert torch.all(
                times[engine.video_indices[conditions:]]
                == torch.tensor(1 - schedule.video_sigmas[index])
            )


def test_face_low_noise_replaces_generated_suffix_and_keeps_prefix_audio():
    engine = engine_fixture()
    state = engine.latent_state
    state.evaluation_index = 0
    attention = engine.denoiser.blocks[0]._veda_attention
    attention.sparse_calls = attention.dense_calls = 0
    video, audio = state.video_latents.clone(), state.audio_latents.clone()
    schedule = refinement_schedule(steps=4, denoise=0.035)
    tables = [torch.zeros(5, 9, 6, 2) for _ in range(50)] + [torch.zeros(5, 3, 2, 2)]
    encoded = torch.full_like(video[:, 1:], 0.4)
    calls = []

    def step(engine, **kwargs):
        attention.sparse_calls += 40
        attention.dense_calls += 10
        calls.append(engine.latent_state.evaluation_index)
        assert engine.latent_state.num_condition_audio_rows == 2
        assert engine.denoiser.blocks[0].weights.qkv_residuals
        engine.latent_state.step(
            video_velocity=torch.ones_like(video) * 0.1, audio_velocity=torch.ones_like(audio)
        )
        return "completed"

    repaint = VideoRepaint(engine, schedule, tables, encoded)
    for expected in range(1, 5):
        result = repaint.advance(step=step)
        assert state.evaluation_index == expected
        assert engine.latent_state is state and engine.schedule is state.schedule
    record = repaint.record
    assert calls == [1, 2, 3, 4] and result == "completed"
    assert (attention.sparse_calls, attention.dense_calls) == (160, 40)
    assert record["sigma"] < 0.31 and state.evaluation_index == 4
    assert torch.equal(state.audio_latents, audio) and torch.equal(
        state.video_latents[:, :1], video[:, :1]
    )
    assert not torch.equal(state.video_latents[:, 1:], video[:, 1:])


def test_face_failure_restores_original_schedule_weights_and_latents():
    engine = engine_fixture()
    state = engine.latent_state
    state.evaluation_index = 0
    weights = tuple(b.weights for b in engine.denoiser.blocks)
    final = engine.final_layer.weights
    video, audio = state.video_latents.clone(), state.audio_latents.clone()
    schedule = refinement_schedule(steps=4, denoise=0.035)
    tables = [torch.zeros(5, 9, 6, 2) for _ in range(50)] + [torch.zeros(5, 3, 2, 2)]
    repaint = VideoRepaint(engine, schedule, tables, torch.full_like(video[:, 1:], 0.4))

    def fail(engine, **kwargs):
        engine.latent_state.video_latents.add_(0.1)
        raise RuntimeError("injected quality failure")

    with pytest.raises(RuntimeError, match="quality failure"):
        repaint.advance(step=fail)
    assert engine.latent_state is state and engine.schedule is state.schedule
    assert engine.final_layer.weights is final
    assert all(b.weights is w for b, w in zip(engine.denoiser.blocks, weights, strict=True))
    assert state.evaluation_index == 0
    assert torch.equal(state.video_latents, video) and torch.equal(state.audio_latents, audio)


def test_bf16_finite_check_covers_every_signed_exponent_pattern():
    from vflash.native.h3_refinement_schedule import finite_cpu_bf16

    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    finite = torch.isfinite(values)
    assert finite_cpu_bf16(values[finite])
    assert not finite_cpu_bf16(values)
    for value in (float("nan"), float("inf"), -float("inf")):
        assert not finite_cpu_bf16(torch.tensor([0, value], dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="CPU BF16"):
        finite_cpu_bf16(torch.ones(1))


def test_table_loader_rejects_unknown_modality_before_file_access(tmp_path):
    from vflash.native.h3_refinement_schedule import load_tables

    with pytest.raises(ValueError, match="unknown refinement"):
        load_tables(tmp_path / "missing", mode="invalid")
