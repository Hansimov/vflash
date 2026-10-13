import shutil

import pytest

from vflash.contracts import ContractError
from vflash.native.h3_audio_mode import without_generated_audio
from vflash.native.h3_native_scheduler import (
    H3NativeLatentState,
    H3NativeSchedule,
    compile_h3_row_timestep_plan,
)
from vflash.pipeline.contracts import VideoRequest

torch = pytest.importorskip("torch")


def fixture():
    return dict(
        initial_video=torch.randn(1, 5, 96),
        initial_audio=torch.randn(1, 4, 32),
        refined_text=torch.randn(1, 2, 8, dtype=torch.bfloat16),
        video_indices=torch.tensor([2, 7, 8, 9, 10]),
        audio_indices=torch.tensor([3, 4, 5, 6]),
        text_indices=torch.tensor([0, 1]),
        token_tags=torch.tensor([0, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1]),
        position_ids=torch.arange(33).reshape(11, 3),
        rotary_cos=torch.randn(11, 4),
        rotary_sin=torch.randn(11, 4),
    )


@pytest.mark.parametrize("nfe", [4, 16, 40])
@pytest.mark.parametrize("conditions", [0, 1])
def test_remove_audio_preserves_original_adaln_indices_and_video_noise(nfe, conditions):
    original = fixture()
    snapshot = {k: v.clone() for k, v in original.items()}
    candidate = without_generated_audio(original, condition_audio_rows=0)
    schedule = H3NativeSchedule.shifted_linear(nfe)

    def plans(t):
        return compile_h3_row_timestep_plan(
            schedule,
            video_indices=t["video_indices"],
            audio_indices=t["audio_indices"],
            text_indices=t["text_indices"],
            num_condition_video_rows=conditions,
            num_condition_audio_rows=0,
            device="cpu",
        )

    retained = torch.tensor([0, 1, 2, 7, 8, 9, 10])
    for before, after in zip(plans(original), plans(candidate), strict=True):
        torch.testing.assert_close(after.timesteps, before.timesteps, rtol=0, atol=0)
        torch.testing.assert_close(
            after.timestep_indices, before.timestep_indices[retained], rtol=0, atol=0
        )
    assert candidate["initial_audio"].shape == (1, 0, 32)
    assert candidate["audio_indices"].numel() == 0
    assert candidate["initial_video"] is original["initial_video"]
    for name in ("position_ids", "rotary_cos", "rotary_sin", "token_tags"):
        torch.testing.assert_close(candidate[name], original[name][retained])
    for k in snapshot:
        torch.testing.assert_close(original[k], snapshot[k], rtol=0, atol=0)


def test_video_only_rejects_audio_references_and_invalid_boolean():
    with pytest.raises(ValueError, match="audio conditioning"):
        without_generated_audio(fixture(), condition_audio_rows=1)
    with pytest.raises(ContractError):
        VideoRequest("test", generate_audio=0)
    with pytest.raises(ContractError):
        VideoRequest("test", generate_audio=False, audio_delivery_profile="web-v1")
    assert VideoRequest("test").generate_audio
    assert not VideoRequest("test", generate_audio=False).generate_audio


def test_zero_audio_latent_state_skips_audio_euler(monkeypatch):
    from vflash.native import h3_native_scheduler as scheduler

    calls = []
    original = scheduler.h3_training_euler_step

    def recorded(sample, *args, **kwargs):
        calls.append(sample.shape)
        assert sample.numel() > 0
        return original(sample, *args, **kwargs)

    monkeypatch.setattr(scheduler, "h3_training_euler_step", recorded)
    state = H3NativeLatentState(
        H3NativeSchedule.shifted_linear(4),
        video_latents=torch.zeros(1, 5, 96),
        audio_latents=torch.empty(1, 0, 32),
        num_condition_video_rows=1,
        num_condition_audio_rows=0,
    )
    for _ in range(4):
        state.step(video_velocity=torch.ones(1, 5, 96), audio_velocity=torch.empty(1, 0, 32))
    assert state.complete and len(calls) == 4
    assert not state.video_latents[:, 0].any()
    assert state.video_latents[:, 1:].abs().sum() > 0


def test_video_only_packer_and_head_never_call_audio_projection(monkeypatch):
    import torch.nn.functional as F

    from vflash.native.h3_native_final_layer import (
        H3NativeFinalLayerWeights,
        _H3FinalLayerOperations,
    )
    from vflash.native.h3_native_input_stage import (
        H3NativeInputProjectionWeights,
        _H3InputPackingOperations,
    )

    hidden = 8
    weights = H3NativeInputProjectionWeights(
        torch.randn(hidden, 96),
        torch.randn(hidden),
        torch.full((hidden, 32), float("nan")),
        torch.full((hidden,), float("nan")),
        hidden,
        96,
        32,
    )
    pack = _H3InputPackingOperations(weights)
    t = without_generated_audio(fixture(), condition_audio_rows=0)
    calls = []
    linear = F.linear

    def observed(x, w, b=None):
        assert x.numel() > 0 and torch.isfinite(w).all()
        calls.append(w.shape)
        return linear(x, w, b)

    monkeypatch.setattr(F, "linear", observed)
    packed = pack(
        video_latents=t["initial_video"],
        audio_latents=t["initial_audio"],
        refined_text=t["refined_text"],
        video_indices=t["video_indices"],
        audio_indices=t["audio_indices"],
        text_indices=t["text_indices"],
        sequence_length=7,
    )
    assert len(calls) == 1 and torch.isfinite(packed).all()
    final_weights = H3NativeFinalLayerWeights(
        torch.zeros(4, 3, 2, hidden, dtype=torch.bfloat16),
        torch.ones(hidden, dtype=torch.bfloat16),
        torch.randn(1, 96, hidden),
        torch.randn(1, 96),
        torch.full((1, 32, hidden), float("nan")),
        torch.full((1, 32), float("nan")),
        torch.ones(4, 1, 1),
        torch.ones(4, 1, 1),
        "test",
    )
    head = _H3FinalLayerOperations(final_weights)
    video, audio = head(
        packed,
        evaluation_index=0,
        timestep_indices=torch.zeros(7, dtype=torch.int64),
        video_indices=t["video_indices"],
        audio_indices=t["audio_indices"],
    )
    assert len(calls) == 2 and video.shape == (1, 5, 96) and torch.isfinite(video).all()
    assert audio.shape == (1, 0, 32)


def test_video_only_decoder_does_not_read_or_decode_audio(monkeypatch):
    from vflash.media import runtime

    decoder = object.__new__(runtime.OfficialMediaDecoder)
    decoder._torch = torch
    decoder.device = torch.device("cpu")
    decoder.video = object()
    decoder.audio = object()

    def load(path, names):
        assert names == ("video_latents",)
        return {"video_latents": torch.zeros(1, 24, 2, 2, 2)}

    monkeypatch.setattr(runtime, "load_safetensor_tensors", load)
    monkeypatch.setattr(
        runtime,
        "decode_official_h3_video_latents",
        lambda *a, **k: torch.zeros(1, 3, 24, 32, 32),
    )
    monkeypatch.setattr(
        runtime,
        "decode_official_h3_audio_latents",
        lambda *a, **k: pytest.fail("audio decoded"),
    )
    _video, audio, stages = decoder._decode_cpu(None, height=32, width=32, decode_audio=False)
    assert audio is None and stages["audio_decode_and_copy"] == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg")
@pytest.mark.parametrize("transport", ["pipe", "file"])
def test_actual_video_only_mp4_has_no_audio_track(tmp_path, transport):
    from vflash.media.encoding import encode_mp4

    video = torch.rand(1, 3, 24, 32, 32)
    result = encode_mp4(
        video,
        None,
        tmp_path / "mute.mp4",
        preset="ultrafast",
        video_input=transport,
        video_threads=2,
    )
    assert result["audio_samples"] == 0 and result["audio_sample_rate"] is None
    assert [x["codec_type"] for x in result["probe"]["streams"]] == ["video"]
    assert result["frames"] == 24 and result["duration_seconds"] == 1
