"""Exercise the same-process conditioning handoff without files or a GPU."""

import hashlib
from types import SimpleNamespace

import pytest

from vflash.native.h3_conditioning_bundle import (
    H3_FIRST_FRAME_CONDITIONING_BUNDLE_SCHEMA_VERSION,
    H3_FIRST_FRAME_POLICY,
    H3ConditioningBundleError,
    H3ConditioningProfile,
    build_h3_in_memory_conditioning,
    consume_h3_in_memory_conditioning,
    h3_target_video_tokens,
)
from vflash.native.h3_native_conditioning_runtime import H3NativeConditioningRuntime
from vflash.native.h3_native_scheduler import H3NativeSchedule


def _live_capture(*, task="ref2va"):
    torch = pytest.importorskip("torch")
    count = 1
    nfe = 16 if task == "i2va" else 4
    profile = H3ConditioningProfile(task, 32, 32, 124, nfe, 12, 3, count, count, 0)
    video = h3_target_video_tokens(width=32, height=32, frames=124) + count
    audio, text = (414 if task == "i2va" else 1), 2
    sequence = video + audio + text
    tensors = {
        "initial_video_latents": torch.zeros(1, video, 96),
        "initial_audio_latents": torch.zeros(1, audio, 32),
        "encoder_hidden_states": torch.zeros(1, text, 5120),
        "token_tags": torch.tensor([0] * video + [2] * audio + [1] * text),
        "position_ids": torch.zeros(sequence, 3, dtype=torch.int64),
        "rotary_cos": torch.ones(sequence, 128),
        "rotary_sin": torch.zeros(sequence, 128),
        "video_indices": torch.arange(video),
        "audio_indices": torch.arange(video, video + audio),
        "text_indices": torch.arange(video + audio, sequence),
        "first_timesteps": torch.tensor([0.5, 0.999]),
        "first_timestep_indices": torch.tensor([1] + [0] * (sequence - 1)),
        "first_time_embeddings": torch.zeros(2, 5376),
        "first_packed_input": torch.arange(sequence * 5376, dtype=torch.float32)
        .reshape(1, sequence, 5376)
        .to(torch.bfloat16),
    }
    prompt = "A scene based on <Picture 1>."
    common_request = {
        "source_case_id": "in-memory-test",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": 1,
        "delivery_profiles": [
            {"temporal_profile": "native-24fps-5s", "frames": 120, "fps": 24}
        ],
    }
    reference = {
        "role": "first_frame",
        "size_bytes": 100,
        "sha256": "d" * 64,
    }
    request = (
        {
            **common_request,
            "first_frame_policy": H3_FIRST_FRAME_POLICY,
            "first_frame": reference,
        }
        if task == "i2va"
        else {
            **common_request,
            "reference_image_policy": "match",
            "references": [
                {
                    "picture_index": 1,
                    "role": "reference",
                    "size_bytes": 100,
                    "sha256": "d" * 64,
                }
            ],
        }
    )
    source = {
        "model_repository": "example/model",
        "model_revision": "test",
        "transformer_sha256": "a" * 64,
        "oracle": "diffusers",
        "oracle_revision": "test",
        "oracle_profile": f"{task}-adapter-bf16-torch-sdpa-sm89",
        "oracle_config_sha256": "b" * 64,
        "oracle_hardware": "sm89",
        "oracle_runtime_sha256": "c" * 64,
    }
    live = build_h3_in_memory_conditioning(
        bundle_id="h3-conditioning-in-memory-test",
        profile=profile,
        request=request,
        source=source,
        schedule=H3NativeSchedule.shifted_linear(nfe),
        tensors=tensors,
        schema_version=(
            H3_FIRST_FRAME_CONDITIONING_BUNDLE_SCHEMA_VERSION if task == "i2va" else 1
        ),
    )
    return live, tensors, source


def test_in_memory_conditioning_is_file_free_and_one_shot():
    live, captured, _source = _live_capture()
    assert live.tensor_bytes == sum(
        tensor.numel() * tensor.element_size() for tensor in captured.values()
    )
    selected = consume_h3_in_memory_conditioning(live)
    assert selected["first_packed_input"] is captured["first_packed_input"]
    assert "encoder_hidden_states" not in selected
    assert set(selected) == {
        "initial_video_latents",
        "initial_audio_latents",
        "token_tags",
        "rotary_cos",
        "rotary_sin",
        "video_indices",
        "audio_indices",
        "text_indices",
        "first_packed_input",
    }
    with pytest.raises(H3ConditioningBundleError, match="already consumed"):
        consume_h3_in_memory_conditioning(live)


def test_native_runtime_reads_live_tensors_without_opening_a_bundle(monkeypatch):
    torch = pytest.importorskip("torch")
    live, captured, source = _live_capture()
    runtime = H3NativeConditioningRuntime.__new__(H3NativeConditioningRuntime)
    runtime._torch = torch
    runtime.artifact = SimpleNamespace(source=source)
    runtime.overlay = SimpleNamespace(schedule=H3NativeSchedule.shifted_linear(4))
    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.load_h3_conditioning_bundle",
        lambda _path: pytest.fail("the same-process handoff opened a persisted bundle"),
    )
    bundle, loaded = runtime._load_request_tensors(live)
    assert bundle is live
    assert loaded["initial_video"] is captured["initial_video_latents"]
    expected = captured["first_packed_input"].index_select(1, captured["text_indices"])
    assert torch.equal(loaded["refined_text"], expected)


def test_in_memory_conditioning_revalidates_tensor_shape_before_consuming():
    live, captured, _source = _live_capture()
    captured["position_ids"].resize_(1, 3)
    with pytest.raises(H3ConditioningBundleError, match="packed shapes"):
        consume_h3_in_memory_conditioning(live)


def test_in_memory_keyframe_handoff_checks_modality_partitions():
    live, captured, _source = _live_capture(task="i2va")
    selected = consume_h3_in_memory_conditioning(live)
    assert selected["initial_audio_latents"].shape == (1, 414, 32)
    assert selected["token_tags"] is captured["token_tags"]
