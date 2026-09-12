"""Exercise the persisted native bundle boundary without models or a GPU."""

import hashlib
import json
from copy import deepcopy

import pytest

from vflash.native.h3_conditioning_bundle import (
    H3_FIRST_FRAME_CONDITIONING_BUNDLE_SCHEMA_VERSION,
    H3_FIRST_FRAME_POLICY,
    H3_FL2VA_CONDITIONING_BUNDLE_SCHEMA_VERSION,
    H3_FL2VA_KEYFRAME_POLICY,
    H3ConditioningBundleError,
    H3ConditioningProfile,
    _validate_request,
    h3_target_audio_tokens,
    h3_target_video_tokens,
    load_h3_conditioning_bundle,
    seal_h3_conditioning_bundle,
)
from vflash.native.h3_native_scheduler import H3NativeSchedule


class ReachedSourceValidation(RuntimeError):
    pass


def write_bundle(directory, count, *, task="ref2va", frames=5, nfe=4):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    profile = H3ConditioningProfile(task, 32, 32, frames, nfe, 12, 3, count, count, 0)
    video = h3_target_video_tokens(width=32, height=32, frames=frames) + count
    audio = h3_target_audio_tokens(frames=frames) if task in {"i2va", "fl2va"} else 1
    text = 2
    sequence = video + audio + text
    values = {
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
        "first_timestep_indices": torch.tensor([1] * count + [0] * (sequence - count)),
        "first_time_embeddings": torch.zeros(2, 5376),
        "first_packed_input": torch.zeros(1, sequence, 5376, dtype=torch.bfloat16),
    }
    save_file(values, directory / "conditioning.safetensors")
    schedule = H3NativeSchedule.shifted_linear(nfe)
    (directory / "scheduler.json").write_text(json.dumps(schedule.to_mapping()))
    prompt = "A scene with ordered reference images."
    request = {
        "source_case_id": "synthetic-image-bundle",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": 1,
        "reference_image_policy": "match",
        "delivery_profiles": [{"temporal_profile": "test", "frames": 5, "fps": 24}],
        "references": [
            {
                "picture_index": i,
                "role": "reference",
                "size_bytes": 100,
                "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            }
            for i in range(1, count + 1)
        ],
    }
    source = {
        "model_repository": "example/model",
        "model_revision": "test",
        "transformer_sha256": "a" * 64,
        "oracle": "test",
        "oracle_revision": "test",
        "oracle_profile": "test",
        "oracle_config_sha256": "b" * 64,
        "oracle_hardware": "cpu",
        "oracle_runtime_sha256": "c" * 64,
    }
    return profile, request, source


@pytest.mark.parametrize(
    ("duration_seconds", "model_frames", "delivery_frames", "audio_rows"),
    [(5, 124, 120, 414), (10, 243, 240, 810)],
)
def test_native_first_frame_bundle_is_distinct_from_ref2va(
    tmp_path, duration_seconds, model_frames, delivery_frames, audio_rows
):
    profile, _request, source = write_bundle(
        tmp_path,
        1,
        task="i2va",
        frames=model_frames,
        nfe=16,
    )
    prompt = "Motion begins from the supplied frame."
    request = {
        "source_case_id": "synthetic-first-frame-bundle",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": 97,
        "first_frame_policy": H3_FIRST_FRAME_POLICY,
        "delivery_profiles": [
            {
                "temporal_profile": f"native-24fps-{duration_seconds}s",
                "frames": delivery_frames,
                "fps": 24,
            }
        ],
        "first_frame": {
            "role": "first_frame",
            "size_bytes": 100,
            "sha256": "d" * 64,
        },
    }
    sealed = seal_h3_conditioning_bundle(
        tmp_path,
        bundle_id="h3-conditioning-first-frame",
        profile=profile,
        request=request,
        source=source,
        schema_version=H3_FIRST_FRAME_CONDITIONING_BUNDLE_SCHEMA_VERSION,
    )
    assert sealed.schema_version == 3
    assert sealed.profile.task == "i2va" and sealed.schedule.nfe == 16
    assert h3_target_audio_tokens(frames=model_frames) == audio_rows
    assert sealed.request["first_frame"]["role"] == "first_frame"
    assert "references" not in sealed.request


@pytest.mark.parametrize(
    ("duration_seconds", "model_frames", "delivery_frames"),
    [(5, 124, 120), (10, 243, 240)],
)
def test_native_fl2va_bundle_binds_both_temporal_anchors(
    tmp_path, duration_seconds, model_frames, delivery_frames
):
    profile, _request, source = write_bundle(
        tmp_path,
        2,
        task="fl2va",
        frames=model_frames,
        nfe=16,
    )
    prompt = "Move from <Picture 1> to <Picture 2>."
    request = {
        "source_case_id": "synthetic-first-last-frame-bundle",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": 97,
        "keyframe_policy": H3_FL2VA_KEYFRAME_POLICY,
        "delivery_profiles": [
            {
                "temporal_profile": f"native-24fps-{duration_seconds}s",
                "frames": delivery_frames,
                "fps": 24,
            }
        ],
        "first_frame": {
            "role": "first_frame",
            "size_bytes": 100,
            "sha256": "d" * 64,
        },
        "last_frame": {
            "role": "last_frame",
            "size_bytes": 101,
            "sha256": "e" * 64,
        },
    }
    sealed = seal_h3_conditioning_bundle(
        tmp_path,
        bundle_id="h3-conditioning-first-last-frame",
        profile=profile,
        request=request,
        source=source,
        schema_version=H3_FL2VA_CONDITIONING_BUNDLE_SCHEMA_VERSION,
    )
    assert sealed.schema_version == 4
    assert sealed.profile.task == "fl2va" and sealed.schedule.nfe == 16
    assert [sealed.request[name]["role"] for name in ("first_frame", "last_frame")] == [
        "first_frame",
        "last_frame",
    ]
    with pytest.raises(H3ConditioningBundleError, match="FL2VA"):
        _validate_request(request, task="i2va", schema_version=4)
    with pytest.raises(H3ConditioningBundleError, match="schema v1"):
        _validate_request(_request, task="fl2va")


@pytest.mark.parametrize(
    "capability,count,strategy,accepted",
    [
        ((8, 9), 1, "single", True),
        ((8, 6), 2, "sequence-head", True),
        ((8, 6), 1, "single", False),
        ((8, 6), 2, "tensor", False),
        ((8, 9), 2, "sequence-head", False),
    ],
)
@pytest.mark.parametrize("schema_version,task", [(3, "i2va"), (4, "fl2va")])
def test_native_keyframe_reader_accepts_only_released_topologies(
    monkeypatch, tmp_path, capability, count, strategy, accepted, schema_version, task
):
    from types import SimpleNamespace

    from vflash.native.h3_native_conditioning_runtime import (
        H3NativeConditioningRuntime,
        H3NativeConditioningRuntimeError,
    )

    runtime = H3NativeConditioningRuntime.__new__(H3NativeConditioningRuntime)
    runtime._torch = object()
    runtime.artifact = SimpleNamespace(
        source={"oracle_profile": f"{task}-base-bf16-torch-sdpa-sm86"}
    )
    runtime.devices, runtime.compute_capability = [None] * count, capability
    runtime.parallel_strategy = strategy
    runtime.overlay = SimpleNamespace(schedule=SimpleNamespace(to_mapping=lambda: {}))
    bundle = SimpleNamespace(
        profile=SimpleNamespace(task=task),
        schema_version=schema_version,
        schedule=SimpleNamespace(to_mapping=lambda: {}),
        source={},
    )
    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.load_h3_conditioning_bundle",
        lambda _: bundle,
    )
    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.validate_conditioning_source",
        lambda *_: (_ for _ in ()).throw(ReachedSourceValidation()),
    )
    if accepted:
        with pytest.raises(ReachedSourceValidation):
            runtime._load_request_tensors(tmp_path)
    else:
        with pytest.raises(H3NativeConditioningRuntimeError, match="qualified"):
            runtime._load_request_tensors(tmp_path)


@pytest.mark.parametrize("count", [1, 3, 4, 9])
def test_native_image_bundle_persists_all_ordered_references(tmp_path, count):
    profile, request, source = write_bundle(tmp_path, count)
    sealed = seal_h3_conditioning_bundle(
        tmp_path,
        bundle_id="h3-conditioning-images",
        profile=profile,
        request=request,
        source=source,
    )
    loaded = load_h3_conditioning_bundle(tmp_path)
    assert loaded.request == request
    assert loaded.profile == profile
    assert loaded.conditioning_sha256 == sealed.conditioning_sha256
    assert loaded.schedule.nfe == 4
    assert len(next(row.tensors for row in loaded.files if row.role == "conditioning")) == 14


@pytest.mark.parametrize("damage", ["ten", "empty", "order", "duplicate"])
def test_native_reader_rejects_invalid_reference_identity_after_roundtrip(tmp_path, damage):
    profile, request, source = write_bundle(tmp_path, 4)
    seal_h3_conditioning_bundle(
        tmp_path,
        bundle_id="h3-conditioning-images",
        profile=profile,
        request=request,
        source=source,
    )
    manifest = tmp_path / "bundle.json"
    value = json.loads(manifest.read_text())
    refs = value["request"]["references"]
    if damage == "ten":
        refs += [
            {
                **refs[0],
                "picture_index": i,
                "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            }
            for i in range(5, 11)
        ]
    elif damage == "empty":
        refs.clear()
    elif damage == "order":
        refs[0], refs[1] = refs[1], refs[0]
    else:
        refs[1] = {**deepcopy(refs[0]), "picture_index": 2}
    manifest.write_text(json.dumps(value))
    with pytest.raises(H3ConditioningBundleError, match=r"reference|Ref2VA"):
        load_h3_conditioning_bundle(tmp_path)
