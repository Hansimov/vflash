"""Exercise the persisted native bundle boundary without models or a GPU."""

import hashlib
import json
from copy import deepcopy

import pytest

from vflash.native.h3_conditioning_bundle import (
    H3_FIRST_FRAME_CONDITIONING_BUNDLE_SCHEMA_VERSION,
    H3_FIRST_FRAME_POLICY,
    H3ConditioningBundleError,
    H3ConditioningProfile,
    h3_target_video_tokens,
    load_h3_conditioning_bundle,
    seal_h3_conditioning_bundle,
)
from vflash.native.h3_native_scheduler import H3NativeSchedule


def write_bundle(directory, count, *, task="ref2va", frames=5, nfe=4):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    profile = H3ConditioningProfile(task, 32, 32, frames, nfe, 12, 3, count, count, 0)
    video = h3_target_video_tokens(width=32, height=32, frames=frames) + count
    audio, text = (414 if task == "i2va" else 1), 2
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


def test_native_first_frame_bundle_is_distinct_from_ref2va(tmp_path):
    profile, _request, source = write_bundle(
        tmp_path,
        1,
        task="i2va",
        frames=124,
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
            {"temporal_profile": "native-24fps-5s", "frames": 120, "fps": 24}
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
    assert sealed.request["first_frame"]["role"] == "first_frame"
    assert "references" not in sealed.request


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
