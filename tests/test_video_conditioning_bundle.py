import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vflash.native.h3_conditioning_bundle import (
    H3_VIDEO_REFERENCE_POLICY,
    H3ConditioningBundleError,
    H3ConditioningProfile,
    _validate_request,
    _validate_video_partitions,
    _validate_video_profile,
)


def video_request(seconds=5):
    frames = seconds * 24
    chunks = (frames - 5) // 17
    latents = chunks * 5 + 2
    return {
        "source_case_id": "video-reference-example",
        "prompt": "A new scene based on <Video 1>.",
        "prompt_sha256": hashlib.sha256(b"A new scene based on <Video 1>.").hexdigest(),
        "seed": 1,
        "reference_video_policy": H3_VIDEO_REFERENCE_POLICY,
        "delivery_profiles": [
            {"temporal_profile": "native-24fps-5s", "frames": 120, "fps": 24}
        ],
        "references": [
            {
                "kind": "video",
                "index": 1,
                "role": "reference",
                "size_bytes": 100,
                "sha256": "a" * 64,
                "source_width": 928,
                "source_height": 512,
                "duration_seconds": seconds,
                "fps": 24,
                "frames": frames,
                "width": 1376,
                "height": 768,
                "vae_input_frames": chunks * 17 + 5,
                "vae_latent_frames": latents,
                "condition_video_rows": latents * 43 * 24,
                "audio_conditioning": False,
                "decoded_rgb_sha256": "b" * 64,
            }
        ],
    }


@pytest.mark.parametrize("seconds", [2, 3, 4, 5])
def test_typed_video_metadata_preserves_its_real_temporal_prefix(seconds):
    request = video_request(seconds)
    validated = _validate_request(request, task="ref2va", schema_version=2)
    assert validated == request
    count = request["references"][0]["condition_video_rows"]
    profile = H3ConditioningProfile("ref2va", 928, 512, 124, 4, 12, 3, count, count, 0)
    _validate_video_profile(profile, validated)


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "image"},
        {"index": True},
        {"index": 2},
        {"audio_conditioning": True},
        {"fps": 12},
        {"frames": 121},
        {"duration_seconds": 8},
        {"width": 1344},
        {"vae_input_frames": 39},
        {"vae_latent_frames": 12},
        {"condition_video_rows": 12384},
        {"source_width": 0},
        {"decoded_rgb_sha256": "invalid"},
        {"unknown": True},
    ],
)
def test_video_metadata_rejects_changed_modality_geometry_or_source_policy(change):
    request = video_request()
    request["references"][0].update(change)
    with pytest.raises(H3ConditioningBundleError):
        _validate_request(request, task="ref2va", schema_version=2)


def test_new_video_schema_never_relabels_an_image_or_text_request():
    request = video_request()
    with pytest.raises(H3ConditioningBundleError):
        _validate_request(request, task="t2va", schema_version=2)
    with pytest.raises(H3ConditioningBundleError):
        _validate_request(request, task="ref2va")
    request["references"].append(deepcopy(request["references"][0]))
    with pytest.raises(H3ConditioningBundleError):
        _validate_request(request, task="ref2va", schema_version=2)


def test_image_and_text_metadata_retain_the_released_schema():
    request = video_request()
    request.pop("reference_video_policy")
    request["reference_image_policy"] = "match"
    request["references"] = [
        {"picture_index": 1, "role": "reference", "size_bytes": 100, "sha256": "a" * 64}
    ]
    assert _validate_request(request, task="ref2va") == request
    request["references"] = []
    assert _validate_request(request, task="t2va") == request


def test_video_prefix_uses_real_timestep_partition_and_visual_presentation(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    from vflash.native.h3_native_scheduler import H3_KEYFRAME_NOISE_AUG

    header = {
        "first_packed_input": {"shape": [1, 6, 5376], "dtype": "BF16"},
        "encoder_hidden_states": {"shape": [1, 2, 5120]},
        "initial_video_latents": {"shape": [1, 3, 96]},
        "initial_audio_latents": {"shape": [1, 414, 32]},
        **{
            name: {"dtype": "I64"}
            for name in (
                "video_indices",
                "audio_indices",
                "text_indices",
                "token_tags",
                "first_timestep_indices",
            )
        },
    }
    monkeypatch.setattr(
        "vflash.native.h3_conditioning_bundle.inspect_safetensors_header", lambda _: header
    )
    values = {
        "video_indices": torch.tensor([0, 1, 2]),
        "audio_indices": torch.tensor([3]),
        "text_indices": torch.tensor([4, 5]),
        "token_tags": torch.tensor([0, 0, 0, 2, 1, 0]),
        "first_timesteps": torch.tensor([0.5, H3_KEYFRAME_NOISE_AUG]),
        "first_timestep_indices": torch.tensor([1, 1, 0, 0, 0, 0]),
    }
    monkeypatch.setattr(
        "vflash.native.h3_tensor_file.load_safetensor_tensors", lambda *_: values
    )
    profile = SimpleNamespace(frames=124, num_condition_video_rows=2)
    _validate_video_partitions(tmp_path / "unused", profile)
    values["first_timestep_indices"][1] = 0
    with pytest.raises(H3ConditioningBundleError, match="prefix"):
        _validate_video_partitions(tmp_path / "unused", profile)
    values["video_indices"][1] = 0
    with pytest.raises(H3ConditioningBundleError, match="partition"):
        _validate_video_partitions(tmp_path / "unused", profile)


@pytest.mark.parametrize(
    "capability,count,schedule",
    [((8, 6), 1, {}), ((8, 9), 2, {}), ((8, 9), 1, {"wrong": True})],
)
def test_video_native_reader_rejects_unqualified_device_or_schedule_before_loading(
    monkeypatch, tmp_path, capability, count, schedule
):
    from vflash.native.h3_native_conditioning_runtime import (
        H3NativeConditioningRuntime,
        H3NativeConditioningRuntimeError,
    )

    runtime = H3NativeConditioningRuntime.__new__(H3NativeConditioningRuntime)
    runtime._torch = object()
    runtime.artifact = SimpleNamespace(
        source={"oracle_profile": "ref2va-adapter-bf16-torch-sdpa-sm89"}
    )
    runtime.devices, runtime.compute_capability = [None] * count, capability
    runtime.overlay = SimpleNamespace(schedule=SimpleNamespace(to_mapping=lambda: schedule))
    bundle = SimpleNamespace(
        profile=SimpleNamespace(task="ref2va"),
        schema_version=2,
        schedule=SimpleNamespace(to_mapping=lambda: {}),
    )
    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.load_h3_conditioning_bundle",
        lambda _: bundle,
    )
    with pytest.raises(H3NativeConditioningRuntimeError, match="one SM89"):
        runtime._load_request_tensors(tmp_path)
