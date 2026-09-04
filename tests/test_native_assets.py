import json
import struct
from types import SimpleNamespace

import pytest

from vflash.native.h3_conditioning_bundle import (
    H3ConditioningBundleError,
    H3ConditioningProfile,
)
from vflash.native.h3_native_conditioning_runtime import (
    H3NativeConditioningRuntime,
    H3NativeConditioningRuntimeError,
    validate_declared_schedule,
)
from vflash.native.h3_native_scheduler import H3NativeSchedule
from vflash.native.h3_tensor_file import H3TensorFileError, inspect_safetensors_header


def conditioning_profile():
    return {
        "task": "ref2va",
        "width": 928,
        "height": 512,
        "frames": 124,
        "nfe": 4,
        "video_flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "reference_token_budget": 198,
        "num_condition_video_rows": 198,
        "num_condition_audio_rows": 0,
    }


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"task": "t2va"}, "Ref2VA"),
        ({"width": 930}, "divisible by 32"),
        ({"nfe": True}, "integer fields"),
        ({"num_condition_video_rows": 0}, "reference budget"),
        ({"num_condition_audio_rows": True}, "prefix counts"),
    ],
)
def test_conditioning_rejects_incompatible_geometry_and_prefixes(change, reason):
    valid = conditioning_profile()
    assert H3ConditioningProfile.from_mapping(valid).num_condition_video_rows == 198
    with pytest.raises(H3ConditioningBundleError, match=reason):
        H3ConditioningProfile.from_mapping({**valid, **change})


@pytest.mark.parametrize(
    "second,reason",
    [
        ({"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]}, "overlap"),
        ({"dtype": "BF16", "shape": [1], "data_offsets": [4, 8]}, "outside"),
        ({"dtype": "BF16", "shape": [True], "data_offsets": [4, 4]}, "shape"),
    ],
)
def test_tensor_reader_rejects_unsafe_header_ranges(tmp_path, second, reason):
    header = json.dumps(
        {
            "first": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
            "second": second,
        }
    ).encode()
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
    with pytest.raises(H3TensorFileError, match=reason):
        inspect_safetensors_header(path)


def test_tensor_reader_rejects_symlinked_payload(tmp_path):
    path = tmp_path / "payload.safetensors"
    path.write_bytes(bytes(8))
    link = tmp_path / "alias.safetensors"
    link.symlink_to(path)
    with pytest.raises(H3TensorFileError, match="regular file"):
        inspect_safetensors_header(link)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"expected_nfe": 8}, "NFE"),
        ({"expected_scheduler": "unsupported"}, "scheduler"),
        ({"expected_video_flow_shift": 6.0}, "sigma grids"),
        ({"expected_audio_flow_shift": 6.0}, "sigma grids"),
    ],
)
def test_declared_profile_cannot_mislabel_the_runtime_schedule(change, reason):
    pytest.importorskip("torch")
    schedule = H3NativeSchedule.shifted_linear(4, video_shift=12.0, audio_shift=3.0)
    declared = {
        "expected_nfe": 4,
        "expected_scheduler": "h3-training-euler",
        "expected_video_flow_shift": 12.0,
        "expected_audio_flow_shift": 3.0,
    }
    validate_declared_schedule(schedule, **declared)
    with pytest.raises(H3NativeConditioningRuntimeError, match=reason):
        validate_declared_schedule(schedule, **{**declared, **change})


def test_conditioning_capture_schedule_can_differ_from_execution(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    runtime = H3NativeConditioningRuntime.__new__(H3NativeConditioningRuntime)
    runtime._torch = torch
    runtime.overlay = SimpleNamespace(schedule=H3NativeSchedule.shifted_linear(4))
    runtime.artifact = SimpleNamespace(source={})
    capture = H3ConditioningProfile.from_mapping(
        {**conditioning_profile(), "nfe": 8, "video_flow_shift": 6.0}
    )
    bundle = SimpleNamespace(directory=tmp_path, source={}, profile=capture)
    packed = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2).to(torch.bfloat16)
    tensors = {
        "video_indices": torch.tensor([0]),
        "audio_indices": torch.tensor([1]),
        "text_indices": torch.tensor([2]),
        "first_packed_input": packed,
        "initial_video_latents": torch.ones(1, 1, 4),
        "initial_audio_latents": torch.ones(1, 1, 2),
        "token_tags": torch.tensor([0, 1, 2]),
        "rotary_cos": torch.ones(3, 2),
        "rotary_sin": torch.zeros(3, 2),
    }
    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.load_h3_conditioning_bundle",
        lambda _: bundle,
    )
    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.load_safetensor_tensor",
        lambda _path, name: tensors[name],
    )
    loaded_bundle, loaded = runtime._load_request_tensors(tmp_path)
    assert loaded_bundle.profile.nfe == 8
    assert runtime.overlay.schedule.nfe == 4
    assert torch.equal(loaded["refined_text"], packed[:, 2:3])
    assert loaded["initial_video"] is tensors["initial_video_latents"]
