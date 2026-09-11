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
    _conditioning_task_matches,
    validate_conditioning_source,
    validate_declared_schedule,
)
from vflash.native.h3_native_scheduler import H3NativeSchedule
from vflash.native.h3_tensor_file import H3TensorFileError, inspect_safetensors_header


def test_conditioning_hardware_is_provenance_but_model_identity_is_bound():
    artifact = {
        "model_repository": "example/model",
        "model_revision": "revision",
        "transformer_sha256": "a" * 64,
        "oracle": "diffusers",
        "oracle_revision": "encoder",
        "oracle_profile": "ref2va-adapter-bf16-torch-sdpa-sm89",
        "oracle_hardware": "sm89",
    }
    capture = {
        **artifact,
        "oracle_profile": "ref2va-adapter-bf16-torch-sdpa-sm86",
        "oracle_hardware": "sm86",
    }
    validate_conditioning_source(capture, artifact)
    for field, value in (
        ("transformer_sha256", "b" * 64),
        ("oracle_revision", "different-encoder"),
        ("oracle_profile", "ref2va-int8-torch-sdpa-sm86"),
    ):
        with pytest.raises(H3NativeConditioningRuntimeError, match="same H3 model"):
            validate_conditioning_source({**capture, field: value}, artifact)


def test_official_base16_i2va_and_fl2va_conditioning_share_one_weight_runtime():
    common = {
        "model_repository": "MiniMaxAI/MiniMax-H3",
        "model_revision": "revision",
        "transformer_sha256": "a" * 64,
        "oracle": "diffusers",
        "oracle_revision": "encoder",
    }
    artifact_source = {
        **common,
        "oracle_profile": "i2va-base-bf16-torch-sdpa-sm89",
    }
    capture_source = {
        **common,
        "oracle_profile": "fl2va-base-bf16-torch-sdpa-sm89",
    }
    validate_conditioning_source(capture_source, artifact_source)
    artifact = SimpleNamespace(
        source=artifact_source,
        weight_profile="minimax-h3-base",
        adapter_execution="none",
    )
    assert _conditioning_task_matches("i2va", artifact)
    assert _conditioning_task_matches("fl2va", artifact)
    assert not _conditioning_task_matches("t2va", artifact)
    assert not _conditioning_task_matches("ref2va", artifact)
    artifact.adapter_execution = "runtime-residual"
    assert not _conditioning_task_matches("fl2va", artifact)


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


def test_t2va_requires_an_empty_reference_prefix():
    profile = {
        **conditioning_profile(),
        "task": "t2va",
        "video_flow_shift": 6.0,
        "reference_token_budget": 0,
        "num_condition_video_rows": 0,
    }
    assert H3ConditioningProfile.from_mapping(profile).task == "t2va"
    with pytest.raises(H3ConditioningBundleError, match="reference prefixes"):
        H3ConditioningProfile.from_mapping({**profile, "num_condition_audio_rows": 1})
    with pytest.raises(H3ConditioningBundleError, match="condition-video"):
        H3ConditioningProfile.from_mapping({**profile, "task": "ref2va"})


def test_i2va_requires_one_video_prefix_and_no_audio_prefix():
    profile = {
        **conditioning_profile(),
        "task": "i2va",
        "nfe": 16,
        "reference_token_budget": 464,
        "num_condition_video_rows": 464,
    }
    assert H3ConditioningProfile.from_mapping(profile).task == "i2va"
    with pytest.raises(H3ConditioningBundleError, match="does not accept audio"):
        H3ConditioningProfile.from_mapping({**profile, "num_condition_audio_rows": 1})


def test_t2va_lora_identity_has_its_own_scale():
    from vflash.native.h3_distilled_lora import (
        H3DistilledLoraError,
        h3_distilled_lora_contract_for_profile,
    )

    contract = h3_distilled_lora_contract_for_profile("lightx-turbo4-v1.0", workflow="t2va")
    assert contract.scaling == 1.0
    assert contract.nfe == 4
    with pytest.raises(H3DistilledLoraError):
        h3_distilled_lora_contract_for_profile("lightx-turbo4-v1.0", workflow="ref2va")


def test_loaded_ref_model_rejects_t2va_before_loading_request_tensors(monkeypatch, tmp_path):
    runtime = H3NativeConditioningRuntime.__new__(H3NativeConditioningRuntime)
    runtime._torch = object()
    runtime.artifact = SimpleNamespace(
        source={"oracle_profile": "ref2va-adapter-bf16-torch-sdpa-sm89"}
    )
    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.load_h3_conditioning_bundle",
        lambda _: SimpleNamespace(profile=SimpleNamespace(task="t2va")),
    )
    with pytest.raises(H3NativeConditioningRuntimeError, match="Base or Ref model"):
        runtime._load_request_tensors(tmp_path)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"task": "t2va"}, "reference prefixes"),
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
    runtime.artifact = SimpleNamespace(
        source={"oracle_profile": "ref2va-adapter-bf16-torch-sdpa-sm89"}
    )
    capture = H3ConditioningProfile.from_mapping(
        {**conditioning_profile(), "nfe": 8, "video_flow_shift": 6.0}
    )
    bundle = SimpleNamespace(
        directory=tmp_path, source=runtime.artifact.source, profile=capture, schema_version=1
    )
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
        "vflash.native.h3_native_conditioning_runtime.load_safetensor_tensors",
        lambda _path, names: {name: tensors[name] for name in names},
    )
    loaded_bundle, loaded = runtime._load_request_tensors(tmp_path)
    assert loaded_bundle.profile.nfe == 8
    assert runtime.overlay.schedule.nfe == 4
    assert torch.equal(loaded["refined_text"], packed[:, 2:3])
    assert loaded["initial_video"] is tensors["initial_video_latents"]
