"""Few-step keyframes retain endpoint structure and bind their own adapter/schedule."""

from types import SimpleNamespace

import pytest

from vflash.model_assets import (
    conditioning_profile_for_request,
    model_profile,
    model_schedule,
    supported_request_modes,
    transformer_identity,
)
from vflash.native.h3_conditioning_bundle import (
    H3ConditioningBundleError,
    _complete_delivery_profile,
    _validate_first_frame_profile,
    _validate_fl2va_profile,
    _validate_last_frame_profile,
)
from vflash.native.h3_distilled_lora import (
    LIGHTX_H3_REF_TURBO8_CONTRACT,
    LIGHTX_H3_TURBO4_V01_CONTRACT,
    LIGHTX_H3_TURBO8_544_CONTRACT,
    LIGHTX_H3_TURBO8_CONTRACT,
)
from vflash.native.h3_native_conditioning_runtime import (
    H3NativeConditioningRuntimeError,
    _conditioning_task_matches,
    validate_conditioning_source,
)
from vflash.native.runner import WEIGHT_PROFILES


@pytest.mark.parametrize("nfe", [4, 8])
@pytest.mark.parametrize(
    "mode,validate,endpoints",
    [
        ("i2va", _validate_first_frame_profile, 1),
        ("fl2va", _validate_fl2va_profile, 2),
    ],
)
def test_keyframe_adapter_execution_contract(nfe, mode, validate, endpoints):
    profile_id = f"{mode}-turbo{nfe}-exact-sm89"
    model = model_profile(profile_id)
    assert model.workflow == "fl2va" and model.transformer_component == "transformer"
    assert model.adapter.alpha == (128 if nfe == 4 else 8)
    assert model.adapter.rank == 128
    assert model.adapter_execution == "runtime-residual"
    assert model.weight_profile == WEIGHT_PROFILES[profile_id]
    p = SimpleNamespace(
        task=mode,
        nfe=nfe,
        width=928,
        height=512,
        frames=124,
        video_flow_shift=6,
        audio_flow_shift=3,
        num_condition_audio_rows=0,
        num_condition_video_rows=endpoints * 29 * 16,
    )
    request = {"delivery_profiles": _complete_delivery_profile(124)}
    validate(p, request)
    # First/last mode cannot silently discard the authoritative last frame.
    p.num_condition_video_rows -= 29 * 16
    with pytest.raises(H3ConditioningBundleError):
        validate(p, request)
    p.num_condition_video_rows += 29 * 16
    p.video_flow_shift = 7
    with pytest.raises(H3ConditioningBundleError):
        validate(p, request)


@pytest.mark.parametrize("nfe", [4, 8])
@pytest.mark.parametrize("mode", ["i2va", "fl2va"])
def test_keyframe_runtime_schedule(nfe, mode):
    pytest.importorskip("torch")
    assert model_schedule(f"{mode}-turbo{nfe}-exact-sm89").nfe == nfe


def test_ref_and_keyframe_eight_step_weights_are_not_interchangeable():
    assert LIGHTX_H3_TURBO8_CONTRACT.revision != LIGHTX_H3_REF_TURBO8_CONTRACT.revision
    assert LIGHTX_H3_TURBO8_CONTRACT.sha256 != LIGHTX_H3_REF_TURBO8_CONTRACT.sha256


@pytest.mark.parametrize("mode", ["i2va", "fl2va"])
def test_original_v01_contract_is_not_the_768p_four_step_contract(mode):
    name = f"{mode}-turbo4-v01-544-exact-sm89"
    model = model_profile(name)
    assert model.adapter == LIGHTX_H3_TURBO4_V01_CONTRACT
    assert model.definition.nfe == 4
    assert (model.definition.video_flow_shift, model.definition.audio_flow_shift) == (12, 3)
    assert model.adapter.scaling == 0.0625 and model.adapter.strength == 1
    assert model.weight_profile == WEIGHT_PROFILES[name]
    assert supported_request_modes(name) == ("i2va", "l2va", "fl2va")
    assert conditioning_profile_for_request(name, "l2va") == "fl2va-turbo4-v01-544-exact-sm89"
    assert transformer_identity(name) != transformer_identity(f"{mode}-turbo4-exact-sm89")


@pytest.mark.parametrize(
    "mode,validate,endpoints",
    [
        ("i2va", _validate_first_frame_profile, 1),
        ("l2va", _validate_last_frame_profile, 1),
        ("fl2va", _validate_fl2va_profile, 2),
    ],
)
def test_original_v01_keeps_explicit_endpoint_conditioning(mode, validate, endpoints):
    profile = SimpleNamespace(
        task=mode,
        nfe=4,
        width=960,
        height=544,
        frames=124,
        video_flow_shift=12,
        audio_flow_shift=3,
        num_condition_audio_rows=0,
        num_condition_video_rows=endpoints * 30 * 17,
    )
    validate(profile, {"delivery_profiles": _complete_delivery_profile(124)})
    profile.num_condition_video_rows -= 30 * 17
    with pytest.raises(H3ConditioningBundleError):
        validate(profile, {"delivery_profiles": _complete_delivery_profile(124)})


@pytest.mark.parametrize("mode", ["i2va", "fl2va"])
def test_544_weights_and_schedule_are_distinct_from_768(mode):
    profile_id = f"{mode}-turbo8-544-exact-sm89"
    model = model_profile(profile_id)
    assert model.adapter == LIGHTX_H3_TURBO8_544_CONTRACT
    assert model.adapter != LIGHTX_H3_TURBO8_CONTRACT
    assert model.weight_profile == WEIGHT_PROFILES[profile_id]
    assert (model.definition.video_flow_shift, model.definition.audio_flow_shift) == (12, 3)
    assert supported_request_modes(profile_id) == ("i2va", "l2va", "fl2va")
    assert conditioning_profile_for_request(profile_id, "l2va") == "fl2va-turbo8-544-exact-sm89"


@pytest.mark.parametrize("mode", ["i2va", "fl2va"])
def test_544_runtime_schedule_is_distinct_from_768(mode):
    pytest.importorskip("torch")
    assert model_schedule(f"{mode}-turbo8-544-exact-sm89") != model_schedule(
        f"{mode}-turbo8-exact-sm89"
    )


@pytest.mark.parametrize(
    "mode,validate,endpoints",
    [
        ("i2va", _validate_first_frame_profile, 1),
        ("l2va", _validate_last_frame_profile, 1),
        ("fl2va", _validate_fl2va_profile, 2),
    ],
)
def test_544_conditioning_preserves_endpoint_count(mode, validate, endpoints):
    profile = SimpleNamespace(
        task=mode,
        nfe=8,
        width=512,
        height=512,
        frames=124,
        video_flow_shift=12,
        audio_flow_shift=3,
        num_condition_audio_rows=0,
        num_condition_video_rows=endpoints * 256,
    )
    request = {"delivery_profiles": _complete_delivery_profile(124)}
    validate(profile, request)
    profile.num_condition_video_rows -= 256
    with pytest.raises(H3ConditioningBundleError):
        validate(profile, request)


def test_544_paired_conditioning_cannot_substitute_768_weights():
    source = transformer_identity("i2va-turbo8-544-exact-sm89")
    paired = transformer_identity("fl2va-turbo8-544-exact-sm89")
    validate_conditioning_source(paired, source)
    with pytest.raises(H3NativeConditioningRuntimeError):
        validate_conditioning_source(transformer_identity("fl2va-turbo8-exact-sm89"), source)
    artifact = SimpleNamespace(
        source=source,
        weight_profile="lightx-turbo8-v1.0-544",
        adapter_execution="runtime-residual",
    )
    assert _conditioning_task_matches("l2va", artifact)
    assert _conditioning_task_matches("fl2va", artifact)
    artifact.weight_profile = "lightx-turbo8-v1.0"
    assert not _conditioning_task_matches("l2va", artifact)
