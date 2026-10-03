"""Few-step keyframes retain endpoint structure and bind their own adapter/schedule."""

from types import SimpleNamespace

import pytest

from vflash.model_assets import model_profile, model_schedule
from vflash.native.h3_conditioning_bundle import (
    H3ConditioningBundleError,
    _complete_delivery_profile,
    _validate_first_frame_profile,
    _validate_fl2va_profile,
)
from vflash.native.h3_distilled_lora import (
    LIGHTX_H3_REF_TURBO8_CONTRACT,
    LIGHTX_H3_TURBO8_CONTRACT,
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
    p.video_flow_shift = 12
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
