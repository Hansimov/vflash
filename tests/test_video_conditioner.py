from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vflash.adapters.conditioning_capture import H3ConditioningCaptureComplete
from vflash.adapters.diffusers_h3 import DiffusersConditioner
from vflash.adapters.video_geometry import reference_video_geometry
from vflash.adapters.video_references import DecodedVideoReference
from vflash.contracts import ContractError
from vflash.model_assets import model_profile
from vflash.native.h3_conditioning_bundle import (
    H3_FL2VA_CONDITIONING_BUNDLE_SCHEMA_VERSION,
    H3_FL2VA_KEYFRAME_POLICY,
    H3_VIDEO_REFERENCE_POLICY,
    _validate_fl2va_profile,
    _validate_request,
    _validate_source,
    _validate_video_profile,
)
from vflash.pipeline.assets import conditioning_source
from vflash.pipeline.contracts import VideoRequest


def decoded_video():
    np = pytest.importorskip("numpy")
    return DecodedVideoReference(
        np.zeros((48, 128, 232, 3), dtype=np.uint8),
        {
            "kind": "video",
            "index": 1,
            "role": "reference",
            "size_bytes": 128,
            "sha256": "a" * 64,
            "source_width": 232,
            "source_height": 128,
            "duration_seconds": 2,
            "fps": 24,
            "frames": 48,
            **reference_video_geometry(232, 128, 48),
            "audio_conditioning": False,
            "decoded_rgb_sha256": "b" * 64,
        },
    )


@pytest.mark.parametrize("bad_prefix,fail", [(False, False), (True, False), (False, True)])
def test_capture_emits_schema2_with_complete_identity_and_always_discards(
    tmp_path, monkeypatch, bad_prefix, fail
):
    decoded = decoded_video()
    rows = decoded.metadata["condition_video_rows"]
    events = []
    module = ModuleType("diffusers.modular_pipelines.minimax_h3")
    module.MiniMaxH3ImageReference = lambda **kwargs: pytest.fail("video relabeled image")
    module.MiniMaxH3VideoReference = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    owner = DiffusersConditioner.__new__(DiffusersConditioner)
    owner.profile = model_profile("ref2va-turbo4-exact-sm89")
    owner.prepared = SimpleNamespace(profile_id=owner.profile.definition.id)
    owner._closed, owner._cuda_active = False, True
    owner.versions = {"diffusers": "0.40.0", "torch": "2.11.0"}
    owner.device, owner.transformer = "unused-cpu-fixture", object()
    owner._torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda _: events.append("synchronize")),
        Generator=lambda: SimpleNamespace(manual_seed=lambda seed: seed),
    )

    class Pipe:
        scheduler = SimpleNamespace(sigmas=[1, 0.5, 0.2, 0.1, 0])
        audio_scheduler = SimpleNamespace(sigmas=[1, 0.7, 0.3, 0.1, 0])

        def __call__(self, **kwargs):
            (reference,) = kwargs["references"]
            assert reference["frames"] is decoded.frames
            assert reference["fps"] == 24 and reference["audio"] is None
            assert kwargs["prompt"] == "<Video 1> changes its action."
            assert kwargs["num_frames"] == 124 and kwargs["num_inference_steps"] == 5
            if fail:
                raise ValueError("official setup failed")
            raise H3ConditioningCaptureComplete

    owner.pipe = Pipe()

    class Capture:
        def __init__(self, _directory):
            pass

        def install(self, transformer):
            assert transformer is owner.transformer
            events.append("install")

        def close(self):
            events.append("capture-close")

        def discard(self):
            events.append("discard")

        def prefix_counts(self):
            return rows - int(bad_prefix), 0

        def reference_token_budget(self, **kwargs):
            return rows

        def finish(self, **kwargs):
            assert kwargs["schema_version"] == 2
            request = kwargs["request"]
            assert request["references"] == [decoded.metadata]
            assert _validate_request(request, task="ref2va", schema_version=2) == request
            _validate_video_profile(kwargs["profile"], request)
            assert _validate_source(kwargs["source"]) == conditioning_source(
                profile_id=owner.prepared.profile_id,
                runtime_versions=owner.versions,
                reference_policy=H3_VIDEO_REFERENCE_POLICY,
            )
            assert (
                kwargs["source"]["oracle_config_sha256"]
                != conditioning_source(
                    profile_id=owner.prepared.profile_id,
                    runtime_versions=owner.versions,
                )["oracle_config_sha256"]
            )
            assert kwargs["video_sigmas"] is owner.pipe.scheduler.sigmas
            assert kwargs["audio_sigmas"] is owner.pipe.audio_scheduler.sigmas
            assert kwargs["update_rule"] == "training_euler"
            return kwargs

    monkeypatch.setattr("vflash.adapters.diffusers_h3.H3ConditioningCaptureSession", Capture)
    request = VideoRequest(
        "<Video 1> changes its action.", reference_video=Path("reference.mp4")
    )
    if bad_prefix or fail:
        with pytest.raises((ValueError, ContractError), match=r"prefix|setup failed"):
            owner.capture(request, (decoded,), tmp_path / "conditioning")
    else:
        owner.capture(request, (decoded,), tmp_path / "conditioning")
        assert events[:3] == ["install", "synchronize", "capture-close"]
    assert events[-1] == "discard"
    assert decoded.frames is not None  # caller, not capture, owns the RGB lifetime
    decoded.close()


@pytest.mark.parametrize(
    "resident_profile",
    ["i2va-base16-bf16-sm89", "fl2va-base16-bf16-sm89"],
)
def test_capture_passes_both_keyframes_and_emits_fl2va_schema4(
    tmp_path, monkeypatch, resident_profile
):
    events = []
    first = SimpleNamespace(image="first-rgb", size_bytes=100, sha256="a" * 64)
    last = SimpleNamespace(image="last-rgb", size_bytes=101, sha256="b" * 64)
    module = ModuleType("diffusers.modular_pipelines.minimax_h3")
    module.MiniMaxH3ImageReference = lambda **kwargs: kwargs
    module.MiniMaxH3VideoReference = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    owner = DiffusersConditioner.__new__(DiffusersConditioner)
    owner.profile = model_profile(resident_profile)
    owner.prepared = SimpleNamespace(profile_id=owner.profile.definition.id)
    owner._closed, owner._cuda_active = False, True
    owner.versions = {"diffusers": "0.40.0", "torch": "2.11.0"}
    owner.device, owner.transformer = "unused-cpu-fixture", object()
    owner._torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda _: events.append("synchronize")),
        Generator=lambda: SimpleNamespace(manual_seed=lambda seed: seed),
    )

    class Pipe:
        scheduler = SimpleNamespace(sigmas=list(range(17)))
        audio_scheduler = SimpleNamespace(sigmas=list(range(17)))

        def __call__(self, **kwargs):
            assert kwargs["image"] == first.image
            assert kwargs["last_image"] == last.image
            assert "references" not in kwargs
            assert kwargs["num_inference_steps"] == 17
            raise H3ConditioningCaptureComplete

    owner.pipe = Pipe()

    class Capture:
        def __init__(self, _directory):
            pass

        def install(self, transformer):
            assert transformer is owner.transformer

        def close(self):
            pass

        def discard(self):
            events.append("discard")

        def prefix_counts(self):
            return 2, 0

        def reference_token_budget(self, **kwargs):
            return 2

        def finish(self, **kwargs):
            assert kwargs["schema_version"] == H3_FL2VA_CONDITIONING_BUNDLE_SCHEMA_VERSION
            request = kwargs["request"]
            assert request["keyframe_policy"] == H3_FL2VA_KEYFRAME_POLICY
            assert [request[name]["sha256"] for name in ("first_frame", "last_frame")] == [
                first.sha256,
                last.sha256,
            ]
            assert _validate_request(request, task="fl2va", schema_version=4) == request
            _validate_fl2va_profile(kwargs["profile"], request)
            assert kwargs["source"] == conditioning_source(
                profile_id=resident_profile,
                request_mode="fl2va",
                runtime_versions=owner.versions,
            )
            return kwargs

    monkeypatch.setattr("vflash.adapters.diffusers_h3.H3ConditioningCaptureSession", Capture)
    request = VideoRequest(
        "Move from <Picture 1> to <Picture 2>.",
        first_frame=Path("first.png"),
        last_frame=Path("last.png"),
        width=32,
        height=32,
        seed=7,
    )
    result = owner.capture(request, (first, last), tmp_path / "conditioning")
    assert result["profile"].task == "fl2va"
    assert result["profile"].num_condition_video_rows == 2
    assert events == ["synchronize", "discard"]
