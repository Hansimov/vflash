"""Mode, adapter, scheduler and hardware cannot be mixed by changing a flag."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vflash.compiler import assets as raw_assets
from vflash.compiler.math import profile_timesteps
from vflash.contracts import ContractError
from vflash.model_assets import (
    COMPLETE_MODEL_PROFILES,
    DEFAULT_MODEL_PROFILE,
    conditioning_profile_for_request,
    model_profile,
    model_schedule,
    supported_request_modes,
    transformer_identity,
    weights_source,
)
from vflash.native.h3_runtime_artifact import _validate_source
from vflash.pipeline.contracts import VideoRequest


def test_released_model_identities_and_adapter_scaling_remain_distinct():
    ref = model_profile()
    base = model_profile("t2va-turbo4-exact-sm89")
    assert ref.transformer_component == "transformer_ref" and ref.adapter.scaling == 0.0625
    assert base.transformer_component == "transformer" and base.adapter.scaling == 1.0
    assert transformer_identity()["transformer_sha256"] == (
        "c5c855614b46cb7954fdeace235d0daf4525dd5ba10599464f4ae9ed76422765"
    )
    assert transformer_identity(base.definition.id)["transformer_sha256"] == (
        "b63ed7be75bb888ac4c46355c12f6273791f26c06773162c0e51152cd94ee05e"
    )
    i2va = model_profile("i2va-base16-bf16-sm89")
    assert i2va.transformer_component == "transformer"
    assert i2va.workflow == "fl2va"
    assert i2va.adapter is None
    assert i2va.weight_profile == "minimax-h3-base"
    assert transformer_identity(i2va.definition.id)["transformer_sha256"] != (
        transformer_identity(base.definition.id)["transformer_sha256"]
    )
    fl2va = model_profile("fl2va-base16-bf16-sm89")
    assert fl2va.definition.mode.value == "fl2va"
    assert fl2va.workflow == "fl2va" and fl2va.adapter is None
    assert fl2va.definition.nfe == 16 and fl2va.weight_profile == "minimax-h3-base"


@pytest.mark.parametrize("hardware", ["sm89", "sm86"])
def test_base16_resident_profiles_accept_both_keyframe_conditioning_modes(hardware):
    i2va = f"i2va-base16-bf16-{hardware}"
    fl2va = f"fl2va-base16-bf16-{hardware}"
    i2va_profile, fl2va_profile = model_profile(i2va), model_profile(fl2va)
    for resident in (i2va, fl2va):
        assert supported_request_modes(resident) == ("i2va", "fl2va")
        assert conditioning_profile_for_request(resident, "i2va") == i2va
        assert conditioning_profile_for_request(resident, "fl2va") == fl2va
    assert i2va_profile.hardware == fl2va_profile.hardware
    assert i2va_profile.workflow == fl2va_profile.workflow == "fl2va"
    assert i2va_profile.adapter is fl2va_profile.adapter is None
    assert (
        i2va_profile.definition.nfe,
        i2va_profile.definition.scheduler,
        i2va_profile.definition.video_flow_shift,
        i2va_profile.definition.audio_flow_shift,
        i2va_profile.definition.precision,
        i2va_profile.definition.attention,
    ) == (
        fl2va_profile.definition.nfe,
        fl2va_profile.definition.scheduler,
        fl2va_profile.definition.video_flow_shift,
        fl2va_profile.definition.audio_flow_shift,
        fl2va_profile.definition.precision,
        fl2va_profile.definition.attention,
    )
    assert model_schedule(i2va).to_mapping() == model_schedule(fl2va).to_mapping()
    assert {
        key: value
        for key, value in transformer_identity(i2va).items()
        if key != "oracle_profile"
    } == {
        key: value
        for key, value in transformer_identity(fl2va).items()
        if key != "oracle_profile"
    }
    assert weights_source(i2va)["base_transformer_sha256"] == weights_source(fl2va)[
        "base_transformer_sha256"
    ]
    with pytest.raises(ContractError, match="request mode"):
        conditioning_profile_for_request(i2va, "t2va")


def test_turbo_and_reference_profiles_remain_single_mode():
    paired = {
        "i2va-base16-bf16-sm89",
        "i2va-base16-bf16-sm86",
        "fl2va-base16-bf16-sm89",
        "fl2va-base16-bf16-sm86",
    }
    for profile_id in COMPLETE_MODEL_PROFILES:
        if profile_id not in paired:
            assert supported_request_modes(profile_id) == (
                model_profile(profile_id).definition.mode.value,
            )


def test_t2_sm86_uses_base_weights_but_requires_its_own_compilation():
    profile = model_profile("t2va-turbo4-exact-sm86")
    sm89 = model_profile("t2va-turbo4-exact-sm89")
    assert profile.transformer_component == "transformer"
    assert profile.adapter == sm89.adapter and profile.adapter.scaling == 1.0
    identity = transformer_identity(profile.definition.id)
    other = transformer_identity(sm89.definition.id)
    assert {k: v for k, v in identity.items() if k != "oracle_profile"} == {
        k: v for k, v in other.items() if k != "oracle_profile"
    }
    source = weights_source(profile.definition.id)
    assert source["oracle_profile"] == "t2va-adapter-bf16-torch-sdpa-sm86"
    assert source["compile_recipe"] == "base4-bf16-runtime-residual-sm86-v1"
    wrong_source = {
        **source,
        "oracle_profile": weights_source(sm89.definition.id)["oracle_profile"],
    }
    with pytest.raises(ValueError, match="fixed"):
        _validate_source(
            wrong_source, weight_profile=profile.adapter.profile_id, schema_version=5
        )


def test_unqualified_complete_profile_fails_before_asset_ingestion(tmp_path):
    adapter = tmp_path / "unread-adapter"
    adapter.write_bytes(b"not a model payload")
    with pytest.raises(ContractError, match="unsupported complete"):
        raw_assets.prepare_weights(
            tmp_path,
            adapter,
            tmp_path / "receipt.json",
            profile_id="ref2va-turbo8-exact-sm89",
        )
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("profile_id", COMPLETE_MODEL_PROFILES)
def test_weights_source_rejects_cross_adapter_and_partial_profile_changes(profile_id):
    source = weights_source(profile_id)
    profile = model_profile(profile_id)
    assert (
        _validate_source(source, weight_profile=profile.weight_profile, schema_version=5)
        == source
    )
    other = (
        "lightx-ref-turbo4-v0.1"
        if profile.definition.mode.value in {"t2va", "i2va"}
        else "lightx-turbo4-v1.0"
    )
    with pytest.raises(ValueError, match="fixed"):
        _validate_source(source, weight_profile=other, schema_version=5)
    for change in (
        {"adapter_alpha": "9"},
        {"compile_recipe": "unqualified"},
        {"replay_case_id": "extra"},
    ):
        with pytest.raises(ValueError, match="fixed"):
            _validate_source(
                {**source, **change},
                weight_profile=profile.weight_profile,
                schema_version=5,
            )


@pytest.mark.parametrize("profile_id", COMPLETE_MODEL_PROFILES)
@pytest.mark.parametrize("video,audio,text", [(2, 1, 1), (6232, 172, 900)])
def test_compiler_timesteps_match_the_native_request_scheduler(profile_id, video, audio, text):
    torch = pytest.importorskip("torch")
    from vflash.native.h3_native_scheduler import compile_h3_row_timestep_plan

    ref = 0 if model_profile(profile_id).definition.mode.value == "t2va" else 198
    plan = compile_h3_row_timestep_plan(
        model_schedule(profile_id),
        video_indices=torch.arange(ref + video),
        audio_indices=torch.arange(ref + video, ref + video + audio),
        text_indices=torch.arange(ref + video + audio, ref + video + audio + text),
        num_condition_video_rows=ref,
        num_condition_audio_rows=0,
        device="cpu",
    )
    rows = profile_timesteps(profile_id)
    expected_counts = (
        (1, 2, 2, 2)
        if ref == 0
        else (2,) + (3,) * (model_profile(profile_id).definition.nfe - 1)
    )
    assert tuple(row.numel() for row in rows) == expected_counts
    assert all(torch.equal(row, step.timesteps) for row, step in zip(rows, plan, strict=True))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("profile_id", COMPLETE_MODEL_PROFILES)
def test_weights_receipt_is_profile_bound_and_legacy_ref4_still_loads(
    profile_id, monkeypatch, tmp_path
):
    path = tmp_path / "weights"
    path.write_bytes(b"immutable official test input")
    expected = {
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(
        raw_assets, "_required_files", lambda *_args: {"weight": (path, expected)}
    )
    receipt = tmp_path / "receipt.json"
    prepared = raw_assets.prepare_weights(tmp_path, path, receipt, profile_id=profile_id)
    assert prepared.profile_id == profile_id
    assert raw_assets.load_prepared_weights(receipt) == prepared
    original = json.loads(receipt.read_text())
    if profile_id == DEFAULT_MODEL_PROFILE:
        legacy = {key: value for key, value in original.items() if key != "profile_id"}
        legacy["kind"] = "h3-ref4-official-weights"
        receipt.write_text(json.dumps(legacy))
        assert raw_assets.load_prepared_ref4_weights(receipt).profile_id == profile_id
    else:
        with pytest.raises(ContractError, match="matching"):
            raw_assets.load_prepared_ref4_weights(receipt)
    changed = {
        **original,
        "profile_id": next(p for p in COMPLETE_MODEL_PROFILES if p != profile_id),
    }
    receipt.write_text(json.dumps(changed))
    with pytest.raises(ContractError, match="fixed compiler"):
        raw_assets.load_prepared_weights(receipt)


def test_base16_raw_assets_omit_the_adapter_contract(tmp_path):
    rows = raw_assets._required_files(tmp_path, None, "i2va-base16-bf16-sm89")
    assert "adapter" not in rows
    with pytest.raises(ContractError, match="does not accept an adapter"):
        raw_assets._required_files(
            tmp_path,
            tmp_path / "unrequested-lora.safetensors",
            "i2va-base16-bf16-sm89",
        )


def test_t2va_request_cannot_contain_an_unbound_picture_label():
    assert VideoRequest("A scene.").audio_delivery_profile == "unchanged"
    assert VideoRequest("A scene.", audio_delivery_profile="web-v1").mode == "t2va"
    assert VideoRequest("A scene.", Path("reference.png")).mode == "ref2va"
    with pytest.raises(ContractError, match="audio delivery"):
        VideoRequest(**{"prompt": "A scene.", "audio_delivery_profile": "broadcast"})
    with pytest.raises(ContractError, match="picture label"):
        VideoRequest("A scene from <Picture 1>.")


@pytest.mark.parametrize("profile_id", COMPLETE_MODEL_PROFILES)
def test_official_call_keeps_mode_specific_inputs_and_reference_order(monkeypatch, profile_id):
    from vflash.adapters.diffusers_h3 import DiffusersConditioner

    module = ModuleType("diffusers.modular_pipelines.minimax_h3")
    module.MiniMaxH3ImageReference = lambda *, image: ("image", image)
    module.MiniMaxH3VideoReference = lambda **kwargs: ("video", kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    conditioner = DiffusersConditioner.__new__(DiffusersConditioner)
    conditioner.profile = model_profile(profile_id)
    generator = SimpleNamespace(manual_seed=lambda seed: ("generator", seed))
    conditioner._torch = SimpleNamespace(Generator=lambda: generator)
    calls = []
    conditioner.pipe = lambda **kwargs: calls.append(kwargs)
    mode = conditioner.profile.definition.mode.value
    refs = (Path("first"), Path("second")) if mode == "ref2va" else ()
    first_frame = Path("frame-zero") if mode in {"i2va", "fl2va"} else None
    last_frame = Path("frame-last") if mode == "fl2va" else None
    request = VideoRequest(
        "A scene.",
        references=refs,
        first_frame=first_frame,
        last_frame=last_frame,
        seed=97,
    )
    decoded = refs or tuple(frame for frame in (first_frame, last_frame) if frame is not None)
    conditioner._invoke(request, tuple(SimpleNamespace(image=p) for p in decoded))
    assert calls[0]["num_inference_steps"] == conditioner.profile.definition.nfe + 1
    assert calls[0]["generator"] == ("generator", 97)
    if refs:
        assert calls[0]["references"] == [("image", p) for p in refs]
    elif last_frame is not None:
        assert calls[0]["image"] == first_frame
        assert calls[0]["last_image"] == last_frame
        assert "references" not in calls[0]
    elif first_frame is not None:
        assert calls[0]["image"] == first_frame and "references" not in calls[0]
    else:
        assert "references" not in calls[0]


def test_i2va_request_owns_one_explicit_first_frame():
    frame = Path("frame-zero.png")
    request = VideoRequest(
        "<Picture 1> is the first frame. Motion begins from it.", first_frame=frame
    )
    assert request.mode == "i2va"
    assert request.first_frame == frame
    assert request.ordered_references == ()
    with pytest.raises(ContractError, match="first frame or Ref2VA"):
        VideoRequest("A scene.", references=(Path("ref.png"),), first_frame=frame)
    with pytest.raises(ContractError, match="picture label"):
        VideoRequest("Continue from <Picture 2>.", first_frame=frame)


def test_fl2va_request_requires_two_explicit_temporal_anchors():
    first, last = Path("frame-zero.png"), Path("frame-last.png")
    request = VideoRequest(
        "Move from <Picture 1> to <Picture 2>.",
        first_frame=first,
        last_frame=last,
    )
    assert request.mode == "fl2va"
    assert (request.first_frame, request.last_frame) == (first, last)
    assert request.ordered_references == ()
    with pytest.raises(ContractError, match="requires a first_frame"):
        VideoRequest("A scene.", last_frame=last)
    with pytest.raises(ContractError, match="picture label"):
        VideoRequest(
            "Move from <Picture 1> through <Picture 3>.",
            first_frame=first,
            last_frame=last,
        )
    with pytest.raises(ContractError, match="first frame or Ref2VA"):
        VideoRequest(
            "A scene.",
            references=(Path("ref.png"),),
            first_frame=first,
            last_frame=last,
        )


@pytest.mark.parametrize("profile_id", COMPLETE_MODEL_PROFILES)
def test_complete_constructor_owns_one_explicit_device_group(monkeypatch, tmp_path, profile_id):
    from vflash.hardware import NvidiaDevice
    from vflash.pipeline.assets import PreparedPipelineAssets
    from vflash.pipeline.contracts import PipelineAssets
    from vflash.pipeline.runtime import H3Pipeline

    profile = model_profile(profile_id)
    assets = PipelineAssets(**{name: tmp_path for name in PipelineAssets.__dataclass_fields__})
    prepared = PreparedPipelineAssets(assets, tmp_path / "receipt", "a" * 64, (), profile_id)
    device = NvidiaDevice(
        0, "primary-test-device", "test", 48, profile.hardware.compute_capability, 300
    )
    peer = (
        NvidiaDevice(1, "peer-test-device", "test", 20, "8.6", 300)
        if profile.hardware.compute_capability == "8.6"
        else None
    )
    plans = []
    monkeypatch.setattr("vflash.pipeline.runtime.media_executables", lambda: None)
    monkeypatch.setattr("vflash.pipeline.runtime.validate_adapter_dependencies", lambda: None)
    monkeypatch.setattr(H3Pipeline, "_load_stages", lambda self, plan: plans.append(plan))
    with H3Pipeline(
        prepared, device=device, peer_device=peer, trust_local_code=True
    ) as pipeline:
        bad = (
            VideoRequest("A scene.", Path("missing"))
            if profile.definition.mode.value == "t2va"
            else VideoRequest("A scene.")
        )
        with pytest.raises(ContractError, match="request mode"):
            pipeline.generate(bad, tmp_path / "bad.mp4")
        assert not pipeline._closed
        assert not plans
        pipeline.prepare()
    assert plans[0].profile.id == profile_id
    assert plans[0].parallel_strategy == ("sequence-head" if peer else "single")
    assert plans[0].gpu_uuids == ((device.uuid, peer.uuid) if peer else (device.uuid,))


def test_pipeline_assets_reject_wrong_mode_adapter_and_gpu_target(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from vflash.pipeline import assets as pipeline_assets
    from vflash.pipeline.contracts import PipelineAssets

    paths = PipelineAssets(**{name: tmp_path for name in PipelineAssets.__dataclass_fields__})
    t2 = model_profile("t2va-turbo4-exact-sm89")
    artifact = SimpleNamespace(
        weight_profile=t2.adapter.profile_id,
        nfe=4,
        adapter_execution="runtime-residual",
        is_complete_block_stack=True,
        target=SimpleNamespace(compute_capability="sm89"),
        source=weights_source(t2.definition.id),
        blocks=(),
    )
    monkeypatch.setattr(pipeline_assets, "_consumed_official_files", lambda *_args: [])
    monkeypatch.setattr(
        pipeline_assets, "load_h3_runtime_artifact", lambda *_a, **_kw: artifact
    )
    schedule = model_schedule(t2.definition.id)
    monkeypatch.setattr(
        pipeline_assets,
        "load_h3_schedule_overlay",
        lambda *_a, **_kw: SimpleNamespace(schedule=schedule),
    )
    rows = pipeline_assets._planned_files(paths, t2.definition.id)
    assert rows[0][2]["sha256"] == t2.adapter.sha256
    with pytest.raises(ContractError, match="profile"):
        pipeline_assets._planned_files(paths, DEFAULT_MODEL_PROFILE)
    artifact.target.compute_capability = "sm86"
    with pytest.raises(ContractError, match="profile"):
        pipeline_assets._planned_files(paths, t2.definition.id)
    artifact.target.compute_capability = "sm89"
    schedule = model_schedule(DEFAULT_MODEL_PROFILE)
    with pytest.raises(ContractError, match=r"shift|schedule"):
        pipeline_assets._planned_files(paths, t2.definition.id)
