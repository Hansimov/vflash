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
    model_profile,
    model_schedule,
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
        _validate_source(source, weight_profile=profile.adapter.profile_id, schema_version=5)
        == source
    )
    other = (
        "lightx-ref-turbo4-v0.1"
        if profile.definition.mode.value == "t2va"
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
                weight_profile=profile.adapter.profile_id,
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
    assert tuple(row.numel() for row in rows) == ((1, 2, 2, 2) if ref == 0 else (2, 3, 3, 3))
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


def test_t2va_request_cannot_contain_an_unbound_picture_label():
    assert VideoRequest("A scene.").mode == "t2va"
    assert VideoRequest("A scene.", Path("reference.png")).mode == "ref2va"
    with pytest.raises(ContractError, match="picture label"):
        VideoRequest("A scene from <Picture 1>.")


@pytest.mark.parametrize("profile_id", COMPLETE_MODEL_PROFILES)
def test_official_call_keeps_mode_specific_inputs_and_reference_order(monkeypatch, profile_id):
    from vflash.adapters.diffusers_h3 import DiffusersConditioner

    module = ModuleType("diffusers.modular_pipelines.minimax_h3")
    module.MiniMaxH3ImageReference = lambda *, image: ("image", image)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    conditioner = DiffusersConditioner.__new__(DiffusersConditioner)
    conditioner.profile = model_profile(profile_id)
    generator = SimpleNamespace(manual_seed=lambda seed: ("generator", seed))
    conditioner._torch = SimpleNamespace(Generator=lambda: generator)
    calls = []
    conditioner.pipe = lambda **kwargs: calls.append(kwargs)
    refs = (
        (Path("first"), Path("second"))
        if conditioner.profile.definition.mode.value == "ref2va"
        else ()
    )
    request = VideoRequest("A scene.", references=refs, seed=97)
    conditioner._invoke(request, tuple(SimpleNamespace(image=p) for p in refs))
    assert calls[0]["num_inference_steps"] == 5
    assert calls[0]["generator"] == ("generator", 97)
    if refs:
        assert calls[0]["references"] == [("image", p) for p in refs]
    else:
        assert "references" not in calls[0]


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
