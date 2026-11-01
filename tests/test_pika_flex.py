from dataclasses import replace
from pathlib import Path

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.pika_flex import (
    PIKA_ADAPTER_SHA256,
    PIKA_BASE_REVISION,
    PIKA_KNOTS,
    PIKA_TARGET_MODULES,
    PikaFlexRequest,
    adapter_partition,
    shifted_sigmas,
    validate_adapter_manifest,
)


def manifest(partition):
    return {
        "format_version": 1,
        "model": "pika-flex",
        "task": partition,
        "release_id": f"pika-flex-{partition}-v1",
        "rank": 128,
        "alpha": 128,
        "trunk": "transformer_ref" if partition == "ref2va" else "transformer",
        "numerics_policy": "fp32_conditioning_v1"
        if partition == "ref2va"
        else "bf16_conditioning_v1",
        "sha256": PIKA_ADAPTER_SHA256[partition],
        "base": {"repo_id": "MiniMaxAI/MiniMax-H3", "revision": PIKA_BASE_REVISION},
        "sampling": {
            "knots": list(PIKA_KNOTS),
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "method": "renoise",
        },
        "target_modules": list(PIKA_TARGET_MODULES),
    }


def test_native_text_has_no_anchor_and_uses_fl_partition():
    assert adapter_partition(PikaFlexRequest("A ferry leaves the pier.").mode) == "fl2va"
    for mode in ("i2va", "l2va", "fl2va"):
        assert adapter_partition(mode) == "fl2va"
    assert adapter_partition("ref2va") == "ref2va"


@pytest.mark.parametrize(
    "changes",
    [
        {"first_frame": Path("first.png")},
        {"mode": "i2va"},
        {"mode": "fl2va", "first_frame": Path("first.png")},
        {"mode": "ref2va", "references": (("audio", Path("voice.wav")),)},
        {"references": (("image", Path("character.png")),)},
        {"mode": "reference"},
        {"resolution": "1080p"},
        {"aspect": "2:1"},
        {"duration_seconds": 5.0},
        {"duration_seconds": True},
        {"seed": True},
        {"prompt": " "},
    ],
)
def test_inputs_cannot_silently_drop_or_reinterpret_references(changes):
    with pytest.raises(ContractError):
        replace(PikaFlexRequest("A ferry leaves the pier."), **changes)


def test_each_temporal_anchor_role_is_explicit():
    first, last = Path("start.png"), Path("end.png")
    assert PikaFlexRequest("Open the sail.", mode="i2va", first_frame=first).last_frame is None
    assert PikaFlexRequest("Fold the sail.", mode="l2va", last_frame=last).first_frame is None
    assert (
        PikaFlexRequest(
            "Fold the sail.", mode="fl2va", first_frame=first, last_frame=last
        ).last_frame
        == last
    )


def test_actual_release_manifest_and_wrong_trunk_or_sampler():
    validate_adapter_manifest(manifest("fl2va"), "t2va")
    validate_adapter_manifest(manifest("ref2va"), "ref2va")
    for change in (
        {"alpha": 8},
        {"trunk": "transformer_ref"},
        {"numerics_policy": "fp32_conditioning_v1"},
        {"sampling": {"method": "euler", "steps": 6}},
        {"sha256": "0" * 64},
    ):
        with pytest.raises(ContractError):
            validate_adapter_manifest({**manifest("fl2va"), **change}, "t2va")
    with pytest.raises(ContractError):
        validate_adapter_manifest(manifest("fl2va"), "ref2va")


def test_trained_ladder_is_not_uniform_six_step_euler():
    video, audio = shifted_sigmas(12), shifted_sigmas(3)
    assert len(video) == len(audio) == 7
    assert video[0] == audio[0] == 1
    assert video[-1] == audio[-1] == 0
    assert video[1] == pytest.approx(36 / 37)
    assert audio[1] == pytest.approx(9 / 10)
    assert video[-2] == pytest.approx(1 / 3)
    assert audio[-2] == pytest.approx(1 / 9)
    with pytest.raises(ContractError):
        shifted_sigmas(6)


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from vflash.adapters import pika_flex_source
    from vflash.pipeline import pika_flex_runtime

    root = tmp_path / "adapters/fl2va"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(manifest("fl2va")))
    (root / "adapter.safetensors").write_bytes(b"test")
    monkeypatch.setattr(pika_flex_runtime, "PIKA_ADAPTER_BYTES", 4)
    calls = []

    def generate(args):
        from vflash.pipeline.pika_flex import PIKA_MODEL_REVISION

        calls.append(args)
        data = {
            "sampling": manifest("fl2va")["sampling"],
            "adapter_sha256": PIKA_ADAPTER_SHA256["fl2va"],
            "adapter_revision": PIKA_MODEL_REVISION,
            "base": manifest("fl2va")["base"],
            "numerics_policy": "bf16_conditioning_v1",
            "task": args.task,
            "prompt": args.prompt,
            "seed": args.seed,
        }
        args.output.write_bytes(b"test-only media")
        args.output.with_suffix(".json").write_text(json.dumps(data))

    pipeline = SimpleNamespace(generate=generate)
    monkeypatch.setattr(pika_flex_source, "validate_pika_environment", lambda: None)
    monkeypatch.setattr(pika_flex_source, "load_pika_source", lambda *a, **k: pipeline)
    runtime = pika_flex_runtime.PikaFlexRuntime(
        source_root=tmp_path,
        base_path=tmp_path,
        adapters_root=root.parent,
        trust_local_code=True,
    )
    return runtime, calls, tmp_path / "take.mp4"


def test_runtime_keeps_text_direct_and_never_adds_legacy_audio(runtime_fixture):
    runtime, calls, output = runtime_fixture
    result = runtime.generate(PikaFlexRequest("A ferry leaves.", seed=42), output)
    assert len(calls) == 1 and calls[0].task == "t2va"
    assert calls[0].first_image is None and calls[0].last_image is None
    assert calls[0].ref == [] and calls[0].seed == 42
    assert result["separate_audio_pass"] is False
    assert result["weight_format"] == "original-bf16"
    with pytest.raises(ContractError, match="new MP4"):
        runtime.generate(PikaFlexRequest("Another ferry."), output)
    assert len(calls) == 1
    runtime.close()
    with pytest.raises(ContractError, match="closed"):
        runtime.generate(PikaFlexRequest("Another ferry."), output.parent / "new.mp4")


def test_reference_partition_never_falls_back_to_fl(runtime_fixture):
    runtime, calls, output = runtime_fixture
    with pytest.raises(ContractError, match="partition"):
        runtime.generate(
            PikaFlexRequest(
                "Follow the reference.", mode="ref2va", references=(("image", Path("ref.png")),)
            ),
            output,
        )
    assert calls == []


def test_failed_execution_cannot_reuse_partial_gpu_state(runtime_fixture):
    runtime, _calls, output = runtime_fixture

    def fail(_):
        raise RuntimeError("simulated GPU failure")

    runtime.pipeline.generate = fail
    with pytest.raises(RuntimeError, match="GPU failure"):
        runtime.generate(PikaFlexRequest("A ferry leaves."), output)
    assert runtime.closed
    runtime.close()


def test_actual_receipt_mismatch_cannot_be_accepted(runtime_fixture):
    import json

    runtime, _calls, output = runtime_fixture
    original = runtime.pipeline.generate

    def mismatch(args):
        original(args)
        metadata = args.output.with_suffix(".json")
        value = json.loads(metadata.read_text())
        value["numerics_policy"] = "wrong-conditioning"
        metadata.write_text(json.dumps(value))

    runtime.pipeline.generate = mismatch
    with pytest.raises(ContractError, match="receipt"):
        runtime.generate(PikaFlexRequest("A ferry leaves."), output)
    assert runtime.closed and output.exists()


def test_source_requires_explicit_trust_before_any_import(tmp_path):
    from vflash.adapters.pika_flex_source import load_pika_source

    with pytest.raises(ContractError, match="trust_local_code"):
        load_pika_source(tmp_path)
    with pytest.raises(ContractError, match="source files"):
        load_pika_source(tmp_path, trust_local_code=True)


def test_reference_environment_rejects_inherited_pipeline_versions(monkeypatch):
    from vflash.adapters import pika_flex_source as source

    monkeypatch.setattr(source.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(source, "version", lambda name: source.PIKA_DEPENDENCIES[name])
    source.validate_pika_environment()
    monkeypatch.setattr(
        source,
        "version",
        lambda name: "2.11.0" if name == "torch" else source.PIKA_DEPENDENCIES[name],
    )
    with pytest.raises(ContractError, match="dependency differs: torch"):
        source.validate_pika_environment()
    monkeypatch.setattr(source.sys, "version_info", (3, 11, 0))
    with pytest.raises(ContractError, match=r"Python 3\.12"):
        source.validate_pika_environment()
