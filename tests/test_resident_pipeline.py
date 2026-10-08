from types import SimpleNamespace as N

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.residency import validate_residency, validate_resident_request


def plan(cap="9.0", memory=79, strategy="single"):
    return N(
        target=N(compute_capability=cap), gpu_memory_gib=memory, parallel_strategy=strategy
    )


def test_explicit_resident_requires_large_h100_and_tested_allocator(monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert validate_residency(plan(), "block-ring", "veda-triton", object()) == "block-ring"
    with pytest.raises(ContractError, match="expandable_segments"):
        validate_residency(plan(), "resident", "veda-triton", object())
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    assert validate_residency(plan(), "resident", "veda-triton", object()) == "resident"
    for candidate in (
        plan("8.9", 48),
        plan("12.0", 95),
        plan(memory=48),
        plan(strategy="sequence-head"),
    ):
        with pytest.raises(ContractError, match="single"):
            validate_residency(candidate, "resident", "veda-triton", object())
    for attention, hybrid in (("torch-flash", object()), ("veda-triton", None)):
        with pytest.raises(ContractError, match="single"):
            validate_residency(plan(), "resident", attention, hybrid)


def test_resident_capacity_rejects_larger_or_different_modes_before_execution():
    request = N(
        mode="i2va", width=1536, height=864, model_frames=362, ordered_references=(object(),)
    )
    validate_resident_request("resident", request)
    for change in (
        {"width": 1920, "height": 1088},
        {"model_frames": 363},
        {"mode": "fl2va"},
        {"ordered_references": (object(), object())},
    ):
        with pytest.raises(ContractError, match="resident I2VA"):
            validate_resident_request("resident", N(**(vars(request) | change)))
        validate_resident_request("block-ring", N(**(vars(request) | change)))


def test_explicit_residency_reaches_native_owner_before_any_cuda_api(monkeypatch):
    import sys
    from pathlib import Path

    from vflash.pipeline.runtime import H3Pipeline

    monkeypatch.setitem(
        sys.modules,
        "torch",
        N(set_num_threads=lambda _: None, get_num_interop_threads=lambda: 1),
    )
    calls = []

    def native(plan, **options):
        calls.append(options["weight_residency"])
        return object()

    monkeypatch.setattr("vflash.pipeline.runtime.NativeEngineSession", native)
    monkeypatch.setattr("vflash.pipeline.runtime.DiffusersConditioner", lambda _: object())
    monkeypatch.setattr("vflash.pipeline.runtime.OfficialMediaDecoder", lambda **_: object())
    pipeline = H3Pipeline.__new__(H3Pipeline)
    pipeline.prepared = N(
        assets=N(
            artifact=Path("artifact"),
            schedule_overlay=Path("schedule"),
            auxiliary_tensor=Path("auxiliary"),
            decoder_directory=Path("decoder"),
        )
    )
    pipeline.weight_residency = "resident"
    pipeline.attention_backend = "veda-triton"
    pipeline.veda_predictor = pipeline.hybrid_model = pipeline.attention_adapter = None
    pipeline.initialization_stages = {}
    pipeline._load_stages(plan())
    assert calls == ["resident"]
