from types import SimpleNamespace as N

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.residency import validate_residency, validate_resident_request


def plan(cap="9.0", memory=79, strategy="single"):
    return N(
        target=N(compute_capability=cap), gpu_memory_gib=memory, parallel_strategy=strategy
    )


def test_explicit_resident_requires_qualified_memory_and_tested_allocator(monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert validate_residency(plan(), "block-ring", "veda-triton", object()) == "block-ring"
    with pytest.raises(ContractError, match="expandable_segments"):
        validate_residency(plan(), "resident", "veda-triton", object())
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    assert validate_residency(plan(), "resident", "veda-triton", object()) == "resident"
    for accepted in (plan("9.0", 75), plan("12.0", 90), plan("12.0", 95)):
        assert validate_residency(accepted, "resident", "veda-triton", object()) == "resident"
    for candidate in (
        plan("8.9", 48),
        plan("12.0", 48),
        plan("12.0", 32),
        plan("12.0", 89.9),
        plan("10.3", 100),
        plan("9.0", 74.9),
        plan(memory=48),
        plan(strategy="sequence-head"),
    ):
        with pytest.raises(ContractError, match="single"):
            validate_residency(candidate, "resident", "veda-triton", object())
    for attention, hybrid in (("torch-flash", object()), ("veda-triton", None)):
        with pytest.raises(ContractError, match="single"):
            validate_residency(plan(), "resident", attention, hybrid)


def test_resident_capacity_uses_real_keyframe_contract():
    from dataclasses import replace
    from pathlib import Path

    from vflash.pipeline.contracts import VideoRequest

    request = VideoRequest(
        prompt="A clock moves.",
        first_frame=Path("first.png"),
        width=1536,
        height=864,
        duration_seconds=15,
    )
    assert request.ordered_references == ()
    validate_resident_request("resident", request)
    validate_resident_request(
        "resident", replace(request, width=512, height=512, duration_seconds=5)
    )
    for change in (
        {"width": 1920, "height": 1088},
        {"last_frame": Path("last.png")},
        {"first_frame": None, "width": 512, "height": 512},
        {"first_frame": None, "references": (Path("ref.png"),)},
    ):
        changed = replace(request, duration_seconds=5, **change)
        with pytest.raises(ContractError, match="resident I2VA"):
            validate_resident_request("resident", changed)
        validate_resident_request("block-ring", changed)


def test_sm120_residency_is_budgeted_by_pixel_frames():
    from dataclasses import replace
    from pathlib import Path

    from vflash.pipeline.contracts import VideoRequest

    sm120 = "i2va-turbo4-v01-544-exact-sm120"
    request = VideoRequest(
        prompt="A clock moves.",
        first_frame=Path("first.png"),
        width=1536,
        height=864,
        duration_seconds=15,
    )
    validate_resident_request("resident", request, sm120)
    for width, height, seconds in (
        (1920, 1088, 5),
        (2720, 1536, 5),
        (2048, 2048, 5),
        (1920, 1088, 10),
    ):
        validate_resident_request(
            "resident",
            replace(request, width=width, height=height, duration_seconds=seconds),
            sm120,
        )
    for width, height, seconds in ((2048, 2048, 10), (1920, 1088, 15)):
        changed = replace(request, width=width, height=height, duration_seconds=seconds)
        with pytest.raises(ContractError, match="pixel-frames"):
            validate_resident_request("resident", changed, sm120)
    # SM90 keeps its measured canvas envelope.
    with pytest.raises(ContractError, match="1536x864 pixels"):
        validate_resident_request(
            "resident",
            replace(request, width=1920, height=1088, duration_seconds=5),
            "i2va-turbo4-v01-544-exact-sm90",
        )


@pytest.mark.parametrize(
    "dense_backend", ["torch-flash", "sageattention2", "cute-flash", "frost-flash", "fa3-flash"]
)
@pytest.mark.parametrize("dense_edges", [3, 5])
def test_explicit_residency_reaches_native_owner_before_any_cuda_api(
    monkeypatch, dense_backend, dense_edges
):
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
        calls.append(
            (
                options["weight_residency"],
                options.get("veda_dense_backend", "torch-flash"),
                options.get("veda_dense_edge_layers", 5),
            )
        )
        return object()

    monkeypatch.setattr("vflash.pipeline.runtime.NativeEngineSession", native)
    monkeypatch.setattr("vflash.pipeline.runtime.DiffusersConditioner", lambda _: object())
    monkeypatch.setattr("vflash.pipeline.runtime.OfficialMediaDecoder", lambda **_: object())
    pipeline = H3Pipeline.__new__(H3Pipeline)
    pipeline._text_encoder_loader = None
    pipeline._native_denoiser_loader = None
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
    pipeline.veda_dense_backend = dense_backend
    pipeline.veda_dense_edge_layers = dense_edges
    pipeline.veda_predictor = pipeline.hybrid_model = pipeline.attention_adapter = None
    pipeline.initialization_stages = {}
    pipeline._load_stages(plan())
    assert calls == [("resident", dense_backend, dense_edges)]
