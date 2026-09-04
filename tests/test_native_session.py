import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.native.runner import NativeEngineSession
from vflash.planner import resolve_plan


@pytest.mark.parametrize(
    "profile_id,capability,memory",
    [
        ("ref2va-turbo4-exact-sm89", "8.9", 48.0),
        ("ref2va-turbo4-exact-sm86", "8.6", 20.0),
    ],
)
def test_session_loads_once_and_keeps_request_accounting_separate(
    monkeypatch, tmp_path, profile_id, capability, memory
):
    loads = []
    calls = []

    @dataclass
    class Result:
        output_path: Path
        nfe: int = 4

    class Runtime:
        def __init__(self, **options):
            loads.append(options)

        def generate_latents(self, bundle, output):
            calls.append((bundle, output))
            return Result(output)

        def metadata(self):
            return {"initialization_seconds": 12.0}

    monkeypatch.setattr(
        "vflash.native.h3_native_conditioning_runtime.H3NativeConditioningRuntime", Runtime
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "prior-selection")
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id=profile_id,
        device=NvidiaDevice(3, "test-uuid", "Test GPU", memory, capability, 320.0),
    )
    session = NativeEngineSession(
        plan,
        artifact=tmp_path / "artifact",
        schedule_overlay=tmp_path / "schedule",
        auxiliary_tensor=tmp_path / "auxiliary",
    )
    first = session.generate(tmp_path / "bundle-a", tmp_path / "first")
    second = session.generate(tmp_path / "bundle-b", tmp_path / "second")

    assert len(loads) == 1
    assert calls == [
        (tmp_path / "bundle-a", tmp_path / "first"),
        (tmp_path / "bundle-b", tmp_path / "second"),
    ]
    assert first["session"]["initialization_charged_seconds"] > 0
    assert second["session"]["initialization_charged_seconds"] == 0
    assert second["session"]["request_index"] == 2
    assert second["generation"]["output_path"] == str(tmp_path / "second")


def test_session_rejects_a_target_outside_its_profile(tmp_path):
    catalog = ProfileCatalog.bundled()
    plan = resolve_plan(
        catalog,
        profile_id="ref2va-turbo4-exact-sm89",
        device=NvidiaDevice(0, "test-uuid", "Test GPU", 48.0, "8.9", 450.0),
    )
    inconsistent = replace(plan, target=catalog.target("sm86-20g-block-ring"))
    with pytest.raises(ContractError, match="public denoiser supports"):
        NativeEngineSession(
            inconsistent,
            artifact=tmp_path / "artifact",
            schedule_overlay=tmp_path / "schedule",
            auxiliary_tensor=tmp_path / "auxiliary",
        )
