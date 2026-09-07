import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.native.h3_mixed_ffnin import PROFILE_ID as MIXED_PROFILE
from vflash.native.runner import NativeEngineSession
from vflash.planner import resolve_plan


@pytest.mark.parametrize(
    "profile_id,capability,memory,strategy,residency",
    [
        ("ref2va-turbo4-exact-sm89", "8.9", 48.0, "single", "default"),
        ("t2va-turbo4-exact-sm89", "8.9", 48.0, "single", "default"),
        ("ref2va-turbo8-exact-sm89", "8.9", 48.0, "single", "block-ring"),
        ("ref2va-turbo4-exact-sm86", "8.6", 20.0, "single", "default"),
        ("ref2va-turbo4-exact-sm86", "8.6", 20.0, "tensor", "default"),
        ("ref2va-turbo4-exact-sm86", "8.6", 20.0, "sequence-head", "block-ring"),
    ],
)
def test_session_loads_once_and_keeps_request_accounting_separate(
    monkeypatch, tmp_path, profile_id, capability, memory, strategy, residency
):
    loads = []
    calls = []
    closes = []

    @dataclass
    class Result:
        output_path: Path
        nfe: int = 4

    class Runtime:
        def __init__(self, **options):
            loads.append(options)

        def generate_latents(self, bundle, output, *, progress_callback=None):
            calls.append((bundle, output))
            if progress_callback is not None:
                for completed in range(1, 5):
                    progress_callback(completed, 4)
            return Result(output)

        def metadata(self):
            return {"initialization_seconds": 12.0}

        def close(self):
            closes.append(True)

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
        peer_device=(
            NvidiaDevice(5, "peer-uuid", "Peer GPU", memory, capability, 320.0)
            if strategy != "single"
            else None
        ),
        strategy=strategy,
    )
    session = NativeEngineSession(
        plan,
        artifact=tmp_path / "artifact",
        schedule_overlay=tmp_path / "schedule",
        auxiliary_tensor=tmp_path / "auxiliary",
        weight_residency=residency,
    )
    first = session.generate(tmp_path / "bundle-a", tmp_path / "first")
    progress = []
    second = session.generate(
        tmp_path / "bundle-b",
        tmp_path / "second",
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    assert progress == [(completed, 4) for completed in range(1, 5)]

    assert len(loads) == 1
    assert calls == [
        (tmp_path / "bundle-a", tmp_path / "first"),
        (tmp_path / "bundle-b", tmp_path / "second"),
    ]
    assert first["session"]["initialization_charged_seconds"] > 0
    assert second["session"]["initialization_charged_seconds"] == 0
    assert second["session"]["request_index"] == 2
    assert second["generation"]["output_path"] == str(tmp_path / "second")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == (
        "test-uuid" if strategy == "single" else "test-uuid,peer-uuid"
    )
    assert loads[0]["expected_task"] == plan.profile.mode.value
    assert loads[0]["parallel_strategy"] == strategy
    assert loads[0]["weight_residency"] == residency
    assert second["parallel"]["strategy"] == strategy
    session.close()
    session.close()
    assert closes == [True]
    with pytest.raises(ContractError, match="session is closed"):
        session.generate(tmp_path / "bundle-c", tmp_path / "third")


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


@pytest.mark.parametrize(
    "profile_id,sidecar,residency",
    [
        (MIXED_PROFILE, None, "default"),
        (MIXED_PROFILE, "weights", "resident"),
        ("ref2va-turbo4-exact-sm89", "weights", "block-ring"),
    ],
)
def test_precision_profile_and_payload_cannot_be_mixed(
    tmp_path, profile_id, sidecar, residency
):
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id=profile_id,
        device=NvidiaDevice(0, "test-uuid", "Test GPU", 48.0, "8.9", 450.0),
    )
    with pytest.raises(ContractError, match="mixed FFN-in profile"):
        NativeEngineSession(
            plan,
            artifact=tmp_path / "artifact",
            schedule_overlay=tmp_path / "schedule",
            auxiliary_tensor=tmp_path / "auxiliary",
            mixed_ffn_in=tmp_path / sidecar if sidecar else None,
            weight_residency=residency,
        )


def test_mixed_session_uses_its_own_runtime_and_forces_two_slot_residency(
    monkeypatch, tmp_path
):
    loads = []
    sidecars = []

    class Runtime:
        def __init__(self, **options):
            loads.append(options)

        def close(self):
            pass

    def choose(sidecar):
        sidecars.append(sidecar)
        return Runtime

    monkeypatch.setattr("vflash.native.h3_mixed_ffnin.mixed_ffn_in_runtime", choose)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id=MIXED_PROFILE,
        device=NvidiaDevice(0, "test-uuid", "Test GPU", 48.0, "8.9", 450.0),
    )
    with NativeEngineSession(
        plan,
        artifact=tmp_path / "artifact",
        schedule_overlay=tmp_path / "schedule",
        auxiliary_tensor=tmp_path / "auxiliary",
        mixed_ffn_in=tmp_path / "weights",
    ):
        pass
    assert sidecars == [tmp_path / "weights"]
    assert loads[0]["weight_residency"] == "block-ring"
    assert loads[0]["expected_nfe"] == 4
    assert loads[0]["expected_weight_profile"] == "lightx-ref-turbo4-v0.1"
    assert loads[0]["parallel_strategy"] == "single"
