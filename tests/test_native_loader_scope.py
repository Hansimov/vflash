"""Native factory ownership without loading model payloads or creating CUDA."""

from types import SimpleNamespace

import pytest

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.native import h3_native_conditioning_runtime as native
from vflash.native import h3_native_denoiser as denoiser
from vflash.native.runner import NativeEngineSession
from vflash.planner import resolve_plan


def test_stack_factory_does_not_replace_default_block_class(tmp_path, monkeypatch):
    original = denoiser.H3NativeBlockBF16Resident
    artifact = SimpleNamespace(
        is_complete_block_stack=True,
        blocks=(SimpleNamespace(index=0), SimpleNamespace(index=1)),
    )
    monkeypatch.setattr(denoiser, "load_h3_runtime_artifact", lambda _: artifact)

    class Stack(denoiser.H3NativeDenoiserBF16Resident):
        def __init__(self, blocks):
            self.blocks = blocks

    def load_a(artifact, index, **options):
        return ("a", index, options["adaln_table"])

    def load_b(artifact, index, **options):
        if index == 1:
            raise ValueError("owned load failure")
        return ("b", index)

    assert Stack.load(
        tmp_path, block_loader=load_a, adaln_table_load=lambda i: i + 10
    ).blocks == (("a", 0, 10), ("a", 1, 11))
    with pytest.raises(ValueError, match="owned load failure"):
        Stack.load(tmp_path, block_loader=load_b)
    assert Stack.load(tmp_path, block_loader=load_a).blocks == (("a", 0, None), ("a", 1, None))
    assert denoiser.H3NativeBlockBF16Resident is original
    with pytest.raises(ValueError, match="callable"):
        Stack.load(tmp_path, block_loader="module.factory")


def test_default_block_placement_stays_strict_bf16():
    torch = pytest.importorskip("torch")
    owner = denoiser.H3NativeBlockBF16Resident.__new__(denoiser.H3NativeBlockBF16Resident)
    values = torch.arange(8).reshape(2, 4).bfloat16()
    weight = denoiser.H3BF16Weight(values, torch.ones(2), 16, None, 4, 2)
    actual = owner._resident_weight(weight, "cpu")
    assert actual.bits == 16 and torch.equal(actual.values, values)
    from dataclasses import replace

    with pytest.raises(ValueError, match="unquantized"):
        owner._resident_weight(replace(weight, bits=8), "cpu")
    assert not torch.cuda.is_initialized()


def test_engine_forwards_factory_without_global_runtime_replacement(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    loads = []
    original_runtime = native.H3NativeConditioningRuntime

    class Runtime:
        def __init__(self, **options):
            loads.append(options)

        def close(self):
            pass

    monkeypatch.setattr(native, "H3NativeConditioningRuntime", Runtime)
    monkeypatch.setattr("vflash.native.runner.require_attention_dependencies", lambda _: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "fixture")
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id="ref2va-turbo4-exact-sm89",
        device=NvidiaDevice(0, "fixture", "RTX 4090", 48, "8.9", 450),
    )
    options = dict(
        artifact=tmp_path,
        schedule_overlay=tmp_path,
        auxiliary_tensor=tmp_path,
        weight_residency="resident",
        attention_backend="torch-flash",
    )
    factories = [lambda *a, **k: "first", lambda *a, **k: "second"]
    for factory in factories:
        with NativeEngineSession(plan, **options, native_denoiser_loader=factory):
            assert loads[-1]["native_denoiser_loader"] is factory
    assert not torch.cuda.is_initialized()
    for bad in ("module.factory", {}, 7):
        with pytest.raises(ContractError, match="callable"):
            NativeEngineSession(plan, **options, native_denoiser_loader=bad)
    with pytest.raises(native.H3NativeConditioningRuntimeError, match="resident"):
        original_runtime(
            artifact_path=tmp_path,
            schedule_overlay_path=tmp_path,
            auxiliary_tensor_path=tmp_path,
            native_denoiser_loader=factories[0],
        )


def test_runtime_components_use_the_owning_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "load_h3_native_input_projection_weights", lambda **kw: kw)
    monkeypatch.setattr(native, "H3NativeInputPacker", lambda value: value)
    monkeypatch.setattr(native, "load_h3_native_final_layer_weights", lambda **kw: kw)
    monkeypatch.setattr(native, "H3NativeFinalLayer", lambda value: value)
    default = native.H3NativeDenoiserBF16Resident
    seen = []
    for marker in ("first", "second"):
        owner = native.H3NativeConditioningRuntime.__new__(native.H3NativeConditioningRuntime)
        owner._torch = SimpleNamespace(
            cuda=SimpleNamespace(empty_cache=lambda: None, synchronize=lambda _: None)
        )
        owner.artifact = object()
        owner.overlay = SimpleNamespace(
            blocks=(),
            load_auxiliary_tensors=lambda: {"time_embeddings": 1, "final_adaln_table": 2},
        )
        owner.device, owner.devices = "cpu", ("cpu",)
        owner.parallel_strategy, owner.weight_residency = "single", "resident"

        def factory(artifact, _marker=marker, **options):
            seen.append((_marker, artifact, options["device"]))
            return _marker

        owner._native_denoiser_loader = factory
        owner._load_components(SimpleNamespace(load=lambda _: None))
        assert owner.denoiser == marker and seen[-1][1] is owner.artifact
    assert [row[0] for row in seen] == ["first", "second"]
    assert native.H3NativeDenoiserBF16Resident is default
