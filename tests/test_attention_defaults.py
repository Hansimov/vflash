import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vflash.attention import require_attention_dependencies, resolve_attention_backend
from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.planner import resolve_plan


def plan(profile="i2va-base16-bf16-sm89", pair=False):
    sm86 = profile.endswith("sm86")
    device = NvidiaDevice(
        0, "primary", "Test GPU", 20 if sm86 else 48, "8.6" if sm86 else "8.9", 320
    )
    return resolve_plan(
        ProfileCatalog.bundled(),
        profile_id=profile,
        device=device,
        peer_device=replace(device, index=1, uuid="peer") if pair else None,
        strategy="sequence-head" if pair else "single",
    )


@pytest.mark.parametrize(
    "profile,pair,expected",
    [
        ("i2va-base16-bf16-sm89", False, "sol-sm89"),
        ("fl2va-base16-bf16-sm89", False, "sol-sm89"),
        ("i2va-base16-bf16-sm89", True, "torch-flash"),
        ("fl2va-base16-bf16-sm89", True, "torch-flash"),
        ("i2va-base16-bf16-sm86", True, "torch-flash"),
        ("ref2va-turbo4-exact-sm89", False, "torch-flash"),
        ("ref2va-turbo8-exact-sm89", False, "torch-flash"),
        ("t2va-turbo4-exact-sm89", False, "torch-flash"),
        ("t2va-turbo4-exact-sm86", True, "torch-flash"),
    ],
)
def test_auto_is_a_fixed_profile_policy_not_dependency_fallback(profile, pair, expected):
    selected = plan(profile, pair)
    assert resolve_attention_backend(selected) == expected
    assert resolve_attention_backend(selected, "torch-flash") == "torch-flash"
    if expected == "torch-flash":
        with pytest.raises(ContractError, match="single-SM89 official Base16"):
            resolve_attention_backend(selected, "sol-sm89")


def test_unknown_backend_and_missing_dependency_are_actionable(monkeypatch):
    with pytest.raises(ContractError, match="unknown attention backend"):
        resolve_attention_backend(plan(), "mystery")
    monkeypatch.setattr("vflash.attention.find_spec", lambda _: None)
    assert resolve_attention_backend(plan()) == "sol-sm89"
    with pytest.raises(ContractError, match=r"python -m vflash\.install_sol"):
        require_attention_dependencies("sol-sm89")
    require_attention_dependencies("torch-flash")


@pytest.mark.parametrize(
    "requested,expected", [("auto", "sol-sm89"), ("torch-flash", "torch-flash")]
)
def test_complete_pipeline_forwards_resolved_backend_even_for_dense(
    tmp_path, monkeypatch, requested, expected
):
    from vflash.pipeline.assets import PreparedPipelineAssets
    from vflash.pipeline.contracts import PipelineAssets
    from vflash.pipeline.runtime import H3Pipeline

    prepared = PreparedPipelineAssets(
        PipelineAssets(**{name: tmp_path for name in PipelineAssets.__dataclass_fields__}),
        tmp_path / "receipt.json",
        "a" * 64,
        (),
        "i2va-base16-bf16-sm89",
    )
    monkeypatch.setattr("vflash.pipeline.runtime.media_executables", lambda: None)
    monkeypatch.setattr("vflash.pipeline.runtime.validate_adapter_dependencies", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda _: None, get_num_interop_threads=lambda: 1),
    )
    calls = []
    stage = SimpleNamespace(close=lambda: None)

    def core(_plan, **options):
        calls.append(options)
        return stage

    monkeypatch.setattr("vflash.pipeline.runtime.NativeEngineSession", core)
    monkeypatch.setattr("vflash.pipeline.runtime.DiffusersConditioner", lambda _: stage)
    monkeypatch.setattr("vflash.pipeline.runtime.OfficialMediaDecoder", lambda **_: stage)
    pipeline = H3Pipeline(
        prepared,
        device=NvidiaDevice(0, "gpu", "4090", 48, "8.9", 450),
        trust_local_code=True,
        attention_backend=requested,
    )
    assert pipeline.attention_backend == expected
    pipeline._load_stages(pipeline._plan)
    assert calls[0]["attention_backend"] == expected
    assert calls[0]["weight_residency"] == "block-ring"
    pipeline.close()


def test_native_plan_cli_reports_selection_without_loading_cuda(monkeypatch, capsys):
    import json

    from vflash.cli import main

    monkeypatch.setattr(
        "vflash.cli.discover_nvidia_devices",
        lambda: (NvidiaDevice(0, "gpu", "4090", 48, "8.9", 450),),
    )
    assert main(["plan", "i2va-base16-bf16-sm89", "--gpu", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["attention_selection"] == {
        "requested": "auto",
        "resolved": "sol-sm89",
        "exact": False,
        "executed": False,
    }


def test_installer_only_patches_the_pinned_stream_abi(tmp_path):
    from vflash.install_sol import patch_stream_abi

    path = tmp_path / "interface.py"
    path.write_text("                stream=stream,\n" * 4)
    patch_stream_abi(path)
    assert path.read_text() == "                stream,\n" * 4
    with pytest.raises(RuntimeError, match="four call sites"):
        patch_stream_abi(path)


def test_installer_extracts_only_the_pinned_package(tmp_path, monkeypatch):
    import io
    import tarfile

    from vflash import install_sol

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name in ("other/private.py", "techniques/sparse_backends/sol_attn/interface.py"):
            member = tarfile.TarInfo(f"Sana-{install_sol.REVISION}/{name}")
            member.size = 5
            archive.addfile(member, io.BytesIO(b"hello"))
    monkeypatch.setattr(install_sol, "urlopen", lambda *a, **kw: io.BytesIO(payload.getvalue()))
    install_sol.fetch_package(tmp_path)
    assert (tmp_path / "sol_attn/interface.py").read_text() == "hello"
    assert list(tmp_path.iterdir()) == [tmp_path / "sol_attn"]
