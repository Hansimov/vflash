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


@pytest.mark.parametrize("mode", ["i2va", "fl2va"])
def test_original_lightx_v01_sol_is_explicit_without_changing_auto(mode):
    selected = plan(f"{mode}-turbo4-v01-544-exact-sm89")
    assert resolve_attention_backend(selected) == "torch-flash"
    assert resolve_attention_backend(selected, "sol-sm89") == "sol-sm89"
    changed = replace(selected, profile=replace(selected.profile, nfe=8))
    with pytest.raises(ContractError, match="Sol requires"):
        resolve_attention_backend(changed, "sol-sm89")


@pytest.mark.parametrize("profile", ["i2va-turbo4-exact-sm89", "i2va-turbo8-544-exact-sm89"])
def test_other_turbo_contracts_do_not_inherit_v01_sol_support(profile):
    with pytest.raises(ContractError, match="Sol requires"):
        resolve_attention_backend(plan(profile), "sol-sm89")


@pytest.mark.parametrize(
    "profile,requested,expected",
    [
        ("i2va-base16-bf16-sm89", "auto", "sol-sm89"),
        ("i2va-base16-bf16-sm89", "torch-flash", "torch-flash"),
        ("i2va-turbo4-v01-544-exact-sm89", "auto", "torch-flash"),
        ("i2va-turbo4-v01-544-exact-sm89", "sol-sm89", "sol-sm89"),
        ("fl2va-turbo4-v01-544-exact-sm89", "sol-sm89", "sol-sm89"),
    ],
)
def test_complete_pipeline_forwards_resolved_backend_even_for_dense(
    tmp_path, monkeypatch, profile, requested, expected
):
    from vflash.pipeline.assets import PreparedPipelineAssets
    from vflash.pipeline.contracts import PipelineAssets
    from vflash.pipeline.runtime import H3Pipeline

    prepared = PreparedPipelineAssets(
        PipelineAssets(**{name: tmp_path for name in PipelineAssets.__dataclass_fields__}),
        tmp_path / "receipt.json",
        "a" * 64,
        (),
        profile,
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


@pytest.mark.parametrize("defect", [None, "symlink", "missing", "oversized"])
def test_installer_checks_pinned_source_and_copies_only_package(tmp_path, monkeypatch, defect):
    import subprocess

    from vflash import install_sol

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    source = upstream / "techniques/sparse_backends"
    (source / "sol_attn").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[build-system]\n")
    if defect != "missing":
        (source / "sol_attn/interface.py").write_text("# source\n")
    (upstream / "unrelated.txt").write_text("not part of the installed package")
    if defect == "symlink":
        (source / "escape").symlink_to("../../unrelated.txt")
    if defect == "oversized":
        (source / "large.bin").write_bytes(b"x" * (256 * 1024 + 1))
    run = subprocess.run
    for args in (
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ],
    ):
        run(["git", "-C", str(upstream), *args], check=True, capture_output=True)
    revision = run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(install_sol, "REVISION", revision)

    def local_source(args, **kwargs):
        args = [
            str(upstream) if arg == "https://github.com/NVlabs/Sana.git" else arg
            for arg in args
        ]
        return run(args, **kwargs)

    monkeypatch.setattr(install_sol.subprocess, "run", local_source)
    output = tmp_path / "package"
    output.mkdir()
    if defect:
        with pytest.raises(RuntimeError):
            install_sol.fetch_package(output)
        assert not list(output.iterdir())
    else:
        install_sol.fetch_package(output)
        assert (output / "sol_attn/interface.py").read_text() == "# source\n"
        assert sorted(p.name for p in output.iterdir()) == ["pyproject.toml", "sol_attn"]


def test_veda_is_explicit_original_v01_only():
    selected = plan("i2va-turbo4-v01-544-exact-sm89")
    assert resolve_attention_backend(selected) == "torch-flash"
    assert resolve_attention_backend(selected, "veda-sm89") == "veda-sm89"
    for selected in (
        plan(),
        plan("ref2va-turbo4-exact-sm89"),
        plan("i2va-base16-bf16-sm86", True),
    ):
        with pytest.raises(ContractError, match=r"original LightX v0\.1"):
            resolve_attention_backend(selected, "veda-sm89")
