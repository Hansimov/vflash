"""Architecture and visible-partition contracts, without starting CUDA."""

import json
import sys
from types import SimpleNamespace

import pytest

from vflash.contracts import ContractError


@pytest.mark.parametrize(
    "arch,cap,memory",
    [("sm90", "9.0", 80), ("sm103", "10.3", 34), ("sm120", "12.0", 32), ("sm120", "12.0", 96)],
)
def test_real_architecture_plan_keeps_original_v01_adapter(arch, cap, memory):
    from vflash.attention import resolve_attention_backend
    from vflash.catalog import ProfileCatalog
    from vflash.hardware import NvidiaDevice
    from vflash.model_assets import model_profile
    from vflash.native.h3_kernel_plan import resolve_h3_kernel_plan
    from vflash.planner import resolve_plan

    profile = "i2va-turbo4-v01-544-exact-" + arch
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id=profile,
        device=NvidiaDevice(0, "GPU-fixture", "Test", memory, cap, 0),
    )
    assert plan.target.compute_capability == cap
    assert model_profile(profile).weight_profile == "lightx-turbo4-v0.1-544"
    assert resolve_attention_backend(plan, "veda-triton") == "veda-triton"
    with pytest.raises(ContractError, match="SM89"):
        resolve_attention_backend(plan, "veda-sm89")
    kernel = resolve_h3_kernel_plan(arch)
    assert (
        kernel.compute_capability == arch
        and "qualification required" in kernel.quality_contract
    )


def test_mig_uses_cuda_visible_memory_and_partition_identity(monkeypatch):
    from vflash import cuda_visible

    parent = "11111111-1111-1111-1111-111111111111"
    child = "22222222-2222-2222-2222-222222222222"
    outputs = [
        f"GPU 0: Parent (UUID: GPU-{parent})\n  MIG Device (UUID: MIG-{child})",
        json.dumps(
            [
                dict(
                    index=0,
                    uuid="GPU-" + child,
                    name="MIG 1g.34gb",
                    memory_gib=33.75,
                    compute_capability="10.3",
                )
            ]
        ),
    ]
    monkeypatch.setattr(
        cuda_visible.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=outputs.pop(0))
    )
    (device,) = cuda_visible.visible_devices()
    assert device.uuid == "MIG-" + child and device.memory_gib == 33.75
    assert device.power_limit_watts == 0  # unavailable, never claim parent power


def test_cuda_identity_mismatch_is_not_admitted(monkeypatch):
    from vflash import cuda_visible
    from vflash.contracts import ContractError

    outputs = [
        "GPU 0: no matching identity",
        json.dumps(
            [
                dict(
                    index=0,
                    uuid="GPU-unknown",
                    name="Test",
                    memory_gib=80,
                    compute_capability="9.0",
                )
            ]
        ),
    ]
    monkeypatch.setattr(
        cuda_visible.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=outputs.pop(0))
    )
    with pytest.raises(ContractError, match="discovery failed"):
        cuda_visible.visible_devices()


@pytest.mark.parametrize("arch", ["sm90", "sm103", "sm120"])
def test_shared_hybrid_reference_constructor_accepts_remote_original_v01(
    monkeypatch, arch, tmp_path
):
    from vflash.adapters import hybrid_reference, modular_config, references

    names = (
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
    )

    class Pipeline:
        component_names = names

        def __init__(self, **kwargs):
            pass

        def update_components(self, **kwargs):
            self.__dict__.update(kwargs)

        def set_progress_bar_config(self, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(MiniMaxH3ModularPipeline=Pipeline)
    )
    monkeypatch.setattr(modular_config, "local_modular_config", lambda *a, **kw: {})
    monkeypatch.setattr(references, "install_match_reference_setup_block", lambda pipe: None)
    owner = SimpleNamespace(
        prepared=SimpleNamespace(
            profile_id="i2va-turbo4-v01-544-exact-" + arch,
            assets=SimpleNamespace(model_directory=tmp_path),
        ),
        transformer=object(),
        pipe=SimpleNamespace(**{name: object() for name in names}),
        versions={},
    )
    graph = hybrid_reference.HybridReferenceGraph(
        owner, {"reference_checkpoint_revision": "42ed227ee7df40d41602854ae760620d6eb651fe"}
    )
    assert graph.pipe.transformer_ref is owner.transformer
    assert all(getattr(graph.pipe, name) is getattr(owner.pipe, name) for name in names)
    owner.prepared.profile_id = "i2va-base16-bf16-sm89"
    with pytest.raises(ValueError, match="original FL"):
        hybrid_reference.HybridReferenceGraph(owner, {})
