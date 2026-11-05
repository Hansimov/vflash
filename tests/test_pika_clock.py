import json

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.pika_flex import PIKA_BASE_REVISION, PIKA_RECIPE, PIKA_SOURCE_REVISION


@pytest.fixture
def clocks(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from vflash.adapters import pika_clock as module

    files, contents = [], {}
    for mode in ("t2va", "i2va"):
        times = module.pika_fl_times(mode)
        name = f"pika6-fl-{'t2va' if mode == 't2va' else 'keyframe'}.safetensors"
        (tmp_path / name).write_bytes(b"fixture")
        files.append(
            {
                "name": name,
                "bytes": 7,
                "plan": {"nfe": 6, "timesteps": [t.tolist() for t in times]},
            }
        )
        tensors = {}
        for step, t in enumerate(times):
            tensors[f"time-{step}"] = torch.zeros((), dtype=torch.float32).expand(len(t), 2688)
            for block in range(51):
                tensors[f"block-{block:03d}-step-{step:02d}"] = torch.zeros(
                    (), dtype=torch.bfloat16
                ).expand(len(t), (18 if block < 50 else 2) * 5376)
        contents[str(tmp_path / name)] = tensors
    proof = {
        "status": "completed",
        "recipe": PIKA_RECIPE,
        "source_revision": PIKA_SOURCE_REVISION,
        "base_revision": PIKA_BASE_REVISION,
        "files": files,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(proof))
    monkeypatch.setattr(module, "load_file", lambda path, **kwargs: dict(contents[path]))
    return module, tmp_path, proof, contents


def test_clock_covers_executed_six_steps_and_keyframe_roles(clocks):
    module, root, _, _ = clocks
    clock = module.PikaSamplingClock(root, device="cpu")
    for mode in ("t2va", "i2va", "fl2va", "l2va"):
        clock.select_mode(mode)
        rows = module.pika_fl_times(mode)
        assert len(rows[0]) == (1 if mode == "t2va" else 2)
        assert len(rows[-1]) == (2 if mode == "t2va" else 3)
        for step, times in enumerate(rows):
            clock.before(None, (), {"timestep": times})
            assert clock.step == step
            assert list(clock.time_embedding(None).shape) == [len(times), 2688]
            clock.after(None, (), None)
        assert clock.completed_forwards == 6
        with pytest.raises(ContractError, match="complete"):
            clock.before(None, (), {"timestep": rows[-1]})
    with pytest.raises(ContractError, match="FL partition"):
        clock.select_mode("ref2va")


def test_wrong_live_clock_and_legacy_euler_table_are_rejected(clocks):
    module, root, proof, _ = clocks
    clock = module.PikaSamplingClock(root, device="cpu")
    clock.select_mode("t2va")
    with pytest.raises(ContractError, match="Executed timestep"):
        clock.before(None, (), {"timestep": module.pika_fl_times("t2va")[1]})
    proof["files"][0]["plan"]["timesteps"][1] = [0.2, 0.2]
    (root / "manifest.json").write_text(json.dumps(proof))
    with pytest.raises(ContractError, match="six renoising"):
        module.PikaSamplingClock(root, device="cpu").select_mode("t2va")


def test_incomplete_or_wrong_precision_table_never_becomes_active(clocks):
    module, root, _, contents = clocks
    path = str(root / "pika6-fl-t2va.safetensors")
    item = contents[path].pop("time-0")
    clock = module.PikaSamplingClock(root, device="cpu")
    with pytest.raises(ContractError, match="coverage"):
        clock.select_mode("t2va")
    contents[path]["time-0"] = item.bfloat16()
    with pytest.raises(ContractError, match="FP32"):
        clock.select_mode("t2va")
    assert clock.plan is None and clock.tensors == {}


@pytest.mark.parametrize("reference_audio", [False, True])
def test_reference_clock_binds_partition_and_exact_clean_audio_time(clocks, reference_audio):
    import torch

    from vflash.pipeline.pika_flex import PIKA_ADAPTER_SHA256

    module, root, proof, contents = clocks
    proof.update(
        partition="ref2va",
        trunk="transformer_ref",
        adapter_sha256=PIKA_ADAPTER_SHA256["ref2va"],
        numerics_policy="fp32_conditioning_v1",
    )
    times = module.pika_times("ref2va", reference_audio=reference_audio)
    name = f"pika6-ref-{'audio' if reference_audio else 'visual'}.safetensors"
    (root / name).write_bytes(b"fixture")
    tensors = {}
    for step, t in enumerate(times):
        expected = module.pika_fl_times("i2va")[step]
        if reference_audio:
            expected = torch.unique(torch.cat([expected, torch.tensor([1.0])]))
        assert torch.equal(t, expected)
        tensors[f"time-{step}"] = torch.zeros((), dtype=torch.float32).expand(len(t), 2688)
        for block in range(51):
            tensors[f"block-{block:03d}-step-{step:02d}"] = torch.zeros(
                (), dtype=torch.bfloat16
            ).expand(len(t), (18 if block < 50 else 2) * 5376)
    contents[str(root / name)] = tensors
    proof["files"] = [
        {"name": name, "bytes": 7, "plan": {"nfe": 6, "timesteps": [t.tolist() for t in times]}}
    ]
    (root / "manifest.json").write_text(json.dumps(proof))
    with pytest.raises(ContractError, match="partition"):
        module.PikaSamplingClock(root, device="cpu")
    clock = module.PikaSamplingClock(root, device="cpu", partition="ref2va")
    clock.select_mode("ref2va", reference_audio=reference_audio)
    for t in times:
        clock.before(None, (), {"timestep": t})
        clock.after(None, (), None)
    assert clock.completed_forwards == 6
    with pytest.raises(ContractError, match="cannot be mixed"):
        clock.select_mode("i2va")
    assert clock.plan is None and clock.tensors == {}
    proof["adapter_sha256"] = PIKA_ADAPTER_SHA256["fl2va"]
    (root / "manifest.json").write_text(json.dumps(proof))
    with pytest.raises(ContractError, match="original base and adapter"):
        module.PikaSamplingClock(root, device="cpu", partition="ref2va")


def test_reference_clocks_cannot_add_audio_to_keyframe_or_accept_implicit_flags():
    from vflash.adapters.pika_clock import pika_times

    with pytest.raises(ContractError, match="Reference audio"):
        pika_times("i2va", reference_audio=True)
    with pytest.raises(ContractError, match="Reference audio"):
        pika_times("ref2va", reference_audio=1)
