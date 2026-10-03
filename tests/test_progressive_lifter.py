import sys
from types import ModuleType

import pytest

from vflash.contracts import ContractError

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("bad", [None, "shape", "nonfinite"])
def test_learned_lifter_keeps_original_clock_without_pixel_roundtrip(
    monkeypatch, tmp_path, bad
):
    from vflash.adapters import vdn_selflift as core

    canvas = dict(width=640, height=640, frames=124)
    low = torch.ones(1, 24, 2, 20, 20)
    sound = torch.ones(1, 8, 4)
    calls = []
    for name, fields in {
        "freevideo_engine.geometry": {"geometry": lambda **kwargs: kwargs},
        "freevideo_engine.two_pass": {
            "plan": lambda *a, **k: dict(
                first=dict(width=320, height=320, frames=124), upscale_target=canvas
            ),
            "crop_latents": lambda value, plan: value,
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)

    class Engine:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            calls.append("close")

    def stage(session, path, seed, state, *, step_callback):
        calls.append((state["phase"], seed))
        if state["phase"] == "prefix":
            return low, sound, dict(sample_seconds=6, nfe=6)
        assert state["video"].shape == (1, 24, 2, 40, 40)
        assert torch.equal(state["video"], torch.full((1, 24, 2, 40, 40), 2.0))
        assert state["audio"] is sound
        state["clock"] = dict(prefix_nfe=6, suffix_nfe=2)
        return state["video"], sound, dict(sample_seconds=2, nfe=2)

    def upscale(value, width, height):
        assert value is low and (width, height) == (640, 640)
        result = torch.full((1, 24, 2, 40, 40), 2.0)
        if bad == "shape":
            result = result[:, :, :1]
        elif bad == "nonfinite":
            result.fill_(float("nan"))
        return result, {"precision": "bf16"}

    def forbidden(*args, **kwargs):
        pytest.fail("Learned rho-zero lifting must not run the pixel VAE")

    monkeypatch.setattr(core, "VDNEngineSession", Engine)
    monkeypatch.setattr(core, "sample_stage", stage)
    monkeypatch.setattr(core, "pixel_vae_anchor", forbidden)
    args = (tmp_path, tmp_path, None, canvas, tmp_path / "target", tmp_path / "first", 17)
    if bad:
        with pytest.raises(ContractError, match="geometry or values"):
            core.sample_selflift(*args, upscale=upscale)
        assert calls == [("prefix", 17), "close"]
    else:
        _, audio, report = core.sample_selflift(*args, upscale=upscale)
        assert audio is sound and report["nfe"] == 8 and report["rho"] == 0
        assert report["strategy"] == "learned6+2"
        assert not report["pixel_vae_roundtrip"]
        assert report["clock"] == dict(prefix_nfe=6, suffix_nfe=2)
        assert calls == [("prefix", 17), "close", ("suffix", 10017), "close"]
