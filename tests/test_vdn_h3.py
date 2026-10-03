import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from vflash.adapters.vdn_h3 import (
    VDNEngineSession,
    first_pass_reference,
    sample_encoded_first_pass,
    sample_latents,
)
from vflash.contracts import ContractError


def test_async_staging_stays_writable():
    torch = pytest.importorskip("torch")

    class Engine:
        def sample(self, *args, **kwargs):
            assert not torch.is_grad_enabled()
            buffer = torch.zeros(4)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(buffer.copy_, torch.ones(4)).result()
            return buffer

    assert sample_latents(Engine(), Path("input.pt"), 1).tolist() == [1] * 4


def test_pixel_canvas_is_restored_after_failure():
    pytest.importorskip("torch")
    target = dict(width=1280, height=704, frames=243)
    first = dict(width=640, height=352, frames=243)

    class Engine:
        canvas = target

        def sample(self, *args, **kwargs):
            assert self.canvas is first
            raise RuntimeError("interrupted")

    engine = Engine()
    with pytest.raises(RuntimeError, match="interrupted"):
        sample_encoded_first_pass(engine, Path("first.pt"), 0, first)
    assert engine.canvas is target
    with pytest.raises(ContractError, match="admitted target"):
        sample_encoded_first_pass(engine, Path("first.pt"), 0, dict(first, width=1281))
    assert engine.canvas is target


def test_first_reference_preserves_alignment_without_black_edges():
    pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (992, 544), (71, 89, 123))
    plan = dict(
        enabled=True,
        first=dict(width=512, height=288),
        crop=dict(left=16, top=16, width=992, height=544),
        upscale_target=dict(width=1024, height=576),
    )
    output = first_pass_reference(image, plan)
    assert output.size == (512, 288)
    assert output.getextrema() == ((71, 71), (89, 89), (123, 123))
    assert image.size == (992, 544)
    with pytest.raises(ContractError, match="target canvas"):
        first_pass_reference(Image.new("RGB", (1280, 720)), plan)


@pytest.fixture
def backend(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    first = dict(width=640, height=352, frames=243)
    target = dict(width=1280, height=704, frames=243)
    calls = []

    class Engine:
        def __init__(self, cache, **options):
            self.canvas = options["canvas"]
            calls.append(("init", options))

        def sample(self, conditioning, seed, **options):
            calls.append((str(conditioning), seed, dict(self.canvas), options))
            return "video", "audio", {"sample_seconds": 3}

        def close(self):
            calls.append(("close",))

    def plan(canvas, enabled, task):
        assert task == "fl2va"
        return dict(
            enabled=enabled,
            first=first if enabled else target,
            second=target,
            upscale_target=target,
            restart_seed_offset=1,
            refine_steps=2,
            upscaler_sha256="not-the-callers-weight",
        )

    modules = {
        "freevideo_engine.paths": {"add_vdn": lambda: None},
        "freevideo_engine.geometry": {"geometry": lambda **kw: kw},
        "freevideo_engine.runtime": {"Engine": Engine},
        "freevideo_engine.two_pass": {"plan": plan, "crop_latents": lambda v, p: v},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    return VDNEngineSession(Path("local-weights"), task="fl2va", canvas=target), calls


def test_two_pass_uses_separate_encoded_inputs_and_schedule(backend):
    session, calls = backend
    stages = []
    video, audio, report = session.sample(
        Path("target.pt"),
        19,
        first_pass_conditioning=Path("first.pt"),
        upscale=lambda v, w, h: (v, {"precision": "bf16"}),
        stage_callback=lambda *args: stages.append(args),
    )
    assert (video, audio) == ("video", "audio")
    assert calls[1][0:2] == ("first.pt", 19)
    assert calls[1][2]["width"] == 640
    assert "allow_smaller_canvas" not in calls[1][3]
    assert calls[2][0:2] == ("target.pt", 20)
    assert calls[2][2]["width"] == 1280
    assert calls[2][3]["refine_steps"] == 2
    assert calls[2][3]["initial_latents"] == ("video", "audio")
    assert len(stages) == 1 and report["sample_seconds"] == 6
    assert "upscaler_sha256" not in report["sampling_plan"]
    assert report["first_pass_conditioning"] == "pixel-encode"
    session.close()
    session.close()
    assert calls.count(("close",)) == 1


def test_full8_has_no_hidden_refinement(backend):
    session, calls = backend
    _, _, report = session.sample(Path("target.pt"), 23)
    assert len(calls) == 2
    assert calls[1][2]["width"] == 1280
    assert "refine_steps" not in calls[1][3]
    assert report["first_pass_conditioning"] == "target-encode"
    session.close()


def test_missing_first_encoding_never_falls_back(backend):
    session, calls = backend
    with pytest.raises(ContractError, match="pixel-encoded"):
        session.sample(Path("target.pt"), 1, upscale=lambda *args: None)
    assert len(calls) == 1
    session.close()


def test_concurrent_and_closed_calls_rejected(backend):
    session, _ = backend
    session._lock.acquire()
    try:
        with pytest.raises(ContractError, match="busy"):
            session.sample(Path("target.pt"), 1)
        with pytest.raises(ContractError, match="busy"):
            session.close()
    finally:
        session._lock.release()
    session.close()
    with pytest.raises(ContractError, match="closed"):
        session.sample(Path("target.pt"), 1)
