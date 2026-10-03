import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.contracts import VideoRequest
from vflash.pipeline.vdn import VDNAssets, VDNKeyframePipeline


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from vflash.pipeline import vdn

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    for name in ("base", "weights", "decoder"):
        (tmp_path / name).mkdir()
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"test-checkpoint")
    assets = VDNAssets(tmp_path / "base", tmp_path / "weights", tmp_path / "decoder", upscaler)
    events = []

    def encode(model, request, plan, directory):
        from PIL import Image

        for anchor in ("first", "last"):
            Image.new("RGB", (request.width, request.height)).save(
                directory / f"{anchor}-reference.png"
            )
        (directory / "target.pt").write_bytes(b"owned-clean-encoding")
        if plan["enabled"]:
            (directory / "first.pt").write_bytes(b"owned-first-pass-encoding")
        events.append(("encode", plan["enabled"]))
        return {"elapsed_seconds": 1.0}

    class Engine:
        def __init__(self, *args, **kwargs):
            events.append(("engine",))

        def sample(self, path, seed, **kwargs):
            two = kwargs["first_pass_conditioning"] is not None
            assert two == (kwargs["upscale"] is not None)
            for _ in range(10 if two else 8):
                kwargs["step_callback"](1.0)
            return torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 8, 3), {"sample_seconds": 1.0}

        def close(self):
            events.append(("engine_close",))

    class Decoder:
        def __init__(self, **kwargs):
            events.append(("decoder",))

        def resume_cuda(self):
            events.append(("resume",))

        def suspend_cuda(self):
            events.append(("suspend",))

        def generate_mp4(self, latent, path, **kwargs):
            assert latent.is_file()
            events.append(("media", kwargs))
            path.write_bytes(b"complete-test-mp4")
            return SimpleNamespace(media={"frames": 120}, stage_durations={"video_decode": 1.0})

        def close(self):
            events.append(("decoder_close",))

    def plan(canvas, enabled, task):
        return dict(enabled=enabled, total_steps=10 if enabled else 8)

    for name, fields in {
        "freevideo_engine.paths": {"add_vdn": lambda: None},
        "freevideo_engine.geometry": {
            "geometry": lambda w, h, frames: dict(width=w, height=h, frames=frames)
        },
        "freevideo_engine.two_pass": {"plan": plan},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(vdn, "encode_conditioning", encode)
    monkeypatch.setattr(vdn, "VDNEngineSession", Engine)
    from vflash.media import runtime

    monkeypatch.setattr(runtime, "OfficialMediaDecoder", Decoder)
    return assets, events, Decoder


@pytest.mark.parametrize("strategy", ["full8", "pixel8+2"])
def test_complete_stage_order_and_decoder_reuse(prepared, tmp_path, strategy):
    assets, events, _ = prepared
    pipeline = VDNKeyframePipeline(assets, strategy=strategy, trust_local_code=True)
    request = VideoRequest(
        "A subject moves.",
        first_frame=tmp_path / "first.png",
        width=512,
        height=512,
        audio_delivery_profile="silent-v1",
        keyframe_delivery_profile="exact-v1",
    )
    progress = []
    for index in range(2):
        output = tmp_path / f"{index}.mp4"
        result = pipeline.generate(request, output, progress=progress.append)
        assert output.read_bytes() == b"complete-test-mp4"
        assert result.request_mode == "i2va"
        assert result.profile_id == f"i2va-vdn-{strategy}-fp8-sm89"
        assert result.stages["media"]["video_decode"] == 1.0
    assert events.count(("decoder",)) == 1
    assert events.index(("engine_close",)) < events.index(("resume",))
    assert events.count(("suspend",)) == 2
    media = next(fields for name, *values in events if name == "media" for fields in values)
    assert media["audio_delivery_profile"] == "silent-v1"
    assert media["first_frame"].mode == "RGB" and media["first_frame"].size == (512, 512)
    assert media["last_frame"] is None
    assert not list(tmp_path.glob("vflash-vdn-*"))
    assert max(p.completed for p in progress if p.stage == "denoising") == (
        10 if strategy == "pixel8+2" else 8
    )
    pipeline.close()
    pipeline.close()
    assert events.count(("decoder_close",)) == 1
    with pytest.raises(ContractError, match="closed"):
        pipeline.generate(request, tmp_path / "closed.mp4")


def test_true_fl2_preserves_two_normalized_endpoints(prepared, tmp_path):
    assets, events, _ = prepared
    pipeline = VDNKeyframePipeline(assets, trust_local_code=True)
    request = VideoRequest(
        "A subject moves.",
        first_frame=tmp_path / "first.png",
        last_frame=tmp_path / "last.png",
        keyframe_delivery_profile="exact-v1",
    )
    result = pipeline.generate(request, tmp_path / "fl.mp4")
    media = next(e[1] for e in events if e[0] == "media")
    assert result.request_mode == "fl2va"
    assert media["last_frame"].mode == "RGB" and media["last_frame"].size == (928, 512)
    pipeline.close()


def test_scoped_pipeline_cache_reuses_only_encoding_and_clears_on_close(prepared, tmp_path):
    from dataclasses import replace

    from PIL import Image

    from vflash.pipeline.contracts import ConditioningReuseScope

    assets, events, _ = prepared
    reference = tmp_path / "input.png"
    Image.new("RGB", (512, 512), "red").save(reference)
    pipeline = VDNKeyframePipeline(assets, trust_local_code=True)
    request = VideoRequest("Motion.", first_frame=reference, width=512, height=512)
    scope = ConditioningReuseScope("accepted-request")
    for i in range(2):
        result = pipeline.generate(
            replace(request, seed=i),
            tmp_path / f"scoped-{i}.mp4",
            conditioning_reuse_scope=scope,
        )
        assert result.stages["encoding"]["conditioning_cache_hit"] == bool(i)
    assert events.count(("encode", False)) == 1
    assert events.count(("engine",)) == 2
    assert pipeline._conditioning_cache.retained_bytes > 0
    pipeline.close()
    assert pipeline._conditioning_cache.retained_bytes == 0


def test_failure_never_publishes_partial_file(prepared, tmp_path, monkeypatch):
    assets, events, decoder = prepared

    def fail(self, latent, output, **kwargs):
        output.write_bytes(b"partial")
        raise RuntimeError("decode interrupted")

    monkeypatch.setattr(decoder, "generate_mp4", fail)
    pipeline = VDNKeyframePipeline(assets, trust_local_code=True)
    output = tmp_path / "result.mp4"
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.generate(VideoRequest("Motion.", first_frame=tmp_path / "first.png"), output)
    assert not output.exists() and not list(tmp_path.glob("vflash-vdn-*"))
    assert ("suspend",) in events
    pipeline.close()


def test_scoped_failure_discards_cached_inputs(prepared, tmp_path, monkeypatch):
    from PIL import Image

    from vflash.pipeline.contracts import ConditioningReuseScope

    assets, _, decoder = prepared
    reference = tmp_path / "input.png"
    Image.new("RGB", (512, 512)).save(reference)
    pipeline = VDNKeyframePipeline(assets, trust_local_code=True)

    def fail(*args, **kwargs):
        raise RuntimeError("cancelled")

    monkeypatch.setattr(decoder, "generate_mp4", fail)
    with pytest.raises(RuntimeError, match="cancelled"):
        pipeline.generate(
            VideoRequest("Motion.", first_frame=reference, width=512, height=512),
            tmp_path / "cancelled.mp4",
            conditioning_reuse_scope=ConditioningReuseScope("accepted-request"),
        )
    assert pipeline._conditioning_cache.retained_bytes == 0
    pipeline.close()


def test_invalid_inputs_do_not_start_model(prepared, tmp_path):
    assets, events, _ = prepared
    with pytest.raises(ContractError, match="trust_local_code"):
        VDNKeyframePipeline(assets)
    pipeline = VDNKeyframePipeline(assets, trust_local_code=True)
    with pytest.raises(ContractError, match="I2VA"):
        pipeline.generate(VideoRequest("Motion."), tmp_path / "video.mp4")
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"original")
    with pytest.raises(ContractError, match="new local"):
        pipeline.generate(VideoRequest("Motion.", first_frame=Path("first.png")), output)
    assert output.read_bytes() == b"original" and not events
    pipeline.close()


def test_selflift_owns_roundtrip_and_eight_steps(prepared, tmp_path, monkeypatch):
    import torch

    from vflash.adapters import vdn_selflift

    assets, events, _ = prepared

    def sample(weights, model, decoder, canvas, target, first, seed, *, step_callback):
        assert target.name == "target.pt" and first.name == "first.pt"
        assert canvas["width"] == 640
        decoder.resume_cuda()
        decoder.suspend_cuda()
        for _ in range(8):
            step_callback(1.0)
        return torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 8, 3), {"nfe": 8}

    monkeypatch.setattr(vdn_selflift, "sample_selflift", sample)
    assets = VDNAssets(assets.official_model, assets.weights, assets.decoder)
    pipeline = VDNKeyframePipeline(assets, strategy="selflift6+2", trust_local_code=True)
    progress = []
    result = pipeline.generate(
        VideoRequest(
            "A subject moves.", first_frame=tmp_path / "first.png", width=640, height=640
        ),
        tmp_path / "progressive.mp4",
        progress=progress.append,
    )
    assert result.stages["denoising"]["nfe"] == 8
    assert events.count(("decoder",)) == 1 and ("engine",) not in events
    assert ("encode", True) in events
    assert [(p.completed, p.total) for p in progress if p.stage == "denoising"] == [
        (i, 8) for i in range(9)
    ]
    pipeline.close()


@pytest.mark.parametrize(
    "changes", [{"width": 512}, {"duration_seconds": 8}, {"last_frame": Path("last.png")}]
)
def test_selflift_rejects_unqualified_modes_before_loading(prepared, tmp_path, changes):
    assets, events, _ = prepared
    pipeline = VDNKeyframePipeline(assets, strategy="selflift6+2", trust_local_code=True)
    args = dict(first_frame=tmp_path / "first.png", width=640, height=640)
    args.update(changes)
    with pytest.raises(ContractError, match="five-second I2VA"):
        pipeline.generate(VideoRequest("Motion.", **args), tmp_path / "unqualified.mp4")
    assert not events
    pipeline.close()
