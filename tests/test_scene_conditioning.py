import threading
from dataclasses import replace
from pathlib import Path

import pytest
from test_conditioning_reuse import _fixture, _run, _State

from vflash.adapters.scene_conditioning import SourceSceneConditioning
from vflash.contracts import ContractError
from vflash.pipeline.contracts import ConditioningReuseScope, VideoRequest
from vflash.pipeline.runtime import H3Pipeline


def _repaint(request):
    return replace(
        request,
        first_frame=Path("restored.png"),
        generate_audio=False,
        audio_delivery_profile="unchanged",
        keyframe_delivery_profile="decoded",
    )


@pytest.mark.parametrize("source_hit", [False, True])
def test_source_capture_composes_with_exact_cache_and_preserves_vae(monkeypatch, source_hit):
    torch = pytest.importorskip("torch")
    reuse, components, reference, request, calls = _fixture(monkeypatch)
    exact = ConditioningReuseScope("one-owner")
    if source_hit:
        _run(reuse, components, reference, request, exact)
    owner = SourceSceneConditioning(request)
    blocks = reuse.conditioner.pipe._blocks.sub_blocks
    originals = dict(blocks)
    source_state = _State()
    owner.check_request(request)
    with reuse.capture(exact, request, (reference,)), owner.capture(reuse.conditioner):
        blocks["text_encoder"](components, source_state)
        blocks["vae_encoder"](components, source_state)
    owner.request_complete()
    assert blocks == originals and calls == {"text_encoder": 1, "vae_encoder": 1}
    saved = source_state.get("prompt_embeds").clone()
    source_state.get("prompt_embeds").zero_()
    with owner.repaint():
        for _ in range(2):
            owner.check_request(_repaint(request))
            state = _State()
            with owner.capture(reuse.conditioner):
                blocks["text_encoder"](components, state)
                blocks["vae_encoder"](components, state)
            assert torch.equal(state.get("prompt_embeds"), saved)
            state.get("prompt_embeds").zero_()
            owner.request_complete()
    assert calls == {"text_encoder": 1, "vae_encoder": 3}
    assert blocks == originals and owner.report["repaint_encoder_calls_elided"] == 2
    owner.close()
    assert owner._features is None
    with pytest.raises(ContractError, match="ended"):
        owner.check_request(request)


def test_phase_request_thread_and_failure_ownership(monkeypatch):
    reuse, components, _, request, _ = _fixture(monkeypatch)
    owner = SourceSceneConditioning(request)
    with pytest.raises(ContractError, match="completed source"), owner.repaint():
        pass
    with pytest.raises(ContractError, match="declared source"):
        owner.check_request(replace(request, prompt="another request"))
    blocks = reuse.conditioner.pipe._blocks.sub_blocks
    original = blocks["text_encoder"]
    with pytest.raises(RuntimeError, match="cancelled"), owner.capture(reuse.conditioner):
        blocks["text_encoder"](components, _State())
        raise RuntimeError("cancelled")
    assert blocks["text_encoder"] is original
    owner.request_complete()
    with pytest.raises(ContractError, match="enter repaint"):
        owner.check_request(request)
    with owner.repaint():
        with pytest.raises(ContractError, match="exact cache"):
            owner.check_request(_repaint(request), reuse_scope=ConditioningReuseScope("scope"))
        for changed in (
            request,
            replace(_repaint(request), prompt="another request"),
            replace(_repaint(request), seed=99),
        ):
            with pytest.raises(ContractError, match="related"):
                owner.check_request(changed)
        with pytest.raises(ContractError, match="completed source"), owner.repaint():
            pass
    errors = []

    def foreign():
        try:
            owner.check_request(request)
        except ContractError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join()
    assert errors and "another thread" in errors[0]
    owner.close()


def test_pipeline_context_restores_after_failure_and_rejects_nesting():
    pipeline = H3Pipeline.__new__(H3Pipeline)
    pipeline._lock = threading.Lock()
    pipeline._closed = False
    request = VideoRequest("source", first_frame=Path("source.png"))
    with (
        pytest.raises(RuntimeError, match="cancelled"),
        pipeline.source_scene_conditioning(request) as owner,
    ):
        with (
            pytest.raises(ContractError, match="unscoped"),
            pipeline.source_scene_conditioning(request),
        ):
            pass
        raise RuntimeError("cancelled")
    assert pipeline._scene_conditioning is None and owner._features is None
    with pipeline.source_scene_conditioning(request):
        pass
    with (
        pytest.raises(ContractError, match="I2VA"),
        pipeline.source_scene_conditioning(VideoRequest("text only")),
    ):
        pass
    assert pipeline._scene_conditioning is None


def test_nonfinite_features_do_not_enter_owner(monkeypatch):
    torch = pytest.importorskip("torch")
    _reuse, components, _, request, _ = _fixture(monkeypatch)
    owner = SourceSceneConditioning(request)

    def block(components, state):
        state.set("prompt_embeds", torch.full((1, 1, 5120), float("nan")))
        state.set("text_token_tags", torch.tensor([0]))
        return components, state

    with pytest.raises(ContractError, match="finite"):
        owner._call(block, components, _State())
    assert owner._features is None


@pytest.mark.parametrize("fail", [False, True])
def test_real_pipeline_routes_scope_and_releases_features_on_failure(
    tmp_path, monkeypatch, fail
):
    from types import SimpleNamespace

    from test_complete_pipeline import _pipeline

    reuse, components, _reference, request, _calls = _fixture(monkeypatch)
    pipeline, _events = _pipeline(fail="native:generate" if fail else None)
    monkeypatch.setattr("vflash.pipeline.runtime.validate_request", lambda *a, **kw: None)
    monkeypatch.setattr(
        "vflash.pipeline.runtime.read_reference",
        lambda path: SimpleNamespace(close=lambda: None),
    )
    capture = pipeline._conditioner.capture

    def owned_capture(request, references, directory, *, scene_conditioning):
        state = _State()
        with scene_conditioning.capture(reuse.conditioner):
            reuse.conditioner.pipe._blocks.sub_blocks["text_encoder"](components, state)
        return capture(request, references, directory)

    pipeline._conditioner.capture = owned_capture
    with pipeline.source_scene_conditioning(request) as owner:
        if fail:
            with pytest.raises(RuntimeError, match="native:generate"):
                pipeline.generate(request, tmp_path / "output.mp4")
            assert owner._features is None and pipeline._closed
        else:
            pipeline.generate(request, tmp_path / "output.mp4")
            with owner.repaint():
                owner.check_request(_repaint(request))
            assert owner.report["source_captures"] == 1
    assert owner._features is None and pipeline._scene_conditioning is None


@pytest.mark.parametrize("memory", [False, True])
def test_adapter_keeps_fixed_source_schema_and_reports_scene_separately(
    tmp_path, monkeypatch, memory
):
    from types import SimpleNamespace

    from vflash.adapters.conditioning_capture import H3ConditioningCaptureComplete
    from vflash.adapters.diffusers_h3 import DiffusersConditioner
    from vflash.model_assets import model_profile
    from vflash.native.h3_conditioning_bundle import _validate_source
    from vflash.pipeline.assets import conditioning_source

    reuse, components, reference, request, _calls = _fixture(monkeypatch)
    reference.size_bytes = 128
    owner = DiffusersConditioner.__new__(DiffusersConditioner)
    owner.pipe = reuse.conditioner.pipe
    owner.profile = model_profile("i2va-base16-bf16-sm89")
    owner.prepared = reuse.conditioner.prepared
    owner.versions = reuse.conditioner.versions
    owner._closed, owner._cuda_active = False, True
    owner.device, owner.transformer = "cpu", object()
    owner._torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: None))
    owner.pipe.scheduler = owner.pipe.audio_scheduler = SimpleNamespace(sigmas=list(range(17)))

    def invoke(_request, _references):
        owner.pipe._blocks.sub_blocks["text_encoder"](components, _State())
        raise H3ConditioningCaptureComplete

    owner._invoke = invoke

    class Capture:
        def __init__(self, _directory):
            pass

        def install(self, _transformer):
            pass

        def close(self):
            pass

        def discard(self):
            pass

        def prefix_counts(self):
            return 1, 0

        def reference_token_budget(self, **_kwargs):
            return 1

        def finish(self, **kwargs):
            assert _validate_source(kwargs["source"]) == conditioning_source(
                profile_id=owner.prepared.profile_id,
                request_mode=request.mode,
                runtime_versions=owner.versions,
                reference_policy="match",
            )
            return kwargs

        finish_in_memory = finish

    monkeypatch.setattr("vflash.adapters.diffusers_h3.H3ConditioningCaptureSession", Capture)
    scene = SourceSceneConditioning(request)
    capture = owner.capture_in_memory if memory else owner.capture
    result = capture(request, (reference,), tmp_path / "conditioning", scene_conditioning=scene)
    assert "scene_conditioning" not in result["source"]
    assert owner.last_capture_diagnostics["scene_conditioning"]["source_captures"] == 1
    assert owner.last_capture_diagnostics["scene_conditioning"]["equivalent"] is False
    scene.close()
