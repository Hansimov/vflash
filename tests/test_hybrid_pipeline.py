from pathlib import Path
from types import SimpleNamespace

import pytest
from test_complete_pipeline import _pipeline

from vflash.contracts import ContractError
from vflash.native.h3_hybrid import HybridModel
from vflash.pipeline import VideoRequest


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile_id", "i2va-turbo4-exact-sm89"),
        ("capability", "8.6"),
        ("strategy", "sequence-head"),
        ("attention_backend", "sol-sm89"),
    ],
)
def test_variant_rejects_other_models_and_unqualified_topologies(tmp_path, field, value):
    options = dict(
        profile_id="i2va-turbo4-v01-544-exact-sm89",
        capability="8.9",
        strategy="single",
        attention_backend="torch-flash",
    )
    options[field] = value
    with pytest.raises(ContractError, match=r"original FL v0\.1"):
        HybridModel(tmp_path).validate(**options)


def test_changed_checkpoint_fails_before_loading_or_retiring_session(tmp_path):
    pipeline, events = _pipeline()
    pipeline.hybrid_model = SimpleNamespace(stamps=lambda: ("new",))
    pipeline._hybrid_stamps = ("old",)
    with pytest.raises(ContractError, match="checkpoint changed"):
        pipeline.prepare()
    assert not events and not pipeline._closed


def test_ref_keyframe_ref_share_owner_and_preserve_real_tasks(tmp_path, monkeypatch):
    from vflash.model_assets import model_profile
    from vflash.pipeline import runtime

    pipeline, events = _pipeline()
    pipeline.profile = model_profile("i2va-turbo4-v01-544-exact-sm89")
    pipeline.hybrid_model = SimpleNamespace(stamps=lambda: ())
    pipeline._hybrid_stamps = ()
    pipeline._core.runtime = SimpleNamespace(
        metadata=lambda: {"model_variant": {"hybrid": True}}
    )
    references = []
    monkeypatch.setattr(
        runtime, "read_reference", lambda p: SimpleNamespace(close=lambda: references.append(p))
    )

    class Bundle:
        pass

    monkeypatch.setattr(runtime, "H3InMemoryConditioning", Bundle)
    tasks = []
    original = pipeline._conditioner.capture

    def capture(request, images, directory):
        tasks.append(request.mode)
        value = original(request, images, directory)
        if request.mode == "ref2va":
            bundle = Bundle()
            bundle.__dict__.update(value.__dict__)
            return bundle
        return value

    pipeline._conditioner.capture = capture
    pipeline._reference_graph = SimpleNamespace(
        capture=capture, close=lambda: events.append("reference-graph:close")
    )
    handoffs = []
    native = pipeline._core.generate

    def generate(bundle, output, **kw):
        handoffs.append(isinstance(bundle, Bundle))
        return native(bundle, output, **kw)

    pipeline._core.generate = generate
    requests = (
        VideoRequest("Reference action", references=(tmp_path / "one.png",)),
        VideoRequest("Keyframe action", first_frame=tmp_path / "start.png"),
        VideoRequest("Reference action again", references=(tmp_path / "two.png",)),
    )
    results = [
        pipeline.generate(request, tmp_path / f"{i}.mp4") for i, request in enumerate(requests)
    ]
    assert tasks == ["ref2va", "i2va", "ref2va"]
    assert handoffs == [True, False, True]
    assert [r.request_mode for r in results] == tasks
    assert all(r.stages["model_variant"] == {"hybrid": True} for r in results)
    assert len(references) == 3
    pipeline.close()
    assert events[-4:] == [
        "reference-graph:close",
        "conditioning:close",
        "media:close",
        "native:close",
    ]


def test_reference_graph_failure_retires_shared_owners(tmp_path, monkeypatch):
    from vflash.pipeline import runtime

    pipeline, events = _pipeline()
    closed = []
    monkeypatch.setattr(
        runtime, "read_reference", lambda p: SimpleNamespace(close=lambda: closed.append(p))
    )

    def fail(*args, **kwargs):
        raise RuntimeError("reference capture failed")

    pipeline._reference_graph = SimpleNamespace(
        capture=fail, close=lambda: events.append("reference-graph:close")
    )
    request = VideoRequest("Reference action", reference=Path("ref.png"))
    with pytest.raises(RuntimeError, match="reference capture failed"):
        pipeline.generate(request, tmp_path / "output.mp4")
    assert pipeline._closed and closed == [Path("ref.png")]
    assert not (tmp_path / "output.mp4").exists()
    assert events[-4:] == [
        "reference-graph:close",
        "conditioning:close",
        "media:close",
        "native:close",
    ]
