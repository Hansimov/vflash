from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.native.h3_conditioning_bundle import H3ConditioningProfile
from vflash.pipeline.contracts import VideoRequest
from vflash.pipeline.runtime import H3Pipeline


def _pipeline(*, fail: str | None = None) -> tuple[H3Pipeline, list[str]]:
    events: list[str] = []

    class Stage:
        def __init__(self, name: str) -> None:
            self.name = name

        def _event(self, action: str) -> None:
            event = f"{self.name}:{action}"
            events.append(event)
            if fail == event:
                raise RuntimeError(event)

        def close(self) -> None:
            self._event("close")

        def resume_cuda(self) -> float:
            self._event("resume")
            return 0.0

        def suspend_cuda(self) -> float:
            self._event("suspend")
            return 0.0

        def capture(self, request, reference, directory):
            self._event("capture")
            directory.mkdir()
            return SimpleNamespace(
                directory=directory,
                bundle_id="h3-conditioning-test",
                source={},
                profile=H3ConditioningProfile(
                    "ref2va",
                    request.width,
                    request.height,
                    124,
                    4,
                    12,
                    3,
                    2,
                    2,
                    0,
                ),
            )

        def generate(self, bundle, output, *, progress_callback):
            self._event("generate")
            output.write_bytes(b"owned latent test placeholder")
            for nfe in range(1, 5):
                progress_callback(nfe, 4)
            return {"generation": {"elapsed_seconds": 0.5, "output_path": str(output)}}

        def generate_mp4(self, latent, output, **kwargs):
            self._event("generate")
            output.write_bytes(b"complete media test placeholder")
            return SimpleNamespace(stage_durations={}, peak_allocated_bytes=4, media=kwargs)

    pipeline = H3Pipeline.__new__(H3Pipeline)
    pipeline._lock = threading.Lock()
    pipeline._closed = pipeline._released = False
    pipeline._core, pipeline._conditioner, pipeline._media = (
        Stage("native"),
        Stage("conditioning"),
        Stage("media"),
    )
    pipeline.prepared = SimpleNamespace(
        check_unchanged=lambda: None, profile_id="ref2va-turbo4-exact-sm89"
    )
    from vflash.model_assets import model_profile

    pipeline.profile = model_profile()
    pipeline.initialization_seconds = 1.0
    pipeline.request_count = 0
    return pipeline, events


@pytest.fixture
def video_request(tmp_path: Path, monkeypatch) -> VideoRequest:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference fixture")
    monkeypatch.setattr(
        "vflash.pipeline.runtime.read_reference",
        lambda path: SimpleNamespace(image=object(), close=lambda: None),
    )
    return VideoRequest("A slow camera move around <Picture 1>.", reference)


def test_two_requests_reuse_the_session_and_retire_each_stage(video_request, tmp_path):
    pipeline, events = _pipeline()
    progress = []
    first = pipeline.generate(video_request, tmp_path / "first.mp4", progress=progress.append)
    second = pipeline.generate(video_request, tmp_path / "second.mp4")
    assert first.stages["request_index"] == 1
    assert second.stages["request_index"] == 2
    assert first.output_path.is_file() and second.output_path.is_file()
    assert (
        events
        == [
            "conditioning:resume",
            "conditioning:capture",
            "conditioning:suspend",
            "native:generate",
            "media:resume",
            "media:generate",
            "media:suspend",
        ]
        * 2
    )
    assert [(row.stage, row.completed) for row in progress] == [
        ("encoding", 0),
        ("encoding", 1),
        ("denoising", 0),
        ("denoising", 1),
        ("denoising", 2),
        ("denoising", 3),
        ("denoising", 4),
        ("decoding", 0),
        ("decoding", 1),
    ]
    assert not list(tmp_path.glob(".vflash-*"))
    pipeline.close()
    pipeline.close()
    assert events[-3:] == ["conditioning:close", "media:close", "native:close"]


def test_total_time_covers_input_loading_and_reference_cleanup(
    video_request, tmp_path, monkeypatch
):
    pipeline, _events = _pipeline()
    clock = [100.0]

    def advance(seconds):
        clock[0] += seconds

    def read(_path):
        advance(3)
        return SimpleNamespace(image=object(), close=lambda: advance(17))

    monkeypatch.setattr("vflash.pipeline.runtime.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("vflash.pipeline.runtime.read_reference", read)
    pipeline.prepared.check_unchanged = lambda: advance(2)
    for stage, method_name, seconds in (
        (pipeline._conditioner, "capture", 7),
        (pipeline._core, "generate", 13),
        (pipeline._media, "generate_mp4", 11),
    ):
        original = getattr(stage, method_name)

        def timed(*args, _original=original, _seconds=seconds, **kwargs):
            advance(_seconds)
            return _original(*args, **kwargs)

        setattr(stage, method_name, timed)
    result = pipeline.generate(video_request, tmp_path / "complete.mp4")
    assert result.elapsed_seconds == 53
    assert result.stages["input_preparation"] == {
        "elapsed_seconds": 5,
        "reference_loading_seconds": 3,
    }
    assert result.stages["encoding"]["elapsed_seconds"] == 7
    assert result.stages["media"]["elapsed_seconds"] == 11
    assert result.stages["session_initialization_seconds"] == 1
    assert not pipeline._lock.locked()


def test_stage_timings_keep_transfer_and_call_costs_inside_outer_duration(
    video_request, tmp_path, monkeypatch
):
    pipeline, _events = _pipeline()
    clock = [0.0]
    monkeypatch.setattr("vflash.pipeline.runtime.time.monotonic", lambda: clock[0])

    def duration(seconds):
        clock[0] += seconds
        return seconds

    for stage, resume, suspend in (
        (pipeline._conditioner, 2, 3),
        (pipeline._media, 5, 7),
    ):
        stage.resume_cuda = lambda value=resume: duration(value)
        stage.suspend_cuda = lambda value=suspend: duration(value)
    for stage, name, seconds in (
        (pipeline._conditioner, "capture", 11),
        (pipeline._media, "generate_mp4", 13),
    ):
        original = getattr(stage, name)

        def timed(*args, _original=original, _seconds=seconds, **kwargs):
            duration(_seconds)
            return _original(*args, **kwargs)

        setattr(stage, name, timed)
    result = pipeline.generate(
        video_request,
        tmp_path / "timed.mp4",
        progress=lambda event: duration(1) if event.stage in {"encoding", "decoding"} else None,
    )
    encoding, media = result.stages["encoding"], result.stages["media"]
    assert encoding["elapsed_seconds"] == 18
    assert encoding["weight_resume_seconds"] == 2
    assert encoding["capture_call_seconds"] == 11
    assert encoding["suspend_seconds"] == 3
    assert media["elapsed_seconds"] == 27
    assert media["weight_resume_seconds"] == 5
    assert media["decode_call_seconds"] == 13
    assert media["suspend_seconds"] == 7
    assert result.elapsed_seconds == 45


@pytest.mark.parametrize(
    "failure",
    [
        "conditioning:capture",
        "conditioning:suspend",
        "native:generate",
        "media:generate",
        "media:suspend",
    ],
)
def test_failed_request_closes_owners_and_does_not_publish(video_request, tmp_path, failure):
    pipeline, events = _pipeline(fail=failure)
    target = tmp_path / "output.mp4"
    with pytest.raises(RuntimeError, match=failure):
        pipeline.generate(video_request, target)
    assert not target.exists() and not list(tmp_path.glob(".vflash-*"))
    assert pipeline._closed and pipeline._released
    assert events[-3:] == ["conditioning:close", "media:close", "native:close"]
    with pytest.raises(ContractError, match="closed"):
        pipeline.generate(video_request, target)


def test_callback_cancellation_before_publication_cleans_up(video_request, tmp_path):
    pipeline, events = _pipeline()

    def cancel(event):
        if event.stage == "decoding" and event.completed == 1:
            raise ValueError("cancel before publication")

    target = tmp_path / "output.mp4"
    with pytest.raises(ValueError, match="cancel before publication"):
        pipeline.generate(video_request, target, progress=cancel)
    assert not target.exists() and not list(tmp_path.glob(".vflash-*"))
    assert events[-3:] == ["conditioning:close", "media:close", "native:close"]


def test_close_from_progress_callback_fails_without_deadlocking(video_request, tmp_path):
    pipeline, events = _pipeline()
    with pytest.raises(ContractError, match="progress callback"):
        pipeline.generate(
            video_request, tmp_path / "output.mp4", progress=lambda _: pipeline.close()
        )
    assert pipeline._released
    assert events[-3:] == ["conditioning:close", "media:close", "native:close"]


def test_bad_input_and_busy_session_do_not_run_models_or_destroy_outputs(
    video_request, tmp_path
):
    pipeline, events = _pipeline()
    target = tmp_path / "output.mp4"
    target.write_bytes(b"existing user video")
    with pytest.raises(ContractError, match="already exists"):
        pipeline.generate(video_request, target)
    assert target.read_bytes() == b"existing user video"
    pipeline._lock.acquire()
    try:
        with pytest.raises(ContractError, match="busy"):
            pipeline.generate(video_request, tmp_path / "new.mp4")
    finally:
        pipeline._lock.release()
    assert not events and not pipeline._closed


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 641},
        {"height": 0},
        {"width": 1344, "height": 768},
        {"prompt": " "},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**63},
    ],
)
def test_request_limits_are_cpu_contracts(tmp_path, changes):
    values = {"prompt": "An example scene.", "reference": tmp_path / "reference.png", **changes}
    with pytest.raises(ContractError):
        VideoRequest(**values)


def test_ordered_multi_reference_request_keeps_labels_and_seed_replacement(tmp_path):
    images = tuple(tmp_path / f"image-{index}.png" for index in range(3))
    request = VideoRequest(
        "<Picture 3> is the setting; <Picture 1> and <Picture 2> are the subjects.",
        references=images,
    )
    assert request.ordered_references == images
    assert replace(request, seed=19).ordered_references == images
    legacy = VideoRequest("<Picture 1> moves.", images[0])
    assert replace(legacy, seed=19).ordered_references == (images[0],)


@pytest.mark.parametrize(
    "values",
    [
        {"prompt": "<Picture 1> without an image"},
        {"references": (Path("one"),) * 4},
        {"reference": Path("one"), "references": (Path("two"),)},
        {"references": [Path("one")]},
        {"references": ("one",)},
        {"prompt": "<Picture 0>", "reference": Path("one")},
        {"prompt": "<Picture 2>", "reference": Path("one")},
    ],
)
def test_ambiguous_or_invalid_reference_contracts_fail_before_cuda(values):
    with pytest.raises(ContractError):
        VideoRequest(**{"prompt": "A scene.", **values})


def test_multi_reference_loading_preserves_order_and_closes_all_images(tmp_path, monkeypatch):
    pipeline, _events = _pipeline()
    paths = tuple(tmp_path / str(index) for index in range(3))
    loaded, closed, captured = [], [], []

    def read(path):
        loaded.append(path)
        return SimpleNamespace(image=path, close=lambda: closed.append(path))

    monkeypatch.setattr("vflash.pipeline.runtime.read_reference", read)
    original = pipeline._conditioner.capture

    def capture(request, references, directory):
        captured.extend(reference.image for reference in references)
        return original(request, references, directory)

    pipeline._conditioner.capture = capture
    pipeline.generate(VideoRequest("Three references.", references=paths), tmp_path / "out.mp4")
    assert loaded == captured == list(paths)
    assert closed == list(reversed(paths))


def test_invalid_later_image_closes_previous_images_without_retiring_models(
    tmp_path, monkeypatch
):
    pipeline, events = _pipeline()
    paths = (tmp_path / "good.png", tmp_path / "bad.png")
    closed = []

    def read(path):
        if path == paths[1]:
            raise ContractError("invalid second image")
        return SimpleNamespace(image=path, close=lambda: closed.append(path))

    monkeypatch.setattr("vflash.pipeline.runtime.read_reference", read)
    with pytest.raises(ContractError, match="second image"):
        pipeline.generate(VideoRequest("A scene.", references=paths), tmp_path / "out.mp4")
    assert closed == [paths[0]]
    assert not events and not pipeline._closed and not pipeline._lock.locked()
    assert not (tmp_path / "out.mp4").exists()


def test_text_only_request_reuses_all_stages_without_reference_loading(tmp_path, monkeypatch):
    from vflash.model_assets import model_profile

    pipeline, events = _pipeline()
    pipeline.profile = model_profile("t2va-turbo4-exact-sm89")
    pipeline.prepared.profile_id = pipeline.profile.definition.id
    monkeypatch.setattr(
        "vflash.pipeline.runtime.read_reference", lambda _: pytest.fail("T2VA loaded an image")
    )
    observed = []
    original = pipeline._conditioner.capture

    def capture(request, references, directory):
        observed.append(references)
        bundle = original(request, references, directory)
        bundle.profile = H3ConditioningProfile(
            "t2va", request.width, request.height, 124, 4, 6, 3, 0, 0, 0
        )
        return bundle

    pipeline._conditioner.capture = capture
    with pipeline:
        first = pipeline.generate(VideoRequest("A scene.", seed=12), tmp_path / "first.mp4")
        second = pipeline.generate(
            VideoRequest("A second scene.", seed=13), tmp_path / "second.mp4"
        )
    assert first.profile_id == second.profile_id == "t2va-turbo4-exact-sm89"
    assert observed == [(), ()]
    assert events.count("native:generate") == 2 and events.count("media:generate") == 2
    assert events[-3:] == ["conditioning:close", "media:close", "native:close"]
