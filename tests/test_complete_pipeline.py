from __future__ import annotations

import threading
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

        def resume_cuda(self) -> None:
            self._event("resume")

        def suspend_cuda(self) -> None:
            self._event("suspend")

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
    pipeline.prepared = SimpleNamespace(check_unchanged=lambda: None)
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
