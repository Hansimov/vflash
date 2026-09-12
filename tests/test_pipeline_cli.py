from pathlib import Path
from types import SimpleNamespace

import pytest

from vflash.cli import build_parser, main
from vflash.pipeline.contracts import PipelineProgress, VideoResult


def test_generate_parser_preserves_reference_order():
    args = build_parser().parse_args(
        [
            "generate",
            "--prepared-assets",
            "receipt.json",
            "--prompt-file",
            "prompt.txt",
            "--reference",
            "second.png",
            "--reference",
            "first.png",
            "--gpu",
            "0",
            "--output",
            "output.mp4",
            "--trust-local-code",
        ]
    )
    assert args.reference == [Path("second.png"), Path("first.png")]
    assert args.duration == 5


def test_generate_parser_exposes_one_explicit_video_path():
    args = build_parser().parse_args(
        [
            "generate",
            "--prepared-assets",
            "receipt.json",
            "--prompt-file",
            "prompt.txt",
            "--reference-video",
            "source.mp4",
            "--gpu",
            "0",
            "--output",
            "result.mp4",
        ]
    )
    assert args.reference_video == Path("source.mp4") and args.reference == []


def test_generate_parser_exposes_first_frame_separately_from_references():
    args = build_parser().parse_args(
        [
            "generate",
            "--prepared-assets",
            "receipt.json",
            "--prompt-file",
            "prompt.txt",
            "--first-frame",
            "frame-zero.png",
            "--duration",
            "10",
            "--gpu",
            "0",
            "--output",
            "result.mp4",
        ]
    )
    assert args.first_frame == Path("frame-zero.png")
    assert args.duration == 10
    assert args.reference == [] and args.reference_video is None


def test_generate_parser_exposes_first_and_last_frames_for_fl2va():
    args = build_parser().parse_args(
        [
            "generate",
            "--prepared-assets",
            "receipt.json",
            "--prompt-file",
            "prompt.txt",
            "--first-frame",
            "frame-zero.png",
            "--last-frame",
            "frame-last.png",
            "--gpu",
            "0",
            "--output",
            "result.mp4",
        ]
    )
    assert (args.first_frame, args.last_frame) == (
        Path("frame-zero.png"),
        Path("frame-last.png"),
    )


def test_generate_cli_runs_one_owned_pipeline_and_keeps_progress_off_stdout(
    tmp_path, monkeypatch, capsys
):
    import vflash.pipeline

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A private test prompt about <Picture 1>.")
    output = tmp_path / "output.mp4"
    events = []
    device = SimpleNamespace(index=4)
    monkeypatch.setattr("vflash.hardware.discover_nvidia_devices", lambda: (device,))
    monkeypatch.setattr(
        vflash.pipeline,
        "load_prepared_pipeline_assets",
        lambda _: SimpleNamespace(profile_id="ref2va-turbo4-exact-sm89"),
    )
    monkeypatch.setattr(
        "vflash.adapters.references.read_reference",
        lambda _: SimpleNamespace(close=lambda: events.append("reference:close")),
    )

    class Pipeline:
        def __init__(self, prepared, **kwargs):
            assert prepared.profile_id == "ref2va-turbo4-exact-sm89"
            assert kwargs == {"device": device, "trust_local_code": True}

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("close")

        def generate(self, request, path, *, progress):
            assert request.ordered_references == (Path("one.png"), Path("two.png"))
            assert request.prompt == prompt.read_text()
            assert request.duration_seconds == 5
            assert path == output
            progress(PipelineProgress("denoising", 1, 4))
            return VideoResult(path, "ref2va-turbo4-exact-sm89", request.seed, 1.5, {}, {})

    monkeypatch.setattr(vflash.pipeline, "H3Pipeline", Pipeline)
    assert (
        main(
            [
                "generate",
                "--prepared-assets",
                "receipt.json",
                "--prompt-file",
                str(prompt),
                "--reference",
                "one.png",
                "--reference",
                "two.png",
                "--gpu",
                "4",
                "--output",
                str(output),
                "--trust-local-code",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"denoising"' in captured.err and '"elapsed_seconds": 1.5' in captured.out
    assert prompt.read_text() not in captured.out + captured.err
    assert events == ["enter", "close"]


def test_generate_cli_requires_explicit_official_code_consent(tmp_path):
    with pytest.raises(SystemExit, match="--trust-local-code"):
        main(
            [
                "generate",
                "--prepared-assets",
                "missing.json",
                "--prompt-file",
                "missing.txt",
                "--reference",
                "missing.png",
                "--gpu",
                "0",
                "--output",
                str(tmp_path / "out.mp4"),
            ]
        )


def test_text_only_and_dual_gpu_flags_are_explicit_cli_contracts():
    args = build_parser().parse_args(
        [
            "generate",
            "--prepared-assets",
            "receipt.json",
            "--prompt-file",
            "prompt.txt",
            "--gpu",
            "0",
            "--output",
            "output.mp4",
            "--trust-local-code",
        ]
    )
    assert args.reference == [] and args.peer_gpu is None and args.strategy is None
    args = build_parser().parse_args(
        [
            "generate",
            "--prepared-assets",
            "receipt.json",
            "--prompt-file",
            "prompt.txt",
            "--reference",
            "ref.png",
            "--gpu",
            "0",
            "--peer-gpu",
            "1",
            "--strategy",
            "sequence-head",
            "--output",
            "output.mp4",
            "--trust-local-code",
        ]
    )
    assert args.peer_gpu == 1 and args.strategy == "sequence-head"
    args = build_parser().parse_args(
        [
            "prepare-pipeline",
            "--assets",
            "assets.json",
            "--receipt",
            "receipt.json",
            "--profile",
            "t2va-turbo4-exact-sm89",
        ]
    )
    assert args.profile == "t2va-turbo4-exact-sm89"
