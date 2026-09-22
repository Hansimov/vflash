import subprocess

import pytest

from vflash.media.encoding import MediaError
from vflash.media.restoration import encode_restored_video


def test_real_restoration_delivery_keeps_source_audio_and_original(tmp_path):
    from PIL import Image

    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=32x32:r=24:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        timeout=30,
    )
    original = source.read_bytes()
    frames = [Image.new("RGB", (32, 32), "blue") for _ in range(24)]
    output = tmp_path / "enhanced.mp4"
    result = encode_restored_video(frames, source, output)
    assert result["audio_delivery"] == "source-packet-copy"
    assert result["frames"] == 24 and source.read_bytes() == original

    def pcm(path):
        return subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "pipe:1",
            ],
            timeout=30,
        )

    assert pcm(source) == pcm(output)
    with pytest.raises(MediaError, match="already exists"):
        encode_restored_video(frames, source, source)
    with pytest.raises(MediaError, match="complete 24 fps"):
        encode_restored_video(frames[:-1], source, tmp_path / "wrong.mp4")
    assert not (tmp_path / "wrong.mp4").exists()


def test_refuses_existing_destination_before_work(tmp_path):
    from PIL import Image

    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep this output")
    with pytest.raises(MediaError, match="already exists"):
        encode_restored_video([Image.new("RGB", (32, 32))], tmp_path / "missing.mp4", output)
    assert output.read_bytes() == b"keep this output"


@pytest.mark.parametrize("memory_policy", ["offload", "resident"])
def test_complete_python_path_decodes_and_publishes_without_mutating_source(
    tmp_path, monkeypatch, memory_policy
):
    pytest.importorskip("cv2")
    from types import SimpleNamespace

    import vflash.restoration as pipeline

    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=32x32:r=24:d=1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
        timeout=30,
    )
    original = source.read_bytes()
    observed = {}

    class Backend:
        def __init__(self, **options):
            observed.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            observed["closed"] = True

        def restore(self, frames, **options):
            assert len(frames) == 24
            assert sum(b - a for a, b in options["segments"]) == 24
            return SimpleNamespace(
                frames=tuple(frame.copy() for frame in frames),
                profile="test-backend",
                elapsed_seconds=0.1,
            )

    monkeypatch.setattr(pipeline, "LocalStcditTiny", Backend)
    output = tmp_path / "new" / "enhanced.mp4"
    result = pipeline.restore_video(
        source,
        output,
        source=tmp_path,
        weights=tmp_path,
        caption="A blue scene.",
        trust_local_code=True,
        memory_policy=memory_policy,
    )
    assert result["media"]["frames"] == 24
    assert result["memory_policy"] == observed["memory_policy"] == memory_policy
    assert result["media"]["audio_delivery"] == "no-source-audio"
    assert observed["closed"] and source.read_bytes() == original
    assert not list(output.parent.glob(".vflash-*"))


def test_restoration_cli_requires_explicit_local_code_trust():
    from vflash.cli import build_parser

    parser = build_parser()
    flags = [
        "restore-video",
        "--source-video",
        "in.mp4",
        "--output",
        "out.mp4",
        "--runtime-code",
        "runtime",
        "--weights",
        "weights",
        "--caption-file",
        "caption.txt",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(flags)
    args = parser.parse_args([*flags, "--trust-local-code"])
    assert args.seed == 42 and args.trust_local_code is True
    assert args.memory_policy == "offload"
    assert (
        parser.parse_args(
            [*flags, "--trust-local-code", "--memory-policy", "resident"]
        ).memory_policy
        == "resident"
    )


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "ffmpeg"),
        subprocess.TimeoutExpired("ffmpeg", 120),
    ],
)
def test_failed_decode_is_actionable_and_cleans_temporary_state(tmp_path, monkeypatch, failure):
    import vflash.restoration as pipeline

    monkeypatch.setattr(pipeline, "media_executables", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(
        pipeline,
        "probe_mp4",
        lambda *_a, **_k: {
            "streams": [
                {
                    "codec_type": "video",
                    "nb_frames": "24",
                    "width": 32,
                    "height": 32,
                    "avg_frame_rate": "24/1",
                    "r_frame_rate": "24/1",
                }
            ]
        },
    )

    def failed(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(pipeline.subprocess, "run", failed)
    with pytest.raises(MediaError, match="decoding failed or timed out"):
        pipeline.restore_video(
            tmp_path / "source.mp4",
            tmp_path / "enhanced.mp4",
            source=tmp_path,
            weights=tmp_path,
            caption="A scene.",
            trust_local_code=True,
        )
    assert not list(tmp_path.iterdir())
