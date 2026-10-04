"""GPU-free tests of optional complete-video enhancement boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from vflash.cli import build_parser
from vflash.flashvsr import FlashVSR, geometry, metadata


@pytest.mark.parametrize("frames", [1, 21, 24, 117, 120, 121, 192, 240])
def test_streaming_keeps_last_frame(frames):
    value = geometry(512, 288, frames, 2)
    assert value.padded_frames % 8 == 1
    assert value.padded_frames - 4 >= frames
    assert (value.width, value.height) == (1024, 576)
    assert value.padded_height == 640


@pytest.mark.parametrize("size", [(512, 512), (960, 544), (736, 992)])
def test_no_aspect_crop(size):
    value = geometry(*size, 120, 4)
    assert (value.width, value.height) == tuple(n * 4 for n in size)
    assert value.padded_width % 128 == value.padded_height % 128 == 0


def test_no_implicit_local_code_execution():
    with pytest.raises(ValueError, match="trust_local_code"):
        FlashVSR(Path("untrusted"), Path("weights"))


def test_explicit_cli():
    args = build_parser().parse_args(
        [
            "upscale-video",
            "--input",
            "a.mp4",
            "--output",
            "b.mp4",
            "--source",
            "code",
            "--weights",
            "weights",
            "--scale",
            "2",
        ]
    )
    assert args.scale == 2 and args.seed == 0 and not args.trust_local_code


def test_fractional_fps_is_not_rounded(monkeypatch):
    monkeypatch.setattr(
        "vflash.flashvsr.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            stdout='{"streams":[{"codec_type":"video","width":512,"height":512,"nb_frames":"120",'
            '"avg_frame_rate":"30000/1001","r_frame_rate":"30000/1001"}]}'
        ),
    )
    assert metadata(Path("test.mp4"))["fps"] == "30000/1001"


@pytest.mark.parametrize("scale", [1, 3, 2.0, True])
def test_explicit_scale(scale):
    with pytest.raises(ValueError):
        geometry(512, 512, 120, scale)
