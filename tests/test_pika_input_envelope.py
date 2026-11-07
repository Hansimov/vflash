"""Pika-specific CPU admission and native four-second clock propagation."""

import subprocess
import sys
from pathlib import Path

import pytest

from vflash.contracts import ContractError
from vflash.pipeline import VideoRequest, validate_request
from vflash.pipeline.pika_av import PikaAVRecipe, PikaAVSession, validate_pika_av_request


@pytest.mark.parametrize("mode", ["t2va", "i2va"])
@pytest.mark.parametrize("duration,frames", [(4, 107), (5, 124)])
@pytest.mark.parametrize("shape", [(512, 512), (1376, 768), (768, 1376)])
def test_exact_native_clock_and_768p_envelope(mode, duration, frames, shape):
    request = VideoRequest(
        "A character turns.",
        width=shape[0],
        height=shape[1],
        first_frame=Path("first.png") if mode == "i2va" else None,
        duration_seconds=duration,
    )
    validate_pika_av_request(request, PikaAVRecipe())
    assert request.model_frames == frames and request.delivery_frames == duration * 24
    if duration == 4 or (mode == "t2va" and shape[0] * shape[1] > 1024**2):
        with pytest.raises(ContractError, match="Pika AV"):
            validate_request(
                request,
                profile_id="i2va-turbo4-v01-544-exact-sm120",
                hybrid=True,
                attention_backend="veda-triton",
            )


@pytest.mark.parametrize(
    "first,last,refs",
    [
        (None, Path("last.png"), ()),
        (Path("first.png"), Path("last.png"), ()),
        (None, None, (Path("reference.png"),)),
    ],
)
def test_short_duration_does_not_expand_unqualified_modes(first, last, refs):
    with pytest.raises(ContractError, match="four-second"):
        VideoRequest(
            "A scene.", first_frame=first, last_frame=last, references=refs, duration_seconds=4
        )


def test_short_large_canvas_is_rejected_before_runtime_or_cuda_access():
    request = VideoRequest(
        "A scene.", first_frame=Path("first.png"), duration_seconds=4, width=1920, height=1088
    )
    session = object.__new__(PikaAVSession)
    with pytest.raises(ValueError, match="bounded Pika"):
        session.generate(request, "unused-output")
    validate_pika_av_request(
        VideoRequest("A scene.", first_frame=Path("first.png"), width=1920, height=1088),
        PikaAVRecipe(),
    )


@pytest.mark.parametrize("duration,shape", [(4, (512, 512)), (5, (1376, 768))])
def test_new_original_scope_does_not_expand_qad(duration, shape):
    request = VideoRequest(
        "A scene.", duration_seconds=duration, width=shape[0], height=shape[1]
    )
    with pytest.raises(ValueError, match="bounded Pika"):
        validate_pika_av_request(request, PikaAVRecipe(visual_weights="nvfp4-qad"))


def test_cpu_admission_import_does_not_initialize_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from vflash.pipeline import VideoRequest
from vflash.pipeline.pika_av import PikaAVRecipe, validate_pika_av_request
import sys
validate_pika_av_request(VideoRequest('A scene.', duration_seconds=4), PikaAVRecipe())
assert 'torch' not in sys.modules
""",
        ],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("mode", ["t2va", "i2va"])
def test_actual_session_propagates_four_seconds_through_picture_audio_and_delivery(
    tmp_path, monkeypatch, mode
):
    from test_pika_av import test_actual_session_counts_freezes_picture_and_restores_owners

    test_actual_session_counts_freezes_picture_and_restores_owners(
        tmp_path,
        monkeypatch,
        mode,
        steps=10,
        fail_audio=False,
        tile_batch=1,
        staged_text=False,
        duration=4,
    )


def test_attention_configuration_tracks_actual_frames_and_keeps_previous_on_invalid_input(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from vflash.adapters.h3_reference_attention import H3ReferenceAttention

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList()

    monkeypatch.setattr(sys.modules[__name__], "attention", lambda *a, **k: None, raising=False)
    model = Model()
    adapter = H3ReferenceAttention(Model)
    try:
        adapter.configure(model, {"width": 1376, "height": 768, "frames": 107}, None)
        assert adapter.profile.frames == 107
        handles = list(adapter.handles)
        with pytest.raises(ValueError, match="aligned model frames"):
            adapter.configure(model, {"width": 1376, "height": 768, "frames": 96}, None)
        assert adapter.handles == handles and adapter.profile.frames == 107
        adapter.configure(model, {"width": 1376, "height": 768}, None)
        assert adapter.profile.frames == 124
    finally:
        adapter.close()
    assert not model._forward_pre_hooks and not torch.cuda.is_initialized()
