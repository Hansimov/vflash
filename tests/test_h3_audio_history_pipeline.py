from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as N

import pytest
from test_complete_pipeline import _pipeline
from test_complete_pipeline import video_request as video_request
from test_h3_audio_pipeline import audio_request
from test_native_av_continuation import fixture

from vflash.contracts import ContractError
from vflash.model_assets import model_profile
from vflash.pipeline import VideoRequest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "fault", [None, "missing", "different_audio", "different_cut", "missing_base"]
)
def test_complete_pipeline_binds_history_before_loading_and_runs_serially(
    video_request, tmp_path, monkeypatch, fault
):
    pipeline, events = _pipeline()
    pipeline.profile = model_profile("i2va-turbo4-v01-544-exact-sm120")
    pipeline.prepared.profile_id = pipeline.profile.definition.id
    pipeline.hybrid_model = object()
    pipeline._check_hybrid_unchanged = lambda: None
    pipeline.weight_residency = "block-ring"
    pipeline.attention_backend = "veda-sm89"
    pipeline.veda_dense_edge_layers = 3
    pipeline.audio_decode_overlap = True
    pipeline._core.runtime = N(metadata=lambda: {"model_variant": {"hybrid": True}})
    pipeline._media.audio = object()
    request = VideoRequest("A native continuing scene.", first_frame=video_request.reference)
    _, prefix, _ = fixture()
    prefix = replace(
        prefix, video=torch.zeros(1, 24, 12, request.height // 16, request.width // 16)
    )
    history = replace(prefix, preserve_video=False)
    if fault == "different_audio":
        history = replace(history, audio=history.audio + 1)
    if fault == "different_cut":
        history = replace(history, source_frames=158)
    refinement = replace(
        audio_request(), native_audio_prefix=None if fault == "missing" else history
    )
    base = None if fault == "missing_base" else prefix

    def generate(bundle, output, *, audio_refinement, native_av_prefix, progress_callback):
        assert audio_refinement is refinement and native_av_prefix is prefix
        # No video_ready_callback: history and modulation caches own serial resources.
        output.write_bytes(b"owned latent test placeholder")
        return {
            "generation": {
                "audio_refinement": {"extra_audio_nfe": 6, "video_latents_identical": True},
                "nfe": 4,
            }
        }

    pipeline._core.generate = generate

    @contextmanager
    def headroom(component, record):
        yield

    monkeypatch.setattr("vflash.media.audio_headroom.audio_decoder_headroom", headroom)
    if fault:
        with pytest.raises(ContractError):
            pipeline.generate(
                request,
                tmp_path / "out.mp4",
                audio_refinement=refinement,
                native_av_prefix=base,
            )
        assert not events
    else:
        result = pipeline.generate(
            request, tmp_path / "out.mp4", audio_refinement=refinement, native_av_prefix=base
        )
        assert result.output_path.is_file()
        assert "audio_video_overlap" not in result.stages
