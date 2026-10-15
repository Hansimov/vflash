from contextlib import contextmanager
from types import SimpleNamespace as N

import pytest
from test_complete_pipeline import _pipeline
from test_complete_pipeline import video_request as video_request

from vflash.contracts import ContractError
from vflash.model_assets import model_profile
from vflash.native.h3_audio_refinement import AudioRefinementRequest
from vflash.native.h3_native_conditioning_runtime import (
    H3NativeConditioningRuntime,
    H3NativeConditioningRuntimeError,
)
from vflash.native.h3_refinement_schedule import refinement_schedule
from vflash.pipeline import VideoRequest

torch = pytest.importorskip("torch")


def audio_request():
    return AudioRefinementRequest(
        refinement_schedule(steps=6, denoise=0.5),
        tuple([torch.zeros(7, 9, 6, 2) for _ in range(50)] + [torch.zeros(7, 3, 2, 2)]),
    )


def test_pipeline_owns_refinement_and_decode_guard(video_request, tmp_path, monkeypatch):
    pipeline, events = _pipeline()
    pipeline.profile = model_profile("i2va-turbo4-v01-544-exact-sm120")
    pipeline.prepared.profile_id = pipeline.profile.definition.id
    pipeline.hybrid_model = object()
    pipeline._check_hybrid_unchanged = lambda: None
    pipeline.weight_residency = "resident"
    pipeline.attention_backend = "veda-triton"
    pipeline.veda_dense_edge_layers = 3
    pipeline._core.runtime = N(metadata=lambda: {"model_variant": {"hybrid": True}})
    pipeline._media.audio = object()
    request = VideoRequest("A person speaks.", first_frame=video_request.reference)
    refinement = audio_request()
    progress = []

    def generate(bundle, output, *, audio_refinement, progress_callback):
        assert audio_refinement is refinement
        output.write_bytes(b"owned latent test placeholder")
        for n in range(1, 11):
            progress_callback(n, 10)
        return {
            "generation": {
                "audio_refinement": {"extra_audio_nfe": 6, "video_latents_identical": True},
                "nfe": 4,
            }
        }

    pipeline._core.generate = generate

    @contextmanager
    def headroom(component, record):
        assert component is pipeline._media.audio
        events.append("headroom:enter")
        try:
            yield
        finally:
            events.append("headroom:exit")
        record.update(linked_stereo=True, gain=0.9)

    monkeypatch.setattr("vflash.media.audio_headroom.audio_decoder_headroom", headroom)
    result = pipeline.generate(
        request, tmp_path / "audio.mp4", audio_refinement=refinement, progress=progress.append
    )
    assert result.output_path.is_file()
    assert result.stages["denoising"]["audio_refinement"]["extra_audio_nfe"] == 6
    assert result.stages["audio_decoder_headroom"]["linked_stereo"]
    assert [(p.completed, p.total) for p in progress if p.stage == "denoising"] == [
        (n, 10) for n in range(11)
    ]
    assert (
        events.index("headroom:enter")
        < events.index("media:generate")
        < events.index("headroom:exit")
    )
    assert not list(tmp_path.glob(".vflash-*"))


@pytest.mark.parametrize(
    "variant", ["bad_type", "block_ring", "ref2va", "profile", "dense10", "mute"]
)
def test_invalid_audio_pipeline_option_fails_before_loading(video_request, tmp_path, variant):
    pipeline, events = _pipeline()
    pipeline.profile = model_profile("i2va-turbo4-v01-544-exact-sm120")
    pipeline.hybrid_model = object()
    pipeline.weight_residency = "resident"
    pipeline.attention_backend = "veda-triton"
    pipeline.veda_dense_edge_layers = 3
    request = VideoRequest("A person speaks.", first_frame=video_request.reference)
    options = {"audio_refinement": audio_request()}
    if variant == "bad_type":
        options["audio_refinement"] = True
    if variant == "block_ring":
        pipeline.weight_residency = "block-ring"
    if variant == "ref2va":
        request = VideoRequest("A scene.", reference=video_request.reference)
    if variant == "profile":
        options["profile_denoise"] = True
    if variant == "dense10":
        pipeline.veda_dense_edge_layers = 5
    if variant == "mute":
        from dataclasses import replace

        from vflash.media.audio_delivery import SILENT_AUDIO_PROFILE

        request = replace(request, audio_delivery_profile=SILENT_AUDIO_PROFILE)
    with pytest.raises(ContractError):
        pipeline.generate(request, tmp_path / "invalid.mp4", **options)
    assert not events and not pipeline._closed and not (tmp_path / "invalid.mp4").exists()


def test_native_rejects_untyped_audio_before_loading(tmp_path):
    runtime = H3NativeConditioningRuntime.__new__(H3NativeConditioningRuntime)
    runtime._closed = False
    runtime._torch = N()
    with pytest.raises(H3NativeConditioningRuntimeError, match="audio refinement"):
        runtime.generate_latents(None, tmp_path / "latents.safetensors", audio_refinement=True)
