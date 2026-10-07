"""Run on the pinned pipeline extra, without weights or any CUDA allocation."""

from types import SimpleNamespace

import pytest


def test_official_video_resize_once_and_image_policy_restored_after_each_request(monkeypatch):
    pytest.importorskip("diffusers")
    import numpy as np
    import torch
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from PIL import Image

    from vflash.adapters.references import install_match_reference_setup_block
    from vflash.native.h3_conditioning_bundle import H3_VIDEO_REFERENCE_POLICY

    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CPU setup attempted CUDA"))
    pipe = SimpleNamespace(
        _blocks=SimpleNamespace(
            sub_blocks={
                "before_encode": MiniMaxH3Ref2VASetupStep(),
            }
        )
    )
    block = install_match_reference_setup_block(pipe)
    components = SimpleNamespace(
        canvas_multiple=32,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        fps=24,
        min_duration=5,
        max_duration=15,
        config=SimpleNamespace(canvas_short_edge=768, canvas_max_pixels=768 * 1344),
    )
    pixels = np.arange(48 * 128 * 232 * 3, dtype=np.uint8).reshape(48, 128, 232, 3)
    source = MiniMaxH3VideoReference(frames=pixels, fps=24, audio=None)
    original = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
        pixels, 24, 124, 32, 768, 768 * 1344, 24
    )
    assert original.shape == (48, 768, 1376, 3)

    def state(reference):
        return PipelineState(
            values={"references": [reference], "height": 352, "width": 640, "num_frames": 124}
        )

    with Image.new("RGB", (232, 128), "white") as image:
        for reference in [
            MiniMaxH3ImageReference(image=image),
            source,
            MiniMaxH3ImageReference(image=image),
        ]:
            request = state(reference)
            block(components, request)
            normalized = request.get("normalized_references")[0]
            if reference.kind == "video":
                assert block.last_metadata["policy"] == H3_VIDEO_REFERENCE_POLICY
                assert np.array_equal(normalized.frames, original)
                assert not normalized.has_audio
                assert source.frames is pixels  # source is never resized in place
            else:
                assert block.last_metadata["policy"] == "match"
                assert normalized.image.size == (224, 128)
                if normalized.image is not image:
                    normalized.image.close()
        # A rejected video must not change the policy for the next image.
        with pytest.raises(ValueError, match="CFR24"):
            block(components, state(MiniMaxH3VideoReference(frames=pixels, fps=12)))
        assert block.last_metadata is None
        request = state(MiniMaxH3ImageReference(image=image))
        block(components, request)
        assert block.last_metadata["policy"] == "match"
        request.get("normalized_references")[0].image.close()
        # Fifteen delivered seconds require 362 internal VAE frames.
        # Nine distinct ordered images retain identity and still use no CUDA.
        references = [
            MiniMaxH3ImageReference(image=Image.new("RGB", (64, 64), (i * 20, 0, 0)))
            for i in range(9)
        ]
        try:
            request = PipelineState(
                values={
                    "references": references,
                    "height": 512,
                    "width": 512,
                    "num_frames": 362,
                }
            )
            block(components, request)
            assert request.get("num_frames") == 362
            normalized = request.get("normalized_references")
            assert len(normalized) == 9
            assert [entry.image.getpixel((0, 0))[0] for entry in normalized] == [
                i * 20 for i in range(9)
            ]
            for entry in normalized:
                entry.image.close()
            request = PipelineState(
                values={
                    "references": references,
                    "height": 512,
                    "width": 512,
                    "num_frames": 379,
                }
            )
            with pytest.raises(ValueError, match="seconds"):
                block(components, request)
        finally:
            for entry in references:
                entry.image.close()
    assert not torch.cuda.is_initialized()


def test_keyframe_layout_pads_fifteen_seconds_without_mutating_pipeline_limits():
    pytest.importorskip("diffusers")
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    from vflash.adapters.conditioning_temporal import install_delivery_frame_layout

    pipe = SimpleNamespace(
        _blocks=SimpleNamespace(
            sub_blocks={
                "before_denoise": SimpleNamespace(
                    sub_blocks={"layout": MiniMaxH3PrepareLayoutStep()}
                )
            }
        )
    )
    install_delivery_frame_layout(pipe)
    block = pipe._blocks.sub_blocks["before_denoise"].sub_blocks["layout"]
    components = SimpleNamespace(
        _execution_device=torch.device("cpu"),
        canvas_multiple=32,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        fps=24,
        min_duration=5,
        max_duration=15,
        vae_spatial_compression_ratio=16,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
    )
    for frames in (124, 345, 362):
        request = PipelineState(
            values={
                "text_token_tags": torch.ones(4, dtype=torch.long),
                "height": 512,
                "width": 512,
                "num_frames": frames,
                "keyframe_anchors": ("first", "last"),
            }
        )
        original, result = block(components, request)
        assert original is components and components.max_duration == 15
        assert result.get("num_frames") == frames
        assert result.get("num_latent_frames") == 5 * ((frames - 5) // 17) + 2
        assert result.get("num_condition_video_rows") == 512
    request = PipelineState(
        values={
            "text_token_tags": torch.ones(4, dtype=torch.long),
            "height": 512,
            "width": 512,
            "num_frames": 379,
        }
    )
    with pytest.raises(ValueError, match="seconds"):
        block(components, request)
    assert components.max_duration == 15 and not torch.cuda.is_initialized()
