"""Actual CPU audio hooks retain the native short clock and restore their owners."""

import pytest
from test_h3_reference_audio import Block, Processor

from vflash.adapters.h3_reference_audio import native_audio_context
from vflash.pipeline import VideoRequest
from vflash.pipeline.pika_av import PikaAVRecipe, validate_pika_av_request

torch = pytest.importorskip("torch")
nn = torch.nn


class Scheduler:
    def step(self):
        return "original"


class AVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block()])
        self.video_projection = nn.Linear(96, 8)
        self.positions = []

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        audio_hidden_states,
        *,
        token_tags,
        timestep,
        timestep_indices,
        video_indices,
        audio_indices,
        text_indices,
        position_ids,
    ):
        # Small real projections/attention exercise all three hooks together.
        count = len(token_tags)
        hidden = torch.empty(1, count, 8)
        hidden[:, video_indices] = self.video_projection(hidden_states)
        hidden[:, text_indices] = encoder_hidden_states
        hidden[:, audio_indices] = audio_hidden_states
        self.positions.append(position_ids[video_indices].clone())
        rotary = (torch.ones(count, 4), torch.zeros(count, 4))
        for block in self.transformer_blocks:
            hidden = block(hidden, timestep, timestep_indices, rotary)
        return hidden


@pytest.mark.parametrize("frames", [32, 37])
@pytest.mark.parametrize("first_frame", [False, True])
def test_complete_pinning_spatial_and_visual_kv_pass_on_each_native_clock(frames, first_frame):
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, scheduler = AVModel().eval(), Scheduler()
        height, width = 3, 5
        condition = height * width if first_frame else 0
        video = torch.randn(1, 24, frames, height * 2, width * 2)
        original = video.clone()
        count = condition + frames * height * width
        hidden = torch.randn(1, count, 96)
        anchor = hidden[:, :condition].clone()
        args = dict(
            hidden_states=hidden,
            encoder_hidden_states=torch.randn(1, 3, 8),
            audio_hidden_states=torch.randn(1, 4, 8),
            text_indices=torch.arange(3),
            audio_indices=torch.arange(3, 7),
            video_indices=torch.arange(7, 7 + count),
            token_tags=torch.cat((torch.ones(3), torch.full((4,), 2), torch.zeros(count))),
            position_ids=torch.arange((7 + count) * 3).reshape(7 + count, 3),
            timestep=torch.tensor([0.2, 0.999]),
            timestep_indices=torch.zeros(7 + count, dtype=torch.long),
        )
        with (
            torch.inference_mode(),
            native_audio_context(
                model,
                scheduler,
                video,
                condition_rows=condition,
                expected_forwards=10,
                reuse_visual_context=True,
                spatial_stride=2,
                latent_frames=frames,
            ) as receipt,
        ):
            for _ in range(10):
                result = model(**args)
                assert torch.isfinite(result).all()
                sample = hidden[0, condition:]
                assert (
                    scheduler.step(None, None, sample, return_dict=False)[0].data_ptr()
                    == sample.data_ptr()
                )
                args["audio_hidden_states"] = args["audio_hidden_states"] + 0.1
            assert receipt["spatial_context"]["forwards"] == 10
            assert receipt["visual_context"]["build_block_calls"] == 1
            assert receipt["visual_context"]["reuse_block_calls"] == 9
            assert receipt["spatial_context"]["retained_video_rows"] == condition + frames * 6
        selected = torch.cat(
            (
                torch.arange(condition),
                torch.arange(frames * height * width)
                .reshape(frames, height, width)[:, ::2, ::2]
                .flatten()
                + condition,
            )
        )
        expected_positions = args["position_ids"][args["video_indices"][selected]]
        assert all(torch.equal(pos, expected_positions) for pos in model.positions)
        torch.testing.assert_close(hidden[:, :condition], anchor, rtol=0, atol=0)
        torch.testing.assert_close(video, original, rtol=0, atol=0)
        assert not model._forward_pre_hooks and scheduler.step() == "original"
        assert type(model.transformer_blocks[0].attn.processor) is Processor
        assert not hasattr(model, "_vflash_spatial_audio_active")
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(before_threads)


@pytest.mark.parametrize("audio_steps", [0, 12])
def test_unqualified_short_audio_recipe_stops_before_execution(audio_steps):
    with pytest.raises(ValueError):
        validate_pika_av_request(
            VideoRequest("A kite turns.", duration_seconds=4),
            PikaAVRecipe(audio_steps=audio_steps),
        )


def test_short_audio_cannot_borrow_the_default_five_second_context():
    model, scheduler = AVModel().eval(), Scheduler()
    source = torch.zeros(1, 24, 32, 6, 10)
    for options in ({}, {"latent_frames": 32, "expected_forwards": 12}):
        with (
            torch.inference_mode(),
            pytest.raises(ValueError),
            native_audio_context(
                model,
                scheduler,
                source,
                condition_rows=0,
                reuse_visual_context=True,
                spatial_stride=2,
                **{"expected_forwards": 10, **options},
            ),
        ):
            pytest.fail("unsupported audio geometry entered the model")
        assert not model._forward_pre_hooks and scheduler.step() == "original"
