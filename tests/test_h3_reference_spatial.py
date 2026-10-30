"""Temporal coverage, live modalities and ownership of spatial audio conditioning."""

import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

from vflash.adapters.h3_reference_audio import fixed_video_audio_pass, native_audio_context
from vflash.adapters.h3_reference_spatial import spatial_video_condition


class Model(nn.Module if torch is not None else object):
    def forward(self, **kwargs):
        return kwargs


class Scheduler:
    def step(self):
        return "original"


@unittest.skipIf(torch is None, "Spatial audio execution requires the pinned Torch image")
class SpatialAudioTests(unittest.TestCase):
    def setUp(self):
        self.model, self.scheduler = Model().eval(), Scheduler()
        self.height, self.width, self.condition = 3, 5, 15
        self.video = torch.randn(1, 24, 37, 6, 10)
        rows, length = 38 * self.condition, 7 + 38 * self.condition
        self.kwargs = dict(
            hidden_states=torch.randn(1, rows, 96),
            encoder_hidden_states=torch.randn(1, 3, 5),
            audio_hidden_states=torch.randn(1, 4, 8),
            text_indices=torch.arange(3),
            audio_indices=torch.arange(3, 7),
            video_indices=torch.arange(7, length),
            position_ids=torch.arange(length * 3).reshape(length, 3),
            token_tags=torch.cat((torch.ones(3), torch.full((4,), 2), torch.zeros(rows))),
            timestep_indices=torch.zeros(length, dtype=torch.long),
            timestep=torch.tensor([0.2, 0.999]),
        )

    def spatial(self):
        return spatial_video_condition(
            self.model, height=self.height, width=self.width, condition_rows=self.condition
        )

    def pin(self):
        return fixed_video_audio_pass(
            self.model, self.scheduler, self.video, condition_rows=self.condition
        )

    def assert_restored(self):
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertFalse(hasattr(self.model, "_vflash_spatial_audio_active"))
        self.assertFalse(hasattr(self.model, "_vflash_fixed_audio_active"))
        self.assertEqual(self.scheduler.step(), "original")

    def test_all_times_and_first_frame_are_retained_with_odd_spatial_sizes(self):
        original_positions = self.kwargs["position_ids"].clone()
        anchor = self.kwargs["hidden_states"][:, : self.condition].clone()
        with torch.inference_mode(), self.pin(), self.spatial() as record:
            actual = self.model(**self.kwargs)
            selected = torch.cat(
                (
                    torch.arange(self.condition),
                    torch.arange(37 * 15).reshape(37, 3, 5)[:, ::2, ::2].flatten()
                    + self.condition,
                )
            )
            expected = self.kwargs["hidden_states"][:, selected]
            torch.testing.assert_close(actual["hidden_states"], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual["hidden_states"][:, :15], anchor, rtol=0, atol=0)
            self.assertEqual(record["retained_video_rows"], 15 + 37 * 6)
            torch.testing.assert_close(
                actual["position_ids"][actual["video_indices"]],
                original_positions[self.kwargs["video_indices"][selected]],
                rtol=0,
                atol=0,
            )
            self.assertTrue((actual["timestep_indices"][actual["video_indices"]] == 1).all())
            self.assertTrue(torch.equal(self.kwargs["position_ids"], original_positions))
        self.assert_restored()

    def test_text_and_audio_remain_live_across_forwards(self):
        with torch.inference_mode(), self.pin(), self.spatial() as record:
            first = self.model(**self.kwargs)
            self.kwargs["audio_hidden_states"] = self.kwargs["audio_hidden_states"] + 1
            self.kwargs["encoder_hidden_states"] = self.kwargs["encoder_hidden_states"] - 1
            second = self.model(**self.kwargs)
            self.assertEqual(record["forwards"], 2)
            for name in ("audio_hidden_states", "encoder_hidden_states"):
                self.assertIs(second[name], self.kwargs[name])
                self.assertFalse(torch.equal(second[name], first[name]))
            for name in ("text_indices", "audio_indices"):
                self.assertTrue(torch.equal(second[name], self.kwargs[name]))
        self.assert_restored()

    def test_invalid_layout_unwinds_both_owners(self):
        with (
            torch.inference_mode(),
            self.assertRaisesRegex(ValueError, "qualified pinned video layout"),
            self.pin(),
            self.spatial(),
        ):
            self.kwargs["video_indices"] = self.kwargs["video_indices"][:-1]
            self.model(**self.kwargs)
        self.assert_restored()

    def test_public_entry_does_not_extrapolate_geometry_or_disable_kv(self):
        self.video = self.video[:, :, :36]
        for reuse in (False, True):
            with (
                torch.inference_mode(),
                self.assertRaisesRegex(ValueError, "qualified five-second"),
                native_audio_context(
                    self.model,
                    self.scheduler,
                    self.video,
                    condition_rows=self.condition,
                    expected_forwards=12,
                    reuse_visual_context=reuse,
                    spatial_stride=2,
                ),
            ):
                self.fail("unqualified input must not enter the model")
            self.assert_restored()


if __name__ == "__main__":
    unittest.main()
