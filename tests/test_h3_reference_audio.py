"""Owned H3 reference-adapter state and live-modality checks (Torch optional)."""

import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

from vflash.adapters.h3_reference_audio import (
    fixed_video_audio_pass,
    frozen_visual_kv,
    native_audio_context,
)


def attention(query, key, value, *, causal=False):
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=causal,
    ).transpose(1, 2)


def apply_minimax_h3_rotary_emb(value, cosine, sine):
    return value * cosine[None, :, None, :] + value.flip(-1) * sine[None, :, None, :]


class Processor:
    def __call__(self, module, hidden, rotary=None, mask=None, length=None):
        query, key, value = [
            projection(hidden).unflatten(-1, (module.heads, -1))
            for projection in (module.to_q, module.to_k, module.to_v)
        ]
        query, key = module.norm_q(query), module.norm_k(key)
        if rotary is not None:
            query = apply_minimax_h3_rotary_emb(query, *rotary)
            key = apply_minimax_h3_rotary_emb(key, *rotary)
        result = attention(query, key, value).flatten(2, 3)
        return module.to_out[1](module.to_out[0](result))


class Attention(nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.to_q, self.to_k, self.to_v = [nn.Linear(8, 8, bias=False) for _ in range(3)]
        self.to_out = nn.ModuleList([nn.Linear(8, 8, bias=False), nn.Identity()])
        self.norm_q, self.norm_k = nn.Identity(), nn.Identity()
        self.processor = Processor()

    def set_processor(self, processor):
        self.processor = processor

    def forward(self, hidden, rotary, mask=None, length=None):
        return self.processor(self, hidden, rotary, mask, length)


class Block(nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.attn = Attention()

    def forward(self, hidden, temb, indices, rotary, mask=None, length=None):
        return hidden + self.attn(hidden, rotary, mask, length)


class Model(nn.Module if torch is not None else object):
    def __init__(self, layers=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block() for _ in range(layers)])

    def forward(
        self,
        hidden_states,
        *,
        token_tags,
        timestep,
        timestep_indices,
        video_indices,
        audio_indices,
        text_indices,
        rotary,
    ):
        hidden = hidden_states
        for block in self.transformer_blocks:
            hidden = block(hidden, timestep, timestep_indices, rotary)
        return hidden


@unittest.skipIf(torch is None, "H3 adapter execution requires the pinned Torch image")
class AudioContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Model().eval().double()
        self.hidden = torch.randn(1, 10, 8, dtype=torch.float64)
        self.kwargs = dict(
            token_tags=torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 2, 2]),
            timestep=torch.tensor([0.2, 0.999]),
            timestep_indices=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0, 0]),
            video_indices=torch.arange(2, 8),
            text_indices=torch.tensor([0, 1]),
            audio_indices=torch.tensor([8, 9]),
            rotary=(torch.randn(10, 4), torch.randn(10, 4)),
        )

    def assert_restored(self):
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertFalse(hasattr(self.model, "_vflash_visual_context_active"))
        self.assertTrue(all("forward" not in b.__dict__ for b in self.model.transformer_blocks))
        self.assertTrue(
            all(type(b.attn.processor) is Processor for b in self.model.transformer_blocks)
        )

    def test_first_forward_matches_and_a_new_request_rebuilds(self):
        with torch.inference_mode():
            expected = self.model(hidden_states=self.hidden, **self.kwargs)
            for _ in range(2):
                with frozen_visual_kv(self.model) as record:
                    first = self.model(hidden_states=self.hidden, **self.kwargs)
                    torch.testing.assert_close(first, expected, rtol=0, atol=0)
                    self.model(hidden_states=self.hidden, **self.kwargs)
                    self.assertEqual(record["build_block_calls"], 2)
                    self.assertEqual(record["reuse_block_calls"], 2)
                    self.assertEqual(record["live_rows"], 4)
                    self.assertTrue(record["approximate"])
                self.assert_restored()

    def test_text_and_audio_stay_live_with_rotary_positions(self):
        self.model = Model(layers=1).eval().double()
        changed = self.hidden.clone()
        live = torch.tensor([0, 1, 8, 9])
        changed[:, live] += 0.25
        with torch.inference_mode():
            expected = self.model(hidden_states=changed, **self.kwargs)
            with frozen_visual_kv(self.model):
                self.model(hidden_states=self.hidden, **self.kwargs)
                result = self.model(hidden_states=changed, **self.kwargs)
            torch.testing.assert_close(
                result[:, live], expected[:, live], rtol=1e-12, atol=1e-12
            )
        self.assert_restored()

    def test_memory_rejection_restores_the_complete_model(self):
        with torch.inference_mode():
            expected = self.model(hidden_states=self.hidden, **self.kwargs)
            with (
                self.assertRaisesRegex(ValueError, "memory headroom"),
                frozen_visual_kv(self.model, maximum_bytes=1),
            ):
                self.model(hidden_states=self.hidden, **self.kwargs)
            self.assert_restored()
            torch.testing.assert_close(
                self.model(hidden_states=self.hidden, **self.kwargs), expected, rtol=0, atol=0
            )

    def test_unpinned_video_and_nested_owners_are_rejected(self):
        with torch.inference_mode():
            with (
                self.assertRaisesRegex(ValueError, "pinned-video"),
                frozen_visual_kv(self.model),
            ):
                self.model(
                    hidden_states=self.hidden,
                    **{**self.kwargs, "timestep": torch.tensor([0.2, 0.7])},
                )
            self.assert_restored()
            with frozen_visual_kv(self.model):
                with (
                    self.assertRaisesRegex(ValueError, "active visual context"),
                    frozen_visual_kv(self.model),
                ):
                    pass
                self.model(hidden_states=self.hidden, **self.kwargs)
            self.assert_restored()

    def test_training_and_parallel_modes_are_rejected_without_hooks(self):
        with self.assertRaisesRegex(ValueError, "inference"), frozen_visual_kv(self.model):
            pass
        with torch.inference_mode():
            self.model.sp_world_size = 2
            with (
                self.assertRaisesRegex(ValueError, "one GPU"),
                frozen_visual_kv(self.model),
            ):
                pass
        self.assert_restored()

    def test_pinned_picture_keeps_anchor_and_other_modality_clocks(self):
        class Echo(nn.Module):
            def forward(self, **kwargs):
                return kwargs

        class Scheduler:
            def step(self):
                return "original"

        model, scheduler = Echo().eval(), Scheduler()
        source = torch.arange(24 * 3 * 4 * 6).reshape(1, 24, 3, 4, 6).float()
        packed = source.reshape(1, 24, 3, 2, 2, 3, 2).permute(0, 2, 3, 5, 1, 4, 6)
        packed = packed.reshape(1, -1, 96).contiguous()
        count = packed.shape[1] + 2
        hidden = torch.randn(1, count, 96)
        anchors = hidden[:, :2].clone()
        indices = torch.zeros(3 + count + 4, dtype=torch.long)
        indices[-4:] = 1
        original = indices.clone()
        video_indices = torch.arange(3, 3 + count)
        with (
            torch.inference_mode(),
            fixed_video_audio_pass(model, scheduler, source, condition_rows=2) as record,
        ):
            result = model(
                hidden_states=hidden,
                timestep=torch.tensor([0.1, 0.3, 0.999]),
                timestep_indices=indices,
                video_indices=video_indices,
            )
            torch.testing.assert_close(hidden[:, 2:], packed, rtol=0, atol=0)
            torch.testing.assert_close(hidden[:, :2], anchors, rtol=0, atol=0)
            self.assertTrue((result["timestep_indices"][video_indices] == 2).all())
            self.assertTrue(torch.equal(result["timestep_indices"][:3], original[:3]))
            self.assertTrue(torch.equal(result["timestep_indices"][-4:], original[-4:]))
            self.assertTrue(torch.equal(indices, original))
            kept = scheduler.step(None, None, hidden[0, 2:], return_dict=False)[0]
            self.assertEqual(kept.data_ptr(), hidden[0, 2:].data_ptr())
            self.assertEqual(record["forwards"], record["pinned_updates"])
        self.assertEqual(scheduler.step(), "original")
        self.assertNotIn("step", scheduler.__dict__)
        self.assertFalse(model._forward_pre_hooks)

    def test_incomplete_audio_pass_rejects_and_restores_owner(self):
        class Scheduler:
            def step(self):
                return "original"

        scheduler = Scheduler()
        source = torch.zeros(1, 24, 3, 4, 6)
        with (
            torch.inference_mode(),
            self.assertRaisesRegex(ValueError, "declared forwards"),
            native_audio_context(
                self.model,
                scheduler,
                source,
                condition_rows=2,
                expected_forwards=12,
                reuse_visual_context=True,
            ),
        ):
            pass
        self.assert_restored()
        self.assertEqual(scheduler.step(), "original")
        self.assertFalse(hasattr(self.model, "_vflash_fixed_audio_active"))


if __name__ == "__main__":
    unittest.main()
