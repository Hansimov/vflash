from dataclasses import replace
from types import SimpleNamespace as N

import pytest

from vflash.native.h3_audio_hidden_cache import frozen_video_hidden

torch = pytest.importorskip("torch")
F = torch.nn.functional


def core_fixture(whole_rows, evaluations=6):
    from contextlib import nullcontext

    from vflash.native.h3_native_denoiser import H3NativeBlockInvocation, _H3BlockOperations
    from vflash.native.h3_tiled_block import tiled_forward

    class CoreBlock(_H3BlockOperations):
        def _linear(self, states, weight):
            return F.linear(states, weight)

        def _attention(self, q, k, v):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)

        def forward_prevalidated(self, hidden, invocation):
            return tiled_forward(
                self,
                hidden,
                invocation,
                rows=1024,
                measure=lambda _: nullcontext(),
                whole_rows=whole_rows,
            )

    torch.manual_seed(816)

    def random(*shape):
        return (torch.randn(*shape) * 0.1).bfloat16()

    blocks = []
    for _ in range(3):
        table = random(evaluations, 2, 6, 8)
        table[:, 0] = table[:1, 0]
        weights = N(
            adaln_table=table,
            qkv=random(36, 8),
            qkv_residuals=(),
            attention_out=random(8, 12),
            attention_out_residual=None,
            ffn_in=random(32, 8),
            ffn_in_residual=None,
            ffn_out=random(8, 16),
            ffn_out_residual=None,
            attention_norm=torch.ones(8).bfloat16(),
            ffn_norm=torch.ones(8).bfloat16(),
            query_norm=torch.ones(4).bfloat16(),
            key_norm=torch.ones(4).bfloat16(),
        )
        blocks.append(
            CoreBlock(
                N(
                    artifact_id="real-core-hidden-test",
                    spec=N(num_attention_heads=3, attention_head_dim=4),
                ),
                weights,
            )
        )
    hidden = torch.randn(1, 17, 8).bfloat16()
    audio = torch.tensor([2, 5, 8])
    mods = torch.zeros(17, dtype=torch.int64)
    mods[audio] = 1
    angles = torch.rand(17, 4)
    invocation = H3NativeBlockInvocation(
        artifact_id="real-core-hidden-test",
        batch_size=1,
        sequence_length=17,
        hidden_size=8,
        device=hidden.device,
        dtype=hidden.dtype,
        evaluation_index=0,
        adaln_indices=mods,
        rotary_cos=angles.cos(),
        rotary_sin=angles.sin(),
    )
    engine = N(
        denoiser=N(blocks=blocks),
        audio_indices=audio,
        latent_state=N(
            num_condition_audio_rows=0,
            num_condition_video_rows=14,
            video_latents=torch.zeros(1, 14, 1),
        ),
    )
    return engine, hidden, invocation


def test_hidden_cache_budget_and_owner_failure_restore_existing_methods():
    engine, hidden, invocation = core_fixture(0)
    block = engine.denoiser.blocks[0]
    original = block._qkv_linear
    block._qkv_linear = original
    with (
        pytest.raises(ValueError, match="headroom"),
        frozen_video_hidden(engine, maximum_bytes=1),
    ):
        block.forward_prevalidated(hidden, invocation)
    assert block._qkv_linear is original and "_attention" not in block.__dict__
    with pytest.raises(ValueError, match="skip evaluations"), frozen_video_hidden(engine):
        block.forward_prevalidated(hidden, invocation)
        block.forward_prevalidated(hidden, replace(invocation, evaluation_index=3))
    assert block._qkv_linear is original and "_attention" not in block.__dict__


@pytest.mark.parametrize("evaluations", [6, 20])
def test_ring_slots_cache_logical_layers_and_match_resident_hidden_reuse(evaluations):
    engine, hidden, invocation = core_fixture(0, evaluations)
    weights = tuple(block.weights for block in engine.denoiser.blocks)

    def run(blocks, ring=False):
        result = []
        for evaluation in range(evaluations):
            states = hidden.clone()
            states[:, engine.audio_indices] += evaluation * 0.125
            for index in range(len(weights)):
                block = blocks[index % 2] if ring else blocks[index]
                if ring:
                    block.weights = weights[index]
                    block._attention_block_index = index
                states = block.forward_prevalidated(
                    states, replace(invocation, evaluation_index=evaluation)
                )
            result.append(states)
        return result

    with frozen_video_hidden(engine, tile_rows=16) as resident:
        expected = run(engine.denoiser.blocks)
    slots = engine.denoiser.blocks[:2]
    engine.denoiser = N(slots=slots, host_blocks=weights)
    with frozen_video_hidden(engine, tile_rows=16) as ring:
        actual = run(slots, ring=True)
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ring["execution_layout"] == "two-slot-ring" and ring["total_blocks"] == 3
    assert ring["cached_bytes"] == resident["cached_bytes"]
    assert ring["build_block_calls"] == 3 and ring["reuse_block_calls"] == 3 * (evaluations - 1)
    assert all(
        not set(b.__dict__) & {"forward_prevalidated", "_attention", "_qkv_linear"}
        for b in slots
    )


def test_ring_cache_requires_current_layer_identity_and_restores_on_failure():
    engine, hidden, invocation = core_fixture(0)
    engine.denoiser = N(
        slots=engine.denoiser.blocks[:2],
        host_blocks=tuple(b.weights for b in engine.denoiser.blocks),
    )
    with pytest.raises(ValueError, match="logical ring block"), frozen_video_hidden(engine):
        engine.denoiser.slots[0].forward_prevalidated(hidden, invocation)
    assert all(
        not set(b.__dict__) & {"forward_prevalidated", "_attention", "_qkv_linear"}
        for b in engine.denoiser.slots
    )
