"""CPU ownership/copy-pair checks; real asynchronous execution needs GPU evidence."""

from types import SimpleNamespace as N

import pytest
from test_h3_audio_refinement import context_engine, fake_attention
from test_native_ring import _cpu_block

from vflash.native import h3_audio_refinement as audio
from vflash.native import h3_native_denoiser as native
from vflash.native.h3_refinement_schedule import refinement_schedule

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("fail", [False, True])
def test_ring_audio_rebinds_all_copy_pairs_without_turbo_and_restores_owner(monkeypatch, fail):
    monkeypatch.setattr(audio, "_attention_view", fake_attention)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    engine = context_engine()
    attention = engine.denoiser.blocks[0]._veda_attention
    ring = N(
        host_blocks=tuple(_cpu_block(i) for i in range(50)),
        slots=tuple(N(weights=_cpu_block(-1), _veda_attention=attention) for _ in range(2)),
        device=torch.device("cpu"),
    )
    ring._ring_copy_pairs = tuple(
        native._bf16_block_copy_pairs(ring.slots[i % 2].weights, w)
        for i, w in enumerate(ring.host_blocks)
    )
    ring._serial_copy_pairs = tuple(
        native._bf16_block_copy_pairs(ring.slots[0].weights, w) for w in ring.host_blocks
    )
    ring._block_weight_bytes = tuple(native._block_tensor_bytes(w) for w in ring.host_blocks)
    ring.host_weight_bytes = sum(ring._block_weight_bytes)
    ring.device_slot_weight_bytes = sum(
        native._block_tensor_bytes(s.weights) for s in ring.slots
    )
    engine.denoiser = ring
    saved = dict(vars(ring))
    slot_weights = [s.weights for s in ring.slots]
    state = engine.latent_state
    video = state.video_latents.clone()
    original_audio = state.audio_latents.clone()
    tables = tuple(
        [torch.full((7, 9, 6, 2), float(i)) for i in range(50)] + [torch.zeros(7, 3, 2, 2)]
    )
    request = audio.AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.5), tables)
    evaluations = []

    def step(view, **kwargs):
        assert view.denoiser is ring and view.latent_state.video_latents.shape[1] == 1452
        assert all(not w.qkv_residuals and w.ffn_out_residual is None for w in ring.host_blocks)
        # The real validated tensor-copy pairs must select all 50 new tables;
        # changing host metadata alone leaves the old four-step copy addresses.
        for index, pairs in enumerate(ring._ring_copy_pairs):
            native._copy_bf16_tensor_pairs_(pairs)
            slot = ring.slots[index % 2]
            assert torch.equal(slot.weights.adaln_table, tables[index])
            assert (
                not slot.weights.qkv_residuals and slot.weights.attention_out_residual is None
            )
        assert ring.slots[0]._veda_attention is not attention
        evaluations.append(view.latent_state.evaluation_index)
        view.latent_state.step(
            video_velocity=torch.ones_like(view.latent_state.video_latents),
            audio_velocity=torch.ones_like(view.latent_state.audio_latents),
        )
        if fail:
            raise RuntimeError("injected streamed audio failure")

    if fail:
        with pytest.raises(RuntimeError, match="streamed audio"):
            audio.refine_audio(
                engine, request, noise=torch.zeros_like(original_audio), step=step
            )
        assert torch.equal(state.audio_latents, original_audio)
    else:
        _, record = audio.refine_audio(
            engine, request, noise=torch.zeros_like(original_audio), step=step
        )
        assert evaluations == [1, 2, 3, 4, 5, 6] and record["video_latents_identical"]
        assert not torch.equal(state.audio_latents, original_audio)
    assert engine.latent_state is state and state.evaluation_index == 4
    assert torch.equal(state.video_latents, video)
    for key, value in saved.items():
        assert getattr(ring, key) is value
    assert all(
        s.weights is w and s._veda_attention is attention
        for s, w in zip(ring.slots, slot_weights, strict=True)
    )
    assert (attention.sparse_calls, attention.dense_calls) == (160, 40)
    assert not torch.cuda.is_initialized()
