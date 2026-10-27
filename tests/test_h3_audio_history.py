"""History survives native audio refinement, including ring caches and rollback."""

from dataclasses import replace
from types import SimpleNamespace as N

import pytest
from test_h3_refinement import engine_fixture
from test_native_av_continuation import fixture
from test_native_ring import _cpu_block

from vflash.native import h3_native_denoiser as native
from vflash.native.h3_audio_refinement import AudioRefinementRequest, refine_audio
from vflash.native.h3_refinement_schedule import refinement_schedule

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("steps,denoise", [(6, 0.5), (12, 0.9), (20, 0.95)])
@pytest.mark.parametrize("fail", [False, True])
def test_history_refinement_preserves_each_step_and_restores_ring(
    monkeypatch, steps, denoise, fail
):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (32 * 1024**3, 48 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    _, prefix, engine = fixture()
    state = engine.latent_state
    state.evaluation_index = 4
    engine.complete = True
    engine.text_indices = torch.arange(3) + engine.audio_indices[-1] + 1
    state.audio_latents.reshape(2, -1, 32)[:, :65].copy_(prefix.audio.permute(0, 2, 1))
    clean = tuple(
        [torch.full((1, 3, 6, 2), 900.0 + i) for i in range(50)]
        + [torch.full((1, 1, 2, 2), 950.0)]
    )
    prefix = replace(prefix, preserve_video=False, clean_timestep_tables=clean)
    engine.final_layer = engine_fixture().final_layer
    ring = N(
        host_blocks=tuple(_cpu_block(i) for i in range(50)),
        slots=tuple(N(weights=_cpu_block(-1)) for _ in range(2)),
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
    saved, slots = dict(vars(ring)), [s.weights for s in ring.slots]
    plan, final = engine.row_timestep_plan, engine.final_layer.weights
    video, audio = state.video_latents.clone(), state.audio_latents.clone()
    tables = tuple(
        [torch.full((steps + 1, 9, 6, 2), float(i)) for i in range(50)]
        + [torch.zeros(steps + 1, 3, 2, 2)]
    )
    request = AudioRefinementRequest(
        refinement_schedule(steps=steps, denoise=denoise),
        tables,
        context_pixels=None,
        evaluations=steps,
        denoise=denoise,
        weight_cache_bytes=1024**2,
        native_audio_prefix=prefix,
    )
    calls = []

    def step(view, **kwargs):
        current = view.latent_state
        assert torch.equal(
            current.audio_latents.reshape(2, -1, 32)[:, :65], prefix.audio.permute(0, 2, 1)
        )
        for i, pairs in enumerate(ring._ring_copy_pairs):
            native._copy_bf16_tensor_pairs_(pairs)
            weights = ring.slots[i % 2].weights
            assert weights.adaln_table.shape == (steps + 1, 12, 6, 2)
            assert not weights.qkv_residuals
            for n, row in enumerate(view.row_timestep_plan):
                index = len(row.timesteps) - 1
                assert weights.adaln_table[n, index * 3 : (index + 1) * 3].eq(900 + i).all()
                assert (
                    row.timesteps[
                        row.timestep_indices[view.audio_indices.reshape(2, -1)[:, :65]]
                    ]
                    .eq(1)
                    .all()
                )
        calls.append(current.evaluation_index)
        current.step(
            video_velocity=torch.ones_like(current.video_latents),
            audio_velocity=torch.ones_like(current.audio_latents),
        )
        if fail:
            raise RuntimeError("cancelled during history refinement")

    if fail:
        with pytest.raises(RuntimeError, match="cancelled"):
            refine_audio(engine, request, noise=torch.zeros_like(audio), step=step)
        assert torch.equal(state.audio_latents, audio)
    else:
        _, report = refine_audio(engine, request, noise=torch.zeros_like(audio), step=step)
        assert calls == list(range(1, steps + 1))
        assert report["native_audio_prefix"]["audio_prefix_exact"]
        assert report["native_audio_prefix"]["evaluations"] == steps
        assert report["weight_cache"]["cached_blocks"] == 50
        result = state.audio_latents.reshape(2, -1, 32)
        assert torch.equal(result[:, :65], prefix.audio.permute(0, 2, 1))
        assert not torch.equal(result[:, 65:], audio.reshape(2, -1, 32)[:, 65:])
    assert torch.equal(state.video_latents, video)
    assert engine.latent_state is state and engine.row_timestep_plan is plan
    assert engine.final_layer.weights is final
    assert all(getattr(ring, key) is value for key, value in saved.items())
    assert all(
        slot.weights is original for slot, original in zip(ring.slots, slots, strict=True)
    )
    assert not torch.cuda.is_initialized()


def test_history_must_match_retained_sound_before_any_engine_mutation():
    _, prefix, engine = fixture()
    engine.latent_state.evaluation_index = 4
    engine.complete = True
    engine.text_indices = torch.arange(3) + engine.audio_indices[-1] + 1
    before = engine.latent_state.audio_latents.clone()
    tables = tuple([torch.zeros(7, 9, 6, 2) for _ in range(50)] + [torch.zeros(7, 3, 2, 2)])
    with pytest.raises(ValueError, match="audio-only"):
        AudioRefinementRequest(
            refinement_schedule(steps=6, denoise=0.5), tables, native_audio_prefix=prefix
        )
    request = AudioRefinementRequest(
        refinement_schedule(steps=6, denoise=0.5),
        tables,
        context_pixels=None,
        native_audio_prefix=replace(prefix, preserve_video=False),
    )
    with pytest.raises(ValueError, match="differs from the retained"):
        refine_audio(engine, request, noise=torch.zeros_like(before))
    assert torch.equal(engine.latent_state.audio_latents, before)
