from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vflash.adapters.conditioning_reuse import (
    H3CleanConditioningReuse,
    _CleanEncoding,
    _EncodingKey,
    _RequestEncodingCache,
)
from vflash.pipeline.contracts import ConditioningReuseScope, VideoRequest


class _State:
    def __init__(self):
        self.values = {}

    def get(self, name):
        return self.values[name]

    def set(self, name, value):
        self.values[name] = value


def _fixture(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {"text_encoder": 0, "vae_encoder": 0}

    class MiniMaxH3FL2VATextEncoderStep:
        def __call__(self, components, state):
            calls["text_encoder"] += 1
            state.set(
                "prompt_embeds",
                torch.full((1, 3, 5120), calls["text_encoder"], dtype=torch.bfloat16),
            )
            state.set("text_token_tags", torch.tensor([0, 1, 0], dtype=torch.int64))
            return components, state

    class MiniMaxH3KeyframeVaeEncoderStep:
        def __call__(self, components, state):
            calls["vae_encoder"] += 1
            state.set(
                "condition_latents",
                [
                    torch.full(
                        (1, 24, 1, 4, 4),
                        calls["vae_encoder"],
                        dtype=torch.float32,
                    )
                ],
            )
            return components, state

    module = ModuleType("diffusers.modular_pipelines.minimax_h3.encoders")
    module.MiniMaxH3FL2VATextEncoderStep = MiniMaxH3FL2VATextEncoderStep
    module.MiniMaxH3KeyframeVaeEncoderStep = MiniMaxH3KeyframeVaeEncoderStep
    monkeypatch.setitem(sys.modules, module.__name__, module)
    text_encoder = SimpleNamespace(dtype=torch.bfloat16)
    pipe = SimpleNamespace(
        _blocks=SimpleNamespace(
            sub_blocks={
                "text_encoder": MiniMaxH3FL2VATextEncoderStep(),
                "vae_encoder": MiniMaxH3KeyframeVaeEncoderStep(),
            }
        ),
        keyframe_encode_seed=42,
        text_encoder_layer=50,
        pixel_mean=(0.5, 0.5, 0.5),
        pixel_std=(0.5, 0.5, 0.5),
        canvas_short_edge=768,
        canvas_max_pixels=768 * 1344,
        canvas_multiple=32,
        text_encoder=text_encoder,
        tokenizer=object(),
        processor=object(),
        vae=object(),
    )
    conditioner = SimpleNamespace(
        pipe=pipe,
        prepared=SimpleNamespace(profile_id="i2va-base16-bf16-sm89"),
        versions={"diffusers": "0.40.0", "torch": "2.11.0"},
    )
    components = SimpleNamespace(
        _execution_device=torch.device("cpu"),
        text_encoder=text_encoder,
    )
    reference = SimpleNamespace(
        image=SimpleNamespace(size=(32, 32)),
        sha256="a" * 64,
    )
    request = VideoRequest(
        "The subject moves from <Picture 1>.",
        first_frame=Path("first.png"),
        width=32,
        height=32,
        seed=7,
    )
    return H3CleanConditioningReuse(conditioner), components, reference, request, calls


def _run(reuse, components, reference, request, scope):
    state = _State()
    with reuse.capture(scope, request, (reference,)) as report:
        blocks = reuse.conditioner.pipe._blocks.sub_blocks
        blocks["text_encoder"](components, state)
        blocks["vae_encoder"](components, state)
    return report, state


def test_clean_keyframe_encoding_hits_across_seeds_but_not_scope_or_content(monkeypatch):
    reuse, components, reference, request, calls = _fixture(monkeypatch)
    scope = ConditioningReuseScope("owner-bound-generation-a")

    miss, first = _run(reuse, components, reference, request, scope)
    hit, second = _run(reuse, components, reference, replace(request, seed=19), scope)
    assert miss["status"] == "miss"
    assert hit["status"] == "hit"
    assert hit["encoder_calls"] == {"text_encoder": 0, "vae_encoder": 0}
    assert calls == {"text_encoder": 1, "vae_encoder": 1}
    assert first.get("prompt_embeds").data_ptr() != second.get("prompt_embeds").data_ptr()
    assert first.get("condition_latents")[0].data_ptr() != (
        second.get("condition_latents")[0].data_ptr()
    )

    other_scope, _ = _run(
        reuse,
        components,
        reference,
        replace(request, seed=23),
        ConditioningReuseScope("owner-bound-generation-b"),
    )
    changed_prompt, _ = _run(
        reuse,
        components,
        reference,
        replace(request, prompt="A different action from <Picture 1>.", seed=29),
        ConditioningReuseScope("owner-bound-generation-b"),
    )
    assert other_scope["status"] == changed_prompt["status"] == "miss"
    assert calls == {"text_encoder": 3, "vae_encoder": 3}


def test_failed_partial_capture_clears_the_entry(monkeypatch):
    reuse, components, reference, request, calls = _fixture(monkeypatch)
    scope = ConditioningReuseScope("owner-bound-generation")
    _run(reuse, components, reference, request, scope)

    with (
        pytest.raises(RuntimeError, match="cancelled"),
        reuse.capture(scope, replace(request, seed=17), (reference,)),
    ):
        blocks = reuse.conditioner.pipe._blocks.sub_blocks
        blocks["text_encoder"](components, _State())
        raise RuntimeError("cancelled")
    report, _state = _run(reuse, components, reference, replace(request, seed=23), scope)
    assert report["status"] == "miss"
    assert calls == {"text_encoder": 2, "vae_encoder": 2}


def test_cache_copies_values_and_expires():
    torch = pytest.importorskip("torch")
    now = [10.0]
    cache = _RequestEncodingCache(ttl_seconds=5, clock=lambda: now[0])
    key = _EncodingKey(ConditioningReuseScope("scope"), "a" * 64, 1)
    value = _CleanEncoding(
        torch.ones((1, 2, 5120), dtype=torch.bfloat16),
        torch.zeros(2, dtype=torch.int64),
        (torch.ones((1, 24, 1, 2, 2), dtype=torch.float32),),
    )
    assert cache.put(key, value)
    first = cache.get(key)
    assert first is not None
    first.prompt_embeds.zero_()
    second = cache.get(key)
    assert second is not None and bool((second.prompt_embeds == 1).all())
    now[0] = 16.0
    assert cache.get(key) is None
    assert cache.cached_bytes == 0

    too_small = _RequestEncodingCache(maximum_bytes=1)
    assert not too_small.put(key, value)


def test_default_cache_covers_serial_long_candidates_and_refreshes_on_hit():
    torch = pytest.importorskip("torch")
    now = [10.0]
    cache = _RequestEncodingCache(clock=lambda: now[0])
    key = _EncodingKey(ConditioningReuseScope("long-sibling-scope"), "b" * 64, 1)
    value = _CleanEncoding(
        torch.ones((1, 2, 5120), dtype=torch.bfloat16),
        torch.zeros(2, dtype=torch.int64),
        (torch.ones((1, 24, 1, 2, 2), dtype=torch.float32),),
    )
    assert cache.put(key, value)
    now[0] += 3500.0
    assert cache.get(key) is not None
    now[0] += 3500.0
    assert cache.get(key) is not None
    now[0] += 3601.0
    assert cache.get(key) is None
