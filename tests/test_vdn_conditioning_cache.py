from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from PIL import Image

from vflash.pipeline.contracts import ConditioningReuseScope
from vflash.pipeline.vdn_conditioning_cache import (
    VDNCleanConditioningCache as ConditioningCache,
)

SIBLING_SCOPE = ConditioningReuseScope("same-accepted-generation")


@dataclass
class Request:
    first_frame: Path
    last_frame: Path | None = None
    mode: str = "i2va"
    prompt: str = "A bird opens its wings."
    width: int = 16
    height: int = 16
    seed: int = 1


@pytest.fixture
def case(tmp_path):
    first, last = tmp_path / "first.png", tmp_path / "last.png"
    Image.new("RGB", (32, 32), "red").save(first)
    Image.new("RGB", (32, 32), "blue").save(last)
    calls = []

    def encoder(model, request, plan, directory):
        calls.append(request)
        # These stand in for immutable serialized CPU embeddings/latents.
        (directory / "target.pt").write_bytes(b"target conditioning")
        (directory / "first-reference.png").write_bytes(request.first_frame.read_bytes())
        if request.mode == "fl2va":
            (directory / "last-reference.png").write_bytes(request.last_frame.read_bytes())
        if plan["enabled"]:
            (directory / "first.pt").write_bytes(b"first pass conditioning")
        return dict(
            elapsed_seconds=30,
            text_seconds=28,
            text_encoder_reused=False,
            token_refiner_reused=False,
        )

    def run(
        cache,
        request=None,
        *,
        model=None,
        plan=None,
        scope=SIBLING_SCOPE,
    ):
        directory = tmp_path / f"attempt-{len(list(tmp_path.glob('attempt-*')))}"
        directory.mkdir()
        report = cache.encode(
            model or tmp_path / "model",
            request or Request(first),
            plan or {"enabled": True, "first": {"width": 8, "height": 8}},
            directory,
            encoder,
            scope=scope,
        )
        return report, directory

    return run, calls, Request(first), last


def test_seed_only_reuses_clean_inputs_and_materializes_all_files(case):
    run, calls, request, _ = case
    cache = ConditioningCache()
    first, a = run(cache, request)
    second, b = run(cache, replace(request, seed=719))
    assert not first["conditioning_cache_hit"] and second["conditioning_cache_hit"]
    assert len(calls) == 1 and second["text_seconds"] == 0
    assert second["text_embedding_reused"] and not second["token_refiner_reused"]
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }


@pytest.mark.parametrize("change", ["prompt", "pixels", "canvas", "mode", "plan", "model"])
def test_all_semantic_inputs_invalidate(case, change):
    run, calls, request, last = case
    cache = ConditioningCache()
    run(cache, request)
    kwargs = {}
    if change == "prompt":
        request = replace(request, prompt="A bird closes its wings.")
    elif change == "pixels":
        Image.new("RGB", (32, 32), "green").save(request.first_frame)
    elif change == "canvas":
        request = replace(request, width=24)
    elif change == "mode":
        request = replace(request, mode="fl2va", last_frame=last)
    elif change == "plan":
        kwargs["plan"] = {"enabled": False}
    else:
        kwargs["model"] = last.parent / "other-model"
    report, _ = run(cache, request, **kwargs)
    assert not report["conditioning_cache_hit"] and len(calls) == 2


def test_ordered_fl_keyframes_and_changed_input_path(case):
    run, calls, request, last = case
    cache = ConditioningCache()
    request = replace(request, mode="fl2va", last_frame=last)
    run(cache, request)
    alias = last.parent / "same-first.png"
    alias.write_bytes(request.first_frame.read_bytes())
    hit, folder = run(cache, replace(request, first_frame=alias, seed=3))
    assert hit["conditioning_cache_hit"] and (folder / "last-reference.png").is_file()
    miss, _ = run(cache, replace(request, first_frame=last, last_frame=request.first_frame))
    assert not miss["conditioning_cache_hit"] and len(calls) == 2


def test_capacity_eviction_oversized_bypass_and_explicit_clear(case):
    run, calls, request, _ = case
    cache = ConditioningCache(max_entries=1)
    run(cache, request)
    run(cache, replace(request, prompt="Different scene."))
    miss, _ = run(cache, request)
    assert not miss["conditioning_cache_hit"] and len(calls) == 3
    assert 0 < cache.retained_bytes <= cache.max_bytes
    cache.clear()
    assert cache.retained_bytes == 0
    tiny = ConditioningCache(max_bytes=1)
    run(tiny, request)
    miss, _ = run(tiny, request)
    assert not miss["conditioning_cache_hit"] and tiny.retained_bytes == 0


def test_no_scope_disables_reuse_and_scope_change_discards_previous_entry(case):
    run, calls, request, _ = case
    cache = ConditioningCache()
    run(cache, request)
    off, _ = run(cache, request, scope=None)
    assert not off["conditioning_cache_hit"] and cache.retained_bytes == 0
    run(cache, request, scope=ConditioningReuseScope("new-generation"))
    old, _ = run(cache, request)
    assert not old["conditioning_cache_hit"] and len(calls) == 4


def test_ttl_and_clear_remove_retained_user_inputs(case):
    run, calls, request, _ = case
    now = [0.0]
    cache = ConditioningCache(ttl_seconds=10, clock=lambda: now[0])
    run(cache, request)
    assert cache.retained_bytes > 0
    now[0] = 11.0
    assert cache.retained_bytes == 0
    miss, _ = run(cache, request)
    assert not miss["conditioning_cache_hit"] and len(calls) == 2
    cache.clear()
    assert cache.retained_bytes == 0


def test_invalid_scope_fails_before_encoding(case):
    run, calls, request, _ = case
    with pytest.raises(ValueError, match="scope"):
        run(ConditioningCache(), request, scope="not-a-typed-scope")
    assert not calls
