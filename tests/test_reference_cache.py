from types import SimpleNamespace

import pytest

from vflash.adapters.reference_cache import ReferenceLatentCache


def reference(identity):
    return SimpleNamespace(
        metadata=dict(
            sha256=identity,
            decoded_rgb_sha256=identity,
            frames=48,
            width=768,
            height=768,
            vae_input_frames=39,
        )
    )


def test_cpu_cache_is_bounded_owned_and_released_after_failure():
    torch = pytest.importorskip("torch")
    value = torch.ones(1, 24, 2, 4, 4)
    size = value.numel() * value.element_size()
    cache = ReferenceLatentCache(maximum_bytes=size)
    with cache.bind(reference("one")):
        cache.put([value])
        value.zero_()
        first = cache.get()[0]
        assert first.sum() == first.numel()
        first.zero_()
        assert cache.get()[0].sum() == first.numel()
        with pytest.raises(ValueError), cache.bind(reference("two")):
            pass
    with pytest.raises(RuntimeError), cache.bind(reference("two")):
        assert cache.get() is None
        cache.put([value])
        raise RuntimeError("caller failure")
    assert cache.key is None and cache.bytes == size
    with cache.bind(reference("one")):
        assert cache.get() is None
    other = ReferenceLatentCache()
    with other.bind(reference("two")):
        assert other.get() is None
    cache.clear()
    assert cache.bytes == 0 and not cache.entries


def test_spatial_pool_retains_each_time_and_full_cache_entry():
    torch = pytest.importorskip("torch")
    from vflash.adapters.reference_cache import spatial_half_latents

    frames = torch.arange(7, dtype=torch.float32)[None, None, :, None, None]
    full = frames.expand(1, 24, 7, 48, 86).contiguous()
    pooled = spatial_half_latents([full])[0]
    assert pooled.shape == (1, 24, 7, 24, 42)
    assert torch.equal(pooled[0, 0, :, 0, 0], torch.arange(7, dtype=torch.float32))
    assert full.shape == (1, 24, 7, 48, 86)
