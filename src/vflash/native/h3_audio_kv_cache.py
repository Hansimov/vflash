"""Bounded post-RoPE video K/V reuse for an owned audio-only quality pass."""

from contextlib import contextmanager
from dataclasses import replace


@contextmanager
def frozen_video_kv(engine, *, maximum_bytes=40 * 1024**3):
    """One audio-only pass; approximate context feedback, never a shared cache.

    Build all blocks normally once, preserving their post-RoPE K/V. Later steps
    compute only audio rows against that context, replacing live audio K/V.
    Frozen non-audio context retains the first evaluation while audio queries
    and audio K/V update on every step. This explicit approximation preserves
    the request-owned six-step schedule and releases its cache on every exit.
    """
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    state = engine.latent_state
    indices = engine.audio_indices
    if (
        state.num_condition_audio_rows
        or not indices.numel()
        or state.num_condition_video_rows != state.video_latents.shape[1]
        or type(maximum_bytes) is not int
        or not 0 < maximum_bytes <= 40 * 1024**3
    ):
        raise ValueError("KV reuse requires bounded audio-only refinement")
    ring = hasattr(engine.denoiser, "host_blocks")
    blocks = engine.denoiser.slots if ring else engine.denoiser.blocks
    block_count = len(engine.denoiser.host_blocks) if ring else len(blocks)
    if not blocks or (ring and len(blocks) != 2) or block_count < len(blocks):
        raise ValueError("KV reuse requires a complete native block layout")
    record = {
        "method": "frozen-video-kv-bf16-v1",
        "approximate": True,
        "maximum_bytes": maximum_bytes,
        "cached_bytes": 0,
        "build_block_calls": 0,
        "reuse_block_calls": 0,
        "execution_layout": "two-slot-ring" if ring else "resident",
        "total_blocks": block_count,
        "rebuilt_projection_rows": 0,
    }
    owners = []
    caches = {}
    plan = {}

    def install(index, block):
        forward, attention = block.forward_prevalidated, block._attention
        saved = {
            name: block.__dict__.get(name) for name in ("forward_prevalidated", "_attention")
        }
        owned = set(block.__dict__) & set(saved)
        owners.append((block, saved, owned))
        current = [None]

        def attend(query, key, value):
            owned_invocation = current[0]
            if owned_invocation is None:
                raise ValueError("audio cache lost its block invocation")
            logical, invocation = owned_invocation
            cached = caches.get(logical)
            if cached is None:
                size = (key.numel() + value.numel()) * key.element_size()
                if not plan:
                    capacity = maximum_bytes
                    if key.device.type == "cuda":
                        free, _ = torch.cuda.mem_get_info(key.device)
                        capacity = min(capacity, max(0, free - 12 * 1024**3))
                    count = block_count
                    expected = size * count
                    if size * block_count > maximum_bytes:
                        raise ValueError(
                            f"frozen-video KV cache exceeds its byte limit: "
                            f"{size * block_count} required, {maximum_bytes} allowed"
                        )
                    if not count or expected > capacity:
                        raise ValueError(
                            "frozen-video KV cache leaves insufficient device headroom"
                        )
                    plan.update(first=block_count - count, bytes_per_block=size)
                    record["sequence_length"] = invocation.sequence_length
                if size != plan["bytes_per_block"]:
                    raise ValueError("audio cache block geometry changed")
                caches[logical] = (key.clone(), value.clone(), invocation)
                record["cached_bytes"] += size
                record["build_block_calls"] += 1
                return attention(query, key, value)
            keys, values, _ = cached
            # Storage is request-owned. Only the live audio slice changes;
            # cached non-audio rows remain exactly the first evaluation's K/V.
            keys.index_copy_(1, indices, key)
            values.index_copy_(1, indices, value)
            if query.device.type == "cuda":
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION], set_priority=True):
                    output = functional.scaled_dot_product_attention(
                        query.transpose(1, 2),
                        keys.transpose(1, 2),
                        values.transpose(1, 2),
                        dropout_p=0.0,
                        is_causal=False,
                    )
            else:
                output = functional.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    keys.transpose(1, 2),
                    values.transpose(1, 2),
                    dropout_p=0.0,
                    is_causal=False,
                )
            record["reuse_block_calls"] += 1
            return output.transpose(1, 2)

        def run(hidden, invocation):
            if invocation.batch_size != 1 or invocation.sequence_length != hidden.shape[1]:
                raise ValueError("audio cache requires one complete owned request")
            if current[0] is not None:
                raise ValueError("KV cache requires serial block ownership")
            logical = getattr(block, "_attention_block_index", None) if ring else index
            if type(logical) is not int or not 0 <= logical < block_count:
                raise ValueError("KV cache lost its logical ring block")
            cached = caches.get(logical)
            current[0] = (logical, invocation)
            try:
                if cached is None:
                    return forward(hidden, invocation)
                previous = cached[2]
                if (
                    invocation.artifact_id != previous.artifact_id
                    or invocation.sequence_length != previous.sequence_length
                    or invocation.evaluation_index != previous.evaluation_index + 1
                ):
                    raise ValueError("audio cache cannot cross requests or skip evaluations")
                selected = replace(
                    invocation,
                    sequence_length=indices.numel(),
                    adaln_indices=invocation.adaln_indices.index_select(0, indices),
                    rotary_cos=invocation.rotary_cos.index_select(0, indices),
                    rotary_sin=invocation.rotary_sin.index_select(0, indices),
                )
                result = forward(hidden.index_select(1, indices), selected)
                output = hidden.clone()
                output.index_copy_(1, indices, result)
                caches[logical] = (cached[0], cached[1], invocation)
                return output
            finally:
                current[0] = None

        block.forward_prevalidated, block._attention = run, attend

    try:
        for index, block in enumerate(blocks):
            install(index, block)
        yield record
    finally:
        for block, saved, owned in reversed(owners):
            for name, value in saved.items():
                if name in owned:
                    setattr(block, name, value)
                else:
                    delattr(block, name)
        caches.clear()
