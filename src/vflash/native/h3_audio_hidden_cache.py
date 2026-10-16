"""Request-owned BF16 pre-projection context reuse for audio-only refinement.

Unlike stored K/V this retains 5376 values per H3 row instead of 14336, but
recomputes QKV projection and key normalization/RoPE on every cached step.
The first evaluation's non-audio context is still an approximation: later
audio updates do not feed back into those rows. It is explicit opt-in and does
not change video latents.
"""

from contextlib import contextmanager
from dataclasses import replace


@contextmanager
def frozen_video_hidden(engine, *, maximum_bytes=40 * 1024**3, tile_rows=16384):
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    state, indices = engine.latent_state, engine.audio_indices
    ring = hasattr(engine.denoiser, "host_blocks")
    blocks = engine.denoiser.slots if ring else engine.denoiser.blocks
    block_count = len(engine.denoiser.host_blocks) if ring else len(blocks)
    if (
        state.num_condition_audio_rows
        or not indices.numel()
        or not blocks
        or (ring and len(blocks) != 2)
        or block_count < len(blocks)
        or state.num_condition_video_rows != state.video_latents.shape[1]
        or type(maximum_bytes) is not int
        or not 0 < maximum_bytes <= 40 * 1024**3
        or type(tile_rows) is not int
        or not 16 <= tile_rows <= 16384
    ):
        raise ValueError("hidden reuse requires bounded audio-only refinement")
    record = dict(
        method="frozen-video-hidden-bf16-v1",
        approximate=True,
        maximum_bytes=maximum_bytes,
        cached_bytes=0,
        build_block_calls=0,
        reuse_block_calls=0,
        rebuilt_projection_rows=0,
        execution_layout="two-slot-ring" if ring else "resident",
        total_blocks=block_count,
    )
    owners, caches = [], {}
    geometry = []

    def install(index, block):
        forward, attention, projection = (
            block.forward_prevalidated,
            block._attention,
            block._qkv_linear,
        )
        names = ("forward_prevalidated", "_attention", "_qkv_linear")
        saved = {name: block.__dict__.get(name) for name in names}
        owners.append((block, saved, set(block.__dict__) & set(names)))
        active = {}

        def project(normalized):
            if not active:
                raise ValueError("hidden cache lost its projection owner")
            start = active["rows"]
            end = start + normalized.shape[1]
            cached = caches[active["index"]]["hidden"]
            if active["build"]:
                if end > cached.shape[1]:
                    raise ValueError("hidden cache projection exceeded the source")
                cached[:, start:end].copy_(normalized)
            else:
                if end > indices.numel():
                    raise ValueError("hidden cache projected extra audio rows")
                cached.index_copy_(1, indices[start:end], normalized)
            active["rows"] = end
            return projection(normalized)

        def attend(query, key, value):
            if active["build"]:
                if active["rows"] != caches[active["index"]]["hidden"].shape[1]:
                    raise ValueError("hidden context was not fully captured")
                return attention(query, key, value)
            if active["rows"] != indices.numel():
                raise ValueError("live audio context was not fully projected")
            invocation = active["invocation"]
            context = caches[active["index"]]["hidden"]
            spec = block.artifact.spec
            shape = (1, context.shape[1], spec.num_attention_heads, spec.attention_head_dim)
            keys, values = (
                torch.empty(shape, device=key.device, dtype=key.dtype),
                torch.empty(shape, device=value.device, dtype=value.dtype),
            )
            for start in range(0, context.shape[1], tile_rows):
                end = min(context.shape[1], start + tile_rows)
                # Call the saved projector so reconstruction cannot rewrite
                # the cache or count as another live-audio projection.
                q, k, v = (
                    part.unflatten(-1, shape[-2:])
                    for part in projection(context[:, start:end]).chunk(3, dim=-1)
                )
                tile = replace(
                    invocation,
                    sequence_length=end - start,
                    adaln_indices=invocation.adaln_indices[start:end],
                    rotary_cos=invocation.rotary_cos[start:end],
                    rotary_sin=invocation.rotary_sin[start:end],
                )
                _, k = block._qk_norm_rotary(q, k, tile)
                keys[:, start:end].copy_(k)
                values[:, start:end].copy_(v)
                record["rebuilt_projection_rows"] += end - start
                del q, k, v, tile
            # Preserve the live audio projection, including any row-tile
            # rounding difference in the rebuilt full-context projection.
            keys.index_copy_(1, indices, key)
            values.index_copy_(1, indices, value)
            args = (query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
            if query.device.type == "cuda":
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION], set_priority=True):
                    output = functional.scaled_dot_product_attention(
                        *args, dropout_p=0.0, is_causal=False
                    )
            else:
                output = functional.scaled_dot_product_attention(
                    *args, dropout_p=0.0, is_causal=False
                )
            record["reuse_block_calls"] += 1
            return output.transpose(1, 2)

        def run(hidden, invocation):
            if (
                active
                or invocation.batch_size != 1
                or hidden.shape[1] != invocation.sequence_length
                or hidden.dtype != torch.bfloat16
            ):
                raise ValueError("hidden cache needs one serial BF16 request")
            # A physical slot carries a different logical layer after each
            # asynchronous weight copy. Cache by that explicit layer index,
            # never by slot number or a method's installation-time weights.
            logical = getattr(block, "_attention_block_index", None) if ring else index
            if type(logical) is not int or not 0 <= logical < block_count:
                raise ValueError("hidden cache lost its logical ring block")
            cached = caches.get(logical)
            build = cached is None
            if build:
                size = hidden.numel() * hidden.element_size()
                if not geometry:
                    capacity = maximum_bytes
                    if hidden.device.type == "cuda":
                        free, _ = torch.cuda.mem_get_info(hidden.device)
                        capacity = min(capacity, max(0, free - 12 * 1024**3))
                    if size * block_count > capacity:
                        raise ValueError("hidden cache exceeds its bytes or device headroom")
                    geometry.append(size)
                    record["sequence_length"] = hidden.shape[1]
                if size != geometry[0]:
                    raise ValueError("hidden cache block geometry changed")
                cached = caches[logical] = dict(
                    hidden=torch.empty_like(hidden), invocation=invocation
                )
                record["cached_bytes"] += size
            else:
                previous = cached["invocation"]
                if (
                    invocation.artifact_id != previous.artifact_id
                    or invocation.sequence_length != previous.sequence_length
                    or invocation.evaluation_index != previous.evaluation_index + 1
                ):
                    raise ValueError("hidden cache cannot cross requests or skip evaluations")
            active.update(build=build, rows=0, invocation=invocation, index=logical)
            try:
                if build:
                    output = forward(hidden, invocation)
                    record["build_block_calls"] += 1
                else:
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
                cached["invocation"] = invocation
                return output
            finally:
                active.clear()

        block.forward_prevalidated, block._attention, block._qkv_linear = run, attend, project

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
