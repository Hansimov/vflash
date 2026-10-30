"""Owned visual K/V reuse for the H3 reference-model audio adapter.

First evaluation builds each layer's visual context. Later evaluations keep
text and audio live; their changes no longer update the cached video rows.
This changes the denoising computation and requires complete audio validation.
The separate fixed-video contract preserves the supplied video latents.
"""

import sys
from contextlib import ExitStack, contextmanager


def _pack_video(video):
    if video.ndim != 5:
        raise ValueError("H3 generated video must have five latent dimensions")
    batch, channels, frames, height, width = video.shape
    if (
        batch != 1
        or channels != 24
        or frames < 1
        or height < 2
        or width < 2
        or height % 2
        or width % 2
    ):
        raise ValueError("H3 generated video has incompatible patch geometry")
    return (
        video.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(batch, -1, channels * 4)
        .contiguous()
    )


@contextmanager
def native_audio_context(
    transformer,
    video_scheduler,
    video,
    *,
    condition_rows,
    expected_forwards,
    reuse_visual_context=False,
    maximum_bytes=40 * 1024**3,
    spatial_stride=1,
):
    """Own one fixed-picture audio pass, optionally reusing visual attention.

    The reference pipeline caller supplies the matched prompt, sampler and
    checkpoint, saves its final native latents, and copies the original video
    bitstream during delivery. False retains full-context computation. A failed
    pass restores state and propagates its error; retries need a fresh generator.
    Spatial stride two or four explicitly reduces spatial audio conditioning
    on five-second I2VA canvases up to two megapixels, with all times retained.
    """
    if type(expected_forwards) is not int or expected_forwards not in (10, 12):
        raise ValueError("This native audio adapter is qualified for 10 or 12 forwards")
    if type(reuse_visual_context) is not bool:
        raise ValueError("Visual context reuse must be explicit")
    if type(spatial_stride) is not int or spatial_stride not in (1, 2, 4):
        raise ValueError("Native audio spatial stride must be 1, 2 or 4")
    # Only the actual five-second, single-first-frame matrix is qualified.
    if spatial_stride > 1 and (
        not reuse_visual_context
        or video.ndim != 5
        or tuple(video.shape[:3]) != (1, 24, 37)
        or max(video.shape[-2:]) * 16 > 2048
        or video.shape[-2] * video.shape[-1] * 256 > 2 * 1024**2
        or condition_rows != (video.shape[-2] // 2) * (video.shape[-1] // 2)
    ):
        raise ValueError(
            "Spatial audio requires qualified five-second I2VA geometry and KV reuse"
        )
    with ExitStack() as stack:
        picture = stack.enter_context(
            fixed_video_audio_pass(
                transformer,
                video_scheduler,
                video,
                condition_rows=condition_rows,
            )
        )
        spatial = None
        if spatial_stride > 1:
            from vflash.adapters.h3_reference_spatial import spatial_video_condition

            spatial = stack.enter_context(
                spatial_video_condition(
                    transformer,
                    height=video.shape[-2] // 2,
                    width=video.shape[-1] // 2,
                    condition_rows=condition_rows,
                    stride=spatial_stride,
                )
            )
        visual = (
            stack.enter_context(
                frozen_visual_kv(
                    transformer,
                    maximum_bytes=maximum_bytes,
                )
            )
            if reuse_visual_context
            else None
        )
        receipt = {
            "picture": picture,
            "visual_context": visual,
            "expected_forwards": expected_forwards,
            "spatial_context": spatial,
        }
        yield receipt
        if (
            picture["forwards"] != expected_forwards
            or picture["pinned_updates"] != expected_forwards
        ):
            raise ValueError("Actual native audio sampling differs from its declared forwards")
        if visual is not None:
            layers = len(transformer.transformer_blocks)
            if visual["build_block_calls"] != layers or visual[
                "reuse_block_calls"
            ] != layers * (expected_forwards - 1):
                raise ValueError("Visual context was not executed for every declared block")
        if spatial is not None and spatial["forwards"] != expected_forwards:
            raise ValueError("Spatial audio context did not cover every declared forward")


@contextmanager
def fixed_video_audio_pass(transformer, video_scheduler, video, *, condition_rows):
    """Keep complete generated picture rows while the caller samples native audio.

    The caller owns this model/scheduler exclusively and uses the original
    image condition and row clock. Text retains its actual per-step time; video
    rows use the existing 0.999 condition row. No source audio is substituted.
    """
    import torch

    fixed = _pack_video(video)
    if (
        type(condition_rows) is not int
        or condition_rows < 1
        or not torch.isfinite(fixed).all()
        or torch.is_grad_enabled()
        or transformer.training
    ):
        raise ValueError("Pinned video requires finite, owned inference input")
    if getattr(transformer, "_vflash_fixed_audio_active", False):
        raise ValueError("This model already owns a pinned-video pass")
    had_step = "step" in video_scheduler.__dict__
    saved_step = video_scheduler.__dict__.get("step")
    record = {
        "forwards": 0,
        "pinned_updates": 0,
        "video_time": 0.999,
        "text_clock": "original per-step video clock",
    }

    def before(_model, _args, kwargs):
        hidden = kwargs["hidden_states"]
        if hidden.shape != (1, condition_rows + fixed.shape[1], 96):
            raise ValueError("Pinned picture differs from the actual video row geometry")
        expected = fixed.to(device=hidden.device, dtype=hidden.dtype)
        if expected.dtype != fixed.dtype:
            raise ValueError("Picture preservation cannot lower its latent precision")
        hidden[:, condition_rows:].copy_(expected)
        clean = (kwargs["timestep"] == 0.999).nonzero().flatten()
        if clean.numel() != 1:
            raise ValueError("Actual compiled clock lacks the image-conditioning row")
        indices = kwargs["timestep_indices"].clone()
        indices[kwargs["video_indices"]] = clean[0]
        kwargs["timestep_indices"] = indices
        record["forwards"] += 1

    def keep_video(model_output, timestep, sample, return_dict=True):
        if return_dict or not torch.equal(sample, fixed[0].to(sample.device)):
            raise ValueError("Pinned video changed or the caller requested another update API")
        record["pinned_updates"] += 1
        return (sample,)

    hook = transformer.register_forward_pre_hook(before, with_kwargs=True)
    transformer._vflash_fixed_audio_active = True
    video_scheduler.step = keep_video
    try:
        yield record
    finally:
        hook.remove()
        del transformer._vflash_fixed_audio_active
        if had_step:
            video_scheduler.step = saved_step
        else:
            del video_scheduler.step


@contextmanager
def frozen_visual_kv(model, *, maximum_bytes=40 * 1024**3):
    import torch

    if torch.is_grad_enabled() or model.training:
        raise ValueError("Visual context reuse requires owned inference in evaluation mode")
    if getattr(model, "sp_world_size", 1) != 1:
        raise ValueError("Visual context reuse supports one GPU")
    if type(maximum_bytes) is not int or not 0 < maximum_bytes <= 40 * 1024**3:
        raise ValueError("Visual context must have a positive limit of at most 40 GiB")
    if getattr(model, "_vflash_visual_context_active", False):
        raise ValueError("This model already owns an active visual context")
    module = sys.modules[type(model.transformer_blocks[0]).__module__]
    owners = []
    caches = {}
    live = None
    geometry = None
    forward_index = -1
    report = {
        "method": "frozen-visual-kv-live-text-audio-v1",
        "approximate": True,
        "build_block_calls": 0,
        "reuse_block_calls": 0,
        "cached_bytes": 0,
        "maximum_bytes": maximum_bytes,
        "live_modalities": ["text", "audio"],
    }

    def before(_model, _args, kwargs):
        nonlocal live, geometry, forward_index
        if kwargs["hidden_states"].shape[0] != 1 or not kwargs["audio_indices"].numel():
            raise ValueError("Visual context requires one audio-bearing sequence")
        video = kwargs["video_indices"]
        next_live = torch.cat((kwargs["text_indices"], kwargs["audio_indices"])).sort().values
        if not (
            kwargs["token_tags"].min() >= 0
            and len(video) + len(next_live) == len(kwargs["token_tags"])
        ):
            raise ValueError("Frozen visual context does not match its owned audio pass")
        if not torch.all(kwargs["timestep"][kwargs["timestep_indices"][video]] == 0.999):
            raise ValueError("Requires pinned-video audio pass")
        if live is None:
            live = next_live
            geometry = len(kwargs["token_tags"])
            if not len(torch.unique(torch.cat((video, live)))) == geometry:
                raise ValueError("Frozen visual context does not match its owned audio pass")
            report.update(total_rows=geometry, live_rows=live.numel(), video_rows=video.numel())
        elif not (geometry == len(kwargs["token_tags"]) and torch.equal(live, next_live)):
            raise ValueError("Frozen visual context does not match its owned audio pass")
        forward_index += 1

    hook = model.register_forward_pre_hook(before, with_kwargs=True)
    model._vflash_visual_context_active = True

    def install(index, block):
        original = block.forward
        processor = block.attn.processor
        had_forward = "forward" in block.__dict__
        saved_forward = block.__dict__.get("forward")
        active = {}
        owners.append((block, processor, had_forward, saved_forward))

        class Processor:
            def __call__(
                self, attn, hidden, rotary_emb=None, attention_mask=None, valid_length=None
            ):
                if not (attention_mask is None and valid_length is None and active):
                    raise ValueError(
                        "Frozen visual context does not match its owned audio pass"
                    )
                query = attn.to_q(hidden).unflatten(-1, (attn.heads, -1))
                key = attn.to_k(hidden).unflatten(-1, (attn.heads, -1))
                value = attn.to_v(hidden).unflatten(-1, (attn.heads, -1))
                query = attn.norm_q(query)
                key = attn.norm_k(key)
                if rotary_emb is not None:
                    query = module.apply_minimax_h3_rotary_emb(query, *rotary_emb)
                    key = module.apply_minimax_h3_rotary_emb(key, *rotary_emb)
                if active["build"]:
                    size = (
                        key.numel() * key.element_size() + value.numel() * value.element_size()
                    )
                    expected = size * len(model.transformer_blocks)
                    if index == 0:
                        limit = maximum_bytes
                        if key.device.type == "cuda":
                            limit = min(
                                limit,
                                max(0, torch.cuda.mem_get_info(key.device)[0] - 12 * 1024**3),
                            )
                        if not expected <= limit:
                            raise ValueError("Visual KV trial exceeds declared memory headroom")
                    caches[index] = {"key": key, "value": value, "step": forward_index}
                    report["cached_bytes"] += size
                    keys, values = (key, value)
                else:
                    saved = caches[index]
                    if not saved["step"] == forward_index - 1:
                        raise ValueError(
                            "Frozen visual context does not match its owned audio pass"
                        )
                    keys, values = (saved["key"], saved["value"])
                    keys.index_copy_(1, live, key)
                    values.index_copy_(1, live, value)
                    saved["step"] = forward_index
                out = (
                    module.attention(query, keys, values, causal=False)
                    .flatten(2, 3)
                    .type_as(query)
                )
                out = attn.to_out[0](out)
                return attn.to_out[1](out)

        def forward(
            hidden, temb, adaln_indices, rotary_emb, attention_mask=None, valid_length=None
        ):
            if not (
                live is not None
                and (not active)
                and (hidden.ndim == 3)
                and (hidden.shape[1] == geometry)
            ):
                raise ValueError("Frozen visual context does not match its owned audio pass")
            build = index not in caches
            active["build"] = build
            try:
                if build:
                    if not forward_index == 0:
                        raise ValueError(
                            "Frozen visual context does not match its owned audio pass"
                        )
                    out = original(
                        hidden, temb, adaln_indices, rotary_emb, attention_mask, valid_length
                    )
                    report["build_block_calls"] += 1
                else:
                    if not (attention_mask is None and valid_length is None):
                        raise ValueError(
                            "Frozen visual context does not match its owned audio pass"
                        )
                    rotary = (
                        tuple(x.index_select(0, live) for x in rotary_emb)
                        if rotary_emb is not None
                        else None
                    )
                    result = original(
                        hidden.index_select(1, live),
                        temb,
                        adaln_indices.index_select(0, live),
                        rotary,
                        None,
                        None,
                    )
                    out = hidden.clone()
                    out.index_copy_(1, live, result)
                    report["reuse_block_calls"] += 1
                return out
            finally:
                active.clear()

        block.forward = forward
        block.attn.set_processor(Processor())

    try:
        for index, block in enumerate(model.transformer_blocks):
            install(index, block)
        yield report
    finally:
        hook.remove()
        del model._vflash_visual_context_active
        for block, processor, had, saved in reversed(owners):
            block.attn.set_processor(processor)
            if had:
                block.forward = saved
            else:
                delattr(block, "forward")
        caches.clear()
