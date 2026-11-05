"""Native FL/Larry audio over an already generated, immutable video latent."""

import time


def frozen_larry_soundtrack(
    runtime,
    request,
    visual,
    *,
    embeddings,
    tags,
    steps,
    on_progress=lambda *_: None,
    strict_rotary=False,
):
    import torch

    from vflash.adapters.h3_reference_audio import native_audio_context
    from vflash.adapters.h3_reference_execution import dense_reference_execution
    from vflash.native.h3_latent_layout import unpack_h3_audio_rows, unpatchify_h3_video_rows

    if (
        steps not in {10, 12}
        or type(steps) is not int
        or runtime.active_partition != "fl2va"
        or runtime.pipe.transformer is not runtime.fl_model
    ):
        raise ValueError("The native soundtrack requires its separate FL transformer")
    pipe, model = runtime.pipe, runtime.fl_model
    original_encode = pipe.encode_prompt
    schedulers = pipe.scheduler, pipe.audio_scheduler
    strength = runtime.adapter.strength
    backend = runtime.attention.backend
    handles, captured = [], {}
    count = 0
    reuse_count = 0

    def encode(prompt, *args, **kwargs):
        nonlocal reuse_count
        if prompt != request.prompt or kwargs.get("images") or kwargs.get("references"):
            raise ValueError("FL audio must retain its exact text-only conditioning")
        reuse_count += 1
        return embeddings.clone(), tags.clone()

    def count_forward(*_):
        nonlocal count
        count += 1

    def callback(p, index, t, values):
        runtime.clock.step = index + 1
        on_progress("denoising", 6 + index + 1, 6 + steps)
        if index + 1 == steps:
            captured["video_latents"] = unpatchify_h3_video_rows(
                values["latents"][None].float().cpu(),
                num_latent_frames=37,
                latent_height=request.height // 16,
                latent_width=request.width // 16,
            )
            captured["audio_latents"] = unpack_h3_audio_rows(
                values["audio_latents"][None].float().cpu()
            )
        return {}

    try:
        pipe.encode_prompt = encode
        runtime.adapter.strength = 1
        runtime.attention.select("sdpa")
        runtime.attention.reset()
        runtime.clock.select(steps, 12.0)
        pipe.scheduler = type(schedulers[0])(shift=12.0)
        pipe.audio_scheduler = type(schedulers[1])(shift=3.0)
        handles = [
            model.register_forward_pre_hook(runtime.clock.before, with_kwargs=True),
            model.register_forward_hook(count_forward),
        ]
        began = time.monotonic()
        with (
            torch.inference_mode(),
            dense_reference_execution(model, qkv=False, rotary=strict_rotary) as execution,
            native_audio_context(
                model,
                pipe.scheduler,
                visual["video_latents"],
                condition_rows=0,
                expected_forwards=steps,
                reuse_visual_context=True,
                spatial_stride=2,
            ) as context,
        ):
            pipe(
                prompt=request.prompt,
                image=None,
                height=request.height,
                width=request.width,
                num_frames=124,
                num_inference_steps=steps,
                flow_shift=12.0,
                audio_flow_shift=3.0,
                guidance_scale=1.0,
                generator=torch.Generator(device="cuda:0").manual_seed(request.seed),
                output_type="latent",
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents", "audio_latents"],
            )
        torch.cuda.synchronize()
        if (
            count != steps
            or reuse_count != 1
            or "video_latents" not in captured
            or "audio_latents" not in captured
            or not torch.equal(captured["video_latents"], visual["video_latents"])
        ):
            raise ValueError("Native audio changed the picture, conditioning or forward count")
        return captured, {
            "role": "larry_audio",
            "partition": "fl2va",
            "forwards": count,
            "seconds": time.monotonic() - began,
            "context": context,
            "attention": "sdpa",
            "execution": execution,
            "prompt_reuses": reuse_count,
            "picture_preserved": True,
        }
    finally:
        for handle in handles:
            handle.remove()
        pipe.encode_prompt = original_encode
        pipe.scheduler, pipe.audio_scheduler = schedulers
        runtime.adapter.strength = strength
        runtime.attention.select(backend)
