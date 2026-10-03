"""Opt-in SelfLift-zero I2VA: six low-resolution and two target-resolution steps.

Each stage owns its engine. VAE round-trip ownership is explicit; no global
sampler patch, downloaded restoration model, frame paste or hidden extra NFE.
"""

import gc
import time
from pathlib import Path

from vflash.adapters.selflift import consistency_lift, sampler
from vflash.adapters.vdn_h3 import VDNEngineSession
from vflash.contracts import ContractError


def sample_stage(session, conditioning, seed, state, *, step_callback=None):
    """Execute the independent schedule with the pinned engine's weight owner."""
    import torch
    from freevideo_engine.keyframes import validate_conditioning
    from freevideo_engine.offload import LayerOffloader
    from src.inference.render import load_prompt

    if not session._lock.acquire(blocking=False):
        raise ContractError("The SelfLift engine is busy")
    engine = session._engine
    try:
        if session._closed or engine.closed:
            raise ContractError("The SelfLift engine is closed")
        canvas = session.canvas
        if session.task != "i2va" or any(
            state[key] != canvas[key] for key in ("width", "height")
        ):
            raise ContractError("SelfLift requires its admitted I2VA canvas")
        prompt, tags, conditions = load_prompt(str(conditioning), "cuda")
        validate_conditioning(
            prompt, tags, conditions, "i2va", canvas["width"], canvas["height"]
        )
        if not conditions or tuple(conditions[0]) != ("first",):
            raise ContractError("SelfLift requires an authoritative first frame")

        class StepTimes(list):
            def append(self, seconds):
                super().append(seconds)
                if step_callback is not None:
                    step_callback(seconds)

        times = StepTimes()
        state["cursor"] = engine.cursor
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        with (
            torch.no_grad(),
            LayerOffloader(
                engine.offload_layers,
                prefetch=engine.prefetch,
                weight_source=engine.stream_weight_source,
            ) as owner,
        ):
            video, audio = sampler(state)(
                engine.transformer,
                prompt,
                tags,
                canvas["frames"],
                8,
                seed,
                "cuda",
                step_seconds=times,
                conditions=conditions,
            )
            offload = owner.stats()
        torch.cuda.synchronize()
        expected = 6 if state["phase"] == "prefix" else 2
        if (
            len(times) != expected
            or not torch.isfinite(video).all()
            or not torch.isfinite(audio).all()
        ):
            raise ContractError("Invalid SelfLift stage result")
        return (
            video.cpu(),
            audio.cpu(),
            dict(
                sample_seconds=time.monotonic() - tick,
                step_seconds=list(times),
                nfe=len(times),
                offload=offload,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            ),
        )
    finally:
        state.pop("cursor", None)
        engine.transformer._freevideo_refined_text = None
        session._lock.release()


def pixel_vae_anchor(low, canvas, official_model: Path, decoder):
    """All-frame floating RGB round trip with deterministic posterior mode."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    from src.inference.render import PIXEL_MEAN, PIXEL_STD
    from torch.nn import functional as F

    from vflash.adapters.conditioning_vae import load_h3_image_conditioning_vae_components
    from vflash.adapters.official_vae import decode_official_h3_video_latents

    decoder.resume_cuda()
    try:
        pixels = decode_official_h3_video_latents(
            decoder.video, low, device=decoder.device, torch_module=torch
        ).cpu()
    finally:
        decoder.suspend_cuda()
    frames = pixels.shape[2]
    images = F.interpolate(
        pixels[0].permute(1, 0, 2, 3),
        size=(canvas["height"], canvas["width"]),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    pixels = images.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    del images
    loaded = load_h3_image_conditioning_vae_components(
        video_component_path=official_model / "vae",
        audio_component_path=official_model / "audio_vae",
        video_class=AutoencoderKLMiniMaxH3,
        audio_class=AutoencoderKLMiniMaxH3Audio,
        device=torch.device("cuda"),
        torch_module=torch,
        init_empty_weights=init_empty_weights,
    )
    vae = loaded.video_vae
    try:
        mean = torch.tensor(PIXEL_MEAN, device="cuda").view(1, 3, 1, 1, 1)
        std = torch.tensor(PIXEL_STD, device="cuda").view(1, 3, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            encoded = vae.encode((pixels.to("cuda") - mean) / std).latent_dist.mode().float()
        lm = encoded.new_tensor(vae.config.latents_mean).view(1, 24, 1, 1, 1)
        ls = encoded.new_tensor(vae.config.latents_std).view(1, 24, 1, 1, 1)
        encoded = ((encoded - lm) / ls).cpu()
        if (
            encoded.shape[2] != low.shape[2]
            or frames != canvas["frames"]
            or not torch.isfinite(encoded).all()
        ):
            raise ContractError("SelfLift VAE anchor has invalid time extent or values")
        return encoded
    finally:
        del vae, loaded, pixels
        gc.collect()
        torch.cuda.empty_cache()


def sample_selflift(
    weights,
    official_model,
    decoder,
    canvas,
    conditioning,
    first_conditioning,
    seed,
    *,
    step_callback=None,
):
    import torch
    from freevideo_engine.geometry import geometry
    from freevideo_engine.two_pass import crop_latents, plan

    if canvas["frames"] != 124 or min(canvas["width"], canvas["height"]) < 640:
        raise ContractError(
            "SelfLift is currently qualified for five-second I2VA, short side >=640"
        )
    selected = plan(canvas, enabled=True, task="i2va")
    low_canvas = geometry(**selected["first"])

    def execute(path, state, stage_canvas, stage_seed):
        tick = time.monotonic()
        session = VDNEngineSession(Path(weights), task="i2va", canvas=stage_canvas)
        load_seconds = time.monotonic() - tick
        try:
            video, audio, report = sample_stage(
                session, path, stage_seed, state, step_callback=step_callback
            )
            return video, audio, dict(report, initialization_seconds=load_seconds)
        finally:
            session.close()
            del session
            gc.collect()
            torch.cuda.empty_cache()

    began = time.monotonic()
    state = dict(phase="prefix", width=low_canvas["width"], height=low_canvas["height"])
    low, audio, prefix = execute(first_conditioning, state, low_canvas, seed)
    tick = time.monotonic()
    pixel = pixel_vae_anchor(
        low, geometry(**selected["upscale_target"]), Path(official_model), decoder
    )
    direct = torch.nn.functional.interpolate(low, size=pixel.shape[2:], mode="nearest")
    pixel, direct = crop_latents(pixel, selected), crop_latents(direct, selected)
    corrected, stats = consistency_lift(direct, pixel, rho=0.6)
    del low, direct, pixel
    anchor_seconds = time.monotonic() - tick
    state = dict(
        phase="suffix",
        width=canvas["width"],
        height=canvas["height"],
        video=corrected,
        audio=audio,
    )
    video, audio, tail = execute(conditioning, state, canvas, seed + 10000)
    return (
        video,
        audio,
        dict(
            strategy="selflift6+2",
            nfe=8,
            prefix=prefix,
            tail=tail,
            rho=0.6,
            **stats,
            anchor_seconds=anchor_seconds,
            elapsed_seconds=time.monotonic() - began,
            sample_seconds=prefix["sample_seconds"] + tail["sample_seconds"],
            sampling_plan=dict(total_steps=8, prefix_steps=6, suffix_steps=2),
            first_pass_conditioning="pixel-encode",
            clock=state["clock"],
        ),
    )
