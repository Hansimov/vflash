"""Owned segmented H3 reference sampling with native fixed-picture audio.

The caller supplies the original FL pipeline, matching Larry activation-space
adapter, original time tables, attention selector and official media decoder.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentedAVRecipe:
    visual_steps: int = 10
    adapted_visual_steps: int = 4
    audio_steps: int = 12
    visual_attention: str = "veda"
    spatial_audio_stride: int = 2

    def __post_init__(self):
        if (
            self.visual_steps != 10
            or self.adapted_visual_steps != 4
            or self.audio_steps != 12
            or self.visual_attention not in {"sage", "veda"}
            or self.spatial_audio_stride not in {1, 2, 4}
        ):
            raise ValueError("Segmented AV requires 4+6 picture and 12 audio evaluations")


class SegmentedAVSession:
    """An exclusive five-second I2VA owner over caller-loaded reference components.

    The pipeline follows the pinned VideoX H3 callable contract. The clock supplies
    original FL time projections and validates actual timestamps. The adapter has
    a mutable strength; attention implements select/reset/configure over that model.
    Loading and source provenance stay with the caller's explicit asset owner.
    """

    def __init__(self, pipeline, clock, adapter, attention, decoder, *, predictor: Path):
        self.pipeline = pipeline
        self.clock = clock
        self.adapter = adapter
        self.attention = attention
        self.decoder = decoder
        self.predictor = predictor
        self._owner = threading.Lock()

    def generate(self, request, output: Path, *, recipe=None, on_progress=lambda *_: None):
        import torch
        from PIL import Image
        from safetensors.torch import save_file

        from vflash.adapters.h3_reference_audio import native_audio_context
        from vflash.adapters.h3_reference_execution import dense_reference_execution
        from vflash.media.overlap import AudioVideoOverlap
        from vflash.native.h3_latent_layout import (
            unpack_h3_audio_rows,
            unpatchify_h3_video_rows,
        )

        recipe = recipe or SegmentedAVRecipe(
            spatial_audio_stride=4 if request.width * request.height > 1376 * 768 else 2
        )
        if (
            request.mode != "i2va"
            or request.duration_seconds != 5
            or not request.generate_audio
            or min(request.width, request.height) < 32
            or max(request.width, request.height) > 2048
            or request.width % 32
            or request.height % 32
            or request.width * request.height > 2 * 1024**2
            or self.pipeline.transformer.training
        ):
            raise ValueError(
                "Reference AV requires qualified five-second I2VA geometry and audio"
            )
        if not self._owner.acquire(blocking=False):
            raise ValueError("This reference session already owns an active request")
        pipe, model = self.pipeline, self.pipeline.transformer
        original_encode = pipe.encode_prompt
        original_schedulers = (pipe.scheduler, pipe.audio_scheduler)
        original_strength = self.adapter.strength
        original_attention = self.attention.backend
        raw_condition, stages = [], []
        conditioning = {"calls": 0, "reuses": 0, "seconds": 0.0}
        started = time.monotonic()

        def encode(prompt, *args, **kwargs):
            if prompt != request.prompt:
                raise ValueError("The native audio pass changed the frozen scene prompt")
            if raw_condition:
                conditioning["reuses"] += 1
                return tuple(value.clone() for value in raw_condition[0])
            before = time.monotonic()
            result = original_encode(prompt, *args, **kwargs)
            conditioning["seconds"] += time.monotonic() - before
            conditioning["calls"] += 1
            on_progress("conditioning", 1, 1)
            raw_condition.append(tuple(value.detach().clone() for value in result))
            return result

        def sample(steps, adapted, backend, *, source=None):
            self.adapter.strength = 0
            self.attention.select(backend)
            self.attention.reset()
            self.attention.configure(
                model, {"width": request.width, "height": request.height}, self.predictor
            )
            self.clock.select(steps, 12.0)
            self.clock.step = 0
            pipe.scheduler = type(original_schedulers[0])(shift=12.0)
            pipe.audio_scheduler = type(original_schedulers[1])(shift=3.0)
            spatial = request.width // 32 * (request.height // 32)
            captured = {}
            actual_forwards = 0
            cuda_starts, cuda_ends = [], []
            role = "audio" if source is not None else "visual"

            def before(*_):
                self.adapter.strength = float(self.clock.step < adapted)
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                cuda_starts.append(event)

            def after(*_):
                nonlocal actual_forwards
                actual_forwards += 1
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                cuda_ends.append(event)

            def callback(_pipe, index, _time, values):
                self.clock.step = index + 1
                on_progress(role, index + 1, steps)
                if index + 1 == steps:
                    captured["video_latents"] = unpatchify_h3_video_rows(
                        values["latents"][spatial:][None].float().cpu(),
                        num_latent_frames=37,
                        latent_height=request.height // 16,
                        latent_width=request.width // 16,
                    )
                    captured["audio_latents"] = unpack_h3_audio_rows(
                        values["audio_latents"][None].float().cpu()
                    )
                return {}

            handles = [
                model.register_forward_pre_hook(self.clock.before, with_kwargs=True),
                model.register_forward_pre_hook(before),
                model.register_forward_hook(after),
            ]
            scope = (
                native_audio_context(
                    model,
                    pipe.scheduler,
                    source,
                    condition_rows=spatial,
                    expected_forwards=steps,
                    reuse_visual_context=True,
                    spatial_stride=recipe.spatial_audio_stride,
                )
                if source is not None
                else nullcontext(None)
            )
            phase_started = time.monotonic()
            try:
                with torch.inference_mode(), scope as context:
                    pipe(
                        prompt=request.prompt,
                        image=image,
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
            finally:
                for handle in reversed(handles):
                    handle.remove()
            if actual_forwards != steps:
                raise ValueError("Actual model evaluations differ from the declared recipe")
            if backend == "veda":
                self.attention.veda.validate_calls(steps)
            if source is not None and not torch.equal(captured["video_latents"], source):
                raise ValueError("Native audio generation changed the selected picture")
            torch.cuda.synchronize()
            stages.append(
                {
                    "role": role,
                    "steps": steps,
                    "adapted_steps": adapted,
                    "attention": backend,
                    "attention_policy": dict(self.attention.veda.metadata())
                    if backend == "veda"
                    else None,
                    "seconds": time.monotonic() - phase_started,
                    "context": context,
                    "transformer_cuda_ms": [
                        a.elapsed_time(b) for a, b in zip(cuda_starts, cuda_ends, strict=True)
                    ],
                }
            )
            return captured

        try:
            output.mkdir(parents=True, exist_ok=True)
            with Image.open(request.first_frame) as opened:
                image = opened.convert("RGB")
            # Reference graphs can stretch the first image; callers must prepare
            # an aspect-preserving image on the exact canvas before this boundary.
            if image.size != (request.width, request.height):
                raise ValueError("Prepare an aspect-preserving first frame on the exact canvas")
            torch.cuda.reset_peak_memory_stats()
            pipe.encode_prompt = encode
            with dense_reference_execution(model) as execution:
                visual = sample(
                    recipe.visual_steps, recipe.adapted_visual_steps, recipe.visual_attention
                )
                save_file(visual, str(output / "visual.safetensors"))
                torch.cuda.empty_cache()
                decode_scope = (
                    AudioVideoOverlap(self.decoder, width=request.width, height=request.height)
                    if request.width * request.height <= 1376 * 768
                    else nullcontext(None)
                )
                with decode_scope as overlap:
                    if overlap is not None:
                        overlap.submit(visual["video_latents"])
                    final = sample(
                        recipe.audio_steps,
                        recipe.audio_steps,
                        "sdpa",
                        source=visual["video_latents"],
                    )
                    predecoded = overlap.finish() if overlap is not None else None
            save_file(final, str(output / "native.safetensors"))
            on_progress("decoding", 0, 1)
            media = self.decoder.generate_mp4(
                output / "native.safetensors",
                output / "native.mp4",
                width=request.width,
                height=request.height,
                duration_seconds=5,
                fps=24,
                audio_delivery_profile=request.audio_delivery_profile,
                video_input="pipe",
                video_threads=16,
                predecoded_video=predecoded,
            )
            on_progress("decoding", 1, 1)
            return {
                "path": output / "native.mp4",
                "stages": stages,
                "conditioning": conditioning,
                "execution": execution,
                "attention_policy": stages[0]["attention_policy"],
                "overlap": dict(overlap.record) if overlap is not None else {"enabled": False},
                "media_stage_seconds": media.stage_durations,
                "media": media.media,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "seconds": time.monotonic() - started,
                "actual_forwards": 22,
                "video_preserved_during_audio": True,
            }
        finally:
            pipe.encode_prompt = original_encode
            pipe.scheduler, pipe.audio_scheduler = original_schedulers
            self.adapter.strength = original_strength
            self.attention.select(original_attention)
            raw_condition.clear()
            self._owner.release()
