"""Prepared Pika picture sampling and optional native Larry soundtrack.

This explicit approximate runtime preserves the original six-evaluation sampler.
It is separate from the full-precision Pika reference pipeline.
"""

import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path


@contextmanager
def selected_clock(model, clock):
    """Swap all prepared projections together and restore even after failure."""
    from vflash.adapters.h3_reference_assets import CachedProjection

    modules = [m for m in model.modules() if isinstance(m, CachedProjection)]
    if len(modules) != 51:
        raise ValueError("Expected all 51 original prepared projections")
    prior = [(m, m.clock) for m in modules]
    embedding = model.time_embedder.forward
    try:
        for module, _ in prior:
            module.clock = clock
        model.time_embedder.forward = clock.time_embedding
        yield
    finally:
        model.time_embedder.forward = embedding
        for module, original in prior:
            module.clock = original


@dataclass(frozen=True)
class PikaAVRecipe:
    audio_steps: int = 10
    visual_attention: str = "veda"
    overlap_decode: bool = True
    decode_tile_batch_size: int = 1
    strict_rotary: bool = False

    def __post_init__(self):
        if (
            type(self.audio_steps) is not int
            or self.audio_steps not in {0, 10, 12}
            or self.visual_attention not in {"sdpa", "veda"}
            or type(self.overlap_decode) is not bool
            or type(self.strict_rotary) is not bool
            or type(self.decode_tile_batch_size) is not int
            or self.decode_tile_batch_size not in {1, 2, 4}
        ):
            raise ValueError(
                "Pika AV requires explicit six-picture and zero/ten/twelve-audio settings"
            )

    @property
    def identity(self):
        return f"pika6-larry{self.audio_steps}-v1" if self.audio_steps else "pika-h3-flex-6-v1"


class PikaAVSession:
    """One exclusively owned FL runtime; five-second text or first-frame inputs.

    Source loading and component lifetime belong to PikaAVRuntime. A soundtrack
    pass uses the same prompt and frozen generated picture, without external Foley.
    """

    def __init__(self, runtime, pika_adapter, pika_clock, *, sampling, predictor=None):
        self.runtime, self.pika_adapter, self.pika_clock = runtime, pika_adapter, pika_clock
        self.sampling = sampling
        self.predictor = predictor
        self.owner = threading.Lock()

    def generate(
        self, request, output, *, recipe=None, on_progress=lambda *_: None, video_crop=None
    ):
        import torch
        from PIL import Image
        from safetensors.torch import save_file

        source = self.sampling
        G = source.geometry
        prepare_conditioning = source.numerics.prepare_conditioning
        configure_numerics_backend = source.numerics.configure_numerics_backend
        BF16_CONDITIONING = source.numerics.BF16_CONDITIONING
        video_rows_to_external = source.packing.video_rows_to_external
        audio_external_to_storage = source.packing.audio_external_to_storage
        sample_ladder = source.sampler.sample_ladder
        H3DiffusionWrapper = source.wrapper.H3DiffusionWrapper
        from types import SimpleNamespace

        from vflash.adapters.h3_reference_audio import native_audio_context
        from vflash.adapters.h3_reference_execution import dense_reference_execution
        from vflash.media.overlap import AudioVideoOverlap
        from vflash.native.h3_latent_layout import (
            unpack_h3_audio_rows,
            unpatchify_h3_video_rows,
        )
        from vflash.pipeline.residency import staged_text_conditioning

        recipe = recipe or PikaAVRecipe()
        if not isinstance(recipe, PikaAVRecipe):
            raise ValueError("A frozen Pika AV recipe is required")
        audio_steps, attention, overlap_enabled = (
            recipe.audio_steps,
            recipe.visual_attention,
            recipe.overlap_decode,
        )
        batched_decode = recipe.decode_tile_batch_size > 1
        if (
            not request.generate_audio
            or request.keyframe_delivery_profile != "decoded"
            or self.runtime.closed
            or self.runtime.pipe.transformer.training
            or request.mode not in {"t2va", "i2va"}
            or request.duration_seconds != 5
            or request.width % 32
            or request.height % 32
            or min(request.width, request.height) < 32
            or request.width * request.height > 2 * 1024**2
            or (batched_decode and request.width * request.height > 960 * 544)
            or audio_steps not in {0, 10, 12}
            or attention not in {"sdpa", "veda"}
            or type(overlap_enabled) is not bool
        ):
            raise ValueError("Explicit bounded Pika six and optional Larry ten/twelve required")
        from vflash.media.canvas import crop_geometry

        crop_geometry(request.width, request.height, video_crop)
        if not self.owner.acquire(blocking=False):
            raise ValueError("The mixed runtime already owns an active request")
        rt = self.runtime
        pipe = rt.pipe
        model = pipe.transformer
        output = Path(output)
        original_encode, original_keyframes = pipe.encode_prompt, pipe.encode_keyframes
        schedulers = (pipe.scheduler, pipe.audio_scheduler)
        strengths = (rt.adapter.strength, self.pika_adapter.strength)
        original_attention = rt.attention.backend
        original_flags = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        image = None
        condition_rows = None
        prompt_cache = []
        events = []
        started = time.monotonic()
        total = 6 + audio_steps
        spatial = request.width // 32 * (request.height // 32) if request.mode == "i2va" else 0
        conditioning = {
            "prompt_calls": 0,
            "prompt_reuses": 0,
            "keyframe_calls": 0,
            "keyframe_reuses": 0,
        }
        execution = {}
        decode_scope_owner = ExitStack()
        decode_batches = {"limit": 1}

        def encode(prompt, *args, **kwargs):
            if prompt != request.prompt:
                raise ValueError("Audio prompt differs from frozen picture prompt")
            if prompt_cache:
                conditioning["prompt_reuses"] += 1
                return tuple(v.clone() for v in prompt_cache[0])
            conditioning["prompt_calls"] += 1
            value = original_encode(prompt, *args, **kwargs)
            prompt_cache.append(tuple(v.detach().clone() for v in value))
            return value

        def encode_frames(images, *args, **kwargs):
            nonlocal condition_rows
            if image is None or len(images) != 1 or images[0].size != image.size:
                raise ValueError("Audio changed the prepared keyframe allocation")
            if condition_rows is not None:
                conditioning["keyframe_reuses"] += 1
                return condition_rows.clone()
            conditioning["keyframe_calls"] += 1
            condition_rows = original_keyframes(images, *args, **kwargs).detach().clone()
            return condition_rows.clone()

        try:
            if recipe.decode_tile_batch_size > 1:
                from vflash.adapters.h3_decode_batches import batch_spatial_tiles

                decode_batches = decode_scope_owner.enter_context(
                    batch_spatial_tiles(
                        rt.decoder.video.module.model, recipe.decode_tile_batch_size
                    )
                )
            output.mkdir(parents=True, exist_ok=True)
            if (output / "native.mp4").exists():
                raise ValueError("New owned output required")
            if request.mode == "i2va":
                with Image.open(request.first_frame) as original_image:
                    image = original_image.convert("RGB")
                if image.size != (request.width, request.height):
                    raise ValueError("Prepare exact canvas keyframe")
            pipe.encode_prompt, pipe.encode_keyframes = encode, encode_frames
            configure_numerics_backend(BF16_CONDITIONING, runtime="inference")
            on_progress("conditioning", 0, 1)
            began = time.monotonic()
            with (
                staged_text_conditioning(
                    getattr(pipe, "text_encoder", None),
                    master=getattr(rt, "text_cpu_master", None),
                    enabled=getattr(rt, "text_encoder_residency", "resident") == "staged",
                ) as text_residency,
                torch.inference_mode(),
            ):
                embeddings, tags = encode(request.prompt, images=[image] if image else None)
                geom = SimpleNamespace(
                    num_latent_frames=37,
                    latent_height=request.height // 16,
                    latent_width=request.width // 16,
                    num_audio_latents=G.audio_latent_num_frames(124),
                )
                cond = {
                    "prompt_embeds": embeddings,
                    "text_token_tags": tags,
                    "geometry": geom,
                    "keyframes": [],
                    "references": [],
                }
                if image:
                    rows = encode_frames([image])
                    cond["keyframes"] = [
                        {
                            "anchor": "first",
                            "latents": video_rows_to_external(
                                rows[None], 1, geom.latent_height, geom.latent_width
                            ),
                        }
                    ]
                cond = prepare_conditioning(cond, "cuda:0", policy=BF16_CONDITIONING)
            torch.cuda.synchronize()
            conditioning["seconds"] = time.monotonic() - began
            conditioning["text_residency"] = text_residency
            on_progress("conditioning", 1, 1)
            on_progress("denoising", 0, total)
            self.pika_clock.select_mode(request.mode)
            rt.adapter.strength = 0
            self.pika_adapter.strength = 1
            rt.attention.select(attention)
            rt.attention.reset()
            if attention == "veda":
                rt.attention.configure(
                    model, {"width": request.width, "height": request.height}, self.predictor
                )
                rt.attention.profile.num_condition_video_rows = spatial
            adapter_before = dict(self.pika_adapter.calls)

            def after(model, args, result):
                self.pika_clock.after(model, args, result)
                on_progress("denoising", self.pika_clock.completed_forwards, total)

            execution = {}
            with torch.inference_mode():
                with dense_reference_execution(
                    model, rotary=recipe.strict_rotary
                ) as visual_execution:
                    began = time.monotonic()
                    with selected_clock(model, self.pika_clock):
                        handles = [
                            model.register_forward_pre_hook(
                                self.pika_clock.before, with_kwargs=True
                            ),
                            model.register_forward_hook(after),
                        ]
                        try:
                            wrapper = H3DiffusionWrapper(model).eval()
                            v, a = sample_ladder(
                                wrapper, cond, seed=request.seed, device="cuda:0"
                            )
                        finally:
                            for handle in handles:
                                handle.remove()
                    torch.cuda.synchronize()
                    if self.pika_clock.completed_forwards != 6:
                        raise ValueError("Actual Pika forward count differs")
                    if any(
                        self.pika_adapter.calls[k] - v != 6 for k, v in adapter_before.items()
                    ):
                        raise ValueError("Not every Pika target executed in all six steps")
                    if attention == "veda":
                        rt.attention.veda.validate_calls(6)
                    visual = {
                        "video_latents": v.permute(0, 2, 1, 3, 4).contiguous().cpu(),
                        "audio_latents": audio_external_to_storage(a[0]).contiguous().cpu(),
                    }
                    events.append(
                        {
                            "role": "pika_joint_visual",
                            "forwards": 6,
                            "seconds": time.monotonic() - began,
                            "attention": attention,
                            "adapter": "original-pika-fl-r128-alpha128",
                            "clock": "original-renoise-six",
                            "weight_format": "unmerged-row-fp8",
                        }
                    )
                    visual_execution["live_bytes_before_release"] = (
                        torch.cuda.memory_allocated()
                    )
                    del v, a, cond, wrapper
                execution["visual"] = visual_execution
                execution["after_visual_live_bytes"] = torch.cuda.memory_allocated()
                execution["after_visual_reserved_bytes"] = torch.cuda.memory_reserved()
                if attention == "veda":
                    events[0]["attention_policy"] = rt.attention.veda.metadata()
                    rt.attention.veda.engine.reset()
                    rt.attention.veda.layout = rt.attention.veda.choice = None
                save_file(visual, str(output / "visual.safetensors"))
                self.pika_adapter.strength = 0
                final = visual
                decode_scope = (
                    AudioVideoOverlap(rt.decoder, width=request.width, height=request.height)
                    if audio_steps and overlap_enabled
                    else nullcontext(None)
                )
                with (
                    dense_reference_execution(
                        model, qkv=False, rotary=recipe.strict_rotary
                    ) as audio_execution,
                    decode_scope as overlap,
                ):
                    if overlap:
                        overlap.submit(visual["video_latents"])
                    if audio_steps:
                        rt.adapter.strength = 1
                        rt.attention.select("sdpa")
                        rt.attention.reset()
                        rt.clock.select(audio_steps, 12.0)
                        pipe.scheduler = type(schedulers[0])(shift=12.0)
                        pipe.audio_scheduler = type(schedulers[1])(shift=3.0)
                        captured = {}
                        count = 0

                        def count_forward(*_):
                            nonlocal count
                            count += 1

                        def callback(p, index, t, values):
                            rt.clock.step = index + 1
                            on_progress("denoising", 6 + index + 1, total)
                            if index + 1 == audio_steps:
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
                            model.register_forward_pre_hook(rt.clock.before, with_kwargs=True),
                            model.register_forward_hook(count_forward),
                        ]
                        began = time.monotonic()
                        try:
                            with native_audio_context(
                                model,
                                pipe.scheduler,
                                visual["video_latents"],
                                condition_rows=spatial,
                                expected_forwards=audio_steps,
                                reuse_visual_context=True,
                                spatial_stride=2,
                            ) as context:
                                pipe(
                                    prompt=request.prompt,
                                    image=image,
                                    height=request.height,
                                    width=request.width,
                                    num_frames=124,
                                    num_inference_steps=audio_steps,
                                    flow_shift=12.0,
                                    audio_flow_shift=3.0,
                                    guidance_scale=1.0,
                                    generator=torch.Generator(device="cuda:0").manual_seed(
                                        request.seed
                                    ),
                                    output_type="latent",
                                    callback_on_step_end=callback,
                                    callback_on_step_end_tensor_inputs=[
                                        "latents",
                                        "audio_latents",
                                    ],
                                )
                        finally:
                            for handle in handles:
                                handle.remove()
                        torch.cuda.synchronize()
                        if count != audio_steps or not torch.equal(
                            captured["video_latents"], visual["video_latents"]
                        ):
                            raise ValueError("Audio count or frozen picture differs")
                        final = captured
                        events.append(
                            {
                                "role": "larry_audio",
                                "forwards": count,
                                "seconds": time.monotonic() - began,
                                "context": context,
                                "attention": "sdpa",
                            }
                        )
                    predecoded = overlap.finish() if overlap else None
                execution["audio"] = audio_execution
            save_file(final, str(output / "native.safetensors"))
            on_progress("decoding", 0, 1)
            media = rt.decoder.generate_mp4(
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
                **({"video_crop": video_crop} if video_crop is not None else {}),
            )
            on_progress("decoding", 1, 1)
            return {
                "recipe": recipe.identity,
                "source_revision": source.source_revision,
                "weight_format": "unmerged-row-fp8",
                "stages": events,
                "conditioning": conditioning,
                "execution": execution,
                "overlap": dict(overlap.record) if overlap else {"applied": False},
                "decode_tile_batches": dict(decode_batches),
                "media_stage_seconds": media.stage_durations,
                "media": media.media,
                "actual_forwards": total,
                "video_preserved_during_audio": True,
                "seconds": time.monotonic() - started,
                "path": str(output / "native.mp4"),
            }
        finally:
            decode_scope_owner.close()
            pipe.encode_prompt, pipe.encode_keyframes = original_encode, original_keyframes
            pipe.scheduler, pipe.audio_scheduler = schedulers
            rt.adapter.strength, self.pika_adapter.strength = strengths
            rt.attention.select(original_attention)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                original_flags
            )
            prompt_cache.clear()
            self.owner.release()
