"""Prepared Pika Ref-picture and separate FL/Larry native-sound session."""

import threading
import time
from pathlib import Path

from vflash.pipeline.pika_reference_recipe import PikaReferenceRecipe


class PikaReferenceAVSession:
    def __init__(self, runtime, *, predictor):
        self.runtime, self.predictor = runtime, predictor
        self.owner = threading.Lock()

    def generate(
        self, request, output, *, recipe=None, on_progress=lambda *_: None, video_crop=None
    ):
        import torch
        from safetensors.torch import save_file

        from vflash.adapters.h3_reference_execution import dense_reference_execution
        from vflash.adapters.pika_reference_conditioning import reference_conditioning
        from vflash.media.canvas import crop_geometry
        from vflash.pipeline.larry_soundtrack import frozen_larry_soundtrack

        rt = self.runtime
        recipe = recipe or PikaReferenceRecipe(
            reference_short_edge=getattr(rt, "reference_short_edge", None)
        )
        if (
            type(recipe) is not PikaReferenceRecipe
            or recipe.reference_short_edge != getattr(rt, "reference_short_edge", None)
            or recipe.overlap_decode
            or recipe.decode_tile_batch_size != 1
            or rt.closed
            or request.mode != "ref2va"
            or request.reference_video is not None
            or not 1 <= len(request.ordered_references) <= 9
            or request.duration_seconds != 5
            or request.width * request.height > 1024**2
            or not request.generate_audio
            or request.keyframe_delivery_profile != "decoded"
        ):
            raise ValueError(
                "Explicit image-reference Pika candidate within five seconds / 1 MP required"
            )
        crop_geometry(request.width, request.height, video_crop)
        if not self.owner.acquire(blocking=False):
            raise ValueError("The reference runtime already owns a request")
        source = rt.sampling
        output = Path(output)
        flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        strengths = rt.adapter.strength, rt.reference_adapter.strength
        backend = rt.attention.backend
        handles = []
        total = 6 + recipe.audio_steps
        stages, residency = [], []
        started = time.monotonic()
        try:
            output.mkdir(parents=True, exist_ok=True)
            if (output / "native.mp4").exists():
                raise ValueError("New owned output required")
            source.numerics.configure_numerics_backend(
                source.numerics.FP32_CONDITIONING, runtime="inference"
            )
            on_progress("conditioning", 0, 1)
            began = time.monotonic()
            prepared = reference_conditioning(
                rt.pipe,
                request,
                source=source,
                conditioning=rt.conditioning_source,
                reference_type=rt.family.pipeline.MiniMaxH3ImageReference,
                reference_short_edge=getattr(rt, "reference_short_edge", None),
            )
            audio_embeddings = audio_tags = None
            if recipe.audio_steps:
                with torch.inference_mode():
                    audio_embeddings, audio_tags = rt.pipe.encode_prompt(
                        request.prompt, device="cuda:0", dtype=torch.bfloat16
                    )
            torch.cuda.synchronize()
            conditioning = {
                **prepared.metadata,
                "seconds": time.monotonic() - began,
                "reference_prompt_calls": 1,
                "audio_text_prompt_calls": int(recipe.audio_steps > 0),
            }
            on_progress("conditioning", 1, 1)
            residency.append(rt.select_transformer("ref2va"))
            model = rt.reference_model
            rt.reference_clock.select_mode("ref2va")
            rt.adapter.strength = 0
            rt.reference_adapter.strength = 1
            rt.attention.select(recipe.visual_attention)
            rt.attention.reset()
            if recipe.visual_attention == "veda":
                rt.attention.configure(
                    model, {"width": request.width, "height": request.height}, self.predictor
                )
                rt.attention.profile.num_condition_video_rows = prepared.metadata[
                    "reference_video_rows"
                ]
            adapter_before = dict(rt.reference_adapter.calls)

            def after(model, args, result):
                rt.reference_clock.after(model, args, result)
                on_progress("denoising", rt.reference_clock.completed_forwards, total)

            on_progress("denoising", 0, total)
            began = time.monotonic()
            with (
                torch.inference_mode(),
                dense_reference_execution(model, rotary=recipe.strict_rotary) as execution,
            ):
                handles = [
                    model.register_forward_pre_hook(
                        rt.reference_clock.before, with_kwargs=True
                    ),
                    model.register_forward_hook(after),
                ]
                try:
                    wrapper = source.wrapper.H3DiffusionWrapper(model).eval()
                    video, audio = source.sampler.sample_ladder(
                        wrapper, prepared.conditional_dict, seed=request.seed, device="cuda:0"
                    )
                finally:
                    for handle in handles:
                        handle.remove()
                    handles = []
                torch.cuda.synchronize()
                if rt.reference_clock.completed_forwards != 6 or any(
                    rt.reference_adapter.calls[k] - value != 6
                    for k, value in adapter_before.items()
                ):
                    raise ValueError("Original Ref adapter targets or six forwards differ")
                visual = {
                    "video_latents": video.permute(0, 2, 1, 3, 4).contiguous().cpu(),
                    "audio_latents": source.packing.audio_external_to_storage(audio[0])
                    .contiguous()
                    .cpu(),
                }
                del video, audio, wrapper
                stage = {
                    "role": "pika_joint_visual",
                    "partition": "ref2va",
                    "forwards": 6,
                    "adapter": "original-pika-ref-r128-alpha128",
                    "numerics_policy": source.numerics.FP32_CONDITIONING,
                    "attention": recipe.visual_attention,
                    "execution": execution,
                    "seconds": time.monotonic() - began,
                }
            if recipe.visual_attention == "veda":
                rt.attention.veda.validate_calls(6)
                stage["attention_policy"] = rt.attention.veda.metadata()
                rt.attention.veda.engine.reset()
                rt.attention.veda.layout = rt.attention.veda.choice = None
            stages.append(stage)
            del prepared
            save_file(visual, str(output / "visual.safetensors"))
            rt.reference_adapter.strength = 0
            final = visual
            if recipe.audio_steps:
                residency.append(rt.select_transformer("fl2va"))
                final, sound = frozen_larry_soundtrack(
                    rt,
                    request,
                    visual,
                    embeddings=audio_embeddings,
                    tags=audio_tags,
                    steps=recipe.audio_steps,
                    on_progress=on_progress,
                    strict_rotary=recipe.strict_rotary,
                )
                stages.append(sound)
            rt.suspend_transformers()
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
                **({"video_crop": video_crop} if video_crop is not None else {}),
            )
            on_progress("decoding", 1, 1)
            return {
                "recipe": recipe.identity,
                "request_mode": "ref2va",
                "source_revision": source.source_revision,
                "weight_format": "unmerged-row-fp8",
                "actual_forwards": total,
                "stages": stages,
                "conditioning": conditioning,
                "transformer_residency": residency,
                "overlap": {"applied": False},
                "decode_tile_batches": {"limit": 1},
                "video_preserved_during_audio": True,
                "media_stage_seconds": media.stage_durations,
                "media": media.media,
                "seconds": time.monotonic() - started,
                "path": str(output / "native.mp4"),
            }
        finally:
            try:
                for handle in handles:
                    handle.remove()
                rt.adapter.strength, rt.reference_adapter.strength = strengths
                rt.attention.select(backend)
                rt.suspend_transformers()
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
                self.owner.release()
