"""Opt-in complete keyframe-to-video pipeline for the pinned VDN8 backend.

Stage ownership is sequential: official multimodal encoding, clean VAE anchors,
trained hybrid sampling, then the existing Vflash official media decoder. No
product credentials, routing or downloaded assets are part of this module.
"""

from __future__ import annotations

import gc
import os
import tempfile
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.adapters.vdn_h3 import VDNEngineSession, first_pass_reference
from vflash.contracts import ContractError
from vflash.pipeline.contracts import PipelineProgress, VideoRequest, VideoResult


@dataclass(frozen=True)
class VDNAssets:
    official_model: Path
    weights: Path
    decoder: Path
    upscaler_checkpoint: Path | None = None

    def validate(self, two_pass: bool) -> None:
        for path in (self.official_model, self.weights, self.decoder):
            if not isinstance(path, Path) or not path.is_dir():
                raise ContractError("VDN requires explicit existing local model directories")
        if two_pass and (
            not isinstance(self.upscaler_checkpoint, Path)
            or not self.upscaler_checkpoint.is_file()
        ):
            raise ContractError("Two-pass generation requires an explicit local upscaler")


def encode_conditioning(
    model: Path, request: VideoRequest, plan: dict, directory: Path
) -> dict:
    """Use raw layer-50 text and ordered clean anchors; never another LoRA's cache."""
    import numpy as np
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from freevideo_engine.keyframes import validate_conditioning
    from PIL import Image, ImageOps
    from src.inference.encode_keyframes import build_presentation, qwen3vl_prompt_embeds
    from src.inference.render import PIXEL_MEAN, PIXEL_STD
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    started = time.monotonic()
    anchors = ["first"] if request.mode == "i2va" else ["first", "last"]
    paths = [request.first_frame] + ([request.last_frame] if request.mode == "fl2va" else [])
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(
                ImageOps.fit(
                    image.convert("RGB"),
                    (request.width, request.height),
                    method=Image.Resampling.LANCZOS,
                )
            )
    for anchor, image in zip(anchors, images, strict=True):
        image.save(directory / f"{anchor}-reference.png")
    processor = Qwen3VLProcessor.from_pretrained(
        model, subfolder="processor", local_files_only=True
    )
    tokens, tags, vision = build_presentation(processor, request.prompt, images)
    encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "36GiB", "cpu": "120GiB"},
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
    )
    try:
        with torch.no_grad():
            text = qwen3vl_prompt_embeds(encoder, processor, tokens, vision, "cuda").cpu()
    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    del processor, vision
    text_seconds = time.monotonic() - started
    tags = torch.tensor(tags, dtype=torch.long)
    vae = (
        AutoencoderKLMiniMaxH3.from_pretrained(model, subfolder="vae", local_files_only=True)
        .eval()
        .requires_grad_(False)
        .to("cuda")
    )
    try:
        stages = [("target", dict(width=request.width, height=request.height), images)]
        if plan["enabled"]:
            stages.append(
                (
                    "first",
                    plan["first"],
                    [first_pass_reference(image, plan) for image in images],
                )
            )
        for name, canvas, refs in stages:
            conditions = []
            for image in refs:
                pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None]
                with torch.no_grad():
                    conditions.append(
                        encode_vae_condition(
                            vae, pixels.to("cuda"), PIXEL_MEAN, PIXEL_STD, 42
                        ).cpu()
                    )
            validate_conditioning(
                text,
                tags,
                (anchors, conditions),
                request.mode,
                canvas["width"],
                canvas["height"],
            )
            torch.save(
                dict(
                    prompt_embeds=text,
                    text_token_tags=tags,
                    keyframe_anchors=anchors,
                    condition_latents=conditions,
                ),
                directory / f"{name}.pt",
            )
    finally:
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    return dict(
        elapsed_seconds=time.monotonic() - started,
        text_seconds=text_seconds,
        text_encoder_reused=False,
        token_refiner_reused=False,
        anchors=len(anchors),
        target_geometry="cover-center-lanczos-rgb-v1",
    )


def upscale_bf16(video: Any, checkpoint: Path, width: int, height: int) -> tuple[Any, dict]:
    """Explicit BF16 LBH v1 variant; no upstream hash/download hooks."""
    import torch
    from freevideo_engine.latent_upscale import LATENTS_MEAN, LATENTS_STD, LatentResizer3D
    from safetensors.torch import load_file

    scale = width / (video.shape[-1] * 16)
    if height != round(video.shape[-2] * 16 * scale) or not 1 <= scale <= 4:
        raise ContractError("Upscaler geometry must preserve aspect ratio")
    if scale == 1:
        return video, dict(scale=1.0, load_seconds=0.0, compute_seconds=0.0)
    started = time.monotonic()
    with torch.device("meta"):
        model = LatentResizer3D()
    model.load_state_dict(load_file(checkpoint), assign=True, strict=True)
    model = model.eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    loaded = time.monotonic() - started
    tick = time.monotonic()
    try:
        with torch.no_grad():
            samples = video.to(device="cuda", dtype=torch.bfloat16)
            mean = samples.new_tensor(LATENTS_MEAN).view(1, 24, 1, 1, 1)
            std = samples.new_tensor(LATENTS_STD).view(1, 24, 1, 1, 1)
            result = model(
                (samples - mean) / std,
                scale=scale,
                target_size=(video.shape[2], height // 16, width // 16),
                enable_chunking=False,
            )
            result = (result * std + mean).float()
        torch.cuda.synchronize()
        if not torch.isfinite(result).all():
            raise ContractError("Upscaler produced nonfinite output")
        return result, dict(
            load_seconds=loaded,
            compute_seconds=time.monotonic() - tick,
            scale=scale,
            precision="bf16",
            checkpoint_transform="extra_per_channel_v1",
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


class VDNKeyframePipeline:
    """Serial complete keyframe generation with an explicit sampling strategy.

    This initial complete adapter reloads encoding and sampling for each request;
    it does not pretend to be a preloaded Base16 pipeline. The CPU media decoder
    is retained between requests. No automatic fallback or hidden output resize.
    """

    def __init__(
        self, assets: VDNAssets, *, strategy: str = "full8", trust_local_code: bool = False
    ) -> None:
        if trust_local_code is not True:
            raise ContractError("Official media components require trust_local_code=True")
        if strategy not in {"full8", "pixel8+2", "selflift6+2"}:
            raise ContractError("VDN strategy must be full8, pixel8+2 or selflift6+2")
        assets.validate(strategy == "pixel8+2")
        self.assets, self.strategy = assets, strategy
        self._lock, self._closed, self._decoder = threading.Lock(), False, None

    def generate(
        self, request: VideoRequest, output_path: Path, *, progress=None
    ) -> VideoResult:
        if not isinstance(request, VideoRequest) or request.mode not in {"i2va", "fl2va"}:
            raise ContractError("VDN complete generation requires I2VA or true FL2VA")
        progressive = self.strategy == "selflift6+2"
        if progressive and (
            request.mode != "i2va"
            or request.duration_seconds != 5
            or min(request.width, request.height) < 640
        ):
            raise ContractError("SelfLift currently accepts five-second I2VA, short side >=640")
        if (
            not isinstance(output_path, Path)
            or output_path.exists()
            or output_path.is_symlink()
        ):
            raise ContractError("Output must be a new local MP4 path")
        if not self._lock.acquire(blocking=False):
            raise ContractError("The VDN pipeline is busy")
        began = time.monotonic()
        try:
            if self._closed:
                raise ContractError("The VDN pipeline is closed")
            from freevideo_engine.paths import add_vdn

            add_vdn()
            import torch
            from freevideo_engine.geometry import geometry
            from freevideo_engine.two_pass import plan
            from safetensors.torch import save_file

            from vflash.media.runtime import OfficialMediaDecoder

            if torch.cuda.get_device_capability() != (8, 9):
                raise ContractError("This complete VDN adapter requires one SM89 GPU")
            torch.set_num_threads(4)
            canvas = geometry(request.width, request.height, frames=request.model_frames)
            two_pass = self.strategy != "full8"
            selected = plan(canvas, enabled=two_pass, task=request.mode)
            if progressive:
                selected["total_steps"] = 8
            output_path.parent.mkdir(parents=True, exist_ok=True)

            def emit(stage, completed=0, total=1):
                if progress:
                    progress(PipelineProgress(stage, completed, total))

            with tempfile.TemporaryDirectory(
                prefix="vflash-vdn-", dir=output_path.parent
            ) as temp:
                directory = Path(temp)
                emit("conditioning")
                encoding = encode_conditioning(
                    self.assets.official_model, request, selected, directory
                )
                emit("conditioning", 1)
                tick = time.monotonic()
                engine = (
                    None
                    if progressive
                    else VDNEngineSession(self.assets.weights, task=request.mode, canvas=canvas)
                )
                initialization = time.monotonic() - tick
                steps = 0

                def on_step(_seconds):
                    nonlocal steps
                    steps += 1
                    emit("denoising", steps, selected["total_steps"])

                emit("denoising", 0, selected["total_steps"])
                try:
                    if progressive:
                        from vflash.adapters.vdn_selflift import sample_selflift

                        if self._decoder is None:
                            self._decoder = OfficialMediaDecoder(
                                video_component=self.assets.decoder / "video_vae",
                                audio_component=self.assets.decoder / "audio_vae",
                                trust_local_code=True,
                            )
                        video, audio, sampling = sample_selflift(
                            self.assets.weights,
                            self.assets.official_model,
                            self._decoder,
                            canvas,
                            directory / "target.pt",
                            directory / "first.pt",
                            request.seed,
                            step_callback=on_step,
                        )
                    else:
                        video, audio, sampling = engine.sample(
                            directory / "target.pt",
                            request.seed,
                            first_pass_conditioning=directory / "first.pt"
                            if two_pass
                            else None,
                            upscale=(
                                lambda v, w, h: upscale_bf16(
                                    v, self.assets.upscaler_checkpoint, w, h
                                )
                            )
                            if two_pass
                            else None,
                            step_callback=on_step,
                        )
                    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
                        raise ContractError("VDN produced nonfinite latents")
                    latent = directory / "latents.safetensors"
                    save_file(
                        dict(
                            video_latents=video.cpu().contiguous(),
                            audio_latents=audio.cpu().contiguous(),
                        ),
                        latent,
                    )
                    del video, audio
                finally:
                    if engine is not None:
                        engine.close()
                del engine
                gc.collect()
                torch.cuda.empty_cache()
                emit("decoding")
                tick = time.monotonic()
                if self._decoder is None:
                    self._decoder = OfficialMediaDecoder(
                        video_component=self.assets.decoder / "video_vae",
                        audio_component=self.assets.decoder / "audio_vae",
                        trust_local_code=True,
                    )
                self._decoder.resume_cuda()
                with ExitStack() as frames:
                    frames.callback(self._decoder.suspend_cuda)
                    exact = request.keyframe_delivery_profile == "exact-v1"
                    from PIL import Image

                    first = (
                        frames.enter_context(Image.open(directory / "first-reference.png"))
                        if exact
                        else None
                    )
                    last = (
                        frames.enter_context(Image.open(directory / "last-reference.png"))
                        if exact and request.mode == "fl2va"
                        else None
                    )
                    media = self._decoder.generate_mp4(
                        latent,
                        directory / "result.mp4",
                        width=request.width,
                        height=request.height,
                        duration_seconds=request.duration_seconds,
                        audio_delivery_profile=request.audio_delivery_profile,
                        keyframe_delivery_profile=request.keyframe_delivery_profile,
                        first_frame=first,
                        last_frame=last,
                    )
                emit("decoding", 1)
                result = VideoResult(
                    output_path,
                    f"{request.mode}-vdn-{self.strategy}-fp8-sm89",
                    request.seed,
                    time.monotonic() - began,
                    dict(
                        encoding=encoding,
                        denoising=sampling,
                        sampling_initialization_seconds=initialization,
                        media=dict(
                            elapsed_seconds=time.monotonic() - tick, **media.stage_durations
                        ),
                    ),
                    media.media,
                    request_mode=request.mode,
                )
                # Publish only a complete file, never overwriting a path that
                # another caller may have created while the GPU was working.
                os.link(directory / "result.mp4", output_path)
                return result
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise ContractError("Cannot close a busy VDN pipeline")
        try:
            if not self._closed:
                if self._decoder is not None:
                    self._decoder.close()
                self._closed = True
        finally:
            self._lock.release()
