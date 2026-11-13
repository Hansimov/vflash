"""Explicit inputs for the pinned original Viggle Remake Comfy implementation.

This model-side adapter builds only a fixed, owned local graph. It does not run a
server, download models, choose hardware, track people or admit a public workflow.
The original H3 FL2VA premerged INT8 models remain separate from Pika/Larry recipes.
"""

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from vflash.media.probe import probe_video

SOURCE_REVISION = "7e5be143702dd53b9154482de673b8519ed2d4d6"
COMFY_REVISION = "355cd6b14c92c19ac3188fe528e47a3f21ef5178"
SIGMAS = {
    "swap": "1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579, 0.0",
    "edit": "1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882, 0.0",
}
DISTILLATION_ADAPTER = "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
DISTILLATION_SIGMAS = "1.0, 0.947368, 0.857143, 0.666667, 0.0"


@dataclass(frozen=True)
class RemakeReference:
    path: Path
    # A real frame of the desired target, not an arbitrary appearance photograph.
    frame: int | None = None

    def __post_init__(self):
        if not isinstance(self.path, Path) or (
            self.frame is not None
            and (type(self.frame) is not int or not 0 <= self.frame < 345)
        ):
            raise ValueError(
                "A reference requires a local image and optional exact target frame"
            )


@dataclass(frozen=True)
class RemakePrefix:
    take: Path
    start: int
    frames: int

    def __post_init__(self):
        if (
            not isinstance(self.take, Path)
            or type(self.start) is not int
            or not 0 <= self.start <= 100000
            or type(self.frames) is not int
            or not 17 <= self.frames <= 323
            or self.frames % 17
        ):
            raise ValueError("A generated prefix requires an explicit 17-frame-grid history")


@dataclass(frozen=True)
class RemakeRequest:
    source: Path
    prompt: str
    mode: Literal["swap", "edit"]
    frames: int
    width: int
    height: int
    references: tuple[RemakeReference, ...]
    soundtrack: Literal["preserve", "silent"]
    seed: int = 0
    prefix: RemakePrefix | None = None
    attention: Literal["full", "veda-r2va"] = "full"
    distillation: Literal["none", "fl2v-v1.2-4step"] = "none"
    source_mask: Path | None = None

    def __post_init__(self):
        if (
            not isinstance(self.source, Path)
            or not isinstance(self.prompt, str)
            or not self.prompt.strip()
            or len(self.prompt) > 65536
            or self.mode not in SIGMAS
            or type(self.frames) is not int
            or not 90 <= self.frames <= 345
            or (self.frames - 5) % 17
            or any(
                type(v) is not int or not 480 <= v <= 3072 or v % 32
                for v in (self.width, self.height)
            )
            or not 480 <= min(self.width, self.height) <= 736
            or not 0.25 <= self.width / self.height <= 4
            or self.width * self.height > 2048**2
            or type(self.seed) is not int
            or not 0 <= self.seed <= 2**64 - 1
            or self.soundtrack not in {"preserve", "silent"}
            or self.attention not in {"full", "veda-r2va"}
            or self.distillation not in {"none", "fl2v-v1.2-4step"}
            or (self.source_mask is not None and not isinstance(self.source_mask, Path))
            or not isinstance(self.references, tuple)
            or not 1 <= len(self.references) <= 4
            or any(not isinstance(ref, RemakeReference) for ref in self.references)
        ):
            raise ValueError("An explicit bounded original Remake request is required")
        for ref in self.references:
            ref.__post_init__()
        pins = [ref.frame for ref in self.references if ref.frame is not None]
        if pins and (len(pins) != len(self.references) or max(pins) >= self.frames):
            raise ValueError("Original Remake pins every reference or uses none")
        if self.prefix is not None:
            if not isinstance(self.prefix, RemakePrefix):
                raise ValueError("Generated history is a separate typed input")
            self.prefix.__post_init__()
            if self.prefix.frames >= self.frames or any(p < self.prefix.frames for p in pins):
                raise ValueError("Target pins cannot replace clean generated history")
        if any(
            int(n) > len(self.references) or int(n) < 1
            for n in re.findall(r"<Picture (\d+)>", self.prompt)
        ):
            raise ValueError("The prompt refers to an image not supplied to this window")
        if any(n != "1" for n in re.findall(r"<Video (\d+)>", self.prompt)):
            raise ValueError("This window has exactly one source video")


def remake_graph(
    request: RemakeRequest,
    *,
    input_directory: Path,
    output_prefix: str,
    cancel_check=None,
) -> dict:
    """Bind owned files to the original graph or an explicit distilled composition.

    Inspect actual frame/canvas metadata before selecting the original nodes. The
    caller owns source display normalization; this decoder consumes raw pixels,
    as the author graph does. Pins must already match that exact target canvas.
    No remote URL or arbitrary node is accepted.
    """
    if not isinstance(request, RemakeRequest):
        raise ValueError("An explicit Remake request is required")
    request.__post_init__()
    if not isinstance(input_directory, Path) or input_directory.is_symlink():
        raise ValueError("The runtime owns a regular input directory")
    directory = input_directory.resolve(strict=True)
    if (
        not directory.is_dir()
        or not isinstance(output_prefix, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}(?:/[A-Za-z0-9_-]{1,80}){0,3}", output_prefix)
    ):
        raise ValueError("An owned input directory and relative output prefix are required")

    def owned(path, video=False):
        resolved = path.resolve(strict=True)
        extensions = {".mp4"} if video else {".png", ".jpg", ".jpeg", ".webp"}
        if (
            path.is_symlink()
            or not path.is_file()
            or not resolved.is_relative_to(directory)
            or path.suffix.lower() not in extensions
            or not 0 < path.stat().st_size <= (2 * 1024**3 if video else 64 * 1024**2)
        ):
            raise ValueError(
                "All model inputs must remain regular files in the owned directory"
            )
        return resolved.relative_to(directory).as_posix()

    source_name = owned(request.source, True)

    def clock(path):
        video, _ = probe_video(path, cancel_check)
        if (
            Fraction(video["r_frame_rate"]) != 24
            or Fraction(video["avg_frame_rate"]) != 24
            or Fraction(video.get("start_time", "0")) != 0
            or (video["width"], video["height"]) != (request.width, request.height)
            or any(s.get("rotation", 0) for s in video.get("side_data_list", []))
        ):
            raise ValueError("Prepared source/history must match the declared clock and canvas")
        return int(video["nb_frames"])

    if clock(request.source) != request.frames:
        raise ValueError("Prepared source must contain every declared model frame")
    if request.prefix:
        owned(request.prefix.take, True)
        if clock(request.prefix.take) < request.prefix.start + request.prefix.frames:
            raise ValueError("The generated take does not cover its complete history interval")
    from PIL import Image

    for ref in request.references:
        owned(ref.path)
        with Image.open(ref.path) as picture:
            width, height = picture.size
        if (
            min(width, height) < 32
            or max(width, height) > 4096
            or not 0.25 <= width / height <= 4
            or (ref.frame is not None and (width, height) != (request.width, request.height))
            or (
                ref.frame is None
                and (
                    min(width, height) != min(request.width, request.height)
                    or width % 32
                    or height % 32
                )
            )
        ):
            raise ValueError(
                "Prepare reference geometry explicitly; target pins cannot stretch"
            )

    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    conditioning = dict(
        clip=["4", 0],
        vae=["5", 0],
        audio_vae=["6", 0],
        source=["8", 0],
        prompt=request.prompt,
        frames=request.frames,
        short_edge=min(request.width, request.height),
        pins=", ".join(str(ref.frame) for ref in request.references if ref.frame is not None),
    )
    if request.soundtrack == "preserve":
        conditioning["audio"] = ["8", 1]
    graph = {
        "1": node(
            "UNETLoader",
            unet_name=(f"viggle_remake_{request.mode}_pruned_int8_convrot.safetensors"),
            weight_dtype="default",
        ),
        "3": node("RemakeKeyframeUntil", model=["1", 0], keyframe_until=0.9),
        "4": node(
            "CLIPLoader",
            clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            type="minimax",
        ),
        "5": node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "6": node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "7": node("LoadVideo", file=source_name),
        "8": node("GetVideoComponents", video=["7", 0]),
        "11": node("RemakeConditioning", **conditioning),
        "12": node("RemakeNoise", positive=["11", 0], seed=request.seed),
        "13": node("BasicGuider", model=["3", 0], conditioning=["11", 0]),
        "14": node("KSamplerSelect", sampler_name="euler"),
        "15": node("ManualSigmas", sigmas=SIGMAS[request.mode]),
        "16": node(
            "SamplerCustomAdvanced",
            noise=["12", 0],
            guider=["13", 0],
            sampler=["14", 0],
            sigmas=["15", 0],
            latent_image=["11", 1],
        ),
        "17": node("VAEDecode", samples=["16", 0], vae=["5", 0]),
        "18": node("VAEDecodeAudio", samples=["16", 0], vae=["6", 0]),
        "19": node("CreateVideo", images=["17", 0], audio=["18", 0], fps=24.0),
        "20": node(
            "SaveVideo",
            video=["19", 0],
            filename_prefix=output_prefix,
            format="mp4",
            codec="h264",
        ),
    }
    if request.distillation == "fl2v-v1.2-4step":
        # The diffusion file already contains Remake's task adapter. Add only
        # the pure FL2V turbo delta, never a fused Remake+turbo task LoRA.
        graph["24"] = node(
            "LoraLoaderBypassModelOnly",
            model=["1", 0],
            lora_name=DISTILLATION_ADAPTER,
            strength_model=1.0,
        )
        graph["3"]["inputs"]["model"] = ["24", 0]
        graph["15"]["inputs"]["sigmas"] = DISTILLATION_SIGMAS
    for index, ref in enumerate(request.references, 1):
        ident = str(index + 8) if index <= 2 else str(index + 30)
        graph[ident] = node("LoadImage", image=owned(ref.path))
        graph["11"]["inputs"][f"ref_{index}"] = [ident, 0]
    if request.prefix is not None:
        graph["21"] = node("LoadVideo", file=owned(request.prefix.take, True))
        graph["22"] = node("GetVideoComponents", video=["21", 0])
        graph["23"] = node(
            "RemakePrefix",
            vae=["5", 0],
            latent=["11", 1],
            prefix=["22", 0],
            prefix_frames=request.prefix.frames,
            prefix_at=request.prefix.start,
        )
        graph["16"]["inputs"]["latent_image"] = ["23", 0]
    if request.source_mask is not None:
        mask_name = owned(request.source_mask)
        with Image.open(request.source_mask) as mask:
            if mask.size != (request.width, request.height):
                raise ValueError("The source mask must match the prepared canvas")
        graph["25"] = node("LoadImage", image=mask_name)
        graph["26"] = node(
            "VFlashRemakeSourceMask",
            positive=["11", 0],
            latent=graph["16"]["inputs"]["latent_image"],
            edit_mask=["25", 0],
        )
        graph["16"]["inputs"]["latent_image"] = ["26", 0]
    if request.attention == "veda-r2va":
        graph["30"] = node(
            "VedaSparseAttention",
            model=["3", 0],
            predictor="minimax_h3_r2va_veda_preview_fp8.safetensors",
            generated_sparsity="trained",
            reference_sparsity="trained",
            full_attention_layers="",
            full_attention_steps="",
            verbose=False,
        )
        graph["13"]["inputs"]["model"] = ["30", 0]
    if request.soundtrack == "silent":
        # The original conditioning still holds clean zero audio latents; the
        # decoded silence is unnecessary and must not introduce a faint waveform.
        del graph["18"]
        del graph["19"]["inputs"]["audio"]
    return graph
