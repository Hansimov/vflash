"""Optional complete-video FlashVSR v1.1 upscaling.

Loads an explicitly pinned external checkout, never ComfyUI's node wrappers.
Preserves full source geometry, all frames, rational FPS and original audio.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

SOURCE_REVISION = "4820b3f02347bddcbbb9a5a85ab7638fe976366e"
MODEL_REVISION = "27561b186ded3402d7c975f4fd722e2885b6135f"
WEIGHT_FILES = {
    "diffusion_pytorch_model_streaming_dmd.safetensors": 5676070392,
    "LQ_proj_in.ckpt": 575694948,
    "TCDecoder.ckpt": 189018333,
}


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    padded_width: int
    padded_height: int
    frames: int
    padded_frames: int


def geometry(width: int, height: int, frames: int, scale: int) -> Geometry:
    if (
        any(type(v) is not int for v in (width, height, frames, scale))
        or min(width, height) < 32
        or frames < 1
        or scale not in (2, 4)
    ):
        raise ValueError("positive video geometry and scale 2 or 4 required")
    w, h = width * scale, height * scale
    # Streaming tiny emits F-4 frames from its 8n+1 input. Round UP, not down.
    padded = max(25, ((frames + 3 + 7) // 8) * 8 + 1)
    return Geometry(w, h, (w + 127) // 128 * 128, (h + 127) // 128 * 128, frames, padded)


def metadata(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    streams = json.loads(result.stdout)["streams"]
    stream = next(s for s in streams if s["codec_type"] == "video")
    rate = Fraction(stream["avg_frame_rate"])
    if rate <= 0 or rate > 60 or rate != Fraction(stream["r_frame_rate"]):
        raise ValueError("unsupported frame rate")
    return dict(
        width=stream["width"],
        height=stream["height"],
        fps=str(rate),
        frames=int(stream.get("nb_frames", 0)),
        audio=any(s["codec_type"] == "audio" for s in streams),
    )


def load_vendor(source: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != SOURCE_REVISION
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    ):
        raise ValueError("FlashVSR source must be the clean pinned revision")
    name = "_vflash_flashvsr_vendor"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, source / "src/__init__.py", submodule_search_locations=[str(source / "src")]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class FlashVSR:
    def __init__(self, source: Path, weights: Path, *, trust_local_code: bool = False):
        if trust_local_code is not True:
            raise ValueError("FlashVSR requires trust_local_code=True")
        import torch

        vendor = load_vendor(source)
        for name, size in WEIGHT_FILES.items():
            if (weights / name).stat().st_size != size:
                raise ValueError("incomplete pinned model file")
        from _vflash_flashvsr_vendor.models import wan_video_dit
        from _vflash_flashvsr_vendor.models.TCDecoder import build_tcdecoder
        from _vflash_flashvsr_vendor.models.utils import Causal_LQ4x_Proj

        # Explicit local mask + sparse Sage, never an unnoticed dense fallback.
        wan_video_dit.USE_BLOCK_ATTN = False
        # Unmasked text cross-attention is separate from LCSA. Do not let an
        # unrelated installed Sage wheel select missing architecture kernels.
        wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
        wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
        wan_video_dit.SAGE_ATTN_AVAILABLE = False
        self.attention_calls = 0
        sparse = wan_video_dit.sparse_sageattn

        def observed_sparse(*args, **kwargs):
            if kwargs.get("mask_id") is None:
                raise RuntimeError("local sparse attention mask was lost")
            self.attention_calls += 1
            return sparse(*args, **kwargs)

        wan_video_dit.sparse_sageattn = observed_sparse
        started = time.monotonic()
        manager = vendor.ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        manager.load_models(
            [str(weights / "diffusion_pytorch_model_streaming_dmd.safetensors")]
        )
        pipe = vendor.FlashVSRTinyLongPipeline.from_model_manager(manager, device="cuda")
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1
        ).to("cuda", dtype=torch.bfloat16)
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(weights / "LQ_proj_in.ckpt", map_location="cpu", weights_only=True),
            strict=True,
        )
        pipe.TCDecoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128],
            device="cuda",
            dtype=torch.bfloat16,
            new_latent_channels=784,
        )
        incompatible = pipe.TCDecoder.load_state_dict(
            torch.load(weights / "TCDecoder.ckpt", map_location="cpu", weights_only=True),
            strict=False,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("tiny decoder weight mismatch")
        pipe.denoising_model().eval().requires_grad_(False)
        pipe.TCDecoder.eval().requires_grad_(False)
        pipe.to("cuda", dtype=torch.bfloat16)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv(prompt_path=str(source / "posi_prompt.pth"))
        pipe.load_models_to_device(["dit", "vae"])
        torch.cuda.synchronize()
        self.pipe, self.load_seconds = pipe, time.monotonic() - started

    def enhance(
        self,
        input_path: Path,
        output_path: Path,
        *,
        scale: int = 2,
        seed: int = 0,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict:
        import numpy as np
        import torch
        import torch.nn.functional as functional

        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError("never overwrite a prior enhancement")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")

        def progress(stage, current, total):
            if on_progress is not None:
                on_progress(stage, current, total)

        progress("preparing", 0, 1)
        total_start = time.monotonic()
        meta = metadata(input_path)
        if meta["width"] * meta["height"] > 1048576 or meta["frames"] > 600:
            raise ValueError("source exceeds qualification budget")
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "2",
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "601",
                "-vsync",
                "0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=90,
        ).stdout
        frames = np.frombuffer(decoded, np.uint8).reshape(-1, meta["height"], meta["width"], 3)
        count = len(frames)
        if not count or count > 600 or (meta["frames"] and meta["frames"] != count):
            raise ValueError("source frame count differs from decoding")
        g = geometry(meta["width"], meta["height"], count, scale)
        if g.width * g.height > 4194304 or count / float(Fraction(meta["fps"])) > 10.001:
            raise ValueError("qualified enhancement budget is at most 4MP and ten seconds")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        prepared = []
        # Keep the conditioning sequence in host RAM; transfer only the active
        # temporal window. Exact scaling followed by edge padding avoids cropping.
        for frame in frames:
            tensor = (
                torch.from_numpy(frame.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to("cuda", dtype=torch.float32)
                / 255
            )
            tensor = functional.interpolate(
                tensor, size=(g.height, g.width), mode="bicubic", align_corners=False
            )
            tensor = functional.pad(
                tensor,
                (0, g.padded_width - g.width, 0, g.padded_height - g.height),
                mode="replicate",
            )
            prepared.append(
                tensor.squeeze(0).to(device="cpu", dtype=torch.bfloat16).mul_(2).sub_(1)
            )
        prepared.extend([prepared[-1]] * (g.padded_frames - count))
        lq = torch.stack(prepared, dim=1).unsqueeze(0)
        del prepared, frames, decoded
        prepare_seconds = time.monotonic() - total_start
        progress("encoding", 1, 1)
        calls_before = self.attention_calls
        model_start = time.monotonic()

        def windows(iterable):
            total = len(iterable)
            progress("denoising", 0, total)
            for index, value in enumerate(iterable):
                yield value
                progress("denoising", index + 1, total)

        with torch.inference_mode():
            video = self.pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=seed,
                LQ_video=lq,
                num_frames=g.padded_frames,
                height=g.padded_height,
                width=g.padded_width,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=2.0 * 768 * 1280 / (g.padded_height * g.padded_width),
                kv_ratio=3.0,
                local_range=11,
                color_fix=False,
                unload_dit=False,
                force_offload=False,
                progress_bar_cmd=windows,
            )
            # The vendor swallows color-correction errors. Execute it explicitly
            # and fail visibly rather than silently publish a color-shifted video.
            video = self.pipe.ColorCorrector(
                video.unsqueeze(0).cpu(),
                lq[:, :, : video.shape[1]],
                chunk_size=8,
                method="adain",
            )[0]
        torch.cuda.synchronize()
        model_seconds = time.monotonic() - model_start
        if video.shape[1] < count or self.attention_calls == calls_before:
            raise RuntimeError("incomplete video or sparse attention did not execute")
        progress("decoding", 1, 1)
        encode_start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=".flashvsr-", dir=output_path.parent) as work:
            encoded = Path(work) / "video.mp4"
            command = [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{g.width}x{g.height}",
                "-r",
                meta["fps"],
                "-i",
                "-",
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-threads",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-t",
                str(float(Fraction(count, 1) / Fraction(meta["fps"]))),
                "-movflags",
                "+faststart",
                str(encoded),
            ]
            with subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            ) as writer:
                try:
                    for index in range(count):
                        frame = video[:, index, : g.height, : g.width]
                        if not torch.isfinite(frame).all():
                            raise RuntimeError("nonfinite enhanced pixels")
                        pixels = (
                            ((frame.float() + 1) * 127.5)
                            .clamp(0, 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .numpy()
                        )
                        writer.stdin.write(pixels.tobytes())
                    writer.stdin.close()
                    if writer.wait(timeout=90):
                        raise RuntimeError("enhanced media encoding failed")
                finally:
                    if writer.poll() is None:
                        writer.kill()
            checked = metadata(encoded)
            if (
                checked["width"],
                checked["height"],
                checked["frames"],
                checked["fps"],
                checked["audio"],
            ) != (g.width, g.height, count, meta["fps"], meta["audio"]):
                raise RuntimeError("enhancement changed geometry/time/audio contract")
            # Same-filesystem exclusive publication; an existing result is never overwritten.
            os.link(encoded, output_path)
        progress("delivery", 1, 1)
        return dict(
            complete=True,
            source_revision=SOURCE_REVISION,
            model_revision=MODEL_REVISION,
            profile="flashvsr-v11-tiny-stream-sparse-sage",
            cross_attention="torch-sdpa",
            scale=scale,
            seed=seed,
            input=meta,
            output=checked,
            padded_frames=g.padded_frames,
            padded_width=g.padded_width,
            padded_height=g.padded_height,
            load_seconds=self.load_seconds,
            prepare_seconds=prepare_seconds,
            model_seconds=model_seconds,
            encode_seconds=time.monotonic() - encode_start,
            elapsed_seconds=time.monotonic() - total_start,
            peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
            sparse_attention_calls=self.attention_calls - calls_before,
        )


def add_flashvsr_command(commands):
    parser = commands.add_parser(
        "upscale-video", help="explicit optional FlashVSR v1.1 enhancement"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-local-code", action="store_true")


def run_flashvsr_command(args):
    model = FlashVSR(args.source, args.weights, trust_local_code=args.trust_local_code)
    result = model.enhance(args.input, args.output, scale=args.scale, seed=args.seed)
    print(json.dumps(result))
    return 0
