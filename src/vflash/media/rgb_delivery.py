"""Video-only RGB delivery, with optional explicitly partial frame decoding."""

import time
import weakref
from contextlib import nullcontext
from pathlib import Path
from traceback import clear_frames

from vflash.media.encoding import MediaError
from vflash.media.rgb_array import create_rgb_array


def validate_rgb_selection(model_frames, required_frames):
    if type(model_frames) is not int or not 1 <= model_frames <= 362:
        raise MediaError("a bounded native model timeline is required")
    if required_frames is not None and (
        type(required_frames) is not tuple
        or not required_frames
        or any(type(i) is not int or not 0 <= i < model_frames for i in required_frames)
        or tuple(sorted(set(required_frames))) != required_frames
    ):
        raise MediaError(
            "required_frames must be a nonempty sorted tuple of unique frame indices"
        )


def generate_rgb(
    decoder,
    latent_path,
    output_path,
    *,
    height,
    width,
    duration_seconds,
    model_frames,
    required_frames=None,
):
    from vflash.media import runtime
    from vflash.media.selective_decode import selective_stream

    decoder._require_open()
    if not decoder._cuda_active:
        raise MediaError("resume the video decoder before generating RGB")
    if any(type(v) is not int or v <= 0 or v % 32 for v in (height, width)):
        raise MediaError("the H3 canvas must be positive and divisible by 32")
    if type(duration_seconds) is not int or not 5 <= duration_seconds <= 15:
        raise MediaError("RGB delivery requires whole seconds from five through fifteen")
    validate_rgb_selection(model_frames, required_frames)
    frames = duration_seconds * 24
    if frames > model_frames:
        raise MediaError("model timeline is shorter than RGB delivery")
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise MediaError("the output path already exists")
    torch = decoder._torch
    started = time.monotonic()
    stages = {}
    array = None
    created = False
    report = {}
    torch.cuda.reset_peak_memory_stats(decoder.device)
    try:
        tick = time.monotonic()
        tensors = runtime.load_safetensor_tensors(latent_path, ("video_latents",))
        stages["latent_load"] = time.monotonic() - tick
        tick = time.monotonic()
        scope = nullcontext()
        if required_frames is not None:
            mask = [i in required_frames for i in range(model_frames)]
            if not hasattr(decoder, "_rgb_decode_signatures"):
                decoder._rgb_decode_signatures = weakref.WeakKeyDictionary()
            scope = selective_stream(
                decoder.video.module.model, mask, report, decoder._rgb_decode_signatures
            )
        with scope:
            pixels = runtime.decode_official_h3_video_latents(
                decoder.video,
                tensors["video_latents"],
                device=decoder.device,
                torch_module=torch,
            )
        if tuple(pixels.shape) != (1, 3, model_frames, height, width):
            raise MediaError("decoded RGB differs from the declared model timeline or canvas")
        if required_frames is not None and not report:
            raise MediaError("decoder bypassed the required temporal dependency plan")
        torch.cuda.synchronize(decoder.device)
        stages["video_decode"] = time.monotonic() - tick
        tick = time.monotonic()
        array = create_rgb_array(output, shape=(model_frames, height, width, 3))
        created = True
        for first in range(0, model_frames, 16):
            chunk = pixels[0, :, first : first + 16]
            if not torch.isfinite(chunk).all():
                raise MediaError("nonfinite decoded RGB pixels")
            array[first : first + 16] = (
                (chunk.permute(1, 2, 3, 0).float() * 255)
                .round()
                .clamp(0, 255)
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )
        if required_frames is not None:
            array[[i for i in range(model_frames) if i not in required_frames]] = 0
        array.flush()
        array._mmap.close()
        array = None
        stages["rgb_array_write"] = time.monotonic() - tick
        peak = int(torch.cuda.max_memory_allocated(decoder.device))
        media = {
            "format": "rgb8-npy",
            "frames": frames,
            "model_frames": model_frames,
            "width": width,
            "height": height,
            "fps": 24,
            "audio_samples": 0,
            "audio_sample_rate": None,
            "duration_seconds": duration_seconds,
            "size_bytes": output.stat().st_size,
            "intermediate_video_encoded": False,
            "required_frames": required_frames,
            "complete_model_timeline": required_frames is None,
        }
        if report:
            media["selective_decode"] = report
        return runtime.MediaResult(output, time.monotonic() - started, stages, peak, media)
    except BaseException as error:
        if array is not None:
            array._mmap.close()
        if created:
            output.unlink(missing_ok=True)
        try:
            torch.cuda.synchronize(decoder.device)
        except BaseException:
            decoder._closed = True
            error.add_note("CUDA completion failed; retire the pipeline process.")
        else:
            clear_frames(error.__traceback__)
        raise
