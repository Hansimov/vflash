"""Explicit, serial VDN/FreeVideo sampling with pixel-encoded keyframe stages.

The caller owns official text/keyframe encoding, model acquisition and MP4 delivery.
This adapter never downloads weights or imports optional CUDA dependencies at import.
FreeVideo is a separately installed Apache-2.0 runtime; see the community backend guide.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError


def sample_latents(engine: Any, conditioning: Path, seed: int, **options: Any) -> Any:
    """Prefetch threads must be able to write the staging tensors we allocate."""
    import torch

    # InferenceMode tensors are thread-local read-only outside their creator's
    # context; upstream asynchronous copy_ requires ordinary no_grad tensors.
    with torch.no_grad():
        return engine.sample(conditioning, seed, **options)


def sample_encoded_first_pass(
    engine: Any, conditioning: Path, seed: int, canvas: dict, **options: Any
) -> Any:
    """Bind already pixel-encoded anchors, not interpolated high-canvas latents.

    Only call while exclusively owning the engine. The target's admitted memory
    placement fits this smaller canvas, and is restored on success or exception.
    """
    target = engine.canvas
    if (
        target is None
        or canvas["frames"] != target["frames"]
        or any(canvas[key] > target[key] for key in ("width", "height"))
        or any(
            type(canvas[key]) is not int or canvas[key] < 256 or canvas[key] % 32
            for key in ("width", "height")
        )
    ):
        raise ContractError("First-pass encoding must stay within the admitted target")
    engine.canvas = canvas
    try:
        return sample_latents(engine, conditioning, seed, **options)
    finally:
        engine.canvas = target


def first_pass_reference(image: Any, sampling_plan: dict) -> Any:
    """Pad in RGB before reduction, preserving the crop's original field of view.

    Returns an owned RGB image for a fresh official VAE encoding. Pixel padding
    is edge replication; no encoded latent is resized and no generated frame is
    overwritten. Reference order remains the caller's declared first/last order.
    """
    import numpy as np
    from PIL import Image

    if not sampling_plan.get("enabled"):
        raise ContractError("First-pass reference requires two-pass generation")
    crop, lift = sampling_plan["crop"], sampling_plan["upscale_target"]
    if image.size != (crop["width"], crop["height"]):
        raise ContractError("Reference must already use the target canvas geometry")
    padding = (
        (crop["top"], lift["height"] - image.height - crop["top"]),
        (crop["left"], lift["width"] - image.width - crop["left"]),
        (0, 0),
    )
    if any(n < 0 for pair in padding for n in pair):
        raise ContractError("Invalid first-pass alignment padding")
    pixels = np.pad(np.asarray(image.convert("RGB")), padding, mode="edge")
    first = sampling_plan["first"]
    return Image.fromarray(pixels).resize(
        (first["width"], first["height"]), Image.Resampling.LANCZOS
    )


class VDNEngineSession:
    """An opt-in fixed-canvas I2VA/FL2VA VDN8 session on one SM89 device.

    Install the pinned upstream dependencies explicitly, then pass an existing
    local rowwise weight directory. CUDA selection belongs to the dedicated
    caller process. Concurrent calls fail rather than mutate a shared canvas.
    The engine owns GPU weights until close(); it does not own account state.
    """

    def __init__(self, weights: Path, *, task: str, canvas: dict) -> None:
        if task not in {"i2va", "fl2va"}:
            raise ContractError("The VDN adapter currently accepts I2VA and true FL2VA")
        from freevideo_engine.geometry import geometry
        from freevideo_engine.paths import add_vdn

        add_vdn()
        import torch
        from freevideo_engine.runtime import Engine

        if torch.cuda.get_device_capability() != (8, 9):
            raise ContractError("This VDN execution configuration is qualified only on SM89")
        self.canvas = geometry(**{k: canvas[k] for k in ("width", "height", "frames")})
        self.task = task
        self._lock = threading.Lock()
        self._closed = False
        self._engine = Engine(
            weights / "rowwise/cache",
            base=weights / "config/h3-base",
            checkpoint=weights / "config/stage-dmd-step-250",
            task=task,
            steps=8,
            attention="sdpa",
            inference_kernels=True,
            ff_chunk=2048,
            projection_chunk=1024,
            head_chunk=16,
            window_batch=4,
            resident_blocks=50,
            offload_refiner=True,
            linear_compute="native-fp8",
            fp8_gemm="auto",
            canvas=self.canvas,
        )

    def sample(
        self,
        conditioning: Path,
        seed: int,
        *,
        first_pass_conditioning: Path | None = None,
        upscale: Any = None,
        step_callback: Any = None,
        stage_callback: Any = None,
    ) -> tuple[Any, Any, dict]:
        """Full eight steps, or RGB-conditioned eight plus two tail steps.

        Two-pass requests must supply separately encoded low-canvas references
        AND a local latent upscaler callable(video, width, height) -> (video, report).
        There is deliberately no latent-resize fallback for missing anchors.
        """
        if (first_pass_conditioning is None) != (upscale is None):
            raise ContractError("Two-pass sampling requires pixel-encoded anchors and upscaler")
        if not self._lock.acquire(blocking=False):
            raise ContractError("This VDN engine is busy")
        try:
            if self._closed:
                raise ContractError("This VDN engine is closed")
            from freevideo_engine.geometry import geometry
            from freevideo_engine.two_pass import crop_latents, plan

            two_pass = first_pass_conditioning is not None
            selected = plan(self.canvas, enabled=two_pass, task=self.task)
            if two_pass:
                video, audio, first = sample_encoded_first_pass(
                    self._engine,
                    first_pass_conditioning,
                    seed,
                    geometry(**selected["first"]),
                    **selected["first"],
                    step_callback=step_callback,
                )
                if stage_callback:
                    stage_callback("first_pass", video, audio, first)
                lift = selected["upscale_target"]
                video, lifted = upscale(video, lift["width"], lift["height"])
                video = crop_latents(video, selected)
                video, audio, tail = sample_latents(
                    self._engine,
                    conditioning,
                    seed + selected["restart_seed_offset"],
                    **selected["second"],
                    initial_latents=(video, audio),
                    refine_steps=selected["refine_steps"],
                    step_callback=step_callback,
                )
                report = dict(
                    sample_seconds=first["sample_seconds"] + tail["sample_seconds"],
                    first=first,
                    upscale=lifted,
                    tail=tail,
                )
            else:
                video, audio, report = sample_latents(
                    self._engine,
                    conditioning,
                    seed,
                    **selected["first"],
                    step_callback=step_callback,
                )
            # An injected explicit upscaler need not use upstream's FP16 file.
            # Never claim its hash as an identity for a different checkpoint.
            selected.pop("upscaler_sha256", None)
            report = dict(
                report,
                sampling_plan=selected,
                first_pass_conditioning="pixel-encode" if two_pass else "target-encode",
            )
            return video, audio, report
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise ContractError("Cannot close a busy VDN engine")
        try:
            if not self._closed:
                self._engine.close()
                self._closed = True
        finally:
            self._lock.release()
