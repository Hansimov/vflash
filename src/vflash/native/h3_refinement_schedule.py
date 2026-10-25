"""Explicit sigma-matched modulation assets for H3 quality passes.

These tables are separately prepared derived weights, never reindexed Turbo
modulation. File sizes, tensor layouts and all BF16 exponent bits are checked
without hashing weight payloads or changing the caller's global thread pool.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def refinement_schedule(*, steps, denoise, video_shift=12.0, audio_shift=3.0):
    from vflash.native.h3_native_scheduler import H3NativeSchedule

    if type(steps) is not int or not 2 <= steps <= 32:
        raise ValueError("refinement requires 2..32 explicit evaluations")
    if (
        type(denoise) not in (int, float)
        or not math.isfinite(denoise)
        or not 0.01 <= denoise < 1
    ):
        raise ValueError("refinement denoise must be finite within [0.01,1)")
    full = H3NativeSchedule.shifted_linear(
        int(steps / denoise), video_shift=video_shift, audio_shift=audio_shift
    )
    # A non-executed full-noise prefix keeps the native schedule invariant.
    # Execution starts at index one, with exactly `steps` low-noise evaluations.
    return H3NativeSchedule(
        (1.0, *full.video_sigmas[-steps - 1 :]),
        (1.0, *full.audio_sigmas[-steps - 1 :]),
    )


def timestep_rows(schedule):
    import torch

    return tuple(
        torch.tensor(
            [1 - video, 1 - audio, schedule.keyframe_noise_aug], dtype=torch.float32
        ).unique(sorted=True)
        for video, audio in zip(
            schedule.video_sigmas[:-1], schedule.audio_sigmas[:-1], strict=True
        )
    )


def finite_cpu_bf16(table):
    """Validate BF16 exponent bits without a global Torch thread-pool change.

    Rental CPU-only modes can expose the whole host's thread count while having
    a small CPU quota. Repeated Torch reductions then dominate table loading.
    This single-threaded, zero-copy input view checks every element, including
    signed NaNs/infinities; it is not a sample or a cached integrity assertion.
    """
    import numpy as np
    import torch

    if table.dtype != torch.bfloat16 or table.device.type != "cpu":
        raise ValueError("finite table validation requires CPU BF16")
    words = table.view(torch.int16).numpy()
    return bool(np.all(np.bitwise_and(words, 0x7F80) != 0x7F80))


def load_tables(directory, *, mode):
    import torch
    from safetensors.torch import load_file

    from vflash.native.h3_native_scheduler import H3NativeSchedule

    if mode not in ("face", "audio"):
        raise ValueError("unknown refinement modality")
    directory = Path(directory)
    manifest = json.loads((directory / "refinement.json").read_text())
    if (
        manifest.get("schema") != 1
        or manifest.get("mode") != mode
        or manifest.get("method") != "hybrid-low-noise-official-modulation-v1"
        or manifest.get("start_evaluation") != 1
    ):
        raise ValueError("refinement tables have a different contract")
    schedule = H3NativeSchedule.from_mapping(manifest["schedule"])
    if schedule != refinement_schedule(steps=manifest["steps"], denoise=manifest["denoise"]):
        raise ValueError("refinement sigma recipe differs")
    expected = [f"block-{i:03d}.safetensors" for i in range(50)] + ["final.safetensors"]
    if [r["file"] for r in manifest["files"]] != expected:
        raise ValueError("refinement table inventory differs")
    tables = []
    for index, record in enumerate(manifest["files"]):
        path = directory / record["file"]
        if path.is_symlink() or path.stat().st_size != record["size"]:
            raise ValueError("refinement table file differs")
        values = load_file(str(path), device="cpu")
        shape = (schedule.nfe, 9, 6, 5376) if index < 50 else (schedule.nfe, 3, 2, 5376)
        table = values.get("adaln.table")
        if (
            set(values) != {"adaln.table"}
            or tuple(table.shape) != shape
            or table.dtype != torch.bfloat16
            or not finite_cpu_bf16(table)
        ):
            raise ValueError("refinement table tensor differs")
        tables.append(table)
    return manifest, schedule, tables
