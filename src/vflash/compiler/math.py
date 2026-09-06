"""Fixed H3 timestep and AdaLN arithmetic, independent of pipeline frameworks."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from vflash.model_assets import DEFAULT_MODEL_PROFILE, model_profile, model_schedule


def profile_timesteps(profile_id: str = DEFAULT_MODEL_PROFILE) -> tuple[Any, ...]:
    """Original distinct-row counts; reference rows exist only for Ref2VA.

    Padding before the timestep MLP changes GEMM selection and FP32 rounding.
    Keep each evaluation's actual rows through both learned projections.
    """
    import torch

    profile = model_profile(profile_id)
    schedule = model_schedule(profile_id)
    rows = []
    for video, audio in zip(
        schedule.video_sigmas[:-1], schedule.audio_sigmas[:-1], strict=True
    ):
        values = [1.0 - video, 1.0 - audio]
        if profile.definition.mode.value == "ref2va":
            values.append(max(1.0 - video, schedule.keyframe_noise_aug))
        rows.append(torch.tensor(values, dtype=torch.float32).unique(sorted=True))
    return tuple(rows)


def ref4_timesteps() -> tuple[Any, ...]:
    return profile_timesteps()


def time_embeddings(
    load: Callable[[str], Any], device: Any, *, profile_id: str = DEFAULT_MODEL_PROFILE
) -> dict[str, Any]:
    """One FP32/TF32 MLP per evaluation, with the original distinct-row counts."""
    import torch
    import torch.nn.functional as functional

    rows = tuple(row.to(device) for row in profile_timesteps(profile_id))
    max_rows = max(row.numel() for row in rows)
    weights = {
        name: load(f"time_embedder.{name}").to(device)
        for name in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")
    }
    frequency = -math.log(10000) * torch.arange(128, dtype=torch.float32, device=device)
    frequency = frequency / 128
    embeddings = torch.zeros((len(rows), max_rows, 2688), dtype=torch.float32, device=device)
    timesteps = torch.zeros((len(rows), max_rows), dtype=torch.float32, device=device)
    counts = torch.tensor([row.numel() for row in rows], dtype=torch.int64, device=device)
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        for index, row in enumerate(rows):
            phases = row[:, None].float() * torch.exp(frequency)[None]
            features = torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)
            hidden = functional.linear(
                features, weights["linear_1.weight"], weights["linear_1.bias"]
            )
            output = functional.linear(
                functional.silu(hidden), weights["linear_2.weight"], weights["linear_2.bias"]
            )
            embeddings[index, : row.numel()].copy_(output)
            timesteps[index, : row.numel()].copy_(row)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    return {"time_embeddings": embeddings, "timesteps": timesteps, "timestep_counts": counts}


def adaln_table(
    embeddings: Any,
    counts: tuple[int, ...],
    weight: Any,
    bias: Any,
    *,
    modalities: int,
    modulations: int,
    hidden_size: int = 5376,
) -> Any:
    """Preserve BF16 projection order; padded rows never participate in GEMMs."""
    import torch
    import torch.nn.functional as functional

    tables = []
    for evaluation, count in enumerate(counts):
        activation = functional.silu(embeddings[evaluation, :count].float()).to(weight.dtype)
        projected = functional.linear(activation, weight, bias)
        table = torch.zeros(
            (embeddings.shape[1] * modalities, modulations, hidden_size),
            device=projected.device,
            dtype=projected.dtype,
        )
        table[: count * modalities] = projected.reshape(
            count * modalities, modulations, hidden_size
        )
        tables.append(table)
    return torch.stack(tables).contiguous()
