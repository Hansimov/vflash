"""MiniMax-H3 packed-row to VAE latent layout conversions."""

from __future__ import annotations

from typing import Any

from vflash.native.errors import VflashNativeError
from vflash.native.h3_native_scheduler import H3_KEYFRAME_NOISE_AUG


class H3LatentLayoutError(VflashNativeError):
    """Packed H3 latent rows do not match the requested VAE geometry."""


H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3LatentLayoutError("H3 latent layout conversion requires PyTorch") from exc
    return torch


def _leading_count(mask: Any) -> int:
    torch = _torch()
    false_rows = torch.nonzero(~mask, as_tuple=False)
    return int(false_rows[0]) if false_rows.numel() else int(mask.numel())


def infer_h3_condition_prefix_counts(
    *,
    captured_timesteps: Any,
    captured_timestep_counts: Any,
    captured_timestep_indices: Any,
    video_indices: Any,
    audio_indices: Any,
) -> tuple[int, int]:
    """Recover immutable Ref2VA video/audio prefix lengths from a replay."""

    torch = _torch()
    tensors = (
        captured_timesteps,
        captured_timestep_counts,
        captured_timestep_indices,
        video_indices,
        audio_indices,
    )
    if any(not isinstance(item, torch.Tensor) or item.device.type != "cpu" for item in tensors):
        raise H3LatentLayoutError("H3 condition-prefix inference requires CPU tensors")
    if (
        captured_timesteps.ndim != 2
        or captured_timestep_counts.ndim != 1
        or captured_timestep_indices.ndim != 2
        or video_indices.ndim != 1
        or audio_indices.ndim != 1
        or not captured_timestep_counts.numel()
    ):
        raise H3LatentLayoutError("H3 condition-prefix replay shapes are invalid")
    count = int(captured_timestep_counts[0])
    unique = captured_timesteps[0, :count]
    rows = unique.index_select(0, captured_timestep_indices[0].to(torch.int64))
    video_rows = rows.index_select(0, video_indices.to(torch.int64))
    audio_rows = rows.index_select(0, audio_indices.to(torch.int64))
    video_count = _leading_count(
        video_rows == torch.tensor(H3_KEYFRAME_NOISE_AUG, dtype=torch.float32)
    )
    audio_count = _leading_count(audio_rows == torch.tensor(1.0, dtype=torch.float32))
    if not bool((video_rows[video_count:] == unique[0]).all()) or not bool(
        (audio_rows[audio_count:] == unique[0]).all()
    ):
        raise H3LatentLayoutError("H3 replay does not use a condition-prefix layout")
    return video_count, audio_count


def h3_video_latent_frame_count(num_frames: int) -> int:
    """Return the released H3 VAE temporal-token count for aligned frames."""

    if (
        not isinstance(num_frames, int)
        or isinstance(num_frames, bool)
        or num_frames <= 0
        or num_frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK
    ):
        raise H3LatentLayoutError("H3 frame count must be positive and have the form 17*n+5")
    return (num_frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK * H3_LATENTS_PER_CHUNK + 2


def unpatchify_h3_video_rows(
    rows: Any,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int = 24,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> Any:
    """Convert batch-one frame-major ``[1, rows, width]`` into VAE latents."""

    torch = _torch()
    if (
        not isinstance(rows, torch.Tensor)
        or rows.ndim != 3
        or int(rows.shape[0]) != 1
        or len(patch_size) != 3
        or any(not isinstance(item, int) or item <= 0 for item in patch_size)
        or min(num_latent_frames, latent_height, latent_width, channels) <= 0
    ):
        raise H3LatentLayoutError("H3 video latent rows or geometry are invalid")
    patch_t, patch_h, patch_w = patch_size
    if num_latent_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise H3LatentLayoutError("H3 video latent geometry is not patch-aligned")
    expected_rows = (
        num_latent_frames // patch_t * (latent_height // patch_h) * (latent_width // patch_w)
    )
    expected_width = channels * patch_t * patch_h * patch_w
    if tuple(rows.shape[1:]) != (expected_rows, expected_width):
        raise H3LatentLayoutError(
            "H3 video latent row shape differs: "
            f"{tuple(rows.shape[1:])} != {(expected_rows, expected_width)}"
        )
    value = rows.reshape(
        1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    value = value.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return value.reshape(
        1,
        channels,
        num_latent_frames,
        latent_height,
        latent_width,
    ).contiguous()


def unpack_h3_audio_rows(
    rows: Any,
    *,
    channels: int = 2,
) -> Any:
    """Convert batch-one audio rows into ``[channels, features, time]``."""

    torch = _torch()
    if (
        not isinstance(rows, torch.Tensor)
        or rows.ndim != 3
        or int(rows.shape[0]) != 1
        or int(rows.shape[1]) <= 0
        or int(rows.shape[2]) <= 0
        or not isinstance(channels, int)
        or isinstance(channels, bool)
        or channels <= 0
        or int(rows.shape[1]) % channels
    ):
        raise H3LatentLayoutError("H3 audio latent rows are invalid")
    frames = int(rows.shape[1]) // channels
    return rows.reshape(channels, frames, int(rows.shape[2])).permute(0, 2, 1).contiguous()
