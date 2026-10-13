"""Project a validated AV conditioning request to video-only packed rows."""

from __future__ import annotations

from typing import Any


def without_generated_audio(
    tensors: dict[str, Any], *, condition_audio_rows: int
) -> dict[str, Any]:
    """Preserve video/text positions and latent noise; remove all target audio.

    Call only after the ordinary bundle validates. Audio-reference semantics
    are deliberately not inferred from a video-only output request.
    """
    import torch

    if type(condition_audio_rows) is not int or condition_audio_rows != 0:
        raise ValueError("video-only generation does not accept audio conditioning")
    tags = tensors["token_tags"]
    audio = tensors["audio_indices"].to(device=tags.device, dtype=torch.int64)
    if audio.ndim != 1 or not audio.numel():
        raise ValueError("expected a validated nonempty AV input")
    keep = torch.ones(tags.numel(), dtype=torch.bool, device=tags.device)
    keep[audio] = False
    selected = torch.nonzero(keep).flatten()
    remap = torch.full((tags.numel(),), -1, dtype=torch.int64, device=tags.device)
    remap[selected] = torch.arange(selected.numel(), device=tags.device)
    result = dict(tensors)
    for name in ("video_indices", "text_indices"):
        old = tensors[name].to(device=tags.device, dtype=torch.int64)
        result[name] = remap[old].to(dtype=tensors[name].dtype)
        if bool((result[name] < 0).any()):
            raise ValueError("packed modalities overlap")
    result["audio_indices"] = tensors["audio_indices"][:0].contiguous()
    result["initial_audio"] = tensors["initial_audio"][:, :0].contiguous()
    for name in ("position_ids", "token_tags", "rotary_cos", "rotary_sin"):
        result[name] = (
            tensors[name].index_select(0, selected.to(tensors[name].device)).contiguous()
        )
    return result
