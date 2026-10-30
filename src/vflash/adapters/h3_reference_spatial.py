"""Approximate spatial conditioning for a qualified fixed-video native audio pass.

All source times, the full first-image condition, text and audio are retained.
Only the generated video's spatial conditioning rows are reduced; the caller's
original complete picture remains pinned by ``fixed_video_audio_pass``.
"""

from contextlib import contextmanager


@contextmanager
def spatial_video_condition(model, *, height, width, condition_rows, stride=2):
    """Own the row-layout adapter inside an already pinned five-second pass."""
    import torch

    if (
        torch.is_grad_enabled()
        or model.training
        or any(type(value) is not int or value < 1 for value in (height, width, condition_rows))
        or type(stride) is not int
        or stride not in (2, 4)
        or condition_rows != height * width
        or not getattr(model, "_vflash_fixed_audio_active", False)
        or getattr(model, "_vflash_spatial_audio_active", False)
    ):
        raise ValueError(
            "Spatial video conditions require an owned fixed-picture inference pass"
        )
    record = {
        "method": "fixed-video-audio-spatial-context-v1",
        "approximate": True,
        "stride": stride,
        "all_source_times_retained": True,
        "condition_frame_retained": True,
        "forwards": 0,
    }
    state = {}

    def before(_model, _args, kwargs):
        hidden, video = kwargs["hidden_states"], kwargs["video_indices"]
        if (
            hidden.shape != (1, len(video), 96)
            or len(video) != 38 * height * width
            or not torch.all(kwargs["timestep"][kwargs["timestep_indices"][video]] == 0.999)
        ):
            raise ValueError("Spatial audio differs from its qualified pinned video layout")
        if not state:
            cells = torch.arange(37 * height * width, device=video.device)
            cells = cells.reshape(37, height, width)
            selected = torch.cat(
                (
                    torch.arange(condition_rows, device=video.device),
                    cells[:, ::stride, ::stride].flatten() + condition_rows,
                )
            )
            keep = (
                torch.cat(
                    (
                        kwargs["text_indices"],
                        kwargs["audio_indices"],
                        video[selected],
                    )
                )
                .sort()
                .values
            )
            if keep.unique().numel() != keep.numel():
                raise ValueError("Spatial audio modality indices overlap")
            inverse = torch.full(
                (len(kwargs["token_tags"]),), -1, dtype=torch.long, device=video.device
            )
            inverse[keep] = torch.arange(len(keep), device=video.device)
            state.update(video=video.clone(), selected=selected, keep=keep, inverse=inverse)
            record.update(
                original_video_rows=len(video),
                retained_video_rows=len(selected),
                original_total_rows=len(kwargs["token_tags"]),
                retained_total_rows=len(keep),
            )
        if not torch.equal(state["video"], video):
            raise ValueError("Spatial audio video layout changed during its owned pass")
        for key in ("position_ids", "token_tags", "timestep_indices"):
            kwargs[key] = kwargs[key].index_select(0, state["keep"])
        kwargs["hidden_states"] = hidden.index_select(1, state["selected"])
        for key in ("text_indices", "audio_indices"):
            kwargs[key] = state["inverse"][kwargs[key]]
            if not torch.all(kwargs[key] >= 0):
                raise ValueError("Spatial audio dropped a live text or audio row")
        kwargs["video_indices"] = state["inverse"][video[state["selected"]]]
        record["forwards"] += 1

    hook = model.register_forward_pre_hook(before, with_kwargs=True)
    model._vflash_spatial_audio_active = True
    try:
        yield record
    finally:
        hook.remove()
        del model._vflash_spatial_audio_active
        state.clear()
