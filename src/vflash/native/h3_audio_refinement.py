"""Owned audio-only refinement with fewer spatial video-context tokens.

Keep all temporal positions and the original video buffer. Uniformly retain
spatial tokens and their original RoPE coordinates; this is approximate context
thinning, not a new low-resolution video render or equivalent audio execution.
The original model weights are shared for this serial, request-owned scope.
"""

import math
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextSelection:
    grid: tuple[int, int, int]
    source_grid: tuple[int, int, int]
    video_rows: object
    packed_rows: object
    video_indices: object
    audio_indices: object
    text_indices: object
    condition_rows: int
    protected_rows: int


def _attention_blocks(engine):
    """One attention owner shared by all resident blocks or both ring slots."""
    denoiser = engine.denoiser
    if hasattr(denoiser, "host_blocks"):
        if len(denoiser.host_blocks) != 50 or len(denoiser.slots) != 2:
            raise ValueError("audio context needs the complete two-slot native ring")
        return denoiser.slots
    blocks = getattr(denoiser, "blocks", ())
    if len(blocks) != 50:
        raise ValueError("audio context needs all 50 native blocks")
    return blocks


def select_context(engine, *, max_pixels=500_000):
    import torch

    if type(max_pixels) is not int or max_pixels != 500_000:
        raise ValueError("audio context reduction supports the qualified 0.5MP budget")
    state = engine.latent_state
    blocks = _attention_blocks(engine)
    if (
        not engine.complete
        or state.num_condition_audio_rows
        or not state.audio_latents.numel()
        or engine._denoise_profile_records is not None
    ):
        raise ValueError("context reduction needs completed AV generation")
    attention = getattr(blocks[0], "_veda_attention", None)
    if attention is None or any(
        getattr(b, "_veda_attention", None) is not attention for b in blocks
    ):
        raise ValueError("context reduction requires one owned Veda layout")
    if attention.dense_layers != frozenset((0, 1, 2, 47, 48, 49)):
        raise ValueError("context reduction keeps the accepted 44/6 attention policy")
    grid = tuple(attention.layout.target.grid)
    if len(grid) != 3 or any(type(v) is not int or v <= 0 for v in grid):
        raise ValueError("invalid source video grid")
    frames, height, width = grid
    spatial = height * width
    conditions = state.num_condition_video_rows
    if conditions not in (0, spatial, 2 * spatial) or state.video_latents.shape != (
        1,
        conditions + frames * spatial,
        96,
    ):
        raise ValueError("only same-canvas temporal keyframe context is supported")
    # H3 packs a 2x2 patch of 16x-downsampled VAE latents per video row.
    if spatial * 32**2 <= max_pixels:
        return None
    ratio = math.sqrt(max_pixels / (spatial * 32**2))
    reduced_h, reduced_w = int(height * ratio), int(width * ratio)
    if min(reduced_h, reduced_w) < 8:
        raise ValueError("context canvas is below the qualified minimum width")
    device = state.device
    ys = torch.linspace(0, height - 1, reduced_h, device=device).round().long()
    xs = torch.linspace(0, width - 1, reduced_w, device=device).round().long()
    plane = (ys[:, None] * width + xs[None, :]).reshape(-1)
    chunks = frames + conditions // spatial
    selected = (torch.arange(chunks, device=device)[:, None] * spatial + plane).reshape(-1)
    old_video = engine.video_indices.index_select(0, selected)
    packed = torch.cat((old_video, engine.audio_indices, engine.text_indices)).sort().values
    if torch.unique(packed).numel() != packed.numel():
        raise ValueError("source packed modalities overlap")
    mapping = torch.full((engine.sequence_length,), -1, dtype=torch.long, device=device)
    mapping[packed] = torch.arange(packed.numel(), device=device)
    new_conditions = conditions // spatial * plane.numel()
    video = mapping[old_video]
    audio, text = mapping[engine.audio_indices], mapping[engine.text_indices]
    protected = new_conditions + audio.numel() + text.numel()
    if not torch.equal(
        video[new_conditions:], torch.arange(protected, packed.numel(), device=device)
    ):
        raise ValueError("reduced context lost its protected prefix or video suffix")
    return ContextSelection(
        (frames, reduced_h, reduced_w),
        grid,
        selected,
        packed,
        video,
        audio,
        text,
        new_conditions,
        protected,
    )


def _attention_view(original, selection):
    from veda_comfy.core.engine import VedaEngine
    from veda_comfy.core.h3_layout import LayoutSpec, SpanSpec

    view = copy(original)
    owner = original.engine
    # Reuse immutable predictor tensors, but not mutable geometry caches/stats.
    view.engine = VedaEngine(
        owner.bundle, owner.generated, owner.reference, owner.backend, owner.device
    )
    view.layout = LayoutSpec(
        selection.packed_rows.numel(),
        SpanSpec("video", selection.protected_rows, selection.grid),
        (),
    )
    view.choice = view.engine.plan_for(view.layout)
    view.sparse_calls = view.dense_calls = 0
    return view


@contextmanager
def reduced_audio_context(engine, *, max_pixels=500_000):
    from vflash.native.h3_native_engine import H3NativeEngine
    from vflash.native.h3_native_scheduler import H3NativeLatentState

    selection = select_context(engine, max_pixels=max_pixels)
    if selection is None:
        yield engine, {"applied": False, "reason": "already_within_pixel_budget"}
        return
    state = engine.latent_state
    reduced = H3NativeLatentState(
        engine.schedule,
        video_latents=state.video_latents.index_select(1, selection.video_rows),
        audio_latents=state.audio_latents.clone(),
        num_condition_video_rows=selection.condition_rows,
        num_condition_audio_rows=0,
    )
    reduced.evaluation_index = state.evaluation_index
    view = H3NativeEngine(
        schedule=engine.schedule,
        latent_state=reduced,
        input_packer=engine.input_packer,
        denoiser=engine.denoiser,
        final_layer=engine.final_layer,
        refined_text=engine.refined_text,
        video_indices=selection.video_indices,
        audio_indices=selection.audio_indices,
        text_indices=selection.text_indices,
        token_tags=engine.token_tags.index_select(0, selection.packed_rows),
        rotary_cos=engine.rotary_cos.index_select(0, selection.packed_rows),
        rotary_sin=engine.rotary_sin.index_select(0, selection.packed_rows),
    )
    blocks = _attention_blocks(engine)
    original = blocks[0]._veda_attention
    attention = _attention_view(original, selection)
    record = {
        "applied": True,
        "method": "uniform-spatial-token-subsample-original-rope",
        "approximate": True,
        "source_grid": list(selection.source_grid),
        "context_grid": list(selection.grid),
        "context_pixels": math.prod(selection.grid[1:]) * 32**2,
        "source_video_rows": state.video_latents.shape[1],
        "context_video_rows": reduced.video_latents.shape[1],
        "all_temporal_positions_retained": True,
        "source_video_buffer_reused": False,
        "dense_layers": sorted(original.dense_layers),
    }
    try:
        for block in blocks:
            block._veda_attention = attention
        yield view, record
    finally:
        for block in blocks:
            block._veda_attention = original


@dataclass(frozen=True)
class AudioRefinementRequest:
    """Six audio evaluations with original video pixels and time positions kept.

    Compiled tables and schedule must describe the same sigma-0.5 six-step pass.
    A None spatial budget explicitly retains the full video context for control.
    Weights belong to the caller and are never downloaded or modified on disk.
    """

    schedule: object
    tables: tuple
    context_pixels: int | None = 500_000

    def __post_init__(self):
        from .h3_native_scheduler import H3NativeSchedule
        from .h3_refinement_schedule import refinement_schedule

        expected = refinement_schedule(steps=6, denoise=0.5)
        if (
            not isinstance(self.schedule, H3NativeSchedule)
            or self.schedule.to_mapping() != expected.to_mapping()
            or len(self.tables) != 51
            or any(table.shape[0] != expected.nfe for table in self.tables)
        ):
            raise ValueError("audio refinement requires matched six-step sigma-0.5 tables")
        if self.context_pixels is not None and (
            type(self.context_pixels) is not int or self.context_pixels != 500_000
        ):
            raise ValueError(
                "audio refinement needs full context or the qualified 0.5MP budget"
            )

    @property
    def steps(self):
        return 6


def _refine_full_audio(engine, request, *, step, noise):
    import time

    import torch

    from .h3_refinement import quality_pass

    state = engine.latent_state
    if not engine.complete or state.num_condition_audio_rows or not state.audio_latents.numel():
        raise ValueError("audio refinement requires completed unconditioned AV output")
    original_audio = state.audio_latents.clone()
    original_video = state.video_latents.clone()
    if noise.shape != original_audio.shape or not noise.isfinite().all():
        raise ValueError("audio refinement noise differs")
    started = time.monotonic()
    try:
        with quality_pass(engine, request.schedule, request.tables, mode="audio"):
            sigma = request.schedule.audio_sigmas[1]
            state.audio_latents.copy_((1 - sigma) * original_audio + sigma * noise)
            for _ in range(request.steps):
                step(engine, capture_packed_input=False)
            if state.device.type == "cuda":
                torch.cuda.synchronize(state.device)
            if (
                not torch.equal(original_video, state.video_latents)
                or not state.audio_latents.isfinite().all()
            ):
                raise ValueError("invalid audio-only refinement output")
    except BaseException:
        state.video_latents.copy_(original_video)
        state.audio_latents.copy_(original_audio)
        raise
    return original_audio, {
        "extra_audio_nfe": request.steps,
        "sigma": sigma,
        "extra_seconds": time.monotonic() - started,
        "video_latents_identical": True,
        "turbo_residuals_disabled": True,
        "new_matched_modulation": True,
        "video_cache": False,
        "upstream_equivalent": False,
    }


def refine_audio(engine, request, *, noise, step=None, progress_callback=None):
    """Commit only successful audio; restore the main trajectory and attention.

    Pass the original initial audio noise, captured before main generation. The
    caller owns this serial engine scope. Main video/audio conditioning, cache
    layout and declared four-step trajectory remain separate from added work.
    """
    import time
    from contextlib import nullcontext

    import torch

    if not isinstance(request, AudioRefinementRequest):
        raise ValueError("audio refinement needs its explicit request")
    if engine.schedule.nfe != 4:
        raise ValueError("audio refinement requires the qualified four-step base")
    if step is None:

        def step(owner, **kwargs):
            return owner.step(**kwargs)

    evaluations = 0

    def advance(owner, **kwargs):
        nonlocal evaluations
        result = step(owner, **kwargs)
        evaluations += 1
        if progress_callback is not None:
            progress_callback(evaluations, request.steps)
        return result

    context = (
        reduced_audio_context(engine, max_pixels=request.context_pixels)
        if request.context_pixels is not None
        else nullcontext((engine, None))
    )
    started = time.monotonic()
    with context as (view, selection):
        baseline, record = _refine_full_audio(view, request, step=advance, noise=noise)
        if view is not engine:
            engine.latent_state.audio_latents.copy_(view.latent_state.audio_latents)
    if engine.latent_state.device.type == "cuda":
        torch.cuda.synchronize(engine.latent_state.device)
    record["extra_seconds"] = time.monotonic() - started
    if selection is not None:
        record["context_reduction"] = selection
    return baseline, record
