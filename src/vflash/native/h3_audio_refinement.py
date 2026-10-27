"""Owned audio-only refinement with fewer spatial video-context tokens.

Keep all temporal positions and the original video buffer. Uniformly retain
spatial tokens and their original RoPE coordinates; this is approximate context
thinning, not a new low-resolution video render or equivalent audio execution.
The original model weights are shared for this serial, request-owned scope.
"""

import math
from contextlib import contextmanager, nullcontext
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


@dataclass(frozen=True)
class EncodedAudioSource:
    """Clean VAE-encoded target rows for repairing an existing soundtrack.

    Video excludes the conditioning keyframe prefix; audio is channel-major.
    Both tensors use normalized H3 model coordinates. The caller retains the
    original picture stream for final muxing, avoiding a video VAE round trip.
    """

    video_rows: object
    audio_rows: object


def install_encoded_audio_source(engine, source):
    """Seed a new owned request from encoded media, with zero video evaluations."""
    import torch

    if not isinstance(source, EncodedAudioSource):
        raise ValueError("audio repair requires explicit encoded source latents")
    state = engine.latent_state
    target = state.video_latents[:, state.num_condition_video_rows :]
    if state.evaluation_index != 0 or state.num_condition_audio_rows:
        raise ValueError("audio repair requires a fresh unconditioned audio request")
    for value, expected in (
        (source.video_rows, target),
        (source.audio_rows, state.audio_latents),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.shape != expected.shape
            or not bool(value.isfinite().all())
        ):
            raise ValueError("encoded source geometry or finite latent values differ")
    video = source.video_rows.to(device=state.device, dtype=target.dtype)
    audio = source.audio_rows.to(device=state.device, dtype=state.audio_latents.dtype)
    if not bool(video.isfinite().all() and audio.isfinite().all()):
        raise ValueError("encoded source overflows native model coordinates")
    target.copy_(video)
    state.audio_latents.copy_(audio)
    state.evaluation_index = state.schedule.nfe
    return {
        "kind": "encoded-av-audio-repair",
        "video_nfe": 0,
        "encoded_video_rows": target.shape[1],
        "encoded_audio_rows": audio.shape[1],
    }


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


# Qualified spatial budgets for the audio pass's video context (0.25MP added in 0.15.1).
CONTEXT_PIXEL_BUDGETS = (250_000, 500_000)


def select_context(engine, *, max_pixels=500_000):
    import torch

    if type(max_pixels) is not int or max_pixels not in CONTEXT_PIXEL_BUDGETS:
        raise ValueError(
            "audio context reduction supports the qualified 0.25MP or 0.5MP budget"
        )
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
    if attention.dense_layers not in {
        frozenset((0, 1, 2, 47, 48, 49)),
        frozenset((0, 1, 2, 3, 46, 47, 48, 49)),
    }:
        raise ValueError("context reduction requires a declared 44/6 or 42/8 partition")
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
    """Explicit audio refinement with original video pixels and time positions kept.

    The default is six steps at denoise 0.5 (shifted audio sigma 0.75).
    Longer candidates use twelve steps at denoise 0.9 or twenty at 0.95.
    Each recipe requires its own matching compiled modulation tables.
    A None spatial budget explicitly retains the full video context for control.
    Hidden reuse freezes first-evaluation non-audio states, with fresh audio and
    rebuilt context K/V on each step. This is an explicit approximation, not an
    exact cache; the request owns all hooks and releases them on every exit.
    weight_cache_bytes optionally retains a prefix of BF16 ring weights on device
    for this pass while keeping slot addresses and arithmetic unchanged. The
    default zero preserves ordinary residency; the scope reserves 20 GiB for work.
    Weights belong to the caller and are never downloaded or modified on disk.
    native_audio_prefix optionally preserves already generated context on every
    evaluation. Its clean tables must match the refinement's original FL weights,
    and preserve_video must be False: this pass freezes the entire video already.
    """

    schedule: object
    tables: tuple
    context_pixels: int | None = 500_000
    reuse_hidden_context: bool = False
    reuse_kv_context: bool = False
    evaluations: int = 6
    denoise: float = 0.5
    dense_attention: bool = False
    weight_cache_bytes: int = 0
    native_audio_prefix: object | None = None

    def __post_init__(self):
        from .h3_av_continuation import NativeAVPrefix
        from .h3_native_scheduler import H3NativeSchedule
        from .h3_refinement_schedule import refinement_schedule

        if self.native_audio_prefix is not None and (
            not isinstance(self.native_audio_prefix, NativeAVPrefix)
            or self.native_audio_prefix.preserve_video
        ):
            raise ValueError("audio refinement history requires an audio-only native prefix")
        if type(self.weight_cache_bytes) is not int or self.weight_cache_bytes < 0:
            raise ValueError("audio weight cache requires a nonnegative integer byte budget")

        if (
            type(self.reuse_hidden_context) is not bool
            or type(self.reuse_kv_context) is not bool
            or (self.reuse_hidden_context and self.reuse_kv_context)
        ):
            raise ValueError(
                "audio context caches require explicit boolean choices with at most one enabled"
            )
        if (
            type(self.evaluations) is not int
            or type(self.denoise) not in (int, float)
            or (self.evaluations, self.denoise)
            not in {
                (6, 0.5),
                (12, 0.9),
                (20, 0.95),
            }
        ):
            raise ValueError("audio requires a matched six, twelve or twenty-step recipe")
        if type(self.dense_attention) is not bool or (
            self.dense_attention and self.evaluations != 20
        ):
            raise ValueError("dense audio context is an explicit twenty-step audio option")
        expected = refinement_schedule(steps=self.evaluations, denoise=self.denoise)
        if (
            not isinstance(self.schedule, H3NativeSchedule)
            or self.schedule.to_mapping() != expected.to_mapping()
            or len(self.tables) != 51
            or any(table.shape[0] != expected.nfe for table in self.tables)
        ):
            raise ValueError(
                "audio refinement requires matched audio schedule and modulation tables"
            )
        if self.context_pixels is not None and (
            type(self.context_pixels) is not int
            or self.context_pixels not in CONTEXT_PIXEL_BUDGETS
        ):
            raise ValueError(
                "audio refinement needs full context or the qualified 0.25MP or 0.5MP budget"
            )

    def matches_native_history(self, prefix):
        """The base pass and refinement must protect the same original audio."""
        import torch

        from .h3_av_continuation import NativeAVPrefix

        history = self.native_audio_prefix
        return (
            isinstance(prefix, NativeAVPrefix)
            and history is not None
            and history.source_frames == prefix.source_frames
            and history.context_frames == prefix.context_frames
            and torch.equal(history.audio, prefix.audio)
        )

    @property
    def steps(self):
        return self.evaluations


@contextmanager
def dense_audio_attention(engine, enabled):
    """Temporarily run every audio-pass context block through its dense backend."""
    if not enabled:
        yield None
        return
    owners = []
    for block in _attention_blocks(engine):
        attention = getattr(block, "_veda_attention", None)
        if attention is None:
            raise ValueError("dense audio context requires the owned Veda attention adapter")
        if all(attention is not row[0] for row in owners):
            owners.append((attention, attention.dense_layers))
    try:
        for attention, _ in owners:
            attention.dense_layers = frozenset(range(50))
        yield {"dense_layers": list(range(50))}
    finally:
        for attention, layers in owners:
            attention.dense_layers = layers


def _refine_full_audio(engine, request, *, step, noise):
    import time

    import torch

    from .h3_audio_hidden_cache import frozen_video_hidden
    from .h3_audio_kv_cache import frozen_video_kv
    from .h3_av_continuation import NativeAVPreservation
    from .h3_refinement import quality_pass
    from .h3_ring_weight_cache import cached_ring_weights

    state = engine.latent_state
    if not engine.complete or state.num_condition_audio_rows or not state.audio_latents.numel():
        raise ValueError("audio refinement requires completed unconditioned AV output")
    original_audio = state.audio_latents.clone()
    original_video = state.video_latents.clone()
    if noise.shape != original_audio.shape or not noise.isfinite().all():
        raise ValueError("audio refinement noise differs")
    prefix = request.native_audio_prefix
    if prefix is not None and not torch.equal(
        original_audio.reshape(2, -1, 32)[:, : prefix.audio.shape[-1]].float().cpu(),
        prefix.audio.permute(0, 2, 1).float().cpu(),
    ):
        raise ValueError("audio refinement history differs from the retained native source")
    history_record = None
    started = time.monotonic()
    try:
        with (
            dense_audio_attention(engine, request.dense_attention),
            quality_pass(engine, request.schedule, request.tables, mode="audio"),
        ):
            sigma = request.schedule.audio_sigmas[1]
            state.audio_latents.copy_((1 - sigma) * original_audio + sigma * noise)
            cache_scope = (
                frozen_video_hidden(engine)
                if request.reuse_hidden_context
                else frozen_video_kv(engine)
                if request.reuse_kv_context
                else nullcontext(None)
            )
            history_scope = (
                NativeAVPreservation(engine, prefix, audio_refinement=True).execution()
                if prefix is not None
                else nullcontext(None)
            )
            # Cache only after history expands the timestep tables; cached addresses
            # must belong to this scope, and be released before its tables restore.
            with (
                history_scope as history,
                cached_ring_weights(engine, request.weight_cache_bytes) as weight_cache,
                cache_scope as cache,
            ):
                for _ in range(request.steps):
                    if history is None:
                        step(engine, capture_packed_input=False)
                    else:
                        history.advance(step=step)
                if history is not None:
                    history_record = history.record()
                if cache is not None and (
                    cache["build_block_calls"] != 50
                    or cache["reuse_block_calls"] != 50 * (request.steps - 1)
                    or cache["rebuilt_projection_rows"]
                    != (
                        0
                        if request.reuse_kv_context
                        else 50 * (request.steps - 1) * cache["sequence_length"]
                    )
                ):
                    raise ValueError("audio cache did not execute every declared evaluation")
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
        **({"native_audio_prefix": history_record} if history_record is not None else {}),
        "extra_audio_nfe": request.steps,
        "sigma": sigma,
        "extra_seconds": time.monotonic() - started,
        "video_latents_identical": True,
        "turbo_residuals_disabled": True,
        "new_matched_modulation": True,
        "video_cache": dict(cache) if cache is not None else False,
        "upstream_equivalent": False,
        "dense_attention": request.dense_attention,
        "weight_cache": dict(weight_cache) if weight_cache is not None else False,
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
