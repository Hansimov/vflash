"""Owned low-noise H3 face repaint over an already loaded native trunk.

The caller owns face selection, video crops, decoding and compositing. This
module neither downloads weights nor changes the ordinary generation default.
PRO 6000 resident-trunk complete-media evidence accompanies the initial API;
block-ring support requires its own target-device qualification.
"""

import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace


def refinement_plan(engine, schedule, *, mode):
    import torch

    from vflash.native.h3_native_scheduler import (
        H3NativeRowTimestepStep,
        compile_h3_row_timestep_plan,
    )

    from .h3_refinement_schedule import timestep_rows

    state = engine.latent_state
    plan = compile_h3_row_timestep_plan(
        schedule,
        video_indices=engine.video_indices,
        audio_indices=engine.audio_indices,
        text_indices=engine.text_indices,
        num_condition_video_rows=state.num_condition_video_rows,
        num_condition_audio_rows=state.num_condition_audio_rows,
        device=state.device,
    )
    result = []
    for step, rows in zip(plan, timestep_rows(schedule), strict=True):
        rows = rows.to(state.device)
        # Frozen rows can remove a timestep from the observed plan. Keep the
        # compiled table's full sorted row numbering instead of indexing by
        # the smaller observed set (which would silently apply wrong AdaLN).
        lookup = torch.searchsorted(rows, step.timesteps)
        if not torch.equal(rows[lookup], step.timesteps):
            raise ValueError("refinement timestep has no compiled modulation")
        indices = lookup[step.timestep_indices]
        clean = int((rows == schedule.keyframe_noise_aug).nonzero().item())
        indices[engine.video_indices if mode == "audio" else engine.audio_indices] = clean
        result.append(H3NativeRowTimestepStep(rows, indices))
    return tuple(result)


@contextmanager
def _ring_modulation(engine, tables):
    """Rebind only modulation storage and its prevalidated copy pairs.

    The two-slot executor captures tensor addresses during construction; merely
    replacing host_blocks would leave stale four-step tables in those pairs.
    Synchronize both entry/exit boundaries before restoring their owned storage.
    """
    import torch

    from .h3_native_denoiser import _bf16_block_copy_pairs, _block_tensor_bytes

    ring = engine.denoiser
    if len(ring.host_blocks) != 50 or len(ring.slots) != 2:
        raise ValueError("face refinement requires the complete two-slot native ring")
    names = (
        "host_blocks",
        "_ring_copy_pairs",
        "_serial_copy_pairs",
        "_block_weight_bytes",
        "host_weight_bytes",
        "device_slot_weight_bytes",
    )
    saved = {name: getattr(ring, name) for name in names}
    weights = [slot.weights for slot in ring.slots]
    torch.cuda.synchronize(engine.latent_state.device)
    try:
        host_tables = [table.to("cpu").contiguous() for table in tables[:-1]]
        host_tables = [t if t.is_pinned() else t.pin_memory() for t in host_tables]
        ring.host_blocks = tuple(
            replace(w, adaln_table=t)
            for w, t in zip(saved["host_blocks"], host_tables, strict=True)
        )
        for slot, table in zip(ring.slots, host_tables[:2], strict=True):
            slot.weights = replace(
                slot.weights, adaln_table=torch.empty_like(table, device=ring.device)
            )
        ring._ring_copy_pairs = tuple(
            _bf16_block_copy_pairs(ring.slots[i % 2].weights, w)
            for i, w in enumerate(ring.host_blocks)
        )
        ring._serial_copy_pairs = tuple(
            _bf16_block_copy_pairs(ring.slots[0].weights, w) for w in ring.host_blocks
        )
        ring._block_weight_bytes = tuple(_block_tensor_bytes(w) for w in ring.host_blocks)
        ring.host_weight_bytes = sum(ring._block_weight_bytes)
        ring.device_slot_weight_bytes = sum(_block_tensor_bytes(s.weights) for s in ring.slots)
        yield
    finally:
        torch.cuda.synchronize(engine.latent_state.device)
        for name, value in saved.items():
            setattr(ring, name, value)
        for slot, weight in zip(ring.slots, weights, strict=True):
            slot.weights = weight


@contextmanager
def quality_pass(engine, schedule, tables, *, mode):
    """One serial pass; never mutate shared checkpoint files or main metadata."""
    from vflash.native.h3_native_scheduler import H3NativeLatentState

    if mode not in ("audio", "face"):
        raise ValueError("unknown refinement modality")
    state = engine.latent_state
    ring = hasattr(engine.denoiser, "host_blocks")
    blocks = engine.denoiser.slots if ring else engine.denoiser.blocks
    block_count = len(engine.denoiser.host_blocks) if ring else len(blocks)
    if ring and mode != "face":
        raise ValueError("block-ring refinement currently supports face repaint only")
    if block_count != 50 or len(tables) != 51 or engine._denoise_profile_records is not None:
        raise ValueError("refinement requires an unprofiled complete 50-block native engine")
    original = (engine.schedule, state, engine.row_timestep_plan, engine.final_layer.weights)
    weights = [block.weights for block in blocks]
    counters = []
    for block in blocks:
        attention = getattr(block, "_veda_attention", None)
        if attention is not None and not any(attention is row[0] for row in counters):
            counters.append((attention, attention.sparse_calls, attention.dense_calls))
    completed = False
    stack = ExitStack()
    try:
        plan = refinement_plan(engine, schedule, mode=mode)
        if ring:
            stack.enter_context(_ring_modulation(engine, tables))
        else:
            for block, table in zip(blocks, tables[:-1], strict=True):
                options = dict(adaln_table=table.to(state.device))
                if mode == "audio":
                    options.update(
                        qkv_residuals=(),
                        attention_out_residual=None,
                        ffn_in_residual=None,
                        ffn_out_residual=None,
                    )
                block.weights = replace(block.weights, **options)
        final = original[3]
        if final.video_head_plans.shape[2] != 1 or final.audio_head_plans.shape[2] != 1:
            raise ValueError("refinement requires the single original output head")
        engine.final_layer.weights = replace(
            final,
            adaln_table=tables[-1].to(state.device),
            video_head_plans=final.video_head_plans[:1].expand(schedule.nfe, -1, -1),
            audio_head_plans=final.audio_head_plans[:1].expand(schedule.nfe, -1, -1),
        )
        engine.latent_state = H3NativeLatentState(
            schedule,
            video_latents=state.video_latents,
            audio_latents=state.audio_latents,
            num_condition_video_rows=state.video_latents.shape[1]
            if mode == "audio"
            else state.num_condition_video_rows,
            num_condition_audio_rows=state.audio_latents.shape[1]
            if mode == "face"
            else state.num_condition_audio_rows,
        )
        engine.latent_state.evaluation_index = 1
        engine.schedule, engine.row_timestep_plan = schedule, plan
        yield
        completed = True
    finally:
        (
            engine.schedule,
            engine.latent_state,
            engine.row_timestep_plan,
            engine.final_layer.weights,
        ) = original
        stack.close()
        if not ring:
            for block, weight in zip(blocks, weights, strict=True):
                block.weights = weight
        # Audio is additional work after the original declared trajectory.
        # Face repaint *replaces* that trajectory, so successful actual calls
        # must remain visible to the engine's final coverage validation.
        if mode == "audio" or not completed:
            for attention, sparse, dense in counters:
                attention.sparse_calls, attention.dense_calls = sparse, dense


class VideoRepaint:
    """One low-noise evaluation per ordinary worker progress event."""

    def __init__(self, engine, schedule, tables, encoded_rows):

        self.engine, self.state, self.schedule = engine, engine.latent_state, schedule
        state = self.state
        suffix = state.video_latents[:, state.num_condition_video_rows :]
        if (
            state.evaluation_index != 0
            or schedule.nfe - 1 != state.schedule.nfe
            or encoded_rows.shape != suffix.shape
            or not encoded_rows.isfinite().all()
        ):
            raise ValueError("face repaint geometry or evaluation contract differs")
        self.video, self.audio = state.video_latents.clone(), state.audio_latents.clone()
        self.encoded = encoded_rows.to(state.device)
        self.started = time.monotonic()
        self.tables = (
            [table.to("cpu").pin_memory() for table in tables]
            if hasattr(engine.denoiser, "host_blocks")
            else [table.to(state.device) for table in tables]
        )
        self.record = None

    def advance(self, *, step):
        import torch

        engine, state, schedule = self.engine, self.state, self.schedule
        evaluation = state.evaluation_index
        if engine.latent_state is not state or state.complete:
            raise ValueError("face repaint lost its original request owner")
        try:
            with quality_pass(engine, schedule, self.tables, mode="face"):
                engine.latent_state.evaluation_index = evaluation + 1
                if evaluation == 0:
                    sigma = schedule.video_sigmas[1]
                    state.video_latents[:, state.num_condition_video_rows :].mul_(sigma).add_(
                        self.encoded, alpha=1 - sigma
                    )
                result = step(engine, capture_packed_input=False)
            state.evaluation_index = evaluation + 1
            if state.complete:
                if state.device.type == "cuda":
                    torch.cuda.synchronize(state.device)
                if (
                    not torch.equal(self.audio, state.audio_latents)
                    or not state.video_latents.isfinite().all()
                ):
                    raise ValueError("invalid face repaint output")
                self.record = {
                    "repaint_nfe": schedule.nfe - 1,
                    "sigma": schedule.video_sigmas[1],
                    "repaint_seconds": time.monotonic() - self.started,
                    "audio_latents_identical": True,
                    "turbo_residuals_retained": True,
                    "new_matched_modulation": True,
                    "incremental_progress": True,
                }
            return result
        except BaseException:
            state.video_latents.copy_(self.video)
            state.audio_latents.copy_(self.audio)
            state.evaluation_index = 0
            raise
