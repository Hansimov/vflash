"""Request-owned native AV target preservation with bounded AV clock rounding.

The context occupies the beginning of the generated target, rather than separate
reference rows. Callers remove that overlap in latent space and decode the joined
sequence together. This experimental primitive does not certify motion quality.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class NativeAVPrefix:
    video: object
    audio: object
    clean_timestep_tables: tuple
    source_frames: int
    context_frames: int
    preserve_video: bool = True

    @classmethod
    def from_latents(
        cls,
        latents,
        *,
        source_frames,
        context_frames,
        clean_timestep_tables,
        preserve_video=True,
    ):
        import torch

        from .h3_latent_layout import h3_video_latent_frame_count

        if (
            type(preserve_video) is not bool
            or type(source_frames) is not int
            or type(context_frames) is not int
            or not 0 < context_frames <= source_frames
            or source_frames % 17 != 5
            or context_frames % 17 != 5
            or context_frames % 3
        ):
            raise ValueError(
                "native AV needs VAE-aligned source and shared 24fps/40Hz overlap boundaries"
            )
        end = h3_video_latent_frame_count(source_frames)
        count = h3_video_latent_frame_count(context_frames)
        video, audio = latents["video_latents"], latents["audio_latents"]
        if (
            not isinstance(video, torch.Tensor)
            or not isinstance(audio, torch.Tensor)
            or video.ndim != 5
            or tuple(video.shape[:2]) != (1, 24)
            or audio.ndim != 3
            or tuple(audio.shape[:2]) != (2, 32)
            or video.shape[2] < end
            or audio.shape[-1] < (source_frames * 5 + 1) // 3
            or min(video.shape[-2:]) < 2
            or any(n % 2 for n in video.shape[-2:])
            or not video.is_floating_point()
            or not audio.is_floating_point()
            or (end - count) % 5
        ):
            raise ValueError("native AV source geometry or temporal phase differs")
        v = (
            video[:, :, end - count : end]
            .detach()
            .to(device="cpu", dtype=torch.float32, copy=True)
        )
        stop = (source_frames * 5 + 1) // 3
        a = (
            audio[:, :, stop - context_frames * 5 // 3 : stop]
            .detach()
            .to(device="cpu", dtype=torch.float32, copy=True)
        )
        if not v.isfinite().all() or not a.isfinite().all():
            raise ValueError("native AV context contains nonfinite values")
        tables = tuple(clean_timestep_tables)
        if len(tables) != 51:
            raise ValueError(
                "native AV context requires 50 matched block tables and a final table"
            )
        for i, table in enumerate(tables):
            shape = (1, 1, 2, 5376) if i == 50 else (1, 3, 6, 5376)
            if (
                not isinstance(table, torch.Tensor)
                or tuple(table.shape) != shape
                or not table.is_floating_point()
                or not table.isfinite().all()
            ):
                raise ValueError("clean timestep modulation table differs")
        return cls(v, a, tables, source_frames, context_frames, preserve_video)

    def join_audio(self, source, generated, *, generated_frames, delivery_frames=None):
        """Join native sound on its global clock, cropping at most one tail tick.

        This also supports a scene change: independently cut the new picture at
        context_frames and decode the joined native audio together. No video
        continuity or subjective sound qualification is implied. An explicit shorter
        delivery_frames may discard an undersupplied model-padding tail; the returned
        tensor retains available padding for decode. Trim decoded PCM to that delivery
        clock. Missing sound within the requested delivery is always rejected.
        """
        import torch

        if (
            type(generated_frames) is not int
            or generated_frames % 17 != 5
            or generated_frames <= self.context_frames
        ):
            raise ValueError("continuation must have a VAE-aligned nonempty suffix")
        stop = (self.source_frames * 5 + 1) // 3
        frames = self.source_frames + generated_frames - self.context_frames
        ticks = (frames * 5 + 1) // 3
        if delivery_frames is not None and (
            type(delivery_frames) is not int or not 0 < delivery_frames <= frames
        ):
            raise ValueError("audio delivery must fit within the joined picture clock")
        sa, na = source["audio_latents"], generated["audio_latents"]
        count = self.audio.shape[-1]
        if (
            sa.ndim != 3
            or na.ndim != 3
            or tuple(sa.shape[:2]) != (2, 32)
            or tuple(na.shape[:2]) != (2, 32)
            or sa.shape[-1] < stop
            or not torch.equal(sa[:, :, stop - count : stop].float().cpu(), self.audio)
            or not torch.equal(na[:, :, :count].float().cpu(), self.audio)
        ):
            raise ValueError(
                "join requires original sound and exactly preserved native audio prefix"
            )
        audio = torch.cat((sa[:, :, :stop], na[:, :, count:]), 2)
        difference = audio.shape[-1] - ticks
        required_ticks = ticks if delivery_frames is None else (delivery_frames * 5 + 2) // 3
        if not -1 <= difference <= 1 or audio.shape[-1] < required_ticks:
            raise ValueError("joined native AV clock is incomplete or differs")
        return audio[:, :, :ticks].contiguous()

    def join(self, source, generated, *, generated_frames):
        """Remove preserved AV overlap and retain the complete sequence clock.

        Source cuts use the nearest 40Hz tick, at most 1/120s from a frame boundary.
        Missing audio is rejected, never synthesized or stretched.
        """
        import torch

        from .h3_latent_layout import h3_video_latent_frame_count

        if not self.preserve_video:
            raise ValueError("audio-only history requires independently cut pictures")
        audio = self.join_audio(source, generated, generated_frames=generated_frames)
        end = h3_video_latent_frame_count(self.source_frames)
        count = self.video.shape[2]
        new_count = h3_video_latent_frame_count(generated_frames)
        frames = self.source_frames + generated_frames - self.context_frames
        sv, nv = source["video_latents"], generated["video_latents"]
        if (
            sv.ndim != 5
            or nv.ndim != 5
            or tuple(sv.shape[:2]) != (1, 24)
            or tuple(nv.shape[:2]) != (1, 24)
            or sv.shape[-2:] != nv.shape[-2:]
            or sv.shape[2] < end
            or nv.shape[2] != new_count
            or not torch.equal(sv[:, :, end - count : end].float().cpu(), self.video)
            or not torch.equal(nv[:, :, :count].float().cpu(), self.video)
        ):
            raise ValueError(
                "join requires the original source and exactly preserved native AV prefix"
            )
        video = torch.cat((sv[:, :, :end], nv[:, :, count:]), 2)
        if video.shape[2] != h3_video_latent_frame_count(frames):
            raise ValueError("joined native AV clock is incomplete or differs")
        return {"video_latents": video.contiguous(), "audio_latents": audio}


def _pack_video(video):
    _, channels, frames, height, width = video.shape
    return (
        video.reshape(1, channels, frames, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(1, frames * height * width // 4, channels * 4)
        .contiguous()
    )


class NativeAVPreservation:
    """One engine's target prefix, with no class or process-global method mutation."""

    def __init__(self, engine, request, *, audio_refinement=False):
        import torch

        from .h3_native_scheduler import H3NativeRowTimestepStep

        if not isinstance(request, NativeAVPrefix):
            raise ValueError("native AV preservation requires an explicit prefix")
        state = engine.latent_state
        spatial = request.video.shape[-2] * request.video.shape[-1] // 4
        if type(audio_refinement) is not bool:
            raise ValueError("native history needs an explicit refinement scope")
        trajectory_matches = (
            state.evaluation_index == 1
            and engine.schedule.nfe in (7, 13, 21)
            and not request.preserve_video
            and state.num_condition_video_rows == state.video_latents.shape[1]
            if audio_refinement
            else state.evaluation_index == 0
            and engine.schedule.nfe == 4
            and state.num_condition_video_rows in (spatial, 2 * spatial)
        )
        if (
            not trajectory_matches
            or state.num_condition_audio_rows
            or state.audio_latents.shape[1] % 2
            or engine._denoise_profile_records is not None
        ):
            raise ValueError(
                "native history requires fresh four-step AV or a frozen-video audio refinement"
            )
        self.engine, self.request, self.state = engine, request, state
        self.expected_evaluations = engine.schedule.nfe - state.evaluation_index
        self.offset = state.num_condition_video_rows
        self.video = _pack_video(request.video).to(device=state.device, dtype=torch.float32)
        self.audio = (
            request.audio.permute(0, 2, 1)
            .contiguous()
            .to(device=state.device, dtype=torch.float32)
        )
        self.ticks = self.audio.shape[1]
        if type(request.preserve_video) is not bool:
            raise ValueError("native history requires an explicit video preservation flag")
        if not request.preserve_video:
            self.video = self.video[:, :0]
        self.rows = self.video.shape[1]
        if (
            self.rows and self.offset + self.rows >= state.video_latents.shape[1]
        ) or self.ticks >= state.audio_latents.shape[1] // 2:
            raise ValueError("native AV prefix must leave a nonempty generated suffix")
        noise = state.video_latents[:, self.offset : self.offset + self.rows].clone()
        self.kept = 0.999 * self.video + 0.001 * noise
        locked_video = engine.video_indices[self.offset : self.offset + self.rows]
        locked_audio = engine.audio_indices.reshape(2, -1)[:, : self.ticks].reshape(-1)
        rows = []
        for row in engine.row_timestep_plan:
            if float(row.timesteps[-1]) >= 1:
                raise ValueError("base trajectory already contains clean audio modulation")
            times = torch.cat((row.timesteps, row.timesteps.new_tensor([1.0])))
            idx = row.timestep_indices.clone()
            pin = torch.searchsorted(times, times.new_tensor([0.999])).item()
            if times[pin] != times.new_tensor(0.999):
                raise ValueError("base trajectory lacks matching clean video modulation")
            idx[locked_video] = pin
            idx[locked_audio] = len(times) - 1
            rows.append(H3NativeRowTimestepStep(times, idx))
        self.plan = tuple(rows)
        self.evaluations = 0
        self.active = False

    def _restore(self):
        value = self.video if self.state.complete else self.kept
        self.state.video_latents[:, self.offset : self.offset + self.rows].copy_(value)
        self.state.audio_latents.reshape(2, -1, 32)[:, : self.ticks].copy_(self.audio)

    @contextmanager
    def preserve_targets(self):
        if self.active or self.evaluations:
            raise ValueError("native AV preservation scopes cannot be reused")
        original = self.engine.row_timestep_plan
        self.active = True
        try:
            self.engine.row_timestep_plan = self.plan
            self._restore()
            yield self
        finally:
            self.engine.row_timestep_plan = original
            self.active = False

    def advance(self, *, step=None):
        if not self.active:
            raise ValueError("native AV preservation scope is not active")
        if step is None:
            self.engine.step()
        else:
            step(self.engine, capture_packed_input=False)
        self._restore()
        self.evaluations += 1

    def record(self):
        import torch

        if not self.state.complete or self.evaluations != self.expected_evaluations:
            raise ValueError("native AV context trajectory is incomplete")
        v = self.state.video_latents[:, self.offset : self.offset + self.rows]
        a = self.state.audio_latents.reshape(2, -1, 32)[:, : self.ticks]
        if not torch.equal(v, self.video) or not torch.equal(a, self.audio):
            raise ValueError("native AV context was not preserved exactly")
        return {
            "context_frames": self.request.context_frames,
            "source_frames": self.request.source_frames,
            "audio_ticks": self.ticks,
            "preserved_video_rows": self.rows,
            "video_prefix_exact": True if self.request.preserve_video else None,
            "preserve_video": self.request.preserve_video,
            "audio_prefix_exact": True,
            "evaluations": self.evaluations,
        }

    @contextmanager
    def execution(self):
        """Own expanded clean-audio modulation, preserving all original turbo weights."""
        import torch

        from .h3_refinement import _ring_modulation

        engine = self.engine
        ring = engine.denoiser
        if not hasattr(ring, "host_blocks") or len(ring.host_blocks) != 50:
            raise ValueError("native AV context currently requires the complete block ring")
        original = engine.final_layer.weights

        def expand(table, extra, modalities):
            result = torch.zeros(
                (
                    engine.schedule.nfe,
                    (max(len(row.timesteps) for row in self.plan)) * modalities,
                    *table.shape[2:],
                ),
                device=table.device,
                dtype=table.dtype,
            )
            for i, row in enumerate(engine.row_timestep_plan):
                size = len(row.timesteps) * modalities
                result[i, :size] = table[i, :size]
                result[i, size : size + modalities] = extra[0].to(table.device, table.dtype)
            return result.contiguous()

        tables = [
            expand(w.adaln_table, extra, 3)
            for w, extra in zip(
                ring.host_blocks, self.request.clean_timestep_tables[:-1], strict=True
            )
        ]
        final = expand(original.adaln_table, self.request.clean_timestep_tables[-1], 1)
        with ExitStack() as scope:
            scope.enter_context(_ring_modulation(engine, [*tables, final], mode="face"))
            engine.final_layer.weights = replace(original, adaln_table=final)
            try:
                with self.preserve_targets():
                    yield self
            finally:
                engine.final_layer.weights = original
