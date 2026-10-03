"""Independent SelfLift-zero equations and H3 progressive-resolution probe.

Based on Wen et al., arXiv:2609.02036, equations 7-10. The paper evaluates
image backbones; this H3 adaptation is opt-in, not the default inference strategy.
No code is copied from the unlicensed community ComfyUI implementation.
"""

import math
import time


def consistency_lift(direct, pixel, *, rho=0.6, minimum=1.0, maximum=1.0):
    import torch

    if direct.shape != pixel.shape or direct.ndim != 5:
        raise ValueError("Lift branches must share BCTHW geometry")
    if not all(math.isfinite(v) for v in (rho, minimum, maximum)) or not (
        0 <= rho <= 1 and 0 <= minimum <= maximum <= 1
    ):
        raise ValueError("Invalid consistency selection or strength")
    delta = pixel.float() - direct.float()
    score = delta.abs().mean(dim=1, keepdim=True)
    if rho == 0:
        return direct.float(), dict(selected_fraction=0.0)
    # H3 temporal latents have phase-dependent scales. A T*H*W quantile
    # switches correction density by temporal phase and makes decoded detail
    # pulse. Select spatial risk independently at each latent time instead;
    # no temporal averaging, changed sampling clock or frame interpolation.
    flat = score.flatten(-2)
    threshold = torch.quantile(flat, 1 - rho, dim=-1, keepdim=True).unsqueeze(-1)
    selected = score >= threshold
    lo = (
        score.masked_fill(~selected, float("inf"))
        .flatten(-2)
        .amin(-1, keepdim=True)
        .unsqueeze(-1)
    )
    hi = flat.amax(-1, keepdim=True).unsqueeze(-1)
    weights = selected * (
        minimum + (maximum - minimum) * ((score - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
    )
    corrected = direct.float() + weights * delta
    return corrected, dict(
        selected_fraction=float(selected.float().mean()),
        selection="spatial_per_time",
    )


def sampler(state):
    """Return a request-local sampler for the pinned FreeVideo Engine interface.

    Prefix: six original-schedule forwards, retain its latest clean estimate and
    the evolved audio state. Suffix: re-noise only video at the SAME next clock,
    continue both modalities for the two remaining forwards. No extra NFE,
    no completed-low-video masquerading as a progressive transition.
    """

    if state.get("phase") not in {"prefix", "suffix"}:
        raise ValueError("SelfLift phase must be prefix or suffix")

    def generate(
        transformer,
        prompt_embeds,
        text_token_tags,
        num_frames,
        num_steps,
        seed,
        device,
        *,
        step_seconds=None,
        conditions=None,
    ):
        import torch
        from diffusers import MiniMaxH3Scheduler
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep,
            patchify_video_latents,
        )
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
            audio_latent_num_frames,
            video_latent_num_frames,
        )
        from src.models.hybrid_transform import iter_hybrids, set_layout
        from src.models.sequence_layout import layout_from_indices

        if num_steps != 8 or num_frames % 17 != 5:
            raise ValueError("Progressive control requires the original eight-step grid")
        anchors, clean = conditions or ((), [])
        h, w = state["height"] // 16, state["width"] // 16
        nf, na = video_latent_num_frames(num_frames, 17, 5), audio_latent_num_frames(num_frames)
        patch = tuple(transformer.config.patch_size)
        channels = transformer.config.in_channels
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags, nf, h, w, na, patch, 2, 2, 0, keyframe_anchors=anchors
        )
        positions, tags, vid, aud, txt, nc, _ = layout
        positions, tags, vid, aud, txt = (
            x.to(device) for x in (positions, tags, vid, aud, txt)
        )
        if next(iter_hybrids(transformer), None) is not None:
            set_layout(
                transformer,
                layout_from_indices(
                    vid[nc:],
                    nf,
                    (h // 2) * (w // 2),
                    seq_len=positions.shape[0],
                    frame_size=(h // 2, w // 2),
                    text_indices=txt,
                ),
            )
        vs, aus = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
        vs.set_timesteps(num_steps, device=device)
        aus.set_timesteps(num_steps, device=device)
        assert len(vs.timesteps) == len(aus.timesteps) == 8
        gen = torch.Generator(device).manual_seed(seed)

        def noise(shape):
            return torch.randn(shape, generator=gen, device=device, dtype=torch.float32)

        fixed = [
            patchify_video_latents(vs.scale_noise(x, 0.999, noise(x.shape)), patch)
            for x in clean
        ]
        start, end = (0, 6) if state["phase"] == "prefix" else (6, 8)
        if start == 0:
            video = noise((1, channels, nf, h, w))
            audio = noise((na * 2, 32))
        else:
            initial = state["video"].to(device).float()
            assert tuple(initial.shape) == (1, channels, nf, h, w)
            video = vs.scale_noise(initial, vs.timesteps[start], noise(initial.shape))
            audio = state["audio"].to(device).permute(0, 2, 1).reshape(-1, 32).contiguous()
            assert tuple(audio.shape) == (na * 2, 32)
        rows = patchify_video_latents(video, patch)
        if fixed:
            rows = torch.cat([*fixed, rows])
        assert rows.shape[0] == vid.numel()
        del fixed, video
        # The full eight-entry cached modulation clock must also resume at six.
        if state.get("cursor") is not None:
            state["cursor"].reset(start)

        def unpack(value):
            value = value.reshape(1, nf, h // patch[1], w // patch[2], channels, *patch)
            return (
                value.permute(0, 4, 1, 5, 2, 6, 3, 7)
                .reshape(1, channels, nf, h, w)
                .contiguous()
            )

        for index in range(start, end):
            began = time.monotonic()
            vt, at = vs.timesteps[index], aus.timesteps[index]
            times = torch.full((positions.shape[0],), float(vt), device=device)
            times[aud], times[vid[:nc]] = at, 0.999
            unique, inverse = torch.unique(times, sorted=True, return_inverse=True)
            pv, pa = transformer(
                hidden_states=rows[None],
                audio_hidden_states=audio[None],
                encoder_hidden_states=prompt_embeds[None],
                timestep=unique,
                timestep_indices=inverse,
                token_tags=tags,
                position_ids=positions,
                video_indices=vid,
                audio_indices=aud,
                text_indices=txt,
                return_dict=False,
            )
            if index == 5:
                estimate = unpack(rows[nc:].float() + (1.0 - vt) * pv[0, nc:].float())
            rows[nc:] = vs.step(pv[0, nc:].float(), vt, rows[nc:], return_dict=False)[0]
            audio = aus.step(pa[0].float(), at, audio, return_dict=False)[0]
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            if step_seconds is not None:
                step_seconds.append(time.monotonic() - began)
        video = estimate if end == 6 else unpack(rows[nc:])
        audio = audio.reshape(2, na, 32).permute(0, 2, 1).contiguous()
        state["clock"] = dict(
            prefix_nfe=6,
            suffix_nfe=2,
            video_resume=float(vs.timesteps[6]),
            audio_resume=float(aus.timesteps[6]),
        )
        return video, audio

    return generate
