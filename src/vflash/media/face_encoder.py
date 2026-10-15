"""Scoped original-H3 VAE encoder profiles for bounded refinement crops.

Temporal clips remain independent with the original padding/token drop. AMP is
an explicitly approximate candidate, not an equivalent or production default.
Only convolution/linear execution changes; weights and posterior sampling stay
owned by the original model. No decoder or attention replacement is used.
"""

from contextlib import contextmanager, nullcontext

PROFILES = {
    "original": (1, "fp32"),
    "batch2-fp32": (2, "fp32"),
    "batch2-fp16": (2, "fp16"),
    "batch4-fp16": (4, "fp16"),
    "batch2-fp16-norm": (2, "fp16"),
    "batch2-fp16-norm-reuse": (2, "fp16"),
    "batch2-fp16-norm-channels-last": (2, "fp16"),
}


def frame_group_norm(module, x, *, channels_last=False, affine=None):
    """Original per-frame groups, with half storage and native FP32 reduction.

    CUDA GroupNorm requires affine tensors matching the input dtype. Use scoped
    FP16 copies, preserving the original FP32 parameters. Native CUDA statistics
    accumulate in FP32. Affine/output rounding is an explicit approximation.
    """
    import torch
    import torch.nn.functional as functional

    if x.ndim != 5 or x.dtype != torch.float16:
        raise ValueError("half frame norm requires NCTHW FP16 activations")
    n, c, t, h, w = x.shape
    frames = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    frames = frames.contiguous(
        memory_format=torch.channels_last if channels_last else torch.contiguous_format
    )
    weight, bias = (
        affine
        if affine is not None
        else (module.weight.to(dtype=x.dtype), module.bias.to(dtype=x.dtype))
    )
    with torch.autocast(x.device.type, enabled=False):
        normalized = functional.group_norm(frames, module.num_groups, weight, bias, module.eps)
    return normalized.reshape(n, t, c, h, w).permute(0, 2, 1, 3, 4)


@contextmanager
def face_encoder(vae, profile="original"):
    import torch

    if profile not in PROFILES:
        raise ValueError("unknown explicit face encoder trial")
    batch, precision = PROFILES[profile]
    report = {
        "profile": profile,
        "chunk_batch": batch,
        "compute_precision": precision,
        "approximate": precision != "fp32",
        "calls": 0,
        "temporal_clips": 0,
    }
    if profile == "original":
        report.update(calls=None, temporal_clips=None)
        yield report
        return
    if vae.config.clip_length != 17 or vae.config.token_drop != 3:
        raise ValueError("face encoder trial requires original H3 temporal geometry")
    owned = "_encode" in vae.__dict__
    saved = vae.__dict__.get("_encode")
    channels_last = profile.endswith("channels-last")
    norm_owners = []

    def encode(x):
        if (
            x.ndim != 5
            or x.shape[0] != 1
            or x.shape[-2] != x.shape[-1]
            or x.shape[-1] not in (320, 384, 512)
        ):
            raise ValueError("face encoder trial requires one complete bounded square crop")
        if not 120 <= x.shape[2] <= 362:
            raise ValueError("face encoder trial requires a bounded native timeline")
        if precision == "fp16" and x.device.type != "cuda":
            raise ValueError("face AMP trial requires CUDA")
        if profile == "batch2-fp16-norm-reuse" and x.shape[2] == 124:
            from vflash.media.clip_reuse import encode_unique

            reuse = {}
            result = encode_unique(vae, x, reuse)
            report["calls"] += 1
            report["temporal_clips"] += reuse["input_clips"]
            report["exact_clip_reuse"] = reuse
            return result
        length = vae.config.clip_length
        tail = (-x.shape[2]) % length
        if tail:
            x = torch.cat((x, x[:, :, -1:].expand(-1, -1, tail, -1, -1)), dim=2)
        chunks = x.split(length, dim=2)
        results = []
        scope = (
            torch.autocast("cuda", dtype=torch.float16)
            if precision == "fp16"
            else nullcontext()
        )
        with scope:
            for offset in range(0, len(chunks), batch):
                group = chunks[offset : offset + batch]
                grouped = torch.cat(group, dim=0)
                if channels_last:
                    grouped = grouped.contiguous(memory_format=torch.channels_last_3d)
                moments = vae._encode_clip(grouped)
                if moments.shape[0] != len(group):
                    raise ValueError("face encoder lost its temporal batch axis")
                # Keep posterior variance/exponent and CPU-seeded sampling FP32.
                results.extend(item.float() for item in moments.split(1, dim=0))
        result = torch.cat(results, dim=2)[:, :, : -vae.config.token_drop]
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite face encoder moments")
        report["calls"] += 1
        report["temporal_clips"] += len(chunks)
        return result

    try:
        if "-norm" in profile:
            norms = [
                m
                for m in vae.encoder.modules()
                if type(m).__name__ == "MiniMaxH3VideoGroupNorm"
            ]
            if not norms or any(
                m.weight.dtype != torch.float32 or m.bias.dtype != torch.float32 for m in norms
            ):
                raise ValueError("expected original FP32 affine H3 frame norms")
            report["fp16_norm_calls"] = 0
            report["norm_affine_compute_precision"] = "fp16"
            report["channels_last"] = channels_last
            for norm in norms:
                norm_owners.append(
                    (norm, "forward" in norm.__dict__, norm.__dict__.get("forward"))
                )
                affine = (
                    norm.weight.to(dtype=torch.float16),
                    norm.bias.to(dtype=torch.float16),
                )

                def forward(x, owner=norm, parameters=affine):
                    result = frame_group_norm(
                        owner, x, channels_last=channels_last, affine=parameters
                    )
                    report["fp16_norm_calls"] += 1
                    return result

                norm.forward = forward
        vae._encode = encode
        yield report
    finally:
        if owned:
            vae._encode = saved
        elif "_encode" in vae.__dict__:
            del vae._encode
        for norm, had_override, override in reversed(norm_owners):
            if had_override:
                norm.forward = override
            else:
                del norm.forward
