"""Exact reuse of repeated padded clips under the same batch-two encoder."""


def unique_clips(chunks, *, equal):
    """Stable exact equality, without content hashing or approximate matching."""
    unique, indices = [], []
    for chunk in chunks:
        match = next((i for i, prior in enumerate(unique) if equal(chunk, prior)), None)
        if match is None:
            match = len(unique)
            unique.append(chunk)
        indices.append(match)
    return unique, indices


def encode_unique(vae, x, report):
    import torch

    if x.ndim != 5 or x.shape[0] != 1 or x.shape[2] != 124:
        raise ValueError("exact clip reuse currently qualifies five-second face inputs only")
    length = vae.config.clip_length
    if length != 17 or vae.config.token_drop != 3:
        raise ValueError("original H3 temporal contract required")
    x = torch.cat((x, x[:, :, -1:].expand(-1, -1, (-x.shape[2]) % length, -1, -1)), dim=2)
    chunks = x.split(length, dim=2)
    unique, mapping = unique_clips(chunks, equal=torch.equal)
    moments = []
    with torch.autocast("cuda", dtype=torch.float16, enabled=x.device.type == "cuda"):
        for start in range(0, len(unique), 2):
            group = unique[start : start + 2]
            # Keep every convolution's original batch-two shape, even when
            # deduplication leaves one unique clip. The extra output is ignored.
            padded = group if len(group) == 2 else [group[0], group[0]]
            value = vae._encode_clip(torch.cat(padded, dim=0))
            if value.shape[0] != 2:
                raise ValueError("encoder changed the batch axis")
            moments.extend(part.float() for part in value[: len(group)].split(1, dim=0))
    result = torch.cat([moments[i] for i in mapping], dim=2)[:, :, : -vae.config.token_drop]
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite reused face moments")
    report.update(
        input_clips=len(chunks),
        unique_clips=len(unique),
        original_batches=(len(chunks) + 1) // 2,
        executed_batches=(len(unique) + 1) // 2,
        batch_size=2,
        exact_inputs=True,
    )
    return result
