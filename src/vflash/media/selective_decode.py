# SPDX-License-Identifier: Apache-2.0
"""Decode only original H3 chunks contributing to explicitly selected frames.

The original temporal writer and overlap blending remain authoritative. Values
outside the selected frame mask are placeholders, never complete video output.
"""

from contextlib import contextmanager
from types import MethodType


def needed_chunks(model, z, head, tail, count, padding, paint):
    total, _, frames = model._decode_temporal_output_frame_plan(z, head, tail, count, padding)
    if (
        len(paint) != frames
        or not 0 < frames <= 1024
        or type(count) is not int
        or not 1 <= count <= 128
        or any(type(value) is not bool for value in paint)
    ):
        raise ValueError("paint mask must cover the exact decoded timeline")
    needed = set()
    logical = 0
    overlap = 0

    def intersects(start, size):
        return any(paint[max(0, start) : min(frames, start + size)])

    if head is not None:
        if intersects(logical, 1):
            needed.add(0)
        logical += 1
    chunk_frames = model.tokens_chunk_size * model.vae_ratio_t
    for i in range(count):
        tokens = max(
            0,
            min((i + 1) * model.tokens_chunk_size + model.token_overlap, z.shape[2])
            - min(i * model.tokens_chunk_size, z.shape[2]),
        )
        # Added isolated endpoint tokens are removed before the ordinary split.
        length = tokens * model.vae_ratio_t
        direct = max(0, min(chunk_frames, length) - model.frame_pre_padding)
        if intersects(logical, direct):
            needed.add(i)
        shared = min(overlap, direct, model.frame_overlap)
        if i and intersects(logical, shared):
            needed.add(i - 1)
        logical += direct
        overlap = (
            max(0, min(chunk_frames, max(0, length - chunk_frames)) - model.frame_pre_padding)
            if model.token_drop > 0
            else 0
        )
    if intersects(logical, overlap):
        needed.add(count - 1)
    logical += overlap
    if tail is not None:
        if intersects(logical, 1):
            needed.add(count - 1)
        logical += 1
    if logical != total:
        raise ValueError("selective decode dependency plan differs from official assembly")
    return needed


@contextmanager
def selective_stream(model, paint, report, signatures, *, verify=False):
    import torch

    stream = model._decode_temporal_streaming
    stream_owned = "_decode_temporal_streaming" in model.__dict__
    stream_saved = model.__dict__.get("_decode_temporal_streaming")

    def run(owner, z, head, tail, count, padding, dtype):
        required = needed_chunks(owner, z, head, tail, count, padding, paint)
        original = owner._adaptive_decode
        owned = "_adaptive_decode" in owner.__dict__
        saved = owner.__dict__.get("_adaptive_decode")
        signature = signatures.get(owner)
        index = 0
        executed = []

        def actual(clip):
            nonlocal signature
            value = original(clip)
            expected = (
                clip.shape[0],
                3,
                clip.shape[2] * owner.vae_ratio_t,
                clip.shape[3] * owner.vae_ratio,
                clip.shape[4] * owner.vae_ratio,
            )
            if tuple(value.shape) != expected:
                raise ValueError("decoder shape differs from its temporal dependency plan")
            current = (value.dtype, value.device)
            if signature is not None and current != signature:
                raise ValueError("decoder dtype or device changed during selective reuse")
            signature = current
            signatures[owner] = current
            return value

        def decode(clip):
            nonlocal index
            current = index
            index += 1
            if current in required or signature is None:
                executed.append(current)
                return actual(clip)
            return torch.zeros(
                (
                    clip.shape[0],
                    3,
                    clip.shape[2] * owner.vae_ratio_t,
                    clip.shape[3] * owner.vae_ratio,
                    clip.shape[4] * owner.vae_ratio,
                ),
                dtype=signature[0],
                device=signature[1],
            )

        try:
            reference = None
            if verify:
                owner._adaptive_decode = actual
                reference = stream(z, head, tail, count, padding, dtype)
            owner._adaptive_decode = decode
            result = stream(z, head, tail, count, padding, dtype)
            if index != count:
                raise ValueError("decoder did not follow the qualified streaming plan")
            report.update(
                chunks=count,
                required_chunks=sorted(required),
                executed_chunks=executed,
                skipped_chunks=count - len(executed),
                painted_frames=sum(paint),
                shadow_comparison=verify,
                shadow_chunks=count if verify else 0,
                total_executed_chunks=len(executed) + (count if verify else 0),
            )
            if reference is not None:
                mask = torch.tensor(paint, dtype=torch.bool, device=result.device)
                same = torch.equal(reference[:, :, mask], result[:, :, mask])
                report["painted_frames_bitwise_equal"] = same
                if not same:
                    raise ValueError("selective decoder changed pixels used by the compositor")
            return result
        finally:
            if owned:
                owner._adaptive_decode = saved
            else:
                del owner._adaptive_decode

    model._decode_temporal_streaming = MethodType(run, model)
    try:
        yield
    finally:
        if stream_owned:
            model._decode_temporal_streaming = stream_saved
        else:
            del model._decode_temporal_streaming
