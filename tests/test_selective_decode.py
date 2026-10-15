import weakref

import pytest
import torch

from vflash.media.selective_decode import needed_chunks, selective_stream


# CPU fixture preserves temporal spans, endpoint trimming and overlap blending.
def decode_temporal_cpu(model, z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype):
    import torch

    if model.training or type(num_chunks) is not int or not 1 <= num_chunks <= 128:
        raise ValueError("CPU frame assembly requires bounded inference chunks")
    total, padded, frames = model._decode_temporal_output_frame_plan(
        z, z_head, z_tail, num_chunks, pad_tokens
    )
    if not 0 < frames <= 1024 or total - padded != frames:
        raise ValueError("invalid temporal frame plan")
    chunk_frames = model.tokens_chunk_size * model.vae_ratio_t
    split_count = int(model.token_drop > 0) + 1
    output = overlap = None
    written = logical = dropped = 0
    copied_bytes = 0

    def write(part):
        nonlocal output, written, logical, dropped, copied_bytes
        count = int(part.shape[2])
        if count <= 0:
            return
        logical += count
        if output is None:
            shape = list(part.shape)
            shape[2] = frames
            # Pageable host storage avoids pinning an entire 4MP/15s movie.
            output = torch.empty(shape, dtype=part.dtype, device="cpu")
        used = min(count, max(0, frames - written))
        if used:
            source = part[:, :, :used]
            output[:, :, written : written + used].copy_(source, non_blocking=False)
            copied_bytes += source.numel() * source.element_size()
            written += used
        dropped += count - used

    for index in range(num_chunks):
        begin = index * model.tokens_chunk_size
        clip_z = z[:, :, begin : begin + model.tokens_chunk_size + model.token_overlap]
        if index == 0 and z_head is not None:
            clip_z = torch.cat((z_head, clip_z), dim=2)
        if index == num_chunks - 1 and z_tail is not None:
            clip_z = torch.cat((clip_z, z_tail), dim=2)
        clip = model._adaptive_decode(clip_z)
        if temporal_cat_dtype is not None and clip.dtype != temporal_cat_dtype:
            clip = clip.to(temporal_cat_dtype)
        if clip.device != z.device:
            clip = clip.to(z.device)
        tail = None
        if index == 0 and z_head is not None:
            write(clip[:, :, model.vae_ratio_t - 1 : model.vae_ratio_t])
            clip = clip[:, :, model.vae_ratio_t :]
        if index == num_chunks - 1 and z_tail is not None:
            tail = clip[:, :, -1:]
            clip = clip[:, :, : -model.vae_ratio_t]
        for split in range(split_count):
            begin = split * chunk_frames
            part = clip[:, :, begin : min(begin + chunk_frames, clip.shape[2])]
            part = part[:, :, model.frame_pre_padding :]
            if split == 0:
                if overlap is not None:
                    part = model.blend(overlap, part, model.frame_overlap, dim=-3)
                    overlap = None
                write(part)
            else:
                # Always break views so the previous complete chunk is released
                # before starting the next decode, even for contiguous slices.
                overlap = part.clone(memory_format=torch.contiguous_format)
            del part
        if index == num_chunks - 1:
            if overlap is not None:
                write(overlap)
                overlap = None
            if tail is not None:
                write(tail)
        del tail, clip, clip_z
    if output is None or (logical, dropped, written) != (total, padded, frames):
        raise ValueError("CPU temporal assembly differs from the official frame plan")
    return output


class Decoder:
    training = False
    tokens_chunk_size = 5
    vae_ratio_t = 4
    vae_ratio = 2
    token_overlap = 2
    token_drop = 3
    frame_overlap = 5
    frame_pre_padding = 3

    def __init__(self):
        self.calls = 0

    def _adaptive_decode(self, z):
        self.calls += 1
        y = z + z.mean(dim=2, keepdim=True)
        return y.repeat_interleave(4, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)

    def _decode_temporal_output_frame_plan(self, z, head, tail, count, padding):
        total = count * 17 + 5 + int(head is not None) + int(tail is not None)
        return total, padding * 4, total - padding * 4

    def _decode_temporal_streaming(self, *args):
        return decode_temporal_cpu(self, *args)

    def blend(self, a, b, extent, dim):
        extent = min(a.shape[dim], b.shape[dim], extent)
        w = torch.arange(extent, dtype=b.dtype).view(1, 1, -1, 1, 1) / extent
        return torch.cat(
            (a[:, :, -extent:] * (1 - w) + b[:, :, :extent] * w, b[:, :, extent:]), 2
        )


@pytest.mark.parametrize("region", ["first", "second", "blend_boundary", "last", "all"])
def test_skip_keeps_every_painted_frame_and_its_overlap_dependency(region):
    model = Decoder()
    z = torch.arange(3 * 37 * 4, dtype=torch.float32).reshape(1, 3, 37, 2, 2) / 1000
    masks = {
        "first": range(0, 41),
        "second": range(57, 120),
        "blend_boundary": [34, 35, 36, 38, 51],
        "last": range(119, 124),
        "all": range(124),
    }
    paint = [i in masks[region] for i in range(124)]
    original = model._decode_temporal_streaming(z, None, None, 7, 0, None)
    report = {}
    with selective_stream(model, paint, report, weakref.WeakKeyDictionary(), verify=True):
        result = model._decode_temporal_streaming(z, None, None, 7, 0, None)
    assert torch.equal(original[:, :, paint], result[:, :, paint])
    assert report["painted_frames_bitwise_equal"]
    assert report["skipped_chunks"] == 0 if region == "all" else report["skipped_chunks"] > 0
    assert (
        "_adaptive_decode" not in model.__dict__
        and "_decode_temporal_streaming" not in model.__dict__
    )
    if region == "blend_boundary":
        assert {1, 2, 3} <= set(report["required_chunks"])


def test_cold_signature_probe_is_bounded_and_warm_skip_uses_original_dtype():
    model = Decoder()
    z = torch.rand(1, 3, 37, 2, 2)
    mask = [57 <= i < 120 for i in range(124)]
    signatures = weakref.WeakKeyDictionary()
    first = {}
    second = {}
    with selective_stream(model, mask, first, signatures):
        a = model._decode_temporal_streaming(z, None, None, 7, 0, None)
    with selective_stream(model, mask, second, signatures):
        b = model._decode_temporal_streaming(z, None, None, 7, 0, None)
    assert first["executed_chunks"] == [0, 3, 4, 5, 6]
    assert second["executed_chunks"] == [3, 4, 5, 6]
    assert torch.equal(a[:, :, mask], b[:, :, mask])


@pytest.mark.parametrize(
    "head,tail,padding", [(True, False, 0), (False, True, 0), (True, True, 1)]
)
def test_endpoints_and_trimmed_padding_stay_on_the_official_timeline(head, tail, padding):
    model = Decoder()
    z = torch.rand(1, 3, 37, 2, 2)
    h = z[:, :, :1] if head else None
    t = z[:, :, -1:] if tail else None
    frames = 124 + int(head) + int(tail) - padding * 4
    mask = [i in [0, 17, 34, frames - 1] for i in range(frames)]
    with selective_stream(model, mask, {}, weakref.WeakKeyDictionary(), verify=True):
        result = model._decode_temporal_streaming(z, h, t, 7, padding, None)
    assert result.shape[2] == frames


def test_mismatched_timeline_fails_before_any_chunk_compute():
    model = Decoder()
    z = torch.zeros(1, 3, 37, 2, 2)
    with pytest.raises(ValueError, match="timeline"):
        needed_chunks(model, z, None, None, 7, 0, [True] * 123)
    assert model.calls == 0
