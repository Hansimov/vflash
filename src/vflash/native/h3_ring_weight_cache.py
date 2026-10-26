"""Scoped BF16 weight residency for repeated passes over a two-slot ring."""

import time
from contextlib import contextmanager


@contextmanager
def cached_ring_weights(engine, budget_bytes):
    """Keep original slot addresses and arithmetic, replacing selected H2D reads.

    The serial engine owner retains its CPU originals. Cached device weights live
    only for this pass, and original copy bindings are restored on every exit.
    Twenty GiB remains available for context construction and overlapping decoding.
    """
    if not budget_bytes:
        yield None
        return
    import torch

    from .h3_native_denoiser import (
        _bf16_block_copy_pairs,
        _block_tensor_bytes,
        _copy_bf16_tensor_pairs_,
        _empty_bf16_block_like,
    )

    ring = engine.denoiser
    if not hasattr(ring, "host_blocks"):
        yield {"applied": False, "reason": "resident-trunk"}
        return
    if len(ring.host_blocks) != 50 or len(ring.slots) != 2:
        raise ValueError("weight reuse requires a complete two-slot BF16 ring")
    device = engine.latent_state.device
    original = {
        name: getattr(ring, name)
        for name in ("host_blocks", "_ring_copy_pairs", "_serial_copy_pairs")
    }
    sizes = tuple(_block_tensor_bytes(block) for block in ring.host_blocks)
    record = {
        "applied": False,
        "requested_bytes": budget_bytes,
        "trunk_bytes": sum(sizes),
        "cached_bytes": 0,
        "cached_blocks": 0,
        "reserve_bytes": 20 * 1024**3,
    }
    torch.cuda.synchronize(device)
    started = time.monotonic()
    try:
        free, _ = torch.cuda.mem_get_info(device)
        available = (
            free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        )
        limit = min(budget_bytes, max(0, available - record["reserve_bytes"]))
        blocks = list(original["host_blocks"])
        for index, (block, size) in enumerate(zip(blocks, sizes, strict=True)):
            if record["cached_bytes"] + size > limit:
                break
            resident = _empty_bf16_block_like(block, device)
            _copy_bf16_tensor_pairs_(_bf16_block_copy_pairs(resident, block))
            blocks[index] = resident
            record["cached_bytes"] += size
            record["cached_blocks"] += 1
        torch.cuda.synchronize(device)
        record["prepare_seconds"] = time.monotonic() - started
        record["applied"] = bool(record["cached_blocks"])
        ring.host_blocks = tuple(blocks)
        ring._ring_copy_pairs = tuple(
            _bf16_block_copy_pairs(ring.slots[i % 2].weights, block)
            for i, block in enumerate(blocks)
        )
        ring._serial_copy_pairs = tuple(
            _bf16_block_copy_pairs(ring.slots[0].weights, block) for block in blocks
        )
        yield record
    finally:
        torch.cuda.synchronize(device)
        for name, value in original.items():
            setattr(ring, name, value)
