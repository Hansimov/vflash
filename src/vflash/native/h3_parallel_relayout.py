"""Exact CUDA relayouts for the two-device H3 sequence/head path.

The collective wire layouts are unchanged.  These kernels only replace
``stack/permute/contiguous`` and ``cat/permute/contiguous`` materializations
with one destination-major copy.  They therefore preserve every BF16 value
while reducing memory traffic on both Ampere (SM86) and Ada (SM89).

The destination-major technique was informed by NVIDIA Sol-Engine's Ulysses
relayout work.  Vflash uses its own layout because the native runtime overlaps
four smaller collectives rather than issuing one variable-size collective.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cache
from typing import Any


class H3ParallelRelayoutError(ValueError):
    """A tensor does not satisfy the exact direct-relayout contract."""


def _torch() -> Any:
    import torch

    return torch


def _direct_relayout_enabled() -> bool:
    value = os.environ.get("VFLASH_H3_DIRECT_RELAYOUT", "1")
    if value not in {"0", "1"}:
        raise H3ParallelRelayoutError("VFLASH_H3_DIRECT_RELAYOUT must be '0' or '1'")
    return value == "1"


def _can_use_direct(*tensors: Any) -> bool:
    if not _direct_relayout_enabled() or not tensors:
        return False
    first = tensors[0]
    return first.is_cuda and all(
        tensor.is_cuda
        and tensor.device == first.device
        and tensor.dtype == first.dtype
        and tensor.stride(-1) == 1
        for tensor in tensors
    )


@cache
def _kernels() -> tuple[Any, Any, Any, Any]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - runtime image contract
        raise H3ParallelRelayoutError("direct H3 relayout requires Triton") from exc

    @triton.jit
    def pack_qkv_kernel(
        output_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        elements,
        batch,
        rows,
        heads_per_chunk,
        width,
        chunks: tl.constexpr,
        query_stride_batch,
        query_stride_row,
        query_stride_head,
        key_stride_batch,
        key_stride_row,
        key_stride_head,
        value_stride_batch,
        value_stride_row,
        value_stride_head,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        cursor = offsets
        column = cursor % width
        cursor //= width
        local_head = cursor % heads_per_chunk
        cursor //= heads_per_chunk
        row = cursor % rows
        cursor //= rows
        batch_index = cursor % batch
        cursor //= batch
        qkv_index = cursor % 3
        cursor //= 3
        destination = cursor % 2
        chunk = cursor // 2
        global_head = (destination * chunks + chunk) * heads_per_chunk + local_head

        query_offset = (
            batch_index * query_stride_batch
            + row * query_stride_row
            + global_head * query_stride_head
            + column
        )
        key_offset = (
            batch_index * key_stride_batch
            + row * key_stride_row
            + global_head * key_stride_head
            + column
        )
        value_offset = (
            batch_index * value_stride_batch
            + row * value_stride_row
            + global_head * value_stride_head
            + column
        )
        query_value = tl.load(query_ptr + query_offset, mask=mask & (qkv_index == 0), other=0.0)
        key_value = tl.load(key_ptr + key_offset, mask=mask & (qkv_index == 1), other=0.0)
        value_value = tl.load(value_ptr + value_offset, mask=mask & (qkv_index == 2), other=0.0)
        packed = tl.where(
            qkv_index == 0, query_value, tl.where(qkv_index == 1, key_value, value_value)
        )
        tl.store(output_ptr + offsets, packed, mask=mask)

    @triton.jit
    def pack_attention_kernel(
        output_ptr,
        attention_ptr,
        elements,
        batch,
        rows,
        local_rows,
        heads,
        width,
        stride_batch,
        stride_row,
        stride_head,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        cursor = offsets
        column = cursor % width
        cursor //= width
        head = cursor % heads
        cursor //= heads
        local_row = cursor % local_rows
        cursor //= local_rows
        batch_index = cursor % batch
        destination = cursor // batch
        source_row = destination * local_rows + local_row
        source_offset = (
            batch_index * stride_batch + source_row * stride_row + head * stride_head + column
        )
        value = tl.load(
            attention_ptr + source_offset,
            mask=mask & (source_row < rows),
            other=0.0,
        )
        tl.store(output_ptr + offsets, value, mask=mask)

    @triton.jit
    def merge_attention_kernel(
        output_ptr,
        chunk_0_ptr,
        chunk_1_ptr,
        chunk_2_ptr,
        chunk_3_ptr,
        elements,
        batch,
        rows,
        heads_per_chunk,
        width,
        stride_source,
        stride_batch,
        stride_row,
        stride_head,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        cursor = offsets
        column = cursor % width
        cursor //= width
        global_head = cursor % (8 * heads_per_chunk)
        cursor //= 8 * heads_per_chunk
        row = cursor % rows
        batch_index = cursor // rows
        destination = global_head // (4 * heads_per_chunk)
        head_in_destination = global_head % (4 * heads_per_chunk)
        chunk = head_in_destination // heads_per_chunk
        local_head = head_in_destination % heads_per_chunk
        source_offset = (
            destination * stride_source
            + batch_index * stride_batch
            + row * stride_row
            + local_head * stride_head
            + column
        )
        value_0 = tl.load(chunk_0_ptr + source_offset, mask=mask & (chunk == 0), other=0.0)
        value_1 = tl.load(chunk_1_ptr + source_offset, mask=mask & (chunk == 1), other=0.0)
        value_2 = tl.load(chunk_2_ptr + source_offset, mask=mask & (chunk == 2), other=0.0)
        value_3 = tl.load(chunk_3_ptr + source_offset, mask=mask & (chunk == 3), other=0.0)
        value = tl.where(
            chunk == 0,
            value_0,
            tl.where(chunk == 1, value_1, tl.where(chunk == 2, value_2, value_3)),
        )
        tl.store(output_ptr + offsets, value, mask=mask)

    return triton, pack_qkv_kernel, pack_attention_kernel, merge_attention_kernel


def pack_qkv_reference(query: Any, key: Any, value: Any, *, chunks: int) -> Any:
    torch = _torch()
    batch, rows, heads, width = query.shape
    return (
        torch.stack((query, key, value), dim=0)
        .reshape(3, batch, rows, 2, chunks, heads // (2 * chunks), width)
        .permute(4, 3, 0, 1, 2, 5, 6)
        .contiguous()
    )


def pack_qkv(query: Any, key: Any, value: Any, *, chunks: int) -> Any:
    """Pack QKV directly into ``(chunk, destination, qkv, batch, row, head, dim)``."""
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise H3ParallelRelayoutError("QKV tensors must have the same four-dimensional shape")
    batch, rows, heads, width = query.shape
    if chunks < 1 or heads % (2 * chunks):
        raise H3ParallelRelayoutError("QKV heads must divide two destinations and all chunks")
    if not _can_use_direct(query, key, value):
        return pack_qkv_reference(query, key, value, chunks=chunks)

    torch = _torch()
    heads_per_chunk = heads // (2 * chunks)
    output = torch.empty(
        (chunks, 2, 3, batch, rows, heads_per_chunk, width),
        dtype=query.dtype,
        device=query.device,
    )
    if output.numel() == 0:
        return output
    triton, kernel, _, _ = _kernels()
    block = 1024
    kernel[(triton.cdiv(output.numel(), block),)](
        output,
        query,
        key,
        value,
        output.numel(),
        batch,
        rows,
        heads_per_chunk,
        width,
        chunks=chunks,
        query_stride_batch=query.stride(0),
        query_stride_row=query.stride(1),
        query_stride_head=query.stride(2),
        key_stride_batch=key.stride(0),
        key_stride_row=key.stride(1),
        key_stride_head=key.stride(2),
        value_stride_batch=value.stride(0),
        value_stride_row=value.stride(1),
        value_stride_head=value.stride(2),
        BLOCK=block,
        num_warps=8,
    )
    return output


def pack_attention_reference(attention: Any, padded_rows: int) -> Any:
    import torch.nn.functional as functional

    batch, rows, heads, width = attention.shape
    if rows != padded_rows:
        attention = functional.pad(attention, (0, 0, 0, 0, 0, padded_rows - rows))
    return (
        attention.reshape(batch, 2, padded_rows // 2, heads, width)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def pack_attention(attention: Any, padded_rows: int) -> Any:
    """Split full-sequence attention output into two destination-major token shards."""
    if attention.ndim != 4 or padded_rows < attention.shape[1] or padded_rows % 2:
        raise H3ParallelRelayoutError("attention padding must be even and cover all rows")
    if not _can_use_direct(attention):
        return pack_attention_reference(attention, padded_rows)

    torch = _torch()
    batch, rows, heads, width = attention.shape
    local_rows = padded_rows // 2
    output = torch.empty(
        (2, batch, local_rows, heads, width),
        dtype=attention.dtype,
        device=attention.device,
    )
    if output.numel() == 0:
        return output
    triton, _, kernel, _ = _kernels()
    block = 1024
    kernel[(triton.cdiv(output.numel(), block),)](
        output,
        attention,
        output.numel(),
        batch,
        rows,
        local_rows,
        heads,
        width,
        attention.stride(0),
        attention.stride(1),
        attention.stride(2),
        BLOCK=block,
        num_warps=8,
    )
    return output


def merge_attention_reference(received: Sequence[Any]) -> Any:
    torch = _torch()
    merged = torch.cat(tuple(received), dim=3)
    _, batch, rows, heads, width = merged.shape
    return merged.permute(1, 2, 0, 3, 4).reshape(batch, rows, heads * 2, width).contiguous()


def merge_attention(received: Sequence[Any]) -> Any:
    """Merge four returned attention chunks directly into global head order."""
    if not received:
        raise H3ParallelRelayoutError("at least one attention chunk is required")
    shape = received[0].shape
    if len(shape) != 5 or any(tensor.shape != shape for tensor in received):
        raise H3ParallelRelayoutError(
            "returned attention chunks must share a five-dimensional shape"
        )
    if len(received) != 4 or not _can_use_direct(*received):
        return merge_attention_reference(received)
    first = received[0]
    if any(tensor.stride() != first.stride() for tensor in received[1:]):
        return merge_attention_reference(received)

    torch = _torch()
    _, batch, rows, heads_per_chunk, width = shape
    output = torch.empty(
        (batch, rows, 8 * heads_per_chunk, width), dtype=first.dtype, device=first.device
    )
    if output.numel() == 0:
        return output
    triton, _, _, kernel = _kernels()
    block = 1024
    kernel[(triton.cdiv(output.numel(), block),)](
        output,
        *received,
        output.numel(),
        batch,
        rows,
        heads_per_chunk,
        width,
        first.stride(0),
        first.stride(1),
        first.stride(2),
        first.stride(3),
        BLOCK=block,
        num_warps=8,
    )
    return output
