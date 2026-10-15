"""Tile rowwise H3 work while retaining complete attention context.

This changes activation storage, not request admission or the attention policy.
The caller owns the block, invocation, optional measurement scopes and capacity.
"""

from dataclasses import replace


def tiled_forward(block, hidden, invocation, *, rows, measure, whole_rows=0):
    import torch

    from vflash.native.h3_native_denoiser import H3NativeDenoiserError, _rms_norm

    if (
        type(rows) is not int
        or not 1024 <= rows <= 65536
        or rows % 16
        or type(whole_rows) is not int
        or whole_rows not in (0, 32768)
        or invocation.artifact_id != block.artifact.artifact_id
        or tuple(hidden.shape)
        != (invocation.batch_size, invocation.sequence_length, invocation.hidden_size)
        or hidden.device != invocation.device
        or hidden.dtype != invocation.dtype
    ):
        raise H3NativeDenoiserError("tiled FP8 block invocation differs from its contract")
    if whole_rows and hidden.shape[1] <= whole_rows:
        # Explicit bounded whole-block execution. The core already owns a
        # phase-bounded whole block; avoid the tiled path's packed QKV copies
        # and output copy-back when the complete row count fits this bound.
        # Call its implementation directly to avoid the FP8 override recursing.
        from vflash.native.h3_native_denoiser import _H3BlockOperations

        with measure("block.whole"):
            return _H3BlockOperations.forward_prevalidated(block, hidden, invocation)
    table = block.weights.adaln_table[invocation.evaluation_index].to(
        device=hidden.device, dtype=hidden.dtype
    )
    spec = block.artifact.spec
    length = hidden.shape[1]
    inner = spec.num_attention_heads * spec.attention_head_dim
    # Separate contiguous Q/K/V planes avoid forcing an attention backend to
    # copy strided whole-sequence views after the rowwise work has been tiled.
    packed = torch.empty(
        (3, *hidden.shape[:-1], inner), device=hidden.device, dtype=hidden.dtype
    )
    with measure("block.qkv_norm_projection"):
        for start in range(0, length, rows):
            stop = min(length, start + rows)
            indices = invocation.adaln_indices[start:stop]
            shift, scale = (table[:, i, :].index_select(0, indices) for i in (0, 1))
            normalized = _rms_norm(
                hidden[:, start:stop], block.weights.attention_norm, eps=block.norm_eps
            )
            normalized = block._modulate(normalized, scale, shift)
            projected = block._qkv_linear(normalized)
            for plane, values in zip(packed, projected.chunk(3, dim=-1), strict=True):
                plane[:, start:stop].copy_(values)
            del projected, plane, values
            del normalized, shift, scale
    query, key, value = packed.unbind(0)
    query, key, value = (
        x.unflatten(-1, (spec.num_attention_heads, spec.attention_head_dim))
        for x in (query, key, value)
    )
    with measure("block.qk_norm_rotary"):
        for start in range(0, length, rows):
            stop = min(length, start + rows)
            tile = replace(
                invocation,
                sequence_length=stop - start,
                adaln_indices=invocation.adaln_indices[start:stop],
                rotary_cos=invocation.rotary_cos[start:stop],
                rotary_sin=invocation.rotary_sin[start:stop],
            )
            q, k = block._qk_norm_rotary(query[:, start:stop], key[:, start:stop], tile)
            # These are new packed QKV storage, never the caller's hidden state.
            # Keep full attention context without full-sequence FP32 norm/RoPE
            # temporaries or duplicate query/key tensors at 4MP/15 seconds.
            query[:, start:stop].copy_(q)
            key[:, start:stop].copy_(k)
            del q, k, tile
    with measure("block.attention"):
        attention = block._attention(query, key, value).flatten(2, 3)
    del packed, query, key, value
    attended = torch.empty_like(hidden)
    with measure("block.attention_projection_residual"):
        for start in range(0, length, rows):
            stop = min(length, start + rows)
            gate = table[:, 2, :].index_select(0, invocation.adaln_indices[start:stop])
            attended[:, start:stop].copy_(
                block._adapted_gate_residual(
                    attention[:, start:stop],
                    block.weights.attention_out,
                    block.weights.attention_out_residual,
                    hidden[:, start:stop],
                    gate,
                )
            )
    del attention, gate, hidden
    # Each FFN tile depends only on its own attended rows. Reuse this block's
    # private buffer after reading the residual; never overwrite the caller's
    # hidden state or retain a second whole-sequence output allocation.
    with measure("block.ffn_norm_projection_residual"):
        for start in range(0, length, rows):
            stop = min(length, start + rows)
            indices = invocation.adaln_indices[start:stop]
            shift, scale, gate = (table[:, i, :].index_select(0, indices) for i in (3, 4, 5))
            normalized = _rms_norm(
                attended[:, start:stop], block.weights.ffn_norm, eps=block.norm_eps
            )
            normalized = block._modulate(normalized, scale, shift)
            ffn = block._ffn_input(normalized)
            del normalized, shift, scale
            attended[:, start:stop].copy_(
                block._adapted_gate_residual(
                    ffn,
                    block.weights.ffn_out,
                    block.weights.ffn_out_residual,
                    attended[:, start:stop],
                    gate,
                )
            )
            del ffn, gate
    return attended
