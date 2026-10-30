"""Owned execution optimizations for the unmerged, row-FP8 H3 reference adapter.

This uses the pinned vflash kernels. It does not select weights, attention,
samplers or product defaults. Qualification remains specific to the actual
reference implementation and hardware; complete-output parity is checked by
the caller before adoption.
"""

import sys
import types
from contextlib import contextmanager


@contextmanager
def dense_reference_execution(
    model, *, qkv=True, elementwise=True, maximum_weight_bytes=8 * 1024**3
):
    """Temporarily fuse an exclusively owned reference model's inference work.

    QKV intermediates live for one attention call. Native fixed-picture audio
    keeps separate projections because its live Q and cached K/V have different
    row layouts. All modifications, including partially failed setup, unwind.
    """
    import torch

    from vflash.native.h3_fused_ops import (
        triton_strict_bf16_gate_residual,
        triton_strict_bf16_modulate,
        triton_strict_bf16_silu_mul,
    )

    if type(qkv) is not bool or type(elementwise) is not bool:
        raise ValueError("Reference execution options must be explicit booleans")
    if type(maximum_weight_bytes) is not int or maximum_weight_bytes < 1:
        raise ValueError("Reference execution needs a positive weight memory bound")
    module = sys.modules[type(model).__module__]
    if model.training or getattr(module, "_vflash_reference_execution_owner", None) is not None:
        raise ValueError("Reference execution requires an exclusively owned evaluation model")
    old_gather = module._index_select_rows
    saved, handles = [], []
    record = {
        "direct_bf16_gathers": 0,
        "fused_qkv": 0,
        "unfused_audio_projections": 0,
        "strict_blocks": 0,
        "strict_swiglu": 0,
        "additional_weight_bytes": 0,
    }

    def replace(obj, name, value):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)

    def gather(table, indices):
        if not torch.is_grad_enabled() and table.dtype == torch.bfloat16:
            record["direct_bf16_gathers"] += 1
            return table.index_select(0, indices)
        # Preserve the source implementation's training/backward behavior.
        return old_gather(table, indices)

    def block_forward(
        block,
        hidden_states,
        temb,
        adaln_indices,
        rotary_emb,
        attention_mask=None,
        valid_length=None,
    ):
        if torch.is_grad_enabled() or model.training:
            raise ValueError("Fused reference blocks require inference without gradients")
        shift, scale, gate, ff_shift, ff_scale, ff_gate = block.adaln_proj(temb)
        normalized = triton_strict_bf16_modulate(
            block.norm1(hidden_states),
            gather(scale, adaln_indices),
            gather(shift, adaln_indices),
        )
        projected = block.attn(normalized, rotary_emb, attention_mask, valid_length)
        hidden_states = triton_strict_bf16_gate_residual(
            hidden_states, gather(gate, adaln_indices), projected
        )
        normalized = triton_strict_bf16_modulate(
            block.norm2(hidden_states),
            gather(ff_scale, adaln_indices),
            gather(ff_shift, adaln_indices),
        )
        projected = block.ff(normalized)
        record["strict_blocks"] += 1
        return triton_strict_bf16_gate_residual(
            hidden_states, gather(ff_gate, adaln_indices), projected
        )

    def swiglu(activation, hidden_states):
        record["strict_swiglu"] += 1
        return triton_strict_bf16_silu_mul(activation.proj(hidden_states))

    try:
        replace(module, "_vflash_reference_execution_owner", model)
        replace(module, "_index_select_rows", gather)
        if elementwise:
            for block in model.transformer_blocks:
                replace(block, "forward", types.MethodType(block_forward, block))
                activation = block.ff.net[0]
                replace(activation, "forward", types.MethodType(swiglu, activation))
        if qkv:
            for block in model.transformer_blocks:
                attention = block.attn
                parts = [attention.to_q, attention.to_k, attention.to_v]
                if not all(
                    hasattr(part, "execution")
                    and hasattr(part, "row_scale")
                    and part.bias is None
                    and part.weight.shape == parts[0].weight.shape
                    and part.execution is parts[0].execution
                    for part in parts
                ):
                    raise ValueError("QKV fusion requires matching cached row-FP8 projections")
                additional = sum(
                    part.weight.numel() * part.weight.element_size()
                    + part.row_scale.numel() * part.row_scale.element_size()
                    for part in parts
                )
                if record["additional_weight_bytes"] + additional > maximum_weight_bytes:
                    raise ValueError("Fused QKV exceeds its additional weight memory bound")
                weight = torch.cat([part.weight.detach() for part in parts])
                scale = torch.cat([part.row_scale for part in parts])
                record["additional_weight_bytes"] += additional
                execution, cache = parts[0].execution, {}

                def clear(*_, cache=cache):
                    cache.clear()

                handles.append(attention.register_forward_pre_hook(clear))
                handles.append(attention.register_forward_hook(clear, always_call=True))
                for index, part in enumerate(parts):
                    original = part.forward

                    def forward(
                        values,
                        *,
                        index=index,
                        original=original,
                        weight=weight,
                        scale=scale,
                        execution=execution,
                        cache=cache,
                    ):
                        if torch.is_grad_enabled() or model.training:
                            raise ValueError(
                                "Fused projections require inference without gradients"
                            )
                        if getattr(model, "_vflash_fixed_audio_active", False):
                            record["unfused_audio_projections"] += 1
                            return original(values)
                        if not cache:
                            if index != 0:
                                raise ValueError(
                                    "Fused attention must evaluate Q before K and V"
                                )
                            cache.update(
                                input=values,
                                output=execution.linear(values, weight, scale).chunk(3, dim=-1),
                            )
                            record["fused_qkv"] += 1
                        if cache["input"] is not values:
                            raise ValueError(
                                "Fused QKV inputs changed inside one attention call"
                            )
                        return cache["output"][index]

                    replace(part, "forward", forward)
        yield record
    finally:
        for handle in reversed(handles):
            handle.remove()
        for obj, name, had, value in reversed(saved):
            if had:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
