"""Optional signed FP32 attention LoRA for the single-SM89 H3 Base16 trunk.

Explicit low-level context; no default, automatic download, training, prompt
handling, TokenRefiner update, or change to the compiled BF16 base artifact.
Use exclusive ownership of the runtime/process for the lifetime of the context.
"""

import math
from contextlib import ExitStack, contextmanager


def _validate_scale(scale):
    if type(scale) not in (int, float) or not math.isfinite(scale) or not -1 <= scale <= 1:
        raise ValueError("finite explicit adapter scale in [-1, 1] required")


def prepare_dit_attention_lora(state, rank):
    """Validate the full FP32 PEFT layout, then select only the 50 DiT blocks.

    The caller supplies the effective alpha/rank multiplier in ``scale``.
    TokenRefiner tensors are validated but intentionally not applied.
    """
    import torch

    if type(rank) is not int or not 1 <= rank <= 256:
        raise ValueError("integer LoRA rank in [1, 256] required")
    shapes = {
        ("qkv_proj", "A"): (rank, 5376),
        ("qkv_proj", "B"): (21504, rank),
        ("out_proj", "A"): (rank, 7168),
        ("out_proj", "B"): (5376, rank),
    }
    modules = [f"blocks.{i}" for i in range(50)] + [
        f"token_refiner.blocks.{i}" for i in range(2)
    ]
    expected = {
        f"pipe.dit.{module}.attn.{projection}.lora_{side}.default.weight": shape
        for module in modules
        for (projection, side), shape in shapes.items()
    }
    if set(state) != set(expected):
        raise ValueError("all 208 PEFT attention tensors, including TokenRefiner, required")
    for name, shape in expected.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
        ):
            raise ValueError("FP32 CPU tensor with the complete declared shape required")
        if not value.isfinite().all():
            raise ValueError("nonfinite LoRA tensor")
    rows = []
    for index in range(50):
        prefix = f"pipe.dit.blocks.{index}.attn."
        up = state[prefix + "qkv_proj.lora_B.default.weight"].reshape(56, 3, 128, rank)
        rows.append(
            dict(
                qkv=(
                    state[prefix + "qkv_proj.lora_A.default.weight"],
                    torch.cat(tuple(up[:, i].reshape(7168, rank) for i in range(3))),
                ),
                attention_out=(
                    state[prefix + "out_proj.lora_A.default.weight"],
                    state[prefix + "out_proj.lora_B.default.weight"],
                ),
            )
        )
    return rows


def _fp32_residual(base, states, down, up, scale, *, chunk_rows=4096):
    import torch
    import torch.nn.functional as functional

    if scale == 0:
        return base
    # Never materialize the full token-by-21504 FP32 QKV update beside a
    # resident 36 GiB backbone. Row chunks preserve the independent linear
    # operation, not a quantized or BF16-merged approximation to the adapter.
    x = states.reshape(-1, states.shape[-1])
    b = base.reshape(-1, base.shape[-1])
    result = torch.empty_like(b)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # Scope precision to these two GEMMs only. Native input/final stages
        # explicitly require their existing "high" kernel-plan policy.
        torch.backends.cuda.matmul.allow_tf32 = False
        for start in range(0, len(x), chunk_rows):
            end = start + chunk_rows
            update = functional.linear(functional.linear(x[start:end].float(), down), up)
            result[start:end] = (b[start:end].float() + update * scale).to(base.dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return result.reshape_as(base)


@contextmanager
def _block_residual(block, row, scale, calls):
    """Wrap only this block's two attention projections, restoring on exit."""
    import torch

    _validate_scale(scale)
    native = block._linear
    had_override = "_linear" in vars(block)
    saved_override = vars(block).get("_linear")
    weights = block.weights
    if any(
        (
            weights.qkv_residuals,
            weights.attention_out_residual is not None,
            weights.ffn_in_residual is not None,
            weights.ffn_out_residual is not None,
        )
    ):
        raise ValueError("adapter-free native block required")
    device = weights.qkv.values.device
    rows = {
        name: tuple(t.to(device=device, dtype=torch.float32) for t in pair)
        for name, pair in row.items()
    }
    if set(rows) != {"qkv", "attention_out"}:
        raise ValueError("exactly the two attention projections required")

    def linear(states, weight):
        base = native(states, weight)
        name = (
            "qkv"
            if weight is weights.qkv
            else "attention_out"
            if weight is weights.attention_out
            else None
        )
        if name is None:
            return base
        calls[name] += 1
        return _fp32_residual(base, states, *rows[name], scale)

    block._linear = linear
    try:
        yield
    finally:
        if had_override:
            block._linear = saved_override
        else:
            del block._linear


@contextmanager
def _method_override(owner, name, replacement):
    had_override = name in vars(owner)
    original = vars(owner).get(name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if had_override:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


@contextmanager
def _ring_residuals(denoiser, rows, scale, calls):
    """Bind logical block rows to reused ring slots on the compute thread.

    LoRA rows remain resident and immutable across the complete invocation.
    The original ring still owns all base-weight copies, CUDA events and slot
    reuse. Never select rows in the asynchronous copy callback: that may run
    ahead of compute and would apply the next block's adapter to the current one.
    """
    from contextlib import ExitStack

    if len(denoiser.host_blocks) != len(rows) or len(denoiser.slots) != 2:
        raise ValueError("complete two-slot ring required")
    gpu_rows = [
        {name: tuple(t.to(device=denoiser.device) for t in pair) for name, pair in row.items()}
        for row in rows
    ]
    native = denoiser.forward_prevalidated

    def forward(hidden_states, invocation, **kwargs):
        next_indices = [0, 1]

        def wrap(slot_index):
            slot = denoiser.slots[slot_index]
            original = slot.forward_prevalidated

            def run(states, current_invocation, **slot_kwargs):
                index = next_indices[slot_index]
                if current_invocation is not invocation or index >= len(gpu_rows):
                    raise ValueError("ring logical block traversal differs")
                next_indices[slot_index] += 2
                with _block_residual(slot, gpu_rows[index], scale, calls):
                    return original(states, current_invocation, **slot_kwargs)

            return run

        with ExitStack() as stack:
            for index, slot in enumerate(denoiser.slots):
                stack.enter_context(_method_override(slot, "forward_prevalidated", wrap(index)))
            output = native(hidden_states, invocation, **kwargs)
            if next_indices != [len(rows), len(rows) + 1]:
                raise ValueError("ring did not consume the complete adapter stack")
            return output

    with _method_override(denoiser, "forward_prevalidated", forward):
        yield


@contextmanager
def apply_dit_attention_lora(runtime, state, *, rank, scale):
    """Apply a reversible DiT-only residual to an exclusively owned Base16 runtime.

    ``state`` is a CPU FP32 PEFT attention state (not a merged backbone).
    ``scale`` is explicit, including any alpha/rank factor; a checkpoint whose
    B tensors are already negated must not receive another negative sign.
    Yield projection call counters. Exiting, including on failure, restores
    original methods; the base weights and conditioning remain unchanged.
    No quality guarantee is implied by successful loading.
    """
    _validate_scale(scale)
    if (
        runtime.compute_capability != (8, 9)
        or runtime.weight_residency not in {"resident", "block-ring"}
        or runtime.parallel_strategy != "single"
        or runtime.overlay.schedule.nfe != 16
        or runtime.artifact.weight_profile != "minimax-h3-base"
        or runtime.artifact.adapter_execution != "none"
    ):
        raise ValueError("single-SM89 adapter-free H3 Base16 runtime required")
    if getattr(runtime, "_fp32_dit_adapter_active", False):
        raise ValueError("nested or concurrent adapters on one runtime are not supported")
    rows = prepare_dit_attention_lora(state, rank)
    calls = dict(qkv=0, attention_out=0)
    with _method_override(runtime, "_fp32_dit_adapter_active", True):
        if runtime.weight_residency == "block-ring":
            with _ring_residuals(runtime.denoiser, rows, scale, calls):
                yield calls
        else:
            if len(runtime.denoiser.blocks) != 50:
                raise ValueError("complete resident Base stack required")
            with ExitStack() as stack:
                for block, row in zip(runtime.denoiser.blocks, rows, strict=True):
                    stack.enter_context(_block_residual(block, row, scale, calls))
                yield calls
