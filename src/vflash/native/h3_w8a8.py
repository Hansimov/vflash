"""Explicit INT8 main matrices with BF16 H3 activations and unmerged adapters.

The provider quantizes each activation row dynamically. Norms, attention,
AdaLN and all six adapter branches retain the original BF16 computation.
No process-wide imports or functions are replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from traceback import clear_frames
from typing import Any

from vflash.native import h3_native_denoiser as native
from vflash.native.h3_pinned_arena import PinnedHostArena

LINEARS = ("qkv", "attention_out", "ffn_in", "ffn_out")


@dataclass(frozen=True)
class W8Weight:
    values: Any
    scales: Any
    input_features: int
    output_features: int
    bits: int = 8
    group_size: None = None


def validate_weight(weight: W8Weight) -> None:
    import torch

    if (
        not isinstance(weight, W8Weight)
        or weight.values.dtype != torch.int8
        or weight.scales.dtype != torch.float32
        or tuple(weight.values.shape) != (weight.output_features, weight.input_features)
        or tuple(weight.scales.shape) != (weight.output_features,)
        or not weight.values.is_contiguous()
        or not weight.scales.is_contiguous()
        or weight.values.device != weight.scales.device
        or weight.bits != 8
        or weight.group_size is not None
    ):
        raise ValueError("W8 requires contiguous INT8 matrices and FP32 per-output-row scales")


def move_weight(weight: W8Weight, device: Any) -> W8Weight:
    validate_weight(weight)
    return replace(weight, values=weight.values.to(device), scales=weight.scales.to(device))


def pin_weight(weight: W8Weight, *, pin: Callable) -> W8Weight:
    validate_weight(weight)
    return replace(weight, values=pin(weight.values), scales=pin(weight.scales))


def empty_weight(weight: W8Weight, device: Any) -> W8Weight:
    import torch

    validate_weight(weight)
    return replace(
        weight,
        values=torch.empty_like(weight.values, device=device),
        scales=torch.empty_like(weight.scales, device=device),
    )


def copy_block(destination: Any, source: Any) -> None:
    """Queue scale copies before the shared ring records its ready event."""
    pairs = []
    for name in LINEARS:
        target, value = getattr(destination, name), getattr(source, name)
        validate_weight(target)
        validate_weight(value)
        if target.scales.shape != value.scales.shape:
            raise ValueError("W8 ring scale layouts differ")
        pairs.append((target.scales, value.scales))
    native._copy_bf16_block_(destination, source)
    for target, value in pairs:
        target.copy_(value, non_blocking=True)


def block_bytes(weights: Any) -> int:
    return native._block_tensor_bytes(weights) + sum(
        weight.scales.numel() * weight.scales.element_size()
        for weight in (getattr(weights, name) for name in LINEARS)
    )


def kitchen_provider() -> Callable:
    """Require the tested CUDA kernel ABI; importing Vflash alone needs no kernel package."""
    import importlib.metadata

    import torch

    try:
        import comfy_kitchen.backends.cuda as kitchen
    except ImportError as exc:
        raise RuntimeError(
            "W8A8 requires the optional comfy-kitchen CUDA kernel package"
        ) from exc
    if (
        importlib.metadata.version("comfy-kitchen") != "0.2.31"
        or torch.__version__ != "2.13.0+cu130"
        or not kitchen._EXT_AVAILABLE
    ):
        raise RuntimeError("W8A8 requires comfy-kitchen 0.2.31 and Torch 2.13.0+cu130")
    return kitchen.int8_linear


class NativeW8Block(native.H3NativeBlockBF16Resident):
    """The shared H3 operations with a distinct, explicit W8 main-weight contract."""

    backend_id = "cuda-w8a8-bf16-adapter-torch-flash-block-v1"
    main_weight_bits = 8
    timing_eligible = False
    _move_main_weight = staticmethod(move_weight)

    def __init__(self, *args, linear_provider: Callable | None = None, **kwargs):
        self.linear_provider = linear_provider or kitchen_provider()
        super().__init__(*args, **kwargs)

    def _linear(self, states: Any, weight: W8Weight) -> Any:
        import torch

        if (
            not isinstance(weight, W8Weight)
            or states.dtype != torch.bfloat16
            or states.device != self.device
        ):
            raise ValueError(
                "W8A8 requires its declared W8 weights and resident BF16 activations"
            )
        return self.linear_provider(
            states, weight.values, weight.scales, out_dtype=torch.bfloat16, convrot=False
        )


class NativeW8Ring(native.H3NativeDenoiserBF16Ring):
    """Reuse the event-ordered two-slot ring, including weight-scale lifetimes."""

    backend_id = "cuda-w8a8-pinned-host-two-slot-event-ring-torch-flash-v1"
    timing_eligible = False
    main_weight_bits = 8
    block_type = NativeW8Block
    _copy_block = staticmethod(copy_block)
    _weight_bytes = staticmethod(block_bytes)

    @staticmethod
    def _empty_block(weights, device):
        return native._empty_bf16_block_like(weights, device, weight_empty=empty_weight)

    @classmethod
    def load_bundle(cls, bundle, *, device="cuda:0", **options):
        """Read only the portable model; no duplicate BF16 matrices are accessed."""
        from vflash.native.h3_w8_bundle import NativeW8Bundle

        if not isinstance(bundle, NativeW8Bundle):
            raise TypeError("the W8 ring requires an explicitly validated NativeW8Bundle")
        # Check the provider before allocating the pinned complete model.
        kitchen_provider()
        host_blocks = []
        with PinnedHostArena() as arena:
            try:
                for index in range(bundle.spec.num_layers):
                    with bundle.mapped_block(index) as weights:
                        host_blocks.append(
                            native._pin_bf16_block(
                                weights, pin=arena.copy, weight_pin=pin_weight
                            )
                        )
                        weights = None
                return cls(bundle, tuple(host_blocks), device=device, **options)
            except BaseException as exc:
                host_blocks.clear()
                clear_frames(exc.__traceback__)
                raise
