"""Explicit approximate Veda attention for original v0.1 on one supported GPU.

The external MIT Veda core owns the predictor and INT8 sparse kernel. The first
and last five layers use explicit Torch Flash or optional SageAttention2. Other
layers keep condition/audio/text connections dense but use approximate INT8
attention. No silent fallback.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from vflash.contracts import ContractError

DENSE_LAYERS = frozenset((*range(5), *range(45, 50)))


def validate_predictor(backend: str, predictor: Path | None) -> None:
    """Check only small metadata before allocating CUDA or loading weights."""
    if backend not in {"veda-sm89", "veda-triton"}:
        if predictor is not None:
            raise ContractError("a Veda predictor requires attention_backend=veda-triton")
        return
    if not isinstance(predictor, Path) or not predictor.is_file():
        raise ContractError("veda-triton requires a local --veda-predictor safetensors file")
    from safetensors import safe_open

    with safe_open(predictor, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected = {
        "format": "miowtion-veda-predictor-v1",
        "num_layers": "50",
        "num_heads": "56",
        "head_dim": "128",
        "keep_ratio": "0.1",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ContractError(
            "Veda predictor does not describe H3 50-layer/56-head keep-0.1 attention"
        )


def target_layout(
    tensors: dict[str, Any], profile: Any
) -> tuple[int, int, tuple[int, int, int]]:
    import torch

    from vflash.native.h3_latent_layout import h3_video_latent_frame_count
    from vflash.native.h3_sol_attention import protected_prefix_length

    prefix, length = protected_prefix_length(tensors, profile.num_condition_video_rows)
    grid = (
        h3_video_latent_frame_count(profile.frames),
        profile.height // 32,
        profile.width // 32,
    )
    positions = tensors["position_ids"].to(device="cpu")
    target = positions[prefix:]
    if positions.shape != (length, 3) or length - prefix != math.prod(grid):
        raise ValueError("Veda target grid differs from its packed request")
    axes = [torch.unique(target[:, axis]) for axis in range(3)]
    if tuple(axis.numel() for axis in axes) != grid or not torch.equal(
        target, torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    ):
        raise ValueError("Veda requires T/H/W row-major target positions")
    return prefix, length, grid


class VedaVideoAttention:
    """One owned predictor per session; geometry and counters reset per request."""

    def __init__(
        self,
        device: Any,
        predictor: Path,
        *,
        backend: str = "veda-sm89",
        dense_backend: str = "torch-flash",
    ) -> None:
        import torch
        from veda_comfy import backends
        from veda_comfy.core.bundle import load_bundle
        from veda_comfy.core.engine import VedaEngine
        from veda_comfy.core.selection import Budget

        if torch.cuda.get_device_capability(device) not in {(8, 9), (9, 0), (10, 3), (12, 0)}:
            raise ValueError("Veda requires SM89, SM90, SM103 or SM120")
        if backend not in {"veda-sm89", "veda-triton"} or (
            backend == "veda-sm89" and torch.cuda.get_device_capability(device) != (8, 9)
        ):
            raise ValueError("Veda backend differs from the device contract")
        self.backend = backend
        from vflash.native.h3_veda_dense import (
            require_dense_dependencies,
            sage_dense,
            validate_dense_backend,
        )

        validate_dense_backend(
            dense_backend,
            attention_backend=backend,
            capability=".".join(map(str, torch.cuda.get_device_capability(device))),
        )
        require_dense_dependencies(dense_backend)
        self.dense_backend = dense_backend
        self.dense_attention = sage_dense if dense_backend == "sageattention2" else None
        resolved = backends.resolve(device)
        if resolved.backend is None or resolved.backend.name != "triton-int8":
            raise RuntimeError("Veda triton-int8 self-test failed; dense fallback is disabled")
        bundle = load_bundle(str(predictor))
        if (bundle.num_layers, bundle.num_heads, bundle.head_dim) != (50, 56, 128):
            raise ValueError("Veda predictor shape does not match H3")
        self.engine = VedaEngine(
            bundle, Budget(ratio=0.1), Budget(ratio=1.0), resolved.backend, device
        )
        self.layout = self.choice = None
        self.sparse_calls = self.dense_calls = 0

    def prepare(self, tensors: dict[str, Any], profile: Any) -> None:
        from veda_comfy.core.h3_layout import LayoutSpec, SpanSpec

        prefix, length, grid = target_layout(tensors, profile)
        self.engine.reset()
        self.layout = LayoutSpec(length, SpanSpec("video", prefix, grid), ())
        self.choice = self.engine.plan_for(self.layout)
        self.sparse_calls = self.dense_calls = 0

    def __call__(self, index: int, dense: Any, q: Any, k: Any, v: Any) -> Any:
        import torch

        if type(index) is not int or not 0 <= index < 50 or self.layout is None:
            raise ValueError("Veda requires a prepared request and logical block index")
        if (
            q.ndim != 4
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape[0] != 1
            or q.shape[1] != self.layout.seq_len
            or q.shape[2:] != (56, 128)
            or any(t.dtype != torch.bfloat16 for t in (q, k, v))
        ):
            raise ValueError("Veda requires matching BF16 H3 BSHD tensors")
        if index in DENSE_LAYERS:
            self.dense_calls += 1
            return (self.dense_attention or dense)(q, k, v)
        value = self.engine.attention(q[0], k[0], v[0], index, self.layout, self.choice.plan)
        self.sparse_calls += 1
        return value.unsqueeze(0)

    def metadata(self) -> dict[str, Any]:
        return {
            "attention_backend": getattr(self, "backend", "veda-sm89"),
            "exact": False,
            "effective_backends": ["triton-int8", self.dense_backend],
            "dense_backend": self.dense_backend,
            "dense_precision": "INT8 QK / FP8 PV"
            if self.dense_backend == "sageattention2"
            else "BF16",
            "dense_layers": sorted(DENSE_LAYERS),
            "generated_keep_ratio": 0.1,
            "condition_keep_ratio": 1.0,
            "protected_connections_dense": True,
            "protected_queries_exact": False,
            "fallback_enabled": False,
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "target_grid": list(self.layout.target.grid) if self.layout else None,
            "exact_trained_plan": self.choice.exact if self.choice else None,
            "kept_video_fraction": self.engine.stats.kept_fraction(),
            "sparse_layer_compute_fraction": self.engine.stats.compute_fraction(),
        }

    def validate_calls(self, nfe: int) -> None:
        if (self.sparse_calls, self.dense_calls) != (40 * nfe, 10 * nfe):
            raise RuntimeError("Veda did not execute the declared logical block coverage")
