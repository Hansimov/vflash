"""Approximate video attention with exact text, keyframe and audio rows.

Uses NVIDIA Sol-Attn's public API; no upstream implementation is vendored.
The conservative SM89 profile fixes tau=0 and never silently falls back.
"""

from __future__ import annotations

from typing import Any


def protected_prefix_length(
    tensors: dict[str, Any], condition_video_rows: int
) -> tuple[int, int]:
    """Validate the packed request once, before entering the denoising loop."""
    import torch

    video = tensors["video_indices"].detach().to(device="cpu", dtype=torch.int64)
    text = tensors["text_indices"].detach().to(device="cpu", dtype=torch.int64)
    audio = tensors["audio_indices"].detach().to(device="cpu", dtype=torch.int64)
    if not 0 <= condition_video_rows < video.numel():
        raise ValueError("Sol requires a nonempty target-video suffix")
    target = video[condition_video_rows:]
    protected = torch.cat((text, video[:condition_video_rows], audio)).sort().values
    prefix = protected.numel()
    length = prefix + target.numel()
    if not torch.equal(protected, torch.arange(prefix)) or not torch.equal(
        target, torch.arange(prefix, length)
    ):
        raise ValueError("Sol requires an exact protected prefix and contiguous video suffix")
    return prefix, length


class SolVideoAttention:
    """A request-scoped SM89 operator; shared by the serial ring's block slots."""

    def __init__(self, device: Any, prefix: int, sequence_length: int) -> None:
        import torch
        from sol_attn import get_sol_attn_backend, sol_attn

        if torch.cuda.get_device_capability(device) != (8, 9):
            raise ValueError("the approximate Sol profile requires SM89")
        if get_sol_attn_backend(device) != "cute_sm89":
            raise ValueError("Sol must select cute_sm89; fallback is not enabled")
        self._operator = sol_attn
        self.prefix = prefix
        self.sequence_length = sequence_length
        self.calls = 0

    def __call__(self, query: Any, key: Any, value: Any) -> Any:
        import torch
        import torch.nn.functional as functional
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if (
            query.ndim != 4
            or query.shape != key.shape
            or query.shape != value.shape
            or query.shape[1] != self.sequence_length
            or query.shape[-1] != 128
            or any(t.dtype != torch.bfloat16 for t in (query, key, value))
        ):
            raise ValueError("Sol attention differs from its BF16 BTHD request layout")
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        output = self._operator(
            q, k, v, tau=0.0, thresh_type="diag", sink_start=0, sink_tokens=self.prefix
        )
        if self.prefix:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION], set_priority=True):
                exact = functional.scaled_dot_product_attention(
                    q[:, : self.prefix].transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    dropout_p=0.0,
                    is_causal=False,
                ).transpose(1, 2)
            output[:, : self.prefix].copy_(exact)
        self.calls += 1
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "attention_backend": "sol-sm89",
            "exact": False,
            "effective_backends": ["cute_sm89", "torch-flash"],
            "tau": 0.0,
            "threshold_type": "diag",
            "protected_prefix_tokens": self.prefix,
            "sequence_tokens": self.sequence_length,
            "protected_queries_exact": True,
            "protected_keys_exact": True,
            "operator_calls": self.calls,
            "fallback_enabled": False,
        }
