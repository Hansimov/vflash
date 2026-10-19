"""FP8 P·V for Veda's sparse layers, paired with the SageAttention2 dense policy on SM120.

Veda's sparse kernel quantises Q and K to INT8 and multiplies P·V in FP16. This variant keeps
its walk, masking, tile selection and INT8 Q·K, and changes only P·V: V is quantised once per
(head, channel) to e4m3, P (already in [0, 1] after the max-subtracted exp2) is scaled to the
e4m3 range, the product accumulates in FP32 and V's scale is applied once at the end. This is
SageAttention2's arithmetic; an FP8 MMA issues at twice the FP16 rate on SM120. A startup
self-test compares it with Veda's own kernel; there is no silent fallback.
"""

from __future__ import annotations

from typing import Any

E4M3_MAX = 448.0
# Veda's INT8 kernel measures ~1-3% against exact attention; FP8 P·V adds about one point.
SELF_TEST_TOLERANCE = 0.05

_kernels: dict[str, Any] = {}


def _compile() -> dict[str, Any]:
    if _kernels:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def quantize_v_kernel(
        V, Vq, Scale, stride_n, stride_h, slots, D: tl.constexpr, CHUNK: tl.constexpr
    ):
        head = tl.program_id(0)
        offs_d = tl.arange(0, D)
        peak = tl.zeros([D], dtype=tl.float32)
        for start in range(0, slots, CHUNK):
            offs_n = start + tl.arange(0, CHUNK)
            x = tl.load(
                V + head * stride_h + offs_n[:, None] * stride_n + offs_d[None, :],
                mask=offs_n[:, None] < slots,
                other=0.0,
            ).to(tl.float32)
            peak = tl.maximum(peak, tl.max(tl.abs(x), 0))
        scale = tl.where(peak > 0.0, peak / 448.0, 1.0)
        tl.store(Scale + head * D + offs_d, scale)
        for start in range(0, slots, CHUNK):
            offs_n = start + tl.arange(0, CHUNK)
            mask = offs_n[:, None] < slots
            x = tl.load(
                V + head * stride_h + offs_n[:, None] * stride_n + offs_d[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                Vq + head * stride_h + offs_n[:, None] * stride_n + offs_d[None, :],
                (x / scale[None, :]).to(tl.float8e4nv),
                mask=mask,
            )

    @triton.jit
    def attention_kernel(
        Q,
        K,
        V,
        Q_scale,
        K_scale,
        V_scale,
        Index,
        Count,
        Valid,
        Out,
        stride_n,
        stride_h,
        stride_vn,
        stride_vh,
        stride_qs,
        stride_ks,
        stride_ih,
        stride_iq,
        n_q_tiles,
        D: tl.constexpr,
        BLK: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        head = tl.program_id(1)
        offs_m = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLK)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :])
        q_scale = tl.load(Q_scale + head * stride_qs + query_tile)
        m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        count = tl.load(Count + head * n_q_tiles + query_tile)
        index_row = Index + head * stride_ih + query_tile * stride_iq
        for i in range(0, count):
            block = tl.load(index_row + i)
            start = block * BLK
            k = tl.load(
                K + head * stride_h + (start + offs_n)[None, :] * stride_n + offs_d[:, None]
            )
            k_scale = tl.load(K_scale + head * stride_ks + block)
            qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
            qk = tl.where(offs_n[None, :] < tl.load(Valid + block), qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(
                V + head * stride_vh + (start + offs_n)[:, None] * stride_vn + offs_d[None, :]
            )
            acc += tl.dot((p * 448.0).to(tl.float8e4nv), v)
            m_i = m_ij
        l_i = tl.where(l_i > 0.0, l_i, 1.0)
        v_scale = tl.load(V_scale + head * D + offs_d)
        acc = acc * (v_scale[None, :] / 448.0) / l_i[:, None]
        tl.store(
            Out + head * stride_h + offs_m[:, None] * stride_n + offs_d[None, :],
            acc.to(Out.type.element_ty),
        )

    _kernels.update(quantize_v=quantize_v_kernel, attention=attention_kernel)
    return _kernels


def quantize_v(v: Any) -> tuple[Any, Any]:
    import torch

    slots, heads, dim = v.shape
    codes = torch.empty(v.shape, dtype=torch.float8_e4m3fn, device=v.device)
    scale = torch.empty((heads, dim), dtype=torch.float32, device=v.device)
    _compile()["quantize_v"][(heads,)](
        v, codes, scale, v.stride(0), v.stride(1), slots, D=dim, CHUNK=128, num_warps=4
    )
    return codes, scale


def attend(
    q: Any,
    k: Any,
    v: Any,
    index: Any,
    count: Any,
    valid_count: Any,
    softmax_scale: float | None = None,
) -> Any:
    """Veda's ``sparse_int8.attend`` contract ([N, H, D] tile-ordered) with FP8 P·V."""
    import torch
    from veda_comfy.kernels.sage import sparse_int8 as base

    slots, heads, dim = q.shape
    scale = (dim**-0.5) if softmax_scale is None else softmax_scale
    q_int8, q_scale = base.quantize(q, base.TILE, pre_scale=scale * base.LOG2E)
    k_int8, k_scale = base.quantize(k, base.KEY_BLOCK)
    blocks, block_count, valid = base.key_blocks(index, count, valid_count, base.KEY_BLOCK)
    v8, v_scale = quantize_v(v)
    out = torch.empty_like(q)
    _compile()["attention"][(slots // base.TILE, heads)](
        q_int8,
        k_int8,
        v8,
        q_scale,
        k_scale,
        v_scale,
        blocks,
        block_count,
        valid,
        out,
        q.stride(0),
        q.stride(1),
        v8.stride(0),
        v8.stride(1),
        q_scale.stride(0),
        k_scale.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        n_q_tiles=slots // base.TILE,
        D=dim,
        BLK=base.KEY_BLOCK,
        BLOCK_M=base.TILE,
        num_warps=base.WARPS,
        num_stages=base.STAGES,
    )
    return out


class FP8PVBackend:
    """Veda's resolved triton-int8 backend with only its sparse P·V product changed."""

    def __init__(self, backend: Any) -> None:
        if getattr(backend, "name", None) != "triton-int8":
            raise ValueError("FP8 P·V wraps only Veda's triton-int8 backend")
        self._backend = backend
        self.name = "triton-int8-fp8pv"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def attend(self, q: Any, k: Any, v: Any, block_mask: Any, layout: Any) -> Any:
        import torch
        from veda_comfy.core import selection

        index, count = selection.tile_index_list(block_mask & layout.kv_ok)
        with torch.no_grad():
            return attend(q, k, v, index, count, layout.valid_count)


def self_test(device: Any) -> float:
    """Relative error against Veda's own INT8/FP16 kernel on a half-sparse problem."""
    import torch
    from veda_comfy.kernels.sage import sparse_int8 as base

    generator = torch.Generator(device=device).manual_seed(0)
    tiles, heads, dim = 8, 4, 128
    q, k, v = (
        torch.randn(
            tiles * base.TILE,
            heads,
            dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )
    index = torch.arange(0, tiles, 2, device=device, dtype=torch.int32)  # every other key tile
    index = index.view(1, 1, -1).expand(heads, tiles, -1).contiguous()
    count = torch.full((heads, tiles), index.shape[-1], device=device, dtype=torch.int32)
    valid = torch.full((tiles,), base.TILE, device=device, dtype=torch.int32)
    reference = base.attend(q, k, v, index, count, valid).float()
    candidate = attend(q, k, v, index, count, valid).float()
    error = ((candidate - reference).norm() / reference.norm()).item()
    if not error < SELF_TEST_TOLERANCE:
        raise RuntimeError(f"FP8 P·V self-test error {error:.3f}; sparse FP8 is disabled")
    return error
