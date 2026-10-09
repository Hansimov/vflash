"""Block parity and intermediate ownership at the attention/FFN boundary."""

import weakref
from types import SimpleNamespace

import pytest

from vflash.native.h3_native_denoiser import H3NativeBlockBF16Resident, _rms_norm


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_block_releases_phase_temporaries_before_large_ffn_allocations(dtype_name):
    torch = pytest.importorskip("torch")
    torch.manual_seed(31)
    dtype = getattr(torch, dtype_name)
    device = torch.device("cpu")

    def random(*shape):
        return torch.randn(*shape, dtype=dtype) / 8

    x = random(1, 8, 16)
    qkv_weight, out_weight, ffn_weight, ffn_out = (
        random(16, 48),
        random(16, 16),
        random(16, 24),
        random(24, 16),
    )
    table = random(2, 3, 6, 16)
    indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    invocation = SimpleNamespace(
        artifact_id="fixture",
        batch_size=1,
        sequence_length=8,
        hidden_size=16,
        device=device,
        dtype=dtype,
        evaluation_index=1,
        adaln_indices=indices,
    )
    instance = H3NativeBlockBF16Resident.__new__(H3NativeBlockBF16Resident)
    instance.artifact = SimpleNamespace(
        artifact_id="fixture", spec=SimpleNamespace(num_attention_heads=2, attention_head_dim=8)
    )
    instance.weights = SimpleNamespace(
        adaln_table=table,
        attention_norm=torch.ones(16, dtype=dtype),
        ffn_norm=torch.ones(16, dtype=dtype),
        attention_out=out_weight,
        attention_out_residual=None,
        ffn_out=ffn_out,
        ffn_out_residual=None,
    )
    instance.norm_eps = 1e-6
    temporaries, modulations, ffn_inputs = [], [], []

    def modulate(value, scale, shift):
        modulations.append((weakref.ref(scale), weakref.ref(shift)))
        return value * (1 + scale) + shift

    def projection(value):
        temporaries.append(weakref.ref(value))
        result = value @ qkv_weight
        temporaries.append(weakref.ref(result))
        return result

    def attention(q, k, v):
        result = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        temporaries.append(weakref.ref(result))
        return result

    def residual(value, weight, adapter, original, gate):
        assert adapter is None
        if ffn_inputs and weight is ffn_out:
            assert ffn_inputs[-1]() is None
        return original + (value @ weight) * gate

    def ffn(value):
        assert all(reference() is None for reference in temporaries)
        assert all(reference() is None for reference in modulations[-1])
        ffn_inputs.append(weakref.ref(value))
        return torch.nn.functional.silu(value @ ffn_weight)

    instance._modulate = modulate
    instance._qkv_linear = projection
    instance._qk_norm_rotary = lambda q, k, invocation: (q, k)
    instance._attention = attention
    instance._adapted_gate_residual = residual
    instance._ffn_input = ffn
    # Independent unoptimized expression expands all six groups at once.
    shift, scale, gate, fshift, fscale, fgate = table[1].index_select(0, indices).unbind(1)
    normalized = modulate(_rms_norm(x, instance.weights.attention_norm, eps=1e-6), scale, shift)
    q, k, v = (normalized @ qkv_weight).reshape(1, 8, 3, 2, 8).unbind(2)
    attended = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        .transpose(1, 2)
        .flatten(2, 3)
    )
    hidden = residual(attended, out_weight, None, x, gate)
    normalized = modulate(
        _rms_norm(hidden, instance.weights.ffn_norm, eps=1e-6), fscale, fshift
    )
    expected = residual(
        torch.nn.functional.silu(normalized @ ffn_weight), ffn_out, None, hidden, fgate
    )
    with torch.no_grad():
        actual = instance.forward_prevalidated(x, invocation)
    assert torch.equal(actual, expected)
