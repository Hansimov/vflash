"""Full-context row tiling against the real H3 block on synthetic CPU tensors."""

from contextlib import nullcontext
from types import SimpleNamespace as N

import pytest

from vflash.native.h3_native_denoiser import H3NativeBlockInvocation, _H3BlockOperations
from vflash.native.h3_tiled_block import tiled_forward

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("whole_rows", [0, 32768])
def test_tiled_projections_keep_full_attention_and_original_input(monkeypatch, whole_rows):
    torch.set_num_threads(2)
    torch.manual_seed(981)

    class Block(_H3BlockOperations):
        def _linear(self, states, weight):
            self.rows.append(states.shape[1])
            return torch.nn.functional.linear(states, weight)

        def _attention(self, q, k, v):
            self.contexts.append(q.shape[1])
            self.contiguous.append(all(x.is_contiguous() for x in (q, k, v)))
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)

        def _qk_norm_rotary(self, q, k, invocation):
            self.norm_rows.append(q.shape[1])
            return super()._qk_norm_rotary(q, k, invocation)

    weights = N(
        adaln_table=torch.randn(1, 3, 6, 8) * 0.1,
        qkv=torch.randn(24, 8) * 0.1,
        qkv_residuals=(),
        attention_out=torch.randn(8, 8) * 0.1,
        attention_out_residual=None,
        ffn_in=torch.randn(32, 8) * 0.1,
        ffn_in_residual=None,
        ffn_out=torch.randn(8, 16) * 0.1,
        ffn_out_residual=None,
        attention_norm=torch.ones(8),
        ffn_norm=torch.ones(8),
        query_norm=torch.ones(4),
        key_norm=torch.ones(4),
    )
    block = Block(
        N(artifact_id="test", spec=N(num_attention_heads=2, attention_head_dim=4)), weights
    )
    block.rows = []
    block.contexts = []
    block.norm_rows = []
    block.contiguous = []
    states = torch.randn(1, 1041, 8)
    untouched = states.clone()
    angles = torch.linspace(0, 1, 1041)[:, None] * torch.tensor([1.0, 2.0, 1.0, 2.0])
    invocation = H3NativeBlockInvocation(
        artifact_id="test",
        batch_size=1,
        sequence_length=1041,
        hidden_size=8,
        device=states.device,
        dtype=states.dtype,
        evaluation_index=0,
        adaln_indices=torch.arange(1041) % 3,
        rotary_cos=angles.cos(),
        rotary_sin=angles.sin(),
    )
    reference = block.forward_prevalidated(states, invocation)
    block.rows.clear()
    block.contexts.clear()
    block.norm_rows.clear()
    block.contiguous.clear()
    buffers = []
    empty_like = torch.empty_like

    def record_buffer(value, *args, **kwargs):
        buffer = empty_like(value, *args, **kwargs)
        if buffer.shape == states.shape:
            buffers.append(buffer)
        return buffer

    monkeypatch.setattr(torch, "empty_like", record_buffer)
    actual = tiled_forward(
        block,
        states,
        invocation,
        rows=1024,
        measure=lambda _: nullcontext(),
        whole_rows=whole_rows,
    )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(states, untouched, rtol=0, atol=0)
    assert max(block.rows) == (1041 if whole_rows else 1024)
    if not whole_rows:
        assert 17 in block.rows
    assert block.contexts == [1041]
    assert block.norm_rows == ([1041] if whole_rows else [1024, 17])
    if not whole_rows:
        assert block.contiguous == [True]
        assert len(buffers) == 1 and actual.data_ptr() == buffers[0].data_ptr()
    assert actual.data_ptr() != states.data_ptr()
    with pytest.raises(ValueError):
        tiled_forward(
            block,
            states,
            invocation,
            rows=1024,
            measure=lambda _: nullcontext(),
            whole_rows=65536,
        )
