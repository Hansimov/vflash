"""Reference execution ownership and CPU algebra; CUDA parity is qualified separately."""

import sys
import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None

from vflash.adapters.h3_reference_execution import dense_reference_execution


def _index_select_rows(table, indices):
    return table.float().index_select(0, indices).to(table.dtype)


class Execution:
    def __init__(self):
        self.calls = 0

    def linear(self, values, weight, scale):
        self.calls += 1
        return F.linear(values, weight)


class Projection(nn.Linear if torch is not None else object):
    def __init__(self, execution):
        super().__init__(8, 8, bias=False, dtype=torch.bfloat16)
        self.execution = execution
        self.register_buffer("row_scale", torch.ones(8, 1))

    def forward(self, values):
        return self.execution.linear(values, self.weight, self.row_scale)


class Attention(nn.Module if torch is not None else object):
    def __init__(self, execution):
        super().__init__()
        self.to_q, self.to_k, self.to_v = [Projection(execution) for _ in range(3)]
        self.change_input = False

    def forward(self, values):
        query = self.to_q(values)
        key = self.to_k(values.clone() if self.change_input else values)
        return torch.cat((query, key, self.to_v(values)), dim=-1)


class Block(nn.Module if torch is not None else object):
    def __init__(self, execution):
        super().__init__()
        self.attn = Attention(execution)


class Model(nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.execution = Execution()
        self.transformer_blocks = nn.ModuleList([Block(self.execution), Block(self.execution)])


@unittest.skipIf(torch is None, "Reference adapter execution requires the pinned Torch image")
class ReferenceExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(88)
        self.model = Model().eval()
        self.attention = self.model.transformer_blocks[0].attn
        self.values = torch.randn(1, 17, 8, dtype=torch.bfloat16)
        self.original_gather = _index_select_rows

    def assert_restored(self):
        self.assertIs(_index_select_rows, self.original_gather)
        self.assertFalse(hasattr(sys.modules[__name__], "_vflash_reference_execution_owner"))
        for block in self.model.transformer_blocks:
            self.assertFalse(block.attn._forward_pre_hooks)
            self.assertFalse(block.attn._forward_hooks)
            for part in (block.attn.to_q, block.attn.to_k, block.attn.to_v):
                self.assertNotIn("forward", part.__dict__)

    def test_per_call_qkv_matches_and_does_not_reuse_previous_input(self):
        with torch.inference_mode():
            expected = [self.attention(x) for x in (self.values, self.values + 1)]
            with dense_reference_execution(self.model, elementwise=False) as record:
                actual = [self.attention(x) for x in (self.values, self.values + 1)]
                self.assertEqual(record["fused_qkv"], 2)
            for first, second in zip(expected, actual, strict=True):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assert_restored()

    def test_fixed_picture_audio_keeps_independent_projections(self):
        with torch.inference_mode():
            expected = self.attention(self.values)
            self.model._vflash_fixed_audio_active = True
            with dense_reference_execution(self.model, elementwise=False) as record:
                actual = self.attention(self.values)
                self.assertEqual(record["fused_qkv"], 0)
                self.assertEqual(record["unfused_audio_projections"], 3)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_restored()

    def test_partial_setup_failure_restores_earlier_blocks(self):
        self.model.transformer_blocks[1].attn.to_k = nn.Linear(8, 8, bias=False)
        with (
            self.assertRaisesRegex(ValueError, "matching cached"),
            dense_reference_execution(self.model, elementwise=False),
        ):
            self.fail("invalid setup must fail before yielding")
        self.assert_restored()

    def test_memory_bound_rejects_before_large_concatenation(self):
        with (
            self.assertRaisesRegex(ValueError, "memory bound"),
            dense_reference_execution(self.model, elementwise=False, maximum_weight_bytes=1),
        ):
            self.fail("invalid memory bound must not yield")
        self.assert_restored()

    def test_exception_and_nested_owner_restore_original_execution(self):
        with (
            torch.inference_mode(),
            self.assertRaisesRegex(RuntimeError, "caller failure"),
            dense_reference_execution(self.model, elementwise=False),
        ):
            with (
                self.assertRaisesRegex(ValueError, "exclusively owned"),
                dense_reference_execution(Model().eval(), elementwise=False),
            ):
                pass
            raise RuntimeError("caller failure")
        self.assert_restored()

    def test_changed_qkv_input_is_rejected_and_next_request_recovers(self):
        with torch.inference_mode(), dense_reference_execution(self.model, elementwise=False):
            self.attention.change_input = True
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                self.attention(self.values)
            self.attention.change_input = False
            self.attention(self.values)
        self.assert_restored()

    def test_bf16_gather_preserves_values_and_training_uses_source(self):
        table = torch.randn(3, 8, dtype=torch.bfloat16)
        indices = torch.tensor([2, 0, 2, 1])
        expected = _index_select_rows(table, indices)
        with dense_reference_execution(self.model, qkv=False, elementwise=False) as record:
            with torch.inference_mode():
                actual = _index_select_rows(table, indices)
            self.assertEqual(record["direct_bf16_gathers"], 1)
            _index_select_rows(table, indices)
            self.assertEqual(record["direct_bf16_gathers"], 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
