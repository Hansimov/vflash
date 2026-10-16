"""Synthetic loader tests runnable with unittest in a dependency-only CPU image."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

if not all(importlib.util.find_spec(name) for name in ("torch", "safetensors")):
    raise unittest.SkipTest("optional CPU tensor dependencies")

import torch
from safetensors.torch import save_file

from vflash.native.h3_fp8_linear import RowFP8Execution, RowFP8Linear
from vflash.native.h3_fp8_runtime import create_row_fp8_native_loaders
from vflash.native.h3_fp8_text import RowFP8TextShard, load_row_fp8_text_encoder


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(16, dtype=torch.bfloat16)

    def forward(self, value):
        return self.norm(self.projection(value))


class NativeFactoryTests(unittest.TestCase):
    def test_owners_do_not_replace_default_block(self):
        from vflash.native import h3_native_denoiser as denoiser

        original = denoiser.H3NativeBlockBF16Resident
        first = create_row_fp8_native_loaders(RowFP8Execution(), weights_directory=Path("a"))
        second = create_row_fp8_native_loaders(RowFP8Execution(), weights_directory=Path("b"))
        self.assertIsNot(first.block_class, second.block_class)
        self.assertIs(denoiser.H3NativeBlockBF16Resident, original)
        values = torch.ones(4, 16).to(torch.float8_e4m3fn)
        scales = torch.ones(4, 1)
        weight = denoiser.H3BF16Weight(values, scales, 8, None, 16, 4)
        owner = first.block_class.__new__(first.block_class)
        actual = owner._resident_weight(weight, "cpu")
        self.assertEqual(actual.bits, 8)
        torch.testing.assert_close(
            owner._linear(torch.ones(2, 16), actual), torch.full((2, 4), 16.0)
        )

    def test_invalid_row_limits_fail_before_loading(self):
        for options in (
            {"token_tile_rows": True},
            {"token_tile_rows": 1025},
            {"token_tile_rows": 65552},
            {"whole_block_rows": 32768},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_row_fp8_native_loaders(
                    RowFP8Execution(), weights_directory=Path("unused"), **options
                )


@unittest.skipUnless(importlib.util.find_spec("accelerate"), "complete-pipeline dependency")
class TextLoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.manual_seed(717)
        self.codes = torch.randn(16, 16).to(torch.float8_e4m3fn)
        self.scales = torch.full((16, 1), 0.125)
        self.bias = torch.randn(16, dtype=torch.bfloat16)
        self.tensors = {
            "projection.weight": self.codes,
            "projection.weight.row_scale": self.scales,
            "projection.bias": self.bias,
        }
        save_file(self.tensors, self.root / "packed.safetensors")
        save_file(
            {
                "norm.weight": torch.ones(16, dtype=torch.bfloat16),
                "norm.bias": torch.zeros(16, dtype=torch.bfloat16),
            },
            self.root / "unchanged.safetensors",
        )
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        name: "unchanged.safetensors" for name in ("norm.weight", "norm.bias")
                    }
                }
            )
        )
        self.shard = RowFP8TextShard(self.root / "packed.safetensors", ("projection.weight",))
        self.execution = RowFP8Execution()

    def load(self, shards=None, **kwargs):
        return load_row_fp8_text_encoder(
            Encoder,
            None,
            (self.shard,) if shards is None else shards,
            self.execution,
            self.root,
            **kwargs,
        )

    def test_real_assignment_and_arithmetic_preserve_unchanged_tensors(self):
        copies = []
        model = self.load(submit=lambda values: copies.append(values))
        self.assertIsInstance(model.projection, RowFP8Linear)
        self.assertEqual(len(copies), 2)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad and not p.is_meta for p in model.parameters()))
        value = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        expected = torch.nn.functional.layer_norm(
            torch.nn.functional.linear(
                value, (self.codes.float() * self.scales).to(value.dtype), self.bias
            ),
            (16,),
        )
        torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
        self.assertEqual(self.execution.calls, {"weight_only": 1, "w8a8": 0})
        self.assertFalse(torch.cuda.is_initialized())

    def test_duplicate_declaration_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load((self.shard, self.shard))

    def test_incomplete_checkpoint_rejected(self):
        del self.tensors["projection.weight.row_scale"]
        save_file(self.tensors, self.root / "packed.safetensors")
        with self.assertRaisesRegex(ValueError, "missing unchanged"):
            self.load()

    def test_invalid_scale_dtype_rejected_during_load(self):
        self.tensors["projection.weight.row_scale"] = self.scales.bfloat16()
        save_file(self.tensors, self.root / "packed.safetensors")
        with self.assertRaisesRegex(ValueError, "invalid dtype"):
            self.load()

    def test_copy_failure_does_not_publish_a_partial_model(self):
        def fail(values):
            raise RuntimeError("copy failed")

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            self.load(submit=fail)
        self.assertIsInstance(self.load().projection, RowFP8Linear)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
