"""Reuse unmerged prepared FP8 base matrices with independent native schedules.

This is an explicit approximate-weight cohort. It does not relabel the cache as
original BF16 weights. No four-step adapters or hybrid modulation are loaded.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from vflash.native.h3_fp8_linear import RowFP8Execution


class CachedMatrix(torch.nn.Linear):
    def __init__(self, original, values, scales, execution):
        torch.nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        assert values.shape == original.weight.shape
        assert values.dtype == torch.float8_e4m3fn
        assert scales.dtype == torch.float32 and scales.shape == (self.out_features, 1)
        assert original.bias is None
        self.weight = torch.nn.Parameter(values.to("cuda:0"), requires_grad=False)
        self.register_buffer("row_scale", scales.to("cuda:0"))
        self.bias = None
        self.execution = execution

    def forward(self, values):
        return self.execution.linear(values, self.weight, self.row_scale)


class CachedProjection(torch.nn.Linear):
    """Frozen timestep projection; the descriptor is not a trainable base matrix."""

    def __init__(self, original, clock, index):
        torch.nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        # Only dtype/geometry are consumed by the model and adapter loader. The
        # actual original projection, including bias, is in the declared table.
        descriptor = torch.zeros((1, 1), device="cuda:0", dtype=torch.bfloat16)
        self.weight = torch.nn.Parameter(
            descriptor.expand(self.out_features, self.in_features), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            descriptor.flatten().expand(self.out_features), requires_grad=False
        )
        self.clock, self.index = clock, index

    def forward(self, values):
        result = self.clock.tensors[f"block-{self.index:03d}-step-{self.clock.step:02d}"]
        assert values.shape == (result.shape[0], self.in_features)
        return result


class SamplingClock:
    def __init__(self, root, *, manifest_name="manifest.json"):
        self.manifest_name = manifest_name
        self.root = Path(root)
        self.manifest = json.loads((self.root / self.manifest_name).read_text())
        assert self.manifest["status"] == "completed"
        self.tensors = {}
        self.step = 0
        self.plan = None

    def select(self, nfe, shift):
        name = f"steps{nfe}-shift{int(shift)}.safetensors"
        row = next(x for x in self.manifest["files"] if x["name"] == name)
        path = self.root / name
        assert path.stat().st_size == row["bytes"]
        self.tensors = load_file(str(path), device="cuda:0")
        self.plan = row["plan"]
        self.step = 0

    def before(self, _model, _args, kwargs):
        expected = torch.tensor(
            self.plan["timesteps"][self.step], device="cuda:0", dtype=torch.float32
        )
        assert torch.equal(kwargs["timestep"], expected), (
            "Actual native clock differs from compiled original weights"
        )

    def time_embedding(self, _values):
        return self.tensors[f"time-{self.step}"]


def validate_cached_structure(model, root, table_root, *, manifest_name="manifest.json"):
    root, table_root = Path(root), Path(table_root)
    expected = dict(model.named_parameters())
    covered = set()

    def check(name, shape):
        assert name in expected and tuple(shape) == tuple(expected[name].shape), name
        covered.add(name)

    directory = root / "model/transformer"
    inventory = json.loads(
        (directory / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    for filename in sorted(set(inventory.values())):
        with safe_open(str(directory / filename), framework="pt", device="cpu") as source:
            for name in source.keys():  # noqa: SIM118 - safetensors has no iterator
                assert not name.startswith("transformer_blocks.")
                check(name, source.get_slice(name).get_shape())
    with safe_open(str(root / "auxiliary.safetensors"), framework="pt", device="cpu") as source:
        for name in source.keys():  # noqa: SIM118 - safetensors has no iterator
            check(name, source.get_slice(name).get_shape())
    for index in range(50):
        prefix = f"transformer_blocks.{index}."
        with safe_open(
            str(root / "artifact/blocks" / f"block-{index:03d}.safetensors"),
            framework="pt",
            device="cpu",
        ) as source:
            rows, cols = source.get_slice("attn.qkv.weight").get_shape()
            assert source.get_slice("attn.qkv.weight.row_scale").get_shape() == [rows, 1]
            for suffix in ("q", "k", "v"):
                check(prefix + "attn.to_" + suffix + ".weight", [rows // 3, cols])
            for target, key in [
                ("attn.to_out.0.weight", "attn.out.weight"),
                ("ff.net.0.proj.weight", "ffn.in.weight"),
                ("ff.net.2.weight", "ffn.out.weight"),
                ("norm1.weight", "norm.attn.weight"),
                ("norm2.weight", "norm.ffn.weight"),
                ("attn.norm_q.weight", "attn.q_norm.weight"),
                ("attn.norm_k.weight", "attn.k_norm.weight"),
            ]:
                check(prefix + target, source.get_slice(key).get_shape())
        covered.update((prefix + "adaln_proj.linear.weight", prefix + "adaln_proj.linear.bias"))
    covered.update(("norm_out.linear.weight", "norm_out.linear.bias"))
    assert covered == set(expected), sorted(set(expected) - covered)
    manifest = json.loads((table_root / manifest_name).read_text())
    assert manifest["status"] == "completed"
    for row in manifest["files"]:
        path = table_root / row["name"]
        assert path.stat().st_size == row["bytes"]
        with safe_open(str(path), framework="pt", device="cpu") as source:
            assert len(list(source.keys())) == row["plan"]["nfe"] * 52
            for step, times in enumerate(row["plan"]["timesteps"]):
                assert source.get_slice(f"time-{step}").get_shape() == [len(times), 2688]
                for index in range(51):
                    assert source.get_slice(
                        f"block-{index:03d}-step-{step:02d}"
                    ).get_shape() == [len(times), (18 if index < 50 else 2) * 5376]
    return len(covered)


def load_cached_trunk(model, root, clock):
    """Assign each declared base matrix once, preserving runtime LoRA independence."""
    root = Path(root)
    execution = RowFP8Execution("w8a8", activation_quantizer="triton", fast_accum=False)
    assigned = set()

    def assign(name, value):
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent)
        assert module._parameters[leaf].shape == value.shape
        module._parameters[leaf] = torch.nn.Parameter(value.to("cuda:0"), requires_grad=False)
        assigned.add(name)

    directory = root / "model/transformer"
    inventory = json.loads(
        (directory / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    for filename in sorted(set(inventory.values())):
        with safe_open(str(directory / filename), framework="pt", device="cpu") as source:
            for name in source.keys():  # noqa: SIM118 - safetensors has no iterator
                assert not name.startswith("transformer_blocks."), (
                    "Conditioning prefix must not embed a different trunk"
                )
                assign(name, source.get_tensor(name))
    with safe_open(str(root / "auxiliary.safetensors"), framework="pt", device="cpu") as source:
        for name in source.keys():  # noqa: SIM118 - safetensors has no iterator
            assign(name, source.get_tensor(name))
    for index, block in enumerate(model.transformer_blocks):
        path = root / "artifact/blocks" / f"block-{index:03d}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as source:
            qkv = source.get_tensor("attn.qkv.weight")
            scales = source.get_tensor("attn.qkv.weight.row_scale")
            for suffix, values, scale in zip(
                ("q", "k", "v"), qkv.chunk(3), scales.chunk(3), strict=True
            ):
                name = "attn.to_" + suffix
                block.set_submodule(
                    name, CachedMatrix(block.get_submodule(name), values, scale, execution)
                )
            for target, key in [
                ("attn.to_out.0", "attn.out.weight"),
                ("ff.net.0.proj", "ffn.in.weight"),
                ("ff.net.2", "ffn.out.weight"),
            ]:
                block.set_submodule(
                    target,
                    CachedMatrix(
                        block.get_submodule(target),
                        source.get_tensor(key),
                        source.get_tensor(key + ".row_scale"),
                        execution,
                    ),
                )
            for target, key in [
                ("norm1.weight", "norm.attn.weight"),
                ("norm2.weight", "norm.ffn.weight"),
                ("attn.norm_q.weight", "attn.q_norm.weight"),
                ("attn.norm_k.weight", "attn.k_norm.weight"),
            ]:
                assign(f"transformer_blocks.{index}." + target, source.get_tensor(key))
        block.adaln_proj.linear = CachedProjection(block.adaln_proj.linear, clock, index)
    model.norm_out.linear = CachedProjection(model.norm_out.linear, clock, 50)
    model.time_embedder.forward = clock.time_embedding
    model.rope.inv_freq = model.rope._compute_inv_freq(device=torch.device("cuda:0"))
    assert not any(v.is_meta for v in model.parameters())
    assert not any(v.is_meta for v in model.buffers())
    model.eval().requires_grad_(False)
    return execution
