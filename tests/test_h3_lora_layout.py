"""Numerical layout/lifetime checks; no model weights or GPU quality claims."""

import pytest

torch = pytest.importorskip("torch")

from vflash.adapters.h3_lora_layout import adapter_names, adapter_scale, converted  # noqa: E402
from vflash.adapters.h3_reference_lora import H3ActivationAdapter  # noqa: E402


def test_fused_projection_mapping_preserves_source_arithmetic():
    torch.manual_seed(19)
    x = torch.randn(3, 7)
    down = torch.randn(16, 7)
    for block, target in (
        ("blocks.49", "transformer_blocks.49"),
        ("token_refiner.blocks.1", "token_refiner.refiner_blocks.1"),
    ):
        up = torch.randn(18, 16)
        parts = converted(f"diffusion_model.{block}.attn.qkv_proj", down, up, "akatz-v1.5")
        assert [name for name, _, _ in parts] == [target + ".attn.to_" + k for k in "qkv"]
        actual = torch.cat(
            [
                torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
                for _, a, b in parts
            ],
            dim=-1,
        )
        torch.testing.assert_close(actual, x @ (up @ down).T)
        up = torch.randn(20, 16)
        name, a, b = converted(f"diffusion_model.{block}.mlp.fc1", down, up, "akatz-v1.5")[0]
        assert name == target + ".ff.net.0.proj"
        original_gate, original_value = (x @ (up @ down).T).chunk(2, dim=-1)
        value, gate = (torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)).chunk(
            2, dim=-1
        )
        torch.testing.assert_close(
            torch.nn.functional.silu(gate) * value,
            torch.nn.functional.silu(original_gate) * original_value,
        )
    assert adapter_scale("akatz-v1.5", {"training_alpha": "8"}, 16) == 1
    assert adapter_scale("larry", None, 128) == 1
    assert adapter_scale("lightx", {"alpha": "8"}, 128) == 0.0625
    with pytest.raises(ValueError, match="finite"):
        adapter_scale("lightx", {"alpha": "nan"}, 128)


def original_fixture(tmp_path):
    from safetensors.torch import save_file

    model = torch.nn.Module()
    weights = {}
    blocks = [f"blocks.{i}" for i in range(50)] + [
        f"token_refiner.blocks.{i}" for i in range(2)
    ]
    for block in blocks:
        for projection, outputs in (
            ("attn.qkv_proj", 6),
            ("attn.out_proj", 2),
            ("mlp.fc1", 4),
            ("mlp.fc2", 2),
        ):
            name = f"diffusion_model.{block}.{projection}"
            a = torch.arange(32, dtype=torch.float32).reshape(16, 2) / 100
            b = torch.arange(outputs * 16, dtype=torch.float32).reshape(outputs, 16) / 200
            weights[name + ".lora_A.weight"] = a
            weights[name + ".lora_B.weight"] = b
            for target, down, up in converted(name, a, b, "akatz-v1.5"):
                parent = model
                names = target.split(".")
                for part in names[:-1]:
                    if part not in parent._modules:
                        parent.add_module(part, torch.nn.Module())
                    parent = parent._modules[part]
                linear = torch.nn.Linear(down.shape[1], up.shape[0], bias=False)
                linear.weight.data.zero_()
                parent.add_module(names[-1], linear)
    path = tmp_path / "original.safetensors"
    save_file(weights, path)
    return model, path, weights


def test_complete_adapter_cpu_hooks_scaling_and_independent_removal(tmp_path):
    model, path, state = original_fixture(tmp_path)
    first = H3ActivationAdapter(model, path, "akatz-v1.5", device="cpu")
    second = H3ActivationAdapter(model, path, "akatz-v1.5", device="cpu")
    assert len(first.calls) == 312
    x = torch.tensor([[2.0, -1.0]])
    q = model.get_submodule("transformer_blocks.0.attn.to_q")
    name = "diffusion_model.blocks.0.attn.qkv_proj"
    expected = x @ (state[name + ".lora_B.weight"][:2] @ state[name + ".lora_A.weight"]).T
    torch.testing.assert_close(q(x), 2 * expected)
    first.strength = 0.25
    second.strength = 0
    torch.testing.assert_close(q(x), 0.25 * expected)
    assert first.calls["transformer_blocks.0.attn.to_q"] == 2 and second.skipped == 1
    first.close()
    torch.testing.assert_close(q(x), torch.zeros_like(expected))
    second.strength = 1
    torch.testing.assert_close(q(x), expected)
    second.close()
    assert all(not item._forward_hooks for item in model.modules())
    assert first.tensor_bytes > 0
    # A late invalid target cleans all hooks installed by that failed adapter.
    model.token_refiner.refiner_blocks._modules["1"].mlp = torch.nn.Identity()
    model.token_refiner.refiner_blocks._modules["1"]._modules.pop("ff")
    with pytest.raises(AttributeError):
        H3ActivationAdapter(model, path, "akatz-v1.5", device="cpu")
    assert all(not item._forward_hooks for item in model.modules())


def test_coverage_rank_and_fused_dimensions_are_not_silently_inferred(tmp_path):
    _, _, weights = original_fixture(tmp_path)
    keys = set(weights)
    assert len(adapter_names(keys, "akatz-v1.5")) == 208
    with pytest.raises(ValueError, match="complete A/B"):
        adapter_names(keys | {"diffusion_model.blocks.0.attn.qkv_proj.alpha"}, "akatz-v1.5")
    damaged = {key.replace("blocks.49.", "blocks.50.") for key in keys}
    with pytest.raises(ValueError, match="coverage"):
        adapter_names(damaged, "akatz-v1.5")
    for a, b in (
        (torch.zeros(8, 2), torch.zeros(6, 8)),
        (torch.zeros(16, 2), torch.zeros(5, 16)),
    ):
        with pytest.raises(ValueError):
            converted("diffusion_model.blocks.0.attn.qkv_proj", a, b, "akatz-v1.5")
    with pytest.raises(ValueError, match="gate/value"):
        converted(
            "diffusion_model.blocks.0.mlp.fc1",
            torch.zeros(16, 2),
            torch.zeros(5, 16),
            "akatz-v1.5",
        )
