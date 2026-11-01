import pytest
from test_pika_flex import manifest


def test_original_pika_pairs_keep_names_rank_scale_and_bf16_execution(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from vflash.adapters.h3_reference_lora import H3ActivationAdapter

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16) for _ in range(312)]
    )
    state = {}
    for i, layer in enumerate(model.layers):
        layer.weight.data.copy_(torch.eye(2))
        state[f"layers.{i}.lora_A.default.weight"] = torch.full((128, 2), 0.30123)
        state[f"layers.{i}.lora_B.default.weight"] = torch.full((2, 128), 0.001234)
    path = tmp_path / "original.safetensors"
    save_file(state, path)
    original_to = torch.Tensor.to

    def cpu_only(tensor, *args, **kwargs):
        if args and args[0] == "cuda:0":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_only)
    adapter = H3ActivationAdapter(model, path, "pika", manifest=manifest("fl2va"))
    sample = torch.tensor([[1, 2]], dtype=torch.bfloat16)
    a = state["layers.0.lora_A.default.weight"].to(torch.bfloat16)
    b = state["layers.0.lora_B.default.weight"].to(torch.bfloat16)
    expected = sample + torch.nn.functional.linear(torch.nn.functional.linear(sample, a), b)
    assert torch.equal(model.layers[0](sample), expected)
    assert len(adapter.calls) == 312 and adapter.calls["layers.0"] == 1
    assert adapter.tensor_bytes == sum(t.numel() * 2 for t in state.values())
    # Exercise the real installed hook, retaining the historical BF16 operation
    # order for fractional strengths and checking disabled adapters too.
    for strength in (0.0, 0.25, 1.0, 1.375):
        adapter.strength = strength
        delta = torch.nn.functional.linear(torch.nn.functional.linear(sample, a), b)
        reference = sample + delta.to(sample.dtype) * 1.0 * strength
        assert torch.equal(model.layers[0](sample), reference)
    adapter.close()
    assert torch.equal(model.layers[0](sample), sample)
    # A late bad target must remove already installed hooks from every earlier layer.
    del model.layers[-1]
    with pytest.raises(AttributeError):
        H3ActivationAdapter(model, path, "pika", manifest=manifest("fl2va"))
    assert all(not layer._forward_hooks for layer in model.layers)


def test_pika_never_inherits_another_adapter_family_or_missing_manifest():
    torch = pytest.importorskip("torch")
    from vflash.adapters.h3_reference_lora import H3ActivationAdapter

    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="family"):
        H3ActivationAdapter(model, "unused", "unknown")
    with pytest.raises(ValueError, match="manifest"):
        H3ActivationAdapter(model, "unused", "pika")
    with pytest.raises(ValueError):
        H3ActivationAdapter(model, "unused", "pika", manifest={**manifest("fl2va"), "alpha": 8})
