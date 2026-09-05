from types import SimpleNamespace

import pytest

from vflash.native import h3_native_denoiser as denoiser
from vflash.native.h3_pinned_arena import PinnedHostArena
from vflash.native.h3_tensor_file import save_safetensors_atomic


def test_loaded_host_weights_outlive_the_mapped_file(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    tensors = {"adaln.table": torch.ones((1, 3, 6, 2), dtype=torch.bfloat16)}
    for name, shape in {
        "attn.qkv": (6, 2),
        "attn.out": (2, 2),
        "ffn.in": (6, 2),
        "ffn.out": (2, 3),
    }.items():
        tensors[name + ".weight"] = torch.full(shape, 2.0, dtype=torch.bfloat16)
        tensors[name + ".scale"] = torch.ones(shape[0], dtype=torch.float16)
    for name in (
        "norm.attn.weight",
        "norm.ffn.weight",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
    ):
        tensors[name] = torch.ones(2, dtype=torch.bfloat16)
    path = tmp_path / "block.safetensors"
    save_safetensors_atomic(path, tensors)
    artifact = SimpleNamespace(
        directory=tmp_path,
        is_complete_block_stack=True,
        adapter_execution="none",
        blocks=(SimpleNamespace(index=0, path=path.name),),
        spec=SimpleNamespace(
            hidden_size=2, num_attention_heads=1, attention_head_dim=2, ffn_dim=3
        ),
        target=SimpleNamespace(
            attention_weight_bits=16, ffn_weight_bits=16, ffn_group_size=None
        ),
    )
    monkeypatch.setattr(denoiser, "load_h3_runtime_artifact", lambda _: artifact)
    # The copying/lifetime contract runs in CPU CI; actual pinning is covered by
    # the target-hardware loader and complete-trajectory checks.
    monkeypatch.setattr(
        denoiser, "PinnedHostArena", lambda: PinnedHostArena(4096, pin_memory=False)
    )

    class HostWeights(denoiser.H3NativeDenoiserBF16Ring):
        def __init__(self, _artifact, host_blocks, **_options):
            self.host_blocks = host_blocks

    loaded = HostWeights.load(tmp_path)
    path.unlink()
    weights = loaded.host_blocks[0]
    assert torch.equal(weights.qkv.values, tensors["attn.qkv.weight"])
    assert torch.equal(weights.qkv.scales, tensors["attn.qkv.scale"])
    assert torch.equal(weights.adaln_table, tensors["adaln.table"])
