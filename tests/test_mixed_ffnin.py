"""Real CPU tensor loading and lifetime checks, without simulated CUDA proof."""

import gc
import hashlib
import json
import weakref
from contextlib import nullcontext
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.native import h3_mixed_ffnin as mixed

torch = pytest.importorskip("torch")
public_native = pytest.importorskip("vflash.native.h3_native_denoiser")


@pytest.fixture
def native():
    return public_native


def make_fixture(root, native):
    from vflash.native.h3_artifact_contract import H3_ARTIFACT_TARGETS, H3Spec
    from vflash.native.h3_runtime_artifact import H3RuntimeArtifact, H3RuntimeArtifactBlock
    from vflash.native.h3_tensor_file import save_safetensors_atomic

    payloads = root / "candidate"
    payloads.mkdir()
    h, inner, ffn = 16, 16, 32
    manifest = {"complete": True, "alpha": 0.5, "blocks": list(mixed.BLOCKS), "layers": []}
    blocks = []
    generator = torch.Generator().manual_seed(149)
    for index in range(50):
        tensors = {"adaln.table": torch.ones(1, 9, 6, h, dtype=torch.bfloat16)}
        for stem, shape in {
            "attn.qkv": (3 * inner, h),
            "attn.out": (h, inner),
            "ffn.in": (2 * ffn, h),
            "ffn.out": (h, ffn),
        }.items():
            if stem == "ffn.in" and index in mixed.BLOCKS:
                # Actual missing entries make an accidental original BF16 load fail.
                continue
            tensors[stem + ".weight"] = torch.randn(*shape, generator=generator).bfloat16()
            tensors[stem + ".scale"] = torch.ones(shape[0], dtype=torch.float16)
        for stem, count in {
            "norm.attn": h,
            "norm.ffn": h,
            "attn.q_norm": 4,
            "attn.k_norm": 4,
        }.items():
            tensors[stem + ".weight"] = torch.ones(count, dtype=torch.bfloat16)
        for stem, n, k in (
            ("attn.q", inner, h),
            ("attn.k", inner, h),
            ("attn.v", inner, h),
            ("attn.out", h, inner),
            ("ffn.in", 2 * ffn, h),
            ("ffn.out", h, ffn),
        ):
            tensors["adapter." + stem + ".down"] = torch.randn(
                2, k, generator=generator
            ).bfloat16()
            tensors["adapter." + stem + ".up"] = torch.randn(
                n, 2, generator=generator
            ).bfloat16()
        path = root / f"block-{index:03d}.safetensors"
        save_safetensors_atomic(path, tensors)
        blocks.append(
            H3RuntimeArtifactBlock(
                index, path.name, path.stat().st_size, "a" * 64, 9, tuple(tensors)
            )
        )
        if index in mixed.BLOCKS:
            directory = payloads / f"block-{index:03d}"
            directory.mkdir()
            row = {"block": index, "files": {}}
            for name, tensor in (
                (
                    "q",
                    torch.randint(
                        -127, 128, (2 * ffn, h), dtype=torch.int8, generator=generator
                    ),
                ),
                ("scale", torch.full((2 * ffn,), 0.125)),
                ("d", torch.linspace(0.5, 2, h)),
            ):
                data = tensor.numpy().tobytes()
                (directory / f"{name}.bin").write_bytes(data)
                row["files"][name] = {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            manifest["layers"].append(row)
    (payloads / "manifest.json").write_text(json.dumps(manifest))
    artifact = H3RuntimeArtifact(
        root,
        "h3-runtime-small-test",
        "",
        "complete",
        "blocks",
        H3_ARTIFACT_TARGETS["rtx4090-48g-sm89-bf16-resident"],
        H3Spec(50, h, 4, 4, ffn, 4, 4, 4, (1, 1, 1)),
        4,
        "lightx-ref-turbo4-v0.1",
        {"oracle_profile": "ref2va-adapter-test"},
        tuple(blocks),
        adapter_execution="runtime-residual",
    )
    manifest = {
        "schema_version": 1,
        "profile_id": mixed.PROFILE_ID,
        "base_signature": mixed.artifact_signature(artifact),
        "alpha": 0.5,
        "layers": manifest["layers"],
    }
    (payloads / "manifest.json").write_text(json.dumps(manifest))
    return artifact, payloads


def load_hosts(artifact, payloads, native, monkeypatch):
    from vflash.native.h3_pinned_arena import PinnedHostArena

    monkeypatch.setattr(mixed, "_kitchen_linear", lambda: None)
    runtime = mixed.mixed_ffn_in_runtime(payloads)
    ring = runtime._block_ring_type

    class CPUHostCapture(ring):
        def __init__(self, artifact, host_blocks, **kwargs):
            self.artifact, self.host_blocks = artifact, host_blocks

    arenas = []

    def arena():
        value = PinnedHostArena(chunk_bytes=16384, pin_memory=False)
        arenas.append(value)
        return value

    monkeypatch.setattr(native, "PinnedHostArena", arena)
    captured = CPUHostCapture.load(artifact, device="cpu")
    return captured.host_blocks, arenas, ring


def test_loader_never_reads_selected_bf16_and_closes_real_mmaps(tmp_path, native, monkeypatch):
    artifact, payloads = make_fixture(tmp_path, native)
    host, arenas, _ = load_hosts(artifact, payloads, native, monkeypatch)
    assert len(host) == 50 and all(a._closed and a._buffer is None for a in arenas)
    assert sum(isinstance(x.ffn_in, mixed.BalancedFFNIn) for x in host) == 31
    for index, block in enumerate(host):
        with native.H3MappedSafetensor(
            artifact.directory / artifact.blocks[index].path
        ) as source:
            for value, name in (
                (block.qkv.values, "attn.qkv.weight"),
                (block.attention_out.values, "attn.out.weight"),
                (block.ffn_out.values, "ffn.out.weight"),
                (block.attention_norm, "norm.attn.weight"),
                (block.ffn_norm, "norm.ffn.weight"),
                (block.query_norm, "attn.q_norm.weight"),
                (block.key_norm, "attn.k_norm.weight"),
                (block.ffn_in_residual.down, "adapter.ffn.in.down"),
                (block.ffn_in_residual.up, "adapter.ffn.in.up"),
            ):
                assert torch.equal(value, source.load(name))
        if index not in mixed.BLOCKS:
            with native.H3MappedSafetensor(
                artifact.directory / artifact.blocks[index].path
            ) as source:
                original = source.load("ffn.in.weight")
                assert torch.equal(block.ffn_in.values, original)
                del original
            assert block.ffn_in.values.dtype == torch.bfloat16
        assert block.qkv.values.dtype == block.ffn_out.values.dtype == torch.bfloat16
        assert block.ffn_in_residual.scaling == 0.0625
    assert sum(mixed.block_bytes(x) for x in host) == sum(a.payload_bytes for a in arenas)
    assert not torch.cuda.is_initialized()


def test_host_arena_storage_releases_without_retained_bf16_backing(
    tmp_path, native, monkeypatch
):
    artifact, payloads = make_fixture(tmp_path, native)

    def scope():
        host, arenas, _ = load_hosts(artifact, payloads, native, monkeypatch)
        storages = {
            x.ffn_in.values.untyped_storage().data_ptr(): x.ffn_in.values.untyped_storage()
            for x in host
        }
        refs = [weakref.ref(x) for x in storages.values()]
        assert all(r() is not None for r in refs)
        return refs, arenas

    refs, arenas = scope()
    gc.collect()
    assert all(r() is None for r in refs)
    assert all(a._buffer is None for a in arenas)


def test_mixed_slot_aliases_capacity_and_copies_only_selected_representation(native):
    base = native.H3BF16Weight(
        torch.zeros(64, 16, dtype=torch.bfloat16),
        torch.ones(64, dtype=torch.float16),
        16,
        None,
        16,
        64,
    )
    slot = mixed.FFNInSlot(base)
    q = (
        torch.arange(1024, dtype=torch.int32)
        .remainder(255)
        .sub(127)
        .to(torch.int8)
        .view(64, 16)
    )
    source = mixed.BalancedFFNIn(q, torch.full((64,), 0.5), torch.ones(16), 2)
    assert (
        slot.quantized.untyped_storage().data_ptr() == base.values.untyped_storage().data_ptr()
    )
    assert slot.capacity_bytes == 2048 + 256 + 64
    slot.copy_from(source)
    assert torch.equal(slot.quantized, q) and slot.block_index == 2
    assert torch.equal(
        base.values.view(torch.uint8).reshape(-1)[1024:], torch.zeros(1024, dtype=torch.uint8)
    )
    retained = replace(base, values=torch.full((64, 16), 3, dtype=torch.bfloat16))
    slot.copy_from(retained)
    assert slot.block_index is None and torch.equal(slot.values, retained.values)
    slot.copy_from(source)
    assert torch.equal(slot.quantized, q)
    with pytest.raises(ValueError, match="dimensions"):
        slot.copy_from(replace(source, values=q[:32], scales=source.scales[:32]))
    assert torch.equal(slot.quantized, q)


def test_real_ring_loop_preserves_four_nfe_route_and_lora_inputs(tmp_path, native, monkeypatch):
    artifact, payloads = make_fixture(tmp_path, native)
    host, _, _ = load_hosts(artifact, payloads, native, monkeypatch)
    states = torch.randn(1, 3, 16).bfloat16()
    original_states = states.clone()
    provider_calls = []

    def provider(x, q, scale, **kwargs):
        assert kwargs == {"out_dtype": torch.bfloat16, "convrot": False}
        assert x.dtype == torch.float32
        provider_calls.append((q.untyped_storage().data_ptr(), x.clone()))
        return (torch.nn.functional.linear(x, q.float()) * scale).bfloat16()

    monkeypatch.setattr(mixed, "_kitchen_linear", lambda: provider)
    runtime = mixed.mixed_ffn_in_runtime(payloads)
    ring = object.__new__(runtime._block_ring_type)
    ring.artifact, ring.host_blocks, ring.device = artifact, host, torch.device("cpu")
    slots = []
    lora_operands = []
    for source in host[:2]:
        slot = object.__new__(ring.block_type)
        weights = native._empty_bf16_block_like(source, "cpu")
        native._H3BlockOperations.__init__(slot, artifact, weights)
        slot.weights = replace(weights, ffn_in=mixed.FFNInSlot(weights.ffn_in))
        slot.device = ring.device
        slot.elementwise_backend = "torch-eager"
        slot.adapter_fusion_backend = "torch-eager"
        original_residual = slot._residual_linear_unscaled

        def residual(self, x, weight, original=original_residual):
            lora_operands.append(x.data_ptr())
            return original(x, weight)

        slot._residual_linear_unscaled = MethodType(residual, slot)

        def forward(self, x, invocation):
            output = self._ffn_input(x)
            assert output.shape == (1, 3, 32) and output.dtype == torch.bfloat16
            return x

        slot.forward_prevalidated = MethodType(forward, slot)
        slots.append(slot)
    ring.slots = tuple(slots)
    ring.copy_stream = SimpleNamespace(wait_event=lambda _: None)
    ring.ready_events = ring.compute_done_events = tuple(
        SimpleNamespace(record=lambda _: None) for _ in range(2)
    )
    fake_torch = SimpleNamespace(
        **{
            **vars(torch),
            "cuda": SimpleNamespace(
                stream=lambda _: nullcontext(),
                current_stream=lambda _: SimpleNamespace(wait_event=lambda _: None),
            ),
        }
    )
    monkeypatch.setattr(native, "_torch", lambda: fake_torch)
    transfers = []

    def copy(dst, src):
        weight = src.ffn_in
        transfers.append(
            weight.values.nbytes
            + (
                weight.scales.nbytes + weight.balance.nbytes
                if isinstance(weight, mixed.BalancedFFNIn)
                else 0
            )
        )
        mixed.copy_block(dst, src)

    ring._copy_block = copy
    for _ in range(4):
        output, _ = ring.forward_prevalidated(states, object())
        assert output is states
    assert len(provider_calls) == 124 and len(lora_operands) == 200
    assert all(x == states.data_ptr() for x in lora_operands) and torch.equal(
        states, original_states
    )
    assert sum(transfers) == 4 * (31 * (1024 + 256 + 64) + 19 * 2048)
    assert len({x[0] for x in provider_calls}) == 2
    expected = states.float() / torch.linspace(0.5, 2, 16)
    assert all(torch.equal(x[1], expected) for x in provider_calls)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["base", "layer", "bytes", "digest", "scale", "symlink"])
def test_sidecar_rejects_wrong_identity_or_payload(tmp_path, native, fault):
    artifact, directory = make_fixture(tmp_path, native)
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    if fault == "base":
        data["base_signature"] = "0" * 64
    elif fault == "layer":
        data["layers"][0]["block"] = 0
    elif fault == "bytes":
        data["layers"][0]["files"]["q"]["bytes"] += 1
    elif fault == "digest":
        data["layers"][0]["files"]["q"]["sha256"] = "0" * 64
    elif fault == "scale":
        target = directory / "block-002" / "scale.bin"
        target.write_bytes(torch.zeros(64).numpy().tobytes())
        data["layers"][0]["files"]["scale"]["sha256"] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
    else:
        target = directory / "block-002" / "q.bin"
        external = tmp_path / "outside.bin"
        target.rename(external)
        target.symlink_to(external)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        payloads = mixed.FFNInPayloads(directory, artifact, verify_content_hashes=True)
        payloads.load(2, output_features=64, input_features=16)


def test_ingest_and_normal_load_have_separate_hash_cost(tmp_path, native, monkeypatch):
    artifact, directory = make_fixture(tmp_path, native)
    verified = mixed.FFNInPayloads(directory, artifact, verify_content_hashes=True)
    assert len(verified.entries) == 31
    original = mixed.hashlib.sha256
    sizes = []

    def digest(data=b""):
        sizes.append(len(data))
        return original(data)

    monkeypatch.setattr(mixed.hashlib, "sha256", digest)
    loaded = mixed.FFNInPayloads(directory, artifact)
    loaded.load(2, output_features=64, input_features=16)
    assert len(sizes) == 2  # Only manifest bytes and existing base block digest metadata.
    assert loaded.manifest_sha256 == verified.manifest_sha256


def test_missing_cuda_provider_is_an_actionable_contract_error(monkeypatch):
    def missing(_):
        raise mixed.PackageNotFoundError("comfy-kitchen")

    monkeypatch.setattr(mixed, "version", missing)
    with pytest.raises(ContractError, match="install the w8 extra"):
        mixed._kitchen_linear()
