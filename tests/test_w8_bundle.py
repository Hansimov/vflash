"""CPU contracts for portable W8 files and precision-specific ring operations."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, replace

import pytest

from vflash.native.h3_native_conditioning_runtime import validate_declared_schedule
from vflash.native.h3_native_denoiser import H3NativeBlockWeights
from vflash.native.h3_native_scheduler import H3NativeSchedule
from vflash.native.h3_w8_bundle import (
    ADAPTER,
    FORMAT,
    MATRIX_SHAPES,
    SCHEDULE_SPECS,
    SOURCE,
    SPEC,
    WEIGHT_PROFILE,
    NativeW8Bundle,
    NativeW8BundleError,
    block_specs,
)
from vflash.native.h3_w8a8 import W8Weight, copy_block, empty_weight, validate_weight

torch = pytest.importorskip("torch")


def sparse_tensor_file(path, specs):
    """Valid headers with sparse zero payloads; no real model is a test dependency."""
    sizes = {"I8": 1, "BF16": 2, "F32": 4, "I64": 8}
    cursor = 0
    header = {}
    for name, (dtype, shape) in sorted(specs.items()):
        count = 1
        for dimension in shape:
            count *= dimension
        end = cursor + count * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, end]}
        cursor = end
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.truncate(8 + len(encoded) + cursor)
    return {
        "size_bytes": path.stat().st_size,
        "sha256": "0" * 64,
        "tensors": {
            name: {"dtype": row["dtype"], "shape": row["shape"]} for name, row in header.items()
        },
    }


@pytest.fixture
def bundle_directory(tmp_path):
    from vflash.native.h3_runtime_auxiliary import H3_RUNTIME_AUXILIARY_SPECS

    (tmp_path / "blocks").mkdir()
    files = {}
    first = tmp_path / "blocks/block-000.safetensors"
    row = sparse_tensor_file(first, block_specs())
    for index in range(50):
        relative = f"blocks/block-{index:03d}.safetensors"
        if index:
            os.link(first, tmp_path / relative)
        files[relative] = row
    for relative, specs in (
        ("auxiliary.safetensors", H3_RUNTIME_AUXILIARY_SPECS),
        ("schedule.safetensors", SCHEDULE_SPECS),
    ):
        files[relative] = sparse_tensor_file(tmp_path / relative, specs)
    schedule = H3NativeSchedule.shifted_linear(4, video_shift=12, audio_shift=3)
    value = {
        "schema_version": 1,
        "format": FORMAT,
        "mode": "ref2va",
        "spec": {**asdict(SPEC), "patch_size": list(SPEC.patch_size)},
        "source": SOURCE,
        "weight_profile": WEIGHT_PROFILE,
        "adapter": ADAPTER,
        "quantization": {
            "weights": "symmetric INT8 per output row",
            "scales": "FP32",
            "activations": "dynamic symmetric INT8 per input row",
            "rotation": False,
            "main_matrices": list(MATRIX_SHAPES),
            "output_dtype": "BF16",
            "weight_formula": (
                "scale=max(max(abs(row)),1e-10)/127; q=clamp(round(row/scale),-127,127)"
            ),
        },
        "schedule": {
            "schema_version": 1,
            "nfe": 4,
            "update_rule": "training_euler",
            "video_sigmas": list(schedule.video_sigmas),
            "audio_sigmas": list(schedule.audio_sigmas),
        },
        "files": files,
        "payload_bytes": sum(row["size_bytes"] for row in files.values()),
    }
    (tmp_path / "bundle.json").write_text(json.dumps(value))
    return tmp_path


def mutate_manifest(path, edit):
    manifest = path / "bundle.json"
    value = json.loads(manifest.read_text())
    edit(value)
    manifest.write_text(json.dumps(value))


def test_model_contract_and_compiled_schedule_are_explicit(bundle_directory):
    bundle = NativeW8Bundle(bundle_directory)
    assert len(bundle.paths) == 52
    assert bundle.target.attention_weight_bits == bundle.target.ffn_weight_bits == 8
    assert bundle.adapter_execution == "runtime-residual"
    validate_declared_schedule(
        bundle.schedule,
        expected_nfe=4,
        expected_scheduler="h3-training-euler",
        expected_video_flow_shift=12,
        expected_audio_flow_shift=3,
    )
    assert bundle.schedule_overlay().base_artifact_id == bundle.artifact_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "t2va"),
        ("weight_profile", "lightx-turbo8-v1.0"),
        ("adapter", {**ADAPTER, "internal_scale": 1.0}),
        ("source", {**SOURCE, "model_revision": "1" * 40}),
    ],
)
def test_foreign_model_contract_is_rejected(bundle_directory, field, value):
    mutate_manifest(bundle_directory, lambda model: model.update({field: value}))
    with pytest.raises(NativeW8BundleError, match="contract"):
        NativeW8Bundle(bundle_directory)


def test_mixed_bf16_matrix_header_is_rejected(bundle_directory):
    target = bundle_directory / "blocks/block-012.safetensors"
    target.unlink()
    specs = block_specs()
    specs["attn.qkv.weight"] = ("BF16", specs["attn.qkv.weight"][1])
    row = sparse_tensor_file(target, specs)

    def edit(value):
        value["files"]["blocks/block-012.safetensors"] = row
        value["payload_bytes"] = sum(r["size_bytes"] for r in value["files"].values())

    mutate_manifest(bundle_directory, edit)
    with pytest.raises(NativeW8BundleError, match="model ABI"):
        NativeW8Bundle(bundle_directory)


def test_missing_and_symlinked_blocks_are_rejected(bundle_directory):
    target = bundle_directory / "blocks/block-012.safetensors"
    target.unlink()
    target.symlink_to(bundle_directory / "blocks/block-000.safetensors")
    with pytest.raises(NativeW8BundleError, match="escaped"):
        NativeW8Bundle(bundle_directory)
    mutate_manifest(
        bundle_directory, lambda model: model["files"].pop("blocks/block-012.safetensors")
    )
    with pytest.raises(NativeW8BundleError, match="exactly 50"):
        NativeW8Bundle(bundle_directory)


def test_bad_scale_exception_does_not_escape_mapped_tensors(bundle_directory):
    bundle = NativeW8Bundle(bundle_directory)
    # Sparse payloads have zero scales. Retain the exception traceback while
    # verifying that the owned mapping can still close instead of leaking views.
    with (
        pytest.raises(NativeW8BundleError, match="positive and finite") as captured,
        bundle.mapped_block(0),
    ):
        pytest.fail("zero scales must not enter the ring")
    assert captured.value.__traceback__ is not None


def test_ring_copy_preserves_dynamic_scales_and_rejects_mixed_precision():
    weight = W8Weight(
        torch.tensor([[1, -2], [3, 4]], dtype=torch.int8), torch.tensor([0.5, 0.25]), 2, 2
    )
    common = dict(
        adaln_table=torch.zeros(1),
        attention_norm=torch.ones(2),
        ffn_norm=torch.ones(2),
        query_norm=torch.ones(2),
        key_norm=torch.ones(2),
    )
    source = H3NativeBlockWeights(
        **common, qkv=weight, attention_out=weight, ffn_in=weight, ffn_out=weight
    )
    destination = replace(
        source,
        **{
            name: empty_weight(weight, "cpu")
            for name in ("qkv", "attention_out", "ffn_in", "ffn_out")
        },
    )
    copy_block(destination, source)
    for name in ("qkv", "attention_out", "ffn_in", "ffn_out"):
        assert torch.equal(getattr(destination, name).values, weight.values)
        assert torch.equal(getattr(destination, name).scales, weight.scales)
        assert getattr(destination, name).scales.data_ptr() != weight.scales.data_ptr()
    with pytest.raises(ValueError, match="INT8"):
        validate_weight(replace(weight, values=weight.values.bfloat16()))


def test_w8_runtime_requires_cuda_without_changing_bf16_module_functions(bundle_directory):
    from vflash.native import h3_native_denoiser as native
    from vflash.native.h3_w8_runtime import NativeW8Runtime

    before = (native.load_h3_native_block, native._copy_bf16_block_)
    if torch.cuda.is_available():
        pytest.skip("this failure-path test is run with CUDA explicitly hidden")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        NativeW8Runtime(bundle_directory)
    assert before == (native.load_h3_native_block, native._copy_bf16_block_)
    assert not torch.cuda.is_initialized()
