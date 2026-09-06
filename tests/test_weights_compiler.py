import hashlib
import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from vflash.compiler import assets
from vflash.compiler.h3 import _publish_directory
from vflash.contracts import ContractError
from vflash.model_assets import ref4_transformer_identity, ref4_weights_source
from vflash.native.h3_runtime_artifact import _validate_source


def test_prepare_and_header_commands_import_without_cuda():
    code = (
        "import sys; import vflash.compiler.__main__; "
        "assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules; "
        "assert 'video_gen' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_raw_receipt_rejects_changes_and_missing_files(monkeypatch, tmp_path):
    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"official-fixed-bytes")
    expected = {
        "size": weight.stat().st_size,
        "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(
        assets, "_required_files", lambda _a, _b, _profile: {"test-weight": (weight, expected)}
    )
    receipt = tmp_path / "receipt.json"
    prepared = assets.prepare_ref4_weights(tmp_path, weight, receipt)
    assert prepared == assets.load_prepared_ref4_weights(receipt)
    with pytest.raises(ContractError, match="already exists"):
        assets.prepare_ref4_weights(tmp_path, weight, receipt)
    original = json.loads(receipt.read_text())
    receipt.write_text(json.dumps({**original, "files": []}))
    with pytest.raises(ContractError, match="incomplete"):
        assets.load_prepared_ref4_weights(receipt)
    receipt.write_text(json.dumps(original))
    weight.write_bytes(b"modified-fixed-bytes")
    with pytest.raises(ContractError, match="changed"):
        prepared.check_unchanged()


def test_raw_verification_does_not_publish_a_bad_hash(monkeypatch, tmp_path):
    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"wrong")
    monkeypatch.setattr(
        assets,
        "_required_files",
        lambda _a, _b, _profile: {"test-weight": (weight, {"size": 5, "sha256": "a" * 64})},
    )
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ContractError, match="bytes differ"):
        assets.prepare_ref4_weights(tmp_path, weight, receipt)
    assert not receipt.exists()


def test_compiled_directory_publication_never_replaces_concurrent_destination(tmp_path):
    staging, destination = tmp_path / "staging", tmp_path / "published"
    staging.mkdir()
    (staging / "new").write_text("result")
    destination.mkdir()
    (destination / "existing").write_text("keep")
    with pytest.raises(FileExistsError):
        _publish_directory(staging, destination)
    assert (destination / "existing").read_text() == "keep"
    assert not (destination / "new").exists()
    free_destination = tmp_path / "free"
    _publish_directory(staging, free_destination)
    assert (free_destination / "new").read_text() == "result"
    assert not staging.exists()


def test_weights_only_source_is_separate_from_legacy_capture_provenance():
    source = ref4_weights_source()
    assert not any("request" in name or "replay" in name for name in source)
    assert source["base_transformer_sha256"] != source["transformer_sha256"]
    assert source["adapter_rank"] == "128"
    assert source["adapter_alpha"] == "8"
    assert source["adapter_strength"] == "1"
    assert (
        _validate_source(source, weight_profile="lightx-ref-turbo4-v0.1", schema_version=5)
        == source
    )
    for change in (
        {"adapter_sha256": "f" * 64},
        {"compile_recipe": "arbitrary"},
        {"request_sha256": "a" * 64},
    ):
        with pytest.raises(ValueError, match="fixed Ref4"):
            _validate_source(
                {**source, **change}, weight_profile="lightx-ref-turbo4-v0.1", schema_version=5
            )
    old = {
        **ref4_transformer_identity(),
        "request_sha256": "a" * 64,
        "replay_case_id": "example",
        "replay_schema_version": "5",
        "replay_packed_input_sha256": "b" * 64,
        "oracle_config_sha256": "c" * 64,
        "oracle_hardware": "sm89",
        "oracle_runtime_sha256": "d" * 64,
        **{
            name: value
            for name, value in source.items()
            if name in {"adapter_repository", "adapter_revision", "adapter_sha256"}
        },
    }
    assert (
        _validate_source(old, weight_profile="lightx-ref-turbo4-v0.1", schema_version=4) == old
    )


@pytest.mark.parametrize(
    "reference,video,audio,text", [(1, 2, 1, 1), (198, 14384, 172, 900), (198, 6232, 172, 32)]
)
def test_request_independent_timesteps_match_runtime_rows(reference, video, audio, text):
    torch = pytest.importorskip("torch")
    from vflash.compiler.math import ref4_timesteps
    from vflash.native.h3_native_scheduler import H3NativeSchedule, compile_h3_row_timestep_plan

    plan = compile_h3_row_timestep_plan(
        H3NativeSchedule.shifted_linear(4),
        video_indices=torch.arange(reference + video),
        audio_indices=torch.arange(reference + video, reference + video + audio),
        text_indices=torch.arange(reference + video + audio, reference + video + audio + text),
        num_condition_video_rows=reference,
        num_condition_audio_rows=0,
        device="cpu",
    )
    rows = ref4_timesteps()
    assert tuple(row.numel() for row in rows) == (2, 3, 3, 3)
    assert all(torch.equal(row, step.timesteps) for row, step in zip(rows, plan, strict=True))


def test_adaln_padding_does_not_influence_real_rows():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    from vflash.compiler.math import adaln_table

    generator = torch.Generator().manual_seed(812)
    embeddings = torch.randn(2, 3, 8, generator=generator)
    embeddings[0, 2] = float("nan")
    weight = torch.randn(24, 8, generator=generator).to(torch.bfloat16)
    bias = torch.randn(24, generator=generator).to(torch.bfloat16)
    actual = adaln_table(
        embeddings, (2, 3), weight, bias, modalities=3, modulations=2, hidden_size=4
    )
    for evaluation, count in enumerate((2, 3)):
        expected = functional.linear(
            functional.silu(embeddings[evaluation, :count]).to(torch.bfloat16), weight, bias
        ).reshape(count * 3, 2, 4)
        assert torch.equal(actual[evaluation, : count * 3], expected)
    assert not actual.isnan().any()
    assert torch.count_nonzero(actual[0, 6:]) == 0


@pytest.mark.parametrize(
    "profile_id",
    ["ref2va-turbo4-exact-sm89", "t2va-turbo4-exact-sm89", "ref2va-turbo4-exact-sm86"],
)
def test_artifact_schema_five_loads_with_no_replay_and_keeps_schema_four(
    monkeypatch, tmp_path, profile_id
):
    from vflash.compiler.h3 import SPEC, compile_target
    from vflash.model_assets import model_profile, weights_source

    profile = model_profile(profile_id)
    TARGET = compile_target(profile_id)
    from vflash.native import h3_runtime_artifact as runtime

    source = weights_source(profile_id)
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    rows = []
    names = sorted(runtime._BLOCK_BASE_TENSOR_NAMES | runtime._BLOCK_RESIDUAL_TENSOR_NAMES)
    for index in range(50):
        relative = f"blocks/block-{index:03d}.safetensors"
        (tmp_path / relative).write_bytes(b"x")
        rows.append(
            {
                "index": index,
                "path": relative,
                "size_bytes": 1,
                "sha256": "a" * 64,
                "adaln_rows": 6 if profile.definition.mode.value == "t2va" else 9,
                "tensors": names,
            }
        )
    manifest = {
        "schema_version": 5,
        "artifact_id": "h3-runtime-test",
        "created_at": "test",
        "status": "complete-block-stack",
        "layout": runtime.H3_RUNTIME_ARTIFACT_LAYOUT,
        "target": asdict(TARGET),
        "spec": asdict(SPEC),
        "nfe": 4,
        "source": source,
        "weight_profile": profile.adapter.profile_id,
        "adapter_execution": "runtime-residual",
        "precision": {
            "attention_weight_bits": 16,
            "attention_activation": "bfloat16",
            "ffn_weight_bits": 16,
            "ffn_group_size": None,
            "ffn_activation": "bfloat16",
            "sensitive": "bfloat16",
            "adaln_table": "bfloat16",
        },
        "compile_environment": {
            "device_type": "cuda",
            "device_name": "test",
            "compute_capability": profile.architecture,
            "torch_version": "2.11.0",
            "cuda_version": "13.0",
            "cudnn_version": "test",
            "adaln_math": "per-evaluation-original-row-count-v1",
            "exact_attention_default": "torch-flash",
        },
        "blocks": rows,
    }
    monkeypatch.setattr(runtime, "inspect_safetensors_header", lambda _p: {})
    monkeypatch.setattr(runtime, "_validate_block_shapes", lambda *_a, **_kw: None)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(manifest))
    assert runtime.load_h3_runtime_artifact(
        tmp_path, verify_content_hashes=False
    ).is_complete_block_stack
    manifest["source"] = {**source, "replay_case_id": "not-allowed"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fixed Ref4"):
        runtime.load_h3_runtime_artifact(tmp_path, verify_content_hashes=False)

    manifest["source"] = source
    manifest["blocks"][0]["adaln_rows"] = 9 if profile.definition.mode.value == "t2va" else 6
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="AdaLN rows"):
        runtime.load_h3_runtime_artifact(tmp_path, verify_content_hashes=False)
    manifest["blocks"][0]["adaln_rows"] = 6 if profile.definition.mode.value == "t2va" else 9
    manifest["target"] = asdict(
        runtime.resolve_h3_artifact_target(
            "rtx4090-48g-sm89-bf16-resident"
            if profile.architecture == "sm86"
            else "rtx3080-20g-sm86-bf16-block-ring"
        )
    )
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target differs"):
        runtime.load_h3_runtime_artifact(tmp_path, verify_content_hashes=False)
