import json
from pathlib import Path

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.assets import load_prepared_pipeline_assets, prepare_pipeline_assets
from vflash.pipeline.contracts import PipelineAssets
from vflash.pipeline.derivation import apply_exact_subset


def fixture(tmp_path, **change):
    file = tmp_path / "config.json"
    file.write_bytes(b"x" * 80)
    value = {
        "schema_version": 1,
        "kind": "exact-h3-consumed-subset",
        "source_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
        "retained_text_layers": 51,
        "text_hidden_state": 50,
        "weights_rehashed": False,
        "source_files_modified": False,
        "omitted": [],
        "changes": {
            "model/text_encoder/config.json": {
                "source_size": 100,
                "size": 80,
                "source_identity": "a" * 64,
                "source_git_blob_sha1": "b" * 40,
                "kind": "text-config-51",
                **change,
            }
        },
    }
    manifest = tmp_path / "derivation.json"
    manifest.write_text(json.dumps(value))
    planned = [("text_encoder/config.json", file, {"size": 100, "git_blob_sha1": "b" * 40})]
    return manifest, planned


def test_explicit_subset_receipt_retains_source_and_no_false_payload_digest(
    tmp_path, monkeypatch
):
    manifest, planned = fixture(tmp_path)
    monkeypatch.setattr(
        "vflash.pipeline.assets._planned_files", lambda assets, profile_id: planned
    )
    assets = PipelineAssets(**{name: tmp_path for name in PipelineAssets.__dataclass_fields__})
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        assert path != planned[0][1], "preparation must not hash or read the weight payload"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ContractError, match="metadata-only"):
        prepare_pipeline_assets(assets, tmp_path / "strict.json", derivation=manifest)
    result = prepare_pipeline_assets(
        assets, tmp_path / "receipt.json", verify_content_hashes=False, derivation=manifest
    )
    assert result.inventory[0]["sha256"] is None
    assert result.inventory[-1]["role"] == "derivation/exact-subset"
    assert len(result.inventory[-1]["sha256"]) == 64
    assert load_prepared_pipeline_assets(result.receipt) == result
    # A receipt cannot silently bind a changed conversion declaration.
    value = json.loads(manifest.read_text())
    value["annotation"] = "changed"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ContractError, match="differs"):
        load_prepared_pipeline_assets(result.receipt)


@pytest.mark.parametrize(
    "change",
    [
        {"source_git_blob_sha1": "c" * 40},
        {"source_size": 101},
        {"size": 101},
        {"kind": "untrusted-conversion"},
    ],
)
def test_subset_cannot_change_pinned_lineage(tmp_path, change):
    with pytest.raises(ContractError, match="source lineage"):
        apply_exact_subset(*fixture(tmp_path, **change))


def test_unlisted_or_unknown_omissions_and_unrequested_derivation_fail(tmp_path):
    manifest, planned = fixture(tmp_path)
    value = json.loads(manifest.read_text())
    value["omitted"] = ["model/text_encoder/model-unknown.safetensors"]
    manifest.write_text(json.dumps(value))
    with pytest.raises(ContractError, match="outside"):
        apply_exact_subset(manifest, planned)
    with pytest.raises(ContractError, match="absolute"):
        apply_exact_subset(Path("relative.json"), planned)
