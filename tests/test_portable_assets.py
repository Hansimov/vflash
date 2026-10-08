"""Portable metadata never hardlinks or changes an ingested source."""

import json
from types import SimpleNamespace

import pytest

from vflash.model_assets import weights_source
from vflash.pipeline import portable
from vflash.pipeline.contracts import PipelineAssets


def test_portable_view_preserves_source_stamps_and_records_lineage(tmp_path, monkeypatch):
    artifact = tmp_path / "source-artifact"
    (artifact / "blocks").mkdir(parents=True)
    block = artifact / "blocks/block-000.safetensors"
    block.write_bytes(b"fixture-block")
    schedule = tmp_path / "source-schedule"
    schedule.mkdir()
    auxiliary = schedule / "schedule.safetensors"
    auxiliary.write_bytes(b"fixture-timing")
    source = weights_source("i2va-turbo4-v01-544-exact-sm89")
    raw = {"artifact_id": "source", "target": {"target_id": "sm89-test"}, "source": source}
    (artifact / "artifact.json").write_text(json.dumps(raw))
    (schedule / "overlay.json").write_text(
        json.dumps(
            {
                "overlay_id": "original",
                "blocks": [],
                "auxiliary": {"path": "schedule.safetensors"},
                "base_artifact": {},
                "source": {},
            }
        )
    )
    original = SimpleNamespace(
        artifact_id="source",
        source=source,
        nfe=4,
        weight_profile="lightx-turbo4-v0.1-544",
        target=SimpleNamespace(target_id="sm89-test"),
    )
    monkeypatch.setattr(portable, "load_h3_runtime_artifact", lambda *a, **k: original)
    monkeypatch.setattr(portable, "load_h3_schedule_overlay", lambda *a, **k: object())
    observed = []

    def prepare(assets, receipt, **kwargs):
        observed.append((assets, receipt, kwargs))
        return assets

    monkeypatch.setattr(portable, "prepare_pipeline_assets", prepare)
    values = {name: tmp_path for name in PipelineAssets.__dataclass_fields__}
    values.update(artifact=artifact, schedule_overlay=schedule)
    before = [
        (p.stat().st_ino, p.stat().st_ctime_ns, p.stat().st_nlink) for p in [block, auxiliary]
    ]
    result = portable.prepare_portable_assets(
        PipelineAssets(**values), tmp_path / "target", "sm120"
    )
    assert (result.artifact / "blocks").is_symlink()
    assert (result.artifact / "blocks/block-000.safetensors").read_bytes() == b"fixture-block"
    assert [
        (p.stat().st_ino, p.stat().st_ctime_ns, p.stat().st_nlink) for p in [block, auxiliary]
    ] == before
    converted = json.loads((result.artifact / "artifact.json").read_text())
    assert converted["portable_from"]["source"] == source and converted["schema_version"] == 7
    assert converted["target"]["compute_capability"] == "sm120"
    assert observed[0][2] == {
        "profile_id": "i2va-turbo4-v01-544-exact-sm120",
        "verify_content_hashes": False,
    }
    with pytest.raises(FileExistsError):
        portable.prepare_portable_assets(PipelineAssets(**values), tmp_path / "target", "sm120")
    with pytest.raises(ValueError, match="unsupported"):
        portable.prepare_portable_assets(PipelineAssets(**values), tmp_path / "other", "sm86")
