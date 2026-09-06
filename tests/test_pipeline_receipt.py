from __future__ import annotations

import hashlib
import json

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.assets import _stamp, load_prepared_pipeline_assets
from vflash.pipeline.contracts import PIPELINE_PROFILE, PipelineAssets


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    asset = tmp_path / "weight.bin"
    asset.write_bytes(b"immutable fixture weights")
    expected = {
        "size": asset.stat().st_size,
        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(
        "vflash.pipeline.assets._planned_files",
        lambda assets, profile_id: [("test-weight", asset, expected)],
    )
    assets = PipelineAssets(**{name: tmp_path for name in PipelineAssets.__dataclass_fields__})
    value = {
        "schema_version": 1,
        "profile_id": PIPELINE_PROFILE,
        "assets": assets.to_mapping(),
        "inventory": [
            {
                "role": "test-weight",
                "path": str(asset),
                "sha256": expected["sha256"],
                "stamp": _stamp(asset),
            }
        ],
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(value))
    return path, asset, value


def test_prepared_asset_replacement_invalidates_receipt(receipt):
    path, asset, _ = receipt
    prepared = load_prepared_pipeline_assets(path)
    prepared.check_unchanged()
    old = asset.with_suffix(".old")
    asset.rename(old)
    asset.write_bytes(old.read_bytes())
    with pytest.raises(ContractError, match="changed"):
        prepared.check_unchanged()


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "wrong-path", "wrong-hash", "bad-stamp"]
)
def test_receipt_binds_all_declared_assets(receipt, mutation):
    path, asset, value = receipt
    row = value["inventory"][0]
    if mutation == "duplicate":
        value["inventory"].append(dict(row))
    elif mutation == "missing":
        value["inventory"] = []
    elif mutation == "wrong-path":
        row["path"] = str(asset.with_suffix(".elsewhere"))
    elif mutation == "wrong-hash":
        row["sha256"] = "0" * 64
    else:
        row["stamp"]["size"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ContractError):
        load_prepared_pipeline_assets(path)


def test_reference_bytes_and_orientation_are_bound_together(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from vflash.adapters.references import read_reference

    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (64, 32), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    decoded = read_reference(source)
    try:
        assert decoded.image.size == (32, 64)
        assert decoded.image.mode == "RGB"
        assert decoded.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    finally:
        decoded.close()
