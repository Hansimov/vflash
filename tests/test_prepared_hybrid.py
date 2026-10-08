"""CPU preparation and asset integrity without source weights or CUDA."""

import json
from types import SimpleNamespace

import pytest

from vflash.native import h3_prepared_hybrid as cache


def test_cpu_preparation_roundtrip_binding_and_tamper(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    table = torch.zeros((4, 9, 6, 5376), dtype=torch.bfloat16)
    artifact = SimpleNamespace(
        artifact_id="fixture", weight_profile="lightx-turbo4-v0.1-544", nfe=4
    )
    overlay = SimpleNamespace(overlay_id="fixture-overlay")
    provenance = {"fixture": True}
    calls = []

    def compile_tables(a, o, c, *, device):
        calls.append(device)
        return SimpleNamespace(tables={i: table for i in range(25, 50)}, provenance=provenance)

    monkeypatch.setattr("vflash.native.h3_hybrid.compile_hybrid_modulation", compile_tables)
    directory = tmp_path / "prepared"
    loaded = cache.prepare(directory, artifact, overlay, object())
    assert calls == ["cpu"] and not torch.cuda.is_initialized()
    assert sorted(loaded.tables) == list(range(25, 50)) and loaded.provenance == provenance
    with pytest.raises(ValueError, match="exists"):
        cache.prepare(directory, artifact, overlay, object())
    with pytest.raises(ValueError, match="different backbone"):
        cache.load(directory, SimpleNamespace(artifact_id="other"), overlay)
    manifest = directory / "hybrid-cache.json"
    data = json.loads(manifest.read_text())
    data["tables"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        cache.inspect(directory)
