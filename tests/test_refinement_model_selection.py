import json
from pathlib import Path

import pytest

from vflash.compiler.refinement import compile_refinement_tables
from vflash.native.h3_refinement_schedule import load_refinement_history_tables

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("hybrid", [False, True])
def test_refinement_and_history_use_the_same_explicit_weight_recipe(
    tmp_path, monkeypatch, hybrid
):
    from safetensors import torch as tensor_files

    from vflash.adapters import checkpoints
    from vflash.compiler import math, refinement
    from vflash.native import h3_hybrid

    opened, saved = [], {}

    class Checkpoint:
        def __init__(self, path):
            self.kind = Path(path).name
            opened.append(self.kind)

        def load(self, name):
            return torch.tensor(1.0 if self.kind == "base" else 2.0)

        def load_many(self, names):
            return {name: self.load(name) for name in names}

    def adaln(timing, counts, weight, bias, *, modalities, modulations):
        assert timing.item() == 1
        return weight.to(torch.bfloat16).expand(
            len(counts), max(counts) * modalities, modulations, 5376
        )

    def save(values, path):
        saved[Path(path).name] = {k: float(v.flatten()[0]) for k, v in values.items()}
        Path(path).write_bytes(b"test owned inventory")

    monkeypatch.setattr(checkpoints, "IndexedCheckpoint", Checkpoint)
    monkeypatch.setattr(h3_hybrid, "validate_hybrid_reference", lambda _: None)
    monkeypatch.setattr(refinement, "time_embeddings", lambda load, rows: load("time"))
    monkeypatch.setattr(math, "adaln_table", adaln)
    monkeypatch.setattr(tensor_files, "save_file", save)
    manifest = compile_refinement_tables(
        Path("base"),
        Path("ref") if hybrid else None,
        tmp_path / "tables",
        steps=6,
        denoise=0.5,
        mode="audio",
        include_native_history=True,
    )
    for i in range(50):
        expected = 2.0 if hybrid and i >= 25 else 1.0
        assert saved[f"block-{i:03}.safetensors"]["adaln.table"] == expected
        assert saved["native-history.safetensors"][f"block-{i:03}"] == expected
    assert saved["native-history.safetensors"]["final"] == 1.0
    assert ("ref" in opened) == hybrid
    assert manifest["native_history"]["source"] == manifest["source"]
    assert len(manifest["files"]) == 51


def test_history_loader_rejects_a_different_recipe_before_reading_tensors(tmp_path):
    manifest = {
        "schema": 1,
        "method": "hybrid-low-noise-official-modulation-v1",
        "source": "FL",
        "native_history": {"file": "native-history.safetensors", "size": 1, "source": "hybrid"},
    }
    (tmp_path / "refinement.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="matching native history"):
        load_refinement_history_tables(tmp_path)
