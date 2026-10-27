from pathlib import Path

import pytest

from vflash.compiler.continuation import compile_native_history_tables


@pytest.mark.parametrize(
    "blocks,reference",
    [
        ((25, 25), Path("ref")),
        ((49, 25), Path("ref")),
        ((50,), Path("ref")),
        ((True,), Path("ref")),
        ((25,), None),
        ((), Path("ref")),
    ],
)
def test_invalid_model_selection_fails_before_checkpoint_io(blocks, reference):
    with pytest.raises(ValueError, match="hybrid blocks"):
        compile_native_history_tables(
            Path("missing"), reference_checkpoint=reference, reference_blocks=blocks
        )


def test_compiler_uses_selected_blocks_and_keeps_base_time_and_head(monkeypatch):
    torch = pytest.importorskip("torch")
    from vflash.adapters import checkpoints
    from vflash.compiler import math, refinement

    loads, reports = [], []

    class Checkpoint:
        def __init__(self, path):
            self.kind = Path(path).name

        def load(self, name):
            loads.append((self.kind, name))
            return torch.tensor(1.0 if self.kind == "base" else 2.0)

        def load_many(self, names):
            return {name: self.load(name) for name in names}

    def time(load, rows):
        assert rows[0].device.type == "cpu" and rows[0].item() == 1.0
        return load("time")

    def adaln(timing, counts, weight, bias, *, modalities, modulations):
        assert timing.item() == 1 and counts == (1,)
        return torch.full((1, modalities, modulations, 5376), weight.item())

    monkeypatch.setattr(checkpoints, "IndexedCheckpoint", Checkpoint)
    monkeypatch.setattr(refinement, "time_embeddings", time)
    monkeypatch.setattr(math, "adaln_table", adaln)
    tables = compile_native_history_tables(
        Path("base"),
        reference_checkpoint=Path("ref"),
        reference_blocks=tuple(range(25, 50)),
        progress=lambda *p: reports.append(p),
    )
    assert len(tables) == 51 and len(reports) == 51
    assert all(t[0, 0, 0, 0] == 1 for t in tables[:25])
    assert all(t[0, 0, 0, 0] == 2 for t in tables[25:50])
    assert tables[-1][0, 0, 0, 0] == 1
    assert loads[0] == ("base", "time")
    assert loads[-1] == ("base", "norm_out.linear.bias")
