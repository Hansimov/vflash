from types import SimpleNamespace

import pytest

from vflash.native.h3_hybrid import HybridModulationOverlay


def fixture():
    calls = []
    base = SimpleNamespace(
        overlay_id="base",
        blocks=tuple(range(50)),
        schedule="FL clock",
        load_block_table=lambda index: calls.append(index) or ("FL", index),
        load_auxiliary_tensors=lambda: {"time": "FL", "final": "FL"},
    )
    artifact = SimpleNamespace(blocks=base.blocks)
    tables = {i: ("Ref", i) for i in range(25, 50)}
    return HybridModulationOverlay(base, artifact, tables, {"experimental": True}), calls


def test_exact_selected_blocks_and_unchanged_auxiliary():
    overlay, calls = fixture()
    assert overlay.schedule == "FL clock"
    assert overlay.load_auxiliary_tensors() == {"time": "FL", "final": "FL"}
    for i in range(50):
        assert overlay.load_block_table(i) == ("FL" if i < 25 else "Ref", i)
    assert calls == list(range(25))
    assert "hybrid-ref" in overlay.overlay_id
    assert overlay.base.overlay_id == "base"


@pytest.mark.parametrize("index", [-1, 50, True, 1.5])
def test_invalid_block_does_not_reach_source(index):
    overlay, calls = fixture()
    with pytest.raises(ValueError):
        overlay.load_block_table(index)
    assert not calls
