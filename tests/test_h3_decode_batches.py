from types import MethodType

import pytest

from vflash.adapters.h3_decode_batches import batch_spatial_tiles

torch = pytest.importorskip("torch")


class Decoder:
    training = False

    def _run_tile_tasks(self, tiles, indices, forward, stacked, cls_agg=None):
        if not stacked:
            return [forward(tiles[i]) for i in indices]
        output = forward(torch.cat([tiles[i] for i in indices]))
        return list(output.split(tiles[0].shape[0]))


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_batch_preserves_original_sample_and_tile_order(limit):
    decoder = Decoder()
    tiles = [torch.arange(24).reshape(2, 3, 4) + i for i in range(7)]
    indices = [6, 2, 1, 5, 0]
    seen = []

    def forward(x):
        seen.append(x.shape[0])
        return x * 2 + 1

    expected = decoder._run_tile_tasks(tiles, indices, forward, False)
    seen.clear()
    with batch_spatial_tiles(decoder, limit) as report:
        actual = decoder._run_tile_tasks(tiles, indices, forward, False)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual, strict=True))
    assert max(seen) <= limit * 2
    assert "_run_tile_tasks" not in vars(decoder)
    if limit > 1:
        assert sum(report["groups"]) == len(indices)


def test_unequal_edge_tiles_do_not_share_a_batch():
    decoder = Decoder()
    tiles = [torch.ones(2, 3, n) for n in (4, 2, 4)]
    with batch_spatial_tiles(decoder, 4) as report:
        actual = decoder._run_tile_tasks(tiles, [0, 1, 2], lambda x: x + 1, False)
    assert [x.shape for x in actual] == [x.shape for x in tiles]
    assert report["groups"] == [1, 1, 1]


def test_failed_or_nested_owners_restore_the_original_instance_override():
    decoder = Decoder()
    original = MethodType(Decoder._run_tile_tasks, decoder)
    decoder._run_tile_tasks = original
    with (
        pytest.raises(RuntimeError, match="decode failed"),
        batch_spatial_tiles(decoder, 4),
    ):
        with (
            pytest.raises(ValueError, match="exclusively owned"),
            batch_spatial_tiles(decoder, 2),
        ):
            pass
        raise RuntimeError("decode failed")
    assert decoder._run_tile_tasks is original
    assert not hasattr(decoder, "_vflash_tile_batch_owner")
    decoder.training = True
    with (
        pytest.raises(ValueError, match="inference decoder"),
        batch_spatial_tiles(decoder, 4),
    ):
        pass
