from types import SimpleNamespace

import pytest

from vflash.adapters.h3_tile_composition import compose_tiles, install_spatial_composition

torch = pytest.importorskip("torch")


def blend(a, b, overlap, dim):
    shape = [1] * b.ndim
    shape[dim] = overlap
    ramp = (torch.arange(overlap, dtype=b.dtype) / overlap).view(shape)
    edge = a.narrow(dim, a.shape[dim] - overlap, overlap) * (1 - ramp)
    edge = edge + b.narrow(dim, 0, overlap) * ramp
    return torch.cat((edge, b.narrow(dim, overlap, b.shape[dim] - overlap)), dim=dim)


def test_diagonal_contributor_and_no_mutation():
    rows = [
        [torch.full((1, 3, 2, 4, 4), float(i * 2 + j + 1)) for j in range(2)] for i in range(2)
    ]
    before = [[t.clone() for t in row] for row in rows]
    out = compose_tiles(rows, [2], [2], blend)
    assert out.shape == (1, 3, 2, 6, 6)
    assert torch.all(out[..., 2, 2] == 1)
    assert torch.all(out[..., 3, 3] == 2.5)
    for a, b in zip(rows, before, strict=True):
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)


def test_aligned_ramp_survives_triple_overlap():
    full = torch.arange(8 * 9, dtype=torch.float64).reshape(1, 1, 1, 8, 9)
    rows = [[full[..., y : y + 5, x : x + 5] for x in (0, 2, 4)] for y in (0, 1, 3)]
    out = compose_tiles(rows, [4, 3], [3, 3], blend)
    torch.testing.assert_close(out, full, rtol=0, atol=1e-12)


def test_single_tile_identity_and_single_axis():
    tile = torch.rand(2, 3, 4, 6, 6)
    assert compose_tiles([[tile]], [], [], blend) is tile
    for rows, yo, xo in (([[tile, tile]], [], [3]), ([[tile], [tile]], [3], [])):
        result = compose_tiles(rows, yo, xo, blend)
        dim = -1 if xo else -2
        expected = torch.cat((tile.narrow(dim, 0, 3), blend(tile, tile, 3, dim)), dim)
        assert torch.equal(result, expected)


def test_invalid_geometry_rejected():
    tile = torch.zeros(1, 3, 1, 4, 4)
    with pytest.raises(ValueError):
        compose_tiles([[tile, tile]], [], [], blend)
    with pytest.raises(ValueError):
        compose_tiles([[tile, tile]], [], [4], blend)


def get_parallel_state():
    return dict(sp_size=1, sp_rank=0)


class Decoder:
    training = False
    parallel_tiling = True
    stack_tiling = True
    vae_ratio = 2
    blend = staticmethod(blend)

    def __init__(self):
        self.calls = []

    def split_tiles(self, length, is_decoder):
        assert length == 6 and is_decoder
        return [0, 2], [4, 4], [2]

    def _local_tile_indices(self, count, rank, size):
        assert get_parallel_state() == dict(sp_size=1, sp_rank=0)
        assert rank == 0 and size == 1
        return list(range(count))

    def _run_tile_tasks(self, tiles, indices, decode, stack):
        assert stack
        return [decode(tiles[i]) for i in indices]

    def decode(self, latent):
        self.calls.append(latent.clone())
        return torch.full((1, 3, 2, 4, 4), float(len(self.calls)))


def test_install_is_instance_local_repeatable_and_preserves_tile_inputs():
    model, other = Decoder(), Decoder()
    latent = torch.arange(18).reshape(1, 1, 2, 3, 3).float()
    install_spatial_composition(model)
    install_spatial_composition(model)
    assert not hasattr(other, "tiled_decode")
    out = model.tiled_decode(latent)
    assert torch.all(out[..., 2, 2] == 1)
    expected = [latent[..., y : y + 2, x : x + 2] for y in (0, 1) for x in (0, 1)]
    assert len(model.calls) == len(expected)
    assert all(torch.equal(a, b) for a, b in zip(model.calls, expected, strict=True))


def test_training_not_silently_changed():
    model = SimpleNamespace(training=True)
    install_spatial_composition(model)
    with pytest.raises(ValueError, match="inference"):
        model.tiled_decode(torch.zeros(1))
