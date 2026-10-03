import pytest

from vflash.adapters.selflift import consistency_lift

torch = pytest.importorskip("torch")


def test_selection_and_pure_routes():
    direct = torch.zeros(1, 2, 1, 2, 2)
    pixel = torch.tensor([1, 2, 3, 4.0]).view(1, 1, 1, 2, 2).expand_as(direct)
    out, stats = consistency_lift(direct, pixel, rho=0.5)
    assert torch.equal(out, torch.tensor([0, 0, 3, 4.0]).view(1, 1, 1, 2, 2).expand_as(out))
    assert stats["selected_fraction"] == 0.5
    assert torch.equal(consistency_lift(direct, pixel, rho=0)[0], direct)
    assert torch.equal(consistency_lift(direct, pixel, rho=1)[0], pixel)


def test_uniform_and_independent_samples():
    direct = torch.zeros(2, 3, 2, 2, 2)
    pixel = torch.ones_like(direct)
    pixel[1] *= 100
    out, _ = consistency_lift(direct, pixel, rho=0.6, minimum=0.5)
    assert torch.isfinite(out).all()
    assert torch.equal(out, pixel * 0.5)


def test_invalid_geometry_or_strength():
    z = torch.ones(1, 2, 1, 2, 2)
    with pytest.raises(ValueError):
        consistency_lift(z, z, rho=float("nan"))
    with pytest.raises(ValueError):
        consistency_lift(z, z, minimum=2)
    with pytest.raises(ValueError):
        consistency_lift(z, z[0])


def test_temporal_phase_scale_cannot_change_spatial_selection():
    spatial = torch.arange(1, 101).float().view(1, 1, 1, 10, 10)
    phase = torch.tensor([1, 1.1, 1.5, 2, 3] * 3).view(1, 1, 15, 1, 1)
    pixel = (phase * spatial).expand(1, 24, 15, 10, 10)
    direct = torch.zeros_like(pixel)
    out, stats = consistency_lift(direct, pixel, rho=0.6)
    selected = out[:, :1] != 0
    assert torch.equal(selected, selected[:, :, :1].expand_as(selected))
    torch.testing.assert_close(selected.float().mean((-2, -1)), torch.full((1, 1, 15), 0.6))
    assert stats["selection"] == "spatial_per_time"
    assert torch.equal(out[selected.expand_as(out)], pixel[selected.expand_as(out)])


def test_temporal_phase_scale_cannot_change_fractional_strength():
    spatial = torch.arange(1, 101).float().view(1, 1, 1, 10, 10)
    phase = torch.tensor([1, 2, 10]).view(1, 1, 3, 1, 1)
    pixel = (phase * spatial).expand(1, 2, 3, 10, 10)
    out, _ = consistency_lift(torch.zeros_like(pixel), pixel, minimum=0.5, maximum=1)
    weights = out / pixel
    torch.testing.assert_close(weights, weights[:, :, :1].expand_as(weights))
