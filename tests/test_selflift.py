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
