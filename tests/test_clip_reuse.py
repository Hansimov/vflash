from types import SimpleNamespace

import pytest

from vflash.media.clip_reuse import encode_unique, unique_clips

torch = pytest.importorskip("torch")


class Encoder:
    config = SimpleNamespace(clip_length=17, token_drop=3)

    def __init__(self):
        self.batches = []

    def _encode_clip(self, x):
        self.batches.append(tuple(x.shape))
        # Independent clip posterior moments, preserving the real temporal shape.
        return x[:, :, ::4] * 1.75 + 0.125


def baseline(vae, x):
    x = torch.cat((x, x[:, :, -1:].expand(-1, -1, 12, -1, -1)), dim=2)
    pieces = list(x.split(17, dim=2))
    values = []
    for i in range(0, len(pieces), 2):
        values.extend(vae._encode_clip(torch.cat(pieces[i : i + 2], dim=0)).split(1, dim=0))
    return torch.cat(values, dim=2)[:, :, :-3]


@pytest.mark.parametrize(
    "segment,expected_unique",
    [("first", 4), ("second", 6), ("moving", 8), ("constant", 1), ("odd", 5)],
)
def test_repeat_elision_preserves_complete_posterior_timeline_and_batch_shape(
    segment, expected_unique
):
    gen = torch.Generator().manual_seed(17)
    x = torch.rand((1, 3, 124, 4, 4), generator=gen)
    if segment == "constant":
        x[:] = x[:, :, :1].clone()
    elif segment == "odd":
        x[:, :, 65:] = x[:, :, 64:65].clone()
    elif segment == "first":
        x[:, :, 41:] = x[:, :, 40:41].clone()
    elif segment == "second":
        x[:, :, :57] = x[:, :, 57:58].clone()
        x[:, :, 120:] = x[:, :, 119:120].clone()
    vae = Encoder()
    reference = baseline(vae, x)
    vae.batches = []
    report = {}
    candidate = encode_unique(vae, x, report)
    assert torch.equal(reference, candidate)
    assert report["unique_clips"] == expected_unique
    assert len(vae.batches) == (expected_unique + 1) // 2
    assert all(shape[:3] == (2, 3, 17) for shape in vae.batches)


def test_exact_reuse_rejects_one_changed_value_and_retains_stable_order():
    a = torch.zeros(1, 3, 17, 2, 2)
    b = a.clone()
    b[0, 2, 16, 1, 1] = 1
    values, mapping = unique_clips([a, a.clone(), b, b.clone(), a], equal=torch.equal)
    assert len(values) == 2 and mapping == [0, 0, 1, 1, 0]


def test_unqualified_timeline_is_rejected():
    with pytest.raises(ValueError, match="five-second"):
        encode_unique(Encoder(), torch.zeros(1, 3, 244, 4, 4), {})
