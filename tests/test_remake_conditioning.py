import pytest

from vflash.pipeline.remake_conditioning import preserved_video_latent

torch = pytest.importorskip("torch")


def test_source_regions_keep_original_pixels_in_latent_and_prior_generated_history():
    video = torch.zeros(1, 24, 7, 4, 4)
    source = torch.full_like(video, 3)
    noise = torch.ones_like(video)
    video[:, :, :2] = 7
    noise[:, :, :2] = 0
    edit_mask = torch.ones(1, 64, 64, 3)
    edit_mask[:, :32] = 0
    result, mask = preserved_video_latent(video, noise, source, edit_mask)
    assert torch.equal(result[:, :, :2], video[:, :, :2])
    assert torch.equal(result[:, :, 2:, :2], source[:, :, 2:, :2])
    assert torch.count_nonzero(result[:, :, 2:, 2:]) == 0
    assert torch.count_nonzero(mask[:, :, :2]) == 0
    assert torch.count_nonzero(mask[:, :, :, :2]) == 0
    assert torch.all(mask[:, :, 2:, 2:] == 1)
    assert torch.all(video[:, :, 2:] == 0)  # Inputs remain reusable for another arm.


def test_fully_editable_mask_keeps_original_execution_latent_and_mask():
    video = torch.randn(1, 24, 7, 4, 4)
    noise = torch.ones_like(video)
    result, mask = preserved_video_latent(
        video,
        noise,
        torch.randn_like(video),
        torch.ones(1, 64, 64, 3),
    )
    assert torch.equal(result, video)
    assert torch.equal(mask, noise)


def test_source_mask_refuses_unaligned_or_unbounded_pixels():
    video = torch.zeros(1, 24, 7, 4, 4)
    for mask in [
        torch.zeros(2, 64, 64, 3),
        torch.zeros(1, 32, 64, 3),
        torch.full((1, 64, 64, 3), float("nan")),
        torch.ones(1, 64, 64, 3) * 2,
    ]:
        with pytest.raises(ValueError, match="static source mask"):
            preserved_video_latent(video, torch.ones_like(video), video, mask)
