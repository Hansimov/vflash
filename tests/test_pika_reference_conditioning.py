from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vflash.adapters.pika_reference_conditioning import reference_conditioning
from vflash.contracts import ContractError
from vflash.pipeline.contracts import VideoRequest


@pytest.fixture
def fixture(tmp_path):
    paths = []
    for n, color in enumerate(["red", "blue"]):
        path = tmp_path / f"{n}.png"
        Image.new("RGB", (48, 32), color).save(path)
        paths.append(path)
    request = VideoRequest(
        "<Picture 2> follows <Picture 1>.", references=tuple(paths), width=1280, height=736
    )
    calls = []

    class Pipe:
        vae = SimpleNamespace(dtype=torch.float32)

        def encode_prompt(self, prompt, *, references, device, dtype):
            calls.append(("text", prompt, [r.image.getpixel((0, 0)) for r in references]))
            assert dtype == torch.bfloat16
            return torch.zeros(1, 3, 4, dtype=dtype), torch.tensor([0, 0, 1])

        def encode_reference_latents(self, refs, *, device):
            calls.append(("latents", len(refs)))
            return [torch.full((1, 24, 1, 2, 4), i + 0.12345) for i in range(len(refs))], []

    def fit(image, geometry, rule):
        assert rule == "tiered" and geometry.width == 1280
        return image.resize((64, 32)), {
            "source": [image.height, image.width],
            "encoded": [32, 64],
        }

    def prepare(cond, device, *, policy):
        assert policy == "fp32_conditioning_v1" and device == "cpu"
        return cond

    conditioning = SimpleNamespace(
        Request=SimpleNamespace, resolve_geometry=lambda r: r, fit_reference_image=fit
    )
    source = SimpleNamespace(
        packing=SimpleNamespace(video_storage_to_external=lambda t: t.permute(1, 0, 2, 3)),
        numerics=SimpleNamespace(
            prepare_conditioning=prepare, FP32_CONDITIONING="fp32_conditioning_v1"
        ),
    )
    return (
        request,
        Pipe(),
        dict(
            source=source,
            conditioning=conditioning,
            reference_type=SimpleNamespace,
            device="cpu",
        ),
        calls,
    )


def test_order_roles_precision_and_geometry_survive_preparation(fixture):
    request, pipe, options, calls = fixture
    result = reference_conditioning(pipe, request, **options)
    cond = result.conditional_dict
    assert calls == [("text", request.prompt, [(255, 0, 0), (0, 0, 255)]), ("latents", 2)]
    assert cond["keyframes"] == [] and len(cond["references"]) == 2
    assert cond["references"][0]["video_latents"].shape == (1, 1, 24, 2, 4)
    assert cond["references"][1]["video_latents"].dtype == torch.float32
    assert cond["references"][1]["video_latents"].flatten()[0] == torch.tensor(1.12345)
    assert result.metadata["reference_video_rows"] == 4
    assert result.metadata["temporal_keyframes"] == 0


def test_keyframes_and_lower_precision_cannot_silently_enter_ref_runtime(fixture):
    request, pipe, options, calls = fixture
    with pytest.raises(ContractError, match="ordered images"):
        reference_conditioning(
            pipe, VideoRequest("First frame", first_frame=Path("x")), **options
        )
    pipe.vae.dtype = torch.bfloat16
    with pytest.raises(ContractError, match="FP32"):
        reference_conditioning(pipe, request, **options)
    assert calls == []


def test_encoder_cannot_return_wrong_reference_count_or_rounded_latents(fixture):
    request, pipe, options, _ = fixture
    pipe.encode_reference_latents = lambda *a, **kw: ([], [])
    with pytest.raises(ContractError, match="ordered modalities"):
        reference_conditioning(pipe, request, **options)
    pipe.encode_reference_latents = lambda *a, **kw: (
        [torch.zeros(1, 24, 1, 2, 4, dtype=torch.bfloat16)] * 2,
        [],
    )
    with pytest.raises(ContractError, match="precision"):
        reference_conditioning(pipe, request, **options)
