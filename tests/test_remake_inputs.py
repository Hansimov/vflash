import subprocess
from dataclasses import replace

import pytest
from PIL import Image

from vflash.pipeline.remake import RemakePrefix, RemakeReference, RemakeRequest, remake_graph


@pytest.fixture
def remake_request(tmp_path):
    source, picture = tmp_path / "source.mp4", tmp_path / "target.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=832x480:rate=24",
            "-frames:v",
            "90",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    Image.new("RGB", (832, 480), "green").save(picture)
    return RemakeRequest(
        source,
        "Apply <Picture 1> to <Video 1>.",
        "edit",
        90,
        832,
        480,
        (RemakeReference(picture, 12),),
        "preserve",
    )


def test_reference_pin_and_generated_prefix_have_distinct_condition_roles(
    tmp_path, remake_request
):
    base = remake_graph(remake_request, input_directory=tmp_path, output_prefix="test/edit")
    assert base["11"]["inputs"]["pins"] == "12"
    assert base["11"]["inputs"]["ref_1"] == ["9", 0]
    assert base["11"]["inputs"]["source"] == ["8", 0]
    with pytest.raises(ValueError, match="clean generated history"):
        replace(remake_request, prefix=RemakePrefix(remake_request.source, 0, 17))
    continued = replace(
        remake_request,
        references=(RemakeReference(remake_request.references[0].path),),
        prefix=RemakePrefix(remake_request.source, 73, 17),
    )
    graph = remake_graph(continued, input_directory=tmp_path, output_prefix="test/continuation")
    assert graph["11"]["inputs"]["pins"] == ""
    assert graph["23"]["inputs"]["prefix_at"] == 73
    assert graph["23"]["inputs"]["prefix_frames"] == 17
    assert graph["16"]["inputs"]["latent_image"] == ["23", 0]
    with pytest.raises(ValueError, match="complete history interval"):
        remake_graph(
            replace(continued, prefix=RemakePrefix(remake_request.source, 74, 17)),
            input_directory=tmp_path,
            output_prefix="test/short",
        )


def test_wrong_clock_canvas_and_pin_geometry_fail_before_execution(tmp_path, remake_request):
    for changed in [replace(remake_request, width=864), replace(remake_request, frames=107)]:
        with pytest.raises(ValueError):
            remake_graph(changed, input_directory=tmp_path, output_prefix="test/mismatch")
    Image.new("RGB", (480, 832), "green").save(remake_request.references[0].path)
    with pytest.raises(ValueError, match="pins cannot stretch"):
        remake_graph(remake_request, input_directory=tmp_path, output_prefix="test/stretch")
    with pytest.raises(ValueError, match="not supplied"):
        replace(remake_request, prompt="Use <Picture 2>")
    with pytest.raises(ValueError, match="pins every reference"):
        replace(
            remake_request,
            references=(
                remake_request.references[0],
                RemakeReference(tmp_path / "other.png"),
            ),
        )


def test_silent_graph_preserves_owned_paths(tmp_path, remake_request):
    graph = remake_graph(
        replace(remake_request, soundtrack="silent"),
        input_directory=tmp_path,
        output_prefix="test/silent",
    )
    assert "audio" not in graph["11"]["inputs"]
    assert "audio" not in graph["19"]["inputs"] and "18" not in graph
    with pytest.raises(ValueError, match="relative output prefix"):
        remake_graph(remake_request, input_directory=tmp_path, output_prefix="../another-owner")
    link = tmp_path / "alias.mp4"
    link.symlink_to(remake_request.source)
    with pytest.raises(ValueError, match="regular files"):
        remake_graph(
            replace(remake_request, source=link),
            input_directory=tmp_path,
            output_prefix="test/symlink",
        )


def test_unpinned_reference_requires_prepared_grid_instead_of_implicit_stretch(
    tmp_path,
    remake_request,
):
    picture = remake_request.references[0].path
    Image.new("RGB", (1024, 1536), "green").save(picture)
    appearance = replace(remake_request, references=(RemakeReference(picture),))
    with pytest.raises(ValueError, match="Prepare reference geometry"):
        remake_graph(appearance, input_directory=tmp_path, output_prefix="test/appearance")
    Image.new("RGB", (480, 736), "green").save(picture)
    graph = remake_graph(appearance, input_directory=tmp_path, output_prefix="test/appearance")
    assert graph["11"]["inputs"]["pins"] == ""
    assert graph["11"]["inputs"]["ref_1"] == ["9", 0]


def test_reference_attention_is_an_explicit_recipe_choice(tmp_path, remake_request):
    dense = remake_graph(remake_request, input_directory=tmp_path, output_prefix="test/edit")
    sparse = remake_graph(
        replace(remake_request, attention="veda-r2va"),
        input_directory=tmp_path,
        output_prefix="test/edit",
    )
    assert "30" not in dense
    assert sparse["30"]["inputs"]["predictor"] == "minimax_h3_r2va_veda_preview_fp8.safetensors"
    assert sparse["30"]["inputs"]["generated_sparsity"] == "trained"
    assert sparse["30"]["inputs"]["reference_sparsity"] == "trained"
    assert sparse["13"]["inputs"]["model"] == ["30", 0]
    del sparse["30"]
    sparse["13"]["inputs"]["model"] = ["3", 0]
    assert sparse == dense
    with pytest.raises(ValueError):
        replace(remake_request, attention="implicit")
