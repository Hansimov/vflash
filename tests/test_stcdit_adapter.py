from types import SimpleNamespace

import pytest

from vflash.adapters.stcdit import StcditTinyRestorer, validate_segments


@pytest.mark.parametrize(
    "segments",
    [[], [(1, 3)], [(0, 2)], [(0, 2), (1, 3)], [(0, 1), (2, 3)], [(0, 4)], [(False, 3)]],
)
def test_rejects_incomplete_or_ambiguous_segmentation(segments):
    with pytest.raises(ValueError):
        validate_segments(segments, 3)


def test_explicit_recipe_preserves_caller_images_and_owns_results():
    from PIL import Image

    frames = [Image.new("RGB", (32, 64), (n, 20, 30)) for n in range(3)]
    before = [frame.tobytes() for frame in frames]
    observed = {}

    def generate(**kwargs):
        observed.update(kwargs)
        kwargs["input_lq"][0].putpixel((0, 0), (255, 0, 0))
        return kwargs["input_lq"]

    result = StcditTinyRestorer(SimpleNamespace(test_tlc_seg=generate)).restore(
        frames, caption="A moving geometric scene.", segments=[(0, 1), (1, 3)], seed=8
    )
    assert before == [frame.tobytes() for frame in frames]
    assert result.frames[0].getpixel((0, 0)) == (255, 0, 0)
    assert all(a is not b for a, b in zip(result.frames, frames, strict=True))
    assert result.frame_count == 3 and result.fps == 24 and result.seed == 8
    assert observed["num_inference_steps"] == 10 and observed["cfg_scale"] == 1
    assert observed["sigma_shift"] == 5 and observed["tile_kernel"] == (21, 8, 4)


@pytest.mark.parametrize("failure", ["count", "canvas", "mode", "exception"])
def test_bad_backend_output_never_overwrites_source(failure):
    from PIL import Image

    source = Image.new("RGB", (32, 32), "blue")
    original = source.tobytes()

    def generate(**_kwargs):
        if failure == "exception":
            raise RuntimeError("backend failed")
        return {
            "count": [],
            "canvas": [Image.new("RGB", (64, 64))],
            "mode": [Image.new("L", (32, 32))],
        }[failure]

    with pytest.raises((ValueError, RuntimeError)):
        StcditTinyRestorer(SimpleNamespace(test_tlc_seg=generate)).restore(
            [source], caption="A blue scene.", segments=[(0, 1)]
        )
    assert source.tobytes() == original


@pytest.mark.parametrize("options", [{"caption": " "}, {"seed": True}, {"seed": -1}])
def test_invalid_request_fails_before_backend(options):
    from PIL import Image

    def generate(**_kwargs):
        pytest.fail("invalid input reached the expensive backend")

    with pytest.raises(ValueError):
        StcditTinyRestorer(SimpleNamespace(test_tlc_seg=generate)).restore(
            [Image.new("RGB", (32, 32))],
            segments=[(0, 1)],
            **({"caption": "A scene.", "seed": 42} | options),
        )


@pytest.mark.parametrize(
    "size,mode", [((33, 32), "RGB"), ((1056, 1024), "RGB"), ((32, 32), "L")]
)
def test_bad_canvas_never_reaches_runtime(size, mode):
    from PIL import Image

    def generate(**_kwargs):
        pytest.fail("invalid geometry reached the backend")

    with pytest.raises(ValueError):
        StcditTinyRestorer(SimpleNamespace(test_tlc_seg=generate)).restore(
            [Image.new(mode, size)], caption="A scene.", segments=[(0, 1)]
        )


def test_local_runtime_requires_explicit_source_trust_before_importing_provider(tmp_path):
    from vflash.adapters.stcdit_runtime import LocalStcditTiny

    with pytest.raises(ValueError, match="trust_local_code"):
        LocalStcditTiny(source=tmp_path, weights=tmp_path)


def test_local_runtime_rejects_incomplete_inventory_before_model_loading(tmp_path):
    from vflash.adapters.stcdit_runtime import validate_local_assets

    with pytest.raises(ValueError, match="wrong-sized"):
        validate_local_assets(tmp_path, tmp_path)


def test_residency_is_explicit_and_only_moves_each_model_once(tmp_path):
    from vflash.adapters.stcdit_runtime import LocalStcditTiny

    with pytest.raises(ValueError, match="memory_policy"):
        LocalStcditTiny(
            source=tmp_path, weights=tmp_path, trust_local_code=True, memory_policy="guess"
        )
    calls = []
    runtime = object.__new__(LocalStcditTiny)
    runtime.pipeline = SimpleNamespace(
        vae=SimpleNamespace(to=lambda device: calls.append(("vae", device))),
        dit=SimpleNamespace(to=lambda device: calls.append(("dit", device))),
        absent=None,
    )
    runtime.device = "cuda:0"
    runtime._resident_models = set()
    runtime._load_retained_models(["vae"])
    runtime._load_retained_models(["dit", "vae", "absent"])
    runtime._load_retained_models([])
    assert calls == [("vae", "cuda:0"), ("dit", "cuda:0")]


def test_cancellation_between_steps_leaves_original_images_usable():
    from PIL import Image

    def generate(**kwargs):
        for _ in kwargs["progress_bar_cmd"](range(10)):
            pass
        pytest.fail("cancelled generation continued")

    def cancelled(stage, completed, _total):
        if stage == "denoising" and completed == 1:
            raise RuntimeError("cancelled")

    source = Image.new("RGB", (32, 32), "red")
    with pytest.raises(RuntimeError, match="cancelled"):
        StcditTinyRestorer(SimpleNamespace(test_tlc_seg=generate)).restore(
            [source], caption="A scene.", segments=[(0, 1)], on_progress=cancelled
        )
    assert source.getpixel((0, 0)) == (255, 0, 0)


def test_motion_segmentation_never_truncates_blank_or_low_feature_input():
    pytest.importorskip("cv2")
    from PIL import Image

    from vflash.adapters.restoration_motion import restoration_segments

    frames = [Image.new("RGB", (32, 32), "black") for _ in range(25)]
    segments = restoration_segments(frames)
    assert validate_segments(segments, 25) == segments
    assert all(end - start <= 9 for start, end in segments)
