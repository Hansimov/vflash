"""Execute the candidate's actual two-model session with bounded CPU stand-ins."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vflash.pipeline import VideoRequest
from vflash.pipeline.pika_av import PikaAVRecipe
from vflash.pipeline.pika_reference_av import PikaReferenceAVSession


@pytest.mark.parametrize("steps", [0, 10, 12])
@pytest.mark.parametrize(
    "failure",
    [None, "conditioning", "picture", "sound", "picture_changed", "decode", "restore"],
)
def test_separate_model_ownership_counts_frozen_picture_and_failure_cleanup(
    tmp_path, monkeypatch, steps, failure
):
    from vflash.adapters.h3_reference_audio import _pack_video

    ref_adapter = NS(strength=0, calls={"target": 0})
    fl_adapter = NS(strength=0)
    clock = NS(completed_forwards=0, tensors={})
    clock.select_mode = lambda mode: setattr(clock, "completed_forwards", 0)
    clock.before = lambda *_: None
    clock.after = lambda *_: setattr(clock, "completed_forwards", clock.completed_forwards + 1)

    class Model(torch.nn.Module):
        def __init__(self, partition):
            super().__init__()
            self.partition = partition

        def forward(self):
            assert rt.active_partition == self.partition and pipe.transformer is self
            if self.partition == "ref2va":
                assert ref_adapter.strength == 1 and fl_adapter.strength == 0
                ref_adapter.calls["target"] += 1
                if failure == "picture":
                    raise RuntimeError("fixture picture")
            else:
                assert ref_adapter.strength == 0 and fl_adapter.strength == 1
                if failure == "sound":
                    raise RuntimeError("fixture sound")

    ref, fl = Model("ref2va").eval(), Model("fl2va").eval()

    class Scheduler:
        def __init__(self, **kwargs):
            pass

    class Pipe:
        transformer = fl
        scheduler, audio_scheduler = Scheduler(), Scheduler()
        fixed = None

        def encode_prompt(self, prompt, **kwargs):
            assert not kwargs.get("references") and not kwargs.get("images")
            return torch.zeros(1, 2, 4), torch.ones(2, dtype=torch.long)

        def __call__(self, **kwargs):
            assert kwargs["image"] is None
            self.encode_prompt(kwargs["prompt"])
            rows = _pack_video(self.fixed)[0]
            if failure == "picture_changed":
                rows = rows + 1
            for step in range(kwargs["num_inference_steps"]):
                self.transformer()
                kwargs["callback_on_step_end"](
                    self, step, None, {"latents": rows, "audio_latents": torch.zeros(414, 32)}
                )

    pipe = Pipe()
    attention = NS(backend="sdpa", reset=lambda: None)
    attention.select = lambda b: setattr(attention, "backend", b)

    def condition(*args, **kwargs):
        if failure == "conditioning":
            raise RuntimeError("fixture conditioning")
        return NS(
            conditional_dict={"keyframes": [], "references": [{"kind": "image"}]},
            metadata={"reference_video_rows": 1, "reference_order": ["image"]},
        )

    def sample(model, condition, **kwargs):
        assert condition["keyframes"] == [] and condition["references"]
        for _ in range(6):
            model()
        return torch.zeros(1, 37, 24, 2, 2), torch.zeros(1, 414, 32)

    source = NS(
        source_revision="fixture",
        numerics=NS(FP32_CONDITIONING="fp32", configure_numerics_backend=lambda *a, **kw: None),
        packing=NS(audio_external_to_storage=lambda x: x),
        sampler=NS(sample_ladder=sample),
        wrapper=NS(H3DiffusionWrapper=lambda model: model),
    )
    switched = []

    def select(partition):
        switched.append(partition)
        rt.active_partition = partition
        pipe.transformer = ref if partition == "ref2va" else fl
        return {"partition": partition}

    def suspend():
        rt.active_partition = None
        pipe.transformer = fl
        if failure == "restore":
            raise RuntimeError("fixture restore")

    def decode(*args, **kwargs):
        assert rt.active_partition is None
        if failure == "decode":
            raise RuntimeError("fixture decode")
        return NS(stage_durations={}, media={})

    rt = NS(
        closed=False,
        pipe=pipe,
        reference_model=ref,
        fl_model=fl,
        reference_adapter=ref_adapter,
        adapter=fl_adapter,
        reference_clock=clock,
        clock=NS(select=lambda *a: None, before=lambda *a: None),
        attention=attention,
        sampling=source,
        conditioning_source=None,
        family=NS(pipeline=NS(MiniMaxH3ImageReference=None)),
        select_transformer=select,
        suspend_transformers=suspend,
        decoder=NS(generate_mp4=decode),
        active_partition=None,
    )

    @contextmanager
    def execution(model, **kwargs):
        assert model is (ref if rt.active_partition == "ref2va" else fl)
        yield {}

    @contextmanager
    def audio_context(model, scheduler, video, **kwargs):
        assert model is fl and kwargs["condition_rows"] == 0 and kwargs["spatial_stride"] == 2
        pipe.fixed = video
        yield {}

    monkeypatch.setattr(
        "vflash.adapters.pika_reference_conditioning.reference_conditioning", condition
    )
    monkeypatch.setattr(
        "vflash.adapters.h3_reference_execution.dense_reference_execution", execution
    )
    monkeypatch.setattr(
        "vflash.adapters.h3_reference_audio.native_audio_context", audio_context
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    generator = torch.Generator
    monkeypatch.setattr(torch, "Generator", lambda **kw: generator())
    session = PikaReferenceAVSession(rt, predictor=tmp_path)
    request = VideoRequest(
        "Use <Picture 1> as the subject.", reference=Path("image"), width=32, height=32
    )
    recipe = PikaAVRecipe(audio_steps=steps, visual_attention="sdpa", overlap_decode=False)
    encode = pipe.encode_prompt
    schedulers = pipe.scheduler, pipe.audio_scheduler
    expected_failure = failure is not None and (
        steps > 0 or failure not in {"sound", "picture_changed"}
    )
    if expected_failure:
        with pytest.raises((RuntimeError, ValueError)):
            session.generate(request, tmp_path / "output", recipe=recipe)
    else:
        result = session.generate(request, tmp_path / "output", recipe=recipe)
        assert result["actual_forwards"] == 6 + steps and result["video_preserved_during_audio"]
        assert result["stages"][0]["partition"] == "ref2va"
        assert switched == (["ref2va", "fl2va"] if steps else ["ref2va"])
        if steps:
            assert result["stages"][1]["partition"] == "fl2va"
            assert result["stages"][1]["prompt_reuses"] == 1
    assert pipe.transformer is fl and rt.active_partition is None
    assert pipe.encode_prompt == encode and (pipe.scheduler, pipe.audio_scheduler) == schedulers
    assert ref_adapter.strength == fl_adapter.strength == 0
    assert not session.owner.locked() and not torch.cuda.is_initialized()
    assert not ref._forward_hooks and not ref._forward_pre_hooks
    assert not fl._forward_hooks and not fl._forward_pre_hooks


@pytest.mark.parametrize("fail_upload", [False, True])
def test_real_residency_switch_keeps_only_selected_trunk_copy(monkeypatch, fail_upload):
    from vflash.pipeline.pika_reference_runtime import PikaReferenceAVRuntime
    from vflash.pipeline.residency import capture_cpu_master

    class Model(torch.nn.Linear):
        def to(self, *, device):
            assert device == "cuda:0"
            self.weight = torch.nn.Parameter(self.weight.detach().clone())
            if self is rt.fl_model and fail_upload:
                raise RuntimeError("fixture upload failure")
            return self

    rt = PikaReferenceAVRuntime.__new__(PikaReferenceAVRuntime)
    rt.closed = False
    rt.reference_model, rt.fl_model = Model(2, 2, bias=False), Model(2, 2, bias=False)
    rt.reference_master = capture_cpu_master(rt.reference_model)
    rt.trunk_cpu_master = capture_cpu_master(rt.fl_model)
    rt.active_partition = None
    rt.pipe = NS(transformer=rt.fl_model)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    rt.select_transformer("ref2va")
    assert (
        rt.reference_model.weight.data_ptr()
        != rt.reference_master.state_dict["weight"].data_ptr()
    )
    assert rt.fl_model.weight.data_ptr() == rt.trunk_cpu_master.state_dict["weight"].data_ptr()
    if fail_upload:
        with pytest.raises(RuntimeError, match="upload failure"):
            rt.select_transformer("fl2va")
    else:
        rt.select_transformer("fl2va")
        assert rt.pipe.transformer is rt.fl_model
    assert (
        rt.reference_model.weight.data_ptr()
        == rt.reference_master.state_dict["weight"].data_ptr()
    )
    rt.suspend_transformers()
    assert rt.active_partition is None and rt.pipe.transformer is rt.fl_model
    assert rt.fl_model.weight.data_ptr() == rt.trunk_cpu_master.state_dict["weight"].data_ptr()
    assert not torch.cuda.is_initialized()


def test_close_cannot_remove_an_active_sessions_models():
    from vflash.pipeline.pika_reference_runtime import PikaReferenceAVRuntime

    rt = PikaReferenceAVRuntime.__new__(PikaReferenceAVRuntime)
    rt.session = PikaReferenceAVSession(rt, predictor=Path("fixture"))
    rt.session.owner.acquire()
    try:
        with pytest.raises(ValueError, match="active reference"):
            rt.close()
    finally:
        rt.session.owner.release()
