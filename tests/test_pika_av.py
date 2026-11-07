from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

from vflash.pipeline.pika_av import PikaAVRecipe


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audio_steps": 6},
        {"audio_steps": True},
        {"visual_attention": "sage"},
        {"overlap_decode": 1},
        {"decode_tile_batch_size": True},
        {"decode_tile_batch_size": 3},
        {"strict_rotary": 1},
    ],
)
def test_recipe_rejects_implicit_or_unqualified_stage_choices(kwargs):
    with pytest.raises(ValueError):
        PikaAVRecipe(**kwargs)


@pytest.mark.parametrize("mode", ["t2va", "i2va"])
@pytest.mark.parametrize("steps", [0, 10, 12])
@pytest.mark.parametrize("fail_audio", [False, True])
@pytest.mark.parametrize("tile_batch", [1, 4])
@pytest.mark.parametrize("staged_text", [False, True])
def test_actual_session_counts_freezes_picture_and_restores_owners(
    tmp_path,
    monkeypatch,
    mode,
    steps,
    fail_audio,
    tile_batch,
    staged_text,
    qad=False,
    duration=5,
):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from vflash.adapters.h3_reference_assets import CachedProjection
    from vflash.adapters.h3_reference_audio import _pack_video
    from vflash.pipeline import VideoRequest
    from vflash.pipeline.pika_av import PikaAVSession

    active = []
    qad_active = []
    adapter = NS(strength=0, calls={"projection": 0})
    clock = NS(completed_forwards=0, time_embedding=lambda x: x)
    clock.select_mode = lambda mode: setattr(clock, "completed_forwards", 0)
    clock.before = lambda *_: None
    clock.after = lambda *_: setattr(clock, "completed_forwards", clock.completed_forwards + 1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Identity()
            self.layers = torch.nn.ModuleList()
            for _ in range(51):
                layer = CachedProjection.__new__(CachedProjection)
                torch.nn.Module.__init__(layer)
                layer.clock = "original"
                self.layers.append(layer)

        def forward(self):
            if adapter.strength or qad_active:
                assert active == [not qad]
                if adapter.strength:
                    adapter.calls["projection"] += 1
            else:
                assert active == [False]
                if fail_audio:
                    raise RuntimeError("fixture audio failure")

    class Scheduler:
        def __init__(self, **_):
            pass

    model = Model().eval()

    class Pipe:
        transformer = model
        scheduler = Scheduler()
        audio_scheduler = Scheduler()
        fixed = None

        def encode_prompt(self, *_args, **_kwargs):
            return torch.zeros(1, 3, 5120), torch.ones(3, dtype=torch.long)

        def encode_keyframes(self, *_args, **_kwargs):
            return torch.zeros(1, 96)

        def __call__(self, **kwargs):
            assert kwargs["num_frames"] == (107 if duration == 4 else 124)
            self.encode_prompt(kwargs["prompt"])
            rows = _pack_video(self.fixed)[0]
            if kwargs["image"] is not None:
                self.encode_keyframes([kwargs["image"]])
                rows = torch.cat([torch.zeros(1, 96), rows])
            for i in range(kwargs["num_inference_steps"]):
                model()
                kwargs["callback_on_step_end"](
                    self, i, None, {"latents": rows, "audio_latents": torch.zeros(414, 32)}
                )

    pipe = Pipe()
    from vflash.pipeline.residency import capture_cpu_master

    class TextEncoder(torch.nn.Linear):
        def to(self, *, device):
            assert device == "cuda:0"
            self.weight = torch.nn.Parameter(self.weight.detach().clone())
            return self

    pipe.text_encoder = TextEncoder(2, 2, bias=False)
    text_master = capture_cpu_master(pipe.text_encoder)

    @contextmanager
    def execution(_model, *, qkv=True, rotary=False):
        assert rotary is (steps == 10 and tile_batch == 4)
        active.append(qkv)
        try:
            yield {"qkv": qkv}
        finally:
            assert active.pop() == qkv

    @contextmanager
    def audio_context(_model, _scheduler, video, **_):
        assert active == [False]
        pipe.fixed = video
        yield {}

    def sample(wrapper, cond, **_):
        frames = 32 if duration == 4 else 37
        assert cond["geometry"].num_latent_frames == frames
        for _ in range(6):
            wrapper()
        return torch.zeros(1, frames, 24, 2, 2), torch.zeros(1, 414, 32)

    sampling = NS(
        source_revision="fixture",
        geometry=NS(
            audio_latent_num_frames=lambda frames: {107: 414, 124: 414}[frames],
            video_latent_num_frames=lambda frames: {107: 32, 124: 37}[frames],
        ),
        numerics=NS(
            prepare_conditioning=lambda value, *_, **__: value,
            configure_numerics_backend=lambda *_, **__: None,
            BF16_CONDITIONING="bf16",
        ),
        packing=NS(
            video_rows_to_external=lambda value, *_: value,
            audio_external_to_storage=lambda value: value,
        ),
        sampler=NS(sample_ladder=sample),
        wrapper=NS(H3DiffusionWrapper=lambda model: model),
    )
    attention = NS(backend="sdpa", select=lambda _: None, reset=lambda: None)
    tile_model = NS(training=False, _run_tile_tasks=lambda *args: [])
    original_tiles = tile_model._run_tile_tasks

    def decode(*_, **kwargs):
        assert kwargs["duration_seconds"] == duration
        return NS(stage_durations={}, media={})

    decoder = NS(
        generate_mp4=decode,
        video=NS(module=NS(model=tile_model)),
    )
    runtime = NS(
        closed=False,
        pipe=pipe,
        adapter=NS(strength=0),
        attention=attention,
        clock=NS(select=lambda *_: None, before=lambda *_: None),
        decoder=decoder,
        text_encoder_residency="staged" if staged_text else "resident",
        text_cpu_master=text_master if staged_text else None,
        trunk_cpu_master=None,
    )
    if qad:

        @contextmanager
        def selected(*, adapters):
            assert all(a.strength == 0 for a in adapters)
            qad_active.append(True)
            record = {"restored": False}
            try:
                yield record
            finally:
                qad_active.pop()
                record["restored"] = True

        runtime.qad_visual = NS(selected=selected)
    session = PikaAVSession(runtime, adapter, clock, sampling=sampling)
    first = tmp_path / "first.png"
    Image.new("RGB", (32, 32)).save(first)
    request = VideoRequest(
        "A paper shape unfolds.",
        width=32,
        height=32,
        first_frame=first if mode == "i2va" else None,
        duration_seconds=duration,
    )
    monkeypatch.setattr(
        "vflash.adapters.h3_reference_execution.dense_reference_execution", execution
    )
    monkeypatch.setattr(
        "vflash.adapters.h3_reference_audio.native_audio_context", audio_context
    )
    for name in ("synchronize", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *_: 0)
    generator = torch.Generator
    monkeypatch.setattr(torch, "Generator", lambda **_: generator())
    original_encode, original_keyframes = pipe.encode_prompt, pipe.encode_keyframes
    prior_schedulers = pipe.scheduler, pipe.audio_scheduler
    events = []
    recipe = PikaAVRecipe(
        audio_steps=steps,
        visual_attention="sdpa",
        overlap_decode=False,
        decode_tile_batch_size=tile_batch,
        strict_rotary=steps == 10 and tile_batch == 4,
        visual_weights="nvfp4-qad" if qad else "original-row-fp8",
    )
    if fail_audio and steps:
        with pytest.raises(RuntimeError, match="fixture audio failure"):
            session.generate(request, tmp_path / "out", recipe=recipe)
    else:
        result = session.generate(
            request,
            tmp_path / "out",
            recipe=recipe,
            on_progress=lambda *args: events.append(args),
        )
        assert result["actual_forwards"] == 6 + steps
        if qad:
            assert result["execution"]["visual_weights"]["restored"]
            assert result["weight_format"] == "nvfp4-qad-merged"
        assert result["recipe"] == recipe.identity and result["video_preserved_during_audio"]
        assert result["conditioning"]["prompt_calls"] == 1
        assert result["conditioning"]["prompt_reuses"] == int(steps > 0)
        assert len(events) == 11 + steps
        assert result["conditioning"]["text_residency"]["applied"] is staged_text
    assert pipe.text_encoder.weight.data_ptr() == text_master.state_dict["weight"].data_ptr()
    assert pipe.encode_prompt == original_encode and pipe.encode_keyframes == original_keyframes
    assert (pipe.scheduler, pipe.audio_scheduler) == prior_schedulers
    assert adapter.strength == runtime.adapter.strength == 0
    assert all(layer.clock == "original" for layer in model.layers)
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert (
        not session.owner.locked()
        and not active
        and not qad_active
        and not torch.cuda.is_initialized()
    )
    assert tile_model._run_tile_tasks is original_tiles
    assert not hasattr(tile_model, "_vflash_tile_batch_owner")


@pytest.mark.parametrize("fail_clock", [False, True])
def test_runtime_replaces_segmented_owner_and_unwinds_partial_load(
    tmp_path, monkeypatch, fail_clock
):
    pytest.importorskip("torch")
    from vflash.pipeline.pika_av_runtime import PikaAVRuntime
    from vflash.pipeline.reference_av_runtime import ReferenceAVRuntime

    closed = []

    def base_init(self, **_):
        self.pipe = NS(transformer="owned-model")
        self.closed = False
        self.session = "old-segmented-session"

    def base_close(self):
        self.closed = True
        self.session = self.pipe = None
        closed.append("base")

    class Adapter:
        def __init__(self, *_, **__):
            self.strength = 1

        def close(self):
            closed.append("pika-adapter")

    def make_clock(_):
        if fail_clock:
            raise ValueError("fixture clock rejection")
        return NS(tensors={"loaded": True})

    monkeypatch.setattr(ReferenceAVRuntime, "__init__", base_init)
    monkeypatch.setattr(ReferenceAVRuntime, "close", base_close)
    monkeypatch.setattr(
        "vflash.adapters.pika_flex_source.load_pika_sampling", lambda *_, **__: NS()
    )
    monkeypatch.setattr("vflash.adapters.h3_reference_lora.H3ActivationAdapter", Adapter)
    monkeypatch.setattr("vflash.adapters.pika_clock.PikaSamplingClock", make_clock)
    kwargs = dict(
        pika_source=tmp_path,
        pika_adapter=tmp_path / "adapter",
        pika_manifest={},
        pika_tables=tmp_path,
        predictor=tmp_path / "predictor",
        trust_local_code=True,
    )
    if fail_clock:
        with pytest.raises(ValueError, match="clock rejection"):
            PikaAVRuntime(**kwargs)
    else:
        runtime = PikaAVRuntime(**kwargs)
        assert runtime.session.runtime is runtime and runtime.pika_adapter.strength == 0
        clock = runtime.pika_clock
        runtime.close()
        assert not clock.tensors and runtime.closed
    assert closed == ["pika-adapter", "base"]


@pytest.mark.parametrize("mode", ["t2va", "i2va"])
@pytest.mark.parametrize("fail_audio", [False, True])
def test_qad_picture_scope_exits_before_original_soundtrack(
    tmp_path, monkeypatch, mode, fail_audio
):
    test_actual_session_counts_freezes_picture_and_restores_owners(
        tmp_path, monkeypatch, mode, 10, fail_audio, 1, False, qad=True
    )


def test_qad_requires_explicit_assets_without_changing_original_recipe_identity():
    from vflash.pipeline.pika_av import PikaAVSession

    assert PikaAVRecipe().identity == "pika6-larry10-v1"
    recipe = PikaAVRecipe(visual_weights="nvfp4-qad")
    assert recipe.identity == "pika-qad6-larry10-v1"
    from vflash.pipeline import VideoRequest

    session = PikaAVSession(NS(), None, None, sampling=None)
    with pytest.raises(ValueError, match="explicit resident visual weights"):
        session.generate(VideoRequest("A scene."), None, recipe=recipe)
