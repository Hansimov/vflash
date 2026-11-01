from types import SimpleNamespace

import pytest

from vflash.contracts import ContractError
from vflash.pipeline.pika_components import PikaComponentCache


@pytest.fixture
def cached(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import vflash.pipeline.pika_components as module

    calls = []

    class Model:
        def __init__(self):
            self.weight = torch.tensor([1.5, 2.5], dtype=torch.bfloat16)

        def to(self, device):
            calls.append(("model_to", str(device)))
            return self

    def build(*args, **kwargs):
        calls.append(("build", kwargs.get("subfolder")))
        return Model()

    def apply(model, **kwargs):
        calls.append(("apply", kwargs))
        model.adapter = torch.zeros_like(model.weight)

    def load(model, path, **kwargs):
        calls.append(("load", path))
        model.adapter.copy_(torch.tensor([0.25, 0.5]))

    class Decoder:
        def __init__(self, *args, **kwargs):
            calls.append(("decoder_load", None))
            self.vae, self.audio_vae = object(), object()

        def to(self, device):
            calls.append(("decoder_to", str(device)))
            return self

    class Encoder:
        def __init__(self, *args, **kwargs):
            self.checkpoint = tmp_path
            self.device = "cpu"

        def free(self, what="all"):
            calls.append(("encoder_free", what))

    modules = {
        "fixture.core.loader": SimpleNamespace(
            build_transformer=build, apply_lora=apply, load_lora_weights=load
        ),
        "fixture.core.conditioning": SimpleNamespace(H3ConditionEncoder=Encoder),
        "fixture.core.vae": SimpleNamespace(H3Decoders=Decoder),
    }
    original_import = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name, *a, **kw: modules.get(name) or original_import(name, *a, **kw),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cache = PikaComponentCache(SimpleNamespace(__package__="fixture"))
    file = tmp_path / "adapter"
    file.write_bytes(b"prepared adapter")
    return cache, calls, file, (build, apply, load, Encoder, Decoder)


def execute(cache, file, partition="transformer"):
    with cache.installed():
        enc = cache.conditioning.H3ConditionEncoder()
        enc._load_vaes()
        enc.free()
        model = cache.loader.build_transformer(file.parent, subfolder=partition)
        cache.loader.apply_lora(model, rank=128, alpha=128)
        cache.loader.load_lora_weights(model, str(file))
        result = (model.weight + model.adapter).clone()
        model.to("cuda")
        cache.vae.H3Decoders(file.parent, device="cuda")
    return result


def test_reuse_preserves_values_and_offloads_before_other_large_stages(cached):
    cache, calls, file, originals = cached
    first = execute(cache, file)
    calls.clear()
    second = execute(cache, file)
    assert first.equal(second)
    assert not any(name in {"build", "apply", "load", "decoder_load"} for name, _ in calls)
    assert calls[0] == ("model_to", "cpu")
    # Decoder CUDA activation cannot precede transformer offload.
    for index, call in enumerate(calls):
        if call == ("decoder_to", "cuda"):
            assert calls[index - 1] == ("model_to", "cpu")
    assert cache.stats["transformer_loads"] == 1
    assert cache.stats["transformer_reuses"] == 1
    assert cache.loader.build_transformer is originals[0]
    assert cache.conditioning.H3ConditionEncoder is originals[3]
    assert cache.vae.H3Decoders is originals[4]
    cache.close()
    assert cache.model is cache.decoders is None


def test_partition_change_releases_previous_trunk_and_loads_correct_adapter(cached):
    cache, calls, file, _ = cached
    execute(cache, file)
    old = cache.model
    execute(cache, file, "transformer_ref")
    assert cache.model is not old
    assert [entry for entry in calls if entry[0] == "build"] == [
        ("build", "transformer"),
        ("build", "transformer_ref"),
    ]
    assert cache.stats["transformer_loads"] == 2


def test_failed_or_changed_adapter_restores_hooks_and_discards_partial_state(cached):
    cache, _calls, file, originals = cached
    execute(cache, file)
    file.write_bytes(b"changed")
    with pytest.raises(ContractError, match="changed"):
        execute(cache, file)
    assert cache.closed and cache.model is cache.decoders is None
    assert cache.loader.build_transformer is originals[0]
    assert cache.conditioning.H3ConditionEncoder is originals[3]
    with pytest.raises(ContractError, match="closed"):
        execute(cache, file)
