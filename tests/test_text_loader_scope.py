"""Constructor ownership and third-party isolation, without CUDA/model loads."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vflash.adapters.diffusers_h3 import DiffusersConditioner
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.pipeline import runtime
from vflash.pipeline.assets import PreparedPipelineAssets
from vflash.pipeline.contracts import PipelineAssets


def prepared(tmp_path):
    return PreparedPipelineAssets(
        PipelineAssets(**{key: tmp_path for key in PipelineAssets.__dataclass_fields__}),
        tmp_path / "receipt.json",
        "a" * 64,
        (),
        "i2va-base16-bf16-sm89",
    )


def test_default_text_loader_preserves_pinned_offline_arguments(tmp_path, monkeypatch):
    seen = []

    class Qwen:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            seen.append((args, kwargs))
            return "original"

    package = ModuleType("transformers")
    package.Qwen3VLForConditionalGeneration = Qwen
    monkeypatch.setitem(sys.modules, "transformers", package)
    original = Qwen.__dict__["from_pretrained"]
    owner = DiffusersConditioner.__new__(DiffusersConditioner)
    owner._text_encoder_loader = None
    owner._torch = SimpleNamespace(bfloat16="bf16")
    assert owner._load_text_encoder(tmp_path) == "original"
    assert seen == [
        (
            (str(tmp_path),),
            {"subfolder": "text_encoder", "dtype": "bf16", "local_files_only": True},
        )
    ]
    assert Qwen.__dict__["from_pretrained"] is original


def test_owned_loaders_do_not_replace_third_party_factory(tmp_path, monkeypatch):
    package = ModuleType("transformers")

    class Qwen:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pytest.fail("custom loading touched the third-party default factory")

    package.Qwen3VLForConditionalGeneration = Qwen
    monkeypatch.setitem(sys.modules, "transformers", package)
    original = Qwen.__dict__["from_pretrained"]
    a, b = (DiffusersConditioner.__new__(DiffusersConditioner) for _ in range(2))
    a._text_encoder_loader = lambda path: ("a", path)
    b._text_encoder_loader = lambda path: ("b", path)
    assert a._load_text_encoder(tmp_path) == ("a", tmp_path)
    assert b._load_text_encoder(tmp_path) == ("b", tmp_path)

    def fail(path):
        raise RuntimeError("load failed")

    a._text_encoder_loader = fail
    with pytest.raises(RuntimeError, match="load failed"):
        a._load_text_encoder(tmp_path)
    assert b._load_text_encoder(tmp_path) == ("b", tmp_path)
    assert Qwen.__dict__["from_pretrained"] is original


@pytest.mark.parametrize("native", [False, True])
def test_real_pipeline_threads_loader_to_one_owned_conditioner(tmp_path, monkeypatch, native):
    torch = pytest.importorskip("torch")
    events = []

    class Stage:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(self.name + ":close")

    def conditioner(assets, *, text_encoder_loader):
        owner = DiffusersConditioner.__new__(DiffusersConditioner)
        owner._text_encoder_loader = text_encoder_loader
        events.append(owner._load_text_encoder(assets.assets.model_directory))
        return Stage("conditioner")

    monkeypatch.setattr(runtime, "media_executables", lambda: None)
    monkeypatch.setattr(runtime, "validate_adapter_dependencies", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    native_calls = []

    def native_stage(*args, **options):
        native_calls.append(options)
        return Stage("native")

    monkeypatch.setattr(runtime, "NativeEngineSession", native_stage)
    monkeypatch.setattr(runtime, "DiffusersConditioner", conditioner)
    monkeypatch.setattr(runtime, "OfficialMediaDecoder", lambda **kw: Stage("media"))
    options = {
        "device": NvidiaDevice(0, "fixture", "RTX 4090", 48, "8.9", 450),
        "trust_local_code": True,
    }

    def native_factory(*a, **k):
        return None

    if native:
        # Constructor/forwarding test only; target capacity gates remain covered
        # by test_resident_pipeline and are not broadened by a trusted loader.
        monkeypatch.setattr(
            "vflash.pipeline.residency.validate_residency",
            lambda plan, choice, attention, hybrid: choice,
        )
        options.update(
            native_denoiser_loader=native_factory,
            weight_residency="resident",
            attention_backend="torch-flash",
        )
    a = runtime.H3Pipeline(prepared(tmp_path), **options, text_encoder_loader=lambda p: "a")
    b = runtime.H3Pipeline(prepared(tmp_path), **options, text_encoder_loader=lambda p: "b")
    try:
        assert events == []  # construction is still lazy
        a.prepare()
        b.prepare()
        a.prepare()
        assert events == ["a", "b"]
        assert len(native_calls) == 2
        for call in native_calls:
            if native:
                assert call["native_denoiser_loader"] is native_factory
            else:
                assert "native_denoiser_loader" not in call
    finally:
        a.close()
        b.close()
    assert events.count("conditioner:close") == 2 and events.count("native:close") == 2
    assert not torch.cuda.is_initialized()


def test_loader_input_is_code_not_a_serialized_class_or_path(tmp_path):
    options = {
        "device": NvidiaDevice(0, "fixture", "RTX 4090", 48, "8.9", 450),
        "trust_local_code": True,
    }
    for invalid in ("external.module.factory", {}, 7):
        with pytest.raises(ContractError, match="callable"):
            runtime.H3Pipeline(prepared(tmp_path), **options, text_encoder_loader=invalid)
        with pytest.raises(ContractError, match="callable"):
            DiffusersConditioner(prepared(tmp_path), text_encoder_loader=invalid)
