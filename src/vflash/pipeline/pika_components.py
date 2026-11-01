"""Bounded component reuse for the pinned Pika reference pipeline.

This opt-in lifecycle adapter leaves the released sampler, tensor dtypes, LoRA
math, conditioning and decoders unchanged. It owns one transformer partition
and one FP32 VAE pair. The transformer moves to CPU before text encoding or
video decoding; text weights load straight onto their device to avoid holding
another full CPU checkpoint beside the retained transformer.

Hooks are scoped to one serialized pipeline call and restored on every exit.
The owning runtime serializes all reference calls within this interpreter.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path

from vflash.contracts import ContractError


class PikaComponentCache:
    def __init__(self, pipeline):
        prefix = pipeline.__package__ + ".core."
        self.loader = importlib.import_module(prefix + "loader")
        self.conditioning = importlib.import_module(prefix + "conditioning")
        self.vae = importlib.import_module(prefix + "vae")
        self.model = None
        self.model_key = None
        self.adapter_key = None
        self.lora_ready = False
        self.decoders = None
        self.decoder_key = None
        self.closed = False
        self.stats = {
            "transformer_loads": 0,
            "transformer_reuses": 0,
            "decoder_loads": 0,
            "decoder_reuses": 0,
        }

    def offload_model(self):
        if self.model is not None:
            self.model.to("cpu")

    def offload_decoders(self):
        if self.decoders is not None:
            self.decoders.to("cpu")

    def close(self):
        # Dropping owned references releases both CPU and CUDA allocations;
        # callers do not retain a model handle across pipeline requests.
        self.model = self.decoders = None
        self.closed = True

    @contextmanager
    def installed(self):
        if self.closed:
            raise ContractError("Pika component cache is closed")
        import torch

        originals = {
            "build": self.loader.build_transformer,
            "apply": self.loader.apply_lora,
            "load": self.loader.load_lora_weights,
            "encoder": self.conditioning.H3ConditionEncoder,
            "decoder": self.vae.H3Decoders,
        }
        cache = self

        def build(path, *, dtype=torch.bfloat16, device="cpu", subfolder=None):
            cache.offload_decoders()
            key = (str(Path(path).resolve()), dtype, subfolder)
            if cache.model_key != key:
                cache.model = None  # Never retain both FL and Ref trunks.
                cache.model = originals["build"](
                    path, dtype=dtype, device=device, subfolder=subfolder
                )
                cache.model_key = key
                cache.adapter_key = None
                cache.lora_ready = False
                cache.stats["transformer_loads"] += 1
            else:
                cache.stats["transformer_reuses"] += 1
            return cache.model

        def apply(model, **kwargs):
            if model is not cache.model:
                raise ContractError("Pika adapter target is not owned")
            if cache.adapter_key is None:
                originals["apply"](model, **kwargs)
                cache.adapter_key = dict(kwargs)
            elif cache.adapter_key != kwargs:
                raise ContractError("Pika adapter configuration changed in a resident session")
            return model

        def load(model, path, *, strict=True):
            if model is not cache.model or strict is not True:
                raise ContractError("Pika requires its owned model and complete adapter")
            identity = Path(path).resolve(strict=True)
            stamp = identity.stat()
            key = (str(identity), stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns)
            if cache.lora_ready:
                if cache.lora_ready != key:
                    raise ContractError("Pika adapter changed after component preparation")
            else:
                originals["load"](model, path, strict=True)
                cache.lora_ready = key

        def decoders(checkpoint_dir, *, device="cpu", dtype=torch.float32):
            cache.offload_model()
            key = (str(Path(checkpoint_dir).resolve()), dtype)
            if cache.decoder_key != key:
                cache.decoders = None
                cache.decoders = originals["decoder"](
                    checkpoint_dir, device=device, dtype=dtype
                )
                cache.decoder_key = key
                cache.stats["decoder_loads"] += 1
            else:
                cache.decoders.to(device)
                cache.stats["decoder_reuses"] += 1
            return cache.decoders

        class Encoder(originals["encoder"]):
            def __init__(self, *args, **kwargs):
                cache.offload_model()
                cache.offload_decoders()
                torch.cuda.empty_cache()
                super().__init__(*args, **kwargs)

            def _load_text_encoder(self):
                if self._te is None:
                    from transformers import (
                        AutoProcessor,
                        AutoTokenizer,
                        Qwen3VLForConditionalGeneration,
                    )

                    root = self.text_encoder_checkpoint
                    tok = AutoTokenizer.from_pretrained(root / "tokenizer")
                    proc = AutoProcessor.from_pretrained(root / "processor")
                    model = Qwen3VLForConditionalGeneration.from_pretrained(
                        root / "text_encoder",
                        dtype=torch.bfloat16,
                        device_map={"": self.device},
                    ).eval()
                    self._te = (tok, proc, model)
                return self._te

            def _load_vaes(self):
                pair = decoders(str(self.checkpoint), device=self.device)
                self._vae, self._audio_vae = pair.vae, pair.audio_vae
                return self._vae, self._audio_vae

            def free(self, what="all"):
                if what in ("vae", "all"):
                    cache.offload_decoders()
                super().free(what)

        self.loader.build_transformer = build
        self.loader.apply_lora = apply
        self.loader.load_lora_weights = load
        self.conditioning.H3ConditionEncoder = Encoder
        self.vae.H3Decoders = decoders
        try:
            yield self
        except BaseException:
            self.close()
            raise
        finally:
            self.loader.build_transformer = originals["build"]
            self.loader.apply_lora = originals["apply"]
            self.loader.load_lora_weights = originals["load"]
            self.conditioning.H3ConditionEncoder = originals["encoder"]
            self.vae.H3Decoders = originals["decoder"]
