"""Load the pinned upstream H3 family without unrelated optional model families."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import textwrap
import types
from pathlib import Path

VIDEOX_REVISION = "13bb3c625e8b1e457d7124d92207b49e00e4a6fe"


def load_reference_modules(source: Path, *, trust_local_code: bool = False):
    if trust_local_code is not True:
        raise ValueError("Loading the local reference implementation requires explicit trust")
    root = source.resolve(strict=True)
    if not (root / "pipeline/pipeline_minimax_h3.py").is_file():
        raise ValueError("Expected the pinned videox_fun package directory")
    existing = sys.modules.get("videox_fun")
    if existing is not None and list(getattr(existing, "__path__", [])) != [str(root)]:
        raise ValueError("Another reference implementation already owns this interpreter")
    if existing is None:
        for part in ("", ".dist", ".models", ".pipeline", ".utils"):
            name = "videox_fun" + part
            module = types.ModuleType(name)
            module.__path__ = [str(root / part[1:].replace(".", "/"))] if part else [str(root)]
            module.__package__ = name
            sys.modules[name] = module
    os.environ["VIDEOX_ATTENTION_TYPE"] = "SDPA"
    fuser = importlib.import_module("videox_fun.dist.fuser")
    dist = sys.modules["videox_fun.dist"]
    for name in (
        "get_sequence_parallel_rank",
        "get_sequence_parallel_world_size",
        "get_sp_group",
    ):
        setattr(dist, name, getattr(fuser, name))
    dist.MiniMaxH3MultiGPUsAttnProcessor = importlib.import_module(
        "videox_fun.dist.minimax_h3_xfuser"
    ).MiniMaxH3MultiGPUsAttnProcessor
    models = sys.modules["videox_fun.models"]
    for module, name in (
        ("minimax_h3_transformer3d", "MiniMaxH3Transformer3DModel"),
        ("minimax_h3_vae", "AutoencoderKLMiniMaxH3"),
        ("minimax_h3_vae_audio", "AutoencoderKLMiniMaxH3Audio"),
    ):
        setattr(
            models, name, getattr(importlib.import_module("videox_fun.models." + module), name)
        )
    scheduler = importlib.import_module(
        "videox_fun.utils.fm_solvers_minimax_h3"
    ).MiniMaxH3Scheduler
    sys.modules["videox_fun.utils"].MiniMaxH3Scheduler = scheduler
    pipeline = importlib.import_module("videox_fun.pipeline.pipeline_minimax_h3")
    transformer = models.MiniMaxH3Transformer3DModel
    if not getattr(transformer, "_vflash_fp32_time_embedding", False):
        original = inspect.getsource(transformer.forward)
        old = (
            "temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))"
            ".to(hidden_states.dtype)"
        )
        if original.count(old) != 1:
            raise ValueError("Reference time projection differs from the pinned implementation")
        # Original FL computes SiLU on the FP32 time embedding before its BF16
        # projection. The upstream reference cast here changes that arithmetic.
        corrected = textwrap.dedent(original).replace(
            old, "temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))"
        )
        namespace = dict(vars(sys.modules[transformer.__module__]))
        exec(compile(corrected, "<h3-reference-fp32-time>", "exec"), namespace)
        transformer.forward = namespace["forward"]
        transformer._vflash_fp32_time_embedding = True
    return types.SimpleNamespace(
        pipeline=pipeline, models=models, scheduler=scheduler, revision=VIDEOX_REVISION
    )
