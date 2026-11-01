"""Original six-step FL timestep projections over explicit prepared base weights.

The table removes request-independent matrix work; it does not replace the
released renoising sampler or grant a quantized runtime quality qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from vflash.contracts import ContractError
from vflash.pipeline.pika_flex import (
    PIKA_BASE_REVISION,
    PIKA_RECIPE,
    PIKA_SOURCE_REVISION,
    shifted_sigmas,
)


def pika_fl_times(mode):
    if mode not in {"t2va", "i2va", "fl2va", "l2va"}:
        raise ContractError("The prepared Pika clock covers the FL partition only")
    video = torch.tensor(shifted_sigmas(12), dtype=torch.float32)
    audio = torch.tensor(shifted_sigmas(3), dtype=torch.float32)
    rows = []
    for v, a in zip(video[:-1], audio[:-1], strict=True):
        clocks = [1.0 - float(v), 1.0 - float(a)]
        if mode != "t2va":
            clocks.append(0.999)
        rows.append(torch.tensor(sorted(set(clocks)), dtype=torch.float32))
    return rows


class PikaSamplingClock:
    def __init__(self, root, *, device="cuda:0"):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        expected = {
            "status": "completed",
            "recipe": PIKA_RECIPE,
            "source_revision": PIKA_SOURCE_REVISION,
            "base_revision": PIKA_BASE_REVISION,
        }
        if any(self.manifest.get(k) != v for k, v in expected.items()):
            raise ContractError("Pika projection tables differ from the fixed original recipe")
        self.device = device
        self.tensors = {}
        self.plan = None
        self.step = self.completed_forwards = 0

    def select_mode(self, mode):
        self.tensors, self.plan = {}, None
        self.step = self.completed_forwards = 0
        expected = pika_fl_times(mode)
        name = f"pika6-fl-{'t2va' if mode == 't2va' else 'keyframe'}.safetensors"
        rows = [row for row in self.manifest["files"] if row["name"] == name]
        if len(rows) != 1:
            raise ContractError(
                "Exactly one prepared Pika clock per conditioning mode is required"
            )
        row = rows[0]
        plan = row["plan"]
        if (
            plan.get("nfe") != 6
            or len(plan.get("timesteps", [])) != 6
            or any(
                not torch.equal(t, torch.tensor(v, dtype=torch.float32))
                for t, v in zip(expected, plan["timesteps"], strict=True)
            )
        ):
            raise ContractError(
                "Prepared clocks must match the original six renoising evaluations"
            )
        file = self.root / name
        if file.stat().st_size != row["bytes"]:
            raise ContractError("Prepared projection size differs from its receipt")
        tensors = load_file(str(file), device=self.device)
        keys = {f"time-{i}" for i in range(6)} | {
            f"block-{b:03d}-step-{i:02d}" for b in range(51) for i in range(6)
        }
        if set(tensors) != keys:
            raise ContractError("Prepared projection coverage is incomplete")
        for step, times in enumerate(expected):
            embed = tensors[f"time-{step}"]
            if list(embed.shape) != [len(times), 2688] or embed.dtype != torch.float32:
                raise ContractError(
                    "Pika time embeddings must preserve original FP32 precision"
                )
            for block in range(51):
                value = tensors[f"block-{block:03d}-step-{step:02d}"]
                if (
                    list(value.shape) != [len(times), (18 if block < 50 else 2) * 5376]
                    or value.dtype != torch.bfloat16
                ):
                    raise ContractError(
                        "Pika modulation must match the original BF16 projections"
                    )
        self.tensors, self.plan = tensors, plan
        self.step = self.completed_forwards = 0

    def before(self, _model, _args, kwargs):
        if self.plan is None or self.completed_forwards >= 6:
            raise ContractError("Pika clock was not selected or its six forwards are complete")
        self.step = self.completed_forwards
        expected = torch.tensor(
            self.plan["timesteps"][self.step],
            device=kwargs["timestep"].device,
            dtype=torch.float32,
        )
        if kwargs["timestep"].dtype != torch.float32 or not torch.equal(
            kwargs["timestep"], expected
        ):
            raise ContractError("Executed timestep differs from the prepared Pika clock")

    def after(self, _model, _args, _result):
        self.completed_forwards += 1

    def time_embedding(self, _values):
        return self.tensors[f"time-{self.step}"]
