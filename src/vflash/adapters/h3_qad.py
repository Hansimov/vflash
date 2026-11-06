"""Owned Pika QAD visual weights over an independently retained FL audio trunk.

The quantized checkpoint already includes its Pika adapter, including the two
text-refiner blocks. Every changed parameter and forward is restored before
native soundtrack sampling. No public/product default is selected here.
"""

import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open

from vflash.native.h3_nvfp4_linear import NVFP4Execution, NVFP4Weight
from vflash.pipeline.pika_flex import PIKA_ADAPTER_SHA256, PIKA_BASE_REVISION

QAD_REPOSITORY = "FastVideo/Pika-H3-Flex-NVFP4-QAD"
QAD_REVISION = "0434eb89466c15ab451cb92fdd20995b4a2d1f1a"
LINEARS = {
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
    "attn.to_out.0": "attn.to_out",
    "ff.net.0.proj": "ff.fc_in",
    "ff.net.2": "ff.fc_out",
}
CLOCK_KEYS = {f"time_embedder.linear_{i}.{leaf}" for i in (1, 2) for leaf in ("weight", "bias")}
CLOCK_KEYS.update({"norm_out.linear.weight", "norm_out.linear.bias"})


def validate_manifest(value):
    expected = {
        "schema": 1,
        "source_repository": QAD_REPOSITORY,
        "source_revision": QAD_REVISION,
        "partition": "transformer",
        "pika_adapter_already_merged": True,
        "merged_adapter_sha256": PIKA_ADAPTER_SHA256["fl2va"],
        "projection_base_revision": PIKA_BASE_REVISION,
        "projection_relation": "all-106-source-tensors-byte-identical-v1",
    }
    if not isinstance(value, dict) or any(
        type(value.get(k)) is not type(v) or value.get(k) != v for k, v in expected.items()
    ):
        raise ValueError(
            "QAD requires its FL source, merged adapter and complete projection provenance"
        )


class H3QADVisualWeights:
    """One model's prepared visual substitution; the caller serializes requests."""

    def __init__(self, model, *, packed_weights: Path, dense_weights: Path, manifest: dict):
        validate_manifest(manifest)
        if model.training or len(model.transformer_blocks) != 50:
            raise ValueError("QAD requires the complete evaluation-mode H3 trunk")
        self.model, self.closed, self.active = model, False, False
        self.owner = threading.Lock()
        self.execution = NVFP4Execution()
        self.matrices, self.dense, self.handles = {}, [], []
        self.cpu_matrices, self.cpu_dense = {}, []
        self.calls, self.refiner_calls = {}, {}
        targets = {
            f"transformer_blocks.{i}.{name}": f"transformer_blocks.{i}.{source}"
            for i in range(50)
            for name, source in LINEARS.items()
        }
        parameters = dict(model.named_parameters())
        projections = {
            f"transformer_blocks.{i}.adaln_proj.linear.{leaf}"
            for i in range(50)
            for leaf in ("weight", "bias")
        }
        expected_dense = set(parameters) - projections - {name + ".weight" for name in targets}
        if len(expected_dense) != 238 or not expected_dense >= CLOCK_KEYS:
            raise ValueError("QAD dense coverage differs from the original H3 model")
        device = self.device = next(model.parameters()).device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError("QAD visual substitution requires its owned SM120 model")
        try:
            with safe_open(str(dense_weights), framework="pt", device="cpu") as source:
                if set(source.keys()) != expected_dense:
                    raise ValueError(
                        "QAD must include every original dense and merged-refiner tensor"
                    )
                for name in sorted(expected_dense):
                    tensor = source.get_tensor(name)
                    if (
                        tensor.shape != parameters[name].shape
                        or tensor.dtype != parameters[name].dtype
                    ):
                        raise ValueError(
                            "QAD dense tensor differs from its model descriptor: " + name
                        )
                    if name in CLOCK_KEYS:
                        # Preparation proved equality of all original projections.
                        continue
                    parent, _, leaf = name.rpartition(".")
                    self.cpu_dense.append((model.get_submodule(parent), leaf, tensor))
            with safe_open(str(packed_weights), framework="pt", device="cpu") as source:
                fields = {
                    "_nvfp4_weight",
                    "_nvfp4_weight_scale",
                    "_nvfp4_alpha",
                    "_weight_global_sf",
                }
                required = {
                    prefix + "::" + field for prefix in targets.values() for field in fields
                }
                required.update(
                    prefix + "::_nvfp4_input_global_sf"
                    for prefix in targets.values()
                    if ".ff." in prefix
                )
                if set(source.keys()) != required:
                    raise ValueError("QAD must cover all 300 original main-block linears")
                unit = torch.ones((), dtype=torch.float32, device="cpu")
                for name, prefix in targets.items():
                    module = model.get_submodule(name)
                    weight = NVFP4Weight(
                        source.get_tensor(prefix + "::_nvfp4_weight"),
                        source.get_tensor(prefix + "::_nvfp4_weight_scale"),
                        source.get_tensor(prefix + "::_nvfp4_alpha"),
                        source.get_tensor(prefix + "::_nvfp4_input_global_sf")
                        if ".ff." in prefix
                        else unit,
                    )
                    if (
                        module.bias is not None
                        or module.in_features != weight.in_features
                        or module.out_features != weight.out_features
                    ):
                        raise ValueError("QAD packed matrix differs from its target")
                    self.cpu_matrices[name] = (module, weight)
            for block in range(2):
                for name in LINEARS:
                    target = f"token_refiner.refiner_blocks.{block}.{name}"
                    module = model.get_submodule(target)
                    self.refiner_calls[target] = 0

                    def observe(*_, key=target):
                        if self.active:
                            self.refiner_calls[key] += 1

                    self.handles.append(module.register_forward_hook(observe))
        except BaseException:
            self.close()
            raise

    def _materialize(self):
        """Create only this visual scope's CUDA copies; mapped source tensors stay owned."""
        if self.matrices or self.dense:
            raise ValueError("Previous QAD visual copies were not released")
        for module, leaf, tensor in self.cpu_dense:
            self.dense.append(
                (module, leaf, torch.nn.Parameter(tensor.to(self.device), requires_grad=False))
            )
        for name, (module, weight) in self.cpu_matrices.items():
            self.matrices[name] = (
                module,
                NVFP4Weight(
                    weight.packed.to(self.device),
                    weight.scales.to(self.device),
                    weight.alpha.to(self.device),
                    weight.input_scale.to(self.device),
                ),
            )

    def _release_materialized(self):
        owned_bytes = sum(t.numel() * t.element_size() for _, _, t in self.dense)
        owned_bytes += sum(
            t.numel() * t.element_size()
            for _, weight in self.matrices.values()
            for t in (weight.packed, weight.scales, weight.alpha, weight.input_scale)
        )
        self.matrices.clear()
        self.dense.clear()
        return owned_bytes

    @contextmanager
    def selected(self, *, adapters):
        if self.closed or not self.owner.acquire(blocking=False):
            raise ValueError("QAD visual weights are closed or already in use")
        saved_parameters, saved_forwards = [], []
        record = {
            "weight_format": "nvfp4-qad-merged",
            "source_revision": QAD_REVISION,
            "linear_contract": self.execution.contract,
            "restored": False,
        }
        try:
            if self.model.training or any(a.strength != 0 for a in adapters):
                raise ValueError("Merged QAD must not receive a second Pika or Larry adapter")
            began = time.monotonic()
            self._materialize()
            record["load_seconds"] = time.monotonic() - began
            self.active = True
            self.calls = dict.fromkeys(self.matrices, 0)
            self.refiner_calls = dict.fromkeys(self.refiner_calls, 0)
            for module, leaf, value in self.dense:
                saved_parameters.append((module, leaf, module._parameters[leaf]))
                module._parameters[leaf] = value

            def replacement(name, weight):
                def forward(values):
                    output = self.execution.linear(values, weight)
                    self.calls[name] += 1
                    return output

                return forward

            for name, (module, weight) in self.matrices.items():
                saved_forwards.append(
                    (module, "forward" in module.__dict__, module.__dict__.get("forward"))
                )
                module.forward = replacement(name, weight)
            yield record
            if (
                len(self.calls) != 300
                or any(n != 6 for n in self.calls.values())
                or len(self.refiner_calls) != 12
                or any(n != 6 for n in self.refiner_calls.values())
            ):
                raise ValueError(
                    "QAD did not execute all merged targets in the original six forwards"
                )
            record.update(
                main_linear_calls=sum(self.calls.values()),
                merged_refiner_calls=sum(self.refiner_calls.values()),
            )
        finally:
            for module, had, original in reversed(saved_forwards):
                if had:
                    module.forward = original
                else:
                    del module.forward
            for module, leaf, original in reversed(saved_parameters):
                module._parameters[leaf] = original
            self.active = False
            record["released_visual_bytes"] = self._release_materialized()
            record["visual_copies_after_scope"] = len(self.matrices) + len(self.dense)
            record["host_storage"] = "source-mapped"
            record["restored"] = True
            self.owner.release()

    def close(self):
        if self.active:
            raise ValueError("Leave the active visual stage before releasing QAD weights")
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._release_materialized()
        self.cpu_matrices.clear()
        self.cpu_dense.clear()
        self.closed = True
