"""Explicit Ref-picture / FL-sound allocation with independent model weights.

This prepared-weight candidate is separate from the qualified FL Pika runtime.
Original CPU storages allow only the active transformer to occupy the GPU;
shared text and media components remain owned by this single runtime.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from vflash.pipeline.reference_av_runtime import ReferenceAVRuntime


class PikaReferenceAVRuntime(ReferenceAVRuntime):
    def __init__(
        self,
        *,
        reference_cache: Path,
        reference_tables: Path,
        reference_adapter: Path,
        reference_manifest: dict,
        pika_source: Path,
        reference_short_edge: int | None = None,
        trust_local_code: bool = False,
        **components,
    ):
        if reference_short_edge is not None and (
            type(reference_short_edge) is not int or reference_short_edge not in (1024, 2048)
        ):
            raise ValueError("Explicit reference short edge must be 1024 or 2048")
        self.reference_short_edge = reference_short_edge
        import torch

        from vflash.adapters.h3_reference_assets import (
            load_cached_trunk,
            validate_cached_structure,
        )
        from vflash.adapters.h3_reference_lora import H3ActivationAdapter
        from vflash.adapters.h3_reference_source import load_reference_modules
        from vflash.adapters.pika_clock import PikaSamplingClock
        from vflash.adapters.pika_flex_source import (
            load_pika_reference_conditioning,
            load_pika_sampling,
        )
        from vflash.pipeline.pika_flex import validate_adapter_manifest
        from vflash.pipeline.residency import capture_cpu_master, restore_cpu_master

        validate_adapter_manifest(reference_manifest, "ref2va")
        if "retain_trunk_cpu" in components or "conditioning_precision" in components:
            raise ValueError(
                "The Ref runtime owns its explicit residency and FP32 conditioning"
            )
        self.reference_model = self.reference_master = self.reference_adapter = None
        self.reference_clock = self.fl_model = self.active_partition = None
        self.sampling = load_pika_sampling(pika_source, trust_local_code=trust_local_code)
        self.conditioning_source = load_pika_reference_conditioning(
            pika_source, trust_local_code=trust_local_code
        )
        self.family = load_reference_modules(
            components["source_root"], trust_local_code=trust_local_code
        )
        super().__init__(
            retain_trunk_cpu=True,
            conditioning_precision="fp32",
            trust_local_code=trust_local_code,
            **components,
        )
        try:
            self.fl_model = self.pipe.transformer
            restore_cpu_master(self.fl_model, self.trunk_cpu_master)
            started = time.monotonic()
            with torch.device("meta"):
                model = self.family.models.MiniMaxH3Transformer3DModel.from_config(
                    json.loads(
                        (reference_cache / "model/transformer_ref/config.json").read_text()
                    )
                )
            validate_cached_structure(
                model, reference_cache, reference_tables, trunk="transformer_ref"
            )
            self.reference_clock = PikaSamplingClock(reference_tables, partition="ref2va")
            load_cached_trunk(
                model,
                reference_cache,
                self.reference_clock,
                trunk="transformer_ref",
                device="cpu",
            )
            self.reference_model = model
            self.reference_master = capture_cpu_master(model)
            self.reference_adapter = H3ActivationAdapter(
                model, reference_adapter, "pika", manifest=reference_manifest
            )
            self.reference_adapter.strength = 0
            self.initialization_stages["reference_model_cpu"] = time.monotonic() - started
            from vflash.pipeline.pika_reference_av import PikaReferenceAVSession

            self.session = PikaReferenceAVSession(self, predictor=components["predictor"])
        except BaseException:
            self.close()
            raise

    def select_transformer(self, partition):
        import torch

        from vflash.pipeline.residency import restore_cpu_master

        if partition not in {"ref2va", "fl2va"} or self.closed:
            raise ValueError("A live explicitly selected transformer partition is required")
        torch.cuda.synchronize()
        began = time.monotonic()
        if self.active_partition == "ref2va":
            restore_cpu_master(self.reference_model, self.reference_master)
        elif self.active_partition == "fl2va":
            restore_cpu_master(self.fl_model, self.trunk_cpu_master)
        self.active_partition = None
        suspended = time.monotonic() - began
        model = self.reference_model if partition == "ref2va" else self.fl_model
        began = time.monotonic()
        model.to(device="cuda:0")
        self.pipe.transformer = model
        self.active_partition = partition
        torch.cuda.synchronize()
        return {
            "partition": partition,
            "suspend_seconds": suspended,
            "upload_seconds": time.monotonic() - began,
            "retained_cpu_bytes": self.reference_master.tensor_bytes
            + self.trunk_cpu_master.tensor_bytes,
        }

    def suspend_transformers(self):
        from vflash.pipeline.residency import restore_cpu_master

        if self.reference_model is not None and self.reference_master is not None:
            restore_cpu_master(self.reference_model, self.reference_master)
        if self.fl_model is not None and self.trunk_cpu_master is not None:
            restore_cpu_master(self.fl_model, self.trunk_cpu_master)
            self.pipe.transformer = self.fl_model
        self.active_partition = None

    def close(self):
        from vflash.pipeline.pika_reference_av import PikaReferenceAVSession

        session = getattr(self, "session", None)
        if isinstance(session, PikaReferenceAVSession) and session.owner.locked():
            raise ValueError("Cannot close an active reference request")
        if self.reference_adapter is not None:
            self.reference_adapter.close()
        if self.reference_clock is not None:
            self.reference_clock.tensors.clear()
        self.reference_model = self.reference_master = self.reference_adapter = None
        self.reference_clock = self.fl_model = self.active_partition = None
        super().close()
