"""Compile explicit low-noise H3 modulation from original FL/Ref weights.

Source checkpoints are read-only. The derived assets contain only modulation
and schedule metadata; they do not bundle another denoiser or a VAE.
"""

from __future__ import annotations

import json
import math
import time

from vflash.native.h3_refinement_schedule import refinement_schedule, timestep_rows


def time_embeddings(load, rows):
    import torch
    import torch.nn.functional as functional

    weights = {
        name: load("time_embedder." + name).float()
        for name in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")
    }
    frequency_count = weights["linear_1.weight"].shape[1] // 2
    frequency = (
        -math.log(10000) * torch.arange(frequency_count, dtype=torch.float32) / frequency_count
    )
    embeddings = torch.zeros((len(rows), max(map(len, rows)), weights["linear_2.bias"].numel()))
    for index, row in enumerate(rows):
        phases = row[:, None] * frequency.exp()[None]
        features = torch.cat((phases.cos(), phases.sin()), dim=-1)
        hidden = functional.linear(
            features, weights["linear_1.weight"], weights["linear_1.bias"]
        )
        embeddings[index, : len(row)] = functional.linear(
            functional.silu(hidden), weights["linear_2.weight"], weights["linear_2.bias"]
        )
    return embeddings


def compile_refinement_tables(
    base_directory, reference_directory, output, *, steps, denoise, mode
):
    import torch
    from safetensors.torch import save_file

    from vflash.adapters.checkpoints import IndexedCheckpoint
    from vflash.compiler.math import adaln_table
    from vflash.native.h3_hybrid import validate_hybrid_reference

    if mode not in ("face", "audio"):
        raise ValueError("unknown refinement modality")
    schedule = refinement_schedule(steps=steps, denoise=denoise)
    base, reference = IndexedCheckpoint(base_directory), IndexedCheckpoint(reference_directory)
    validate_hybrid_reference(reference)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = timestep_rows(schedule)
    counts = tuple(map(len, rows))
    records = []
    with torch.inference_mode():
        timing = time_embeddings(base.load, rows)
        for index in range(50):
            checkpoint = base if index < 25 else reference
            prefix = f"transformer_blocks.{index}.adaln_proj.linear."
            weights = checkpoint.load_many((prefix + "weight", prefix + "bias"))
            table = adaln_table(
                timing,
                counts,
                weights[prefix + "weight"],
                weights[prefix + "bias"],
                modalities=3,
                modulations=6,
            )
            if table.shape != (steps + 1, 9, 6, 5376) or not table.isfinite().all():
                raise ValueError("invalid actual H3 modulation table")
            path = output / f"block-{index:03d}.safetensors"
            save_file({"adaln.table": table}, str(path))
            records.append({"file": path.name, "size": path.stat().st_size})
            del table, weights
        final = adaln_table(
            timing,
            counts,
            base.load("norm_out.linear.weight"),
            base.load("norm_out.linear.bias"),
            modalities=1,
            modulations=2,
        )
        if not final.isfinite().all():
            raise ValueError("invalid final modulation")
        path = output / "final.safetensors"
        save_file({"adaln.table": final}, str(path))
        records.append({"file": path.name, "size": path.stat().st_size})
    manifest = {
        "schema": 1,
        "method": "hybrid-low-noise-official-modulation-v1",
        "mode": mode,
        "steps": steps,
        "denoise": denoise,
        "start_evaluation": 1,
        "schedule": schedule.to_mapping(),
        "files": records,
        "source": "original-FL-time-and-head; FL-blocks-0-24; Ref-blocks-25-49",
        "turbo_residuals": "retained" if mode == "face" else "disabled-during-refinement",
        "compile_seconds": time.monotonic() - started,
        "preparation_device": "cpu",
    }
    (output / "refinement.json").write_text(json.dumps(manifest, indent=2))
    return manifest
