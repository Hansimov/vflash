"""Compile caller-owned clean-context modulation for native AV continuation."""

from pathlib import Path


def compile_native_history_tables(
    base_checkpoint: Path,
    *,
    reference_checkpoint: Path | None = None,
    reference_blocks: tuple[int, ...] = (),
    progress=None,
) -> tuple:
    """Read local H3 checkpoints on CPU and return the 51 timestep-one tables.

    Match the active pipeline's checkpoint and hybrid block selection. This does
    not fetch weights, change a prepared artifact, or qualify output quality.
    Compile once per model selection and reuse the returned tables. The caller
    owns thread count, storage and any deadline through the progress callback.
    """
    if (
        type(reference_blocks) is not tuple
        or any(type(i) is not int or not 0 <= i < 50 for i in reference_blocks)
        or tuple(sorted(set(reference_blocks))) != reference_blocks
        or (reference_checkpoint is None) != (not reference_blocks)
    ):
        raise ValueError("reference checkpoint and sorted unique hybrid blocks must agree")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")

    import torch

    from vflash.adapters.checkpoints import IndexedCheckpoint
    from vflash.compiler.math import adaln_table
    from vflash.compiler.refinement import time_embeddings

    base = IndexedCheckpoint(Path(base_checkpoint))
    reference = (
        IndexedCheckpoint(Path(reference_checkpoint))
        if reference_checkpoint is not None
        else None
    )
    chosen = frozenset(reference_blocks)
    tables = []
    with torch.device("cpu"), torch.inference_mode():
        timing = time_embeddings(base.load, (torch.tensor([1.0], dtype=torch.float32),))
        for index in range(50):
            checkpoint = reference if index in chosen else base
            prefix = f"transformer_blocks.{index}.adaln_proj.linear."
            weights = checkpoint.load_many((prefix + "weight", prefix + "bias"))
            table = adaln_table(
                timing,
                (1,),
                weights[prefix + "weight"],
                weights[prefix + "bias"],
                modalities=3,
                modulations=6,
            ).contiguous()
            if table.shape != (1, 3, 6, 5376) or not table.isfinite().all():
                raise ValueError("clean-context block modulation differs from H3")
            tables.append(table)
            del weights
            if progress is not None:
                progress(index + 1, 51)
        final = adaln_table(
            timing,
            (1,),
            base.load("norm_out.linear.weight"),
            base.load("norm_out.linear.bias"),
            modalities=1,
            modulations=2,
        ).contiguous()
        if final.shape != (1, 1, 2, 5376) or not final.isfinite().all():
            raise ValueError("clean-context output modulation differs from H3")
        tables.append(final)
        if progress is not None:
            progress(51, 51)
    return tuple(tables)
