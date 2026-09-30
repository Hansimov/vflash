"""Compose the released H3 spatial tiles without losing overlap contributors.

The decoder, tile plan, local position coordinates and temporal path are unchanged.
Sequential crossfades must use already-composited neighbours: raw neighbours lose
the diagonal contribution where horizontal and vertical overlaps intersect.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


def compose_tiles(
    rows: list[list[Any]], y_overlap: list[int], x_overlap: list[int], blend: Any
) -> Any:
    """Blend complete horizontal strips before composing them vertically.

    This preserves all contributors for triple overlaps too, using sequential
    crossfade weights, not an order-independent normalized overlap-add scheme.
    The incoming tiles are never modified.
    """
    import torch

    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("a nonempty rectangular tile grid is required")
    if len(y_overlap) != len(rows) - 1 or len(x_overlap) != len(rows[0]) - 1:
        raise ValueError("overlap plan differs from the tile grid")

    def append(previous: Any, incoming: Any, extent: int, dim: int) -> Any:
        if not 0 < extent < min(previous.shape[dim], incoming.shape[dim]):
            raise ValueError("positive overlap smaller than both tiles required")
        prefix = previous.narrow(dim, 0, previous.shape[dim] - extent)
        return torch.cat((prefix, blend(previous, incoming, extent, dim)), dim=dim)

    canvas = None
    for row_index, row in enumerate(rows):
        strip = row[0]
        for index, tile in enumerate(row[1:]):
            strip = append(strip, tile, x_overlap[index], -1)
        canvas = (
            strip if canvas is None else append(canvas, strip, y_overlap[row_index - 1], -2)
        )
    return canvas


def _decode_tiles(model: Any, latent: Any) -> Any:
    if model.training:
        raise ValueError("the Vflash media decoder requires inference mode")
    if model.parallel_tiling:
        # The pinned wrapper sets parallel_tiling even for its one-rank group.
        state = model._local_tile_indices.__func__.__globals__["get_parallel_state"]()
        if state["sp_size"] != 1 or state["sp_rank"] != 0:
            raise ValueError("the Vflash media decoder requires one spatial decode rank")
    height, width = (value * model.vae_ratio for value in latent.shape[-2:])
    ys, heights, yo = model.split_tiles(height, True)
    xs, widths, xo = model.split_tiles(width, True)
    tiles = [
        latent[
            ...,
            y // model.vae_ratio : (y + h) // model.vae_ratio,
            x // model.vae_ratio : (x + w) // model.vae_ratio,
        ]
        for y, h in zip(ys, heights, strict=True)
        for x, w in zip(xs, widths, strict=True)
    ]
    indices = model._local_tile_indices(len(tiles), 0, 1)
    decoded = model._run_tile_tasks(tiles, indices, model.decode, model.stack_tiling)
    rows = [
        [decoded[i * len(xs) + j].to(latent.device) for j in range(len(xs))]
        for i in range(len(ys))
    ]
    result = compose_tiles(rows, yo, xo, model.blend)
    if result.shape[-2:] != (height, width):
        raise ValueError("composited canvas differs from the latent geometry")
    return result


def install_spatial_composition(model: Any) -> None:
    """Bind the inference compositor to this decoder instance, not its class.

    The media stage owns one device even when the denoiser uses several devices.
    Repeated CPU/CUDA preparations keep the same instance-local implementation.
    """
    model.tiled_decode = MethodType(_decode_tiles, model)
