"""Bounded batches of existing H3 decoder tiles, scoped to one owned instance.

Tile coordinates, temporal state and overlap composition remain unchanged.
Batch-dependent convolution rounding can change decoded pixels.
"""

from contextlib import contextmanager
from types import MethodType


@contextmanager
def batch_spatial_tiles(model, limit):
    if type(limit) is not int or limit not in (1, 2, 4):
        raise ValueError("expected a bounded tile batch")
    if model.training or getattr(model, "_vflash_tile_batch_owner", False):
        raise ValueError("an exclusively owned inference decoder is required")
    report = {"limit": limit, "groups": [], "tiles": 0, "fallback_calls": 0}
    original = model._run_tile_tasks
    missing = object()
    previous = vars(model).get("_run_tile_tasks", missing)

    def run(self, tiles, indices, forward_fn, stack_tiling, cls_agg=None):
        if cls_agg is not None or stack_tiling or limit == 1:
            report["fallback_calls"] += 1
            return original(tiles, indices, forward_fn, stack_tiling, cls_agg)
        output, group = [], []

        def flush():
            if not group:
                return
            output.extend(original(tiles, group, forward_fn, True))
            report["groups"].append(len(group))
            report["tiles"] += len(group)
            group.clear()

        for index in indices:
            if group and (len(group) == limit or tiles[index].shape != tiles[group[0]].shape):
                flush()
            group.append(index)
        flush()
        return output

    model._vflash_tile_batch_owner = True
    model._run_tile_tasks = MethodType(run, model)
    try:
        yield report
    finally:
        if previous is missing:
            del model._run_tile_tasks
        else:
            model._run_tile_tasks = previous
        del model._vflash_tile_batch_owner
