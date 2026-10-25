"""Deduplicate identical complete BF16 modulation rows, retaining every table bit."""

import numpy as np


def pack_table(table):
    import torch

    if table.device.type != "cpu" or table.dtype != torch.bfloat16 or table.ndim != 4:
        raise ValueError("compact modulation requires a complete CPU BF16 table")
    rows = table.contiguous().view(torch.int16).numpy().reshape(-1, *table.shape[2:])
    unique, indices, lookup = [], [], {}
    for row in rows:
        key = row.tobytes()
        index = lookup.get(key)
        if index is None:
            index = len(unique)
            lookup[key] = index
            unique.append(row)
        indices.append(index)
    return {
        "adaln.rows": torch.from_numpy(np.stack(unique)).view(torch.bfloat16),
        "adaln.indices": torch.tensor(indices, dtype=torch.int32).reshape(table.shape[:2]),
    }


def unpack_table(values, schedule, *, final=False):
    import torch

    if set(values) != {"adaln.rows", "adaln.indices"}:
        raise ValueError("compact modulation inventory differs")
    rows, indices = values["adaln.rows"], values["adaln.indices"]
    shape = (schedule.nfe, 3 if final else 9)
    if (
        rows.device.type != "cpu"
        or rows.dtype != torch.bfloat16
        or rows.ndim != 3
        or rows.shape[1] != (2 if final else 6)
        or rows.shape[2] <= 0
        or not 0 < rows.shape[0] <= shape[0] * shape[1]
        or indices.device.type != "cpu"
        or indices.dtype != torch.int32
        or tuple(indices.shape) != shape
    ):
        raise ValueError("compact modulation geometry differs")
    offsets = indices.numpy()
    if offsets.min() < 0 or offsets.max() >= rows.shape[0]:
        raise ValueError("compact modulation row index differs")
    # Exact CPU bit copies preserve the original full tensor and caller's thread pool.
    bits = rows.view(torch.int16).numpy()[offsets]
    return torch.from_numpy(bits).view(torch.bfloat16)
