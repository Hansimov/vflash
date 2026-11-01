from types import SimpleNamespace

import pytest

from vflash.native.cuda_memory import available_cuda_bytes


@pytest.mark.parametrize(
    "free,reserved,allocated,expected",
    [
        (4, 20, 8, 16),  # inactive reservation can serve the next audio pass
        (4, 20, 20, 4),  # active video decode cannot be counted as reusable
        (4, 20, 21, 4),  # changing allocator observations never add negative capacity
        (90, 20, 0, 96),  # clamp an asynchronous snapshot to the physical allocation
    ],
)
def test_available_capacity_preserves_live_and_foreign_allocations(
    free, reserved, allocated, expected
):
    gib = 1024**3
    cuda = SimpleNamespace(
        mem_get_info=lambda _: (free * gib, 96 * gib),
        memory_reserved=lambda _: reserved * gib,
        memory_allocated=lambda _: allocated * gib,
    )
    assert available_cuda_bytes(SimpleNamespace(cuda=cuda), "cuda:0") == expected * gib
