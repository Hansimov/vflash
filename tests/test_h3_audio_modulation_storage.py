import numpy as np
import pytest

from vflash.native.h3_audio_modulation_storage import pack_table, unpack_table
from vflash.native.h3_refinement_schedule import refinement_schedule

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("steps,denoise", [(6, 0.5), (12, 0.9), (20, 0.95)])
@pytest.mark.parametrize("final", [False, True])
def test_compact_storage_restores_every_bit_including_unconsumed_rows(steps, denoise, final):
    schedule = refinement_schedule(steps=steps, denoise=denoise)
    shape = (schedule.nfe, 3 if final else 9, 2 if final else 6, 7)
    table = torch.randn(shape).bfloat16()
    table[2:, 0] = table[1, 0]
    # Preserve signed zeros and arbitrary exponent bits; the table loader owns finite checks.
    table.view(torch.int16)[0, 0, 0, :3] = torch.tensor([-32768, 32767, -1], dtype=torch.int16)
    threads = torch.get_num_threads()
    packed = pack_table(table)
    restored = unpack_table(packed, schedule, final=final)
    assert np.array_equal(table.view(torch.int16).numpy(), restored.view(torch.int16).numpy())
    assert sum(v.numel() * v.element_size() for v in packed.values()) < table.numel() * 2
    assert torch.get_num_threads() == threads
    packed["adaln.indices"][0, 0] = packed["adaln.rows"].shape[0]
    with pytest.raises(ValueError, match="row index"):
        unpack_table(packed, schedule, final=final)
