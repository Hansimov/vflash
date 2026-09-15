import pytest

from vflash.native import h3_native_denoiser as denoiser


def _cpu_block(value, *, residual_scaling=0.25):
    torch = pytest.importorskip("torch")

    def tensor(*shape):
        return torch.full(shape, float(value), dtype=torch.float32)

    def weight():
        return denoiser.H3BF16Weight(tensor(2, 2), None, 16, None, 2, 2)

    def residual():
        return denoiser.H3LowRankResidualWeights(tensor(1, 2), tensor(2, 1), residual_scaling)

    return denoiser.H3NativeBlockWeights(
        tensor(1, 6, 2),
        weight(),
        weight(),
        weight(),
        weight(),
        tensor(2),
        tensor(2),
        tensor(2),
        tensor(2),
        (residual(),),
        residual(),
        residual(),
        residual(),
    )


def test_prevalidated_copy_pairs_reuse_tensor_mapping():
    torch = pytest.importorskip("torch")
    destination = _cpu_block(0)
    source = _cpu_block(3)

    pairs = denoiser._bf16_block_copy_pairs(destination, source)
    assert isinstance(pairs, tuple)
    assert len(pairs) == 17

    denoiser._copy_bf16_tensor_pairs_(pairs)
    assert all(torch.equal(target, value) for target, value in pairs)

    # The cached mapping retains tensor identities, not snapshots, so later
    # source contents still reach the fixed destination without revalidation.
    source.adaln_table.fill_(7)
    denoiser._copy_bf16_tensor_pairs_(pairs)
    assert torch.equal(destination.adaln_table, source.adaln_table)


def test_prevalidated_copy_pairs_reject_residual_layout_drift():
    destination = _cpu_block(0, residual_scaling=0.5)
    source = _cpu_block(3, residual_scaling=0.25)

    with pytest.raises(denoiser.H3NativeDenoiserError, match="residual scaling differs"):
        denoiser._bf16_block_copy_pairs(destination, source)


@pytest.mark.parametrize(
    "order", [("ring", "ring"), ("serial", "serial"), ("ring", "serial"), ("serial", "ring")]
)
def test_back_to_back_invocations_preserve_inflight_weights(order):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the asynchronous weight-ring contract requires CUDA")

    class Slot:
        def __init__(self):
            self.weights = torch.zeros(1, device="cuda")

        def forward_prevalidated(self, value, _invocation):
            # Keep the consumer queued while the CPU submits the next request.
            torch.cuda._sleep(1_000_000)
            return value * self.weights

    ring = denoiser.H3NativeDenoiserBF16Ring.__new__(denoiser.H3NativeDenoiserBF16Ring)
    ring.device = torch.device("cuda:0")
    ring.host_blocks = tuple(
        torch.tensor([value], pin_memory=True) for value in (2.0, 3.0, 5.0, 7.0)
    )
    ring.slots = (Slot(), Slot())
    ring.copy_stream = torch.cuda.Stream()
    ring.ready_events = tuple(torch.cuda.Event() for _ in range(2))
    ring.compute_done_events = tuple(torch.cuda.Event() for _ in range(2))
    ring._ring_copy_pairs = tuple(
        ((ring.slots[index % 2].weights, host_block),)
        for index, host_block in enumerate(ring.host_blocks)
    )
    ring._serial_copy_pairs = tuple(
        ((ring.slots[0].weights, host_block),) for host_block in ring.host_blocks
    )
    methods = {
        "ring": ring.forward_prevalidated,
        "serial": ring.forward_prevalidated_serial,
    }
    value = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    outputs = [methods[mode](value, None)[0] for mode in order]
    torch.cuda.synchronize()

    assert [output.item() for output in outputs] == [210.0, 210.0]
