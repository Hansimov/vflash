import pytest

from vflash.native import h3_native_denoiser as denoiser


@pytest.mark.parametrize(
    "order", [("ring", "ring"), ("serial", "serial"), ("ring", "serial"), ("serial", "ring")]
)
def test_back_to_back_invocations_preserve_inflight_weights(monkeypatch, order):
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
    monkeypatch.setattr(
        denoiser, "_copy_bf16_block_", lambda dst, src: dst.copy_(src, non_blocking=True)
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
