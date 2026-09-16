from types import SimpleNamespace

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


def test_opt_in_ring_profile_reports_copy_wait_and_compute_without_changing_output(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the asynchronous weight-ring contract requires CUDA")

    class Slot:
        def __init__(self):
            self.weights = torch.zeros(1, device="cuda")

        def forward_prevalidated(self, value, _invocation):
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
    monkeypatch.setattr(denoiser, "_block_tensor_bytes", lambda _block: 4)

    ring.begin_denoise_profile()
    output = ring.forward_prevalidated(
        torch.ones(1, device="cuda"), SimpleNamespace(evaluation_index=3)
    )[0]
    torch.cuda.synchronize()
    profile = ring.finish_denoise_profile()

    assert output.item() == 210.0
    assert profile["device_index"] == 0
    assert profile["totals"]["copies"] == 4
    assert profile["totals"]["copied_bytes"] == 16
    assert profile["totals"]["collective_calls"] == 0
    evaluation = profile["evaluations"][0]
    assert evaluation["evaluation_index"] == 3
    assert len(evaluation["h2d_active_seconds"]) == 4
    assert len(evaluation["ready_wait_seconds"]) == 4
    assert len(evaluation["block_compute_seconds"]) == 4
    assert evaluation["compute_span_seconds"] >= 0


def test_ring_profile_aggregates_block_and_attention_phase_events():
    class Event:
        def __init__(self, milliseconds):
            self.milliseconds = milliseconds

        def elapsed_time(self, other):
            return other.milliseconds - self.milliseconds

    ring = denoiser.H3NativeDenoiserBF16Ring.__new__(denoiser.H3NativeDenoiserBF16Ring)
    ring.device = SimpleNamespace(index=2)
    ring._denoise_profile_records = [
        {
            "evaluation_index": 7,
            "copies": [(0, 16, Event(0), Event(10))],
            "ready_waits": [(0, Event(10), Event(12))],
            "block_compute": [(0, Event(12), Event(112))],
            "block_phases": [
                {
                    "block_index": 0,
                    "phases": [
                        ("qkv_projection", Event(20), Event(50)),
                        ("attention", Event(50), Event(90)),
                    ],
                    "attention_detail": {
                        "qkv_pack": (Event(50), Event(55)),
                        "flash_sdpa": [
                            (Event(60), Event(70)),
                            (Event(72), Event(82)),
                        ],
                    },
                    "block_detail": {
                        "ffn_input_projection": [(Event(90), Event(100))],
                        "ffn_input_silu_mul": [(Event(100), Event(102))],
                    },
                }
            ],
            "copy_span_start": Event(0),
            "copy_span_end": Event(10),
            "compute_span_start": Event(12),
            "compute_span_end": Event(112),
            "collective": {
                "calls": 2,
                "transmitted_bytes": 32,
                "issue_seconds": 0.01,
                "wait_seconds": 0.02,
            },
        }
    ]

    profile = ring.finish_denoise_profile()

    evaluation = profile["evaluations"][0]
    assert evaluation["profiled_blocks"] == 1
    assert evaluation["attention_chunks_profiled"] == 2
    assert evaluation["block_phase_seconds"] == {
        "qkv_projection": pytest.approx(0.03),
        "attention": pytest.approx(0.04),
    }
    assert evaluation["block_detail_seconds"] == {
        "ffn_input_projection": pytest.approx(0.01),
        "ffn_input_silu_mul": pytest.approx(0.002),
    }
    assert evaluation["attention_critical_path_seconds"] == {
        "qkv_pack": pytest.approx(0.005),
        "flash_sdpa": pytest.approx(0.02),
    }
    assert profile["phase_totals"] == {
        "profiled_blocks": 1,
        "block_phase_seconds": evaluation["block_phase_seconds"],
        "block_detail_seconds": evaluation["block_detail_seconds"],
        "attention_chunks_profiled": 2,
        "attention_critical_path_seconds": evaluation["attention_critical_path_seconds"],
    }


def test_profiled_block_helpers_preserve_projection_and_pointwise_order():
    instance = denoiser._H3BlockOperations.__new__(denoiser._H3BlockOperations)
    weight = object()
    adapter = object()
    instance.weights = SimpleNamespace(ffn_in=weight, ffn_in_residual=adapter)
    calls = []
    markers = iter(range(7))

    def adapted(states, selected_weight, selected_adapter):
        calls.append(("projection", states, selected_weight, selected_adapter))
        return "projected"

    def silu_mul(projected):
        calls.append(("silu", projected))
        return "activated"

    def gate_residual(residual, gate, projected):
        calls.append(("residual", residual, gate, projected))
        return "combined"

    instance._adapted_linear = adapted
    instance._silu_mul = silu_mul
    instance._gate_residual = gate_residual
    detail = {}

    assert (
        instance._profiled_ffn_input("normalized", detail=detail, event=lambda: next(markers))
        == "activated"
    )
    assert (
        instance._profiled_adapted_gate_residual(
            "states",
            weight,
            adapter,
            "residual",
            "gate",
            prefix="ffn_output",
            detail=detail,
            event=lambda: next(markers),
        )
        == "combined"
    )

    assert calls == [
        ("projection", "normalized", weight, adapter),
        ("silu", "projected"),
        ("projection", "states", weight, adapter),
        ("residual", "residual", "gate", "projected"),
    ]
    assert detail == {
        "ffn_input_projection": [(0, 1)],
        "ffn_input_silu_mul": [(1, 2)],
        "ffn_output_projection": [(3, 4)],
        "ffn_output_gate_residual": [(4, 5)],
    }
