from types import SimpleNamespace

import pytest

from vflash.native import h3_attention_lora as bridge
from vflash.native.h3_attention_lora import _block_residual as block_residual
from vflash.native.h3_attention_lora import _fp32_residual as fp32_residual

torch = pytest.importorskip("torch")
functional = torch.nn.functional


def test_fp32_residual_matches_peft_without_rounding_weights():
    torch.manual_seed(8)
    states = torch.randn(2, 3).bfloat16()
    base = torch.randn(2, 4).bfloat16()
    down, up = torch.randn(2, 3), torch.randn(4, 2)
    original = base.clone()
    for scale in (1, -1, 0):
        update = functional.linear(functional.linear(states.float(), down), up)
        expected = (base + update * scale).bfloat16()
        assert torch.equal(fp32_residual(base, states, down, up, scale), expected)
    assert torch.equal(original, base)
    assert fp32_residual(base, states, down, up, 0) is base


def test_row_chunks_preserve_batch_shape_and_fp32_residual():
    torch.manual_seed(9)
    states = torch.randn(2, 7, 3).bfloat16()
    base = torch.randn(2, 7, 4).bfloat16()
    down, up = torch.randn(2, 3), torch.randn(4, 2)
    expected = fp32_residual(base, states, down, up, -1)
    actual = fp32_residual(base, states, down, up, -1, chunk_rows=3)
    assert actual.shape == base.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class Block:
    def __init__(self):
        self.weights = SimpleNamespace(
            qkv=SimpleNamespace(values=torch.ones(4, 3).bfloat16()),
            attention_out=SimpleNamespace(values=torch.ones(4, 3).bfloat16()),
            ffn=SimpleNamespace(values=torch.ones(4, 3).bfloat16()),
            qkv_residuals=(),
            attention_out_residual=None,
            ffn_in_residual=None,
            ffn_out_residual=None,
        )

    def _linear(self, states, weight):
        return functional.linear(states, weight.values)


@pytest.mark.parametrize("override", (True, False))
def test_only_attention_is_changed_and_exception_restores(override):
    block = Block()
    native = block._linear
    if override:
        block._linear = native
    row = {name: (torch.ones(2, 3), torch.ones(4, 2)) for name in ("qkv", "attention_out")}
    calls = dict(qkv=0, attention_out=0)
    states = torch.ones(2, 3).bfloat16()
    with pytest.raises(RuntimeError, match="sentinel"), block_residual(block, row, -1, calls):
        assert (block._linear(states, block.weights.qkv) == -3).all()
        assert (block._linear(states, block.weights.attention_out) == -3).all()
        assert (block._linear(states, block.weights.ffn) == 3).all()
        raise RuntimeError("sentinel")
    assert calls == dict(qkv=1, attention_out=1)
    assert ("_linear" in vars(block)) == override
    assert torch.equal(
        block._linear(states, block.weights.qkv), native(states, block.weights.qkv)
    )


def test_does_not_stack_on_existing_native_adapter():
    block = Block()
    block.weights.attention_out_residual = object()
    with (
        pytest.raises(ValueError, match="adapter-free"),
        block_residual(block, {}, -1, dict(qkv=0, attention_out=0)),
    ):
        pytest.fail("must not enter")
    assert "_linear" not in vars(block)


class RingBlock(Block):
    def forward_prevalidated(self, states, invocation):
        return self._linear(states, self.weights.qkv)


class Ring:
    def __init__(self, count=4):
        self.host_blocks = [object() for _ in range(count)]
        self.slots = (RingBlock(), RingBlock())
        self.device = torch.device("cpu")

    def forward_prevalidated(self, states, invocation, *, checkpoint_blocks=frozenset()):
        output = []
        for index in range(len(self.host_blocks)):
            output.append(self.slots[index % 2].forward_prevalidated(states, invocation))
        return output, checkpoint_blocks


@pytest.mark.parametrize("scale", (-1, 0))
def test_ring_uses_logical_block_not_slot_and_resets_each_invocation(scale):
    from vflash.native.h3_attention_lora import _ring_residuals as ring_trunk_residuals

    ring = Ring()
    rows = [
        {
            name: (torch.ones(1, 3), torch.full((4, 1), float(i + 1)))
            for name in ("qkv", "attention_out")
        }
        for i in range(4)
    ]
    calls = dict(qkv=0, attention_out=0)
    states = torch.ones(2, 3).bfloat16()
    baseline, _ = ring.forward_prevalidated(states, object())
    with ring_trunk_residuals(ring, rows, scale, calls):
        for _ in range(2):
            output, checkpoints = ring.forward_prevalidated(
                states, object(), checkpoint_blocks=frozenset({3})
            )
            assert checkpoints == frozenset({3})
            for index, value in enumerate(output):
                assert torch.equal(value, baseline[index] + scale * 3 * (index + 1))
    assert calls == dict(qkv=8, attention_out=0)
    assert "forward_prevalidated" not in vars(ring)
    assert all("forward_prevalidated" not in vars(slot) for slot in ring.slots)
    assert all("_linear" not in vars(slot) for slot in ring.slots)
    after, _ = ring.forward_prevalidated(states, object())
    assert all(torch.equal(a, b) for a, b in zip(after, baseline, strict=True))


def test_ring_failure_restores_slot_and_denoiser_overrides():
    from vflash.native.h3_attention_lora import _ring_residuals as ring_trunk_residuals

    ring = Ring()
    rows = [
        {name: (torch.ones(1, 3), torch.ones(4, 1)) for name in ("qkv", "attention_out")}
        for _ in range(4)
    ]

    def fail(*args):
        raise RuntimeError("sentinel")

    ring.slots[1].forward_prevalidated = fail
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        ring_trunk_residuals(ring, rows, -1, dict(qkv=0, attention_out=0)),
    ):
        ring.forward_prevalidated(torch.ones(2, 3).bfloat16(), object())
    assert ring.slots[1].forward_prevalidated is fail
    assert "forward_prevalidated" not in vars(ring)
    assert all("_linear" not in vars(slot) for slot in ring.slots)


def state_fixture():
    shapes = {
        ("qkv_proj", "A"): (1, 5376),
        ("qkv_proj", "B"): (21504, 1),
        ("out_proj", "A"): (1, 7168),
        ("out_proj", "B"): (5376, 1),
    }
    modules = [f"blocks.{i}" for i in range(50)] + [
        f"token_refiner.blocks.{i}" for i in range(2)
    ]
    return {
        f"pipe.dit.{m}.attn.{p}.lora_{s}.default.weight": torch.ones(shape)
        for m in modules
        for (p, s), shape in shapes.items()
    }


def test_full_layout_and_interleaved_qkv_mapping():
    state = state_fixture()
    key = "pipe.dit.blocks.0.attn.qkv_proj.lora_B.default.weight"
    state[key] = torch.arange(21504, dtype=torch.float32).reshape(-1, 1)
    rows = bridge.prepare_dit_attention_lora(state, 1)
    assert len(rows) == 50
    actual = rows[0]["qkv"][1]
    assert actual[0] == 0 and actual[128] == 384
    assert actual[7168] == 128 and actual[14336] == 256
    assert state[key][128] == 128
    refiner = "pipe.dit.token_refiner.blocks.0.attn.out_proj.lora_B.default.weight"
    state[refiner] = torch.empty(0)
    with pytest.raises(ValueError, match="complete declared shape"):
        bridge.prepare_dit_attention_lora(state, 1)
    del state[refiner]
    with pytest.raises(ValueError, match="208"):
        bridge.prepare_dit_attention_lora(state, 1)


@pytest.mark.parametrize("scale", (float("nan"), float("inf"), True, 2, -2, "-1"))
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match="explicit adapter scale"):
        bridge._validate_scale(scale)


def test_runtime_is_explicit_exclusive_and_restored(monkeypatch):
    row = {name: (torch.ones(2, 3), torch.ones(4, 2)) for name in ("qkv", "attention_out")}
    monkeypatch.setattr(bridge, "prepare_dit_attention_lora", lambda state, rank: [row] * 50)
    runtime = SimpleNamespace(
        compute_capability=(8, 9),
        weight_residency="resident",
        parallel_strategy="single",
        overlay=SimpleNamespace(schedule=SimpleNamespace(nfe=16)),
        artifact=SimpleNamespace(weight_profile="minimax-h3-base", adapter_execution="none"),
        denoiser=SimpleNamespace(blocks=[Block() for _ in range(50)]),
    )
    with bridge.apply_dit_attention_lora(runtime, {}, rank=8, scale=-1) as calls:
        assert runtime._fp32_dit_adapter_active
        runtime.denoiser.blocks[0]._linear(
            torch.ones(1, 3).bfloat16(), runtime.denoiser.blocks[0].weights.qkv
        )
        assert calls["qkv"] == 1
        with (
            pytest.raises(ValueError, match="nested"),
            bridge.apply_dit_attention_lora(runtime, {}, rank=8, scale=-1),
        ):
            pytest.fail("nested adapter entered")
    assert not hasattr(runtime, "_fp32_dit_adapter_active")
    assert all("_linear" not in vars(b) for b in runtime.denoiser.blocks)
    runtime.compute_capability = (8, 6)
    with (
        pytest.raises(ValueError, match="single-SM89"),
        bridge.apply_dit_attention_lora(runtime, {}, rank=8, scale=-1),
    ):
        pytest.fail("unqualified architecture entered")
