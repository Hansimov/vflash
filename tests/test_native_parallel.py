import pytest

torch = pytest.importorskip("torch")

from vflash.native.h3_native_denoiser import (  # noqa: E402
    H3BF16Weight,
    H3LowRankResidualWeights,
    H3NativeBlockWeights,
)
from vflash.native.h3_parallel import (  # noqa: E402
    _pack_attention,
    _pack_qkv,
    _unpack_attention,
    _unpack_qkv,
    shard_h3_block,
)


def _weights():
    generator = torch.Generator().manual_seed(18)

    def tensor(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64)

    def weight(inputs, outputs):
        return H3BF16Weight(tensor(outputs, inputs), torch.ones(1), 16, None, inputs, outputs)

    def adapter(inputs, outputs):
        return H3LowRankResidualWeights(tensor(2, inputs), tensor(outputs, 2), 0.0625)

    return H3NativeBlockWeights(
        tensor(4, 6, 6),
        weight(6, 24),
        weight(8, 6),
        weight(6, 24),
        weight(12, 6),
        tensor(6),
        tensor(6),
        tensor(2),
        tensor(2),
        tuple(adapter(6, 8) for _ in range(3)),
        adapter(8, 6),
        adapter(6, 24),
        adapter(12, 6),
    )


def test_tensor_shards_preserve_branch_and_lora_order():
    weights = _weights()
    shards = [shard_h3_block(weights, rank) for rank in (0, 1)]
    for name, branches in (("qkv", 3), ("ffn_in", 2)):
        values = [getattr(shard, name).values for shard in shards]
        restored = torch.cat([value.reshape(branches, -1, 6) for value in values], dim=1)
        assert torch.equal(restored.reshape(-1, 6), getattr(weights, name).values)
    for name in ("attention_out", "ffn_out"):
        assert torch.equal(
            torch.cat([getattr(shard, name).values for shard in shards], dim=1),
            getattr(weights, name).values,
        )
    for branch in range(3):
        assert torch.equal(
            torch.cat([shard.qkv_residuals[branch].up for shard in shards]),
            weights.qkv_residuals[branch].up,
        )
    packed = torch.cat([shard.ffn_in_residual.up.reshape(2, -1, 2) for shard in shards], dim=1)
    assert torch.equal(packed.reshape(-1, 2), weights.ffn_in_residual.up)


@pytest.mark.parametrize("name,inputs", [("attention_out", 8), ("ffn_out", 12)])
def test_row_parallel_lora_reduces_down_projection_before_up(name, inputs):
    weights = _weights()
    shards = [shard_h3_block(weights, rank) for rank in (0, 1)]
    states = torch.arange(inputs * 3, dtype=torch.float64).reshape(3, inputs) / 32
    partials, down = [], []
    for rank, shard in enumerate(shards):
        x = states.chunk(2, dim=-1)[rank]
        partials.append(x @ getattr(shard, name).values.T)
        down.append(x @ getattr(shard, name + "_residual").down.T)
    adapter = getattr(weights, name + "_residual")
    actual = sum(partials) + (sum(down) @ adapter.up.T) * adapter.scaling
    expected = states @ getattr(weights, name).values.T
    expected += ((states @ adapter.down.T) @ adapter.up.T) * adapter.scaling
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rows", [7, 8])
@pytest.mark.parametrize("chunks", [1, 4])
def test_sequence_head_exchange_restores_tokens_and_excludes_padding(rows, chunks):
    generator = torch.Generator().manual_seed(91)
    tensors = [torch.randn((1, rows, 8, 2), generator=generator) for _ in range(3)]
    local_rows = (rows + 1) // 2
    padded = [
        torch.nn.functional.pad(t, (0, 0, 0, 0, 0, local_rows * 2 - rows)) for t in tensors
    ]
    sent = [
        _pack_qkv(
            *(t[:, rank * local_rows : (rank + 1) * local_rows] for t in padded), chunks=chunks
        )
        for rank in (0, 1)
    ]
    attention_sent = [[], []]
    heads = 4 // chunks
    for rank in (0, 1):
        for chunk in range(chunks):
            received = torch.stack([source[chunk, rank] for source in sent])
            query, key, value = _unpack_qkv(received, rows)
            begin = rank * 4 + chunk * heads
            for actual, full in zip((query, key, value), tensors, strict=True):
                assert torch.equal(actual, full[:, :, begin : begin + heads])
            assert key.shape[1] == rows
            attention = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            ).transpose(1, 2)
            attention_sent[rank].append(_pack_attention(attention, local_rows * 2))
    restored = []
    for rank in (0, 1):
        received_chunks = [
            torch.stack([source[chunk][rank] for source in attention_sent])
            for chunk in range(chunks)
        ]
        restored.append(_unpack_attention(torch.cat(received_chunks, dim=3)))
    actual = torch.cat(restored, dim=1)[:, :rows]
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(tensor.transpose(1, 2) for tensor in tensors)
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_failed_rank_aborts_waiting_peer_and_rejects_reuse(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    from vflash.native import h3_parallel
    from vflash.native.h3_native_denoiser import H3NativeDenoiserError

    released = Event()
    entered = Event()
    aborted = []

    class Group:
        def __init__(self, rank):
            self.rank = rank

        def abort(self):
            aborted.append(self.rank)
            released.set()

    pair = h3_parallel._DevicePair.__new__(h3_parallel._DevicePair)
    pair.devices = (0, 1)
    pair.groups = (Group(0), Group(1))
    pair.pool = ThreadPoolExecutor(max_workers=2)
    pair.closed = False
    monkeypatch.setattr(
        h3_parallel,
        "_torch",
        lambda: SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda device: None),
            inference_mode=torch.inference_mode,
        ),
    )

    def action(rank):
        if rank == 0:
            assert entered.wait(timeout=2)
            raise RuntimeError("original rank failure")
        entered.set()
        assert released.wait(timeout=2), "peer collective was not aborted"
        return None

    with pytest.raises(RuntimeError, match="original rank failure"):
        pair.run(action)
    assert aborted == [0, 1]
    with pytest.raises(H3NativeDenoiserError, match="lane is closed"):
        pair.run(action)
    pair.close()
