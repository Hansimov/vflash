"""Two-device BF16 execution with weight TP or sequence/head partitioning.

One process owns both CUDA contexts, a shared host store and two device rings.
The two threads issue work on separate devices; NCCL exchanges only activations.
No framework model graph or distributed launcher is required.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import replace
from datetime import timedelta
from traceback import clear_frames
from typing import Any

from vflash.native.h3_native_denoiser import (
    H3NativeBlockBF16Resident,
    H3NativeBlockWeights,
    H3NativeDenoiserBF16Ring,
    H3NativeDenoiserError,
    _pin_bf16_block,
    _torch,
    load_h3_native_block,
)
from vflash.native.h3_pinned_arena import PinnedHostArena
from vflash.native.h3_runtime_artifact import H3RuntimeArtifact
from vflash.native.h3_tensor_file import H3MappedSafetensor


def _split_matrix(
    tensor: Any, rank: int, *, axis: int, parts: int = 1, pin_memory: bool = False
) -> Any:
    """Copy a compact shard, preserving Q/K/V and SiLU/value branch ordering."""
    torch = _torch()
    if rank not in {0, 1} or tensor.ndim != 2 or axis not in {0, 1}:
        raise H3NativeDenoiserError("invalid two-device matrix partition")
    if parts < 1 or tensor.shape[axis] % (2 * parts):
        raise H3NativeDenoiserError("matrix branches cannot be divided into two shards")
    shape = list(tensor.shape)
    shape[axis] //= 2
    output = torch.empty(shape, dtype=tensor.dtype, device=tensor.device, pin_memory=pin_memory)
    width = tensor.shape[axis] // parts
    for part in range(parts):
        source = tensor.narrow(axis, part * width + rank * width // 2, width // 2)
        output.narrow(axis, part * width // 2, width // 2).copy_(source)
    return output


def shard_h3_block(
    weights: H3NativeBlockWeights, rank: int, *, pin_memory: bool = False
) -> H3NativeBlockWeights:
    """Shard all four trunk projections and their six native LoRA branches."""

    def column(weight: Any, parts: int = 1) -> Any:
        values = _split_matrix(weight.values, rank, axis=0, parts=parts, pin_memory=pin_memory)
        return replace(weight, values=values, output_features=values.shape[0])

    def row(weight: Any) -> Any:
        values = _split_matrix(weight.values, rank, axis=1, pin_memory=pin_memory)
        return replace(weight, values=values, input_features=values.shape[1])

    def column_adapter(adapter: Any, parts: int = 1) -> Any:
        if adapter is None:
            return None
        up = _split_matrix(adapter.up, rank, axis=0, parts=parts, pin_memory=pin_memory)
        return replace(adapter, up=up)

    def row_adapter(adapter: Any) -> Any:
        if adapter is None:
            return None
        down = _split_matrix(adapter.down, rank, axis=1, pin_memory=pin_memory)
        return replace(adapter, down=down)

    return replace(
        weights,
        qkv=column(weights.qkv, 3),
        attention_out=row(weights.attention_out),
        ffn_in=column(weights.ffn_in, 2),
        ffn_out=row(weights.ffn_out),
        qkv_residuals=tuple(column_adapter(r) for r in weights.qkv_residuals),
        attention_out_residual=row_adapter(weights.attention_out_residual),
        ffn_in_residual=column_adapter(weights.ffn_in_residual, 2),
        ffn_out_residual=row_adapter(weights.ffn_out_residual),
    )


def _pack_qkv(query: Any, key: Any, value: Any, *, chunks: int = 1) -> Any:
    torch = _torch()
    batch, rows, heads, width = query.shape
    return (
        torch.stack((query, key, value), dim=0)
        .reshape(3, batch, rows, 2, chunks, heads // (2 * chunks), width)
        .permute(4, 3, 0, 1, 2, 5, 6)
        .contiguous()
    )


def _unpack_qkv(received: Any, total_rows: int) -> tuple[Any, Any, Any]:
    _, _, batch, rows, heads, width = received.shape
    full = received.permute(1, 2, 0, 3, 4, 5).reshape(3, batch, rows * 2, heads, width)
    return tuple(tensor[:, :total_rows] for tensor in full.unbind(0))


def _pack_attention(attention: Any, padded_rows: int) -> Any:
    import torch.nn.functional as functional

    batch, rows, heads, width = attention.shape
    if rows != padded_rows:
        attention = functional.pad(attention, (0, 0, 0, 0, 0, padded_rows - rows))
    return (
        attention.reshape(batch, 2, padded_rows // 2, heads, width)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def _unpack_attention(received: Any) -> Any:
    _, batch, rows, heads, width = received.shape
    return received.permute(1, 2, 0, 3, 4).reshape(batch, rows, heads * 2, width).contiguous()


class _SequenceHeadBlock(H3NativeBlockBF16Resident):
    """Full-width GEMMs on token shards; exact attention on complete head shards."""

    group: Any
    total_rows: int

    def _attention(self, query: Any, key: Any, value: Any) -> Any:
        torch = _torch()
        chunks = 4
        qkv_send = _pack_qkv(query, key, value, chunks=chunks)
        qkv_receive = torch.empty_like(qkv_send)
        # Queue QKV transfers before inserting any attention dependency into
        # the compute stream. Later head groups can transfer while an earlier
        # group computes; the return exchanges overlap subsequent attention.
        inbound = [
            self.group.alltoall_base(qkv_receive[index], qkv_send[index], [], [])
            for index in range(chunks)
        ]
        outbound, sends, receives = [], [], []
        for index in range(chunks):
            inbound[index].wait()
            query, key, value = _unpack_qkv(qkv_receive[index], self.total_rows)
            # A padded token is never admitted to the attention softmax.
            attention = super()._attention(query, key, value)
            sent = _pack_attention(attention, qkv_receive.shape[4] * 2)
            received = torch.empty_like(sent)
            sends.append(sent)
            receives.append(received)
            outbound.append(self.group.alltoall_base(received, sent, [], []))
        for work in outbound:
            work.wait()
        return _unpack_attention(torch.cat(receives, dim=3))


class _TensorBlock(H3NativeBlockBF16Resident):
    """Column/row tensor parallelism including the low-rank residual GEMMs."""

    group: Any

    def _adapted_gate_residual(
        self, states: Any, weight: Any, adapter: Any, residual: Any, gate: Any
    ) -> Any:
        import torch.nn.functional as functional

        base = self._linear(states, weight)
        down = functional.linear(states, adapter.down) if adapter is not None else None
        base_work = self.group.allreduce([base])
        if down is not None:
            # Reduce before the replicated up projection. Reducing separate
            # full-width LoRA outputs would add rounding and communication.
            self.group.allreduce([down]).wait()
            update = functional.linear(down, adapter.up)
        base_work.wait()
        if adapter is None:
            return self._gate_residual(residual, gate, base)
        from vflash.native.h3_fused_ops import triton_strict_bf16_adapter_gate_residual

        return triton_strict_bf16_adapter_gate_residual(
            base,
            update,
            residual,
            gate,
            scaling=adapter.scaling,
            block_size=self.elementwise_block_size,
        )


class _SequenceRing(H3NativeDenoiserBF16Ring):
    block_type = _SequenceHeadBlock


class _TensorRing(H3NativeDenoiserBF16Ring):
    block_type = _TensorBlock


class _DevicePair:
    """Own the two NCCL ranks without global torch.distributed process state."""

    def __init__(self, devices: tuple[Any, Any]) -> None:
        import torch.distributed as distributed

        self.devices = devices
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vflash-rank")
        self.closed = False
        store = distributed.HashStore()

        def create(rank: int) -> Any:
            _torch().cuda.set_device(devices[rank])
            options = distributed.ProcessGroupNCCL.Options()
            options._timeout = timedelta(seconds=120)
            return distributed.ProcessGroupNCCL(store, rank, 2, options)

        futures = [self.pool.submit(create, rank) for rank in (0, 1)]
        try:
            self.groups = tuple(future.result() for future in futures)
        except BaseException:
            self.closed = True
            self.pool.shutdown(wait=True)
            for future in futures:
                if future.exception() is None:
                    with suppress(Exception):
                        future.result().abort()
            raise

    def run(self, action: Callable[[int], Any]) -> list[Any]:
        if self.closed:
            raise H3NativeDenoiserError("the two-device execution lane is closed")

        def invoke(rank: int) -> Any:
            torch = _torch()
            torch.cuda.set_device(self.devices[rank])
            with torch.inference_mode():
                return action(rank)

        futures = [self.pool.submit(invoke, rank) for rank in (0, 1)]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        if any(future.exception() is not None for future in done):
            # Unblock the peer collective. A failed lane is never reused.
            self.closed = True
            for group in self.groups:
                with suppress(Exception):
                    group.abort()
            self.pool.shutdown(wait=True, cancel_futures=True)
        return [future.result() for future in futures]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            for group in self.groups:
                group.shutdown()
        finally:
            self.pool.shutdown(wait=True)


def _partition_invocation(
    states: Any, invocation: Any, devices: tuple[Any, Any], *, sequence: bool
) -> list[tuple[Any, Any]]:
    torch = _torch()
    import torch.nn.functional as functional

    total = states.shape[1]
    rows = (total + 1) // 2 if sequence else total
    values = []
    for rank, device in enumerate(devices):
        begin = rank * rows if sequence else 0
        end = min(total, begin + rows)
        shard = states[:, begin:end].to(device).contiguous()
        indices = invocation.adaln_indices[begin:end].to(device)
        cos = invocation.rotary_cos[begin:end].to(device)
        sin = invocation.rotary_sin[begin:end].to(device)
        padding = rows - shard.shape[1]
        if padding:
            shard = functional.pad(shard, (0, 0, 0, padding))
            indices = functional.pad(indices, (0, padding))
            cos = functional.pad(cos, (0, 0, 0, padding))
            sin = functional.pad(sin, (0, 0, 0, padding))
        values.append(
            (
                shard,
                replace(
                    invocation,
                    sequence_length=rows,
                    device=torch.device(device),
                    adaln_indices=indices,
                    rotary_cos=cos,
                    rotary_sin=sin,
                ),
            )
        )
    return values


class H3NativeDenoiserParallel:
    """Two SM86 devices cooperatively execute one complete native request."""

    backend_id = "cuda-bf16-two-device-nccl-block-ring-v1"

    @classmethod
    def load(
        cls,
        artifact: H3RuntimeArtifact,
        *,
        devices: tuple[Any, Any],
        strategy: str,
        adaln_table_load: Callable[[int], Any] | None = None,
    ) -> H3NativeDenoiserParallel:
        torch = _torch()
        devices = tuple(torch.device(device) for device in devices)
        if (
            strategy not in {"tensor", "sequence-head"}
            or len(devices) != 2
            or devices[0] == devices[1]
            or artifact.target.compute_capability != "sm86"
            or any(torch.cuda.get_device_capability(device) != (8, 6) for device in devices)
            or artifact.spec.num_attention_heads % 2
            or (strategy == "sequence-head" and artifact.spec.num_attention_heads % 8)
            or artifact.spec.ffn_dim % 2
        ):
            raise H3NativeDenoiserError("parallel execution requires two distinct SM86 devices")
        host_blocks: list[list[H3NativeBlockWeights]] = [[], []]
        with PinnedHostArena() as arena:
            for row in artifact.blocks:
                with H3MappedSafetensor(artifact.directory / row.path) as mapped:
                    weights = None
                    try:
                        _, weights = load_h3_native_block(
                            artifact,
                            row.index,
                            adaln_table=(
                                adaln_table_load(row.index) if adaln_table_load else None
                            ),
                            tensor_load=mapped.load,
                        )
                        if strategy == "tensor":
                            for rank in (0, 1):
                                # Pre-pinning individual shards would reintroduce
                                # the allocator's per-tensor power-of-two waste.
                                host_blocks[rank].append(
                                    _pin_bf16_block(
                                        shard_h3_block(weights, rank), pin=arena.copy
                                    )
                                )
                        else:
                            pinned = _pin_bf16_block(weights, pin=arena.copy)
                            host_blocks[0].append(pinned)
                            host_blocks[1].append(pinned)
                    except BaseException as exc:
                        host_blocks[0].clear()
                        host_blocks[1].clear()
                        pinned = None
                        clear_frames(exc.__traceback__)
                        raise
                    finally:
                        weights = None
        ring_artifact = artifact
        if strategy == "tensor":
            ring_artifact = replace(
                artifact,
                spec=replace(
                    artifact.spec,
                    num_attention_heads=artifact.spec.num_attention_heads // 2,
                    ffn_dim=artifact.spec.ffn_dim // 2,
                ),
            )
        pair = None
        self = None
        ring_type = _TensorRing if strategy == "tensor" else _SequenceRing
        rings = []
        try:
            pair = _DevicePair(devices)
            # A failed generator expression can keep its call arguments alive
            # through the traceback even after clear_frames(). Use the same
            # ordered construction with explicit owners that can be released.
            for rank, device in enumerate(devices):
                rings.append(ring_type(ring_artifact, tuple(host_blocks[rank]), device=device))
            for rank, ring in enumerate(rings):
                for slot in ring.slots:
                    slot.group = pair.groups[rank]
            self = cls()
            self.artifact = artifact
            self.strategy = strategy
            self.devices = devices
            self.device = devices[0]
            self.rings = tuple(rings)
            self.pair = pair
            self.host_weight_bytes = (
                sum(ring.host_weight_bytes for ring in rings)
                if strategy == "tensor"
                else rings[0].host_weight_bytes
            )
            return self
        except BaseException as exc:
            try:
                if pair is not None:
                    pair.close()
            finally:
                rings.clear()
                host_blocks.clear()
                self = pinned = ring = slot = None
                clear_frames(exc.__traceback__)
            raise

    def prepare_invocation(self, states: Any, **kwargs: Any) -> Any:
        return self.rings[0].prepare_invocation(states, **kwargs)

    def forward_prevalidated(self, states: Any, invocation: Any) -> tuple[Any, dict[int, Any]]:
        torch = _torch()
        sequence = self.strategy == "sequence-head"
        shards = _partition_invocation(states, invocation, self.devices, sequence=sequence)

        def run(rank: int) -> Any:
            if sequence:
                for slot in self.rings[rank].slots:
                    slot.total_rows = states.shape[1]
            return self.rings[rank].forward_prevalidated(*shards[rank])[0]

        outputs = self.pair.run(run)
        if sequence:
            return torch.cat((outputs[0], outputs[1].to(self.device)), dim=1)[
                :, : states.shape[1]
            ], {}
        # The secondary rank's last gate is part of the request boundary too.
        completed = torch.cuda.Event()
        completed.record(torch.cuda.current_stream(self.devices[1]))
        torch.cuda.current_stream(self.device).wait_event(completed)
        return outputs[0], {}

    def close(self) -> None:
        self.pair.close()
