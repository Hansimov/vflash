"""CPU-only contract tests; run unchanged in the native Torch runtime image."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import tempfile
import unittest
import weakref
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from functools import partial
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import pytest

from vflash.native import h3_native_denoiser as denoiser
from vflash.native import h3_parallel as parallel
from vflash.native.h3_artifact_contract import H3Spec, resolve_h3_artifact_target
from vflash.native.h3_pinned_arena import PinnedHostArena
from vflash.native.h3_runtime_artifact import H3RuntimeArtifact, H3RuntimeArtifactBlock
from vflash.native.h3_tensor_file import H3MappedSafetensor, save_safetensors_atomic

torch = pytest.importorskip("torch")


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def bytes_of(value):
    return value.contiguous().view(torch.uint8).reshape(-1)


def fixture(directory: Path, count: int = 3):
    spec = H3Spec(count, 8, 8, 2, 12, 4, 2, 2, (1, 2, 2))
    shapes = {
        "adaln.table": (4, 1, 6, 8),
        "attn.qkv.weight": (48, 8),
        "attn.qkv.scale": (48,),
        "attn.out.weight": (8, 16),
        "attn.out.scale": (8,),
        "ffn.in.weight": (24, 8),
        "ffn.in.scale": (24,),
        "ffn.out.weight": (8, 12),
        "ffn.out.scale": (8,),
        "norm.attn.weight": (8,),
        "norm.ffn.weight": (8,),
        "attn.q_norm.weight": (2,),
        "attn.k_norm.weight": (2,),
    }
    for role, in_features, out_features in (
        ("attn.q", 8, 16),
        ("attn.k", 8, 16),
        ("attn.v", 8, 16),
        ("attn.out", 16, 8),
        ("ffn.in", 8, 24),
        ("ffn.out", 12, 8),
    ):
        shapes[f"adapter.{role}.down"] = (2, in_features)
        shapes[f"adapter.{role}.up"] = (out_features, 2)
    rows = []
    for index in range(count):
        values = {}
        for sequence, (name, shape) in enumerate(shapes.items()):
            size = 1
            for dim in shape:
                size *= dim
            values[name] = (
                ((torch.arange(size) % 43) / 8 + sequence + index)
                .to(torch.bfloat16)
                .reshape(shape)
            )
        path = directory / f"block-{index}.safetensors"
        save_safetensors_atomic(path, values)
        rows.append(
            H3RuntimeArtifactBlock(
                index,
                path.name,
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                1,
                tuple(shapes),
            )
        )
    return H3RuntimeArtifact(
        directory,
        "h3-runtime-arena-cpu",
        "cpu-only",
        "ready",
        "row-major-reference-v1",
        resolve_h3_artifact_target("rtx3080-20g-sm86-bf16-block-ring"),
        spec,
        4,
        "test-fixture",
        {},
        tuple(rows),
        adapter_execution="runtime-residual",
    )


class CpuRing(denoiser.H3NativeDenoiserBF16Ring):
    """Replace only GPU construction; inherit the actual host loader unchanged."""

    def __init__(self, artifact, host_blocks, *, device, **kwargs):
        self.artifact = artifact
        self.host_blocks = host_blocks
        self.device = device
        self.slots = ()
        self.host_weight_bytes = sum(denoiser._block_tensor_bytes(row) for row in host_blocks)


class CpuPair:
    def __init__(self, devices):
        self.groups = (None, None)
        self.closed = False

    def close(self):
        self.closed = True


class ArenaTests(unittest.TestCase):
    def assert_tensor_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(bytes_of(actual), bytes_of(expected)))

    def assert_weights_equal(self, actual, expected):
        actual_tensors, expected_tensors = list(tensors(actual)), list(tensors(expected))
        self.assertEqual(len(actual_tensors), len(expected_tensors))
        for left, right in zip(actual_tensors, expected_tensors, strict=True):
            self.assert_tensor_equal(left, right)
        for name in ("qkv", "attention_out", "ffn_in", "ffn_out"):
            for key in ("bits", "group_size", "input_features", "output_features"):
                self.assertEqual(
                    getattr(getattr(actual, name), key), getattr(getattr(expected, name), key)
                )
        for left, right in zip(actual.qkv_residuals, expected.qkv_residuals, strict=True):
            self.assertEqual(left.scaling, right.scaling)

    def test_dtype_alignment_contiguity_and_byte_identity(self):
        arena = PinnedHostArena(4096, pin_memory=False)
        inputs = [
            torch.arange(6).to(torch.bfloat16).reshape(2, 3),
            torch.arange(15).to(torch.float16).reshape(3, 5),
            torch.tensor([1.25, -2.75, float("inf"), float("nan")]),
            torch.tensor([2**61 + 13, -(2**60)], dtype=torch.int64),
            torch.arange(20, dtype=torch.float64).reshape(4, 5).t(),
            torch.tensor([True, False, True]),
        ]
        outputs = [arena.copy(source) for source in inputs]
        storage = outputs[0].untyped_storage()
        for index, (source, output) in enumerate(zip(inputs, outputs, strict=True)):
            self.assert_tensor_equal(output, source)
            self.assertIs(output.untyped_storage(), storage)
            offset = output.data_ptr() - storage.data_ptr()
            self.assertEqual(offset, index * 256)
            self.assertEqual(output.data_ptr() % source.element_size(), 0)
        self.assertEqual(arena.allocation_count, 1)
        self.assertEqual(
            arena.payload_bytes, sum(row.numel() * row.element_size() for row in inputs)
        )
        # Mutating one returned view cannot alter another view or its source.
        before = outputs[1].clone()
        outputs[0].zero_()
        self.assert_tensor_equal(outputs[1], before)
        self.assertFalse(torch.equal(outputs[0], inputs[0]))

    def test_views_outlive_closed_arena_and_then_release_storage(self):
        arena = PinnedHostArena(512, pin_memory=False)
        source = torch.arange(300, dtype=torch.int16)
        first = arena.copy(source[:150])
        second = arena.copy(source[150:])
        first_owner = weakref.ref(first.untyped_storage())
        second_owner = weakref.ref(second.untyped_storage())
        self.assertIsNot(first.untyped_storage(), second.untyped_storage())
        arena.close()
        del arena
        gc.collect()
        self.assertIsNotNone(first_owner())
        self.assertIsNotNone(second_owner())
        self.assert_tensor_equal(first, source[:150])
        self.assert_tensor_equal(second, source[150:])
        del first, second
        gc.collect()
        self.assertIsNone(first_owner())
        self.assertIsNone(second_owner())

    def test_boundary_overflow_alignment_and_invalid_shapes(self):
        arena = PinnedHostArena(512, pin_memory=False)
        a = arena.copy(torch.arange(256, dtype=torch.uint8))
        b = arena.copy(torch.arange(128, dtype=torch.int16))
        c = arena.copy(torch.ones(1, dtype=torch.uint8))
        self.assertIs(a.untyped_storage(), b.untyped_storage())
        self.assertIsNot(a.untyped_storage(), c.untyped_storage())
        self.assertEqual(arena.allocation_count, 2)
        with self.assertRaisesRegex(ValueError, "fit entirely"):
            arena.copy(torch.zeros(513, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "fit entirely"):
            arena.copy(torch.empty(0))
        with self.assertRaisesRegex(ValueError, "dense CPU"):
            arena.copy(torch.ones(1, requires_grad=True))
        with self.assertRaises(ValueError):
            PinnedHostArena(768, pin_memory=False)
        arena.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            arena.copy(torch.ones(1))

    def test_failed_allocation_closes_owner_without_invalidating_prior_views(self):
        arena = PinnedHostArena(256, pin_memory=False)
        output = arena.copy(torch.arange(256, dtype=torch.uint8))
        reference = weakref.ref(output.untyped_storage())
        source = torch.ones(1)
        real_empty = torch.empty

        def fail(*args, **kwargs):
            if args == (256,) and kwargs.get("dtype") == torch.uint8:
                raise MemoryError("injected slab allocation")
            return real_empty(*args, **kwargs)

        with (
            patch.object(torch, "empty", new=fail),
            self.assertRaisesRegex(MemoryError, "injected slab"),
        ):
            arena.copy(source)
        self.assertTrue(arena._closed)
        self.assertIsNone(arena._buffer)
        self.assert_tensor_equal(output, torch.arange(256, dtype=torch.uint8))
        del output
        gc.collect()
        self.assertIsNone(reference())

    def test_production_defaults_request_native_pinned_uint8_allocations(self):
        calls = []
        real_empty = torch.empty

        def capture(*args, **kwargs):
            calls.append((args, dict(kwargs)))
            # Inspect the true production request, but use a tiny pageable
            # allocation so this test never initializes a CUDA driver.
            return real_empty(*args, **{**kwargs, "pin_memory": False})

        arena = PinnedHostArena(512)
        source = torch.arange(16, dtype=torch.bfloat16)
        with patch.object(torch, "empty", new=capture):
            output = arena.copy(source)
        self.assertEqual(
            calls, [((512,), {"dtype": torch.uint8, "device": "cpu", "pin_memory": True})]
        )
        self.assert_tensor_equal(output, source)
        self.assertEqual(PinnedHostArena().chunk_bytes, 2 * 1024**3)
        arena.close()

    def test_failed_cpu_copy_releases_new_slab(self):
        arena = PinnedHostArena(512, pin_memory=False)
        source = torch.ones(3)
        allocated = []
        real_empty = torch.empty

        def capture(*args, **kwargs):
            value = real_empty(*args, **kwargs)
            allocated.append(weakref.ref(value.untyped_storage()))
            return value

        def fail_copy(*args, **kwargs):
            raise RuntimeError("injected CPU copy")

        with (
            patch.object(torch, "empty", new=capture),
            patch.object(torch.Tensor, "copy_", new=fail_copy),
            self.assertRaisesRegex(RuntimeError, "injected CPU copy"),
        ):
            arena.copy(source)
        gc.collect()
        self.assertTrue(arena._closed)
        self.assertTrue(all(reference() is None for reference in allocated))
        self.assertEqual(arena.payload_bytes, 0)

    def loader_context(self, stack, *, failing_copy=None):
        arenas, mappings, owners = [], [], []

        class TrackingArena(PinnedHostArena):
            def __init__(self):
                super().__init__(2048, pin_memory=False)
                self.calls = 0
                arenas.append(self)

            def copy(self, source):
                self.calls += 1
                if failing_copy == self.calls:
                    raise MemoryError("injected mapped copy")
                output = super().copy(source)
                owners.append(weakref.ref(output.untyped_storage()))
                return output

        class TrackingMapped(H3MappedSafetensor):
            def __init__(self, path):
                super().__init__(path)
                mappings.append((self, self._mapping))

        real_empty = torch.empty

        def forbid_early_pin(*args, **kwargs):
            if kwargs.get("pin_memory"):
                raise AssertionError("per-tensor pinning escaped the arena")
            return real_empty(*args, **kwargs)

        def forbid_tensor_pin(*args, **kwargs):
            raise AssertionError("Tensor.pin_memory escaped the arena")

        stack.enter_context(patch.object(torch, "empty", new=forbid_early_pin))
        stack.enter_context(patch.object(torch.Tensor, "pin_memory", new=forbid_tensor_pin))
        for module in (denoiser, parallel):
            stack.enter_context(patch.object(module, "PinnedHostArena", new=TrackingArena))
            stack.enter_context(patch.object(module, "H3MappedSafetensor", new=TrackingMapped))
        return arenas, mappings, owners

    def assert_closed(self, arenas, mappings):
        self.assertTrue(all(arena._closed and arena._buffer is None for arena in arenas))
        self.assertTrue(
            all(
                wrapper._mapping is None and wrapper._handle.closed and raw.closed
                for wrapper, raw in mappings
            )
        )

    def test_actual_ring_loader_all_fields_overlay_and_cross_block_sharing(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            artifact = fixture(Path(directory))
            hashes = [
                hashlib.sha256((artifact.directory / row.path).read_bytes()).hexdigest()
                for row in artifact.blocks
            ]
            expected = [
                denoiser.load_h3_native_block(artifact, row.index)[1] for row in artifact.blocks
            ]
            overlays = [
                torch.full_like(row.adaln_table, 41 + index)
                for index, row in enumerate(expected)
            ]
            arenas, mappings, _owners = self.loader_context(stack)
            ring = CpuRing.load(artifact, device="cpu", adaln_table_load=overlays.__getitem__)
            self.assert_closed(arenas, mappings)
            self.assertEqual(len(arenas), 1)
            block_owners = []
            for index, (actual, source) in enumerate(
                zip(ring.host_blocks, expected, strict=True)
            ):
                self.assert_weights_equal(actual, replace(source, adaln_table=overlays[index]))
                block_owners.append(
                    {
                        row.untyped_storage().data_ptr()
                        for row in tensors(actual)
                        if row._base is not None
                    }
                )
                for field in ("qkv", "attention_out", "ffn_in", "ffn_out"):
                    self.assertIsNone(getattr(actual, field).scales._base)
                self.assertEqual(
                    hashlib.sha256(
                        (artifact.directory / artifact.blocks[index].path).read_bytes()
                    ).hexdigest(),
                    hashes[index],
                )
            self.assertTrue(any(left & right for left, right in pairwise(block_owners)))
            self.assertGreater(arenas[0].allocation_count, 1)

    def test_mapped_allocation_error_preserves_error_and_releases_partial_slabs(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            artifact = fixture(Path(directory))
            arenas, mappings, owners = self.loader_context(stack, failing_copy=5)
            with self.assertRaisesRegex(MemoryError, "injected mapped copy"):
                CpuRing.load(artifact, device="cpu")
            self.assert_closed(arenas, mappings)
            gc.collect()
            self.assertTrue(all(owner() is None for owner in owners))

    def test_ring_constructor_error_releases_loaded_host_views(self):
        class FailingRing(CpuRing):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("injected ring construction")

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            artifact = fixture(Path(directory))
            arenas, mappings, owners = self.loader_context(stack)
            with self.assertRaisesRegex(RuntimeError, "injected ring construction"):
                FailingRing.load(artifact, device="cpu")
            self.assert_closed(arenas, mappings)
            gc.collect()
            self.assertTrue(all(owner() is None for owner in owners))

    def run_parallel(self, strategy, *, failing_copy=None):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            artifact = fixture(Path(directory))
            expected = [
                denoiser.load_h3_native_block(artifact, row.index)[1] for row in artifact.blocks
            ]
            if strategy == "tensor":
                expected_by_rank = [
                    [parallel.shard_h3_block(row, rank) for row in expected] for rank in (0, 1)
                ]
            else:
                expected_by_rank = [expected, expected]
            arenas, mappings, owners = self.loader_context(stack, failing_copy=failing_copy)
            stack.enter_context(
                patch.object(torch.cuda, "get_device_capability", return_value=(8, 6))
            )
            stack.enter_context(patch.object(parallel, "_DevicePair", new=CpuPair))
            stack.enter_context(patch.object(parallel, "_TensorRing", new=CpuRing))
            stack.enter_context(patch.object(parallel, "_SequenceRing", new=CpuRing))
            if failing_copy:
                with self.assertRaisesRegex(MemoryError, "injected mapped copy"):
                    parallel.H3NativeDenoiserParallel.load(
                        artifact, devices=("cuda:0", "cuda:1"), strategy=strategy
                    )
                self.assert_closed(arenas, mappings)
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))
                return
            result = parallel.H3NativeDenoiserParallel.load(
                artifact, devices=("cuda:0", "cuda:1"), strategy=strategy
            )
            self.assert_closed(arenas, mappings)
            self.assertEqual(len(arenas), 1)
            for rank in (0, 1):
                for actual, expected_block in zip(
                    result.rings[rank].host_blocks, expected_by_rank[rank], strict=True
                ):
                    self.assert_weights_equal(actual, expected_block)
            if strategy == "sequence-head":
                for left, right in zip(
                    result.rings[0].host_blocks, result.rings[1].host_blocks, strict=True
                ):
                    self.assertIs(left, right)
                self.assertEqual(result.host_weight_bytes, result.rings[0].host_weight_bytes)
            else:
                self.assertEqual(
                    result.host_weight_bytes,
                    sum(ring.host_weight_bytes for ring in result.rings),
                )
            result.close()
            self.assertTrue(result.pair.closed)

    def test_actual_tp_loader_does_not_pin_before_arena(self):
        self.run_parallel("tensor")

    def test_actual_sequence_loader_keeps_one_shared_host_copy(self):
        self.run_parallel("sequence-head")

    def test_tp_mapped_failure_releases_both_partial_rank_stores(self):
        self.run_parallel("tensor", failing_copy=30)

    def retained_error(self, action):
        # assertRaises strips traceback frames itself and would hide a loader
        # retention bug. Keep the actual exception/traceback alive throughout
        # the memory assertion instead.
        try:
            action()
        except BaseException as error:
            return error
        self.fail("the injected error did not occur")

    def test_retained_constructor_traceback_releases_single_and_both_parallel_ranks(self):
        for strategy, failure_rank in (
            ("single", 0),
            ("tensor", 0),
            ("tensor", 1),
            ("sequence-head", 0),
            ("sequence-head", 1),
        ):
            with (
                self.subTest(strategy=strategy, failure_rank=failure_rank),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                artifact = fixture(Path(directory))
                arenas, mappings, owners = self.loader_context(stack)
                calls = 0

                class FailingRing(CpuRing):
                    def __init__(self, *args, _failure_rank=failure_rank, **kwargs):
                        nonlocal calls
                        index = calls
                        calls += 1
                        if index == _failure_rank:
                            raise RuntimeError("retained constructor failure")
                        super().__init__(*args, **kwargs)

                if strategy == "single":
                    action = partial(FailingRing.load, artifact, device="cpu")
                else:
                    stack.enter_context(
                        patch.object(torch.cuda, "get_device_capability", return_value=(8, 6))
                    )
                    stack.enter_context(patch.object(parallel, "_DevicePair", new=CpuPair))
                    stack.enter_context(patch.object(parallel, "_TensorRing", new=FailingRing))
                    stack.enter_context(
                        patch.object(parallel, "_SequenceRing", new=FailingRing)
                    )
                    action = partial(
                        parallel.H3NativeDenoiserParallel.load,
                        artifact,
                        devices=("cuda:0", "cuda:1"),
                        strategy=strategy,
                    )
                error = self.retained_error(action)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("retained constructor failure", str(error))
                self.assertIsNotNone(error.__traceback__)
                self.assert_closed(arenas, mappings)
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))

    def test_retained_device_pair_constructor_traceback_releases_host_store(self):
        class FailingPair:
            def __init__(self, devices):
                raise RuntimeError("retained pair constructor failure")

        for strategy in ("tensor", "sequence-head"):
            with (
                self.subTest(strategy=strategy),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                artifact = fixture(Path(directory))
                arenas, mappings, owners = self.loader_context(stack)
                stack.enter_context(
                    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6))
                )
                stack.enter_context(patch.object(parallel, "_DevicePair", new=FailingPair))
                error = self.retained_error(
                    partial(
                        parallel.H3NativeDenoiserParallel.load,
                        artifact,
                        devices=("cuda:0", "cuda:1"),
                        strategy=strategy,
                    )
                )
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("retained pair constructor failure", str(error))
                self.assertIsNotNone(error.__traceback__)
                self.assert_closed(arenas, mappings)
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))

    def test_retained_loading_traceback_releases_previous_blocks_and_lora_copies(self):
        for strategy in ("single", "tensor", "sequence-head"):
            for failure in (5, 13, 30, 50):
                with (
                    self.subTest(strategy=strategy, failure=failure),
                    tempfile.TemporaryDirectory() as directory,
                    ExitStack() as stack,
                ):
                    artifact = fixture(Path(directory))
                    arenas, mappings, owners = self.loader_context(stack, failing_copy=failure)
                    if strategy == "single":
                        action = partial(CpuRing.load, artifact, device="cpu")
                    else:
                        stack.enter_context(
                            patch.object(
                                torch.cuda, "get_device_capability", return_value=(8, 6)
                            )
                        )
                        stack.enter_context(patch.object(parallel, "_DevicePair", new=CpuPair))
                        stack.enter_context(patch.object(parallel, "_TensorRing", new=CpuRing))
                        stack.enter_context(
                            patch.object(parallel, "_SequenceRing", new=CpuRing)
                        )
                        action = partial(
                            parallel.H3NativeDenoiserParallel.load,
                            artifact,
                            devices=("cuda:0", "cuda:1"),
                            strategy=strategy,
                        )
                    error = self.retained_error(action)
                    self.assertIsInstance(error, MemoryError)
                    self.assertIn("injected mapped copy", str(error))
                    self.assertIsNotNone(error.__traceback__)
                    self.assert_closed(arenas, mappings)
                    gc.collect()
                    self.assertTrue(all(owner() is None for owner in owners))

    def test_retained_parallel_wrapper_constructor_error_releases_built_rings(self):
        class FailingParallel(parallel.H3NativeDenoiserParallel):
            def __init__(self):
                raise RuntimeError("retained parallel wrapper constructor failure")

        for strategy in ("tensor", "sequence-head"):
            with (
                self.subTest(strategy=strategy),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                artifact = fixture(Path(directory))
                arenas, mappings, owners = self.loader_context(stack)
                stack.enter_context(
                    patch.object(torch.cuda, "get_device_capability", return_value=(8, 6))
                )
                stack.enter_context(patch.object(parallel, "_DevicePair", new=CpuPair))
                stack.enter_context(patch.object(parallel, "_TensorRing", new=CpuRing))
                stack.enter_context(patch.object(parallel, "_SequenceRing", new=CpuRing))
                error = self.retained_error(
                    partial(
                        FailingParallel.load,
                        artifact,
                        devices=("cuda:0", "cuda:1"),
                        strategy=strategy,
                    )
                )
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("retained parallel wrapper constructor failure", str(error))
                self.assertIsNotNone(error.__traceback__)
                self.assert_closed(arenas, mappings)
                gc.collect()
                self.assertTrue(all(owner() is None for owner in owners))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArenaTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_git": torch.version.git_version,
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "cuda_initialized": torch.cuda.is_initialized(),
                "pinned_allocation_tested": False,
                "cpu_only": True,
            },
            indent=2,
        )
    )
    sys.exit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
