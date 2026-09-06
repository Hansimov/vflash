import gc
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vflash.native import h3_native_conditioning_runtime as native
from vflash.native import h3_parallel as parallel
from vflash.native.runner import NativeEngineSession


def owned_runtime(events):
    torch = pytest.importorskip("torch")
    runtime = native.H3NativeConditioningRuntime.__new__(native.H3NativeConditioningRuntime)
    storage_refs = []

    def owner(**extra):
        tensor = torch.ones(8)
        storage_refs.append(tensor.untyped_storage()._weak_ref())
        return SimpleNamespace(weight=tensor, **extra)

    def sync(device):
        # Every allocation must stay live through completion of both devices.
        assert all(not torch.UntypedStorage._expired(ref) for ref in storage_refs)
        events.append(("sync", device))

    runtime._torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=sync))
    runtime.devices = ("cuda:0", "cuda:1")
    runtime.input_packer = owner()
    runtime.final_layer = owner()
    runtime.denoiser = owner(close=lambda: events.append(("close", "transport")))
    runtime._closed = runtime._released = False
    return runtime, storage_refs


def test_close_drains_group_then_releases_owned_storage_with_retained_session():
    torch = pytest.importorskip("torch")
    events = []
    runtime, storage_refs = owned_runtime(events)
    session = NativeEngineSession.__new__(NativeEngineSession)
    session.runtime, session.closed = runtime, False
    try:
        with session:
            pass
        gc.collect()
        assert session.closed and session.runtime is None
        assert all(torch.UntypedStorage._expired(ref) for ref in storage_refs)
        assert events == [("sync", "cuda:0"), ("sync", "cuda:1"), ("close", "transport")]
        session.close()
        runtime.close()
        assert len(events) == 3
        with pytest.raises(native.H3NativeConditioningRuntimeError, match="closed"):
            runtime.metadata()
    finally:
        for ref in storage_refs:
            torch.UntypedStorage._free_weak_ref(ref)


def test_failed_synchronization_fences_storage_until_explicit_close_retry():
    torch = pytest.importorskip("torch")
    runtime, storage_refs = owned_runtime([])
    original = runtime._torch.cuda.synchronize

    def fail(_device):
        raise RuntimeError("completion unavailable")

    runtime._torch.cuda.synchronize = fail
    try:
        with pytest.raises(RuntimeError, match="completion unavailable"):
            runtime.close()
        assert runtime._closed and not runtime._released
        assert all(not torch.UntypedStorage._expired(ref) for ref in storage_refs)
        with pytest.raises(native.H3NativeConditioningRuntimeError, match="closed"):
            runtime.generate_latents(None, None)
        runtime._torch.cuda.synchronize = original
        runtime.close()
        assert all(torch.UntypedStorage._expired(ref) for ref in storage_refs)
    finally:
        for ref in storage_refs:
            torch.UntypedStorage._free_weak_ref(ref)


@pytest.mark.parametrize("failed", [None, "rank-0", "rank-1", "executor"])
def test_pair_close_distinguishes_unusable_from_released(failed):
    torch = pytest.importorskip("torch")
    events = []
    runtime, refs = owned_runtime(events)
    pair = parallel._DevicePair.__new__(parallel._DevicePair)
    pair.closed = pair._released = False

    def shutdown(name):
        events.append(("shutdown", name))
        if failed == name:
            raise RuntimeError("injected " + name)

    pair.groups = tuple(
        SimpleNamespace(shutdown=lambda rank=rank: shutdown("rank-" + str(rank)))
        for rank in (0, 1)
    )
    pair.pool = SimpleNamespace(shutdown=lambda **_: shutdown("executor"))
    runtime.denoiser.close = pair.close
    try:
        if failed is None:
            runtime.close()
            assert pair._released and runtime._released
            assert all(torch.UntypedStorage._expired(ref) for ref in refs)
            runtime.close()
            pair.close()
        else:
            with pytest.raises(RuntimeError, match="injected " + failed):
                runtime.close()
            assert pair.closed and runtime._closed
            assert not pair._released and not runtime._released
            assert all(not torch.UntypedStorage._expired(ref) for ref in refs)
            with pytest.raises(parallel.H3NativeDenoiserError, match=r"cleanup.*not confirmed"):
                runtime.close()
            assert runtime.denoiser is not None and not runtime._released
        assert [event for event in events if event[0] == "shutdown"] == [
            ("shutdown", "rank-0"),
            ("shutdown", "rank-1"),
            ("shutdown", "executor"),
        ]
    finally:
        for ref in refs:
            torch.UntypedStorage._free_weak_ref(ref)


def test_failed_pair_execution_cannot_turn_abort_into_confirmed_cleanup(monkeypatch):
    torch = pytest.importorskip("torch")
    events = []
    runtime, refs = owned_runtime(events)
    pair = parallel._DevicePair.__new__(parallel._DevicePair)
    pair.closed = pair._released = False
    pair.devices = ("cuda:0", "cuda:1")
    pair.pool = ThreadPoolExecutor(max_workers=2)
    pair.groups = tuple(
        SimpleNamespace(abort=lambda rank=rank: events.append(("abort", rank)))
        for rank in (0, 1)
    )
    monkeypatch.setattr(
        parallel,
        "_torch",
        lambda: SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda *_: None), inference_mode=nullcontext
        ),
    )
    runtime.denoiser.close = pair.close

    def action(rank):
        if rank == 0:
            raise RuntimeError("failed rank")
        return rank

    try:
        with pytest.raises(RuntimeError, match="failed rank"):
            pair.run(action)
        assert ("abort", 0) in events and ("abort", 1) in events
        with pytest.raises(parallel.H3NativeDenoiserError, match=r"cleanup.*not confirmed"):
            runtime.close()
        assert not runtime._released and not pair._released
        assert all(not torch.UntypedStorage._expired(ref) for ref in refs)
    finally:
        pair.pool.shutdown(wait=True)
        for ref in refs:
            torch.UntypedStorage._free_weak_ref(ref)


@pytest.mark.parametrize("completion_available", [True, False])
def test_failed_constructor_preserves_error_and_releases_only_after_completion(
    monkeypatch, tmp_path, completion_available
):
    torch = pytest.importorskip("torch")
    artifact = SimpleNamespace(
        is_complete_block_stack=True,
        weight_profile="lightx-ref-turbo4-v0.1",
        adapter_execution="runtime-residual",
        source={},
        artifact_id="h3-test",
        target=SimpleNamespace(compute_capability="sm89", target_id="test"),
    )
    overlay = SimpleNamespace(schedule=object(), target_id="test", base_artifact_id="h3-test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))

    def synchronize(*_):
        if not completion_available:
            raise RuntimeError("completion unavailable")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(native, "load_h3_runtime_artifact", lambda *_a, **_kw: artifact)
    monkeypatch.setattr(native, "load_h3_schedule_overlay", lambda *_a, **_kw: overlay)
    monkeypatch.setattr(native, "validate_declared_schedule", lambda *_a, **_kw: None)
    monkeypatch.setattr(native, "load_h3_runtime_auxiliary", lambda p: SimpleNamespace(path=p))
    refs = []

    def fail(self, _store):
        tensor = torch.ones(8)
        refs.append(tensor.untyped_storage()._weak_ref())
        self.input_packer = SimpleNamespace(weight=tensor)
        raise ValueError("original weight failure")

    monkeypatch.setattr(native.H3NativeConditioningRuntime, "_load_components", fail)
    try:
        with pytest.raises(ValueError, match="original weight failure") as failure:
            native.H3NativeConditioningRuntime(
                artifact_path=tmp_path,
                schedule_overlay_path=tmp_path,
                auxiliary_tensor_path=tmp_path,
            )
        gc.collect()
        assert failure.value.__traceback__ is not None
        assert refs
        assert all(torch.UntypedStorage._expired(ref) == completion_available for ref in refs)
        if not completion_available:
            assert "exit the CUDA worker process" in failure.value.__notes__[0]
    finally:
        for ref in refs:
            torch.UntypedStorage._free_weak_ref(ref)
