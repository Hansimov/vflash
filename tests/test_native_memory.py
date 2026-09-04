from types import SimpleNamespace

from vflash.native.h3_native_conditioning_runtime import H3NativeConditioningRuntime


def test_host_allocation_and_retained_cache_are_reported_separately():
    cuda = SimpleNamespace(
        memory=SimpleNamespace(
            host_memory_stats=lambda: {
                "active_bytes.current": 32,
                "allocated_bytes.current": 64,
            }
        ),
        memory_allocated=lambda _device: 128,
        memory_reserved=lambda _device: 256,
        get_device_name=lambda _device: "Test GPU",
    )
    runtime = H3NativeConditioningRuntime.__new__(H3NativeConditioningRuntime)
    runtime._torch = SimpleNamespace(cuda=cuda, get_float32_matmul_precision=lambda: "high")
    runtime.device = "cuda:0"
    runtime.compute_capability = (8, 6)
    runtime.artifact = SimpleNamespace(
        artifact_id="test", weight_profile="test", adapter_execution="runtime-residual"
    )
    runtime.overlay = SimpleNamespace(overlay_id="test", schedule=SimpleNamespace(nfe=4))
    runtime.denoiser = SimpleNamespace(host_weight_bytes=24)
    runtime.initialization_seconds = 1.0
    runtime.initialization_stages = {}
    runtime.attention_backend = "torch-flash"
    runtime.attention_strict_prefix_evaluations = 0
    runtime.attention_schedule = ("torch-flash",) * 4

    assert runtime.metadata()["memory"] == {
        "weight_residency": "block-ring",
        "pinned_host_weight_payload_bytes": 24,
        "pinned_host_allocated_bytes": 32,
        "pinned_host_reserved_bytes": 64,
        "device_allocated_bytes": 128,
        "device_reserved_bytes": 256,
    }
