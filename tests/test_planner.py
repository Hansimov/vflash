import pytest

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice
from vflash.planner import resolve_plan


def device(*, capability: str, memory: float, index: int = 0) -> NvidiaDevice:
    return NvidiaDevice(index, "test-device", "Test GPU", memory, capability, 320.0)


def test_resolves_hardware_specialized_plan() -> None:
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id="ref2va-turbo4-exact-sm86",
        device=device(capability="8.6", memory=20.0),
    )
    assert plan.target.id == "sm86-20g-block-ring"


def test_sm86_turbo4_preview_resolves_without_research_opt_in() -> None:
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id="ref2va-turbo4-exact-sm86",
        device=device(capability="8.6", memory=20.0),
    )
    assert plan.target.weight_residency == "block-ring"
    assert plan.target.block_prefetch_slots == 2
    with pytest.raises(ContractError, match="has no target"):
        resolve_plan(
            ProfileCatalog.bundled(),
            profile_id="ref2va-turbo4-exact-sm86",
            device=device(capability="8.9", memory=48.0),
        )


def test_rejects_unsupported_gpu() -> None:
    with pytest.raises(ContractError, match="has no target"):
        resolve_plan(
            ProfileCatalog.bundled(),
            profile_id="ref2va-turbo4-exact-sm86",
            device=device(capability="9.0", memory=80.0),
        )


def test_two_devices_resolve_an_explicit_strategy():
    from dataclasses import replace

    first = device(capability="8.6", memory=20.0)
    second = replace(first, index=1, uuid="peer-device")
    plan = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id="ref2va-turbo4-exact-sm86",
        device=first,
        peer_device=second,
        strategy="tensor",
    )
    assert plan.parallel_strategy == "tensor"
    assert plan.gpu_uuids == (first.uuid, second.uuid)
    assert plan.peer_device.index == 1
    default = resolve_plan(
        ProfileCatalog.bundled(),
        profile_id="ref2va-turbo4-exact-sm86",
        device=first,
        peer_device=second,
    )
    assert default.parallel_strategy == "sequence-head"


@pytest.mark.parametrize(
    "failure", ["duplicate", "architecture", "memory", "missing", "strategy"]
)
def test_rejects_invalid_parallel_device_groups(failure):
    from dataclasses import replace

    first = device(capability="8.6", memory=20.0)
    second = replace(first, index=1, uuid="peer-device")
    strategy = "tensor"
    if failure == "duplicate":
        second = first
    elif failure == "architecture":
        second = replace(second, compute_capability="8.9")
    elif failure == "memory":
        second = replace(second, memory_gib=10.0)
    elif failure == "missing":
        second = None
    elif failure == "strategy":
        strategy = "single"
    with pytest.raises(ContractError):
        resolve_plan(
            ProfileCatalog.bundled(),
            profile_id="ref2va-turbo4-exact-sm86",
            device=first,
            peer_device=second,
            strategy=strategy,
        )
