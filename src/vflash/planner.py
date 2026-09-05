"""Resolve one immutable profile onto one device or a cooperating pair."""

from __future__ import annotations

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError, ExecutionPlan, PeerDevice
from vflash.hardware import NvidiaDevice


def resolve_plan(
    catalog: ProfileCatalog,
    *,
    profile_id: str,
    device: NvidiaDevice,
    peer_device: NvidiaDevice | None = None,
    strategy: str | None = None,
) -> ExecutionPlan:
    strategy = strategy or ("sequence-head" if peer_device is not None else "single")
    if strategy not in {"single", "tensor", "sequence-head"}:
        raise ContractError("parallel strategy must be single, tensor or sequence-head")
    if (strategy == "single") != (peer_device is None):
        raise ContractError("a parallel strategy requires exactly two selected GPUs")
    profile = catalog.profile(profile_id)
    candidates = [
        catalog.target(target_id)
        for target_id in profile.target_ids
        if catalog.target(target_id).compute_capability == device.compute_capability
        and catalog.target(target_id).minimum_memory_gib <= device.memory_gib
    ]
    if not candidates:
        raise ContractError(
            f"profile {profile.id} has no target for {device.name} "
            f"(sm{device.compute_capability.replace('.', '')}, {device.memory_gib:.1f} GiB)"
        )
    target = max(candidates, key=lambda item: item.minimum_memory_gib)
    if peer_device is not None and (
        device.uuid == peer_device.uuid
        or target.compute_capability != "8.6"
        or peer_device.compute_capability != "8.6"
        or peer_device.memory_gib < target.minimum_memory_gib
    ):
        raise ContractError("parallel Ref2VA requires two distinct SM86 GPUs with 20 GiB each")
    return ExecutionPlan(
        profile=profile,
        target=target,
        gpu_index=device.index,
        gpu_uuid=device.uuid,
        gpu_name=device.name,
        gpu_memory_gib=device.memory_gib,
        parallel_strategy=strategy,
        peer_device=(
            PeerDevice(
                peer_device.index, peer_device.uuid, peer_device.name, peer_device.memory_gib
            )
            if peer_device is not None
            else None
        ),
    )
