"""Resolve one immutable profile onto one physical GPU."""

from __future__ import annotations

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError, ExecutionPlan
from vflash.hardware import NvidiaDevice


def resolve_plan(
    catalog: ProfileCatalog,
    *,
    profile_id: str,
    device: NvidiaDevice,
) -> ExecutionPlan:
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
    return ExecutionPlan(
        profile=profile,
        target=target,
        gpu_index=device.index,
        gpu_uuid=device.uuid,
        gpu_name=device.name,
        gpu_memory_gib=device.memory_gib,
    )
