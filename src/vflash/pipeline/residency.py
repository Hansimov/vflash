"""Zero-copy host-master residency primitives for sequential H3 GPU stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from vflash.contracts import ContractError


class CPUResidencyError(RuntimeError):
    """A module cannot participate in the CPU-master residency contract."""


@dataclass(frozen=True)
class CPUModuleMaster:
    state_dict: Any
    non_persistent_buffers: tuple[tuple[str, str, Any], ...]
    tensor_bytes: int


def capture_cpu_master(module: Any) -> CPUModuleMaster:
    """Retain prepared CPU storages while ``Module.to`` uploads new storages."""

    state_dict = module.state_dict()
    non_persistent: list[tuple[str, str, Any]] = []
    tensors = list(state_dict.values())
    for module_name, child in module.named_modules():
        for name in child._non_persistent_buffers_set:
            tensor = child._buffers.get(name)
            if tensor is not None:
                detached = tensor.detach()
                non_persistent.append((module_name, name, detached))
                tensors.append(detached)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise CPUResidencyError("the CPU master contains accelerator tensors")
    unique_storages: dict[tuple[int, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        unique_storages[key] = storage.nbytes()
    return CPUModuleMaster(
        state_dict=state_dict,
        non_persistent_buffers=tuple(non_persistent),
        tensor_bytes=sum(unique_storages.values()),
    )


def restore_cpu_master(module: Any, master: CPUModuleMaster) -> None:
    """Swap retained CPU storages into a module without a CUDA-to-host copy."""

    module.load_state_dict(master.state_dict, strict=True, assign=True)
    for module_name, name, tensor in master.non_persistent_buffers:
        child = module.get_submodule(module_name) if module_name else module
        child._buffers[name] = tensor


def validate_residency(plan: Any, choice: str, attention: str, hybrid: Any) -> str:
    if choice == "block-ring":
        return choice
    if choice != "resident":
        raise ContractError("pipeline weight_residency must be block-ring or resident")
    if (
        plan.target.compute_capability != "9.0"
        or plan.gpu_memory_gib < 75
        or plan.parallel_strategy != "single"
        or attention != "veda-triton"
        or hybrid is None
    ):
        raise ContractError(
            "resident pipeline requires a single >=75GiB SM90 hybrid Veda allocation"
        )
    allocator = os.environ.get(
        "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    )
    options = {
        k.strip(): v.strip()
        for token in allocator.split(",")
        if ":" in token
        for k, v in [token.split(":", 1)]
    }
    if options.get("expandable_segments") != "True":
        raise ContractError(
            "resident pipeline requires PYTORCH_ALLOC_CONF=expandable_segments:True "
            "before CUDA starts"
        )
    return choice


def validate_resident_request(choice: str, request: Any) -> None:
    if choice != "resident":
        return
    if (
        request.mode != "i2va"
        or request.width * request.height > 1536 * 864
        or request.model_frames > 362
        or request.first_frame is None
    ):
        raise ContractError(
            "resident I2VA supports one first frame, at most 1536x864 pixels and 15 seconds; "
            "use block-ring for other requests"
        )
