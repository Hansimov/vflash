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


def resident_min_gib(compute_capability: str) -> int:
    """Smallest card admitted to resident weights: 75 GiB on SM90, 90 GiB on SM120.

    VFLASH_SM120_RESIDENT_GIB lowers the SM120 gate (24-90) for a deployment that serves only
    shapes whose measured peak fits a smaller card, e.g. 30 for a 32 GiB RTX 5090 limited to
    1 MP / 5 s requests by its scheduler; a larger shape then fails with CUDA OOM.
    """
    if compute_capability != "12.0":
        return 75
    value = os.environ.get("VFLASH_SM120_RESIDENT_GIB", "90")
    if not value.isdigit() or not 24 <= int(value) <= 90:
        raise ContractError("VFLASH_SM120_RESIDENT_GIB must be an integer from 24 to 90")
    return int(value)


def validate_residency(plan: Any, choice: str, attention: str, hybrid: Any) -> str:
    if choice == "block-ring":
        return choice
    if choice != "resident":
        raise ContractError("pipeline weight_residency must be block-ring or resident")
    if (
        plan.target.compute_capability not in {"9.0", "12.0"}
        or plan.gpu_memory_gib < resident_min_gib(plan.target.compute_capability)
        or plan.parallel_strategy != "single"
        or attention != "veda-triton"
        or hybrid is None
    ):
        raise ContractError(
            "resident pipeline requires a single hybrid Veda allocation: "
            ">=75GiB SM90 or >=90GiB SM120"
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


# Resident memory follows the denoised token count. SM90 keeps its measured canvas
# envelope; 96 GiB SM120 is budgeted by canvas pixels x model frames, which admits
# both 1536x864 for 15 s and a 4 MP canvas for 5 s within native-HD limits.
RESIDENT_SM120_MAX_PIXEL_FRAMES = 2048 * 2048 * 124


def validate_resident_request(
    choice: str, request: Any, profile_id: str | None = None, *, hybrid: bool = False
) -> None:
    if choice != "resident":
        return
    pixels = request.width * request.height
    if (profile_id or "").endswith("-sm120"):
        within = (
            pixels <= 2048 * 2048
            and pixels * request.model_frames <= RESIDENT_SM120_MAX_PIXEL_FRAMES
        )
        envelope = "at most 2048x2048x124 pixel-frames (4 MP for 5 s, 1536x864 for 15 s)"
    else:
        within = pixels <= 1536 * 864
        envelope = "at most 1536x864 pixels"
    keyframe_modes = hybrid and (profile_id or "").endswith("-sm120")
    supported = (
        request.mode in {"i2va", "l2va", "fl2va", "ref2va"}
        if keyframe_modes
        else (request.mode == "i2va" and request.first_frame is not None)
    )
    if keyframe_modes and request.mode == "ref2va" and len(request.ordered_references) > 1:
        within = within and pixels <= 960 * 544 and request.duration_seconds == 5
    if not supported or not within or request.model_frames > 362:
        modes = "hybrid keyframe/Ref modes" if keyframe_modes else "I2VA with one first frame"
        raise ContractError(
            f"resident {modes} supports {envelope} and 15 seconds; "
            "use block-ring for other requests"
        )
