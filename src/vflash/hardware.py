"""NVIDIA device discovery for explicit Vflash targets and optional visible CUDA partitions."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass

from vflash.contracts import ContractError


@dataclass(frozen=True, slots=True)
class NvidiaDevice:
    index: int
    uuid: str
    name: str
    memory_gib: float
    compute_capability: str
    power_limit_watts: float


def discover_nvidia_devices() -> tuple[NvidiaDevice, ...]:
    if os.environ.get("VFLASH_DEVICE_DISCOVERY") == "cuda-visible":
        from vflash.cuda_visible import visible_devices

        return visible_devices()
    command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap,power.limit",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ContractError("nvidia-smi could not enumerate GPUs") from exc
    devices = []
    for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
        if len(row) != 6:
            raise ContractError("nvidia-smi returned an unexpected GPU row")
        devices.append(
            NvidiaDevice(
                index=int(row[0]),
                uuid=row[1],
                name=row[2],
                memory_gib=float(row[3]) / 1024,
                compute_capability=row[4],
                power_limit_watts=float(row[5]),
            )
        )
    if not devices:
        raise ContractError("nvidia-smi found no GPUs")
    return tuple(devices)
