"""Isolated CUDA-visible device probe, including MIG; parent never initializes CUDA."""

import json
import re
import subprocess
import sys

from vflash.contracts import ContractError

PROBE = """import json,torch
rows=[]
for i in range(torch.cuda.device_count()):
 p=torch.cuda.get_device_properties(i)
 rows.append(dict(index=i,uuid=str(p.uuid),name=p.name,memory_gib=p.total_memory/1024**3,
                  compute_capability=f"{p.major}.{p.minor}"))
print(json.dumps(rows))
"""


def visible_devices():
    from vflash.hardware import NvidiaDevice

    try:
        listed = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
        identities = re.findall(r"UUID: ((?:GPU|MIG)-[0-9a-fA-F-]{36})", listed)
        raw = subprocess.run(
            [sys.executable, "-c", PROBE],
            capture_output=True,
            text=True,
            timeout=45,
            check=True,
        ).stdout
        if len(raw) > 65536:
            raise ValueError("oversized probe")
        rows = json.loads(raw)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
            raise ValueError("no visible CUDA device")
        devices = []
        for row in rows:
            suffix = row["uuid"].removeprefix("GPU-").removeprefix("MIG-").lower()
            matches = [v for v in identities if v[4:].lower() == suffix]
            if len(matches) != 1 or not 0 < row["memory_gib"] <= 256:
                raise ValueError("CUDA and NVIDIA identities differ")
            devices.append(
                NvidiaDevice(
                    index=row["index"],
                    uuid=matches[0],
                    name=row["name"],
                    memory_gib=row["memory_gib"],
                    compute_capability=row["compute_capability"],
                    power_limit_watts=0.0,
                )
            )
        return tuple(devices)
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError):
        raise ContractError(
            "CUDA-visible GPU/MIG discovery failed; inspect private diagnostics"
        ) from None
