"""Run ``python -m vflash.compiler`` for explicit offline preparation or compilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vflash.contracts import ContractError
from vflash.model_assets import COMPLETE_MODEL_PROFILES, DEFAULT_MODEL_PROFILE

from .assets import load_prepared_weights, prepare_weights
from .h3 import compile_assets, validate_weight_headers


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile fixed official H3 BF16 weights")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="verify raw official weight bytes on CPU")
    prepare.add_argument(
        "--profile", choices=COMPLETE_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE
    )
    prepare.add_argument("--transformer", required=True, type=Path)
    prepare.add_argument("--adapter", required=True, type=Path)
    prepare.add_argument("--receipt", required=True, type=Path)
    check = commands.add_parser(
        "check", help="check an existing receipt and tensor headers on CPU"
    )
    check.add_argument("--receipt", required=True, type=Path)
    compile_command = commands.add_parser(
        "compile", help="compile on an exclusively assigned matching GPU"
    )
    compile_command.add_argument("--receipt", required=True, type=Path)
    compile_command.add_argument("--output", required=True, type=Path)
    compile_command.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepared = prepare_weights(
            args.transformer,
            args.adapter,
            args.receipt,
            profile_id=args.profile,
            progress=lambda count, total: print(f"weights {count}/{total}", flush=True),
        )
    else:
        prepared = load_prepared_weights(args.receipt)
    if args.command in {"prepare", "check"}:
        print(json.dumps(validate_weight_headers(prepared), sort_keys=True))
        return
    from vflash.hardware import discover_nvidia_devices

    devices = [device for device in discover_nvidia_devices() if device.index == args.gpu]
    if len(devices) != 1:
        raise ContractError("the selected GPU index was not discovered")
    result = compile_assets(
        prepared,
        args.output,
        device=devices[0],
        progress=lambda stage, count, total: print(f"{stage} {count}/{total}", flush=True),
    )
    print(
        json.dumps(
            {
                "artifact": str(result.artifact),
                "schedule_overlay": str(result.schedule_overlay),
                "auxiliary_tensor": str(result.auxiliary_tensor),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
