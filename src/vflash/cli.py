"""Command line interface for profile inspection and hardware planning."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import discover_nvidia_devices
from vflash.planner import resolve_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vflash")
    parser.add_argument("--catalog", type=Path, help="use an explicit profile catalog")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="show the physical NVIDIA devices Vflash can see")

    commands.add_parser("profiles", help="list the available GPU and model configurations")

    plan = commands.add_parser(
        "plan", help="resolve a profile onto one physical GPU or a cooperating pair"
    )
    plan.add_argument("profile_id")
    plan.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi index")
    plan.add_argument(
        "--peer-gpu", type=int, help="second physical GPU for cooperative execution"
    )
    plan.add_argument("--strategy", choices=("single", "tensor", "sequence-head"))
    denoise = commands.add_parser(
        "denoise",
        help="run a native Ref2VA distilled profile from a conditioning bundle",
    )
    denoise.add_argument("profile_id")
    denoise.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi index")
    denoise.add_argument(
        "--peer-gpu", type=int, help="second physical GPU for cooperative execution"
    )
    denoise.add_argument("--strategy", choices=("single", "tensor", "sequence-head"))
    denoise.add_argument("--bundle", type=Path, required=True)
    denoise.add_argument("--artifact", type=Path, required=True)
    denoise.add_argument("--schedule-overlay", type=Path, required=True)
    denoise.add_argument("--auxiliary-tensor", type=Path, required=True)
    denoise.add_argument("--output-latents", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps([asdict(item) for item in discover_nvidia_devices()], indent=2))
            return 0
        catalog = (
            ProfileCatalog.load(args.catalog) if args.catalog else ProfileCatalog.bundled()
        )
        if args.command == "profiles":
            rows = [
                {
                    "id": profile.id,
                    "mode": profile.mode.value,
                    "availability": profile.availability.value,
                    "nfe": profile.nfe,
                    "attention": profile.attention.backend,
                    "exact_attention": profile.attention.exact,
                    "target_ids": list(profile.target_ids),
                }
                for profile in catalog.profiles
            ]
            print(json.dumps(rows, indent=2))
            return 0
        devices = {device.index: device for device in discover_nvidia_devices()}
        if args.gpu not in devices:
            raise ContractError(f"GPU index {args.gpu} was not found")
        if args.peer_gpu is not None and args.peer_gpu not in devices:
            raise ContractError(f"GPU index {args.peer_gpu} was not found")
        plan = resolve_plan(
            catalog,
            profile_id=args.profile_id,
            device=devices[args.gpu],
            peer_device=devices.get(args.peer_gpu),
            strategy=args.strategy,
        )
        if args.command == "denoise":
            from vflash.native.runner import denoise_conditioning_bundle

            result = denoise_conditioning_bundle(
                plan,
                bundle=args.bundle,
                artifact=args.artifact,
                schedule_overlay=args.schedule_overlay,
                auxiliary_tensor=args.auxiliary_tensor,
                output_latents=args.output_latents,
            )
            print(json.dumps(result, indent=2))
            return 0
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"vflash: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
