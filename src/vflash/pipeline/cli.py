"""Local complete-video commands, with no model imports during CLI discovery."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from vflash.contracts import ContractError


def add_pipeline_commands(commands: argparse._SubParsersAction) -> None:
    prepare = commands.add_parser(
        "prepare-pipeline", help="verify local complete-pipeline assets and write a receipt"
    )
    prepare.add_argument("--assets", type=Path, required=True, help="six-path asset JSON")
    prepare.add_argument("--receipt", type=Path, required=True, help="new local receipt path")
    generate = commands.add_parser(
        "generate", help="generate a complete Ref4 MP4 from a prompt and ordered images"
    )
    generate.add_argument("--prepared-assets", type=Path, required=True)
    generate.add_argument("--prompt-file", type=Path, required=True, help="UTF-8 prompt file")
    generate.add_argument(
        "--reference",
        type=Path,
        action="append",
        required=True,
        help="local image; repeat in <Picture N> order, up to three images",
    )
    generate.add_argument("--width", type=int, default=928)
    generate.add_argument("--height", type=int, default=512)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", type=Path, required=True, help="new MP4 output path")
    generate.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi index")
    generate.add_argument(
        "--trust-local-code",
        action="store_true",
        help="allow official decoder code from the verified local model snapshot",
    )


def run_pipeline_command(args: argparse.Namespace) -> int:
    from vflash.pipeline import (
        PipelineAssets,
        VideoRequest,
        load_prepared_pipeline_assets,
        prepare_pipeline_assets,
    )

    if args.command == "prepare-pipeline":
        prepared = prepare_pipeline_assets(PipelineAssets.from_json(args.assets), args.receipt)
        print(json.dumps({"receipt": str(prepared.receipt), "sha256": prepared.receipt_sha256}))
        return 0

    if not args.trust_local_code:
        raise ContractError("review the official decoder code and pass --trust-local-code")
    with args.prompt_file.open(encoding="utf-8") as handle:
        prompt = handle.read(65537)
    request = VideoRequest(
        prompt=prompt,
        references=tuple(args.reference),
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    if args.output.exists() or args.output.is_symlink():
        raise ContractError("the output path already exists")
    # Catch malformed or absent inputs before the costly model load. Generate
    # binds its own decoded bytes again, so a changed file is never trusted here.
    from vflash.adapters.references import read_reference

    for path in request.ordered_references:
        reference = read_reference(path)
        reference.close()
    prepared = load_prepared_pipeline_assets(args.prepared_assets)
    from vflash.hardware import discover_nvidia_devices
    from vflash.pipeline import H3Pipeline

    devices = {device.index: device for device in discover_nvidia_devices()}
    if args.gpu not in devices:
        raise ContractError(f"GPU index {args.gpu} was not found")
    with H3Pipeline(prepared, device=devices[args.gpu], trust_local_code=True) as pipeline:
        result = pipeline.generate(
            request,
            args.output,
            progress=lambda event: print(
                json.dumps(asdict(event)), file=sys.stderr, flush=True
            ),
        )
    print(json.dumps(asdict(result), default=str, indent=2))
    return 0
