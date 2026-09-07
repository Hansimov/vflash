"""Local complete-video commands, with no model imports during CLI discovery."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from vflash.contracts import ContractError
from vflash.model_assets import COMPLETE_MODEL_PROFILES, DEFAULT_MODEL_PROFILE, model_profile


def add_pipeline_commands(commands: argparse._SubParsersAction) -> None:
    prepare = commands.add_parser(
        "prepare-pipeline", help="verify local complete-pipeline assets and write a receipt"
    )
    prepare.add_argument(
        "--profile", choices=COMPLETE_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE
    )
    prepare.add_argument("--assets", type=Path, required=True, help="six-path asset JSON")
    prepare.add_argument("--receipt", type=Path, required=True, help="new local receipt path")
    generate = commands.add_parser(
        "generate", help="generate an MP4 from text, ordered images or one reference video"
    )
    generate.add_argument("--prepared-assets", type=Path, required=True)
    generate.add_argument("--prompt-file", type=Path, required=True, help="UTF-8 prompt file")
    generate.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="local image; repeat in <Picture N> order up to three times; omit for T2VA",
    )
    generate.add_argument("--width", type=int, default=928)
    generate.add_argument("--height", type=int, default=512)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", type=Path, required=True, help="new MP4 output path")
    generate.add_argument("--gpu", type=int, required=True, help="physical nvidia-smi index")
    generate.add_argument(
        "--reference-video",
        type=Path,
        help="one local 2-5s MP4/MOV/WebM labeled <Video 1>; excludes --reference",
    )
    generate.add_argument(
        "--peer-gpu", type=int, help="second SM86 GPU for cooperative denoising"
    )
    generate.add_argument(
        "--strategy",
        choices=("sequence-head",),
        help="qualified complete dual-SM86 strategy; default sequence-head",
    )
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
        prepared = prepare_pipeline_assets(
            PipelineAssets.from_json(args.assets), args.receipt, profile_id=args.profile
        )
        print(json.dumps({"receipt": str(prepared.receipt), "sha256": prepared.receipt_sha256}))
        return 0

    if not args.trust_local_code:
        raise ContractError("review the official decoder code and pass --trust-local-code")
    with args.prompt_file.open(encoding="utf-8") as handle:
        prompt = handle.read(65537)
    request = VideoRequest(
        prompt=prompt,
        references=tuple(args.reference),
        reference_video=args.reference_video,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    if args.output.exists() or args.output.is_symlink():
        raise ContractError("the output path already exists")
    # H3Pipeline now validates/decodes once, before its first CUDA load. Do not
    # decode every video twice just to duplicate this same CPU input boundary.
    prepared = load_prepared_pipeline_assets(args.prepared_assets)
    if request.mode != model_profile(prepared.profile_id).definition.mode.value:
        raise ContractError("the request mode differs from the prepared pipeline profile")
    if args.strategy is not None and args.peer_gpu is None:
        raise ContractError("a parallel strategy requires --peer-gpu")
    from vflash.hardware import discover_nvidia_devices
    from vflash.pipeline import H3Pipeline

    devices = {device.index: device for device in discover_nvidia_devices()}
    if args.gpu not in devices:
        raise ContractError(f"GPU index {args.gpu} was not found")
    options = {}
    if args.peer_gpu is not None:
        if args.peer_gpu not in devices:
            raise ContractError(f"GPU index {args.peer_gpu} was not found")
        options = {"peer_device": devices[args.peer_gpu], "strategy": args.strategy}
    with H3Pipeline(
        prepared, device=devices[args.gpu], trust_local_code=True, **options
    ) as pipeline:
        result = pipeline.generate(
            request,
            args.output,
            progress=lambda event: print(
                json.dumps(asdict(event)), file=sys.stderr, flush=True
            ),
        )
    print(json.dumps(asdict(result), default=str, indent=2))
    return 0
