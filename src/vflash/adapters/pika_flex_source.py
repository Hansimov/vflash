"""Explicit, pinned local Pika code import; never downloads or installs code."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from vflash.contracts import ContractError
from vflash.pipeline.pika_flex import PIKA_SOURCE_REVISION


def load_pika_source(root: Path, *, trust_local_code: bool = False):
    if trust_local_code is not True:
        raise ContractError("Pika source loading requires explicit trust_local_code=True")
    root = root.resolve(strict=True)
    receipt = json.loads(files("vflash").joinpath("data/pika_flex_source.json").read_text())
    if receipt["source_revision"] != PIKA_SOURCE_REVISION:
        raise ContractError("Pika source inventory revision differs")
    package = root / "src/pika_h3"
    actual = {
        str(p.relative_to(root))
        for p in package.rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml"}
    }
    if actual != set(receipt["files"]):
        raise ContractError("Pika source files differ from the pinned release")
    for name, expected in receipt["files"].items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ContractError("Pika source content differs from the pinned release")
    for name, module in tuple(sys.modules.items()):
        if name == "pika_h3" or name.startswith("pika_h3."):
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(package):
                raise ContractError("Another Pika implementation owns this interpreter")
    entry = str(root / "src")
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return importlib.import_module("pika_h3.pipeline")


PIKA_DEPENDENCIES = {
    "torch": "2.13.0+cu129",
    "torchvision": "0.28.0+cu129",
    "transformers": "5.15.0",
    "peft": "0.20.0",
    "av": "18.1.0",
    "numpy": "2.5.2",
    "pillow": "12.3.0",
    "pyyaml": "6.0.3",
    "safetensors": "0.8.0",
    "huggingface-hub": "1.30.0",
    "accelerate": "1.14.0",
}


def validate_pika_environment():
    """The faithful reference entry cannot inherit another pipeline's runtime."""
    if sys.version_info[:2] != (3, 12):
        raise ContractError("The Pika reference runtime requires isolated Python 3.12")
    for name, expected in PIKA_DEPENDENCIES.items():
        actual = version(name)
        if actual != expected:
            raise ContractError(
                f"Pika reference dependency differs: {name} {actual} != {expected}"
            )
