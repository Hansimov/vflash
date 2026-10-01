"""Public runtime metadata must describe the wheel and documentation version."""

import tomllib
from pathlib import Path

import vflash


def test_package_version_matches_project():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert vflash.__version__ == project["project"]["version"]
