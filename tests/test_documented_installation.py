"""Installation commands in reader documentation must resolve to this release."""

import re
import tomllib
from pathlib import Path


def test_documented_installation_uses_the_package_version():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    pages = sorted(root.glob("README*.md")) + sorted((root / "docs").rglob("*.md"))
    commands = [
        (page.relative_to(root), line.strip())
        for page in pages
        for line in page.read_text().splitlines()
        if "github.com/Hansimov/vflash" in line
        and re.match(r"\s*(?:git clone|(?:python -m )?pip install)\s", line)
    ]
    assert commands
    for page, command in commands:
        if command.startswith("git clone"):
            assert f"--branch v{version} " in command, page
        elif "/releases/download/" in command:
            assert f"/v{version}/vflash-{version}-" in command, page
