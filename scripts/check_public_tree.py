#!/usr/bin/env python3
"""Reject credentials and machine-specific identifiers from the public tree."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
MAX_TEXT_BYTES = 2_000_000
PATTERNS = {
    "credential assignment": re.compile(
        r"(?i)(?:HF_TOKEN|SUDO_PASSWORD|GITHUB_TOKEN|DOCKER_PASSWORD)\s*=\s*[^\s$<{]+"
    ),
    "Hugging Face token": re.compile(r"\bhf_" + r"[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[opusr]_" + r"[A-Za-z0-9]{20,}\b"),
    "private home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "physical GPU UUID": re.compile(r"\bGPU-" + r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F-]+)?\b"),
}


def candidate_paths() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in completed.stdout.split(b"\0"):
        if raw:
            paths.append((ROOT / raw.decode()).resolve())
    return tuple(paths)


def main() -> int:
    findings: list[str] = []
    for path in candidate_paths():
        if path == SELF or not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT)
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {label}")
    if findings:
        print("Public-tree check failed:")
        print("\n".join(f"- {finding}" for finding in findings))
        return 1
    print("Public-tree check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
