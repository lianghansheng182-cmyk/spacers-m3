#!/usr/bin/env python3
"""Convenience wrapper for the OrbitGuard-RS recovery ablation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "reliability" / "run_fault_injection.py"),
            "--config",
            str(REPO_ROOT / "configs" / "reliability" / "orbitguard_v1.yaml"),
        ],
        cwd=REPO_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
