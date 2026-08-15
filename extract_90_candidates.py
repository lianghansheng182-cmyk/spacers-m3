#!/usr/bin/env python3
"""Copy the 90 eval_v1 images from a local full VRSBench image directory.

This helper is optional because the current PR already includes the 90-image
internal evaluation subset. It is kept for reproducibility when rebuilding the
subset from an independently downloaded VRSBench copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "datasets" / "vrsbench_eval_v1" / "candidate_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "vrsbench_eval_v1" / "images"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the official VRSBench source images.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    source_dir = args.images_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not MANIFEST.is_file():
        print(f"[FAIL] manifest not found: {MANIFEST}")
        return 1
    if not source_dir.is_dir():
        print(f"[FAIL] source image directory not found: {source_dir}")
        return 1

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []
    for row in rows:
        filename = Path(row["image"]).name
        src = source_dir / filename
        dst = output_dir / filename
        if not src.is_file():
            missing.append(filename)
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"[INFO] manifest samples: {len(rows)}")
    print(f"[INFO] copied: {copied}")
    if missing:
        print(f"[FAIL] missing: {len(missing)}")
        for filename in missing:
            print(f"  - {filename}")
        return 1

    print("[PASS] all eval_v1 images are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
