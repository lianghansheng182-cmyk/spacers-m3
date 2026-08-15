#!/usr/bin/env python3
"""Materialize a valid eval manifest by applying explicit bbox repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "datasets" / "vrsbench_eval_v1"
DEFAULT_INPUT = DEFAULT_DIR / "candidate_manifest.jsonl"
DEFAULT_REPAIRS = DEFAULT_DIR / "bbox_repairs.json"
DEFAULT_OUTPUT = DEFAULT_DIR / "resolved_manifest.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--repairs", type=Path, default=DEFAULT_REPAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    repairs_payload = json.loads(args.repairs.read_text(encoding="utf-8"))
    repairs = repairs_payload.get("bbox_repairs", {})
    output_rows = []

    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        repair = repairs.get(row.get("id"))
        if repair:
            if repair.get("source_bbox") != row.get("bbox"):
                raise ValueError(
                    f"{row.get('id')}: source_bbox does not match candidate_manifest"
                )
            row["bbox_original"] = row["bbox"]
            row["bbox"] = repair["effective_bbox"]
            row["bbox_repair_reason"] = repair.get("reason")
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] wrote {len(output_rows)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
