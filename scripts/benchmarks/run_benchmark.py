#!/usr/bin/env python3
"""Thin dispatcher for the frozen benchmark entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3-VL-2B-Instruct"
DEFAULT_ADAPTER_PATH = REPO_ROOT / "results" / "lora" / "qwen3_vl_2b_lora_v1" / "adapter_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen SPACERS-M3 benchmark adapter.")
    parser.add_argument(
        "benchmark",
        choices=("vrsbench", "mme-rs-lite", "xlrs-lite", "levir-cc"),
        help="Benchmark adapter to run.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--precision", choices=("bf16", "int8", "int4"), default="int4")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke runs.")
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    return parser.parse_args()


def add_common(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--model-path",
            str(args.model_path),
            "--adapter-path",
            str(args.adapter_path),
            "--precision",
            args.precision,
            "--max-pixels",
            str(args.max_pixels),
        ]
    )


def build_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable]
    if args.benchmark == "vrsbench":
        command.append(str(REPO_ROOT / "scripts" / "lora" / "evaluate_lora_adapter.py"))
        add_common(command, args)
        command.extend(
            [
                "--input",
                str(REPO_ROOT / "datasets" / "vrsbench_eval_v1" / "resolved_manifest.jsonl"),
                "--output-dir",
                str(args.output_dir or REPO_ROOT / "results" / "lora" / "qwen3_vl_2b_lora_v1_int4_90"),
                "--prompt-version",
                "v1",
                "--grounding-max-new-tokens",
                "32",
                "--stop-grounding-after-brace",
            ]
        )
        return command

    if args.benchmark == "mme-rs-lite":
        command.append(str(REPO_ROOT / "scripts" / "benchmarks" / "evaluate_mme_realworld_rs_lite.py"))
        add_common(command, args)
        command.extend(["--output-dir", str(args.output_dir or REPO_ROOT / "results" / "mme_realworld_rs")])
    elif args.benchmark == "xlrs-lite":
        command.append(str(REPO_ROOT / "scripts" / "benchmarks" / "evaluate_xlrs_bench_lite.py"))
        add_common(command, args)
        command.extend(["--output-dir", str(args.output_dir or REPO_ROOT / "results" / "xlrs_bench" / "full_lite_int4")])
    elif args.benchmark == "levir-cc":
        command.append(str(REPO_ROOT / "scripts" / "benchmarks" / "evaluate_levir_cc.py"))
        add_common(command, args)
        command.extend(["--output-dir", str(args.output_dir or REPO_ROOT / "results" / "levir_cc")])
    else:
        raise AssertionError(args.benchmark)

    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def main() -> int:
    args = parse_args()
    command = build_command(args)
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=str(REPO_ROOT), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
