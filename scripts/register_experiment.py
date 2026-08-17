#!/usr/bin/env python3
"""Register an existing experiment directory into benchmark_report and CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.experiment_registry import ExperimentProfile, register_existing_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register existing experiment outputs.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--precision", default="BF16")
    parser.add_argument("--dataset", default="vrsbench_mini")
    parser.add_argument("--input-path", default="datasets/mini_eval/mini_eval.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image-size", default="max_pixels=1048576")
    parser.add_argument("--operator", default="朱福康")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = ExperimentProfile(
        experiment_name=args.experiment_name,
        base_model=args.base_model,
        precision=args.precision,
        dataset=args.dataset,
        input_path=args.input_path,
        results_dir=str(args.results_dir),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        lora_adapter=args.lora_adapter,
        operator=args.operator,
        image_size=args.image_size,
    )
    report = register_existing_results(profile)
    print(f"[PASS] Registered experiment '{args.experiment_name}'")
    print(f"[PASS] Report saved with experiment section: git_commit={report['experiment'].get('git_commit')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
