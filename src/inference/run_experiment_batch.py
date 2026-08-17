#!/usr/bin/env python3
"""Experiment-aware batch eval wrapper for base model and future LoRA runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.experiment_registry import (  # noqa: E402
    ExperimentProfile,
    append_registry_row,
    generate_experiment_report,
    read_git_commit,
    write_experiment_json,
)
from src.utils.precision_backend import (  # noqa: E402
    get_precision_backend,
    precision_catalog,
    validate_precision_for_run,
)


DEFAULT_INPUT = REPO_ROOT / "datasets" / "mini_eval" / "mini_eval.jsonl"


def _as_repo_path(path: Path) -> str:
    try:
        return path.expanduser().resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or validate a unified VRSBench mini_eval experiment."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--base-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora-adapter", type=Path, default=None)
    parser.add_argument("--precision", default="BF16")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--dataset", default="vrsbench_mini")
    parser.add_argument("--operator", default="朱福康")
    parser.add_argument("--register", action="store_true", help="Append experiment_registry.csv")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only rebuild benchmark_report.json and optional registry from existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate experiment config and precision backend without running inference.",
    )
    return parser.parse_args()


def _validate_lora_adapter(path: Path | None) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {resolved}")


def _build_profile(args: argparse.Namespace) -> ExperimentProfile:
    return ExperimentProfile(
        experiment_name=args.experiment_name,
        base_model=args.base_model_id,
        precision=args.precision,
        dataset=args.dataset,
        input_path=_as_repo_path(args.input),
        results_dir=_as_repo_path(args.output_dir),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        lora_adapter=str(args.lora_adapter) if args.lora_adapter else None,
        git_commit=read_git_commit(),
        operator=args.operator,
        image_size=f"max_pixels={args.max_pixels}",
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    backend = get_precision_backend(args.precision)

    if args.dry_run:
        profile = _build_profile(args)
        _validate_lora_adapter(args.lora_adapter)
        payload = {
            "profile": profile.to_dict(),
            "precision_backend": {
                "mode": backend.mode.value,
                "runtime_status": backend.runtime_status,
                "load_strategy": backend.load_strategy,
                "notes": backend.notes,
            },
            "precision_catalog": precision_catalog(),
            "lora_adapter_ready": args.lora_adapter is None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[PASS] Dry-run validated experiment '{args.experiment_name}'")
        return 0

    profile = _build_profile(args)
    write_experiment_json(profile, output_dir)

    if args.report_only:
        report = generate_experiment_report(output_dir)
        if args.register:
            append_registry_row(profile, report)
        print(f"[PASS] Experiment report refreshed at {output_dir / 'benchmark_report.json'}")
        return 0

    validate_precision_for_run(args.precision)
    _validate_lora_adapter(args.lora_adapter)
    if args.lora_adapter is not None:
        raise RuntimeError(
            "LoRA adapter path was provided, but adapter loading is not enabled yet. "
            "Record the adapter in experiment.json now and enable loading after training completes."
        )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "src/inference/run_batch.py"),
        "--model-path",
        str(args.base_model_path.expanduser().resolve()),
        "--input",
        str(args.input.expanduser().resolve()),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-pixels",
        str(args.max_pixels),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if completed.returncode not in (0, 20):
        return completed.returncode

    report = generate_experiment_report(output_dir)
    if args.register:
        append_registry_row(profile, report)
    print(f"[PASS] Experiment completed: success={report['overall']['success_count']} failure={report['overall']['failure_count']}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
