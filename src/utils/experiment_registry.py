"""Experiment profile, registry helpers, and benchmark report enrichment."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.benchmark_metrics import (
    PROJECT_ROOT,
    generate_benchmark_report,
    load_samples,
    load_json,
    _relative,
)
from src.utils.precision_backend import PrecisionMode, get_precision_backend, normalize_precision

REGISTRY_PATH = PROJECT_ROOT / "results" / "experiment_registry.csv"
EXPERIMENT_FILENAME = "experiment.json"


@dataclass
class ExperimentProfile:
    experiment_name: str
    base_model: str
    precision: str
    dataset: str
    input_path: str
    results_dir: str
    batch_size: int = 1
    max_new_tokens: int = 256
    lora_adapter: str | None = None
    git_commit: str | None = None
    operator: str = "朱福康"
    image_size: str | None = None
    sample_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["precision"] = normalize_precision(self.precision).value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentProfile":
        return cls(
            experiment_name=str(payload["experiment_name"]),
            base_model=str(payload["base_model"]),
            precision=str(payload.get("precision") or PrecisionMode.BF16.value),
            dataset=str(payload.get("dataset") or "vrsbench_mini"),
            input_path=str(payload.get("input_path") or "datasets/mini_eval/mini_eval.jsonl"),
            results_dir=str(payload["results_dir"]),
            batch_size=int(payload.get("batch_size") or 1),
            max_new_tokens=int(payload.get("max_new_tokens") or 256),
            lora_adapter=str(payload["lora_adapter"]) if payload.get("lora_adapter") else None,
            git_commit=str(payload["git_commit"]) if payload.get("git_commit") else None,
            operator=str(payload.get("operator") or "朱福康"),
            image_size=str(payload["image_size"]) if payload.get("image_size") else None,
            sample_count=int(payload["sample_count"]) if payload.get("sample_count") is not None else None,
        )


def read_git_commit(project_root: Path | None = None) -> str | None:
    root = project_root or PROJECT_ROOT
    candidates = ["git", r"F:\Program Files\Git\cmd\git.exe"]
    for git_exe in candidates:
        try:
            completed = subprocess.run(
                [git_exe, "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        commit = completed.stdout.strip()
        if commit:
            return commit
    return None


def directory_size_mb(path: Path | None) -> float | None:
    if path is None:
        return None
    resolved = path.expanduser()
    if not resolved.exists():
        return None
    total_bytes = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
    return round(total_bytes / 1024 / 1024, 2)


def resolve_project_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def write_experiment_json(profile: ExperimentProfile, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = profile.to_dict()
    if not payload.get("git_commit"):
        payload["git_commit"] = read_git_commit()
    output_path = results_dir / EXPERIMENT_FILENAME
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_experiment_json(results_dir: Path) -> dict[str, Any]:
    path = results_dir / EXPERIMENT_FILENAME
    if not path.exists():
        return {}
    return load_json(path)


def build_experiment_section(
    profile: ExperimentProfile | None,
    report: Mapping[str, Any],
    *,
    base_model_path: Path | None = None,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    overall = report.get("overall", {})
    vqa = report.get("vqa", {})
    grounding = report.get("grounding", {})
    model_config = report.get("model_config", {})

    base_model = profile.base_model if profile else model_config.get("model_name")
    precision = profile.precision if profile else model_config.get("precision")
    lora_adapter = profile.lora_adapter if profile else None
    git_commit = profile.git_commit if profile and profile.git_commit else read_git_commit()

    return {
        "experiment_name": profile.experiment_name if profile else None,
        "git_commit": git_commit,
        "base_model": base_model,
        "lora_adapter": lora_adapter,
        "precision": precision,
        "batch_size": profile.batch_size if profile else model_config.get("batch_size"),
        "dataset": profile.dataset if profile else "vrsbench_mini",
        "base_model_size_mb": directory_size_mb(base_model_path),
        "adapter_size_mb": directory_size_mb(adapter_path),
        "load_time_s": overall.get("model_load_time_s"),
        "mean_latency_s": overall.get("mean_latency_s"),
        "p95_latency_s": overall.get("p95_latency_s"),
        "peak_vram_mb": overall.get("peak_vram_mb"),
        "success_count": overall.get("success_count"),
        "failure_count": overall.get("failure_count"),
        "failure_rate": overall.get("failure_rate"),
        "vqa_accuracy": vqa.get("vqa_accuracy"),
        "grounding_mean_iou": grounding.get("grounding_mean_iou"),
        "grounding_acc_at_0_5": grounding.get("acc_at_0_5"),
        "grounding_acc_at_0_7": grounding.get("acc_at_0_7"),
        "precision_backend": get_precision_backend(str(precision or PrecisionMode.BF16.value)).runtime_status,
    }


def enrich_benchmark_report(report: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    experiment_payload = load_experiment_json(results_dir)
    profile = ExperimentProfile.from_dict(experiment_payload) if experiment_payload else None
    run_config = load_json(results_dir / "run_config.json") if (results_dir / "run_config.json").exists() else {}

    base_model_path = resolve_project_path(
        str(run_config.get("model_path") or run_config.get("base_model_path") or "")
    )
    adapter_path = resolve_project_path(profile.lora_adapter if profile else run_config.get("lora_adapter"))

    enriched = dict(report)
    enriched["experiment"] = build_experiment_section(
        profile,
        report,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    return enriched


def generate_experiment_report(
    results_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_path = output_path or results_dir / "benchmark_report.json"
    report = generate_benchmark_report(results_dir, output_path)
    enriched = enrich_benchmark_report(report, results_dir)
    output_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return enriched


def _next_experiment_id(rows: Sequence[Mapping[str, str]]) -> str:
    max_id = 0
    for row in rows:
        value = row.get("experiment_id", "")
        if value.startswith("EXP-"):
            try:
                max_id = max(max_id, int(value.split("-", 1)[1]))
            except ValueError:
                continue
    return f"EXP-{max_id + 1:03d}"


def append_registry_row(
    profile: ExperimentProfile,
    report: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
    status: str = "success",
) -> dict[str, str]:
    registry_path = registry_path or REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    rows: list[dict[str, str]] = []
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows.extend(reader)

    experiment = report.get("experiment", {})
    model_config = report.get("model_config", {})
    overall = report.get("overall", {})
    vqa = report.get("vqa", {})

    model_label = profile.base_model
    if profile.lora_adapter:
        model_label = f"{profile.base_model}+lora"

    row = {name: "" for name in fieldnames}
    row.update(
        {
            "experiment_id": _next_experiment_id(rows),
            "date": date.today().isoformat(),
            "operator": profile.operator,
            "git_commit": experiment.get("git_commit") or profile.git_commit or "",
            "model": model_label,
            "precision": str(profile.precision),
            "dataset": profile.dataset,
            "task": profile.experiment_name,
            "image_count": str(profile.sample_count or overall.get("sample_count") or ""),
            "image_size": profile.image_size or str(model_config.get("image_resolution") or ""),
            "max_new_tokens": str(profile.max_new_tokens),
            "latency_mean_s": str(overall.get("mean_latency_s") or ""),
            "latency_p95_s": str(overall.get("p95_latency_s") or ""),
            "peak_vram_mb": str(overall.get("peak_vram_mb") or ""),
            "model_size_mb": str(experiment.get("base_model_size_mb") or ""),
            "metric_name": "vqa_accuracy",
            "metric_score": str(vqa.get("vqa_accuracy") or ""),
            "status": status,
            "result_file": _relative(Path(profile.results_dir) / "benchmark_report.json"),
            "notes": (
                f"adapter={profile.lora_adapter or 'none'}; "
                f"grounding_iou={experiment.get('grounding_mean_iou')}; "
                f"acc@0.5={experiment.get('grounding_acc_at_0_5')}; "
                f"acc@0.7={experiment.get('grounding_acc_at_0_7')}; "
                f"load_time_s={experiment.get('load_time_s')}"
            ),
        }
    )

    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)
    return row


def register_existing_results(profile: ExperimentProfile) -> dict[str, Any]:
    results_dir = resolve_project_path(profile.results_dir)
    if results_dir is None:
        raise ValueError("results_dir is required")
    profile.results_dir = _relative(results_dir)
    if profile.sample_count is None:
        samples, _, _ = load_samples(results_dir)
        profile.sample_count = len(samples)
    if not profile.git_commit:
        profile.git_commit = read_git_commit()
    write_experiment_json(profile, results_dir)
    report = generate_experiment_report(results_dir)
    append_registry_row(profile, report)
    return report
