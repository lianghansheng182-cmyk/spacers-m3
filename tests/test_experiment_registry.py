import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.benchmark_metrics import build_benchmark_report, load_samples
from src.utils.experiment_registry import (
    ExperimentProfile,
    append_registry_row,
    build_experiment_section,
    directory_size_mb,
    enrich_benchmark_report,
    generate_experiment_report,
    write_experiment_json,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_RESULTS = Path(__file__).resolve().parents[1] / "results" / "vrsbench_mini" / "qwen3_vl_2b_zero_shot"


def test_directory_size_mb_on_existing_results():
    if not REAL_RESULTS.exists():
        return
    size = directory_size_mb(REAL_RESULTS)
    assert size is not None
    assert size > 0


def test_experiment_section_contains_required_fields():
    profile = ExperimentProfile(
        experiment_name="demo",
        base_model="Qwen/Qwen3-VL-2B-Instruct",
        precision="BF16",
        dataset="vrsbench_mini",
        input_path="datasets/mini_eval/mini_eval.jsonl",
        results_dir="results/demo",
        git_commit="abc123",
    )
    report = {
        "overall": {
            "mean_latency_s": 1.0,
            "p95_latency_s": 2.0,
            "peak_vram_mb": 100.0,
            "model_load_time_s": 3.0,
            "success_count": 30,
            "failure_count": 0,
            "failure_rate": 0.0,
        },
        "vqa": {"vqa_accuracy": 0.5},
        "grounding": {
            "grounding_mean_iou": 0.56,
            "acc_at_0_5": 0.7,
            "acc_at_0_7": 0.4,
        },
        "model_config": {"batch_size": 1, "precision": "BF16"},
    }
    section = build_experiment_section(profile, report)
    assert section["experiment_name"] == "demo"
    assert section["git_commit"] == "abc123"
    assert section["vqa_accuracy"] == 0.5
    assert section["grounding_acc_at_0_5"] == 0.7


def test_register_existing_results_with_temp_registry():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()
        (results_dir / "predictions.jsonl").write_text(
            (FIXTURES_DIR / "benchmark_predictions.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (results_dir / "metrics.json").write_text(
            (FIXTURES_DIR / "benchmark_metrics.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (results_dir / "summary.json").write_text(
            json.dumps({"model_load_time_s": 1.5, "precision": "BF16", "batch_size": 1}),
            encoding="utf-8",
        )
        (results_dir / "run_config.json").write_text(
            json.dumps({"model_path": "models/Qwen3-VL-2B-Instruct", "max_pixels": 1048576}),
            encoding="utf-8",
        )

        registry_path = Path(tmp) / "experiment_registry.csv"
        registry_path.write_text(
            "experiment_id,date,operator,git_commit,model,precision,dataset,task,image_count,image_size,max_new_tokens,latency_mean_s,latency_p95_s,peak_vram_mb,model_size_mb,metric_name,metric_score,status,result_file,notes\n",
            encoding="utf-8",
        )

        profile = ExperimentProfile(
            experiment_name="fixture_demo",
            base_model="Qwen/Qwen3-VL-2B-Instruct",
            precision="BF16",
            dataset="vrsbench_mini",
            input_path="datasets/mini_eval/mini_eval.jsonl",
            results_dir=str(results_dir),
        )
        write_experiment_json(profile, results_dir)
        report = generate_experiment_report(results_dir)
        append_registry_row(profile, report, registry_path=registry_path)

        rows = list(csv.DictReader(registry_path.open(encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "EXP-001"
        assert (results_dir / "benchmark_report.json").exists()


def test_real_results_enrichment_if_present():
    if not (REAL_RESULTS / "predictions.jsonl").exists():
        return
    profile = ExperimentProfile(
        experiment_name="qwen3_vl_2b_zero_shot_bf16",
        base_model="Qwen/Qwen3-VL-2B-Instruct",
        precision="BF16",
        dataset="vrsbench_mini",
        input_path="datasets/mini_eval/mini_eval.jsonl",
        results_dir=str(REAL_RESULTS.relative_to(Path(__file__).resolve().parents[1])),
    )
    write_experiment_json(profile, REAL_RESULTS)
    samples, _, metrics = load_samples(REAL_RESULTS)
    report = build_benchmark_report(samples, metrics=metrics)
    enriched = enrich_benchmark_report(report, REAL_RESULTS)
    assert enriched["experiment"]["experiment_name"] == "qwen3_vl_2b_zero_shot_bf16"
    assert enriched["experiment"]["mean_latency_s"] is not None


if __name__ == "__main__":
    test_directory_size_mb_on_existing_results()
    test_experiment_section_contains_required_fields()
    test_register_existing_results_with_temp_registry()
    test_real_results_enrichment_if_present()
    print("[PASS] All experiment registry tests passed")
