"""Criticality scoring for LoRA fault-injection experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def criticality_score(
    *,
    vqa_degradation: float,
    iou_degradation: float,
    acc05_degradation: float,
    parse_failure_delta: float,
    inference_failure_rate: float,
) -> float:
    """Task-driven score used for Critical/Important/Normal protection.

    Formula:
      0.35 * max(VQA drop, 0)
    + 0.35 * max(Grounding mean IoU drop, 0)
    + 0.15 * max(Acc@0.5 drop, 0)
    + 0.10 * parse failure delta
    + 0.05 * inference failure rate
    """

    return round(
        0.35 * max(vqa_degradation, 0.0)
        + 0.35 * max(iou_degradation, 0.0)
        + 0.15 * max(acc05_degradation, 0.0)
        + 0.10 * max(parse_failure_delta, 0.0)
        + 0.05 * max(inference_failure_rate, 0.0),
        6,
    )


def protection_class(score: float) -> str:
    if score >= 0.08:
        return "Critical"
    if score >= 0.025:
        return "Important"
    return "Normal"


def summarize_fault_runs(runs: list[dict[str, Any]], healthy: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["group_key"]].append(run)

    rows: list[dict[str, Any]] = []
    for group_key, items in sorted(grouped.items()):
        vqa_drop = sum(healthy["vqa_accuracy"] - item["metrics"]["vqa_accuracy"] for item in items) / len(items)
        iou_drop = sum(healthy["grounding_mean_iou"] - item["metrics"]["grounding_mean_iou"] for item in items) / len(items)
        acc05_drop = sum(healthy["grounding_acc_at_0_5"] - item["metrics"]["grounding_acc_at_0_5"] for item in items) / len(items)
        parse_delta = sum(item["metrics"]["parse_failures"] - healthy["parse_failures"] for item in items) / len(items)
        failure_rate = sum(item["metrics"]["inference_failures"] / max(item["metrics"]["sample_count"], 1) for item in items) / len(items)
        score = criticality_score(
            vqa_degradation=vqa_drop,
            iou_degradation=iou_drop,
            acc05_degradation=acc05_drop,
            parse_failure_delta=parse_delta,
            inference_failure_rate=failure_rate,
        )
        rows.append(
            {
                "group_key": group_key,
                "runs": len(items),
                "mean_vqa_degradation": round(vqa_drop, 6),
                "mean_iou_degradation": round(iou_drop, 6),
                "mean_acc05_degradation": round(acc05_drop, 6),
                "mean_parse_failure_delta": round(parse_delta, 6),
                "mean_inference_failure_rate": round(failure_rate, 6),
                "criticality_score": score,
                "protection_class": protection_class(score),
            }
        )
    rows.sort(key=lambda row: row["criticality_score"], reverse=True)
    return rows
