"""Task-aware compute-budget routing for remote-sensing MLLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ResourceMode = Literal["fixed", "eco", "balanced", "precision", "auto"]
TilingMode = Literal["none", "grid_2x2"]


@dataclass(frozen=True)
class SpaceBudgetDecision:
    resource_mode: str
    precision: str
    max_pixels: int
    tiling_mode: TilingMode
    reason: str


@dataclass(frozen=True)
class SpaceBudgetProfile:
    name: str
    max_pixels: int
    description: str


# These values are intentionally expressed as pixel budgets rather than raw
# token counts because Qwen3-VL processors expose image budget control through
# per-image `max_pixels`. The realized token count is audited from
# `image_grid_thw` and image-token counts in processor outputs.
SPACEBUDGET_PROFILES: dict[str, SpaceBudgetProfile] = {
    "eco": SpaceBudgetProfile(
        name="ECO",
        max_pixels=512 * 512,
        description="Low visual budget for power-limited or coarse recognition runs.",
    ),
    "balanced": SpaceBudgetProfile(
        name="BALANCED",
        max_pixels=1024 * 1024,
        description="Current fixed single-view benchmark budget.",
    ),
    "precision": SpaceBudgetProfile(
        name="PRECISION",
        max_pixels=2048 * 2048,
        description="Higher single-view visual budget for detail-sensitive tasks.",
    ),
}


def visual_patch_tokens_from_grid(image_grid_thw: list[list[int]] | None) -> int:
    if not image_grid_thw:
        return 0
    return sum(int(t) * int(h) * int(w) for t, h, w in image_grid_thw)


def language_visual_tokens_from_grid(
    image_grid_thw: list[list[int]] | None,
    spatial_merge_size: int = 2,
) -> int:
    patch_tokens = visual_patch_tokens_from_grid(image_grid_thw)
    divisor = max(int(spatial_merge_size), 1) ** 2
    return patch_tokens // divisor


def normalized_task_label(task: str | None, question: str | None = None, category: str | None = None) -> str:
    joined = " ".join(str(value or "").lower() for value in (task, question, category))
    if "count" in joined or "how many" in joined or "number of" in joined:
        return "counting"
    if "ground" in joined or "bbox" in joined or "where is" in joined or "located" in joined or "position" in joined:
        return "spatial"
    if "color" in joined or "small" in joined or "tiny" in joined:
        return "small_object"
    if "regional land use" in joined or "land use" in joined:
        return "regional_recognition"
    if "spatial relationship" in joined or "relative" in joined:
        return "spatial"
    return "general"


def choose_budget(
    *,
    resource_mode: ResourceMode,
    task: str | None,
    question: str | None,
    category: str | None,
    original_width: int | None,
    original_height: int | None,
    fixed_max_pixels: int,
    allow_tiling: bool = False,
) -> SpaceBudgetDecision:
    """Return deterministic compute-budget settings.

    `auto` uses only pre-inference metadata: task/category/question text,
    image dimensions, and the caller-provided resource mode. It never consults
    the ground-truth answer or model prediction.
    """

    if resource_mode == "fixed":
        return SpaceBudgetDecision(
            resource_mode="fixed",
            precision="INT4 NF4",
            max_pixels=fixed_max_pixels,
            tiling_mode="none",
            reason=f"fixed caller max_pixels={fixed_max_pixels}",
        )

    if resource_mode in SPACEBUDGET_PROFILES:
        return SpaceBudgetDecision(
            resource_mode=resource_mode,
            precision="INT4 NF4",
            max_pixels=SPACEBUDGET_PROFILES[resource_mode].max_pixels,
            tiling_mode="none",
            reason=f"explicit {SPACEBUDGET_PROFILES[resource_mode].name} profile",
        )

    if resource_mode != "auto":
        raise ValueError(f"unsupported resource_mode: {resource_mode}")

    label = normalized_task_label(task, question, category)
    pixels = int(original_width or 0) * int(original_height or 0)
    megapixels = pixels / 1_000_000 if pixels else 0.0

    category_text = str(category or "").lower()
    question_text = str(question or "").lower()

    if "overall counting" in category_text:
        return SpaceBudgetDecision(
            resource_mode="auto",
            precision="INT4 NF4",
            max_pixels=SPACEBUDGET_PROFILES["eco"].max_pixels,
            tiling_mode="none",
            reason="overall-counting showed no reliable gain from higher visual budget",
        )

    if label == "counting" and "complex reasoning" not in category_text:
        return SpaceBudgetDecision(
            resource_mode="auto",
            precision="INT4 NF4",
            max_pixels=SPACEBUDGET_PROFILES["eco"].max_pixels,
            tiling_mode="none",
            reason="counting tasks showed weak visual-budget sensitivity; conserve tokens",
        )

    if "object motion state" in category_text or "complex reasoning" in category_text:
        return SpaceBudgetDecision(
            resource_mode="auto",
            precision="INT4 NF4",
            max_pixels=SPACEBUDGET_PROFILES["balanced"].max_pixels,
            tiling_mode="none",
            reason="category showed measurable ECO-to-BALANCED gain",
        )

    if label in {"spatial", "regional_recognition"}:
        tiling_mode: TilingMode = "grid_2x2" if allow_tiling and label == "regional_recognition" else "none"
        return SpaceBudgetDecision(
            resource_mode="auto",
            precision="INT4 NF4",
            max_pixels=SPACEBUDGET_PROFILES["balanced"].max_pixels,
            tiling_mode=tiling_mode,
            reason=f"{label} task benefits from moderate detail on {megapixels:.2f}MP image",
        )

    if label == "small_object" and ("color" in category_text or "color" in question_text):
        return SpaceBudgetDecision(
            resource_mode="auto",
            precision="INT4 NF4",
            max_pixels=SPACEBUDGET_PROFILES["balanced"].max_pixels,
            tiling_mode="none",
            reason="color/small-object questions need moderate detail, but PRECISION did not improve MME color",
        )

    return SpaceBudgetDecision(
        resource_mode="auto",
        precision="INT4 NF4",
        max_pixels=SPACEBUDGET_PROFILES["eco"].max_pixels,
        tiling_mode="none",
        reason=f"general or low-detail task label={label}",
    )


def summarize_token_values(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        rank = (len(ordered) - 1) * p
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        if lower == upper:
            return float(ordered[lower])
        weight = rank - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "p50": round(pct(0.50), 6),
        "p95": round(pct(0.95), 6),
        "min": min(values),
        "max": max(values),
    }
