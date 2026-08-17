"""Unified precision backend interface for batch / LoRA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PrecisionMode(str, Enum):
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


@dataclass(frozen=True)
class PrecisionBackend:
    mode: PrecisionMode
    torch_dtype: str
    runtime_status: str
    load_strategy: str
    notes: str

    @property
    def is_runnable(self) -> bool:
        return self.runtime_status == "ready"


SUPPORTED_PRECISIONS = tuple(PrecisionMode)
RUNNABLE_PRECISIONS = (PrecisionMode.BF16,)


def normalize_precision(value: str) -> PrecisionMode:
    normalized = value.strip().upper()
    aliases = {
        "BF16": PrecisionMode.BF16,
        "BFLOAT16": PrecisionMode.BF16,
        "INT8": PrecisionMode.INT8,
        "INT4": PrecisionMode.INT4,
        "FP16": PrecisionMode.BF16,
        "FP32": PrecisionMode.BF16,
    }
    if normalized not in aliases:
        supported = ", ".join(mode.value for mode in SUPPORTED_PRECISIONS)
        raise ValueError(f"Unsupported precision '{value}'. Supported: {supported}")
    return aliases[normalized]


def get_precision_backend(precision: str | PrecisionMode) -> PrecisionBackend:
    mode = precision if isinstance(precision, PrecisionMode) else normalize_precision(precision)

    if mode is PrecisionMode.BF16:
        return PrecisionBackend(
            mode=mode,
            torch_dtype="bfloat16",
            runtime_status="ready",
            load_strategy="from_pretrained",
            notes="Current default baseline path. Uses transformers from_pretrained with BF16 on CUDA.",
        )

    if mode is PrecisionMode.INT8:
        return PrecisionBackend(
            mode=mode,
            torch_dtype="int8",
            runtime_status="planned",
            load_strategy="quantized_load_not_enabled",
            notes="Interface reserved. Do not run large-scale INT8 experiments until backend is enabled.",
        )

    return PrecisionBackend(
        mode=mode,
        torch_dtype="int4",
        runtime_status="planned",
        load_strategy="quantized_load_not_enabled",
        notes="Interface reserved. Do not run large-scale INT4 experiments until backend is enabled.",
    )


def validate_precision_for_run(precision: str | PrecisionMode) -> PrecisionBackend:
    backend = get_precision_backend(precision)
    if not backend.is_runnable:
        raise RuntimeError(
            f"Precision {backend.mode.value} is not runnable yet "
            f"({backend.runtime_status}). {backend.notes}"
        )
    return backend


def precision_catalog() -> dict[str, Any]:
    return {
        mode.value: {
            "torch_dtype": get_precision_backend(mode).torch_dtype,
            "runtime_status": get_precision_backend(mode).runtime_status,
            "load_strategy": get_precision_backend(mode).load_strategy,
            "notes": get_precision_backend(mode).notes,
        }
        for mode in SUPPORTED_PRECISIONS
    }
