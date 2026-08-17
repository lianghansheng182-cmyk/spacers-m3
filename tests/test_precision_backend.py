import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.precision_backend import (
    PrecisionMode,
    get_precision_backend,
    normalize_precision,
    precision_catalog,
    validate_precision_for_run,
)


def test_normalize_precision_aliases():
    assert normalize_precision("bf16") is PrecisionMode.BF16
    assert normalize_precision("INT8") is PrecisionMode.INT8
    assert normalize_precision("int4") is PrecisionMode.INT4


def test_bf16_is_runnable():
    backend = get_precision_backend("BF16")
    assert backend.is_runnable is True
    assert backend.torch_dtype == "bfloat16"


def test_int8_int4_are_planned_only():
    for mode in ("INT8", "INT4"):
        backend = get_precision_backend(mode)
        assert backend.is_runnable is False
        assert backend.runtime_status == "planned"


def test_validate_precision_blocks_planned_modes():
    try:
        validate_precision_for_run("INT8")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "not runnable" in str(exc)


def test_precision_catalog_lists_all_modes():
    catalog = precision_catalog()
    assert set(catalog) == {"BF16", "INT8", "INT4"}


if __name__ == "__main__":
    test_normalize_precision_aliases()
    test_bf16_is_runnable()
    test_int8_int4_are_planned_only()
    test_validate_precision_blocks_planned_modes()
    test_precision_catalog_lists_all_modes()
    print("[PASS] All precision backend tests passed")
