"""
Step 0D verification: the schema contract must be provably correct before
any real data exists. Five cases per PROJECT_PLAN.md Step 0D item 6.
"""

import numpy as np
import pandas as pd
import pytest

from schema.unified import (
    F1_ONLY_COLUMNS,
    INTERSECTION_COLUMNS,
    UNIFIED_COLUMNS,
    SchemaViolation,
    project,
    validate_schema,
)


def _correct_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {col: rng.normal(size=10) for col in UNIFIED_COLUMNS}
    )


def test_correct_frame_passes():
    X = _correct_frame()
    validate_schema(X, mode="full")  # must not raise


def test_reordered_columns_raise():
    X = _correct_frame()[list(reversed(UNIFIED_COLUMNS))]
    with pytest.raises(SchemaViolation):
        validate_schema(X, mode="full")


def test_missing_column_raises():
    X = _correct_frame().drop(columns=[UNIFIED_COLUMNS[0]])
    with pytest.raises(SchemaViolation):
        validate_schema(X, mode="full")


def test_project_returns_correct_subset_sizes():
    X = _correct_frame()
    assert list(project(X, mode="intersection").columns) == INTERSECTION_COLUMNS
    assert len(INTERSECTION_COLUMNS) == 7
    assert list(project(X, mode="F1_only").columns) == F1_ONLY_COLUMNS
    assert len(F1_ONLY_COLUMNS) == 3


def test_invalid_mode_raises():
    X = _correct_frame()
    with pytest.raises(SchemaViolation):
        validate_schema(X, mode="not_a_real_mode")
    with pytest.raises(SchemaViolation):
        project(X, mode="not_a_real_mode")


def test_nan_is_permitted_not_a_violation():
    # B_ONLY columns are legitimately all-NaN on Dataset A -- this must NOT raise.
    X = _correct_frame()
    X["time_central"] = np.nan
    validate_schema(X, mode="full")  # must not raise
